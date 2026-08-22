"""Unit tests for the shared action-policy helpers (`brain/arch/_policy.py`).

Covers the discrete-mode scenarios: the helpers must reproduce the per-brain
inline numerics exactly (byte-equivalent migration). Continuous (tanh-squashed
Gaussian) helpers are tested when added by the continuous-action-heads work.
"""

from __future__ import annotations

import math

import torch
from quantumnematode.brain.arch._policy import (
    categorical_evaluate_torch,
    categorical_logprob_entropy_from_probs,
    categorical_logprob_entropy_torch,
    categorical_sample_torch,
    ppo_clip_policy_loss,
    reinforce_policy_loss,
)


class TestCategoricalSampleTorch:
    """The torch-backend sampler must match the inline MLP-PPO / connectome-PPO path."""

    def test_matches_inline_categorical_under_same_seed(self) -> None:
        logits = torch.tensor([0.5, -1.2, 0.3, 0.9])

        # Inline reference (the pre-refactor per-brain code).
        torch.manual_seed(123)
        probs_ref = torch.softmax(logits, dim=-1)
        dist_ref = torch.distributions.Categorical(probs_ref)
        action_ref = int(dist_ref.sample().item())
        log_prob_ref = dist_ref.log_prob(torch.tensor(action_ref))
        entropy_ref = dist_ref.entropy()

        # Helper, same RNG state.
        torch.manual_seed(123)
        action, log_prob, entropy, probs = categorical_sample_torch(logits)

        assert action == action_ref
        assert torch.equal(probs, probs_ref)
        assert torch.equal(log_prob, log_prob_ref)
        assert torch.equal(entropy, entropy_ref)

    def test_outputs_finite_and_action_in_range(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(4)
        action, log_prob, entropy, probs = categorical_sample_torch(logits)
        assert 0 <= action < 4
        assert torch.isfinite(log_prob)
        assert torch.isfinite(entropy)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)


class TestCategoricalEvaluateTorch:
    """The torch-backend batch evaluator must match the inline PPO-update path."""

    def test_matches_inline_batch_logprob_and_entropy(self) -> None:
        logits = torch.tensor([[0.5, -1.2, 0.3, 0.9], [0.1, 0.2, -0.5, 0.0]])
        actions = torch.tensor([2, 0])

        probs_ref = torch.softmax(logits, dim=-1)
        dist_ref = torch.distributions.Categorical(probs_ref)
        log_probs_ref = dist_ref.log_prob(actions)
        entropy_ref = dist_ref.entropy().mean()

        log_probs, entropy = categorical_evaluate_torch(logits, actions)

        assert torch.equal(log_probs, log_probs_ref)
        assert torch.equal(entropy, entropy_ref)


class TestCategoricalLogprobEntropyTorch:
    """The given-action log-prob/entropy helper (LSTM/CfC update path)."""

    def test_matches_inline_categorical_for_given_action(self) -> None:
        logits = torch.tensor([0.5, -1.2, 0.3, 0.9])
        action = 2

        probs_ref = torch.softmax(logits, dim=-1)
        dist_ref = torch.distributions.Categorical(probs_ref)
        log_prob_ref = dist_ref.log_prob(torch.tensor(action))
        entropy_ref = dist_ref.entropy()

        log_prob, entropy, probs = categorical_logprob_entropy_torch(logits, action)

        assert torch.equal(log_prob, log_prob_ref)
        assert torch.equal(entropy, entropy_ref)
        assert torch.equal(probs, probs_ref)

    def test_is_differentiable(self) -> None:
        logits = torch.tensor([0.5, -1.2, 0.3, 0.9], requires_grad=True)
        log_prob, entropy, _ = categorical_logprob_entropy_torch(logits, 1)
        (log_prob + entropy).backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_close_to_manual_log_softmax_within_tolerance(self) -> None:
        # Tolerance: torch log-prob vs the manual log(softmax)+eps the
        # LSTM/CfC brains used. Deviation is float32 round-off for taken actions.
        logits = torch.tensor([1.3, -0.4, 0.8, 0.1])
        probs = torch.softmax(logits, dim=-1)
        for action in range(4):
            manual = float(torch.log(probs[action] + 1e-8))
            log_prob, _, _ = categorical_logprob_entropy_torch(logits, action)
            assert abs(float(log_prob) - manual) < 1e-5

    def test_entropy_close_to_manual_within_tolerance(self) -> None:
        # Tolerance: torch entropy vs the manual -sum(p*log(p+1e-10))
        # the LSTM/CfC/spiking brains used. Includes a saturated case to lock in
        # the no-log(0)/NaN guarantee (torch's Categorical.entropy clamps log).
        for logits in (
            torch.tensor([1.3, -0.4, 0.8, 0.1]),  # diffuse
            torch.tensor([6.0, -2.0, 0.0, -1.0]),  # peaked
            torch.tensor([60.0, 0.0, 0.0, -30.0]),  # saturated (a prob underflows)
        ):
            probs = torch.softmax(logits, dim=-1)
            manual = float(-torch.sum(probs * torch.log(probs + 1e-10)))
            _, entropy, _ = categorical_logprob_entropy_torch(logits, 0)
            assert torch.isfinite(entropy)
            assert abs(float(entropy) - manual) < 1e-5


class TestPPOClipPolicyLoss:
    """The shared clipped surrogate must match the inline per-brain term."""

    def test_matches_inline_surrogate(self) -> None:
        new_log_probs = torch.tensor([-0.5, -1.0, -0.2])
        old_log_probs = torch.tensor([-0.7, -0.9, -0.3])
        advantages = torch.tensor([1.0, -2.0, 0.5])
        clip_epsilon = 0.2

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        loss_ref = -torch.min(surr1, surr2).mean()

        loss = ppo_clip_policy_loss(new_log_probs, old_log_probs, advantages, clip_epsilon)
        assert torch.equal(loss, loss_ref)


class TestCategoricalLogprobEntropyFromProbs:
    """The probs-based scorer serves the Family-C ε-mixture policies."""

    @staticmethod
    def _epsilon_mixture(
        logits: torch.Tensor,
        epsilon: float,
        temperature: float,
    ) -> torch.Tensor:
        """Rebuild the exact ε-greedy mixture the Family-C brains construct."""
        softmax_probs = torch.softmax(logits / temperature, dim=-1)
        uniform = torch.ones_like(softmax_probs) / softmax_probs.shape[-1]
        return (1 - epsilon) * softmax_probs + epsilon * uniform

    def test_matches_inline_categorical_on_a_mixture(self) -> None:
        """Scoring an ε-mixture must equal the inline Categorical ops on it."""
        logits = torch.tensor([1.4, -0.6, 0.2, -1.1])
        probs = self._epsilon_mixture(logits, epsilon=0.1, temperature=1.5)
        action = 2

        dist_ref = torch.distributions.Categorical(probs)
        log_prob_ref = dist_ref.log_prob(torch.tensor(action))
        entropy_ref = dist_ref.entropy()

        log_prob, entropy, returned = categorical_logprob_entropy_from_probs(probs, action)

        assert torch.equal(log_prob, log_prob_ref)
        assert torch.equal(entropy, entropy_ref)
        assert returned is probs

    def test_mixture_is_not_a_softmax_of_the_logits(self) -> None:
        """Guards the reason this helper exists at all.

        If the ε-mixture happened to equal ``softmax(logits)``, the logits-based
        helper would have sufficed and the Family-C migration would be scoring
        the wrong distribution without any test noticing.
        """
        logits = torch.tensor([1.4, -0.6, 0.2, -1.1])
        probs = self._epsilon_mixture(logits, epsilon=0.1, temperature=1.5)
        action = 2

        from_probs, _, _ = categorical_logprob_entropy_from_probs(probs, action)
        from_logits, _, _ = categorical_logprob_entropy_torch(logits, action)

        assert not torch.allclose(from_probs, from_logits, rtol=0, atol=1e-4)

    def test_deviation_from_the_manual_epsilon_floored_form_is_within_tolerance(self) -> None:
        """Dropping the ``+1e-8`` / ``+1e-10`` floors stays inside the declared 1e-7 bar."""
        logits = torch.tensor([1.4, -0.6, 0.2, -1.1])
        probs = self._epsilon_mixture(logits, epsilon=0.05, temperature=1.0)
        action = 1

        # The pre-migration inline expressions, verbatim.
        log_prob_manual = torch.log(probs[action] + 1e-8)
        entropy_manual = -torch.sum(probs * torch.log(probs + 1e-10))

        log_prob, entropy, _ = categorical_logprob_entropy_from_probs(probs, action)

        assert torch.allclose(log_prob, log_prob_manual, rtol=0, atol=1e-7)
        assert torch.allclose(entropy, entropy_manual, rtol=0, atol=1e-7)

    def test_renormalises_a_drifted_vector(self) -> None:
        """Documented behaviour: ``Categorical`` normalises rather than rejecting."""
        probs = torch.tensor([0.2, 0.2, 0.2, 0.2])  # sums to 0.8, not 1.0
        log_prob, entropy, _ = categorical_logprob_entropy_from_probs(probs, 0)

        assert torch.allclose(log_prob, torch.log(torch.tensor(0.25)), rtol=0, atol=1e-6)
        assert torch.isfinite(entropy)


class TestLogitsScorerDelegation:
    """The logits scorer is now a softmax front-end over the probs scorer."""

    def test_delegation_is_byte_exact_for_the_already_migrated_brains(self) -> None:
        """Refactoring the logits helper must not move any migrated brain."""
        logits = torch.tensor([0.5, -1.2, 0.3, 0.9])
        action = 3

        # The pre-refactor body of ``categorical_logprob_entropy_torch``.
        probs_ref = torch.softmax(logits, dim=-1)
        dist_ref = torch.distributions.Categorical(probs_ref)
        log_prob_ref = dist_ref.log_prob(torch.tensor(action))
        entropy_ref = dist_ref.entropy()

        log_prob, entropy, probs = categorical_logprob_entropy_torch(logits, action)

        assert torch.equal(log_prob, log_prob_ref)
        assert torch.equal(entropy, entropy_ref)
        assert torch.equal(probs, probs_ref)

    def test_delegation_agrees_with_calling_the_probs_scorer_directly(self) -> None:
        logits = torch.tensor([0.5, -1.2, 0.3, 0.9])
        action = 1

        via_logits = categorical_logprob_entropy_torch(logits, action)
        via_probs = categorical_logprob_entropy_from_probs(torch.softmax(logits, dim=-1), action)

        assert torch.equal(via_logits[0], via_probs[0])
        assert torch.equal(via_logits[1], via_probs[1])


class TestReinforcePolicyLoss:
    """The shared REINFORCE term must match the inline per-brain loss."""

    def test_matches_inline_vectorised_form(self) -> None:
        """Byte-exact for brains that already wrote the vectorised expression."""
        log_probs = torch.tensor([-0.5, -1.0, -0.2, -1.7])
        advantages = torch.tensor([1.0, -2.0, 0.5, 0.3])

        loss_ref = -(log_probs * advantages).mean()
        loss = reinforce_policy_loss(log_probs, advantages)

        assert torch.equal(loss, loss_ref)

    def test_matches_loop_accumulated_form_within_tolerance(self) -> None:
        """``qrc`` / ``mlpreinforce`` accumulated in a Python loop then divided.

        Same mathematics, different association order — a float32 reorder, so
        this is the declared ~1e-7 case rather than byte-exact.
        """
        torch.manual_seed(11)
        log_probs = torch.randn(64)
        advantages = torch.randn(64)

        loop_loss = torch.tensor(0.0)
        for t in range(len(log_probs)):
            loop_loss = loop_loss - log_probs[t] * advantages[t]
        loop_loss = loop_loss / len(log_probs)

        loss = reinforce_policy_loss(log_probs, advantages)

        assert torch.allclose(loss, loop_loss, rtol=0, atol=1e-7)

    def test_gradient_flows_to_log_probs(self) -> None:
        log_probs = torch.tensor([-0.5, -1.0], requires_grad=True)
        advantages = torch.tensor([1.0, -2.0])

        reinforce_policy_loss(log_probs, advantages).backward()

        assert log_probs.grad is not None
        assert torch.isfinite(log_probs.grad).all()


class TestCategoricalInternalClamp:
    """``Categorical`` has its own floor — larger than the one the migration removed.

    Found in review of this migration. The scorers' docstrings originally claimed
    "no epsilon floor is applied", which is wrong: ``Categorical`` runs
    ``clamp_probs`` internally, pinning probabilities to
    ``[finfo(dtype).eps, 1 - finfo(dtype).eps]`` — ``1.19e-7`` in float32, versus
    the ``1e-8`` additive floor the per-brain code used. These tests pin the real
    behaviour so it is known rather than rediscovered.
    """

    def test_clamp_threshold_is_float32_eps(self) -> None:
        from torch.distributions.utils import clamp_probs

        eps = torch.finfo(torch.float32).eps
        probs = torch.tensor([1e-9, 1e-5, 1.0 - 1e-5 - 1e-9], dtype=torch.float32)
        assert float(clamp_probs(probs)[0]) == eps
        assert eps > 1e-8, "the internal clamp is LARGER than the removed 1e-8 floor"

    def test_log_prob_saturates_below_the_clamp(self) -> None:
        """Below the clamp the log-prob stops tracking ``p`` at all."""
        saturated = math.log(torch.finfo(torch.float32).eps)
        for p in (1e-8, 1e-9, 1e-12):
            probs = torch.tensor([p, 1.0 - p], dtype=torch.float32)
            log_prob, _entropy, _probs = categorical_logprob_entropy_from_probs(probs, 0)
            assert abs(float(log_prob) - saturated) < 1e-3

    def test_gradient_is_zeroed_below_the_clamp(self) -> None:
        """The behavioural change that matters: a clamped entry gets no gradient.

        ``clamp`` has zero gradient outside its range, so a timestep whose action
        has fallen below ``1.19e-7`` under the updated policy contributes nothing
        to the policy gradient — where ``log(p + 1e-8)`` contributed an enormous
        one. Better-conditioned (it removes a variance spike), but a change.
        """
        below = torch.tensor([1e-9, 1.0 - 1e-9], dtype=torch.float32, requires_grad=True)
        categorical_logprob_entropy_from_probs(below, 0)[0].backward()
        assert below.grad is not None
        assert float(below.grad[0]) == 0.0

        above = torch.tensor([1e-4, 1.0 - 1e-4], dtype=torch.float32, requires_grad=True)
        categorical_logprob_entropy_from_probs(above, 0)[0].backward()
        assert above.grad is not None
        assert float(above.grad[0]) > 0.0

    def test_the_clamp_is_unreachable_for_the_epsilon_mixture_brains(self) -> None:
        """Family C floors ``p`` at ``eps/n_actions``, far above the clamp.

        ``exploration_schedule`` decays ε to 30% of its initial value, never to
        zero, so at the configured ``exploration_epsilon: 0.1`` with 4 actions the
        mixture guarantees ``p >= 0.0075`` — ~63,000x the clamp threshold.
        """
        clamp = torch.finfo(torch.float32).eps
        epsilon_final, n_actions = 0.1 * (1.0 - 0.7), 4
        floor = epsilon_final / n_actions

        saturating = torch.tensor([40.0, 0.0, -40.0, -80.0])
        softmax_probs = torch.softmax(saturating, dim=-1)
        mixture = (1 - epsilon_final) * softmax_probs + epsilon_final / n_actions

        assert float(mixture.min()) >= floor * 0.999
        assert float(mixture.min()) > clamp * 1000
