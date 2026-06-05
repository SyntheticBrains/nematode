"""Unit tests for the shared action-policy helpers (`brain/arch/_policy.py`).

Covers the discrete-mode scenarios at
``openspec/changes/add-continuous-2d-and-action-heads/specs/continuous-action-policy/spec.md``:
the helpers must reproduce the per-brain inline numerics exactly (byte-equivalent
migration). Continuous (tanh-squashed Gaussian) helpers are tested when added by
the continuous-action-heads work.
"""

from __future__ import annotations

import torch
from quantumnematode.brain.arch._policy import (
    categorical_evaluate_torch,
    categorical_logprob_entropy_torch,
    categorical_sample_torch,
    ppo_clip_policy_loss,
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
        # Option B tolerance: torch log-prob vs the manual log(softmax)+eps the
        # LSTM/CfC brains used. Deviation is float32 round-off for taken actions.
        logits = torch.tensor([1.3, -0.4, 0.8, 0.1])
        probs = torch.softmax(logits, dim=-1)
        for action in range(4):
            manual = float(torch.log(probs[action] + 1e-8))
            log_prob, _, _ = categorical_logprob_entropy_torch(logits, action)
            assert abs(float(log_prob) - manual) < 1e-5

    def test_entropy_close_to_manual_within_tolerance(self) -> None:
        # Option B tolerance: torch entropy vs the manual -sum(p*log(p+1e-10))
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
