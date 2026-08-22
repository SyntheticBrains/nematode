"""Migration regression for the direct-copy brains onto ``_policy.py`` (Task 4).

Four brains, three families:

- ``qliflstm`` — **Family B**. Torch ``logits`` are in hand at rollout, so both
  halves score through ``categorical_logprob_entropy_torch`` and the rollout and
  update use the same expression.
- ``qsnnppo`` — **Family B, NumPy-logits variant**. Its rollout forward pass is
  NumPy, so there is no torch ``logits`` tensor to score; the sampled probability
  vector is scored directly instead, cast to float32 to match the update's dtype.
- ``qsnnreinforce`` — **Family C**. Despite the brain's name this path is a
  PPO-clipped surrogate over an ε-greedy mixture.
- ``env/mlpppo_predator_brain`` — **Family A**, byte-exact.

Every comparison is same-process, same-tensor, same-RNG-state against the exact
pre-migration expression; no stored float goldens (see
``test_mlpppo_policy_migration.py`` for why).
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math

import numpy as np
import torch
from quantumnematode.brain.arch._policy import (
    categorical_evaluate_torch,
    categorical_logprob_entropy_from_probs,
    categorical_logprob_entropy_torch,
    categorical_sample_torch,
    ppo_clip_policy_loss,
)

_SEED = 4711
_N = 1000
_ENTROPY_TOLERANCE = 5e-7
_MODEL_TOLERANCE = 2e-6
_FLOAT32_SOFTMAX_FLOOR = 1e-6


class TestFamilyAPredatorBrain:
    """The PPO predator controller is byte-exact on both halves."""

    def test_rollout_sampling_matches_inline_reference(self) -> None:
        torch.manual_seed(_SEED)
        logits = torch.randn(4)

        torch.manual_seed(99)
        probs_ref = torch.softmax(logits, dim=-1)
        dist_ref = torch.distributions.Categorical(probs_ref)
        action_tensor = dist_ref.sample()
        idx_ref = int(action_tensor.item())
        log_prob_ref = dist_ref.log_prob(action_tensor)

        torch.manual_seed(99)
        idx, log_prob, _entropy, probs = categorical_sample_torch(logits)

        assert idx == idx_ref
        assert torch.equal(probs, probs_ref)
        assert torch.equal(log_prob, log_prob_ref)

    def test_update_matches_inline_reference(self) -> None:
        torch.manual_seed(_SEED)
        logits = torch.randn(8, 4)
        actions = torch.randint(0, 4, (8,))
        old_log_probs = torch.randn(8)
        advantages = torch.randn(8)
        clip_epsilon = 0.2

        probs_ref = torch.softmax(logits, dim=-1)
        dist_ref = torch.distributions.Categorical(probs_ref)
        new_ref = dist_ref.log_prob(actions)
        entropy_ref = dist_ref.entropy().mean()
        ratio = torch.exp(new_ref - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        loss_ref = -torch.min(surr1, surr2).mean()

        new_log_probs, entropy = categorical_evaluate_torch(logits, actions)
        loss = ppo_clip_policy_loss(new_log_probs, old_log_probs, advantages, clip_epsilon)

        assert torch.equal(new_log_probs, new_ref)
        assert torch.equal(entropy, entropy_ref)
        assert torch.equal(loss, loss_ref)


class TestFamilyBQlifLstm:
    """``qliflstm`` scores both halves from the same torch logits."""

    def test_scoring_consumes_no_rng_from_either_stream(self) -> None:
        """The real invariant behind "the sampler is untouched".

        The migration adds scoring calls around an unchanged ``rng.choice``. If
        any of those calls drew from the NumPy or torch RNG, every subsequent
        sampled action would shift and the trajectory would silently diverge —
        the one thing this migration promises cannot happen.

        An earlier version of this test reseeded ``default_rng`` twice and
        compared the draws to each other, which only demonstrated that NumPy is
        deterministic and never touched the migrated code at all.
        """
        rng = np.random.default_rng(_SEED)
        probs_np = np.array([0.4, 0.3, 0.2, 0.1])
        logits = torch.log(torch.as_tensor(probs_np, dtype=torch.float32))

        # Baseline: draws with no scoring interleaved.
        baseline = [int(rng.choice(4, p=probs_np)) for _ in range(64)]
        torch.manual_seed(_SEED)
        torch_baseline = torch.randn(8)

        # Same streams, with every migrated scorer called between draws.
        rng = np.random.default_rng(_SEED)
        interleaved = []
        for _ in range(64):
            interleaved.append(int(rng.choice(4, p=probs_np)))
            categorical_logprob_entropy_torch(logits, 0)
            categorical_logprob_entropy_from_probs(
                torch.as_tensor(probs_np, dtype=torch.float32),
                1,
            )
            categorical_evaluate_torch(logits.unsqueeze(0), torch.tensor([2]))
        torch.manual_seed(_SEED)
        for _ in range(8):
            categorical_logprob_entropy_torch(logits, 3)
        torch_after = torch.randn(8)

        assert interleaved == baseline
        assert torch.equal(torch_after, torch_baseline)

    def test_logprob_deviation_is_the_epsilon_floor_being_removed(self) -> None:
        """Old-vs-new is exactly ``-log1p(eps/p)``; see design D3."""
        torch.manual_seed(_SEED)
        exercised_a_real_deviation = False

        for scale in (1.0, 2.0, 3.0):
            for _ in range(_N // 3):
                logits = torch.randn(4) * scale
                probs = torch.softmax(logits, dim=-1)
                for action in range(4):
                    p = float(probs[action])
                    if p < _FLOAT32_SOFTMAX_FLOOR:
                        continue
                    floored = float(torch.log(probs[action] + 1e-8))
                    got, _entropy, _probs = categorical_logprob_entropy_torch(logits, action)
                    deviation = float(got) - floored
                    assert abs(deviation - (-math.log1p(1e-8 / p))) < _MODEL_TOLERANCE
                    if abs(deviation) > 1e-7:
                        exercised_a_real_deviation = True

        assert exercised_a_real_deviation

    def test_entropy_within_declared_tolerance(self) -> None:
        """The ``+1e-10`` entropy floor is damped by each term's factor of ``p``."""
        torch.manual_seed(_SEED)
        for _ in range(_N):
            logits = torch.randn(4) * 2.0
            probs = torch.softmax(logits, dim=-1)
            entropy_ref = -torch.sum(probs * torch.log(probs + 1e-10))
            _lp, entropy, _p = categorical_logprob_entropy_torch(logits, 0)
            assert torch.allclose(entropy, entropy_ref, rtol=0, atol=_ENTROPY_TOLERANCE)


class TestFamilyBQsnnPpoNumpyLogits:
    """``qsnnppo``'s rollout forward pass is NumPy, so it scores the probs vector."""

    @staticmethod
    def _numpy_softmax(logits: np.ndarray) -> np.ndarray:
        """Reproduce the brain's inline NumPy softmax exactly."""
        exp_probs = np.exp(logits - np.max(logits))
        return exp_probs / np.sum(exp_probs)

    def test_scored_distribution_is_the_one_sampled_from(self) -> None:
        """The rollout scores its own NumPy vector, not a torch re-derivation.

        This is the deliberate difference from ``qliflstm``: there is no torch
        ``logits`` tensor at rollout time, and re-deriving one would score a
        distribution the sampler never saw.
        """
        rng = np.random.default_rng(_SEED)
        logits_np = (rng.random(4) - 0.5) * 6.0
        probs_np = self._numpy_softmax(logits_np)
        action = int(rng.choice(4, p=probs_np))

        scored, _entropy, returned = categorical_logprob_entropy_from_probs(
            torch.as_tensor(probs_np, dtype=torch.float32),
            action,
        )
        expected = torch.distributions.Categorical(
            torch.as_tensor(probs_np, dtype=torch.float32),
        ).log_prob(torch.tensor(action))

        assert torch.equal(scored, expected)
        assert returned.dtype == torch.float32

    def test_residual_is_dominated_by_the_softmax_backend_not_the_cast(self) -> None:
        """What actually limits rollout/update agreement here — measured.

        The rollout builds its probability vector with NumPy and the update with
        torch. Those two vectors already differ by ~4e-7 relative before any
        scoring happens, and that difference — not the dtype of the tensor handed
        to the scorer — is what bounds the ratio.

        An earlier version of this test asserted that ``as_tensor(..., float32)``
        beat ``from_numpy`` for ratio symmetry. It does not: across sample sizes
        the two swap places, so the choice is noise inside the softmax-backend
        gap. The float32 cast is kept for dtype discipline (a float64 tensor has
        no business flowing into an otherwise-float32 pipeline), not because it
        measurably tightens the ratio.
        """
        rng = np.random.default_rng(_SEED)
        worst_cast = worst_raw = worst_softmax_gap = 0.0

        for _ in range(_N):
            logits_np = (rng.random(4) - 0.5) * 6.0
            probs_np = self._numpy_softmax(logits_np)
            probs_t32 = torch.softmax(torch.tensor(logits_np, dtype=torch.float32), dim=-1)
            action = int(rng.choice(4, p=probs_np))

            update, _e, _p = categorical_logprob_entropy_from_probs(probs_t32, action)
            cast, _e, _p = categorical_logprob_entropy_from_probs(
                torch.as_tensor(probs_np, dtype=torch.float32),
                action,
            )
            raw, _e, _p = categorical_logprob_entropy_from_probs(
                torch.from_numpy(probs_np),
                action,
            )
            worst_cast = max(worst_cast, abs(math.exp(float(update) - float(cast)) - 1.0))
            worst_raw = max(worst_raw, abs(math.exp(float(update) - float(raw)) - 1.0))
            worst_softmax_gap = max(
                worst_softmax_gap,
                abs(float(probs_t32[action]) / float(probs_np[action]) - 1.0),
            )

        # Both dtype choices land in the same place, and both are negligible
        # against the clip band.
        assert worst_cast < 1e-6
        assert worst_raw < 1e-6
        # The backend difference is the same order as the residual it explains.
        assert worst_softmax_gap > worst_cast / 10


class TestFamilyCQsnnReinforce:
    """``qsnnreinforce`` is PPO-clipped over an ε-mixture despite its name."""

    @staticmethod
    def _mixture(logits: np.ndarray, epsilon: float, temperature: float) -> np.ndarray:
        """Reproduce the brain's inline NumPy ε-mixture exactly."""
        scaled = logits / temperature
        exp_probs = np.exp(scaled - np.max(scaled))
        softmax_probs = exp_probs / np.sum(exp_probs)
        return (1 - epsilon) * softmax_probs + epsilon * np.ones(len(logits)) / len(logits)

    def test_mixture_is_not_a_softmax_of_the_logits(self) -> None:
        rng = np.random.default_rng(_SEED)
        logits_np = (rng.random(4) - 0.5) * 6.0
        probs = self._mixture(logits_np, 0.1, 1.2)
        action = 1

        from_mixture, _e, _p = categorical_logprob_entropy_from_probs(
            torch.as_tensor(probs, dtype=torch.float32),
            action,
        )
        from_logits, _e, _p = categorical_logprob_entropy_torch(
            torch.tensor(logits_np, dtype=torch.float32),
            action,
        )
        assert not torch.allclose(from_mixture, from_logits, rtol=0, atol=1e-4)

    def test_epsilon_mixture_floors_probability_well_above_the_float32_floor(self) -> None:
        """The exploration floor is why Family C cannot reach the tail regime."""
        rng = np.random.default_rng(_SEED)
        epsilon, n_actions = 0.03, 4
        for _ in range(_N):
            logits_np = (rng.random(n_actions) - 0.5) * 40.0  # deliberately saturating
            probs = self._mixture(logits_np, epsilon, 1.0)
            assert probs.min() >= epsilon / n_actions * 0.999
            assert probs.min() > _FLOAT32_SOFTMAX_FLOOR

    def test_surrogate_with_entropy_kept_separate(self) -> None:
        torch.manual_seed(_SEED)
        log_probs = torch.randn(16)
        old_log_probs = torch.randn(16)
        advantages = torch.randn(16)
        clip_epsilon, coef = 0.2, 0.01
        mean_entropy = torch.tensor(1.1)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        loss_ref = -torch.min(surr1, surr2).mean() - coef * mean_entropy

        loss = (
            ppo_clip_policy_loss(log_probs, old_log_probs, advantages, clip_epsilon)
            - coef * mean_entropy
        )
        assert torch.equal(loss, loss_ref)
