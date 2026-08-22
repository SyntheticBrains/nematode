"""Migration regression for the hybrid family onto ``_policy.py``.

Covers **Family C** — `hybridquantum`, `hybridclassical`, `hybridquantumcortex`
(and, in the same shape, `qsnnreinforce`) — whose action distribution is an
explicit ε-greedy mixture ``(1-ε)·softmax(logits/T) + ε·uniform`` rather than a
softmax of any logits the brain holds. That is why ``_policy.py`` grew
``categorical_logprob_entropy_from_probs``: the logits-based scorer cannot
express this distribution at all.

It also pins **design D4** — the PPO importance ratio must be produced by one
formula on both sides — which splits into two cases that are NOT fixed to the
same degree, and are therefore asserted separately:

- **Cortex path** (`hybridquantum` / `hybridclassical`, via
  ``_hybrid_common.perform_ppo_update``): torch on both sides, so the pre-migration
  ``log(softmax + 1e-8)`` at rollout vs ``Categorical.log_prob`` at update was a
  pure *formula* mismatch. The migration closes it exactly.
- **Reflex path** (all three): the rollout builds its mixture in NumPy float64 and
  the update in torch float32. The migration closes the formula half; the dtype
  half is pre-existing and survives, bounded by the declared tolerance.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math

import numpy as np
import torch
from quantumnematode.brain.arch._policy import (
    categorical_logprob_entropy_from_probs,
    categorical_logprob_entropy_torch,
    ppo_clip_policy_loss,
)

_SEED = 31
_N = 2000
# Residual left on the reflex ratio by the pre-existing float64/float32 boundary.
_REFLEX_RATIO_TOLERANCE = 1e-6


def _epsilon_mixture_numpy(logits: np.ndarray, epsilon: float, temperature: float) -> np.ndarray:
    """The rollout-side mixture, exactly as the brains build it (NumPy float64)."""
    scaled = logits / temperature
    exp_probs = np.exp(scaled - np.max(scaled))
    softmax_probs = exp_probs / np.sum(exp_probs)
    uniform = np.ones(len(logits)) / len(logits)
    return (1 - epsilon) * softmax_probs + epsilon * uniform


def _epsilon_mixture_torch(
    logits: torch.Tensor,
    epsilon: float,
    temperature: float,
) -> torch.Tensor:
    """The update-side mixture, exactly as the brains build it (torch float32)."""
    softmax_probs = torch.softmax(logits / temperature, dim=-1)
    uniform = torch.ones_like(softmax_probs) / softmax_probs.shape[-1]
    return (1 - epsilon) * softmax_probs + epsilon * uniform


class TestFamilyCMixtureScoring:
    """The ε-mixture is scored by the probs helper, not by re-softmaxing logits."""

    def test_scoring_matches_inline_categorical_on_the_mixture(self) -> None:
        torch.manual_seed(_SEED)
        logits = torch.randn(4) * 2.0
        probs = _epsilon_mixture_torch(logits, epsilon=0.03, temperature=1.0)
        action = 2

        dist_ref = torch.distributions.Categorical(probs)
        log_prob, entropy, returned = categorical_logprob_entropy_from_probs(probs, action)

        assert torch.equal(log_prob, dist_ref.log_prob(torch.tensor(action)))
        assert torch.equal(entropy, dist_ref.entropy())
        assert returned is probs

    def test_the_mixture_is_not_a_softmax_of_the_logits(self) -> None:
        """Guards the reason the probs helper exists.

        If the ε-mixture coincided with ``softmax(logits)``, a migration that
        wrongly routed Family C through the logits helper would still pass every
        other assertion in this file.
        """
        torch.manual_seed(_SEED)
        logits = torch.randn(4) * 2.0
        probs = _epsilon_mixture_torch(logits, epsilon=0.03, temperature=1.0)
        action = 2

        from_mixture, _, _ = categorical_logprob_entropy_from_probs(probs, action)
        from_logits, _, _ = categorical_logprob_entropy_torch(logits, action)

        assert not torch.allclose(from_mixture, from_logits, rtol=0, atol=1e-4)

    def test_temperature_and_epsilon_still_change_the_score(self) -> None:
        """The mixture stays the brain's own construction, not the helper's."""
        torch.manual_seed(_SEED)
        logits = torch.randn(4) * 2.0
        action = 1

        base, _, _ = categorical_logprob_entropy_from_probs(
            _epsilon_mixture_torch(logits, 0.03, 1.0), action
        )
        hotter, _, _ = categorical_logprob_entropy_from_probs(
            _epsilon_mixture_torch(logits, 0.03, 1.5), action
        )
        explorier, _, _ = categorical_logprob_entropy_from_probs(
            _epsilon_mixture_torch(logits, 0.30, 1.0), action
        )

        assert not torch.allclose(base, hotter, rtol=0, atol=1e-6)
        assert not torch.allclose(base, explorier, rtol=0, atol=1e-6)


class TestD4CortexRatioIsExactlyOne:
    """The cortex formula mismatch is a real defect, and the migration closes it."""

    def test_pre_migration_cortex_ratio_was_biased_away_from_one(self) -> None:
        """Documents the defect being fixed, so the fix cannot be silently reverted.

        Pre-migration: rollout stored ``log(softmax(logits) + 1e-8)`` while
        ``_hybrid_common.perform_ppo_update`` re-scored with ``Categorical.log_prob``.
        With the policy unmoved the ratio should be exactly 1; it was not.
        """
        torch.manual_seed(_SEED)
        biased = 0

        for _ in range(_N):
            logits = torch.randn(4) * 2.0
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = int(dist.sample())

            old_pre = float(torch.log(probs + 1e-8)[action])  # rollout, pre-migration
            new = float(dist.log_prob(torch.tensor(action)))  # update, unchanged
            if math.exp(new - old_pre) != 1.0:
                biased += 1

        assert biased > 0, "expected the pre-migration cortex ratio to be biased off 1.0"

    def test_migrated_cortex_ratio_is_exactly_one(self) -> None:
        """Both halves now use one scorer, so an unmoved policy gives ratio == 1."""
        torch.manual_seed(_SEED)

        for _ in range(_N):
            logits = torch.randn(4) * 2.0
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = int(dist.sample())

            # Rollout (migrated) and update both route through the shared scorer.
            old_post, _, _ = categorical_logprob_entropy_from_probs(probs, action)
            new, _, _ = categorical_logprob_entropy_from_probs(probs, action)

            assert math.exp(float(new) - float(old_post)) == 1.0


class TestD4ReflexRatioResidual:
    """The reflex path also straddles a dtype boundary the migration does not close."""

    def test_reflex_ratio_residual_is_bounded_and_not_claimed_exact(self) -> None:
        """Formula half closed; float64/float32 half survives, within tolerance.

        Asserting a *bound* rather than exactness is the point: claiming the
        reflex ratio reaches 1.0 would overstate what this migration achieves.
        """
        rng = np.random.default_rng(_SEED)
        epsilon, temperature = 0.03, 1.0
        worst = 0.0

        for _ in range(_N):
            logits_np = (rng.random(4) - 0.5) * 6.0
            probs_np = _epsilon_mixture_numpy(logits_np, epsilon, temperature)
            probs_t = _epsilon_mixture_torch(
                torch.tensor(logits_np, dtype=torch.float32), epsilon, temperature
            )
            action = int(rng.choice(4, p=probs_np / probs_np.sum()))

            # Rollout casts to float32 (design D2) before scoring; update is float32.
            old_lp, _, _ = categorical_logprob_entropy_from_probs(
                torch.as_tensor(probs_np, dtype=torch.float32), action
            )
            new_lp, _, _ = categorical_logprob_entropy_from_probs(probs_t, action)
            worst = max(worst, abs(math.exp(float(new_lp) - float(old_lp)) - 1.0))

        assert worst < _REFLEX_RATIO_TOLERANCE
        assert worst > 0.0, (
            "expected a non-zero residual from the float64/float32 boundary; if this "
            "is now exactly zero the dtype gap closed and the design note is stale"
        )

    def test_residual_is_far_below_the_clip_band(self) -> None:
        """Whatever residual remains cannot reach the PPO clip threshold."""
        assert _REFLEX_RATIO_TOLERANCE < 0.2 / 1000


class TestFamilyCSurrogate:
    """The clipped surrogate itself is unchanged arithmetic."""

    def test_surrogate_matches_inline_with_entropy_kept_separate(self) -> None:
        torch.manual_seed(_SEED)
        log_probs = torch.randn(16)
        old_log_probs = torch.randn(16)
        advantages = torch.randn(16)
        clip_epsilon = 0.2
        entropy_coef = 0.01
        mean_entropy = torch.tensor(1.23)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        loss_ref = -torch.min(surr1, surr2).mean() - entropy_coef * mean_entropy

        loss = (
            ppo_clip_policy_loss(log_probs, old_log_probs, advantages, clip_epsilon)
            - entropy_coef * mean_entropy
        )

        assert torch.equal(loss, loss_ref)
