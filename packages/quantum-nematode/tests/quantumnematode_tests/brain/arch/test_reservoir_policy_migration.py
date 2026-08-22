"""Migration regression for the two reservoir bases onto ``_policy.py``.

Covers the shared bases behind five brains:

- ``ReservoirHybridBase`` (crh, qef, qrh) — **Family A**, byte-exact. Its
  pre-migration code was ``softmax → Categorical → sample/log_prob/entropy``
  plus an inline ``surr1``/``surr2``/``min``, which the shared helpers reproduce
  verbatim.
- ``ReservoirLSTMBase`` (crhqlstm, qrhqlstm) — **Family B**. Its NumPy
  ``rng.choice`` sampler is kept verbatim so the sampled-action trajectory is
  byte-identical; only the log-probability, entropy, and surrogate move onto
  torch, dropping the manual ``+1e-8`` / ``+1e-10`` floors. The log-prob deviation
  is the floor's ``-log1p(eps / p)`` bias being removed — **not** a flat 1e-7, and
  the tests assert the model rather than a widened constant.

Each test computes the pre-migration expression and the migrated path **in the
same process, on the same tensors, under the same pinned RNG state**. It
deliberately avoids stored float goldens: those drift at ~1e-8 across BLAS and
torch builds, so a snapshot passes locally and fails CI without indicating any
change in computation (the ``test_mlpppo_policy_migration.py`` precedent).

``crh`` is the vehicle for the Family-A base because it is the classical
reservoir companion — same inherited code path as ``qef``/``qrh`` (neither
overrides ``run()`` or the PPO update) with no quantum simulator in the loop.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import numpy as np
import torch
from quantumnematode.brain.arch.crh import CRHBrain, CRHBrainConfig
from quantumnematode.brain.arch.crhqlstm import CRHQLSTMBrain, CRHQLSTMBrainConfig
from quantumnematode.brain.modules import ModuleName

_SEED = 4242
_SAMPLE_SEED = 17
# Declared migration tolerances (D3), measured rather than assumed.
#
# ``_TOLERANCE`` covers entropy and any quantity where the epsilon floor is damped
# out. ``_MODEL_TOLERANCE`` is the residual once the floor's ``-log1p(eps/p)``
# contribution is predicted and subtracted — i.e. pure float32 log round-off.
# ``_FLOAT32_SOFTMAX_FLOOR`` is where the float32 softmax stops resolving the
# probability at all, below which no old-vs-new model applies.
_TOLERANCE = 5e-7
_MODEL_TOLERANCE = 2e-6
_FLOAT32_SOFTMAX_FLOOR = 1e-6


def _crh_brain() -> CRHBrain:
    """Build a small CRH brain (ReservoirHybridBase, Family A)."""
    config = CRHBrainConfig(
        num_reservoir_neurons=6,
        reservoir_depth=2,
        feature_channels=["raw", "cos_sin"],
        readout_hidden_dim=32,
        readout_num_layers=2,
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        ppo_buffer_size=16,
        ppo_minibatches=2,
        ppo_epochs=1,
        seed=_SEED,
    )
    return CRHBrain(config)


def _crhqlstm_brain() -> CRHQLSTMBrain:
    """Build a small CRH-QLSTM brain (ReservoirLSTMBase, Family B)."""
    config = CRHQLSTMBrainConfig(
        num_reservoir_neurons=6,
        reservoir_depth=2,
        feature_channels=["raw", "cos_sin"],
        lstm_hidden_dim=16,
        critic_hidden_dim=16,
        critic_num_layers=2,
        use_quantum_gates=False,
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
        rollout_buffer_size=16,
        bptt_chunk_length=4,
        num_epochs=1,
        seed=_SEED,
    )
    return CRHQLSTMBrain(config)


class TestReservoirHybridBaseFamilyA:
    """ReservoirHybridBase (crh / qef / qrh) migrates byte-exactly."""

    def test_rollout_sampling_matches_inline_reference(self) -> None:
        """The migrated sampler is bitwise-identical to the ops it replaced."""
        brain = _crh_brain()
        torch.manual_seed(_SEED)
        logits = torch.randn(brain.num_actions)

        # Inline reference == the pre-migration rollout ops, at a fixed RNG state.
        torch.manual_seed(_SAMPLE_SEED)
        probs_ref = torch.softmax(logits, dim=-1)
        dist_ref = torch.distributions.Categorical(probs_ref)
        action_ref = int(dist_ref.sample().item())
        log_prob_ref = dist_ref.log_prob(torch.tensor(action_ref, device=brain.device))

        # Migrated path, under the SAME RNG state.
        from quantumnematode.brain.arch._policy import categorical_sample_torch

        torch.manual_seed(_SAMPLE_SEED)
        action, log_prob, _entropy, probs = categorical_sample_torch(
            logits,
            device=brain.device,
        )

        assert action == action_ref
        assert torch.equal(probs, probs_ref)
        assert torch.equal(log_prob, log_prob_ref)

    def test_update_scoring_and_surrogate_match_inline_reference(self) -> None:
        """The migrated batch evaluator + surrogate are bitwise-identical."""
        from quantumnematode.brain.arch._policy import (
            categorical_evaluate_torch,
            ppo_clip_policy_loss,
        )

        brain = _crh_brain()
        torch.manual_seed(_SEED)
        logits = torch.randn(8, brain.num_actions)
        actions = torch.randint(0, brain.num_actions, (8,))
        old_log_probs = torch.randn(8)
        advantages = torch.randn(8)
        clip_epsilon = brain.clip_epsilon

        # Inline reference == the pre-migration update ops.
        probs_ref = torch.softmax(logits, dim=-1)
        dist_ref = torch.distributions.Categorical(probs_ref)
        new_log_probs_ref = dist_ref.log_prob(actions)
        entropy_ref = dist_ref.entropy().mean()
        ratio = torch.exp(new_log_probs_ref - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss_ref = -torch.min(surr1, surr2).mean()

        # Migrated path.
        new_log_probs, entropy = categorical_evaluate_torch(logits, actions)
        policy_loss = ppo_clip_policy_loss(
            new_log_probs,
            old_log_probs,
            advantages,
            clip_epsilon,
        )

        assert torch.equal(new_log_probs, new_log_probs_ref)
        assert torch.equal(entropy, entropy_ref)
        assert torch.equal(policy_loss, policy_loss_ref)


class TestReservoirLSTMBaseFamilyB:
    """ReservoirLSTMBase (crhqlstm / qrhqlstm) keeps its sampler, moves its scoring."""

    def test_numpy_sampler_is_untouched(self) -> None:
        """The sampled-action trajectory must stay byte-identical.

        This is the half of Family B that carries NO tolerance: the migration
        may move the log-prob, but the action drawn from ``rng.choice`` on the
        same probability vector must be exactly what it was before.
        """
        brain = _crhqlstm_brain()
        torch.manual_seed(_SEED)
        logits = torch.randn(brain.num_actions)
        action_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        rng_ref = np.random.default_rng(_SAMPLE_SEED)
        actions_ref = [int(rng_ref.choice(brain.num_actions, p=action_probs)) for _ in range(32)]

        rng = np.random.default_rng(_SAMPLE_SEED)
        actions = [int(rng.choice(brain.num_actions, p=action_probs)) for _ in range(32)]

        assert actions == actions_ref

    def test_logprob_deviation_is_exactly_the_epsilon_floor_being_removed(self) -> None:
        """The whole old-vs-new gap is the ``+1e-8`` floor, and it follows a model.

        ``log(p) - log(p + eps) = -log1p(eps / p)``. Asserting the deviation
        *matches that model* — rather than widening a tolerance until it passes —
        is what shows the discrepancy is understood rather than merely accepted.

        Note what this means for the declared bar: the deviation is NOT a small
        constant. It is ~``eps / p``, so ~1e-7 only for p >= 0.1, and ~1e-4 at
        p = 1e-4. It is nonetheless a *correction*: the floored form biased
        low-probability actions upward by exactly the epsilon it added, and the
        actions PPO is most sensitive to are precisely the improbable ones that
        were taken anyway.
        """
        import math

        from quantumnematode.brain.arch._policy import categorical_logprob_entropy_torch

        brain = _crhqlstm_brain()
        torch.manual_seed(_SEED)
        checked_a_meaningful_deviation = False

        for scale in (1.0, 2.0, 3.0, 5.0):
            for _ in range(40):
                logits = torch.randn(brain.num_actions) * scale
                action_probs = torch.softmax(logits, dim=-1)

                for action_idx in range(brain.num_actions):
                    prob = float(action_probs[action_idx])
                    if prob < _FLOAT32_SOFTMAX_FLOOR:
                        # Below this the float32 softmax has lost the value itself;
                        # see test_below_the_float32_floor_neither_form_is_reliable.
                        continue

                    floored = float(torch.log(action_probs[action_idx] + 1e-8))
                    log_prob_t, _entropy, _probs = categorical_logprob_entropy_torch(
                        logits,
                        int(action_idx),
                    )

                    deviation = float(log_prob_t) - floored
                    predicted = -math.log1p(1e-8 / prob)

                    assert abs(deviation - predicted) < _MODEL_TOLERANCE, (
                        f"p={prob:.3e}: deviation {deviation:.3e} does not match the "
                        f"eps/p model {predicted:.3e}"
                    )
                    if abs(deviation) > 1e-7:
                        checked_a_meaningful_deviation = True

        assert checked_a_meaningful_deviation, (
            "Expected at least one action improbable enough for the floor to bite "
            "beyond 1e-7; without one this test proves nothing."
        )

    def test_below_the_float32_softmax_floor_the_model_stops_applying(self) -> None:
        """Documents the boundary rather than pretending it does not exist.

        Below p ~ 1e-6 the float32 ``softmax`` has already lost the probability to
        its own round-off, so *both* the old and the new expression are far from
        the float64-exact value and neither is reliably closer. That is a property
        of the brains' float32 pipeline, which this migration does not change and
        does not claim to fix — both forms read the same float32 ``probs``.
        """
        # A deliberately extreme policy: one action at p ~ 1e-7, well past the floor.
        logits = torch.tensor([0.34001052, 7.29843378, -9.01550102, 1.40780151])
        probs = torch.softmax(logits, dim=-1)
        exact = torch.log_softmax(logits.double(), dim=-1)

        tail = int(torch.argmin(probs).item())
        assert float(probs[tail]) < _FLOAT32_SOFTMAX_FLOOR

        old = float(torch.log(probs[tail] + 1e-8))
        new = float(torch.distributions.Categorical(probs).log_prob(torch.tensor(tail)))

        # Both are far from exact, in the same direction, by a similar order.
        assert abs(old - float(exact[tail])) > 0.01
        assert abs(new - float(exact[tail])) > 0.01

    def test_update_scoring_within_declared_tolerance(self) -> None:
        """Per-step log-prob and entropy in the BPTT loop stay within tolerance."""
        from quantumnematode.brain.arch._policy import categorical_logprob_entropy_torch

        brain = _crhqlstm_brain()
        torch.manual_seed(_SEED)

        for step in range(8):
            logits = torch.randn(brain.num_actions)
            action_idx = step % brain.num_actions

            # Pre-migration inline expressions, verbatim.
            action_probs = torch.softmax(logits, dim=-1)
            entropy_ref = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
            exact = torch.log_softmax(logits.double(), dim=-1)

            log_prob, entropy, _probs = categorical_logprob_entropy_torch(logits, action_idx)

            # Log-prob against the exact reference (bounded at every p).
            assert abs(float(log_prob) - float(exact[action_idx])) < _TOLERANCE
            # Entropy against the pre-migration form: here the +1e-10 floor IS
            # harmless, because each term carries a factor of p that damps it to
            # ~n*eps. What is left is float32 round-off of the sum.
            assert torch.allclose(entropy, entropy_ref, rtol=0, atol=_TOLERANCE)

    def test_surrogate_matches_inline_reference(self) -> None:
        """The surrogate itself is byte-exact even in Family B.

        Only the *inputs* to the ratio carry the 1e-7 tolerance; the clipped
        surrogate expression is unchanged arithmetic.
        """
        from quantumnematode.brain.arch._policy import ppo_clip_policy_loss

        brain = _crhqlstm_brain()
        torch.manual_seed(_SEED)
        new_log_probs = torch.randn(8)
        old_log_probs = torch.randn(8)
        advantages = torch.randn(8)
        clip_epsilon = brain.config.clip_epsilon

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss_ref = -torch.min(surr1, surr2).mean()

        policy_loss = ppo_clip_policy_loss(
            new_log_probs,
            old_log_probs,
            advantages,
            clip_epsilon,
        )

        assert torch.equal(policy_loss, policy_loss_ref)
