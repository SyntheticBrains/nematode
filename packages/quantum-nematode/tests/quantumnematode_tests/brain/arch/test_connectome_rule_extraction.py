# pyright: reportPrivateUsage=false
"""Byte-equivalence suite for the ConnectomePPORule extraction.

Covers the `PPO update routed through the learning-rule seam` requirement of
``openspec/changes/add-l4-trace-substrate/specs/connectome-ppo-brain/spec.md``:
the extracted rule must produce bit-equal parameters to the frozen
pre-extraction reference (``_legacy_connectome_update_reference.py``, M1
pattern) from identical state, with identical RNG streams afterwards. No
golden float constants (policy-migration precedent — they drift across BLAS
builds); every assertion is a same-process two-construction comparison.
"""

from __future__ import annotations

import subprocess
import sys

import torch

from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.learning_rules.ppo import ConnectomePPOBatch

from ._legacy_connectome_update_reference import legacy_perform_ppo_update

_SEED = 2026
_DRIVE_SEED = 4053
_N_STEPS = 8


def _make_brain(**cfg_overrides: object) -> ConnectomePPOBrain:
    cfg = ConnectomePPOBrainConfig(seed=_SEED, **cfg_overrides)  # type: ignore[arg-type]
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _make_params(strength: float = 0.42, angle: float = 0.13) -> BrainParams:
    return BrainParams(
        food_gradient_strength=strength,
        food_gradient_direction=angle,
    )


def _drive_without_update(brain: ConnectomePPOBrain, n_steps: int = _N_STEPS) -> None:
    """Fill the rollout buffer deterministically WITHOUT triggering an update.

    ``episode_done`` stays False and the buffer (default size 2048) never
    fills, so both brains reach the update call with identical buffered
    experience. The torch global RNG is re-seeded by the caller before each
    drive so action sampling matches across brains.
    """
    for step in range(n_steps):
        params = _make_params(strength=0.3 + 0.05 * step, angle=0.1 * step - 0.3)
        brain.run_brain(params, reward=None, input_data=None, top_only=False, top_randomize=False)
        brain.learn(params, reward=0.1 * (step % 3), episode_done=False)


def _all_learnables(brain: ConnectomePPOBrain) -> list[torch.Tensor]:
    return [
        *brain.topology.learnable_parameters,
        *brain.critic.parameters(),
    ]


class TestRuleExtractionByteEquivalence:
    """Extracted rule vs frozen pre-extraction reference: bit-equal."""

    def _assert_equivalent(self, **cfg_overrides: object) -> None:
        brain_legacy = _make_brain(**cfg_overrides)
        brain_rule = _make_brain(**cfg_overrides)

        # Identical initial parameters (both __init__s reset the global seed).
        for p_a, p_b in zip(
            _all_learnables(brain_legacy),
            _all_learnables(brain_rule),
            strict=True,
        ):
            assert torch.equal(p_a, p_b)

        torch.manual_seed(_DRIVE_SEED)
        _drive_without_update(brain_legacy)
        torch.manual_seed(_DRIVE_SEED)
        _drive_without_update(brain_rule)

        rng_before = torch.get_rng_state()
        legacy_perform_ppo_update(brain_legacy)
        rng_after_legacy = torch.get_rng_state()
        torch.set_rng_state(rng_before)
        report = brain_rule._rule.step(
            brain_rule.topology,
            ConnectomePPOBatch(
                buffer=brain_rule.buffer,
                unpack_batched=brain_rule._unpack_state_batched,
                last_value=brain_rule.last_value,
            ),
        )
        rng_after_rule = torch.get_rng_state()

        for p_a, p_b in zip(
            _all_learnables(brain_legacy),
            _all_learnables(brain_rule),
            strict=True,
        ):
            assert torch.equal(p_a, p_b)
        assert torch.equal(rng_after_legacy, rng_after_rule)
        assert report.policy_loss is not None
        assert report.total_loss is not None

    def test_discrete_strict(self) -> None:
        self._assert_equivalent()

    def test_discrete_soft_prior(self) -> None:
        self._assert_equivalent(chemical_mask_mode="soft_prior")

    def test_continuous_strict(self) -> None:
        self._assert_equivalent(action_mode="continuous")


class TestFreezeShortCircuit:
    """Freeze / empty-buffer paths return a report with None loss fields."""

    def test_freeze_updates_returns_none_report_and_moves_nothing(self) -> None:
        brain = _make_brain(freeze_updates=True)
        torch.manual_seed(_DRIVE_SEED)
        _drive_without_update(brain)
        before = [p.detach().clone() for p in _all_learnables(brain)]
        report = brain._rule.step(
            brain.topology,
            ConnectomePPOBatch(
                buffer=brain.buffer,
                unpack_batched=brain._unpack_state_batched,
                last_value=brain.last_value,
            ),
        )
        assert report.policy_loss is None
        assert report.total_loss is None
        for p_before, p_after in zip(before, _all_learnables(brain), strict=True):
            assert torch.equal(p_before, p_after)

    def test_empty_buffer_returns_none_report(self) -> None:
        brain = _make_brain()
        report = brain._rule.step(
            brain.topology,
            ConnectomePPOBatch(
                buffer=brain.buffer,
                unpack_batched=brain._unpack_state_batched,
                last_value=brain.last_value,
            ),
        )
        assert report.policy_loss is None


class TestLossTelemetry:
    """`Loss telemetry flows to tracking` scenario."""

    def test_update_appends_finite_policy_loss(self) -> None:
        brain = _make_brain()
        torch.manual_seed(_DRIVE_SEED)
        for step in range(_N_STEPS):
            params = _make_params()
            brain.run_brain(
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            brain.learn(params, reward=0.1, episode_done=(step == _N_STEPS - 1))
        assert len(brain.history_data.losses) >= 1
        assert all(torch.isfinite(torch.tensor(loss)) for loss in brain.history_data.losses)
        assert brain.latest_data.loss is not None

    def test_frozen_updates_append_nothing(self) -> None:
        brain = _make_brain(freeze_updates=True)
        torch.manual_seed(_DRIVE_SEED)
        for step in range(_N_STEPS):
            params = _make_params()
            brain.run_brain(
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            brain.learn(params, reward=0.1, episode_done=(step == _N_STEPS - 1))
        assert brain.history_data.losses == []


class TestImportDiscipline:
    """Decision 3b: the rule package survives being the first import."""

    def test_learning_rules_ppo_as_first_import(self) -> None:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", "import quantumnematode.learning_rules.ppo"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
