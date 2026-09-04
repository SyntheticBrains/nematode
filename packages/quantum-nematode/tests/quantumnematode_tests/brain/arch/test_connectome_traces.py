# pyright: reportPrivateUsage=false
"""Tests for the persistent activity-trace substrate (eligibility traces).

Covers the `Persistent activity-trace substrate` requirement of
``openspec/changes/add-l4-trace-substrate/specs/connectome-ppo-brain/spec.md``:
byte-identical when off (034 ``TestWiringControl`` template), closed-form
decay recurrence when on, masked support, episode-boundary reset, PPO-path
independence, and training bit-invariance while no rule consumes ``E``.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.learning_rules.ppo import ConnectomePPOBatch

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


def _drive(brain: ConnectomePPOBrain, n_steps: int = _N_STEPS, *, final_done: bool) -> None:
    for step in range(n_steps):
        params = _make_params(strength=0.3 + 0.05 * step, angle=0.1 * step - 0.3)
        brain.run_brain(params, reward=None, input_data=None, top_only=False, top_randomize=False)
        brain.learn(
            params,
            reward=0.1 * (step % 3),
            episode_done=(final_done and step == n_steps - 1),
        )


class TestTracesOff:
    """`Traces off is byte-identical` scenario."""

    def test_off_is_byte_identical_to_default(self) -> None:
        default = _make_brain()
        explicit_off = _make_brain(enable_activity_traces=False)
        for name in ("m_chem", "w_chem", "g_gap", "food_gains", "readout"):
            assert torch.equal(
                getattr(default.topology, name),
                getattr(explicit_off.topology, name),
            )

    def test_off_allocates_no_buffer(self) -> None:
        brain = _make_brain()
        assert not hasattr(brain.topology, "activity_traces")
        assert "activity_traces" not in dict(brain.topology.named_buffers())

    def test_on_leaves_weight_init_byte_identical(self) -> None:
        off = _make_brain()
        on = _make_brain(enable_activity_traces=True)
        for name in ("m_chem", "w_chem", "g_gap", "food_gains", "readout"):
            assert torch.equal(getattr(off.topology, name), getattr(on.topology, name))


class TestTraceRecurrence:
    """`Trace recurrence matches the closed form` scenario."""

    def test_recurrence_matches_manual_accumulation(self) -> None:
        brain = _make_brain(enable_activity_traces=True, trace_decay=0.8)
        topology = brain.topology
        expected = torch.zeros_like(topology.activity_traces)
        torch.manual_seed(_DRIVE_SEED)
        for step in range(5):
            params = _make_params(strength=0.2 + 0.1 * step, angle=0.05 * step)
            brain.run_brain(
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            # ``run_brain`` stores the post-K hidden implicitly; recompute the
            # expected recurrence from the same settled state by re-running the
            # forward on the buffered pending state with traces detoured.
            state_t = torch.tensor(brain._pending_state, dtype=torch.float32)
            food, distal, mechano, zone, thermo = brain._unpack_state(state_t)
            with torch.no_grad():
                topology.enable_activity_traces = False
                _, h = topology.forward_with_hidden(
                    food,
                    predator_distal_features=distal,
                    predator_mechano_features=mechano,
                    predator_contact_zone=zone,
                    thermotaxis_features=thermo,
                )
                topology.enable_activity_traces = True
            expected = 0.8 * expected + topology.m_chem * torch.outer(h, h)
            assert torch.allclose(topology.activity_traces, expected, rtol=0.0, atol=0.0)

    def test_traces_are_zero_off_edges(self) -> None:
        brain = _make_brain(enable_activity_traces=True)
        torch.manual_seed(_DRIVE_SEED)
        _drive(brain, n_steps=4, final_done=False)
        traces = brain.topology.activity_traces
        off_edges = brain.topology.m_chem == 0
        assert torch.all(traces[off_edges] == 0)
        assert traces.abs().sum() > 0

    def test_deterministic_across_identical_runs(self) -> None:
        brains = []
        for _ in range(2):
            brain = _make_brain(enable_activity_traces=True)
            torch.manual_seed(_DRIVE_SEED)
            _drive(brain, n_steps=4, final_done=False)
            brains.append(brain)
        assert torch.equal(brains[0].topology.activity_traces, brains[1].topology.activity_traces)


class TestTraceLifecycle:
    """`Traces reset at episode boundaries` scenario."""

    def test_prepare_episode_zeroes_traces(self) -> None:
        brain = _make_brain(enable_activity_traces=True)
        torch.manual_seed(_DRIVE_SEED)
        _drive(brain, n_steps=4, final_done=False)
        assert brain.topology.activity_traces.abs().sum() > 0
        brain.prepare_episode()
        assert torch.all(brain.topology.activity_traces == 0)

    def test_reset_traces_is_noop_when_disabled(self) -> None:
        brain = _make_brain()
        brain.prepare_episode()  # must not raise


class TestTrainingBitInvariance:
    """`Training is bit-invariant while no rule consumes traces` scenario."""

    def test_parameters_bit_equal_traces_on_vs_off(self) -> None:
        brain_off = _make_brain()
        brain_on = _make_brain(enable_activity_traces=True)
        torch.manual_seed(_DRIVE_SEED)
        _drive(brain_off, final_done=True)
        torch.manual_seed(_DRIVE_SEED)
        _drive(brain_on, final_done=True)
        assert len(brain_off.history_data.losses) >= 1  # the update genuinely ran
        params_off = [*brain_off.topology.learnable_parameters, *brain_off.critic.parameters()]
        params_on = [*brain_on.topology.learnable_parameters, *brain_on.critic.parameters()]
        for p_off, p_on in zip(params_off, params_on, strict=True):
            assert torch.equal(p_off, p_on)

    def test_batched_update_leaves_traces_unchanged(self) -> None:
        brain = _make_brain(enable_activity_traces=True)
        torch.manual_seed(_DRIVE_SEED)
        _drive(brain, final_done=False)
        before = brain.topology.activity_traces.detach().clone()
        brain._rule.step(
            brain.topology,
            ConnectomePPOBatch(
                buffer=brain.buffer,
                unpack_batched=brain._unpack_state_batched,
                last_value=brain.last_value,
            ),
        )
        assert torch.equal(brain.topology.activity_traces, before)


class TestTraceConfigValidation:
    """`trace_decay` bounds enforce at load (tracker Decision B.6)."""

    @pytest.mark.parametrize("bad_decay", [-0.1, 1.0, 1.5])
    def test_out_of_range_trace_decay_raises(self, bad_decay: float) -> None:
        with pytest.raises(ValidationError):
            ConnectomePPOBrainConfig(seed=_SEED, trace_decay=bad_decay)

    def test_in_range_trace_decay_accepted(self) -> None:
        cfg = ConnectomePPOBrainConfig(seed=_SEED, trace_decay=0.0)
        assert cfg.trace_decay == 0.0
