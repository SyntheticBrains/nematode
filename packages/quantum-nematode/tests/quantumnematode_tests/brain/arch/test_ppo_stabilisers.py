"""PPO stabilisers and the placebo shuffle, on the connectome PPO rule and the MLP-PPO brain.

Covers the learning-rules requirement "PPO stabilisers and the placebo shuffle": the KL stop ends
the update before the crossing step; weight clipping bounds each tensor by its construction scale;
Adam's settings survive a load; the placebo changes only the minibatch order; off is byte-identical.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams, MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.arch import _ppo_stabilisers as stab
from quantumnematode.brain.arch import _telemetry as tm
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.modules import ModuleName

if TYPE_CHECKING:
    from pathlib import Path


# ── The pieces ───────────────────────────────────────────────────────────────────────────────


class TestPieces:
    def test_adam_keeps_pytorch_s_defaults_when_off(self) -> None:
        param = torch.nn.Parameter(torch.ones(2))
        group = stab.make_adam([param], 0.01, stab.PPOStabiliserConfig()).param_groups[0]
        assert group["eps"] == 1e-8
        assert group["betas"] == (0.9, 0.999)

    def test_adam_takes_the_settings(self) -> None:
        param = torch.nn.Parameter(torch.ones(2))
        config = stab.PPOStabiliserConfig(adam_eps=1e-4, adam_beta2=0.99)
        group = stab.make_adam([param], 0.01, config).param_groups[0]
        assert group["eps"] == 1e-4
        assert group["betas"] == (0.9, 0.99)

    def test_the_settings_survive_a_load(self) -> None:
        """Scenario "Adam's settings survive a load"."""
        param = torch.nn.Parameter(torch.ones(2))
        saved = stab.make_adam([param], 0.01, stab.PPOStabiliserConfig()).state_dict()
        config = stab.PPOStabiliserConfig(adam_eps=1e-4, adam_beta2=0.99)
        optimizer = stab.make_adam([param], 0.01, config)
        optimizer.load_state_dict(saved)
        assert optimizer.param_groups[0]["eps"] == 1e-8
        stab.reapply_adam(optimizer, config)
        assert optimizer.param_groups[0]["eps"] == 1e-4
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.99)

    def test_weight_clip_bounds_by_construction_scale(self) -> None:
        """Scenario "Weight clipping bounds each tensor by its construction scale"."""
        weight = torch.nn.Parameter(torch.tensor([0.5, -1.0]))
        bias = torch.nn.Parameter(torch.zeros(2))
        clip = stab.WeightClip([weight, bias], 2.0)
        with torch.no_grad():
            weight.copy_(torch.tensor([5.0, -0.1]))
            bias.copy_(torch.tensor([7.0, -7.0]))
        clip.apply()
        assert weight.tolist() == [2.0, pytest.approx(-0.1)]
        assert bias.tolist() == [7.0, -7.0]

    def test_the_kl_stop_waits_for_its_window_then_reads_the_median(self) -> None:
        stop = stab.KLStop(3.0, window=4)
        assert not stop.exceeded(100.0)
        for kl in (0.01, 0.02, 0.01, 0.03):
            stop.record(kl)
        assert not stop.exceeded(0.045)
        assert stop.exceeded(0.046)

    def test_the_gate_counts_steps_and_records_the_largest(self) -> None:
        stop = stab.KLStop(1.0, window=1)
        stop.record(0.001)
        gate = stab.UpdateGate(stop)
        same = torch.zeros(4)
        assert gate.allow(same, same)
        assert not gate.allow(same + 1.0, same)
        assert gate.stopped
        assert gate.stepped == 1
        gate.close()
        assert stop.history[-1] == pytest.approx(stab.approx_kl(same + 1.0, same))

    def test_without_a_stop_every_minibatch_steps(self) -> None:
        gate = stab.UpdateGate(None)
        assert all(gate.allow(torch.zeros(2), torch.ones(2)) for _ in range(3))
        assert gate.stepped == 3
        assert not gate.stopped

    def test_the_placebo_stream_is_its_own(self) -> None:
        a, b = stab.shuffle_rng(7), stab.shuffle_rng(7)
        assert a.random() == b.random()
        assert stab.shuffle_rng(7).random() != np.random.default_rng(7).random()


# ── On the brains ────────────────────────────────────────────────────────────────────────────


def _connectome(**overrides: object) -> ConnectomePPOBrain:
    torch.manual_seed(0)
    config = ConnectomePPOBrainConfig(
        seed=5,
        action_mode="continuous",
        rollout_buffer_size=16,
        num_minibatches=2,
        num_epochs=4,
        **overrides,  # type: ignore[arg-type]
    )
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU)


def _mlp(**overrides: object) -> MLPPPOBrain:
    torch.manual_seed(0)
    config = MLPPPOBrainConfig(
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
        action_mode="continuous",
        rollout_buffer_size=16,
        num_minibatches=2,
        num_epochs=4,
        seed=5,
        **overrides,  # type: ignore[arg-type]
    )
    return MLPPPOBrain(config=config)


def _rule(brain: ConnectomePPOBrain) -> Any:
    rule = brain._ppo_rule
    assert rule is not None
    return rule


def _drive(brain: Any, steps: int = 48) -> None:
    torch.manual_seed(1)
    for step in range(steps):
        params = BrainParams(
            food_concentration=0.1 * (step % 5),
            food_lateral_gradient=0.05 * (step % 3) - 0.05,
            food_dconcentration_dt=0.01 * step,
        )
        brain.run_brain(params, reward=None, input_data=None, top_only=False, top_randomize=False)
        brain.learn(params, reward=0.1 * (step % 3), episode_done=step % 13 == 12)


def _state(brain: Any) -> dict[str, torch.Tensor]:
    return {
        f"{c.name}.{k}": v.detach().clone()
        for c in brain.get_weight_components().values()
        for k, v in c.state.items()
        if isinstance(v, torch.Tensor)
    }


@pytest.mark.parametrize("build", [_connectome, _mlp], ids=["connectome", "mlp"])
def test_off_is_byte_identical(build: Any) -> None:
    """Scenario "Off is byte-identical": explicit defaults are the implicit build, run for run."""
    implicit = build()
    explicit = build(
        kl_stop_factor=None,
        weight_clip=None,
        adam_eps=None,
        adam_beta2=None,
        rollout_shuffle_stream=False,
    )
    _drive(implicit)
    _drive(explicit)
    a, b = _state(implicit), _state(explicit)
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert implicit.rng.random() == explicit.rng.random()
    assert implicit.buffer.rng.random() == explicit.buffer.rng.random()
    training = implicit.get_weight_components()["training_state"].state
    assert not {k for k in training if "kl" in k or "clip" in k or "adam" in k or "shuffle" in k}


@pytest.mark.parametrize("build", [_connectome, _mlp], ids=["connectome", "mlp"])
def test_the_kl_stop_ends_updates_and_is_recorded(build: Any, tmp_path: Path) -> None:
    """Scenario "The KL stop ends the update before the crossing step"."""
    brain = build(kl_stop_factor=1e-6)
    stop = stab.KLStop(1e-6, window=1)
    if isinstance(brain, ConnectomePPOBrain):
        _rule(brain).kl_stop = stop
    else:
        brain._kl_stop = stop
    sink = tm.UpdateTelemetry(tmp_path / "updates.jsonl")
    brain.set_update_telemetry(sink)
    _drive(brain)
    sink.close()
    lines = [json.loads(x) for x in (tmp_path / "updates.jsonl").read_text().splitlines()]

    def full(line: dict[str, Any]) -> int:
        size = line["batch_size"]
        return 4 * -(-size // max(1, size // 2))

    assert any(line["kl_stopped"] for line in lines[1:])
    for line in lines:
        assert line["minibatches_stepped"] <= full(line)
        if line["kl_stopped"]:
            assert line["minibatches_stepped"] < full(line)


@pytest.mark.parametrize("build", [_connectome, _mlp], ids=["connectome", "mlp"])
def test_weight_clip_holds_every_tensor_at_its_construction_scale(build: Any) -> None:
    """At kappa 1, no clipped tensor grows past its largest magnitude at construction."""
    brain = build(weight_clip=1.0)
    clip = _rule(brain).weight_clip if isinstance(brain, ConnectomePPOBrain) else brain._weight_clip
    assert clip is not None
    assert clip.bounds
    _drive(brain)
    for param, bound in clip.bounds:
        assert float(param.detach().abs().max()) <= bound + 1e-6


@pytest.mark.parametrize("build", [_connectome, _mlp], ids=["connectome", "mlp"])
def test_the_optimiser_carries_the_adam_settings(build: Any) -> None:
    """Both learners' optimisers are built with the configured second-moment settings."""
    brain = build(adam_eps=1e-4, adam_beta2=0.99)
    optimizer = _rule(brain).optimizer if isinstance(brain, ConnectomePPOBrain) else brain.optimizer
    assert optimizer.param_groups[0]["eps"] == 1e-4
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.99)


@pytest.mark.parametrize("build", [_connectome, _mlp], ids=["connectome", "mlp"])
def test_the_placebo_changes_the_order_and_spares_the_brain_s_generator(build: Any) -> None:
    """Scenario "The placebo changes only the minibatch order"."""
    off, on = build(), build(rollout_shuffle_stream=True)
    assert on.buffer.rng is not on.rng
    before = on.rng.bit_generator.state
    _drive(on)
    _drive(off)
    assert on.rng.bit_generator.state == before
    a, b = _state(off), _state(on)
    assert any(not torch.equal(a[k], b[k]) for k in a)
