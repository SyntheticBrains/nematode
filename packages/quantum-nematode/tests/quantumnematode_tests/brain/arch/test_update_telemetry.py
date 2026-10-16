"""Per-update PPO telemetry on the MLP-PPO and connectome PPO brains.

Covers the training-telemetry requirement "Per-update PPO telemetry": each statistic against a hand
computation; scenario "Telemetry does not change the run" (outputs, final weights and the next value
of every generator are identical with it on and off); scenario "One line per update".
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams, MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.arch import _telemetry as tm
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.modules import ModuleName

if TYPE_CHECKING:
    from pathlib import Path

_FIELDS = {
    "episode",
    "update",
    "entropy",
    "log_std",
    "approx_kl",
    "clip_fraction",
    "value_loss",
    "explained_variance",
    "grad_norm",
    "clipped_minibatch_fraction",
    "actor_grad",
    "critic_grad",
    "adv_mean",
    "adv_std",
    "adv_max_abs",
    "action_mean_abs",
    "action_mean_large_fraction",
    "weight_norms",
    "saturated_fraction",
    "srank",
}


# ── Hand computations ────────────────────────────────────────────────────────────────────────


class TestStatistics:
    def test_approx_kl_is_the_low_variance_estimate(self) -> None:
        new, old = torch.tensor([0.0, -1.0]), torch.tensor([-0.5, -1.0])
        r = math.exp(0.5)
        assert tm.approx_kl(new, old) == pytest.approx(((r - 1 - 0.5) + 0.0) / 2)

    def test_identical_policies_have_zero_kl_and_no_clipping(self) -> None:
        logp = torch.tensor([-0.3, -1.2, -2.0])
        assert tm.approx_kl(logp, logp) == 0.0
        assert tm.clip_fraction(logp, logp, 0.2) == 0.0

    def test_clip_fraction_counts_ratios_outside_the_band(self) -> None:
        old = torch.zeros(4)
        new = torch.log(torch.tensor([1.0, 1.1, 1.3, 0.7]))
        assert tm.clip_fraction(new, old, 0.2) == pytest.approx(0.5)

    def test_explained_variance(self) -> None:
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert tm.explained_variance(returns, returns) == pytest.approx(1.0)
        assert tm.explained_variance(returns, torch.full((4,), 2.5)) == pytest.approx(0.0)
        assert tm.explained_variance(torch.ones(3), torch.zeros(3)) is None

    def test_srank_reads_a_fixed_number_of_rows(self) -> None:
        record = tm.UpdateRecord(
            returns=torch.arange(200.0),
            raw_advantages=torch.ones(200),
            old_values=torch.zeros(200),
            clip_epsilon=0.2,
            max_grad_norm=0.5,
        )
        record.begin_epoch()
        record.read_grads([], [])
        record.minibatch(
            new_log_probs=torch.zeros(200),
            old_log_probs=torch.zeros(200),
            head_out=torch.zeros(200, 2),
            features=torch.eye(200),
            entropy=0.0,
            value_loss=0.0,
            grad_total=0.0,
        )
        line = record.finish(log_std=None, norms={}, activation="tanh", continuous=False)
        assert line["srank_rows"] == tm.SRANK_ROWS
        assert line["srank"] == tm.SRANK_ROWS

    def test_srank(self) -> None:
        assert tm.srank(torch.eye(4)) == 4
        assert tm.srank(torch.outer(torch.arange(1.0, 6.0), torch.ones(3))) == 1
        assert tm.srank(torch.zeros(3, 3)) == 0

    def test_saturation_by_activation(self) -> None:
        h = torch.tensor([[0.995, -0.999, 0.5, 0.0]])
        assert tm.saturated_fraction(h, "tanh") == pytest.approx(0.5)
        assert tm.saturated_fraction(h, "relu") == pytest.approx(0.25)

    def test_grad_norm_and_weight_norms(self) -> None:
        layer = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[3.0, 4.0]]))
        layer.weight.grad = torch.tensor([[1.0, 2.0]])
        assert tm.grad_norm(layer.parameters()) == pytest.approx(math.sqrt(5))
        assert tm.weight_norms({"head": layer}) == {"head.weight": pytest.approx(5.0)}

    def test_a_record_assembles_one_update(self) -> None:
        record = tm.UpdateRecord(
            returns=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            raw_advantages=torch.tensor([1.0, -1.0, 3.0, -3.0]),
            old_values=torch.tensor([0.0, 3.0, 0.0, 7.0]),
            clip_epsilon=0.2,
            max_grad_norm=0.5,
        )
        for epoch in range(2):
            record.begin_epoch()
            for _ in range(2):
                layer = torch.nn.Linear(1, 1, bias=False)
                layer.weight.grad = torch.tensor([[2.0 + epoch]])
                record.read_grads(layer.parameters(), [])
                record.minibatch(
                    new_log_probs=torch.zeros(2),
                    old_log_probs=torch.zeros(2),
                    head_out=torch.tensor([[3.0, 0.0], [1.0, -1.0]]),
                    features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    entropy=1.0 + epoch,
                    value_loss=2.0,
                    grad_total=0.4 + epoch,
                    penultimate=torch.zeros(2, 2),
                )
        line = record.finish(
            log_std=torch.tensor([-0.5, -1.5]),
            norms={"w": 1.0},
            activation="tanh",
            continuous=True,
        )
        assert line["adv_mean"] == 0.0
        assert line["adv_std"] == pytest.approx(math.sqrt(5))
        assert line["adv_max_abs"] == 3.0
        assert line["entropy"] == pytest.approx(1.5)
        assert line["actor_grad"] == pytest.approx(2.5)
        assert line["critic_grad"] == 0.0
        assert line["clipped_minibatch_fraction"] == pytest.approx(0.5)
        assert line["action_mean_abs"] == pytest.approx(5 / 4)
        assert line["action_mean_large_fraction"] == pytest.approx(1 / 4)
        assert line["log_std"] == pytest.approx(-1.0)
        assert line["srank"] == 2
        assert line["srank_rows"] == 4
        assert line["settling_residual"] == pytest.approx(1.0)
        assert line["batch_size"] == 4

    def test_the_off_record_keeps_nothing(self) -> None:
        record = tm.record_for(
            None,
            returns=torch.zeros(2),
            raw_advantages=torch.zeros(2),
            old_values=[torch.zeros(1)],
            clip_epsilon=0.2,
            max_grad_norm=0.5,
        )
        assert isinstance(record, tm.NoRecord)
        record.begin_epoch()
        record.minibatch(features=torch.zeros(1))

    def test_the_sink_stamps_and_counts(self, tmp_path: Path) -> None:
        sink = tm.UpdateTelemetry(tmp_path / "t" / "updates.jsonl")
        assert not (tmp_path / "t").exists()
        sink.episode = 3
        sink.write({"x": 1})
        sink.write({"x": 2})
        sink.close()
        lines = [json.loads(x) for x in (tmp_path / "t" / "updates.jsonl").read_text().splitlines()]
        assert [(x["episode"], x["update"], x["x"]) for x in lines] == [(3, 0, 1), (3, 1, 2)]


# ── On and off are the same run ──────────────────────────────────────────────────────────────

_STEPS = 40


def _params(step: int) -> BrainParams:
    return BrainParams(
        food_concentration=0.1 * (step % 5),
        food_lateral_gradient=0.05 * (step % 3) - 0.05,
        food_dconcentration_dt=0.01 * step,
        food_gradient_strength=0.2 + 0.01 * step,
        food_gradient_direction=0.3 * step,
    )


def _connectome() -> ConnectomePPOBrain:
    config = ConnectomePPOBrainConfig(
        seed=5,
        action_mode="continuous",
        rollout_buffer_size=16,
        num_minibatches=2,
        num_epochs=2,
    )
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU)


def _mlp() -> MLPPPOBrain:
    config = MLPPPOBrainConfig(
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
        action_mode="continuous",
        rollout_buffer_size=16,
        num_minibatches=2,
        num_epochs=2,
        seed=5,
    )
    return MLPPPOBrain(config=config)


def _drive(brain: Any) -> list[str]:
    torch.manual_seed(0)
    outputs = []
    for step in range(_STEPS):
        params = _params(step)
        actions = brain.run_brain(
            params,
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )
        outputs.append(repr(actions))
        brain.learn(params, reward=0.1 * (step % 3), episode_done=step % 13 == 12)
    return outputs


def _state(brain: Any) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for component in brain.get_weight_components().values():
        for key, value in component.state.items():
            if isinstance(value, torch.Tensor):
                out[f"{component.name}.{key}"] = value.detach().clone()
    return out


def _generators(brain: Any) -> tuple[float, float, float]:
    return (
        float(brain.rng.random()),
        float(brain.buffer.rng.random()),
        float(torch.rand(1).item()),
    )


@pytest.mark.parametrize("build", [_connectome, _mlp], ids=["connectome", "mlp"])
def test_telemetry_does_not_change_the_run(build: Any, tmp_path: Path) -> None:
    """Scenario "Telemetry does not change the run", and "One line per update"."""
    off = build()
    off_outputs = _drive(off)
    off_state, off_next = _state(off), _generators(off)

    on = build()
    sink = tm.UpdateTelemetry(tmp_path / "updates.jsonl")
    on.set_update_telemetry(sink)
    on_outputs = _drive(on)
    on_state, on_next = _state(on), _generators(on)
    sink.close()

    assert on_outputs == off_outputs
    assert on_state.keys() == off_state.keys()
    for key, value in off_state.items():
        assert torch.equal(on_state[key], value), key
    assert on_next == off_next

    lines = [json.loads(x) for x in (tmp_path / "updates.jsonl").read_text().splitlines()]
    assert len(lines) >= 2
    assert [line["update"] for line in lines] == list(range(len(lines)))
    for line in lines:
        assert line.keys() >= _FIELDS
        assert all(np.isfinite(v) for v in line["weight_norms"].values())


def test_the_connectome_reports_its_settling_residual(tmp_path: Path) -> None:
    """The connectome's line carries the settled state's distance from a fixed point."""
    brain = _connectome()
    sink = tm.UpdateTelemetry(tmp_path / "updates.jsonl")
    brain.set_update_telemetry(sink)
    _drive(brain)
    sink.close()
    line = json.loads((tmp_path / "updates.jsonl").read_text().splitlines()[0])
    assert line["settling_residual"] >= 0.0
    assert line["saturation_rule"] == "tanh_0.99"
    assert any(k.startswith("critic.") for k in line["weight_norms"])


def test_a_frozen_brain_writes_no_line(tmp_path: Path) -> None:
    """A frozen floor makes no update, so it writes nothing."""
    config = ConnectomePPOBrainConfig(
        seed=5,
        action_mode="continuous",
        rollout_buffer_size=16,
        freeze_updates=True,
    )
    brain = ConnectomePPOBrain(config=config, device=DeviceType.CPU)
    sink = tm.UpdateTelemetry(tmp_path / "updates.jsonl")
    brain.set_update_telemetry(sink)
    _drive(brain)
    assert not (tmp_path / "updates.jsonl").exists()


def test_a_plastic_rule_refuses_telemetry(tmp_path: Path) -> None:
    """A plastic rule makes no PPO update, so telemetry is refused there."""
    config = ConnectomePPOBrainConfig(
        seed=5,
        learning_rule="three_factor",
        enable_activity_traces=True,
    )
    brain = ConnectomePPOBrain(config=config, device=DeviceType.CPU)
    with pytest.raises(AttributeError, match="update telemetry"):
        brain.set_update_telemetry(tm.UpdateTelemetry(tmp_path / "u.jsonl"))
