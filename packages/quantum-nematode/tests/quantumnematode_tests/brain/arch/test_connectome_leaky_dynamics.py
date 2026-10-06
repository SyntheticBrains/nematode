# pyright: reportPrivateUsage=false
"""Across-step leaky-integrator dynamics on the connectome brain.

Covers the connectome-ppo-brain requirement "Across-step leaky-integrator dynamics": settling is
unchanged; the step is stable on the raw gap weights; state carries across steps and resets at
episode start; replay reproduces the rollout, across an episode boundary and with a partial final
chunk; unsupported combinations are refused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._policy import categorical_evaluate_torch
from quantumnematode.brain.arch._ppo_buffer import ChunkedRolloutBuffer, RolloutBuffer
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.learning_rules.ppo import ConnectomePPOBatch
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[6]
_HARD350 = (
    _REPO_ROOT
    / "configs"
    / "scenarios"
    / "foraging"
    / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350.yml"
)
_SEED = 2026


def _brain(**overrides: object) -> ConnectomePPOBrain:
    cfg = ConnectomePPOBrainConfig(seed=_SEED, **overrides)  # type: ignore[arg-type]
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _hard350(**overrides: object) -> ConnectomePPOBrain:
    container = load_simulation_config(str(_HARD350)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    cfg = container.config.model_copy(update={"seed": _SEED, **overrides})
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _params(step: int) -> BrainParams:
    return BrainParams(
        food_gradient_strength=0.3 + 0.05 * (step % 7),
        food_gradient_direction=0.1 * (step % 5) - 0.3,
    )


def _drive(brain: ConnectomePPOBrain, n_steps: int, done_at: tuple[int, ...] = ()) -> None:
    """Act and learn for ``n_steps``, ending an episode after each step in ``done_at``."""
    torch.manual_seed(_SEED)
    for step in range(n_steps):
        brain.run_brain(
            _params(step),
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )
        done = step in done_at
        brain.learn(_params(step), reward=0.1 * (step % 3), episode_done=done)
        if done:
            brain.prepare_episode()


# ── Configuration ────────────────────────────────────────────────────────────────────────────


class TestConfiguration:
    def test_leaky_validates_and_settling_is_the_default(self) -> None:
        assert ConnectomePPOBrainConfig().dynamics == "settling"
        ConnectomePPOBrainConfig(dynamics="leaky", membrane_tau_steps=5.0, bptt_chunk_length=8)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"membrane_tau_steps": 2.0},
            {"bptt_chunk_length": 8},
            {"input_gain": 4.0},
        ],
    )
    def test_leaky_pins_are_refused_under_settling(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="read only under dynamics='leaky'"):
            ConnectomePPOBrainConfig(**overrides)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            (
                {"learning_rule": "three_factor", "enable_activity_traces": True},
                "requires learning_rule='ppo'",
            ),
            ({"enable_activity_traces": True}, "enable_activity_traces"),
            ({"plasticity_node_noise": 0.1}, "plasticity_node_noise=0.0"),
        ],
    )
    def test_unsupported_combinations_are_refused(
        self,
        overrides: dict[str, object],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ConnectomePPOBrainConfig(dynamics="leaky", **overrides)  # type: ignore[arg-type]

    def test_a_copied_config_is_refused_at_construction(self) -> None:
        """``model_copy`` skips validators, so the brain repeats the check."""
        cfg = ConnectomePPOBrainConfig(seed=_SEED).model_copy(
            update={"dynamics": "leaky", "enable_activity_traces": True},
        )
        with pytest.raises(ValueError, match="enable_activity_traces"):
            ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


# ── Settling is untouched ────────────────────────────────────────────────────────────────────


class TestSettlingUnchanged:
    def test_settling_allocates_no_leaky_state_and_keeps_its_buffer(self) -> None:
        brain = _brain()
        assert brain.topology.dynamics == "settling"
        assert not hasattr(brain.topology, "membrane")
        assert type(brain.buffer) is RolloutBuffer

    def test_the_leaky_state_is_not_saved_with_the_weights(self) -> None:
        brain = _brain(dynamics="leaky")
        saved = brain.topology.state_dict()
        assert "membrane" not in saved
        assert "implicit_inverse" not in saved
        assert set(saved) == set(_brain().topology.state_dict())

    def test_the_zero_start_batched_forward_is_refused_under_leaky(self) -> None:
        brain = _brain(dynamics="leaky")
        with pytest.raises(RuntimeError, match="forward_sequence"):
            brain.topology.forward_with_hidden_batched(torch.zeros(3, 2))


# ── Stability on the raw gap weights ─────────────────────────────────────────────────────────


class TestStability:
    @pytest.mark.parametrize("tau", [0.05, 0.2, 1.0, 5.0])
    @pytest.mark.parametrize("depth", [1, 4])
    def test_the_implicit_operator_contracts(self, tau: float, depth: int) -> None:
        topo = _brain(dynamics="leaky", membrane_tau_steps=tau, forward_pass_depth=depth).topology
        alpha = 1.0 / (depth * tau)
        norm = torch.linalg.matrix_norm(topo.implicit_inverse.double(), ord=2).item()
        assert norm <= 1.0 / (1.0 + alpha) + 1e-5

    def test_an_explicit_step_would_diverge_on_these_weights(self) -> None:
        """The stiffness the implicit step exists for: explicit Euler's step matrix exceeds 1."""
        topo = _brain(dynamics="leaky").topology
        gap = topo.g_gap.double()
        laplacian = torch.diag(gap.sum(dim=1)) - gap
        explicit = torch.eye(topo.n_neurons, dtype=torch.float64) - topo.leaky_alpha * (
            torch.eye(topo.n_neurons, dtype=torch.float64) + laplacian
        )
        assert torch.linalg.eigvalsh(explicit).abs().max().item() > 1.0

    def test_potentials_stay_bounded_over_a_long_episode(self) -> None:
        topo = _hard350(dynamics="leaky", membrane_tau_steps=0.2).topology
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            peak = 0.0
            for _ in range(3000):
                food = torch.randn(topo.n_food_features, generator=gen) * 10.0
                topo.forward_with_hidden(food)
                peak = max(peak, topo.membrane.abs().max().item())
        assert np.isfinite(peak)
        assert peak < 1e3


# ── Carry and reset ──────────────────────────────────────────────────────────────────────────


class TestCarryAndReset:
    def test_the_second_step_starts_where_the_first_ended(self) -> None:
        topo = _brain(dynamics="leaky").topology
        food = torch.tensor([0.5, -0.2])
        with torch.no_grad():
            first, _ = topo.forward_with_hidden(food)
            after_first = topo.membrane.clone()
            second, _ = topo.forward_with_hidden(food)
            expected = topo._leaky_substeps(
                after_first,
                topo._sensor_current(food, None, None, None, None),
            )
        assert not torch.equal(first, second)
        assert torch.allclose(topo.membrane, expected)

    def test_a_new_episode_starts_from_zero(self) -> None:
        brain = _brain(dynamics="leaky")
        _drive(brain, 5)
        assert brain.topology.membrane.abs().sum().item() > 0.0
        brain.prepare_episode()
        assert torch.equal(brain.topology.membrane, torch.zeros_like(brain.topology.membrane))

    def test_the_input_gain_scales_the_sensor_current(self) -> None:
        food = torch.tensor([0.5, -0.2])
        plain = _brain(dynamics="leaky").topology
        scaled = _brain(dynamics="leaky", input_gain=8.0).topology
        with torch.no_grad():
            plain.forward_with_hidden(food)
            scaled.forward_with_hidden(food)
            current = plain._sensor_current(food, None, None, None, None)
            expected = scaled._leaky_substeps(torch.zeros(scaled.n_neurons), 8.0 * current)
        assert torch.allclose(scaled.membrane, expected)
        assert not torch.allclose(scaled.membrane, plain.membrane)

    def test_the_brain_uses_the_chunked_buffer(self) -> None:
        assert type(_brain(dynamics="leaky").buffer) is ChunkedRolloutBuffer


# ── Chunks ───────────────────────────────────────────────────────────────────────────────────


class TestChunks:
    def _buffer(self, n: int, dones: tuple[int, ...]) -> ChunkedRolloutBuffer:
        buffer = ChunkedRolloutBuffer(100, torch.device("cpu"), rng=np.random.default_rng(0))
        for t in range(n):
            buffer.add(
                state=np.array([float(t)]),
                action=t % 4,
                log_prob=torch.tensor(-1.0),
                value=torch.tensor([0.0]),
                reward=0.0,
                done=t in dones,
                start_state=torch.full((3,), float(t)),
            )
        return buffer

    def test_padding_mask_and_restarts(self) -> None:
        buffer = self._buffer(37, dones=(10, 25))
        (batch,) = buffer.get_chunk_minibatches(1, 16, torch.zeros(37), torch.arange(37.0))
        order = np.random.default_rng(0).permutation(3)
        index = {int(c): i for i, c in enumerate(order)}
        mask = batch["mask"][[index[0], index[1], index[2]]]
        restart = batch["restart"][[index[0], index[1], index[2]]].reshape(-1)
        assert mask.sum().item() == 37
        assert not mask[2, 5:].any()
        assert set(torch.nonzero(restart[:37]).flatten().tolist()) == {0, 11, 16, 26, 32}
        states = batch["states"][[index[0], index[1], index[2]]].reshape(-1)[:37]
        assert states.tolist() == [float(t) for t in range(37)]
        advantages = batch["advantages"][batch["mask"]]
        assert advantages.mean().item() == pytest.approx(0.0, abs=1e-6)

    def test_add_needs_the_start_state(self) -> None:
        buffer = ChunkedRolloutBuffer(10, torch.device("cpu"))
        with pytest.raises(ValueError, match="start_state"):
            buffer.add(np.zeros(1), 0, torch.tensor(0.0), torch.tensor([0.0]), 0.0, done=False)


# ── Replay reproduces the rollout ────────────────────────────────────────────────────────────


class TestReplay:
    def test_the_rule_replays_the_brains_rollout(self) -> None:
        """Every stored step re-scores to its rollout log-probability at unchanged parameters.

        41 steps with an episode boundary inside the second chunk and a partial third chunk. Three
        minibatches need 33 steps before an end-of-episode update, so the boundary at step 20
        keeps the buffer whole.
        """
        brain = _brain(dynamics="leaky", rollout_buffer_size=100, num_minibatches=3)
        _drive(brain, 41, done_at=(20,))
        assert len(brain.buffer) == 41
        rule = brain._require_ppo_rule("test")
        batch = ConnectomePPOBatch(
            buffer=brain.buffer,
            unpack_batched=brain._unpack_state_batched,
            last_value=None,
            chunk_length=16,
        )
        returns, advantages = torch.zeros(41), torch.zeros(41)
        seen = 0
        with torch.no_grad():
            for minibatch in rule._minibatches(batch, returns, advantages):
                head_out, _hidden, flat = rule._replay(brain.topology, batch, minibatch)
                log_probs, _ = categorical_evaluate_torch(head_out, flat["actions"])
                assert torch.allclose(log_probs, flat["old_log_probs"], atol=1e-5)
                seen += log_probs.shape[0]
        assert seen == 41

    def test_the_sequence_forward_matches_the_rollout_on_hard350(self) -> None:
        """The continuous klinotaxis cell, topology level, a restart mid-chunk."""
        topo = _hard350(dynamics="leaky", input_gain=512.0).topology
        gen = torch.Generator().manual_seed(1)
        food = torch.randn(2, 8, topo.n_food_features, generator=gen)
        restart = torch.zeros(2, 8, dtype=torch.bool)
        restart[:, 0] = True
        restart[1, 3] = True
        starts = torch.zeros(2, 8, topo.n_neurons)
        logits = torch.zeros(2, 8, 2)
        with torch.no_grad():
            for c in range(2):
                topo.reset_membrane()
                for t in range(8):
                    if restart[c, t]:
                        topo.reset_membrane()
                    starts[c, t] = topo.membrane
                    logits[c, t], _ = topo.forward_with_hidden(food[c, t])
            replayed, _ = topo.forward_sequence(food, None, None, None, None, starts, restart)
        assert torch.allclose(replayed, logits, atol=1e-5)

    def test_an_update_waits_for_a_chunk_per_minibatch(self) -> None:
        """An episode ending before two chunks' worth of steps carries its experience forward."""
        brain = _brain(dynamics="leaky", rollout_buffer_size=100, num_minibatches=2)
        _drive(brain, 13, done_at=(12,))
        assert len(brain.buffer) == 13
        _drive(brain, 7, done_at=(6,))
        assert len(brain.buffer) == 0

    def test_a_leaky_update_moves_the_weights(self) -> None:
        brain = _brain(dynamics="leaky", rollout_buffer_size=40, num_minibatches=2)
        before = brain.topology.w_chem.detach().clone()
        _drive(brain, 40)
        assert len(brain.buffer) == 0
        assert not torch.equal(before, brain.topology.w_chem.detach())
