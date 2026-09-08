"""Connectome weight persistence and the reported continuous action mean.

A warm-started connectome arm is a config plus a ``.pt`` file, so a load must reproduce
the saved brain bit for bit, refuse a file from another wiring or std mode before touching
anything, carry PPO's critic and optimiser only while PPO is live, and return the plastic
rule's running state to its construction values.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._policy import continuous_deterministic_action
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.weights import WeightPersistence, load_weights, save_weights
from quantumnematode.learning_rules.three_factor import ThreeFactorRule
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[6]
_ARMS = _REPO_ROOT / "configs" / "scenarios" / "foraging_predator_thermal"
_STEM = "connectomeppo_small_continuous2d_combined_klinotaxis"
_PPO = _ARMS / f"{_STEM}.yml"
_PLASTIC = _ARMS / f"{_STEM}_plastic.yml"
_PLASTIC_REWIRED = _ARMS / f"{_STEM}_plastic_rewired_null.yml"
_MLP = _ARMS / "mlpppo_small_continuous2d_combined_klinotaxis.yml"
_SEED = 11
_STEPS = 12


def _cfg(path: Path, **overrides: object) -> ConnectomePPOBrainConfig:
    config = load_simulation_config(str(path)).brain
    assert config is not None
    assert isinstance(config.config, ConnectomePPOBrainConfig)
    return config.config.model_copy(update={"seed": _SEED, **overrides})


def _brain(path: Path, **overrides: object) -> ConnectomePPOBrain:
    return ConnectomePPOBrain(config=_cfg(path, **overrides), device=DeviceType.CPU)


def _drive(brain: ConnectomePPOBrain, steps: int = _STEPS) -> None:
    brain.prepare_episode()
    torch.manual_seed(_SEED)
    for step in range(steps):
        brain.run_brain(
            BrainParams(
                food_gradient_strength=0.2 + 0.05 * step,
                food_gradient_direction=0.3 * step - 1.0,
            ),
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )
        brain.learn(BrainParams(), reward=0.5 * (step % 2) - 0.2, episode_done=(step == steps - 1))


_TRANSIENT = {"activity_traces", "prev_activity", "prev_activity_valid"}


def _params(brain: ConnectomePPOBrain) -> dict[str, torch.Tensor]:
    """Every persisted tensor: parameters and wiring buffers, never the per-episode traces."""
    return {
        k: v.detach().clone() for k, v in brain.topology.state_dict().items() if k not in _TRANSIENT
    }


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


class TestProtocolAndComponents:
    def test_implements_the_protocol(self) -> None:
        assert isinstance(_brain(_PLASTIC), WeightPersistence)

    def test_ppo_components_follow_the_rule(self) -> None:
        assert set(_brain(_PPO).get_weight_components()) == {
            "topology",
            "value",
            "optimizer",
            "training_state",
        }
        assert set(_brain(_PLASTIC).get_weight_components()) == {"topology", "training_state"}

    def test_topology_component_carries_the_wiring_buffers_not_the_traces(self) -> None:
        state = _brain(_PLASTIC).get_weight_components()["topology"].state
        assert "m_chem" in state
        assert "g_gap" in state
        assert "w_chem" in state
        assert "readout" in state
        assert not _TRANSIENT & set(state)

    def test_training_state_records_the_configuration(self) -> None:
        ts = _brain(_PLASTIC).get_weight_components()["training_state"].state
        assert ts["learning_rule"] == "three_factor"
        assert ts["wiring"] == "wild_type"
        assert ts["weight_init"] == "degree_scaled"
        assert ts["continuous_std_mode"] == "state_independent"
        assert "episode_count" not in ts

    def test_unknown_component_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown weight components"):
            _brain(_PLASTIC).get_weight_components(components={"policy"})


class TestRoundTrip:
    @pytest.mark.parametrize("path", [_PPO, _PLASTIC], ids=["ppo", "plastic"])
    def test_round_trip_is_bit_identical(self, path: Path, tmp_path: Path) -> None:
        source = _brain(path)
        _drive(source)
        file = tmp_path / "w.pt"
        save_weights(source, file)
        target = _brain(path, seed=_SEED + 1)
        assert not _same(_params(source), _params(target))
        load_weights(target, file)
        assert _same(_params(source), _params(target))

    def test_ppo_critic_and_optimizer_round_trip(self, tmp_path: Path) -> None:
        source = _brain(_PPO)
        _drive(source)
        file = tmp_path / "w.pt"
        save_weights(source, file)
        target = _brain(_PPO, seed=_SEED + 1)
        load_weights(target, file)
        for a, b in zip(source.critic.parameters(), target.critic.parameters(), strict=True):
            assert torch.equal(a, b)
        assert target.optimizer.state_dict()["param_groups"][0]["lr"] == pytest.approx(
            source.optimizer.state_dict()["param_groups"][0]["lr"],
        )

    def test_load_resets_the_buffer_and_the_rule_state(self, tmp_path: Path) -> None:
        source = _brain(_PLASTIC)
        _drive(source)
        file = tmp_path / "w.pt"
        save_weights(source, file)
        target = _brain(_PLASTIC, seed=_SEED + 1)
        _drive(target)
        rule = target._rule
        assert isinstance(rule, ThreeFactorRule)
        assert rule.baseline != 0.0
        load_weights(target, file)
        assert rule.baseline == 0.0
        assert rule.modulator_scale.count == 0
        assert all(s.count == 0 for s in rule.trace_scales)
        assert rule.modulator_centre.count == 0
        assert len(target.buffer) == 0
        for trace in target.topology.eligibility_traces:
            assert torch.count_nonzero(trace) == 0

    def test_load_reanchors_the_homeostatic_targets_to_the_loaded_weights(
        self,
        tmp_path: Path,
    ) -> None:
        """A loaded policy keeps its norms: the targets follow the weights, not the init."""
        source = _brain(_PLASTIC)
        with torch.no_grad():
            source.topology.w_chem.mul_(6.0)  # a clone with inflated norms, as cloning produces
        file = tmp_path / "w.pt"
        save_weights(source, file)
        target = _brain(_PLASTIC, seed=_SEED + 1)
        rule = target._rule
        assert isinstance(rule, ThreeFactorRule)
        before = rule.norm_targets[0].clone()
        load_weights(target, file)
        mask = target.topology.m_chem
        loaded_norms = torch.sqrt(((target.topology.w_chem * mask) ** 2).sum(0))
        assert torch.allclose(rule.norm_targets[0], loaded_norms, atol=1e-5)
        assert not torch.allclose(rule.norm_targets[0], before)

    def test_plastic_brain_ignores_ppo_components(self, tmp_path: Path) -> None:
        source = _brain(_PPO)
        _drive(source)
        file = tmp_path / "w.pt"
        save_weights(source, file)
        target = _brain(_PLASTIC, seed=_SEED + 1)
        load_weights(target, file)  # the file carries value + optimizer; must not raise
        assert torch.equal(target.topology.w_chem, source.topology.w_chem)

    def test_ppo_brain_keeps_a_fresh_critic_without_ppo_components(self, tmp_path: Path) -> None:
        source = _brain(_PLASTIC)
        _drive(source)
        file = tmp_path / "w.pt"
        save_weights(source, file)
        target = _brain(_PPO, seed=_SEED + 1)
        before = [p.detach().clone() for p in target.critic.parameters()]
        load_weights(target, file)
        for a, b in zip(before, target.critic.parameters(), strict=True):
            assert torch.equal(a, b)
        assert torch.equal(target.topology.w_chem, source.topology.w_chem)


class TestRefusals:
    @pytest.mark.parametrize(
        ("saved", "receiving"),
        [(_PLASTIC, _PLASTIC_REWIRED), (_PLASTIC_REWIRED, _PLASTIC)],
        ids=["wild-type-into-rewired", "rewired-into-wild-type"],
    )
    def test_wiring_mismatch_is_refused_before_mutation(
        self,
        saved: Path,
        receiving: Path,
        tmp_path: Path,
    ) -> None:
        file = tmp_path / "w.pt"
        save_weights(_brain(saved), file)
        target = _brain(receiving)
        before = _params(target)
        with pytest.raises(ValueError, match="different wiring"):
            load_weights(target, file)
        assert _same(before, _params(target))

    def test_std_mode_mismatch_is_refused_before_mutation(self, tmp_path: Path) -> None:
        file = tmp_path / "w.pt"
        save_weights(_brain(_PPO), file)
        target = _brain(_PPO, continuous_std_mode="state_dependent")
        before = _params(target)
        with pytest.raises(ValueError, match="std mode"):
            load_weights(target, file)
        assert _same(before, _params(target))


class TestContinuousMean:
    def test_connectome_mean_is_the_noiseless_action(self) -> None:
        brain = _brain(_PLASTIC)
        brain.prepare_episode()
        params = BrainParams(food_gradient_strength=0.4, food_gradient_direction=0.2)
        action = brain.run_brain(
            params,
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )[0]
        assert action.continuous_mean is not None
        state = torch.from_numpy(brain.preprocess(params))
        food, distal, mechano, zone, thermo = brain._unpack_state(state)
        with torch.no_grad():
            mean, _ = brain.topology.forward_with_hidden(
                food,
                predator_distal_features=distal,
                predator_mechano_features=mechano,
                predator_contact_zone=zone,
                thermotaxis_features=thermo,
            )
            expected = continuous_deterministic_action(mean, brain._action_low, brain._action_high)
        assert action.continuous_mean == pytest.approx((expected[0].item(), expected[1].item()))
        assert action.continuous is not None

    def test_mlp_mean_is_the_noiseless_action(self) -> None:
        config = load_simulation_config(str(_MLP)).brain
        assert config is not None
        assert isinstance(config.config, MLPPPOBrainConfig)
        brain = MLPPPOBrain(
            config=config.config.model_copy(update={"seed": _SEED}),
            device=DeviceType.CPU,
        )
        brain.prepare_episode()
        params = BrainParams(food_gradient_strength=0.4, food_gradient_direction=0.2)
        action = brain.run_brain(
            params,
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )[0]
        assert action.continuous_mean is not None
        assert 0.0 <= action.continuous_mean[0] <= 1.0
        assert -1.0 <= action.continuous_mean[1] <= 1.0

    def test_discrete_brain_reports_none(self) -> None:
        discrete = (
            _REPO_ROOT / "configs" / "scenarios" / "foraging" / "connectomeppo_small_klinotaxis.yml"
        )
        brain = _brain(discrete)
        brain.prepare_episode()
        action = brain.run_brain(
            BrainParams(food_gradient_strength=0.4, food_gradient_direction=0.2),
            reward=None,
            input_data=None,
            top_only=True,
            top_randomize=False,
        )[0]
        assert action.continuous_mean is None
