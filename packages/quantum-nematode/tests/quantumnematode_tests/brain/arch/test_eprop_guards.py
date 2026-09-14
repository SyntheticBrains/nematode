"""What an e-prop configuration is refused for, and where the refusal has to live.

Every check here exists twice in the package: once as a pydantic validator and once as a guard the
brain re-runs at construction. That is not redundancy -- ``model_copy`` skips validators, and
``model_copy`` is how the campaign runner derives its arms, so a validator alone would let a derived
arm run in a configuration nobody would have accepted from a file.

Four refusals, each for a result that would otherwise be wrong rather than absent:

* a perturbation alongside e-prop changes the forward pass the trace describes while contributing
  nothing to the trace -- a result attributable to neither mechanism;
* a perturbation SET restricts draws that are never drawn, so the arm would carry a name for a
  restriction it never had;
* the learning-signal routing decides which units the signal can reach AT ALL, so it is the arm:
  taking it from a default would mean an arm nobody chose, and naming it where nothing reads it
  would mean an arm that was never run;
* a discrete head has a different score function, and deriving one from a head this build was not
  written for would be a guess presented as a gradient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from quantumnematode.brain.arch._mlp_topology import MLPTopology
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
    _reject_unsupported_plasticity_modes,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.learning_rules import ThreeFactorRule
from quantumnematode.utils.config_loader import load_simulation_config
from torch import nn

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "configs").is_dir():
    _root = _root.parent
_CONFIGS = _root / "configs" / "scenarios" / "foraging"
_STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop"
_ROUTINGS = ("symmetric", "random_motor", "random", "scalar")
_ARMS = (*_ROUTINGS, "plastic_readout", "readout_only")


def _config(**overrides: Any) -> ConnectomePPOBrainConfig:
    """Build a valid e-prop config, then apply the override under test."""
    simulation = load_simulation_config(str(_CONFIGS / f"{_STEM}_random.yml"))
    assert simulation.brain is not None
    base = simulation.brain.config
    assert isinstance(base, ConnectomePPOBrainConfig)
    fields = base.model_dump()
    fields.update(overrides)
    return ConnectomePPOBrainConfig(**fields)


def _derived(**overrides: Any) -> ConnectomePPOBrainConfig:
    """Apply the override the way the campaign runner does: through model_copy, no validators."""
    simulation = load_simulation_config(str(_CONFIGS / f"{_STEM}_random.yml"))
    assert simulation.brain is not None
    base = simulation.brain.config
    assert isinstance(base, ConnectomePPOBrainConfig)
    return base.model_copy(update=overrides)


class TestAPerturbationAlongsideEprop:
    def test_the_validator_refuses_it(self) -> None:
        with pytest.raises(ValueError, match=r"requires plasticity_node_noise=0\.0"):
            _config(plasticity_node_noise=0.1)

    def test_the_construction_guard_refuses_it(self) -> None:
        with pytest.raises(ValueError, match=r"requires plasticity_node_noise=0\.0"):
            _reject_unsupported_plasticity_modes(_derived(plasticity_node_noise=0.1))


class TestAPerturbationSetAlongsideEprop:
    def test_the_validator_refuses_it(self) -> None:
        with pytest.raises(ValueError, match="no perturbation to restrict"):
            _config(plasticity_perturbation_set="motor", plasticity_node_noise=0.0)

    def test_the_construction_guard_refuses_it(self) -> None:
        with pytest.raises(ValueError, match="no perturbation to restrict"):
            _reject_unsupported_plasticity_modes(_derived(plasticity_perturbation_set="motor"))


class TestTheRoutingIsTheArm:
    def test_eprop_without_a_routing_is_refused(self) -> None:
        # An arm must not come from a default: which units the signal can reach is the experiment.
        with pytest.raises(ValueError, match="requires plasticity_learning_signal"):
            _config(plasticity_learning_signal=None)

    def test_the_construction_guard_refuses_it_too(self) -> None:
        with pytest.raises(ValueError, match="requires plasticity_learning_signal"):
            _reject_unsupported_plasticity_modes(_derived(plasticity_learning_signal=None))

    def test_a_routing_without_eprop_is_refused(self) -> None:
        # A config naming a routing nothing reads would be read as an arm that had one.
        with pytest.raises(ValueError, match="read only under"):
            _config(
                plasticity_eligibility="node_perturbation",
                plasticity_node_noise=0.1,
                plasticity_learning_signal="random",
            )

    def test_an_unset_routing_is_fine_without_eprop(self) -> None:
        # Every committed config round-trips through model_dump, which sets every field; only an
        # actual value may be refused, never the absence of one.
        config = _config(
            plasticity_eligibility="node_perturbation",
            plasticity_node_noise=0.1,
            plasticity_learning_signal=None,
        )
        assert config.plasticity_learning_signal is None


class TestADiscreteHeadIsRefused:
    def test_the_construction_guard_refuses_it(self) -> None:
        with pytest.raises(ValueError, match="continuous head only"):
            _reject_unsupported_plasticity_modes(_derived(action_mode="discrete"))


class TestTheDenseYardstickRefusesEprop:
    def test_it_refuses_at_construction(self) -> None:
        # The topology supports the mode so the rule's positive control can drive it directly; this
        # brain has no per-step call site to fold the signal in, and a refusal at load is the
        # difference between a config that cannot run and a run whose trace nothing ever credits.
        config = MLPPPOBrainConfig(
            sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
            learning_rule="three_factor",
            enable_activity_traces=True,
            plasticity_eligibility="eprop",
            plasticity_learning_signal="random",
            action_mode="continuous",
        )
        with pytest.raises(ValueError, match="not available on this brain"):
            MLPPPOBrain(config=config, device=DeviceType.CPU)


class TestTheArmsDifferByOneKey:
    """A claim of matched arms only survives if the files differ where they say they do."""

    @staticmethod
    def _brain_keys(name: str) -> dict[str, Any]:
        loaded = yaml.safe_load((_CONFIGS / f"{name}.yml").read_text())
        return loaded["brain"]["config"]

    @pytest.mark.parametrize("routing", ["symmetric", "random_motor", "scalar"])
    def test_the_learning_arms_differ_in_the_routing_alone(self, routing: str) -> None:
        reference = self._brain_keys(f"{_STEM}_random")
        arm = self._brain_keys(f"{_STEM}_{routing}")
        differing = {key for key in set(reference) | set(arm) if reference.get(key) != arm.get(key)}
        assert differing == {"plasticity_learning_signal"}
        assert arm["plasticity_learning_signal"] == routing

    def test_the_frozen_arm_differs_in_the_freeze_alone(self) -> None:
        reference = self._brain_keys(f"{_STEM}_random")
        frozen = self._brain_keys(f"{_STEM}_frozen")
        differing = {
            key for key in set(reference) | set(frozen) if reference.get(key) != frozen.get(key)
        }
        assert differing == {"freeze_updates"}
        assert frozen["freeze_updates"] is True

    @pytest.mark.parametrize("routing", [*_ARMS, "frozen"])
    def test_every_arm_pins_the_measured_operating_point(self, routing: str) -> None:
        # R.1c's and R.1d's operating point, written out rather than inherited, so the comparison
        # with node perturbation's 3.751 and 3.150 on this cell is against the same conditions.
        keys = self._brain_keys(f"{_STEM}_{routing}")
        assert keys["plasticity_eligibility"] == "eprop"
        assert keys["plasticity_node_noise"] == 0.0
        assert "plasticity_perturbation_set" not in keys
        assert keys["plasticity_normalise_trace"] is True
        assert keys["plasticity_normalise_modulator"] is True
        assert keys["plasticity_homeostasis"] is True
        assert keys["plasticity_rate"] == 0.001
        assert keys["trace_decay"] == 0.9
        assert keys["initial_log_std"] == -1.0
        assert keys["forward_pass_depth"] == 4


class TestTheRuleRefusesAMismatchedTopology:
    def test_a_hebbian_topology_under_the_eprop_rule_is_refused(self) -> None:
        topology = MLPTopology(
            nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1)),
            enable_activity_traces=True,
            trace_decay=0.9,
            plastic_layers="hidden",
        )
        with pytest.raises(ValueError, match="needs a topology built under the same mode"):
            ThreeFactorRule(
                topology,
                plasticity_rate=0.001,
                weight_decay=0.001,
                weight_bound=3.0,
                baseline_rate=0.01,
                freeze_updates=False,
                modulated=True,
                eligibility="eprop",
                device=torch.device("cpu"),
            )


class TestTheArmsConstruct:
    @pytest.mark.parametrize("routing", [*_ARMS, "frozen"])
    def test_each_config_builds_a_brain(self, routing: str) -> None:
        simulation = load_simulation_config(str(_CONFIGS / f"{_STEM}_{routing}.yml"))
        assert simulation.brain is not None
        config = simulation.brain.config
        assert isinstance(config, ConnectomePPOBrainConfig)
        config.seed = 1
        brain = ConnectomePPOBrain(config=config, device=DeviceType.CPU)
        assert brain.topology.eligibility == "eprop"


class TestThePlasticReadoutArm:
    """The arm stage 1 says can work, and the three things that would make it something else."""

    def test_it_differs_from_the_broad_arm_in_the_readout_alone(self) -> None:
        keys = TestTheArmsDifferByOneKey._brain_keys
        reference = keys(f"{_STEM}_random")
        arm = keys(f"{_STEM}_plastic_readout")
        differing = {k for k in set(reference) | set(arm) if reference.get(k) != arm.get(k)}
        assert differing == {"plasticity_plastic_readout"}
        assert arm["plasticity_plastic_readout"] is True

    def test_it_needs_the_eprop_eligibility(self) -> None:
        # Under the Hebbian eligibility a plastic output layer takes its own OUTPUT as the
        # post-synaptic factor and its rows self-amplify, which Logbook 040 measured.
        with pytest.raises(ValueError, match="requires plasticity_eligibility='eprop'"):
            _config(
                plasticity_eligibility="hebbian",
                plasticity_learning_signal=None,
                plasticity_plastic_readout=True,
            )

    def test_a_frozen_plastic_readout_arm_is_refused(self) -> None:
        # There is no such arm: with no update no tensor moves, so one frozen floor serves them all.
        with pytest.raises(ValueError, match="no such arm"):
            _reject_unsupported_plasticity_modes(
                _derived(plasticity_plastic_readout=True, freeze_updates=True),
            )

    def test_the_seam_exposes_the_readout_as_a_second_plastic_tensor(self) -> None:
        simulation = load_simulation_config(str(_CONFIGS / f"{_STEM}_plastic_readout.yml"))
        assert simulation.brain is not None
        config = simulation.brain.config
        assert isinstance(config, ConnectomePPOBrainConfig)
        config.seed = 1
        topology = ConnectomePPOBrain(config=config, device=DeviceType.CPU).topology
        assert [tuple(w.shape) for w in topology.plastic_weights] == [(302, 302), (2, 4)]
        assert [tuple(x.shape) for x in topology.eligibility_traces] == [(302, 302), (2, 4)]
        assert [tuple(m.shape) for m in topology.plastic_masks] == [(302, 302), (2, 4)]
        assert [tuple(v.shape) for v in topology.plastic_post_activities] == [(302,), (2,)]
        # The readout is [action, class]: its post-synaptic units are the action dimensions on axis
        # 0, so one unit's incoming weights are a ROW.
        assert topology.plastic_fan_in_axes == [0, 1]

    def test_the_readout_is_excluded_from_homeostasis(self) -> None:
        # The rescale returns each unit's incoming norm to construction, and the readout's SCALE is
        # part of what this arm asks about: R.1d measured +4.51 foods from that scale alone.
        simulation = load_simulation_config(str(_CONFIGS / f"{_STEM}_plastic_readout.yml"))
        assert simulation.brain is not None
        config = simulation.brain.config
        assert isinstance(config, ConnectomePPOBrainConfig)
        config.seed = 1
        topology = ConnectomePPOBrain(config=config, device=DeviceType.CPU).topology
        assert topology.plastic_homeostasis == [True, False]

    def test_the_frozen_readout_arms_expose_one_tensor(self) -> None:
        simulation = load_simulation_config(str(_CONFIGS / f"{_STEM}_random.yml"))
        assert simulation.brain is not None
        config = simulation.brain.config
        assert isinstance(config, ConnectomePPOBrainConfig)
        config.seed = 1
        topology = ConnectomePPOBrain(config=config, device=DeviceType.CPU).topology
        assert len(topology.plastic_weights) == 1
        assert topology.plastic_homeostasis == [True]


class TestTheReadoutOnlyControl:
    """What a positive plastic-readout result needs to mean what it claims."""

    @staticmethod
    def _topology(name: str) -> Any:
        simulation = load_simulation_config(str(_CONFIGS / f"{_STEM}_{name}.yml"))
        assert simulation.brain is not None
        config = simulation.brain.config
        assert isinstance(config, ConnectomePPOBrainConfig)
        config.seed = 1
        return ConnectomePPOBrain(config=config, device=DeviceType.CPU).topology

    def test_it_differs_from_the_plastic_readout_arm_in_one_key(self) -> None:
        keys = TestTheArmsDifferByOneKey._brain_keys
        reference = keys(f"{_STEM}_plastic_readout")
        arm = keys(f"{_STEM}_readout_only")
        differing = {k for k in set(reference) | set(arm) if reference.get(k) != arm.get(k)}
        assert differing == {"plasticity_plastic_tensors"}
        assert arm["plasticity_plastic_tensors"] == "readout_only"

    def test_it_withholds_the_chemical_matrix(self) -> None:
        # The readout is an 8-parameter linear map over four pooled motor-class means, so "a local
        # rule learns this substrate" and "a small linear readout on frozen recurrent features
        # learns this cell" predict the same success. This arm is the difference.
        topology = self._topology("readout_only")
        assert [tuple(w.shape) for w in topology.plastic_weights] == [(2, 4)]
        assert [tuple(x.shape) for x in topology.eligibility_traces] == [(2, 4)]
        assert [tuple(m.shape) for m in topology.plastic_masks] == [(2, 4)]
        assert [tuple(v.shape) for v in topology.plastic_post_activities] == [(2,)]
        assert topology.plastic_fan_in_axes == [1]
        assert topology.plastic_homeostasis == [False]

    def test_the_aligned_lists_stay_the_same_length(self) -> None:
        # One place decides what is exposed; every aligned list derives from it, so a trace can
        # never be paired with another tensor's mask.
        for name in ("plastic_readout", "readout_only", "random"):
            topology = self._topology(name)
            lengths = {
                len(topology.plastic_weights),
                len(topology.eligibility_traces),
                len(topology.plastic_masks),
                len(topology.plastic_fan_in_axes),
                len(topology.plastic_homeostasis),
                len(topology.plastic_post_activities),
            }
            assert len(lengths) == 1, f"{name} exposes lists of differing lengths: {lengths}"

    def test_withholding_needs_a_plastic_readout(self) -> None:
        # Without one it would leave nothing plastic at all: a frozen control wearing a learning
        # arm's name.
        with pytest.raises(ValueError, match="requires plasticity_plastic_readout"):
            _config(plasticity_plastic_tensors="readout_only", plasticity_plastic_readout=False)
