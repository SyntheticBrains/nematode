"""The motor readout's width, and the one property the L.1 comparison rests on.

`pooled` maps four motor-class MEANS to the action -- 8 parameters, each of the 39 motor neurons
carrying `1/|class|` of its class's influence. `per_neuron` gives each neuron its own weight, 78
parameters, so the learner can read *which* neuron fires rather than only which class.

The per-neuron map is initialised by EXPANDING the pooled draw rather than drawn afresh, which is
what makes the two widths comparable:

* the orthogonal draw happens at the pooled shape either way, so **the RNG stream is untouched** and
  a pooled arm is byte-identical to the runs made before this option existed;
* the two widths compute **the same policy** at initialisation, so a wide arm does not start from
  different behaviour;
* the anatomical contrast expands the same way, so a frozen arm's policy does not depend on width.

That last equality is **analytic, not bitwise**. A slice `mean()` and a dot product with pre-divided
weights round differently, and the classes are unequal (11, 7, 12, 9) so the divisors are not powers
of two. The action is sampled around the mean, so two arms differing at 1e-8 diverge within an
episode -- which is why a frozen arm is run at each width rather than shared. These tests pin the
distinction, because collapsing it is what an earlier draft of L.1 did.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from quantumnematode.brain.arch.connectome_ppo import (
    _MOTOR_CLASSES,
    _N_ACTIONS,
    _SPEED_ACTION_INDEX,
    _TURN_ACTION_INDEX,
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[6]
_ARM = (
    _REPO_ROOT
    / "configs"
    / "scenarios"
    / "foraging"
    / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop_readout_only.yml"
)
_SEED = 7
_N_MOTOR = 39
_POOLED_PARAMS = 8
_WIDE_PARAMS = 78
# Analytic equality, float32 arithmetic: the two paths agree on the action mean to ~1e-8.
_POLICY_TOL = 1e-6


def _brain(width: str, seed: int = _SEED, **overrides: object) -> ConnectomePPOBrain:
    config = load_simulation_config(str(_ARM)).brain
    assert config is not None
    assert isinstance(config.config, ConnectomePPOBrainConfig)
    updated = config.config.model_copy(
        update={"seed": seed, "readout_width": width, **overrides},
    )
    torch.manual_seed(seed)  # the readout's orthogonal draw uses torch's global RNG
    return ConnectomePPOBrain(config=updated, device=DeviceType.CPU)


@pytest.fixture(scope="module")
def pooled() -> ConnectomePPOBrain:
    """Build the 8-parameter arm once for the module."""
    return _brain("pooled")


@pytest.fixture(scope="module")
def wide() -> ConnectomePPOBrain:
    """Build the 78-parameter arm once for the module."""
    return _brain("per_neuron")


def _readout(brain: ConnectomePPOBrain) -> torch.Tensor:
    value = brain.topology.readout
    assert isinstance(value, torch.Tensor)
    return value


def _mu(brain: ConnectomePPOBrain, h: torch.Tensor) -> torch.Tensor:
    """Return the action mean this width produces, through its own pooling path."""
    with torch.no_grad():
        return _readout(brain) @ brain.topology._pool_motor(h)


def _hidden(brain: ConnectomePPOBrain) -> torch.Tensor:
    torch.manual_seed(99)
    return torch.randn(brain.topology.n_neurons)


class TestTheShapes:
    def test_pooled_is_the_eight_parameter_map(self, pooled: ConnectomePPOBrain) -> None:
        assert tuple(_readout(pooled).shape) == (2, len(_MOTOR_CLASSES))
        assert _readout(pooled).numel() == _POOLED_PARAMS

    def test_per_neuron_is_the_seventy_eight_parameter_map(self, wide: ConnectomePPOBrain) -> None:
        assert tuple(_readout(wide).shape) == (2, _N_MOTOR)
        assert _readout(wide).numel() == _WIDE_PARAMS

    def test_the_motor_classes_are_unequal(self, pooled: ConnectomePPOBrain) -> None:
        # VB 11, DB 7, VA 12, DA 9. The expansion divisor is PER CLASS; a single constant would
        # silently reweight the pools.
        sizes = [stop - start for start, stop in pooled.topology._motor_class_slices]
        assert sizes == [11, 7, 12, 9]
        assert sum(sizes) == _N_MOTOR
        assert len(set(sizes)) > 1


class TestTheRngStreamIsUntouched:
    """The property that keeps a pooled arm byte-identical to the runs made before this option."""

    def test_every_other_parameter_is_bitwise_identical(
        self,
        pooled: ConnectomePPOBrain,
        wide: ConnectomePPOBrain,
    ) -> None:
        other = dict(wide.topology.named_parameters())
        for name, param in pooled.topology.named_parameters():
            if name == "readout":
                continue
            assert torch.equal(param, other[name]), f"{name} differs; the draw order moved"

    def test_every_float_buffer_is_bitwise_identical(
        self,
        pooled: ConnectomePPOBrain,
        wide: ConnectomePPOBrain,
    ) -> None:
        other = dict(wide.topology.named_buffers())
        for name, buf in pooled.topology.named_buffers():
            if not buf.is_floating_point() or buf.shape != other[name].shape:
                continue
            assert torch.equal(buf, other[name]), f"{name} differs; the draw order moved"


class TestTheSamePolicyButNotTheSameRun:
    def test_the_widths_agree_on_the_action_mean(
        self,
        pooled: ConnectomePPOBrain,
        wide: ConnectomePPOBrain,
    ) -> None:
        h = _hidden(pooled)
        assert torch.allclose(_mu(pooled, h), _mu(wide, h), atol=_POLICY_TOL)

    def test_but_not_bitwise_and_that_is_why_floors_are_run_at_each_width(
        self,
        pooled: ConnectomePPOBrain,
        wide: ConnectomePPOBrain,
    ) -> None:
        # Pinned deliberately. An earlier draft of L.1 ran two frozen floors instead of four, on the
        # reasoning that a frozen arm cannot depend on a width it computes the same function at.
        # It is the same FUNCTION and not the same ARITHMETIC: the action is sampled around this
        # mean, so the arms diverge within an episode and a shared floor would be the wrong control.
        h = _hidden(pooled)
        delta = (_mu(pooled, h) - _mu(wide, h)).abs().max().item()
        assert delta > 0.0, "if these ever become bitwise equal, the floor count can be revisited"
        assert delta < _POLICY_TOL


class TestTheExpansion:
    def test_each_neuron_carries_its_class_weight_over_the_class_size(
        self,
        pooled: ConnectomePPOBrain,
        wide: ConnectomePPOBrain,
    ) -> None:
        pooled_w, wide_w = _readout(pooled), _readout(wide)
        for class_index, (start, stop) in enumerate(pooled.topology._motor_class_slices):
            size = stop - start
            expected = pooled_w[:, class_index : class_index + 1] / float(size)
            assert torch.allclose(wide_w[:, start:stop], expected.expand(2, size), atol=0.0), (
                f"class {_MOTOR_CLASSES[class_index]} expanded with the wrong divisor"
            )

    def test_the_expansion_is_in_the_pools_own_flat_order(
        self,
        wide: ConnectomePPOBrain,
    ) -> None:
        # The readout's columns must line up with `_pool_motor`'s selection, or every weight is
        # attached to the wrong neuron and nothing downstream would say so.
        assert int(wide.topology._motor_flat_indices.numel()) == _N_MOTOR
        assert _readout(wide).shape[1] == int(wide.topology._motor_flat_indices.numel())


class TestTheAnatomicalContrast:
    """`set_anatomical_readout` writes a derivable map; it must be the same map at both widths."""

    @staticmethod
    def _applied(width: str) -> ConnectomePPOBrain:
        brain = _brain(width)
        brain.topology.set_anatomical_readout()
        return brain

    def test_it_expands_to_the_same_policy(self) -> None:
        a, b = self._applied("pooled"), self._applied("per_neuron")
        h = _hidden(a)
        assert torch.allclose(_mu(a, h), _mu(b, h), atol=_POLICY_TOL)

    def test_the_contrasts_survive_the_expansion(self) -> None:
        # Turn is dorsal-minus-ventral, speed is forward-minus-backward. Per neuron, every member of
        # a class must carry its class's sign.
        brain = self._applied("per_neuron")
        readout = _readout(brain)
        for class_index, cls in enumerate(_MOTOR_CLASSES):
            start, stop = brain.topology._motor_class_slices[class_index]
            turn = readout[_TURN_ACTION_INDEX, start:stop]
            speed = readout[_SPEED_ACTION_INDEX, start:stop]
            assert torch.all(turn > 0) if cls.startswith("D") else torch.all(turn < 0), cls
            assert torch.all(speed > 0) if cls.endswith("B") else torch.all(speed < 0), cls

    def test_a_frozen_arm_keeps_its_own_width(self) -> None:
        assert tuple(_readout(self._applied("pooled")).shape) == (2, len(_MOTOR_CLASSES))
        assert tuple(_readout(self._applied("per_neuron")).shape) == (2, _N_MOTOR)


class TestTheEligibilityFollowsTheWidth:
    @pytest.mark.parametrize(
        ("width", "columns"),
        [("pooled", len(_MOTOR_CLASSES)), ("per_neuron", _N_MOTOR)],
    )
    def test_the_readout_trace_and_its_presynaptic_factor_are_sized_to_the_readout(
        self,
        width: str,
        columns: int,
    ) -> None:
        brain = _brain(width)
        trace = brain.topology.readout_trace
        pooled_motor = brain.topology.pooled_motor
        assert isinstance(trace, torch.Tensor)
        assert isinstance(pooled_motor, torch.Tensor)
        assert tuple(trace.shape) == (2, columns)
        assert tuple(pooled_motor.shape) == (columns,)

    @pytest.mark.parametrize("width", ["pooled", "per_neuron"])
    def test_pool_motor_returns_the_readouts_input(self, width: str) -> None:
        brain = _brain(width)
        pre = brain.topology._pool_motor(_hidden(brain))
        assert pre.shape[-1] == _readout(brain).shape[1]


class TestTheSymmetricProjectionFollowsTheWidth:
    """`random` is what L.0 and R.2 run, but symmetric must not keep a 4-wide map either."""

    @pytest.mark.parametrize(
        ("width", "columns"),
        [("pooled", len(_MOTOR_CLASSES)), ("per_neuron", _N_MOTOR)],
    )
    def test_it_reaches_only_the_motor_pool_at_either_width(self, width: str, columns: int) -> None:
        brain = _brain(width, plasticity_learning_signal="symmetric")
        projection = brain.topology.learning_signal_projection()
        assert projection is not None
        assert tuple(projection.shape) == (brain.topology.n_neurons, 2)
        assert _readout(brain).shape[1] == columns
        outside = torch.ones(brain.topology.n_neurons, dtype=torch.bool)
        outside[brain.topology._motor_flat_indices] = False
        assert torch.all(projection[outside] == 0.0), "the projection leaked outside the pool"

    def test_per_neuron_carries_each_neurons_own_column_with_no_class_share(self) -> None:
        brain = _brain("per_neuron", plasticity_learning_signal="symmetric")
        projection = brain.topology.learning_signal_projection()
        assert projection is not None
        expected = _readout(brain).detach().t()
        assert torch.equal(projection[brain.topology._motor_flat_indices], expected)


class TestTheActionCountIsNotTheInputWidth:
    def test_they_are_equal_only_by_coincidence(self) -> None:
        # `_N_ACTIONS` is the discrete action count; the readout's input is the motor-class count.
        # Both are 4, which is why reading one for the other went unnoticed until the width moved.
        assert len(_MOTOR_CLASSES) == _N_ACTIONS

    def test_the_readouts_input_is_keyed_to_the_motor_pool_not_the_action_set(
        self,
        pooled: ConnectomePPOBrain,
        wide: ConnectomePPOBrain,
    ) -> None:
        assert _readout(pooled).shape[1] == len(_MOTOR_CLASSES)
        assert _readout(wide).shape[1] == _N_MOTOR != _N_ACTIONS


class TestACheckpointDoesNotCrossWidths:
    """Established rather than assumed, and the mechanism is named.

    `_PLASTICITY_IDENTITY` is the wrong home for this: it holds keys whose semantics are *the same
    shapes, a different meaning*, which is exactly what a width change is NOT. The readout is a
    parameter whose shape moves with the width, so `load_state_dict`'s own size check refuses the
    load first, naming the tensor and both shapes. Pinned so that a future refactor which makes the
    shapes agree cannot let a cross-width load through in silence.
    """

    def test_a_pooled_checkpoint_is_refused_by_a_wide_brain(self) -> None:
        components = _brain("pooled").get_weight_components()
        with pytest.raises(ValueError, match=r"do not match this brain's shapes"):
            _brain("per_neuron").load_weight_components(components)

    def test_and_the_other_way_round(self) -> None:
        components = _brain("per_neuron").get_weight_components()
        with pytest.raises(ValueError, match=r"do not match this brain's shapes"):
            _brain("pooled").load_weight_components(components)

    @pytest.mark.parametrize(("src", "dst"), [("pooled", "per_neuron"), ("per_neuron", "pooled")])
    def test_a_rejected_load_mutates_nothing(self, src: str, dst: str) -> None:
        """The rejection must be clean, not partial.

        `load_state_dict` skips a mismatched tensor and raises at the END -- after copying every
        tensor that DID match. So a rejected cross-width load used to leave the brain holding the
        file's `w_chem` beside its own readout: a mixed state no config describes, and one nothing
        downstream could detect. The shapes are now checked before anything is written.
        """
        # A different seed so the source's tensors genuinely differ from the destination's.
        components = _brain(src, seed=101).get_weight_components()
        brain = _brain(dst)
        before = {n: t.detach().clone() for n, t in brain.topology.named_parameters()}
        with pytest.raises(ValueError, match=r"Nothing was loaded"):
            brain.load_weight_components(components)
        for name, tensor in brain.topology.named_parameters():
            assert torch.equal(tensor.detach(), before[name]), (
                f"{name} was mutated by a refused load"
            )

    def test_the_same_width_still_loads(self) -> None:
        # The guard must refuse the width change and nothing else.
        components = _brain("per_neuron").get_weight_components()
        _brain("per_neuron").load_weight_components(components)

    def test_readout_width_is_not_in_the_plasticity_identity_tuple(self) -> None:
        # It is not a plasticity key, and adding it there would imply the shapes agree.
        assert "readout_width" not in ConnectomePPOBrain._PLASTICITY_IDENTITY
