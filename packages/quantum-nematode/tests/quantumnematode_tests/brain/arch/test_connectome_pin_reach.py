"""A.2's pins: whether each level actually reaches the learner the arm runs.

A sweep's entire output is a claim about settings, which makes it uniquely exposed to a setting that
is **accepted, validated and then never read**. `plasticity_rate` passes its `gt=0.0` bound on a PPO
arm and a hundred committed configs set it; under PPO nothing reads it. `readout_width` under
`mlpppo` is dropped with a log warning rather than an error. Either produces an arm that looks
swept, scores cleanly, and establishes nothing — and no amount of care about the statistics would
show it, because the arm is not wrong, it is absent.

So each pin's reach is asserted here per learner, in both directions: the three construction pins
must change what the arm computes, and the two rule pins must change the reading learner while
provably leaving a PPO arm alone.

**`topology.trace_decay` is the cautionary case, and it is why an attribute is not evidence.** It is
stored on the topology at 0.9 whether the learner reads traces or not, so a test asserting "the
attribute changed" would pass on a PPO arm that ignores it entirely. What separates the two is
whether the trace substrate exists at all.

The last class carries protocol principle 6's 2026-09-21 clause into this panel: a level must change
only what it names, **at the point the arms diverge**. A.1's `dense_mask` drew 91,204 values from a
generator the rollout buffer also consumes, moving PPO's minibatch order as well as the weights, and
every parameter check passed because the divergence only exists once training starts. Two of A.2's
pins change parameter counts, so the same question is live here and is answered the same way.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.config_loader import load_simulation_config

_REPO_ROOT = Path(__file__).resolve().parents[6]
_CONFIGS = _REPO_ROOT / "configs" / "scenarios" / "foraging"
_STEM = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"

_PPO_ARM = _STEM
_PPO_FROZEN = f"{_STEM}_frozen"
_READ_ARM = f"{_STEM}_eprop_readout_only"
_READ_FROZEN = f"{_STEM}_eprop_frozen"

_SEED = 7
# klinotaxis sensing: [concentration, lateral gradient, dC/dt].
_FOOD = torch.tensor([0.5, 0.2, -0.1])


def _brain(stem: str, **overrides: Any) -> ConnectomePPOBrain:
    """Construct one arm from its committed config, with pins overridden for the probe cases."""
    brain_config = load_simulation_config(str(_CONFIGS / f"{stem}.yml")).brain
    assert brain_config is not None
    assert isinstance(brain_config.config, ConnectomePPOBrainConfig)
    updated = brain_config.config.model_copy(update={"seed": _SEED, **overrides})
    torch.manual_seed(_SEED)  # the readout's orthogonal draw uses torch's global RNG
    return ConnectomePPOBrain(config=updated, device=DeviceType.CPU)


def _parameters(brain: ConnectomePPOBrain) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in brain.topology.named_parameters()}


class TestTheConstructionPinsReachBothLearners:
    """Readout width, settling depth and initial noise change what the arm computes."""

    @pytest.mark.parametrize("stem", [_PPO_ARM, _READ_ARM])
    def test_readout_width_changes_the_readout_it_names(self, stem: str) -> None:
        pooled = _brain(stem, readout_width="pooled")
        wide = _brain(stem, readout_width="per_neuron")
        assert pooled.topology.readout.numel() == 8
        assert wide.topology.readout.numel() == 78

    @pytest.mark.parametrize("stem", [_PPO_ARM, _READ_ARM])
    @pytest.mark.parametrize("depth", [2, 3, 6])
    def test_forward_pass_depth_changes_what_the_arm_computes(self, stem: str, depth: int) -> None:
        # Not merely the attribute: the settling loop runs a different number of hops, so the motor
        # activations for one fixed input differ. An attribute nothing reads would pass the first
        # assertion and fail the second, which is the whole point of making both.
        centre = _brain(stem, forward_pass_depth=4)
        other = _brain(stem, forward_pass_depth=depth)
        assert other.topology.forward_pass_depth == depth
        with torch.no_grad():
            assert not torch.equal(centre.topology.forward(_FOOD), other.topology.forward(_FOOD))

    @pytest.mark.parametrize("stem", [_PPO_ARM, _READ_ARM])
    @pytest.mark.parametrize("value", [-1.5, -0.5, 0.5])
    def test_initial_log_std_changes_the_action_noise(self, stem: str, value: float) -> None:
        arm = _brain(stem, initial_log_std=value)
        assert torch.allclose(arm.topology.log_std.detach(), torch.tensor(value))


class TestTheRulePinsReachTheReadingLearnerOnly:
    """The two learning-only pins, in both directions — the failure this panel is exposed to."""

    @pytest.mark.parametrize("rate", [0.0001, 0.01])
    def test_plasticity_rate_reaches_the_reading_learner(self, rate: float) -> None:
        arm = _brain(_READ_ARM, plasticity_rate=rate)
        assert arm._rule.plasticity_rate == rate

    def test_plasticity_rate_cannot_reach_a_ppo_arm(self) -> None:
        # The config accepts it, the bound passes, and the rule that runs has no such quantity.
        arm = _brain(_PPO_ARM, plasticity_rate=0.0001)
        assert arm._uses_ppo
        assert not hasattr(arm._rule, "plasticity_rate")

    @pytest.mark.parametrize("decay", [0.5, 0.99])
    def test_trace_decay_reaches_the_reading_learner(self, decay: float) -> None:
        arm = _brain(_READ_ARM, trace_decay=decay)
        assert arm.topology.trace_decay == decay
        # The substrate the pin governs has to exist, or the number is decoration.
        assert arm.config.enable_activity_traces
        assert arm.topology.activity_traces is not None

    def test_trace_decay_cannot_reach_a_ppo_arm(self) -> None:
        # The attribute moves and nothing reads it: a PPO arm has no trace substrate at all — the
        # buffer is never even allocated. This is why the reach test is written against the
        # substrate rather than against the field, which would have passed here.
        arm = _brain(_PPO_ARM, trace_decay=0.5)
        assert arm.topology.trace_decay == 0.5
        assert not arm.config.enable_activity_traces
        assert not hasattr(arm.topology, "activity_traces")

    @pytest.mark.parametrize("stem", [_PPO_FROZEN, _READ_FROZEN])
    @pytest.mark.parametrize(("pin", "value"), [("plasticity_rate", 0.0001), ("trace_decay", 0.5)])
    def test_neither_rule_pin_reaches_a_frozen_arm(self, stem: str, pin: str, value: float) -> None:
        # This is what licenses one frozen floor serving every level of these two pins: a frozen arm
        # performs no updates, so neither pin can move it, and the centre's floor is the correct
        # control at the rate and decay levels rather than a substitute for a missing one.
        centre = _parameters(_brain(stem))
        other = _parameters(_brain(stem, **{pin: value}))
        assert centre.keys() == other.keys()
        for name, param in centre.items():
            assert torch.equal(param, other[name]), f"{name} moved under {pin} on a frozen arm"


class TestALevelChangesOnlyWhatItNames:
    """Protocol principle 6's 2026-09-21 clause, at the point the arms diverge.

    ``rng`` is shared with the rollout buffer, whose minibatch permutation consumes it. A level
    taking a different NUMBER of values from it would move PPO's minibatch order as well as the
    thing it names — two manipulations under one name, invisible in the initial parameters. Two of
    A.2's pins change parameter counts, so this is asserted rather than read off a comment.
    """

    def test_the_buffer_shares_the_brain_generator(self) -> None:
        # If this stops being true the rest of the class is checking nothing.
        arm = _brain(_PPO_ARM)
        assert arm.buffer.rng is arm.rng

    @pytest.mark.parametrize(
        ("pin", "value"),
        [
            ("readout_width", "per_neuron"),
            ("forward_pass_depth", 2),
            ("forward_pass_depth", 6),
            ("initial_log_std", -1.0),
            ("initial_log_std", 0.5),
        ],
    )
    def test_every_level_leaves_the_shared_generator_where_the_centre_did(
        self,
        pin: str,
        value: Any,
    ) -> None:
        centre = _brain(_PPO_FROZEN)
        other = _brain(_PPO_FROZEN, **{pin: value})
        expected = centre.rng.permutation(64)
        actual = other.rng.permutation(64)
        assert (expected == actual).all(), (
            f"{pin}={value!r} left the shared generator in a different state, so it would move "
            "PPO's minibatch order as well as the quantity it names"
        )

    @pytest.mark.parametrize(("pin", "value"), [("plasticity_rate", 0.0001), ("trace_decay", 0.5)])
    def test_a_rule_pin_leaves_the_shared_generator_untouched_too(
        self,
        pin: str,
        value: float,
    ) -> None:
        centre = _brain(_READ_ARM)
        other = _brain(_READ_ARM, **{pin: value})
        assert (centre.rng.permutation(64) == other.rng.permutation(64)).all()
