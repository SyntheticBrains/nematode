"""Tests for learning-rule selection on the connectome brain.

Two rules now share one brain class, which is what keeps the panel's arms
differing in exactly the dimension under test. That only holds if the
default path is untouched and the plastic path genuinely leaves the PPO
machinery alone — both pinned here.
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

_SEED = 909
_STEPS = 6


def _make(**overrides: object) -> ConnectomePPOBrain:
    cfg = ConnectomePPOBrainConfig(
        seed=_SEED,
        action_mode="continuous",
        **overrides,  # type: ignore[arg-type]
    )
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


def _plastic(**overrides: object) -> ConnectomePPOBrain:
    return _make(learning_rule="three_factor", enable_activity_traces=True, **overrides)


def _drive(brain: ConnectomePPOBrain, steps: int = _STEPS) -> None:
    brain.prepare_episode()
    torch.manual_seed(_SEED + 1)
    for step in range(steps):
        brain.run_brain(
            BrainParams(
                food_gradient_strength=0.3 + 0.05 * step,
                food_gradient_direction=0.1 * step - 0.2,
            ),
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )
        brain.learn(BrainParams(), reward=0.4 * (step % 3) - 0.2, episode_done=(step == steps - 1))


class TestRulePairingValidation:
    """A plasticity rule with no trace would train silently and do nothing."""

    def test_three_factor_without_traces_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires enable_activity_traces"):
            ConnectomePPOBrainConfig(learning_rule="three_factor")

    def test_three_factor_with_traces_accepted(self) -> None:
        cfg = ConnectomePPOBrainConfig(learning_rule="three_factor", enable_activity_traces=True)
        assert cfg.learning_rule == "three_factor"

    def test_default_is_ppo(self) -> None:
        assert ConnectomePPOBrainConfig().learning_rule == "ppo"


class TestConstructionParity:
    """Selecting a rule must not perturb the substrate it trains."""

    def test_all_parameters_but_readout_are_identical(self) -> None:
        """The PPO critic's init consumes RNG under both selections."""
        ppo = _make(enable_activity_traces=True)
        plastic = _plastic()

        for name in ("w_chem", "food_gains", "g_gap", "m_chem"):
            assert torch.equal(
                getattr(ppo.topology, name),
                getattr(plastic.topology, name),
            ), name

    def test_default_config_matches_explicit_ppo(self) -> None:
        default = _make()
        explicit = _make(learning_rule="ppo")
        assert torch.equal(default.topology.w_chem, explicit.topology.w_chem)
        assert torch.equal(default.topology.readout, explicit.topology.readout)


class TestAnatomicalReadout:
    """Frozen decoding must respect what the motor pools mean."""

    def test_plastic_readout_encodes_the_contrasts(self) -> None:
        readout = _plastic().topology.readout.detach()
        speed, turn = readout[0], readout[1]
        # Pool order is (VB, DB, VA, DA): V/D ventral/dorsal, B/A forward/backward.
        assert speed[0] > 0  # VB, forward drive
        assert speed[1] > 0  # DB, forward drive
        assert speed[2] < 0  # VA, backward drive
        assert speed[3] < 0  # DA, backward drive
        assert turn[1] > 0  # DB, dorsal
        assert turn[3] > 0  # DA, dorsal
        assert turn[0] < 0  # VB, ventral
        assert turn[2] < 0  # VA, ventral

    def test_anatomical_readout_is_orthonormal(self) -> None:
        """Same scale and conditioning as the random orthogonal init."""
        readout = _plastic().topology.readout.detach()
        assert readout[0].norm().item() == pytest.approx(1.0)
        assert readout[1].norm().item() == pytest.approx(1.0)
        assert torch.dot(readout[0], readout[1]).item() == pytest.approx(0.0, abs=1e-6)

    def test_ppo_readout_is_unchanged(self) -> None:
        """Previously measured PPO results must stay reproducible."""
        readout = _make().topology.readout.detach()
        anatomical = _plastic().topology.readout.detach()
        assert not torch.equal(readout, anatomical)

    def test_readout_identical_across_wiring_arms(self) -> None:
        """The decoder must not confound the wild-type vs rewired contrast."""
        wild = _plastic()
        rewired = _plastic(wiring="rewired_degree_preserving")
        assert torch.equal(wild.topology.readout, rewired.topology.readout)


class TestPlasticUpdateCadence:
    """One update per environment step, with no rollout involved."""

    def test_one_update_per_step(self) -> None:
        brain = _plastic()
        _drive(brain, steps=_STEPS)
        assert len(brain.history_data.plasticity_prediction_error) == _STEPS

    def test_buffer_stays_empty(self) -> None:
        brain = _plastic()
        _drive(brain)
        assert len(brain.buffer) == 0

    def test_no_value_is_computed(self) -> None:
        """Action selection must not require a value head that does not exist."""
        brain = _plastic()
        _drive(brain)
        assert brain.last_value is None

    def test_weights_change(self) -> None:
        brain = _plastic()
        before = brain.topology.w_chem.detach().clone()
        _drive(brain)
        assert not torch.equal(before, brain.topology.w_chem)

    def test_only_chemical_weights_change(self) -> None:
        brain = _plastic()
        gains = brain.topology.food_gains.detach().clone()
        readout = brain.topology.readout.detach().clone()
        _drive(brain)
        assert torch.equal(gains, brain.topology.food_gains)
        assert torch.equal(readout, brain.topology.readout)

    def test_telemetry_is_recorded(self) -> None:
        brain = _plastic()
        _drive(brain)
        assert len(brain.history_data.plasticity_baseline) == _STEPS
        assert len(brain.history_data.plasticity_mean_abs_delta) == _STEPS
        assert len(brain.history_data.plasticity_saturated_fraction) == _STEPS


class TestPpoOnlyAccessors:
    """A question that does not apply should say so, not raise from a shim."""

    @pytest.mark.parametrize(
        ("attribute", "phrase"),
        [("critic", "value head"), ("optimizer", "optimiser")],
    )
    def test_accessor_explains_itself(self, attribute: str, phrase: str) -> None:
        brain = _plastic()
        with pytest.raises(AttributeError, match=phrase) as excinfo:
            getattr(brain, attribute)
        assert "three_factor" in str(excinfo.value)

    def test_accessors_still_work_under_ppo(self) -> None:
        brain = _make()
        assert brain.critic is not None
        assert brain.optimizer is not None


class TestTraceLifecycleUnderThePlasticRule:
    """Eligibility resets with the episode, not across it."""

    def test_first_step_of_an_episode_accrues_nothing(self) -> None:
        brain = _plastic()
        brain.prepare_episode()
        torch.manual_seed(_SEED + 2)
        brain.run_brain(
            BrainParams(food_gradient_strength=0.5, food_gradient_direction=0.1),
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )
        assert torch.all(brain.topology.activity_traces == 0)

    def test_prepare_episode_clears_trace_and_history(self) -> None:
        brain = _plastic()
        _drive(brain)
        assert brain.topology.activity_traces.abs().sum() > 0

        brain.prepare_episode()

        assert torch.all(brain.topology.activity_traces == 0)
        assert torch.all(brain.topology.prev_activity == 0)


class TestStabilisationBoundsValidation:
    """Unusable plasticity settings fail before a run, not during one."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("plasticity_rate", 0.0),  # a zero rate makes the rule a no-op
            ("plasticity_rate", -0.1),
            ("plasticity_weight_decay", -0.1),
            ("plasticity_weight_decay", 1.0),  # >= 1 inverts weights each step
            ("plasticity_weight_bound", 0.0),  # collapses every synapse to zero
            ("plasticity_weight_bound", -1.0),
            ("plasticity_baseline_rate", 0.0),  # baseline would never track
            ("plasticity_baseline_rate", 1.5),
        ],
    )
    def test_out_of_range_setting_rejected(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError, match=field):
            ConnectomePPOBrainConfig(
                learning_rule="three_factor",
                enable_activity_traces=True,
                **{field: value},  # type: ignore[arg-type]
            )

    def test_in_range_settings_accepted(self) -> None:
        cfg = ConnectomePPOBrainConfig(
            learning_rule="three_factor",
            enable_activity_traces=True,
            plasticity_rate=0.05,
            plasticity_weight_decay=0.01,
            plasticity_weight_bound=2.0,
            plasticity_baseline_rate=0.1,
        )
        assert cfg.plasticity_rate == 0.05


class TestRewardCreditsTheStepThatEarnedIt:
    """Alignment is inclusive of the current step, by design."""

    def test_reward_gates_eligibility_including_its_own_step(self) -> None:
        """The synapses that produced the action are the ones credited.

        The trace is updated during the forward pass that selects an action,
        and the reward for that action arrives on the following ``learn``.
        Excluding the current step would be equally implementable and would
        silently change what the rule credits, so the inclusive semantics
        are pinned here.

        Driven from the second step onward, since the first accrues no
        eligibility and so cannot distinguish the two conventions.
        """
        # Decay off: it is proportional to the weights rather than the trace,
        # so it would mask the alignment this test isolates.
        brain = _plastic(plasticity_weight_decay=0.0)
        brain.prepare_episode()
        torch.manual_seed(_SEED + 3)

        def act(step: int) -> None:
            brain.run_brain(
                BrainParams(
                    food_gradient_strength=0.4 + 0.05 * step,
                    food_gradient_direction=0.1 * step,
                ),
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )

        # Step 0 accrues nothing; step 1 is the first with eligibility.
        act(0)
        brain.learn(BrainParams(), reward=0.0)
        act(1)

        trace_before_update = brain.topology.activity_traces.detach().clone()
        assert trace_before_update.abs().sum() > 0

        weights_before = brain.topology.w_chem.detach().clone()
        brain.learn(BrainParams(), reward=1.0)
        applied = brain.topology.w_chem - weights_before

        # The applied change must be the plasticity rate times the prediction
        # error times the trace that ALREADY INCLUDES this step's
        # contribution. Had the rule gated only strictly-earlier eligibility,
        # the expected tensor below would be the previous step's trace and
        # this would not hold.
        errors = brain.history_data.plasticity_prediction_error
        expected = brain.config.plasticity_rate * errors[-1] * trace_before_update

        assert torch.allclose(applied, expected, rtol=0.0, atol=1e-7)
        assert torch.all(applied[~brain.topology.m_chem] == 0)


class TestBoundClearsInitialisation:
    """The clamp must not modify the substrate before learning starts."""

    def test_default_bound_does_not_clip_initial_weights(self) -> None:
        """Otherwise the plastic arm starts from a different substrate.

        Chemical weights are drawn N(0, 1/sqrt(chemical in-degree)), whose
        tail reaches well past 1.0 on this connectome. A bound at or below
        that tail would clamp a handful of synapses on the first update,
        so this arm would no longer begin where the frozen-weights baseline
        it is compared against begins.
        """
        for seed in (7, 909, 4242):
            brain = ConnectomePPOBrain(
                config=ConnectomePPOBrainConfig(
                    seed=seed,
                    action_mode="continuous",
                    learning_rule="three_factor",
                    enable_activity_traces=True,
                ),
                device=DeviceType.CPU,
            )
            edges = brain.topology.m_chem
            largest = brain.topology.w_chem.detach()[edges].abs().max().item()
            assert largest < brain.config.plasticity_weight_bound, seed

    def test_initial_weights_survive_a_no_op_update(self) -> None:
        """A zero prediction error must leave the substrate exactly as found."""
        brain = _plastic(plasticity_weight_decay=0.0)
        before = brain.topology.w_chem.detach().clone()
        brain.prepare_episode()
        torch.manual_seed(_SEED + 4)
        brain.run_brain(
            BrainParams(food_gradient_strength=0.5, food_gradient_direction=0.0),
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )
        brain.learn(BrainParams(), reward=0.0)

        assert torch.equal(before, brain.topology.w_chem)
