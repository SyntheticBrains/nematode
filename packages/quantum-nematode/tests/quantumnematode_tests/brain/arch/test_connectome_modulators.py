"""Serotonin and PDF on the connectome brain.

Covers the connectome-ppo-brain requirements "The neuromodulator field" (off is unchanged;
serotonin inhibits its receptor cells; ingestion reaches serotonin through settling; PDF inhibits
NSM; an old checkpoint still loads; a concentration persists and decays; replay sees the rollout's
concentrations) and "Modulator knockouts" (a tph-1 knockout removes serotonin; a pdfr-1 knockout
removes every PDFR-1 effect).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
    ConnectomeTopology,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome.loader import load_emmons_2024_hermaphrodite
from quantumnematode.connectome.neuromodulators import class_members, load_circuit
from quantumnematode.learning_rules.ppo import ConnectomePPOBatch
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[6]
_LAWNS = (
    _REPO / "configs/scenarios/foraging/"
    "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_connectome_e005.yml"
)
_ZERO = torch.zeros(3)


def _brain(**overrides: object) -> ConnectomePPOBrain:
    container = load_simulation_config(str(_LAWNS)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    config = container.config.model_copy(
        update={
            "seed": 11,
            "rollout_buffer_size": 16,
            "num_minibatches": 2,
            "num_epochs": 2,
            **overrides,
        },
    )
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU)


def _params(step: int, *, ingested: float | None = None) -> BrainParams:
    return BrainParams(
        food_concentration=0.1 * (step % 5),
        food_lateral_gradient=0.05 * (step % 3) - 0.05,
        food_dconcentration_dt=0.01 * step,
        satiety=100.0 + 10.0 * (step % 4),
        max_satiety=300.0,
        food_ingested=ingested,
    )


def _settle(topo: ConnectomeTopology, modulators: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        _, hidden = topo.forward_with_hidden(torch.zeros(3), modulators=modulators)
    return hidden


def _cells(topo: ConnectomeTopology, *classes: str) -> list[int]:
    return [topo._idx[n] for c in classes for n in class_members(c, topo.neuron_names)]


def _act(brain: ConnectomePPOBrain, params: BrainParams) -> None:
    brain.run_brain(params, reward=None, input_data=None, top_only=False, top_randomize=False)


# ── The circuit ──────────────────────────────────────────────────────────────────────────────


class TestCircuit:
    def test_every_cited_class_maps_to_its_left_and_right_cells(self) -> None:
        circuit = load_circuit(sorted(load_emmons_2024_hermaphrodite().neurons))
        assert circuit.serotonin_release == ("NSML", "NSMR")
        assert circuit.pdf_release == ("AVBL", "AVBR", "PVPL", "PVPR", "SIAVL", "SIAVR")
        assert circuit.mod1 == ("AIYL", "AIYR", "ASIL", "ASIR", "RIFL", "RIFR")
        assert circuit.pdfr1 == ("AIYL", "AIYR", "RIML", "RIMR", "RIAL", "RIAR")
        assert circuit.pdf_inhibits == ("NSML", "NSMR")
        assert (circuit.mod1_sign, circuit.pdfr1_sign) == (-1, 1)
        assert circuit.timescale_s == 60.0
        assert circuit.skipped == ()

    def test_an_unpaired_class_maps_to_itself(self) -> None:
        assert class_members("AVM", ["AVM", "AVBL"]) == ["AVM"]
        assert class_members("RIF", ["AVM"]) == []

    def test_a_connectome_without_a_receptor_cell_is_refused(self) -> None:
        names = sorted(load_emmons_2024_hermaphrodite().neurons)
        stripped = [n for n in names if not n.startswith(("AIY", "RIM", "RIA"))]
        with pytest.raises(ValueError, match="PDFR-1"):
            load_circuit(stripped)


# ── Off is unchanged ─────────────────────────────────────────────────────────────────────────


class TestOff:
    def test_off_allocates_nothing(self) -> None:
        brain = _brain()
        assert not brain.topology.modulators
        assert "modulator_rho" not in brain.topology.state_dict()
        assert brain.preprocess(_params(1, ingested=0.5)).shape == (3,)
        assert brain.modulator_concentrations is None

    def test_turning_it_on_moves_no_other_tensor(self) -> None:
        off, on = _brain().topology, _brain(modulators=True).topology
        on_state = on.state_dict()
        assert set(on_state) - set(off.state_dict()) == {"modulator_rho"}
        for name, value in off.state_dict().items():
            assert torch.equal(value, on_state[name]), name
        assert on.learnable_parameters[-1] is on.modulator_rho
        assert len(on.learnable_parameters) == len(off.learnable_parameters) + 1

    def test_with_no_concentration_and_nothing_eaten_the_forward_is_unchanged(self) -> None:
        off, on = _brain().topology, _brain(modulators=True).topology
        food = torch.tensor([0.3, -0.2, 0.1])
        with torch.no_grad():
            mean_off, h_off = off.forward_with_hidden(food)
            mean_on, h_on = on.forward_with_hidden(food, modulators=_ZERO)
        assert torch.equal(mean_off, mean_on)
        assert torch.equal(h_off, h_on)


# ── The field's effects ──────────────────────────────────────────────────────────────────────


class TestField:
    def test_strengths_start_at_one(self) -> None:
        topo = _brain(modulators=True).topology
        assert torch.allclose(topo.modulator_strengths(), torch.ones(4))

    def test_serotonin_inhibits_its_receptor_cells(self) -> None:
        topo = _brain(modulators=True).topology
        cells = _cells(topo, "AIY", "ASI", "RIF")
        base = _settle(topo, _ZERO)[cells]
        dosed = _settle(topo, torch.tensor([0.5, 0.0, 0.0]))[cells]
        assert bool((dosed < base).all())

    def test_pdf_excites_its_receptor_cells(self) -> None:
        topo = _brain(modulators=True).topology
        cells = _cells(topo, "RIM", "RIA")
        base = _settle(topo, _ZERO)[cells]
        dosed = _settle(topo, torch.tensor([0.0, 0.5, 0.0]))[cells]
        assert bool((dosed > base).all())

    def test_ingestion_reaches_nsm_at_the_end_of_settling(self) -> None:
        topo = _brain(modulators=True).topology
        nsm = _cells(topo, "NSM")
        base = _settle(topo, _ZERO)[nsm]
        fed = _settle(topo, torch.tensor([0.0, 0.0, 0.5]))[nsm]
        assert bool((fed > base + 0.1).all())

    def test_pdf_inhibits_nsm(self) -> None:
        topo = _brain(modulators=True).topology
        nsm = _cells(topo, "NSM")
        fed = torch.tensor([0.0, 0.0, 0.5])
        base = _settle(topo, fed)[nsm]
        inhibited = _settle(topo, fed + torch.tensor([0.0, 0.5, 0.0]))[nsm]
        assert bool((inhibited < base).all())

    def test_a_concentration_decays_by_its_timescale(self) -> None:
        brain = _brain(modulators=True)
        brain._concentrations = np.array([1.0, 1.0], dtype=np.float32)
        brain._update_concentrations(torch.zeros(brain.topology.n_neurons))
        expected = np.exp(-5.0 / 60.0)
        assert brain.modulator_concentrations == pytest.approx((expected, expected))

    def test_release_raises_the_concentration_for_the_next_step(self) -> None:
        brain = _brain(modulators=True)
        _act(brain, _params(0, ingested=1.0))
        serotonin, _ = brain.modulator_concentrations or (0.0, 0.0)
        assert serotonin > 0.0
        state = brain.preprocess(_params(1, ingested=0.0))
        assert state[-3] == pytest.approx(serotonin)

    def test_the_episode_starts_with_no_modulator(self) -> None:
        brain = _brain(modulators=True)
        for step in range(3):
            _act(brain, _params(step, ingested=1.0))
        brain.prepare_episode()
        assert brain.modulator_concentrations == (0.0, 0.0)

    def test_the_drive_is_a_fraction_of_a_full_bite(self) -> None:
        params = _params(0, ingested=0.01).model_copy(update={"max_food_ingested": 0.02})
        assert _brain(modulators=True).preprocess(params)[-1] == pytest.approx(0.5)

    def test_the_satiety_link_amplifies_ingestion_by_hunger(self) -> None:
        plain = _brain(modulators=True).preprocess(_params(0, ingested=0.4))
        linked = _brain(modulators=True, modulator_satiety_link=True).preprocess(
            _params(0, ingested=0.4),
        )
        assert plain[-1] == pytest.approx(0.4)
        assert linked[-1] == pytest.approx(0.4 * (2.0 - 100.0 / 300.0))


# ── Replay ───────────────────────────────────────────────────────────────────────────────────


class TestReplay:
    def test_replay_sees_the_rollout_s_concentrations(self) -> None:
        brain = _brain(modulators=True)
        states, means = [], []
        for step in range(8):
            params = _params(step, ingested=0.3 * (step % 2))
            state = torch.from_numpy(brain.preprocess(params))
            with torch.no_grad():
                food, *_ = brain._unpack_state(state)
                mean, _ = brain.topology.forward_with_hidden(
                    food,
                    modulators=brain._unpack_modulators(state),
                )
            states.append(state)
            means.append(mean)
            _act(brain, params)
        stacked = torch.stack(states)
        assert bool((stacked[1:, -3:-1] > 0).all())
        batch = ConnectomePPOBatch(
            buffer=brain.buffer,
            unpack_batched=brain._unpack_state_batched,
            unpack_modulators=brain._unpack_modulators,
            last_value=None,
        )
        rule = brain._ppo_rule
        assert rule is not None
        with torch.no_grad():
            replayed, _, _ = rule._replay(brain.topology, batch, {"states": stacked})
        assert torch.allclose(replayed, torch.stack(means), atol=1e-5)

    def test_the_strengths_learn(self) -> None:
        brain = _brain(modulators=True)
        before = brain.topology.modulator_rho.detach().clone()
        torch.manual_seed(0)
        for step in range(16):
            params = _params(step, ingested=0.5 * (step % 2))
            _act(brain, params)
            brain.learn(params, reward=0.1 * (step % 3), episode_done=False)
        assert not torch.equal(before, brain.topology.modulator_rho.detach())


# ── Signs under PPO ──────────────────────────────────────────────────────────────────────────


class TestSignsUnderPPO:
    def test_ppo_holds_every_grounded_synapse_to_its_sign(self) -> None:
        """Scenario "PPO keeps grounded signs": after updates no grounded weight crosses zero."""
        brain = _brain(
            modulators=True,
            synapse_signs="atlas",
            enforce_synapse_signs=True,
            learning_rate=0.05,
        )
        signs = cast("torch.Tensor", brain.topology.chem_sign)
        torch.manual_seed(0)
        for step in range(32):
            params = _params(step, ingested=0.5 * (step % 2))
            _act(brain, params)
            brain.learn(params, reward=float(step % 3) - 1.0, episode_done=False)
        weights = brain.topology.w_chem.detach()
        assert bool((weights[signs > 0] >= 0).all())
        assert bool((weights[signs < 0] <= 0).all())

    def test_without_enforcement_ppo_may_cross_zero(self) -> None:
        """The same run unenforced lets some grounded weight change sign, so the test bites."""
        brain = _brain(modulators=True, synapse_signs="atlas", learning_rate=0.05)
        signs = cast("torch.Tensor", brain.topology.chem_sign)
        torch.manual_seed(0)
        for step in range(32):
            params = _params(step, ingested=0.5 * (step % 2))
            _act(brain, params)
            brain.learn(params, reward=float(step % 3) - 1.0, episode_done=False)
        weights = brain.topology.w_chem.detach()
        crossed = (weights[signs > 0] < 0).sum() + (weights[signs < 0] > 0).sum()
        assert int(crossed) > 0


# ── Checkpoints ──────────────────────────────────────────────────────────────────────────────


class TestCheckpoints:
    def test_an_unmodulated_checkpoint_loads_and_acts_identically(self) -> None:
        saved = _brain()
        with torch.no_grad():
            saved.topology.w_chem.mul_(1.5)
        fresh = _brain(seed=12)
        fresh.load_weight_components(saved.get_weight_components())
        food = torch.tensor([0.3, -0.2, 0.1])
        with torch.no_grad():
            assert torch.equal(
                fresh.topology.forward_with_hidden(food)[0],
                saved.topology.forward_with_hidden(food)[0],
            )

    def test_a_modulated_checkpoint_round_trips(self) -> None:
        saved = _brain(modulators=True)
        with torch.no_grad():
            saved.topology.modulator_rho.copy_(torch.tensor([0.1, -1.0, 2.0, 0.5]))
        fresh = _brain(modulators=True, seed=12)
        fresh.load_weight_components(saved.get_weight_components())
        assert torch.equal(fresh.topology.modulator_rho, saved.topology.modulator_rho)

    @pytest.mark.parametrize("knockout", ["tph1", "pdfr1"])
    def test_a_trained_network_loads_into_a_knockout(self, knockout: str) -> None:
        saved = _brain(modulators=True)
        mutant = _brain(modulators=True, modulator_knockout=knockout, seed=12)
        mutant.load_weight_components(saved.get_weight_components())
        assert torch.equal(mutant.topology.w_chem, saved.topology.w_chem)


# ── Knockouts ────────────────────────────────────────────────────────────────────────────────


class TestKnockouts:
    def test_tph1_holds_serotonin_at_zero_and_no_mod1_current_flows(self) -> None:
        brain = _brain(modulators=True, modulator_knockout="tph1")
        for step in range(4):
            _act(brain, _params(step, ingested=1.0))
        concentrations = brain.modulator_concentrations
        assert concentrations is not None
        assert concentrations[0] == 0.0
        current = brain.topology.modulator_current(torch.tensor([1.0, 0.0, 0.0]))
        assert torch.count_nonzero(current) == 0

    def test_pdfr1_removes_the_receptor_current_and_the_inhibition_of_nsm(self) -> None:
        topo = _brain(modulators=True, modulator_knockout="pdfr1").topology
        current = topo.modulator_current(torch.tensor([0.0, 1.0, 0.0]))
        assert torch.count_nonzero(current) == 0
        intact = _brain(modulators=True).topology.modulator_current(torch.tensor([0.0, 1.0, 0.0]))
        assert bool((intact[_cells(topo, "NSM")] < 0).all())
        assert bool((intact[_cells(topo, "RIM", "RIA")] > 0).all())

    def test_a_knockout_leaves_the_other_modulator(self) -> None:
        topo = _brain(modulators=True, modulator_knockout="pdfr1").topology
        current = topo.modulator_current(torch.tensor([1.0, 0.0, 1.0]))
        assert bool((current[_cells(topo, "ASI", "RIF")] < 0).all())
        assert bool((current[_cells(topo, "NSM")] > 0).all())

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"modulator_knockout": "tph1"}, "require modulators"),
            ({"modulator_satiety_link": True}, "require modulators"),
            ({"modulators": True, "dynamics": "leaky"}, "dynamics='settling'"),
            (
                {
                    "modulators": True,
                    "learning_rule": "three_factor",
                    "enable_activity_traces": True,
                },
                "requires learning_rule='ppo'",
            ),
        ],
    )
    def test_unsupported_combinations_are_refused(
        self,
        overrides: dict[str, object],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ConnectomePPOBrainConfig(**overrides)  # type: ignore[arg-type]
