"""The data-placed neuromodulator layer on the connectome brain.

Covers the neuromodulator-layer spec's scenarios "The default is unchanged", "pdfr-1 removes both
PDF couples", "The restricted layer matches bitwise", "A nine-minute timescale" and the knockout
and layer refusals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch import connectome_ppo as cp
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.connectome.neuromodulator_layer import hand_placed_spec
from quantumnematode.connectome.neuromodulators import load_circuit
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[6]
_LAWNS = (
    _REPO / "configs/scenarios/foraging/"
    "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_connectome_e005.yml"
)


def _brain(**overrides: object) -> ConnectomePPOBrain:
    container = load_simulation_config(str(_LAWNS)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    config = container.config.model_copy(
        update={"seed": 11, "rollout_buffer_size": 16, "num_minibatches": 2, "num_epochs": 2}
        | {"modulators": True}
        | overrides,
    )
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU)


def _restricted(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> ConnectomePPOBrain:
    """Build a data-layer brain whose layer is the hand-placed circuit written as a spec."""
    monkeypatch.setattr(
        cp,
        "_data_layer_spec",
        lambda _config, names: hand_placed_spec(load_circuit(names)),
    )
    return _brain(modulator_layer="data", **overrides)


def _params(step: int) -> BrainParams:
    return BrainParams(
        food_concentration=0.1 * (step % 5),
        food_lateral_gradient=0.05 * (step % 3) - 0.05,
        food_dconcentration_dt=0.01 * step,
        satiety=100.0 + 10.0 * (step % 4),
        max_satiety=300.0,
        food_ingested=0.3 if step % 4 == 0 else None,
    )


def _act(brain: ConnectomePPOBrain, params: BrainParams) -> np.ndarray:
    actions = brain.run_brain(
        params,
        reward=None,
        input_data=None,
        top_only=False,
        top_randomize=False,
    )
    return np.asarray(actions[0].continuous_mean)


class TestTheDefault:
    def test_the_default_layer_is_hand_placed(self) -> None:
        """Modulators with no layer named build the hand-placed field and its four strengths."""
        brain = _brain()
        assert brain.topology.modulator_layer == "hand_placed"
        assert brain.topology.modulator_rho.shape == (4,)
        assert brain._n_concentrations == 2


class TestTheRestrictedLayer:
    @pytest.mark.parametrize(
        ("knockout", "genes"),
        [("none", []), ("tph1", ["tph-1"]), ("pdfr1", ["pdfr-1"])],
    )
    def test_the_restricted_layer_matches_the_hand_placed_field(
        self,
        monkeypatch: pytest.MonkeyPatch,
        knockout: str,
        genes: list[str],
    ) -> None:
        """Currents match bitwise on reachable inputs; whole steps match within float32 rounding.

        A producer knockout holds its ligand at zero, so its concentration input is zero in both;
        the hand-placed field also zeroes MOD-1's row, which then carries nothing either way.
        """
        hand = _brain(modulator_knockout=knockout)
        data = _restricted(monkeypatch, modulator_knockout_genes=genes)
        if knockout != "tph1":
            assert torch.equal(hand.topology.modulator_map, data.topology.modulator_map)
        rng = torch.Generator().manual_seed(3)
        inputs = torch.rand((64, 3), generator=rng)
        for k in data._knocked_ligands:
            inputs[:, k] = 0.0
        with torch.no_grad():
            assert torch.equal(
                hand.topology.modulator_current(inputs),
                data.topology.modulator_current(inputs),
            )
        for step in range(30):
            a, b = _act(hand, _params(step)), _act(data, _params(step))
            np.testing.assert_allclose(a, b, atol=1e-5)
            np.testing.assert_allclose(hand._concentrations, data._concentrations, atol=1e-6)


class TestKnockouts:
    def test_pdfr1_removes_both_pdf_rows(self) -> None:
        """A pdfr-1 knockout zeroes every couple through PDFR-1; every other row is unchanged."""
        whole = _brain(modulator_layer="data")
        knocked = _brain(modulator_layer="data", modulator_knockout_genes=["pdfr-1"])
        spec = whole.topology.modulator_spec
        for i, row in enumerate(spec.rows):
            if row.receptor == "pdfr-1":
                assert not knocked.topology.modulator_map[i].any()
                assert whole.topology.modulator_map[i].any()
            else:
                assert torch.equal(
                    knocked.topology.modulator_map[i],
                    whole.topology.modulator_map[i],
                )

    def test_a_producer_gene_holds_its_ligand_at_zero(self) -> None:
        """tph-1 holds serotonin at zero, step after step."""
        brain = _brain(modulator_layer="data", modulator_knockout_genes=["tph-1"])
        serotonin = brain.topology.modulator_spec.ligands.index("serotonin")
        for step in range(10):
            _act(brain, _params(step))
            assert brain._concentrations[serotonin] == 0.0

    def test_an_unknown_gene_is_refused(self) -> None:
        """A gene the layer carries neither as producer nor receptor is refused at construction."""
        with pytest.raises(ValueError, match="neither a producer nor a receptor gene"):
            _brain(modulator_layer="data", modulator_knockout_genes=["unc-1"])


class TestRefusals:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"modulator_layer": "data", "modulator_knockout": "tph1"}, "modulator_knockout_genes"),
            ({"modulator_knockout_genes": ["tph-1"]}, "require modulator_layer='data'"),
            (
                {"modulator_layer": "data", "modulator_layer_null": "receptor_placement"},
                "set together",
            ),
            ({"modulators": False, "modulator_layer": "data"}, "require modulators"),
        ],
    )
    def test_settings_that_cannot_run_are_refused(
        self,
        overrides: dict[str, object],
        match: str,
    ) -> None:
        """Each knob refuses the other layer, and the layer refuses to run without the field."""
        with pytest.raises(ValueError, match=match):
            _brain(**overrides)

    def test_a_timescale_for_an_unknown_ligand_is_refused(self) -> None:
        """Naming a ligand the layer does not carry is refused."""
        with pytest.raises(ValueError, match="does not carry"):
            _brain(modulator_layer="data", modulator_timescales_s={"no-such-1": 30.0})


class TestTheDataLayer:
    def test_fitted_rows_start_at_zero_and_fixed_rows_at_one(self) -> None:
        """Fitted couples carry no current until learned; fixed ones start at strength 1."""
        brain = _brain(modulator_layer="data")
        spec = brain.topology.modulator_spec
        strengths = brain.topology.modulator_layer_strengths().detach()
        for i, row in enumerate(spec.rows):
            assert strengths[i].item() == pytest.approx(0.0 if row.sign is None else 1.0)
        assert brain._n_concentrations == len(spec.ligands)

    def test_a_nine_minute_timescale_holds_its_state(self) -> None:
        """At a 1-s step, τ = 540 s leaves 1/e of a stopped ligand after 540 steps."""
        brain = _brain(
            modulator_layer="data",
            modulator_step_seconds=1.0,
            modulator_timescales_s={"serotonin": 540.0},
        )
        topology = brain.topology
        k = topology.modulator_spec.ligands.index("serotonin")
        row = next(
            i
            for i, r in enumerate(topology.modulator_spec.rows)
            if r.ligand == k and r.sign is not None
        )
        brain._concentrations[k] = 1.0
        quiet = torch.full((topology.n_neurons,), -1.0)
        crossed = None
        for step in range(1, 600):
            brain._update_concentrations(quiet)
            if crossed is None and brain._concentrations[k] <= np.exp(-1.0):
                crossed = step
        assert crossed is not None
        assert abs(crossed - 540) <= 1
        inputs = torch.zeros(brain._n_concentrations + 1)
        inputs[k] = 0.5
        with torch.no_grad():
            half = topology.modulator_current(inputs)
            inputs[k] = 1.0
            full = topology.modulator_current(inputs)
        cells = topology.modulator_map[row] != 0
        assert torch.allclose(full[cells], 2.0 * half[cells])
