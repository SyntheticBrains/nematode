"""The encoder procedure's options on the connectome brain, and cited receptor signs on its layer.

Covers the connectome-encoder spec's scenarios "Off is byte-identical", "The cap bounds each
output", "Centres at zero leave the readout unchanged", "The graded box's centre", "The floor
bounds the inhibitory scale", "A floor outside the box is refused", "Prefix property", "The counts
with the new options" and "The observer sees each step and changes nothing"; and the
neuromodulator-layer spec's "Empty is byte-identical", "PDFR-1 and SER-5 fix their couples", "The
sign travels with its couple under the null", "An unresolved receptor is refused", "Every signed
entry is checked" and "ser-5 keeps its own magnitude".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.weights import WeightComponent
from quantumnematode.connectome import encoder as enc
from quantumnematode.connectome import neuromodulator_layer as layer
from quantumnematode.utils.config_loader import load_simulation_config

_REPO = Path(__file__).resolve().parents[6]
_ASSAY = (
    _REPO / "configs/scenarios/foraging/"
    "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_modulated_s1_primitives"
    "_assay_scheer_lawn.yml"
)
ENCODED: dict[str, object] = {
    "modulator_layer": "data",
    "modulator_satiety_link": True,
    "synapse_signs": "per_connection",
    "gap_normalisation": "strength",
    "encoder": "compact",
    "encoder_modulator_tying": "per_ligand",
}
PROCEDURE: dict[str, object] = {
    "encoder_readout": "soft_cap",
    "encoder_pool_centres": True,
    "encoder_box": "graded",
    "encoder_inhibition_floor": 0.09,
    "encoder_speed_noise": "ou",
    "modulator_receptor_signs": ["pdfr-1", "ser-5"],
}


def _config(**overrides: object) -> ConnectomePPOBrainConfig:
    """Return the encoded small-lawn brain's config with overrides, unvalidated as model_copy is."""
    container = load_simulation_config(str(_ASSAY)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    return container.config.model_copy(update={"seed": 11, **ENCODED, **overrides})


def _brain(**overrides: object) -> ConnectomePPOBrain:
    """Build an encoded brain with overrides on the CPU; construction applies the refusals."""
    return ConnectomePPOBrain(config=_config(**overrides), device=DeviceType.CPU)


def _vector(brain: ConnectomePPOBrain, seed: int = 3) -> np.ndarray:
    """Return a seeded unit vector away from the box's edges, sized to the brain's encoder."""
    n = len(brain.topology.encoder.entries)  # type: ignore[union-attr]
    return np.random.default_rng(seed).uniform(0.15, 0.85, size=n)


def _params(step: int) -> BrainParams:
    """Return step ``step``'s varying senses, satiety and intake."""
    return BrainParams(
        food_concentration=0.1 * (step % 5),
        food_lateral_gradient=0.05 * (step % 3) - 0.05,
        food_dconcentration_dt=0.01 * step,
        food_contact=0.4,
        satiety=100.0 + 10.0 * (step % 4),
        max_satiety=300.0,
        food_ingested=0.3 if step % 4 == 0 else None,
    )


def _means(brain: ConnectomePPOBrain, steps: int = 6) -> np.ndarray:
    """Return the brain's squashed action means over a short episode of varying senses."""
    brain.prepare_episode()
    out = []
    for t in range(steps):
        actions = brain.run_brain(_params(t), top_only=False, top_randomize=False)
        out.append(actions[0].continuous_mean)
    return np.asarray(out)


def _values(brain: ConnectomePPOBrain) -> dict[str, float]:
    """Return each encoder parameter's value, by name, as the brain carries it."""
    entries = brain.topology.encoder.entries  # type: ignore[union-attr]
    z = enc.read_encoder(brain)
    return {e.name: e.value(float(u)) for e, u in zip(entries, z, strict=True)}


class TestDefaults:
    def test_off_is_byte_identical(self) -> None:
        """Covers "Off is byte-identical": the options at their defaults change nothing."""
        plain = _brain()
        explicit = _brain(
            encoder_readout="linear",
            encoder_pool_centres=False,
            encoder_box="wide",
            encoder_inhibition_floor=None,
            modulator_receptor_signs=[],
        )
        z = _vector(plain)
        enc.write_encoder(plain, z)
        enc.write_encoder(explicit, z)
        np.testing.assert_array_equal(_means(plain), _means(explicit))
        assert set(plain.topology.state_dict()) == set(explicit.topology.state_dict())
        assert not any("pool_centres" in k for k in plain.topology.state_dict())

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("encoder_readout", "soft_cap"),
            ("encoder_pool_centres", True),
            ("encoder_box", "graded"),
            ("encoder_inhibition_floor", 0.1),
        ],
    )
    def test_the_options_need_the_encoder(self, key: str, value: object) -> None:
        """Each readout or box option away from its default is refused without the encoder."""
        with pytest.raises(ValueError, match=key):
            ConnectomePPOBrainConfig(**{key: value})  # type: ignore[arg-type]

    def test_receptor_signs_need_the_data_layer(self) -> None:
        """Cited receptor signs are refused without the data layer."""
        with pytest.raises(ValueError, match="modulator_receptor_signs"):
            ConnectomePPOBrainConfig(modulator_receptor_signs=["pdfr-1"])


class TestReadout:
    def test_the_cap_bounds_each_output(self) -> None:
        """Covers "The cap bounds each output": capped outputs are 2·tanh(x/2) of linear ones."""
        linear = _brain()
        capped = _brain(encoder_readout="soft_cap")
        entries = linear.topology.encoder.entries  # type: ignore[union-attr]
        z = _vector(linear)
        z[[i for i, e in enumerate(entries) if e.name.endswith(".bias")]] = 1.0
        enc.write_encoder(linear, z)
        enc.write_encoder(capped, z)
        features = torch.from_numpy(
            np.random.default_rng(5).uniform(-1, 1, size=(16, linear.topology.n_food_features)),
        ).float()
        with torch.no_grad():
            raw, _ = linear.topology.forward_with_hidden_batched(features)
            soft, _ = capped.topology.forward_with_hidden_batched(features)
        assert float(raw.abs().max()) > enc.READOUT_CAP
        np.testing.assert_allclose(
            soft.numpy(),
            enc.READOUT_CAP * np.tanh(raw.numpy() / enc.READOUT_CAP),
            atol=1e-6,
        )
        assert float(soft.abs().max()) < enc.READOUT_CAP

    def test_centres_at_zero_leave_the_readout_unchanged(self) -> None:
        """Covers "Centres at zero leave the readout unchanged" and the "Prefix property"."""
        without = _brain()
        with_centres = _brain(encoder_pool_centres=True)
        z = _vector(without)
        enc.write_encoder(without, z)
        enc.write_encoder(with_centres, np.concatenate([z, np.full(len(enc.POOLS), 0.5)]))
        np.testing.assert_array_equal(_means(without), _means(with_centres))

    def test_centres_move_the_readout(self) -> None:
        """Centres away from zero change the actions, so the readout reads them."""
        brain = _brain(encoder_pool_centres=True)
        z = _vector(brain)
        enc.write_encoder(brain, z)
        before = _means(brain)
        z[-len(enc.POOLS) :] = 0.9
        enc.write_encoder(brain, z)
        assert not np.allclose(before, _means(brain))

    def test_the_centres_are_learnable(self) -> None:
        """The pool centres are among PPO's learnable parameters."""
        brain = _brain(encoder_pool_centres=True)
        centres = brain.topology.encoder.pool_centres  # type: ignore[union-attr]
        assert centres is not None
        assert any(p is centres for p in brain.topology.learnable_parameters)


class TestBox:
    def test_the_graded_box_centre(self) -> None:
        """Covers "The graded box's centre": wiring scales at 0.375, sensory gains at 0."""
        values = _values(_brain(encoder_box="graded"))
        for name in ("wiring.excitatory", "wiring.inhibitory", "wiring.gap"):
            assert values[name] == pytest.approx(0.375)
        assert all(abs(v) < 1e-7 for k, v in values.items() if k.startswith("sensory."))
        assert values["wiring.unsigned.serotonin"] == pytest.approx(0.0)

    def test_the_floor_bounds_the_inhibitory_scale(self) -> None:
        """Covers "The floor bounds the inhibitory scale": its unit 0 is the floor."""
        brain = _brain(encoder_box="graded", encoder_inhibition_floor=0.09)
        entries = brain.topology.encoder.entries  # type: ignore[union-attr]
        z = enc.read_encoder(brain)
        z[[e.name for e in entries].index("wiring.inhibitory")] = 0.0
        enc.write_encoder(brain, z)
        assert _values(brain)["wiring.inhibitory"] == pytest.approx(0.09)

    @pytest.mark.parametrize(("box", "floor"), [("graded", 0.75), ("graded", -0.01), ("wide", 1.5)])
    def test_a_floor_outside_the_box_is_refused(self, box: str, floor: float) -> None:
        """Covers "A floor outside the box is refused"."""
        with pytest.raises(ValueError, match="encoder_inhibition_floor"):
            _brain(encoder_box=box, encoder_inhibition_floor=floor)


class TestCounts:
    @pytest.mark.parametrize(
        ("overrides", "count"),
        [
            ({}, 132),
            ({"encoder_speed_noise": "ou"}, 133),
            ({"encoder_pool_centres": True}, 136),
            ({"encoder_pool_centres": True, "encoder_speed_noise": "ou"}, 137),
            (PROCEDURE, 137),
        ],
    )
    def test_the_counts_with_the_new_options(
        self,
        overrides: dict[str, object],
        count: int,
    ) -> None:
        """Covers "The counts with the new options": the centres last, the timescale before them."""
        entries = _brain(**overrides).topology.encoder.entries  # type: ignore[union-attr]
        assert len(entries) == count
        if overrides.get("encoder_pool_centres"):
            assert [e.name for e in entries[-4:]] == [f"readout.centre.{p}" for p in enc.POOLS]

    def test_a_procedure_vector_round_trips(self) -> None:
        """A vector with every procedure option on reads back as written."""
        brain = _brain(**PROCEDURE)
        z = _vector(brain)
        enc.write_encoder(brain, z)
        np.testing.assert_allclose(enc.read_encoder(brain), z, atol=1e-6)


class TestObserver:
    def test_the_observer_sees_each_step_and_changes_nothing(self) -> None:
        """Covers "The observer sees each step and changes nothing"."""
        brain = _brain(**PROCEDURE)
        z = _vector(brain)
        enc.write_encoder(brain, z)
        quiet = _means(brain)
        seen: list[tuple[torch.Tensor, torch.Tensor]] = []
        brain.topology.forward_observer = lambda mean, h: seen.append((mean, h))
        watched = _means(brain)
        brain.topology.forward_observer = None
        np.testing.assert_array_equal(quiet, watched)
        assert len(seen) == len(watched)
        assert not seen[0][0].requires_grad
        assert seen[0][1].shape[-1] == brain.topology.n_neurons

    def test_an_observer_that_edits_its_arguments_changes_nothing(self) -> None:
        """The observer receives copies: editing them in place leaves the forward's outputs."""
        brain = _brain(**PROCEDURE)
        enc.write_encoder(brain, _vector(brain))
        quiet = _means(brain)

        def vandal(mean: torch.Tensor, h: torch.Tensor) -> None:
            mean.add_(100.0)
            h.zero_()

        brain.topology.forward_observer = vandal
        edited = _means(brain)
        brain.topology.forward_observer = None
        np.testing.assert_array_equal(quiet, edited)


class TestCheckpoints:
    @staticmethod
    def _saved(brain: ConnectomePPOBrain) -> dict[str, WeightComponent]:
        saved = brain.get_weight_components(components={"topology", "training_state"})
        return {k: WeightComponent(name=k, state=dict(v.state)) for k, v in saved.items()}

    def test_matching_weights_load(self) -> None:
        """Weights saved under the procedure's options load into a brain with the same options."""
        brain = _brain(**PROCEDURE)
        z = _vector(brain)
        enc.write_encoder(brain, z)
        other = _brain(**PROCEDURE)
        other.load_weight_components(self._saved(brain))
        np.testing.assert_allclose(enc.read_encoder(other), z, atol=1e-6)

    def test_another_readout_mode_is_refused(self) -> None:
        """Same shapes, another readout: a soft-capped brain refuses linear weights and back."""
        capped, linear = _brain(encoder_readout="soft_cap"), _brain()
        with pytest.raises(ValueError, match="encoder_readout"):
            capped.load_weight_components(self._saved(linear))
        with pytest.raises(ValueError, match="encoder_readout"):
            linear.load_weight_components(self._saved(capped))

    def test_other_cited_signs_are_refused(self) -> None:
        """Same per-ligand shapes, other cited signs: the load is refused either way."""
        signed, unsigned = _brain(modulator_receptor_signs=["pdfr-1", "ser-5"]), _brain()
        assert len(signed.topology.encoder.entries) == len(unsigned.topology.encoder.entries)  # type: ignore[union-attr]
        with pytest.raises(ValueError, match="modulator_receptor_signs"):
            signed.load_weight_components(self._saved(unsigned))
        with pytest.raises(ValueError, match="modulator_receptor_signs"):
            unsigned.load_weight_components(self._saved(signed))
        with pytest.raises(ValueError, match="modulator_receptor_signs"):
            _brain(modulator_receptor_signs=["pdfr-1"]).load_weight_components(
                self._saved(signed),
            )


class TestReceptorSigns:
    def test_empty_is_byte_identical(self) -> None:
        """Covers "Empty is byte-identical"."""
        names = sorted(_brain().topology.neuron_names)
        assert layer.load_data_layer(names) == layer.load_data_layer(names, receptor_signs=())

    def test_pdfr1_and_ser5_fix_their_couples(self) -> None:
        """Covers "PDFR-1 and SER-5 fix their couples"."""
        names = sorted(_brain().topology.neuron_names)
        plain = layer.load_data_layer(names)
        signed = layer.load_data_layer(names, receptor_signs=("pdfr-1", "ser-5"))
        changed = {
            b.name: (b.sign, b.sign_source)
            for a, b in zip(plain.rows, signed.rows, strict=True)
            if a != b
        }
        assert changed == {
            "pdf-1:pdfr-1": (1, layer.CITED_SIGN_SOURCE),
            "pdf-2:pdfr-1": (1, layer.CITED_SIGN_SOURCE),
            "serotonin:ser-5": (1, layer.CITED_SIGN_SOURCE),
        }

    def test_ser5_keeps_its_own_magnitude(self) -> None:
        """Covers "ser-5 keeps its own magnitude": its own group, the count unchanged at 132."""
        entries = _brain(modulator_receptor_signs=["ser-5"]).topology.encoder.entries  # type: ignore[union-attr]
        names = [e.name for e in entries]
        assert "modulator.fixed.serotonin.ser-5" in names
        assert "modulator.fixed.serotonin" in names
        assert "modulator.fitted.serotonin" not in names
        assert len(entries) == 132

    def test_the_sign_travels_with_its_couple_under_the_null(self) -> None:
        """Covers "The sign travels with its couple under the null"."""
        wild = _brain(modulator_receptor_signs=["pdfr-1"])
        null = _brain(
            modulator_receptor_signs=["pdfr-1"],
            modulator_layer_null="receptor_placement",
            modulator_layer_null_seed=7,
        )
        rows = {r.name: r for r in null.topology.modulator_spec.rows}
        for name in ("pdf-1:pdfr-1", "pdf-2:pdfr-1"):
            assert rows[name].sign == 1
            assert rows[name].sign_source == layer.CITED_SIGN_SOURCE
        assert [e.name for e in null.topology.encoder.entries] == [  # type: ignore[union-attr]
            e.name
            for e in wild.topology.encoder.entries  # type: ignore[union-attr]
        ]

    @pytest.mark.parametrize("receptor", ["not-a-receptor", "seb-3"])
    def test_an_unresolved_receptor_is_refused(self, receptor: str) -> None:
        """Covers "An unresolved receptor is refused": unknown, or with a null sign."""
        names = sorted(_brain().topology.neuron_names)
        with pytest.raises(ValueError, match=receptor):
            layer.load_data_layer(names, receptor_signs=(receptor,))

    def test_every_signed_entry_is_checked(self) -> None:
        """Covers "Every signed entry is checked": a sign only with a checked quote."""
        table = json.loads(layer.RECEPTOR_COUPLINGS_PATH.read_text())["receptors"]
        for gene, entry in table.items():
            if entry["sign"] is not None:
                assert entry["verified"], gene
                assert entry["verified"]["date"], gene
                assert entry["quote"], gene
