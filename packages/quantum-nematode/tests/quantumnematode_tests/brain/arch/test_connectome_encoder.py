"""C.2's compact encoder on the connectome brain.

Covers the connectome-encoder spec's scenarios "The encoder is off", "A 5-s modulator step is
refused", "A null is encoded with the wild type's parameters", "Two seeds, one vector", "Lateral
drives the two sides oppositely", "Food contact reaches only the dopaminergic mechanosensors", "A
vector round-trips", "A reused brain equals a fresh one", "Per-ligand tying shares a ligand's
strengths", "The batched controller sees a write", "The monoamine timescale changes only monoamines"
and "tax-4 keeps the non-tax-4 classes", and the encoder's refusals. Also the connectome-ppo-brain
scenario "The encoded brain changes only on the corrected edges".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch import connectome_ppo as cp
from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomePPOBrainConfig
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.weights import WeightComponent
from quantumnematode.connectome import encoder as enc
from quantumnematode.connectome.loader import load_emmons_2024_hermaphrodite
from quantumnematode.connectome.signs import per_connection_signs
from quantumnematode.evaluation.batched import BatchedEvaluator
from quantumnematode.evaluation.controllers import BatchedConnectome
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
}
NULL_WIRINGS = (
    "rewired_degree_preserving",
    "rewired_chemical_only",
    "rewired_gap_junctions_held",
    "rewired_boundary_held",
    "rewired_gap_junctions_only",
    "rewired_gap_outliers_held",
    "rewired_gap_outliers_scattered",
)


def _config(**overrides: object) -> ConnectomePPOBrainConfig:
    container = load_simulation_config(str(_ASSAY)).brain
    assert container is not None
    assert isinstance(container.config, ConnectomePPOBrainConfig)
    return container.config.model_copy(update={"seed": 11, **ENCODED, **overrides})


def _brain(**overrides: object) -> ConnectomePPOBrain:
    return ConnectomePPOBrain(config=_config(**overrides), device=DeviceType.CPU)


def _vector(brain: ConnectomePPOBrain, seed: int = 3) -> np.ndarray:
    n = len(brain.topology.encoder.entries)  # type: ignore[union-attr]
    return np.random.default_rng(seed).uniform(0.15, 0.85, size=n)


def _params(step: int, *, contact: float = 0.4) -> BrainParams:
    return BrainParams(
        food_concentration=0.1 * (step % 5),
        food_lateral_gradient=0.05 * (step % 3) - 0.05,
        food_dconcentration_dt=0.01 * step,
        food_contact=contact,
        satiety=100.0 + 10.0 * (step % 4),
        max_satiety=300.0,
        food_ingested=0.3 if step % 4 == 0 else None,
    )


def _means(brain: ConnectomePPOBrain, steps: int = 6) -> np.ndarray:
    brain.prepare_episode()
    out = []
    for t in range(steps):
        actions = brain.run_brain(_params(t), top_only=False, top_randomize=False)
        out.append(actions[0].continuous_mean)
    return np.asarray(out)


def _injection(brain: ConnectomePPOBrain, features: list[float]) -> np.ndarray:
    with torch.no_grad():
        return brain.topology._sensor_current(
            torch.tensor(features, dtype=torch.float32),
            None,
            None,
            None,
            None,
        ).numpy()


class TestTheDefault:
    def test_the_encoder_is_off_by_default(self) -> None:
        """Without the encoder nothing is attached and the learnable set is the pre-option one."""
        container = load_simulation_config(str(_ASSAY)).brain
        assert container is not None
        brain = ConnectomePPOBrain(
            config=container.config.model_copy(update={"seed": 11}),  # type: ignore[union-attr]
            device=DeviceType.CPU,
        )
        topology = brain.topology
        assert topology.encoder is None
        assert not hasattr(topology, "readout_bias")
        assert brain._n_food_features == 3
        assert topology.learnable_parameters[:3] == [
            topology.w_chem,
            topology.food_gains,
            topology.readout,
        ]
        assert not any(k.startswith("encoder.") for k in topology.state_dict())
        components = brain.get_weight_components(components={"training_state"})
        assert "encoder" not in components["training_state"].state

    def test_the_tying_needs_the_encoder(self) -> None:
        with pytest.raises(ValueError, match="encoder_modulator_tying"):
            ConnectomePPOBrainConfig(encoder_modulator_tying="per_ligand")


class TestRefusals:
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("synapse_signs", "atlas"),
            ("gap_normalisation", "degree"),
            ("gap_coupling", "ohmic"),
            ("modulator_satiety_link", False),
            ("modulator_layer", "hand_placed"),
            ("weight_init", "count_scaled"),
            ("chemical_weight_scale", 0.5),
            ("gap_weight_scale", 0.5),
            ("readout_width", "per_neuron"),
            ("modulator_timescales_s", {"dopamine": 30.0}),
            ("enable_thermotaxis_projection", True),
            ("initial_log_std", -1.0),
        ],
    )
    def test_a_setting_the_encoder_fixes_is_refused(self, key: str, value: object) -> None:
        with pytest.raises(ValueError, match="encoder='compact' fixes"):
            _brain(**{key: value})

    def test_a_five_second_modulator_step_is_refused(self) -> None:
        """Covers "A 5-s modulator step is refused": the brain refuses the unconverted default."""
        with pytest.raises(ValueError, match="modulator_step_seconds"):
            _brain(modulator_step_seconds=5.0)

    def test_a_step_the_loader_does_not_bring_to_the_worm_is_refused(self, tmp_path: Path) -> None:
        """A modulator step written at the worm's own step is converted below it, and refused."""
        text = _ASSAY.read_text().replace(
            "    action_space: primitives\n",
            "    action_space: primitives\n"
            "    modulator_layer: data\n"
            "    modulator_satiety_link: true\n"
            "    synapse_signs: per_connection\n"
            "    gap_normalisation: strength\n"
            "    encoder: compact\n"
            "    modulator_step_seconds: 1.0\n",
        )
        path = tmp_path / "encoded.yml"
        path.write_text(text)
        with pytest.raises(ValueError, match="come to the 1-s step"):
            load_simulation_config(str(path))
        path.write_text(text.replace("    modulator_step_seconds: 1.0\n", ""))
        loaded = load_simulation_config(str(path))
        assert loaded.brain.config.modulator_step_seconds == pytest.approx(1.0)  # type: ignore[union-attr]

    def test_an_unknown_tying_is_refused(self) -> None:
        """A tying copied in past the field's validation is refused at construction."""
        with pytest.raises(ValueError, match="Unsupported encoder_modulator_tying"):
            _brain(encoder_modulator_tying="per_receptor")

    def test_a_modulator_step_off_a_one_second_reference_step_is_refused(
        self,
        tmp_path: Path,
    ) -> None:
        """With the reference step 1 s nothing is converted, and 0.5 s is refused at load."""
        text = (
            _ASSAY.read_text()
            .replace("    max_step_mm: 1.0\n", "    max_step_mm: 0.2\n")
            .replace("max_steps: 720\n", "max_steps: 3600\n")
            .replace(
                "    action_space: primitives\n",
                "    action_space: primitives\n"
                "    modulator_layer: data\n"
                "    modulator_satiety_link: true\n"
                "    synapse_signs: per_connection\n"
                "    gap_normalisation: strength\n"
                "    encoder: compact\n"
                "    modulator_step_seconds: 0.5\n",
            )
        )
        path = tmp_path / "reference_step.yml"
        path.write_text(text)
        with pytest.raises(ValueError, match=re.escape("come to the 1-s step; it comes to 0.5 s")):
            load_simulation_config(str(path))
        path.write_text(text.replace("modulator_step_seconds: 0.5", "modulator_step_seconds: 1.0"))
        loaded = load_simulation_config(str(path))
        assert loaded.brain.config.modulator_step_seconds == 1.0  # type: ignore[union-attr]

    def test_the_validator_refuses_too(self) -> None:
        with pytest.raises(ValueError, match="encoder='compact' fixes"):
            ConnectomePPOBrainConfig(encoder="compact")


class TestTheParameterTable:
    def test_per_couple_has_196_parameters(self) -> None:
        brain = _brain()
        names = enc.encoder_parameters(brain.topology.modulator_spec)
        assert len(names) == 196
        assert len(set(names)) == 196
        blocks = [e.block for e in brain.topology.encoder.entries]  # type: ignore[union-attr]
        assert {b: blocks.count(b) for b in dict.fromkeys(blocks)} == {
            "wiring": 7,
            "sensory": 36,
            "readout": 25,
            "noise": 5,
            "modulator": 121,
            "timescale": 2,
        }

    def test_the_policy_learns_exactly_the_encoder(self) -> None:
        """Every tensor the optimiser holds is the encoder's; the timescales are held."""
        brain = _brain()
        topology = brain.topology
        encoder = topology.encoder
        assert encoder is not None
        learnable = topology.learnable_parameters
        assert sum(p.numel() for p in learnable) == 196 - 2
        held = {id(topology.w_chem), id(topology.food_gains)}
        assert not held & {id(p) for p in learnable}
        assert not topology.w_chem.requires_grad
        assert not topology.food_gains.requires_grad
        optimised = {id(p) for g in brain.optimizer.param_groups for p in g["params"]}
        critic = {id(p) for p in brain.critic.parameters()}
        assert optimised == {id(p) for p in learnable} | critic

    def test_fixed_sign_strengths_are_on_a_log_scale(self) -> None:
        """A fixed-sign strength centres on its range's geometric mean; a signed one on 0."""
        brain = _brain()
        encoder = brain.topology.encoder
        assert encoder is not None
        fixed = [e for e in encoder.entries if e.block == "modulator" and e.low > 0]
        signed = [e for e in encoder.entries if e.block == "modulator" and e.low < 0]
        assert len(fixed) == 17
        assert len(signed) == 104
        for entry in fixed:
            assert entry.kind == "log"
            assert entry.value(0.5) == pytest.approx(np.sqrt(entry.low * entry.high))
        assert all(entry.value(0.5) == 0.0 for entry in signed)

    def test_the_box_centre_is_written_at_construction(self) -> None:
        brain = _brain()
        np.testing.assert_allclose(enc.read_encoder(brain), 0.5, atol=1e-6)
        assert brain.topology.encoder.timescales_s.tolist() == pytest.approx(  # type: ignore[union-attr]
            [np.sqrt(10.0 * 3000.0)] * 2,
        )


class TestTheWiring:
    def test_the_encoded_brain_changes_only_on_the_corrected_edges(self) -> None:
        """Only the López-Cruz edges move: to the inhibitory matrix, at their magnitude."""
        full = _brain().topology
        earlier = _brain(sign_override_exclude=["lopezcruz2019"]).topology
        a, b = full.encoder, earlier.encoder
        assert a is not None
        assert b is not None
        assert torch.equal(a.chem_grouped, b.chem_grouped)
        moved = (a.chem_positive != b.chem_positive) | (a.chem_negative != b.chem_negative)
        names = full.neuron_names
        edges = {(names[i], names[j]) for i, j in moved.nonzero().tolist()}
        assert len(edges) == 16
        assert {post for _, post in edges} == {"AIAL", "AIAR", "ADEL", "ADER"}
        i, j = moved.nonzero().unbind(1)
        assert torch.all(a.chem_positive[i, j] == 0)
        assert torch.all(b.chem_negative[i, j] == 0)
        # The inhibitory matrix holds its counts negated, at the same normalised magnitude.
        assert torch.equal(a.chem_negative[i, j], -b.chem_positive[i, j])
        assert torch.all(b.chem_positive[i, j] > 0)

    def test_weights_follow_the_counts_signs_and_scales(self) -> None:
        brain = _brain()
        z = _vector(brain)
        enc.write_encoder(brain, z)
        entries = brain.topology.encoder.entries  # type: ignore[union-attr]
        value = {e.name: e.value(u) for e, u in zip(entries, z, strict=True)}
        connectome = load_emmons_2024_hermaphrodite()
        signs = {edge: entry.sign for edge, entry in per_connection_signs(connectome).items()}
        idx = brain.topology._idx
        norm: dict[str, float] = {}
        for syn in connectome.chemical_synapses:
            norm[syn.post] = norm.get(syn.post, 0.0) + float(syn.weight) ** 2
        with torch.no_grad():
            chem = brain.topology.encoder.chemical().numpy()  # type: ignore[union-attr]
        groups = {"DA": "dopamine", "5-HT": "serotonin", "octopamine": "octopamine", None: "none"}
        seen_unsigned = set()
        for syn in connectome.chemical_synapses:
            base = float(syn.weight) / np.sqrt(norm[syn.post])
            sign = signs[syn.pre, syn.post]
            if sign > 0:
                expected = value["wiring.excitatory"] * base
            elif sign < 0:
                expected = -value["wiring.inhibitory"] * base
            else:
                group = groups[connectome.neurons[syn.pre].neurotransmitter]
                seen_unsigned.add(group)
                expected = value[f"wiring.unsigned.{group}"] * base
            assert chem[idx[syn.pre], idx[syn.post]] == pytest.approx(expected, rel=1e-5, abs=1e-7)
        assert seen_unsigned == set(enc.TRANSMITTER_GROUPS)
        assert np.count_nonzero(chem) == len(connectome.chemical_synapses)

    def test_gaps_are_strength_normalised_under_one_scale(self) -> None:
        brain = _brain()
        z = _vector(brain)
        enc.write_encoder(brain, z)
        scale = brain.topology.encoder.entries[2].value(z[2])  # type: ignore[union-attr]
        connectome = load_emmons_2024_hermaphrodite()
        strength: dict[str, float] = {}
        for gj in connectome.gap_junctions:
            for name in (gj.neuron_a, gj.neuron_b):
                strength[name] = strength.get(name, 0.0) + float(gj.weight)
        idx = brain.topology._idx
        with torch.no_grad():
            gap = brain.topology.gap_matrix().numpy()
        for gj in connectome.gap_junctions:
            expected = scale * gj.weight / np.sqrt(strength[gj.neuron_a] * strength[gj.neuron_b])
            assert gap[idx[gj.neuron_a], idx[gj.neuron_b]] == pytest.approx(expected, rel=1e-5)
            assert gap[idx[gj.neuron_b], idx[gj.neuron_a]] == pytest.approx(expected, rel=1e-5)

    def test_two_seeds_one_vector(self) -> None:
        """Covers "Two seeds, one vector": equal weights and equal actions."""
        first, second = _brain(seed=11), _brain(seed=12)
        z = _vector(first)
        for brain in (first, second):
            enc.write_encoder(brain, z)
        with torch.no_grad():
            for read in (
                lambda b: b.topology.encoder.chemical(),
                lambda b: b.topology.gap_matrix(),
                lambda b: b.topology.encoder.injection(),
            ):
                assert torch.equal(read(first), read(second))
        np.testing.assert_array_equal(_means(first), _means(second))

    def test_the_composition_takes_a_candidate_axis(self) -> None:
        brain = _brain()
        encoder = brain.topology.encoder
        assert encoder is not None
        scales = torch.tensor([[1.0, 2.0], [0.5, 0.25], [3.0, 0.0]])
        gains = torch.tensor([[0.1, -0.2, 0.3, -0.4], [0.0, 1.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]])
        batch = enc.compose_chemical(
            scales,
            gains,
            encoder.chem_positive,
            encoder.chem_negative,
            encoder.chem_grouped,
        )
        assert batch.shape == (3, encoder.n_neurons, encoder.n_neurons)
        for k in range(3):
            one = enc.compose_chemical(
                scales[k],
                gains[k],
                encoder.chem_positive,
                encoder.chem_negative,
                encoder.chem_grouped,
            )
            torch.testing.assert_close(batch[k], one)
        sensory = torch.randn(3, encoder.n_sensory, generator=torch.Generator().manual_seed(1))
        maps = enc.compose_injection(
            sensory.index_select(-1, encoder.sensory_gain_index),
            encoder.sensory_features,
            encoder.sensory_neurons,
            encoder.sensory_signs,
            encoder.n_neurons,
        )
        assert maps.shape == (3, len(enc.SENSORY_FEATURES), encoder.n_neurons)


class TestTheSensoryInputs:
    def test_lateral_drives_the_two_sides_oppositely(self) -> None:
        brain = _brain()
        enc.write_encoder(brain, _vector(brain))
        current = _injection(brain, [0.0, 0.7, 0.0, 0.0])
        idx = brain.topology._idx
        for cls in enc.AMPHID_CLASSES:
            right, left = current[idx[f"{cls}R"]], current[idx[f"{cls}L"]]
            assert right != 0.0
            assert right == pytest.approx(-left)
        targets = {idx[f"{cls}{side}"] for cls in enc.AMPHID_CLASSES for side in "LR"}
        assert set(np.flatnonzero(current)) == targets

    def test_concentration_and_derivative_are_shared_by_the_pair(self) -> None:
        brain = _brain()
        enc.write_encoder(brain, _vector(brain))
        idx = brain.topology._idx
        for features in ([0.6, 0.0, 0.0, 0.0], [0.0, 0.0, -0.4, 0.0]):
            current = _injection(brain, features)
            for cls in enc.AMPHID_CLASSES:
                assert current[idx[f"{cls}R"]] == pytest.approx(current[idx[f"{cls}L"]])

    def test_food_contact_reaches_only_the_dopaminergic_mechanosensors(self) -> None:
        brain = _brain()
        z = _vector(brain)
        enc.write_encoder(brain, z)
        current = _injection(brain, [0.0, 0.0, 0.0, 0.8])
        idx = brain.topology._idx
        contact = {name for members in enc.CONTACT_CLASSES.values() for name in members}
        assert {brain.topology.neuron_names[i] for i in np.flatnonzero(current)} == contact
        entries = brain.topology.encoder.entries  # type: ignore[union-attr]
        for cls, members in enc.CONTACT_CLASSES.items():
            k = next(i for i, e in enumerate(entries) if e.name == f"sensory.{cls}.contact")
            for name in members:
                assert current[idx[name]] == pytest.approx(0.8 * entries[k].value(z[k]), rel=1e-6)

    def test_the_brain_reads_food_contact(self) -> None:
        brain = _brain()
        enc.write_encoder(brain, _vector(brain))
        state = brain.preprocess(_params(2, contact=0.6))
        assert state[0] == pytest.approx(0.2)
        assert state[3] == pytest.approx(0.6)
        assert brain.preprocess(_params(2, contact=1.7))[3] == 1.0
        assert brain.preprocess(BrainParams(food_concentration=0.1))[3] == 0.0


class TestTheVector:
    @pytest.mark.parametrize("tying", ["per_couple", "per_ligand"])
    def test_a_vector_round_trips(self, tying: str) -> None:
        brain = _brain(encoder_modulator_tying=tying)
        z = _vector(brain)
        enc.write_encoder(brain, z)
        np.testing.assert_allclose(enc.read_encoder(brain), z, atol=1e-6)

    def test_a_vector_of_the_wrong_length_is_refused(self) -> None:
        brain = _brain()
        with pytest.raises(ValueError, match="196 parameters"):
            enc.write_encoder(brain, np.full(132, 0.5))

    def test_a_reused_brain_equals_a_fresh_one(self) -> None:
        reused = _brain()
        for seed in (4, 5, 6):
            enc.write_encoder(reused, _vector(reused, seed))
            _means(reused)
        fresh = _brain(seed=99)
        enc.write_encoder(fresh, _vector(fresh, 6))
        np.testing.assert_array_equal(_means(reused), _means(fresh))
        config = load_simulation_config(str(_ASSAY)).model_copy(update={"max_steps": 120})
        seeds = [53_100_000 + i for i in range(4)]
        runs = [
            BatchedEvaluator(config, BatchedConnectome(b), seeds).run() for b in (reused, fresh)
        ]
        for name in ("x", "y", "heading", "primitive"):
            np.testing.assert_array_equal(getattr(runs[0], name), getattr(runs[1], name))

    def test_the_batched_controller_sees_a_write(self) -> None:
        brain = _brain()
        controller = BatchedConnectome(brain)
        enc.write_encoder(brain, _vector(brain, 8))
        fresh = _brain(seed=21)
        enc.write_encoder(fresh, _vector(fresh, 8))
        config = load_simulation_config(str(_ASSAY)).model_copy(update={"max_steps": 120})
        seeds = [53_100_010 + i for i in range(4)]
        written = BatchedEvaluator(config, controller, seeds).run()
        built = BatchedEvaluator(config, BatchedConnectome(fresh), seeds).run()
        for name in ("x", "y", "heading", "primitive"):
            np.testing.assert_array_equal(getattr(written, name), getattr(built, name))

    def test_the_weights_persist_with_their_decay(self) -> None:
        brain = _brain()
        z = _vector(brain)
        enc.write_encoder(brain, z)
        saved = brain.get_weight_components(components={"topology", "training_state"})
        assert saved["training_state"].state["encoder"] == "compact"
        other = _brain(seed=12)
        other.load_weight_components(
            {k: WeightComponent(name=k, state=dict(v.state)) for k, v in saved.items()},
        )
        np.testing.assert_allclose(enc.read_encoder(other), z, atol=1e-6)
        np.testing.assert_array_equal(other._modulator_decay, brain._modulator_decay)


class TestTying:
    def test_per_ligand_tying_shares_a_ligands_strengths(self) -> None:
        """Covers "Per-ligand tying shares a ligand's strengths": 132 parameters, one per group."""
        brain = _brain(encoder_modulator_tying="per_ligand")
        topology = brain.topology
        spec = topology.modulator_spec
        assert len(enc.encoder_parameters(spec, "per_ligand")) == 132
        assert sum(p.numel() for p in topology.learnable_parameters) == 130
        enc.write_encoder(brain, _vector(brain))
        with torch.no_grad():
            strengths = topology.modulator_layer_strengths().numpy()
        by_group: dict[tuple[str, bool], set[float]] = {}
        for i, row in enumerate(spec.rows):
            key = (enc._tied_name(spec, i), row.sign is not None)
            by_group.setdefault(key, set()).add(float(strengths[i]))
        assert len(by_group) == 57
        assert all(len(values) == 1 for values in by_group.values())
        fitted = [k for k in by_group if k[0].startswith("fitted.")]
        assert len(fitted) == 52
        assert not topology.modulator_layer_w.requires_grad

    def test_per_couple_strengths_land_on_their_rows(self) -> None:
        brain = _brain()
        z = _vector(brain)
        enc.write_encoder(brain, z)
        entries = brain.topology.encoder.entries  # type: ignore[union-attr]
        start = next(i for i, e in enumerate(entries) if e.block == "modulator")
        with torch.no_grad():
            strengths = brain.topology.modulator_layer_strengths().numpy()
        expected = [e.value(u) for e, u in zip(entries[start:-2], z[start:-2], strict=True)]
        np.testing.assert_allclose(strengths, expected, rtol=1e-5)


class TestTimescales:
    def test_the_monoamine_timescale_changes_only_monoamines(self) -> None:
        brain = _brain()
        z = _vector(brain)
        enc.write_encoder(brain, z)
        before = np.asarray(brain._modulator_decay).copy()
        z[-2] = 0.05
        enc.write_encoder(brain, z)
        after = np.asarray(brain._modulator_decay)
        ligands = brain.topology.modulator_spec.ligands
        changed = {ligands[k] for k in np.flatnonzero(before != after)}
        assert changed == set(enc.MONOAMINES)
        tau = brain.topology.encoder.entries[-2].value(0.05)  # type: ignore[union-attr]
        k = ligands.index("serotonin")
        assert after[k] == pytest.approx(np.exp(-1.0 / tau), rel=1e-6)


class TestMutants:
    def test_tax4_keeps_the_non_tax4_classes(self) -> None:
        brain = _brain()
        spec = brain.topology.modulator_spec
        z = _vector(brain)
        mutant = enc.tax4_vector(spec, z)
        enc.write_encoder(brain, mutant)
        entries = brain.topology.encoder.entries  # type: ignore[union-attr]
        sensory = brain.topology.encoder.sensory.detach().numpy()  # type: ignore[union-attr]
        names = [e.name for e in entries if e.block == "sensory"]
        zeroed = [n for n, g in zip(names, sensory, strict=True) if g == 0.0]
        assert len(zeroed) == 21
        assert {n.split(".")[1] for n in zeroed} == set(enc.TAX4_CLASSES)
        kept = [i for i, e in enumerate(entries) if e.name.split(".")[1] not in enc.TAX4_CLASSES]
        np.testing.assert_array_equal(mutant[kept], z[kept])

    @pytest.mark.parametrize("gene", ["tph-1", "pdfr-1"])
    def test_a_knockout_keeps_the_parameter_table(self, gene: str) -> None:
        wild = _brain()
        mutant = _brain(modulator_knockout_genes=[gene])
        spec = mutant.topology.modulator_spec
        assert enc.encoder_parameters(spec) == enc.encoder_parameters(wild.topology.modulator_spec)
        z = _vector(wild)
        enc.write_encoder(mutant, z)
        np.testing.assert_allclose(enc.read_encoder(mutant), z, atol=1e-6)


def _assert_chemical_weights(
    brain: ConnectomePPOBrain,
    connectome: Any,
    signs: dict[tuple[str, str], int],
    z: np.ndarray,
) -> None:
    """Assert each edge's encoded weight from this connectome's own counts, signs and groups."""
    entries = brain.topology.encoder.entries  # type: ignore[union-attr]
    value = {e.name: e.value(u) for e, u in zip(entries, z, strict=True)}
    norm: dict[str, float] = {}
    for syn in connectome.chemical_synapses:
        norm[syn.post] = norm.get(syn.post, 0.0) + float(syn.weight) ** 2
    groups = {"DA": "dopamine", "5-HT": "serotonin", "octopamine": "octopamine", None: "none"}
    idx = brain.topology._idx
    with torch.no_grad():
        chem = brain.topology.encoder.chemical().numpy()  # type: ignore[union-attr]
    for syn in connectome.chemical_synapses:
        base = float(syn.weight) / np.sqrt(norm[syn.post])
        sign = signs[syn.pre, syn.post]
        if sign > 0:
            expected = value["wiring.excitatory"] * base
        elif sign < 0:
            expected = -value["wiring.inhibitory"] * base
        else:
            group = groups[connectome.neurons[syn.pre].neurotransmitter]
            expected = value[f"wiring.unsigned.{group}"] * base
        assert chem[idx[syn.pre], idx[syn.post]] == pytest.approx(expected, rel=1e-5, abs=1e-7)


class TestNulls:
    @pytest.mark.parametrize("wiring", NULL_WIRINGS)
    def test_a_null_is_encoded_with_the_wild_types_parameters(
        self,
        wiring: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Covers "A null is encoded with the wild type's parameters", and its mutants build.

        Each edge's weight follows the null's own counts and normalisations, its origin's sign, and
        its presynaptic neuron's transmitter group.
        """
        captured: dict[str, Any] = {}
        rewire = cp.rewire_degree_preserving

        def capture(*args: Any, **kwargs: Any) -> Any:
            null_connectome = rewire(*args, **kwargs)
            captured["connectome"], captured["origins"] = null_connectome, kwargs["origins"]
            return null_connectome

        monkeypatch.setattr(cp, "rewire_degree_preserving", capture)
        wild = _brain()
        names = enc.encoder_parameters(wild.topology.modulator_spec)
        z = _vector(wild)
        for genes in ([], ["tph-1"], ["pdfr-1"]):
            null = _brain(wiring=wiring, rewire_seed=5, modulator_knockout_genes=genes)
            assert enc.encoder_parameters(null.topology.modulator_spec) == names
            for vector in (z, enc.tax4_vector(null.topology.modulator_spec, z)):
                enc.write_encoder(null, vector)
                assert np.all(np.isfinite(_means(null, steps=2)))
        encoder = null.topology.encoder
        assert encoder is not None
        placed = (
            (encoder.chem_positive != 0)
            | (encoder.chem_negative != 0)
            | encoder.chem_grouped.any(0)
        )
        assert torch.equal(placed, null.topology.m_chem)
        enc.write_encoder(null, z)
        signs = cp._per_connection_edge_signs(
            load_emmons_2024_hermaphrodite(),
            captured["connectome"],
            captured["origins"],
        )
        _assert_chemical_weights(null, captured["connectome"], signs, z)

    def test_the_boundary_held_null_holds_the_encoded_boundary(self) -> None:
        """Every edge leaving an encoded sensor or entering a pooled motor neuron is kept."""
        connectome = load_emmons_2024_hermaphrodite()
        sensory, motor = cp.boundary_neurons(connectome, encoded=True)
        default_sensory, _ = cp.boundary_neurons(connectome)
        assert {"ASHL", "ASHR", "CEPDL", "PDER"} <= sensory
        assert sensory != default_sensory
        wild = _brain()
        null = _brain(wiring="rewired_boundary_held", rewire_seed=5)
        idx = wild.topology._idx
        rows = [idx[name] for name in sorted(sensory)]
        cols = [idx[name] for name in sorted(motor)]
        wild_mask, null_mask = wild.topology.m_chem, null.topology.m_chem
        assert torch.equal(null_mask[rows], wild_mask[rows])
        assert torch.equal(null_mask[:, cols], wild_mask[:, cols])
        assert not torch.equal(null_mask, wild_mask)

    def test_the_receptor_placement_null_is_encoded(self) -> None:
        wild = _brain()
        names = enc.encoder_parameters(wild.topology.modulator_spec)
        z = _vector(wild)
        for genes in ([], ["tph-1"], ["pdfr-1"]):
            null = _brain(
                modulator_layer_null="receptor_placement",
                modulator_layer_null_seed=4,
                modulator_knockout_genes=genes,
            )
            assert enc.encoder_parameters(null.topology.modulator_spec) == names
            enc.write_encoder(null, enc.tax4_vector(null.topology.modulator_spec, z))
            assert np.all(np.isfinite(_means(null, steps=2)))

    def test_an_unsigned_edge_outside_the_groups_is_refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = cp._per_connection_edge_signs

        def unsign_acetylcholine(*args: Any) -> dict[tuple[str, str], int]:
            signs = original(*args)
            first = next(edge for edge in signs if edge[0] == "AIYL")
            return {**signs, first: 0}

        monkeypatch.setattr(cp, "_per_connection_edge_signs", unsign_acetylcholine)
        with pytest.raises(ValueError, match="in no unsigned group"):
            _brain()


class TestUnderPPO:
    def test_an_update_moves_only_the_encoders_parameters(self) -> None:
        brain = _brain(freeze_updates=False, rollout_buffer_size=8, num_minibatches=2, num_epochs=1)
        enc.write_encoder(brain, _vector(brain))
        w_chem = brain.topology.w_chem.detach().clone()
        timescales = brain.topology.encoder.timescales_s.clone()  # type: ignore[union-attr]
        before = enc.read_encoder(brain)
        brain.prepare_episode()
        for t in range(9):
            brain.run_brain(_params(t), top_only=False, top_randomize=False)
            brain.learn(_params(t), reward=0.1 * (t % 3) - 0.1)
        after = enc.read_encoder(brain)
        assert not np.allclose(before, after)
        assert torch.equal(brain.topology.w_chem, w_chem)
        assert torch.equal(brain.topology.encoder.timescales_s, timescales)  # type: ignore[union-attr]


class TestTheSettlingTanh:
    def test_it_agrees_with_torch_and_does_not_depend_on_the_batch(self) -> None:
        """Within one float32 ulp of torch's tanh, and each row the same alone or in a batch."""
        x = torch.randn(64, 302, generator=torch.Generator().manual_seed(2)) * 3.0
        out = cp._settling_tanh(x)
        torch.testing.assert_close(out, torch.tanh(x), rtol=0.0, atol=1.2e-7)
        for i in (0, 31, 63):
            assert torch.equal(cp._settling_tanh(x[i : i + 1]), out[i : i + 1])
            assert torch.equal(cp._settling_tanh(x[i]), out[i])

    def test_a_gradient_takes_torchs_tanh(self) -> None:
        x = torch.randn(4, 302, generator=torch.Generator().manual_seed(3), requires_grad=True)
        y = cp._settling_tanh(x)
        assert y.requires_grad
        y.sum().backward()
        assert x.grad is not None
        torch.testing.assert_close(x.grad, 1.0 - torch.tanh(x.detach()) ** 2)
