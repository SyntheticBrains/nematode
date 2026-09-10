"""Neurotransmitter identities: the vendored atlas, its normalisation, and the committed table."""

from __future__ import annotations

import hashlib

import pytest
from quantumnematode.connectome import neurotransmitters as nt
from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite
from quantumnematode.connectome.neurons import (
    NEURON_CLASSIFICATION,
    NEURON_CO_TRANSMITTERS,
)
from quantumnematode.connectome.neurotransmitters import (
    ATLAS_NAME_MAP,
    ATLAS_PATH,
    ATLAS_SHA256,
    TRANSMITTER_SIGN,
    normalise_identity,
    read_atlas_transmitters,
    release_identity,
    sign_for,
)

# The coverage this substrate change records. These are the figures the design and the
# registered test are written against, so a silent shift in either the atlas or the sign table
# has to fail here rather than quietly move the experiment's premise.
_EXPECTED_SIGNED_NEURONS = 268
_EXPECTED_SYNAPSES = 3709
_EXPECTED_GROUNDED = 3176
_EXPECTED_EXCITATORY = 2962
_EXPECTED_INHIBITORY = 214


class TestVendoredFile:
    def test_present_and_unmodified(self) -> None:
        assert ATLAS_PATH.is_file(), f"{ATLAS_PATH} missing; run `git lfs pull`"
        digest = hashlib.sha256(ATLAS_PATH.read_bytes()).hexdigest()
        assert digest == ATLAS_SHA256


class TestNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ACh", ("ACh", None)),
            ("*ACh", ("ACh", None)),
            ("ACh - NEW", ("ACh", None)),
            ("*ACh - NEW", ("ACh", None)),
            ("GABA (uptake)", ("GABA", "uptake")),
            ("betaine (uptake) - NEW", ("betaine", "uptake")),
            ("unknown (orphan)", ("unknown", "orphan")),
            ("Unknown (orphan, unc-47 expression)", ("unknown", "orphan, unc-47 expression")),
        ],
    )
    def test_annotation_is_stripped(self, raw: str, expected: tuple[str, str | None]) -> None:
        assert normalise_identity(raw) == expected

    def test_uptake_is_not_release(self) -> None:
        assert release_identity("GABA (uptake)") is None
        assert release_identity("betaine (uptake) - NEW") is None
        assert release_identity("GABA") == "GABA"

    def test_orphans_have_no_release_identity(self) -> None:
        assert release_identity("unknown (orphan)") is None
        assert release_identity("Unknown (orphan, unc-47 expression)") is None


class TestAtlasRead:
    def test_covers_every_canonical_neuron(self) -> None:
        atlas = read_atlas_transmitters()
        assert set(atlas) == set(NEURON_CLASSIFICATION)
        assert len(atlas) == 302

    def test_the_joint_labels_are_mapped(self) -> None:
        atlas = read_atlas_transmitters()
        for joint, canonical in ATLAS_NAME_MAP.items():
            assert joint not in atlas
            assert canonical in atlas


class TestCommittedTable:
    def test_matches_what_the_atlas_says(self) -> None:
        """The committed literals are the atlas's, entry for entry."""
        atlas = read_atlas_transmitters()
        committed = {name: value[1] for name, value in NEURON_CLASSIFICATION.items()}
        assert committed == atlas

    def test_reaches_the_connectome_loader(self) -> None:
        neurons = load_cook_2019_hermaphrodite().neurons
        assert neurons["AVAL"].neurotransmitter == "ACh"
        assert neurons["ADEL"].neurotransmitter == "DA"
        assert sum(1 for n in neurons.values() if n.neurotransmitter) == 280


class TestDerivedSigns:
    def test_sign_table(self) -> None:
        assert sign_for("ACh") == 1
        assert sign_for("Glu") == 1
        assert sign_for("GABA") == -1
        for modulatory in ("DA", "5-HT", "octopamine", "tyramine"):
            assert sign_for(modulatory) is None
        assert sign_for(None) is None
        assert set(TRANSMITTER_SIGN) == {"ACh", "Glu", "GABA"}

    def test_coverage_is_the_documented_one(self) -> None:
        atlas = read_atlas_transmitters()
        signed = {n for n, t in atlas.items() if sign_for(t)}
        assert len(signed) == _EXPECTED_SIGNED_NEURONS
        synapses = load_cook_2019_hermaphrodite().chemical_synapses
        assert len(synapses) == _EXPECTED_SYNAPSES
        grounded = [s for s in synapses if sign_for(atlas.get(s.pre))]
        assert len(grounded) == _EXPECTED_GROUNDED
        assert sum(1 for s in grounded if sign_for(atlas[s.pre]) == 1) == _EXPECTED_EXCITATORY
        assert sum(1 for s in grounded if sign_for(atlas[s.pre]) == -1) == _EXPECTED_INHIBITORY


class TestCoTransmitterIdentities:
    """The atlas writes identities across three columns; the first alone hides co-transmitters."""

    def test_the_known_co_transmitters_carry_both_identities(self) -> None:
        identities = nt.read_atlas_identities()
        assert identities["ADFL"] == ("ACh", "5-HT")
        assert identities["HSNL"] == ("ACh", "5-HT")
        assert identities["RIML"] == ("Glu", "tyramine")
        assert identities["RICL"] == ("octopamine", "Glu")

    def test_uptake_only_neurons_carry_no_amine(self) -> None:
        identities = nt.read_atlas_identities()
        # AIM and RIH are annotated "5-HT (uptake)": they recover the amine without making it.
        assert identities["AIML"] == ("Glu",)
        assert identities["RIH"] == ("ACh",)

    def test_synthesis_with_uptake_is_release(self) -> None:
        # ADF's "5-HT (synthesis + uptake)" is a releasing neuron, unlike a bare "(uptake)".
        assert nt.release_identity("5-HT (synthesis + uptake)") == "5-HT"
        assert nt.release_identity("5-HT (uptake)") is None

    def test_the_atlas_own_hedges_yield_no_release_identity(self) -> None:
        identities = nt.read_atlas_identities()
        for name in ("I5", "MI", "VC4", "VC5"):
            assert all(value not in nt.AMINERGIC for value in identities[name]), name
        assert identities["PVWL"] == ()
        assert nt.release_identity("bas-1-depen unknown monoamine?") is None

    def test_the_aminergic_set_is_the_registered_eighteen(self) -> None:
        assert nt.aminergic_neurons(nt.read_atlas_identities()) == {
            "ADEL",
            "ADER",
            "CEPDL",
            "CEPDR",
            "CEPVL",
            "CEPVR",
            "PDEL",
            "PDER",
            "NSML",
            "NSMR",
            "ADFL",
            "ADFR",
            "HSNL",
            "HSNR",
            "RICL",
            "RICR",
            "RIML",
            "RIMR",
        }

    def test_the_committed_co_transmitter_table_matches_the_atlas(self) -> None:
        identities = nt.read_atlas_identities()
        expected = {name: rest[1:] for name, rest in identities.items() if len(rest) > 1}
        assert expected == NEURON_CO_TRANSMITTERS

    def test_an_excluded_first_column_yields_no_primary_identity(self) -> None:
        # The primary comes from the FIRST identity column itself, not from the compacted list:
        # if that column is excluded, taking identities[0] would promote a co-transmitter into
        # the sign table and silently change a synapse's sign.
        assert nt.release_identity("5-HT (uptake)") is None
        assert nt.release_identity("ACh") == "ACh"
        cells = ("5-HT (uptake)", "ACh")
        primary = nt.release_identity(cells[0]) if cells[0] else None
        compacted = tuple(i for i in (nt.release_identity(c) for c in cells if c) if i)
        assert primary is None
        assert compacted == ("ACh",)  # what the compacted list would have promoted

    def test_sign_grounding_is_unchanged_by_the_extra_columns(self) -> None:
        # The primary identity is what signs come from, and it is the first column's.
        primary = nt.read_atlas_transmitters()
        identities = nt.read_atlas_identities()
        for name, carried in identities.items():
            assert primary[name] == (carried[0] if carried else None)
        for name in ("ADFL", "RIML", "SMDDL"):
            assert nt.sign_for(primary[name]) == nt.sign_for(NEURON_CLASSIFICATION[name][1])


class TestTheInstructivePathway:
    def test_the_reach_is_the_registered_one(self) -> None:
        connectome = load_cook_2019_hermaphrodite()
        instructed = nt.instructed_neurons(connectome)
        assert len(instructed) == 169
        synapses = connectome.chemical_synapses
        reached = sum(1 for synapse in synapses if synapse.post in instructed)
        assert reached == 2636
        assert reached / len(synapses) == pytest.approx(0.711, abs=0.001)

    def test_a_rewired_connectome_derives_its_own_pathway(self) -> None:
        import numpy as np
        from quantumnematode.connectome.rewiring import rewire_degree_preserving

        connectome = load_cook_2019_hermaphrodite()
        rewired = rewire_degree_preserving(connectome, np.random.default_rng(7))
        assert nt.instructed_neurons(rewired) != nt.instructed_neurons(connectome)
