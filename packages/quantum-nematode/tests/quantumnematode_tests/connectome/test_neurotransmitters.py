"""Neurotransmitter identities: the vendored atlas, its normalisation, and the committed table."""

from __future__ import annotations

import hashlib

import pytest
from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite
from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION
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
