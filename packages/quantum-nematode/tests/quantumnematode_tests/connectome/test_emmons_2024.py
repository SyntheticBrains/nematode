"""The Emmons 2024 release of the Cook 2019 matrices: the file, its wiring, and its muscles.

Covers two connectome-substrate requirements. "The Emmons 2024 release of the Cook 2019
connectome is vendored and loadable": the vendored file is the one recorded, the chemical wiring is
the 2019 file's, and the gap junctions differ only by the 2023 addition. "Chemical synapses onto the
body wall muscles are loadable": every body wall muscle is innervated, the 2019 file holds the same
entries, and a sheet that does not list every muscle is refused.

The counts are pinned rather than recomputed from the same code, because they are what the file was
checked against before it was vendored.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from quantumnematode.connectome import (
    Connectome,
    NeuromuscularJunction,
    load_cook_2019_hermaphrodite,
    load_emmons_2024_hermaphrodite,
    load_emmons_2024_neuromuscular,
)
from quantumnematode.connectome.loader import (
    COOK_2019_HERMAPHRODITE_PATH,
    EMMONS_2024_HERMAPHRODITE_PATH,
    EMMONS_2024_SHA256,
    _parse_neuromuscular_sheet,
    _read_sheet,
)
from quantumnematode.connectome.muscles import BODY_WALL_MUSCLE_QUADRANTS, BODY_WALL_MUSCLES
from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION

if TYPE_CHECKING:
    from pathlib import Path

_PROVENANCE = EMMONS_2024_HERMAPHRODITE_PATH.parent / "PROVENANCE.md"

_CHEMICAL = 3709
_CHEMICAL_SECTIONS = 20965
_AUTAPSES = 38
_GAP_PAIRS = 1095
_GAP_SECTIONS = 5864
# (pair) -> (Cook 2019 SI 5 sections, Emmons 2024 sections); None where the pair is absent.
_GAP_DIFFERENCES = {
    ("ALML", "BDUL"): (None, 23),
    ("ALMR", "BDUR"): (None, 23),
    ("BDUL", "PLML"): (23, 37),
    ("BDUR", "PLMR"): (23, 37),
}
_NEUROMUSCULAR = 956
_NEUROMUSCULAR_NEURONS = 162
_NEUROMUSCULAR_SECTIONS = 5515


@pytest.fixture(scope="module")
def emmons() -> Connectome:
    """Load the Emmons 2024 connectome once per module."""
    return load_emmons_2024_hermaphrodite()


@pytest.fixture(scope="module")
def cook() -> Connectome:
    """Load the Cook 2019 SI 5 connectome once per module, for comparison."""
    return load_cook_2019_hermaphrodite()


@pytest.fixture(scope="module")
def neuromuscular() -> list[NeuromuscularJunction]:
    """Load the Emmons 2024 file's synapses onto body wall muscle once per module."""
    return load_emmons_2024_neuromuscular()


def _gap_table(connectome: Connectome) -> dict[tuple[str, str], int]:
    return {(gj.neuron_a, gj.neuron_b): gj.weight for gj in connectome.gap_junctions}


class TestTheVendoredFile:
    """Scenario: the vendored file is the one recorded."""

    def test_the_digest_matches(self) -> None:
        """The bytes hash to the recorded SHA256."""
        digest = hashlib.sha256(EMMONS_2024_HERMAPHRODITE_PATH.read_bytes()).hexdigest()
        assert digest == EMMONS_2024_SHA256

    def test_provenance_records_the_digest(self) -> None:
        """PROVENANCE.md carries the same digest the loader checks."""
        assert EMMONS_2024_SHA256 in _PROVENANCE.read_text()

    def test_a_changed_file_is_refused(self, tmp_path: Path) -> None:
        """A copy with one byte appended is refused before it is parsed, by both loaders."""
        altered = tmp_path / EMMONS_2024_HERMAPHRODITE_PATH.name
        altered.write_bytes(EMMONS_2024_HERMAPHRODITE_PATH.read_bytes() + b"\0")
        with pytest.raises(ValueError, match="SHA256"):
            load_emmons_2024_hermaphrodite(altered)
        with pytest.raises(ValueError, match="SHA256"):
            load_emmons_2024_neuromuscular(altered)

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        """A missing file is an error naming LFS, not an empty connectome."""
        with pytest.raises(FileNotFoundError, match="git lfs pull"):
            load_emmons_2024_hermaphrodite(tmp_path / "absent.xlsx")


class TestTheChemicalWiring:
    """Scenario: the chemical wiring is the 2019 file's."""

    def test_the_neurons_are_the_302(self, emmons: Connectome, cook: Connectome) -> None:
        assert list(emmons.neurons) == list(cook.neurons)
        assert len(emmons.neurons) == 302

    def test_the_chemical_synapses_are_identical(
        self,
        emmons: Connectome,
        cook: Connectome,
    ) -> None:
        assert emmons.chemical_synapses == cook.chemical_synapses

    def test_the_chemical_counts(self, emmons: Connectome) -> None:
        synapses = emmons.chemical_synapses
        assert len(synapses) == _CHEMICAL
        assert sum(s.weight for s in synapses) == _CHEMICAL_SECTIONS
        assert sum(1 for s in synapses if s.pre == s.post) == _AUTAPSES

    def test_source_and_version_name_the_release(self, emmons: Connectome) -> None:
        assert emmons.source == "emmons_2024_hermaphrodite"
        assert "Emmons 2024" in emmons.version
        assert "Cook et al. 2019" in emmons.version

    def test_output_is_sorted(self, emmons: Connectome) -> None:
        pairs = [(s.pre, s.post) for s in emmons.chemical_synapses]
        assert pairs == sorted(pairs)
        gaps = [(g.neuron_a, g.neuron_b) for g in emmons.gap_junctions]
        assert gaps == sorted(gaps)


class TestTheGapJunctions:
    """Scenario: the gap junctions differ only by the 2023 addition."""

    def test_the_gap_counts(self, emmons: Connectome) -> None:
        assert len(emmons.gap_junctions) == _GAP_PAIRS
        assert sum(g.weight for g in emmons.gap_junctions) == _GAP_SECTIONS

    def test_only_the_bdu_pairs_differ(self, emmons: Connectome, cook: Connectome) -> None:
        """ALM-BDU is new and BDU-PLM is larger; every other pair is unchanged."""
        new, old = _gap_table(emmons), _gap_table(cook)
        differences = {
            pair: (old.get(pair), new.get(pair))
            for pair in set(new) | set(old)
            if old.get(pair) != new.get(pair)
        }
        assert differences == _GAP_DIFFERENCES

    def test_the_cook_2019_loader_is_unchanged(self, cook: Connectome) -> None:
        """Moving the Cook 2019 loader onto the shared parse left its output as it was."""
        assert cook.source == "cook_2019_hermaphrodite"
        assert len(cook.chemical_synapses) == _CHEMICAL
        assert len(cook.gap_junctions) == 1093
        assert sum(g.weight for g in cook.gap_junctions) == 5790


class TestTheBodyWallMuscles:
    """The 95 muscles, as the matrices name them."""

    def test_the_quadrants(self) -> None:
        counts = Counter(name.rstrip("0123456789") for name in BODY_WALL_MUSCLES)
        assert counts == dict(BODY_WALL_MUSCLE_QUADRANTS)
        assert counts == {"dBWML": 24, "dBWMR": 24, "vBWML": 23, "vBWMR": 24}
        assert len(set(BODY_WALL_MUSCLES)) == 95

    def test_each_quadrant_runs_from_one(self) -> None:
        for quadrant, count in BODY_WALL_MUSCLE_QUADRANTS.items():
            positions = [
                int(name.removeprefix(quadrant))
                for name in BODY_WALL_MUSCLES
                if name.rstrip("0123456789") == quadrant
            ]
            assert positions == list(range(1, count + 1))


class TestTheNeuromuscularConnections:
    """Scenario: every body wall muscle is innervated."""

    def test_the_counts(self, neuromuscular: list[NeuromuscularJunction]) -> None:
        assert len(neuromuscular) == _NEUROMUSCULAR
        assert len({j.pre for j in neuromuscular}) == _NEUROMUSCULAR_NEURONS
        assert sum(j.weight for j in neuromuscular) == _NEUROMUSCULAR_SECTIONS

    def test_every_muscle_is_innervated(
        self,
        neuromuscular: list[NeuromuscularJunction],
    ) -> None:
        assert {j.muscle for j in neuromuscular} == set(BODY_WALL_MUSCLES)

    def test_every_source_is_a_neuron(self, neuromuscular: list[NeuromuscularJunction]) -> None:
        assert {j.pre for j in neuromuscular} <= set(NEURON_CLASSIFICATION)

    def test_sorted_and_unique(self, neuromuscular: list[NeuromuscularJunction]) -> None:
        pairs = [(j.pre, j.muscle) for j in neuromuscular]
        assert pairs == sorted(pairs)
        assert len(set(pairs)) == len(pairs)

    def test_numbering_runs_head_to_tail(self, neuromuscular: list[NeuromuscularJunction]) -> None:
        """A head motor neuron reaches position 1 and the last DA reaches position 24."""
        weights = {(j.pre, j.muscle): j.weight for j in neuromuscular}
        assert weights[("DA9", "dBWML24")] == 4
        assert ("SMDDL", "dBWML1") in weights

    def test_the_2019_file_holds_the_same_entries(
        self,
        neuromuscular: list[NeuromuscularJunction],
    ) -> None:
        """Scenario: the 2019 file holds the same entries."""
        parsed = _parse_neuromuscular_sheet(
            _read_sheet(COOK_2019_HERMAPHRODITE_PATH, "hermaphrodite chemical"),
            valid_neurons=set(NEURON_CLASSIFICATION),
        )
        assert parsed == {(j.pre, j.muscle): j.weight for j in neuromuscular}


def _sheet(columns: list[str]) -> pd.DataFrame:
    """Build a minimal sheet in the Cook 2019 layout: names in row 2 and column 2, data after."""
    width = 3 + len(columns)
    return pd.DataFrame(
        [
            [None] * width,
            [None] * width,
            [None, None, None, *columns],
            [None, None, "DA9", *([1] * len(columns))],
        ],
    )


class TestAnIncompleteSheetIsRefused:
    """Scenario: a sheet that does not list every muscle is refused."""

    def test_a_complete_sheet_parses(self) -> None:
        edges = _parse_neuromuscular_sheet(
            _sheet(list(BODY_WALL_MUSCLES)),
            valid_neurons={"DA9"},
        )
        assert len(edges) == 95

    def test_a_missing_muscle_is_named(self) -> None:
        with pytest.raises(ValueError, match="vBWMR24"):
            _parse_neuromuscular_sheet(
                _sheet(list(BODY_WALL_MUSCLES[:-1])),
                valid_neurons={"DA9"},
            )

    def test_a_repeated_muscle_is_named(self) -> None:
        with pytest.raises(ValueError, match="dBWML1'"):
            _parse_neuromuscular_sheet(
                _sheet([*BODY_WALL_MUSCLES, "dBWML1"]),
                valid_neurons={"DA9"},
            )
