"""Tests for the *C. elegans* connectome loader."""

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from quantumnematode.connectome import (
    Connectome,
    load_cook_2019_hermaphrodite,
    load_witvliet_2021_adult,
    loader,
)

# Loose lower bounds — see add-connectome-substrate spec for rationale.
# Observed: 3709 chemical / 1093 gap-junction on Cook 2019 herm.
MIN_CHEMICAL_SYNAPSES = 3000
MIN_GAP_JUNCTIONS = 600

# Witvliet 2021 adult covers the nerve ring (~150-200 neurons of 302).
MIN_WITVLIET_NEURONS = 100
MAX_WITVLIET_NEURONS = 250


class TestLoadCook2019Hermaphrodite:
    """Verify the Cook 2019 hermaphrodite loader returns expected counts + shapes."""

    @pytest.fixture(scope="class")
    def connectome(self) -> Connectome:
        """Load the Cook 2019 hermaphrodite connectome once per class."""
        return load_cook_2019_hermaphrodite()

    def test_neuron_count_is_302(self, connectome: Connectome) -> None:
        """Connectome has exactly 302 neurons."""
        assert len(connectome.neurons) == 302

    def test_chemical_synapse_count_above_lower_bound(self, connectome: Connectome) -> None:
        """Chemical-synapse count is above the loose lower bound."""
        assert len(connectome.chemical_synapses) > MIN_CHEMICAL_SYNAPSES

    def test_gap_junction_count_above_lower_bound(self, connectome: Connectome) -> None:
        """Gap-junction count is above the loose lower bound."""
        assert len(connectome.gap_junctions) > MIN_GAP_JUNCTIONS

    def test_source_field_is_canonical(self, connectome: Connectome) -> None:
        """Source field records the canonical dataset identifier."""
        assert connectome.source == "cook_2019_hermaphrodite"

    def test_version_records_publication(self, connectome: Connectome) -> None:
        """Version string references the Cook 2019 paper."""
        assert "Cook" in connectome.version
        assert "2019" in connectome.version

    @pytest.mark.parametrize(
        "name",
        [
            "ASEL",
            "ASER",
            "AFDL",
            "AFDR",
            "ASHL",
            "ASHR",
            "ADLL",
            "ADLR",
            "AWAL",
            "AWAR",
            "AWCL",
            "AWCR",
            "URXL",
            "URXR",
            "BAGL",
            "BAGR",
        ],
    )
    def test_canonical_sensory_neurons_classified_as_sensory(
        self,
        connectome: Connectome,
        name: str,
    ) -> None:
        """Canonical sensory neurons have cell_class == 'sensory'."""
        assert connectome.neurons[name].cell_class == "sensory"

    @pytest.mark.parametrize(
        "name",
        ["VB1", "VB11", "DB1", "DB7", "VA1", "VA12", "DA1", "DA9", "VC1", "VC6", "DD1", "DD6"],
    )
    def test_canonical_motor_neurons_classified_as_motor(
        self,
        connectome: Connectome,
        name: str,
    ) -> None:
        """Canonical motor neurons have cell_class == 'motor'."""
        assert connectome.neurons[name].cell_class == "motor"

    def test_loader_is_deterministic(self, connectome: Connectome) -> None:
        """Calling the loader twice produces byte-equivalent output."""
        c2 = load_cook_2019_hermaphrodite()
        assert c2.chemical_synapses == connectome.chemical_synapses
        assert c2.gap_junctions == connectome.gap_junctions
        assert c2.neurons == connectome.neurons

    def test_chemical_synapses_sorted_by_pre_post(self, connectome: Connectome) -> None:
        """Chemical synapses are returned sorted by (pre, post)."""
        keys = [(s.pre, s.post) for s in connectome.chemical_synapses]
        assert keys == sorted(keys)

    def test_gap_junctions_sorted_by_pair(self, connectome: Connectome) -> None:
        """Gap junctions are returned sorted by (neuron_a, neuron_b)."""
        keys = [(g.neuron_a, g.neuron_b) for g in connectome.gap_junctions]
        assert keys == sorted(keys)


class TestLoadWitvliet2021Adult:
    """Verify the Witvliet 2021 adult loader returns the nerve-ring subset."""

    @pytest.fixture(scope="class")
    def connectome(self) -> Connectome:
        """Load the Witvliet 2021 adult connectome once per class."""
        return load_witvliet_2021_adult()

    def test_neuron_count_in_nerve_ring_range(self, connectome: Connectome) -> None:
        """Witvliet 2021 covers the nerve ring (~150-200 neurons of 302)."""
        assert MIN_WITVLIET_NEURONS < len(connectome.neurons) < MAX_WITVLIET_NEURONS

    def test_has_chemical_synapses(self, connectome: Connectome) -> None:
        """Witvliet 2021 has at least one chemical synapse."""
        assert len(connectome.chemical_synapses) > 0

    def test_has_gap_junctions(self, connectome: Connectome) -> None:
        """Witvliet 2021 has at least one gap junction."""
        assert len(connectome.gap_junctions) > 0

    def test_source_field_records_dataset(self, connectome: Connectome) -> None:
        """Source identifier references Witvliet adult dataset."""
        assert "witvliet" in connectome.source.lower()
        assert "adult" in connectome.source.lower()


class TestSheetCache:
    """A workbook sheet is parsed once per process, and every read gets its own copy."""

    def test_a_read_is_a_copy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Changing one read's frame leaves the next read as the file has it."""
        monkeypatch.setattr(loader, "_SHEETS", {})
        sheet = "hermaphrodite chemical"
        first = loader._read_sheet(loader.COOK_2019_HERMAPHRODITE_PATH, sheet)
        expected = first.copy()
        first.iloc[3, 3] = -1
        second = loader._read_sheet(loader.COOK_2019_HERMAPHRODITE_PATH, sheet)
        pd.testing.assert_frame_equal(second, expected)

    def test_a_changed_file_is_parsed_again(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A sheet is parsed once until its file's modification time changes."""
        parses: list[str] = []
        real_read = pd.read_excel

        def counting_read(*args, **kwargs):
            parses.append(kwargs["sheet_name"])
            return real_read(*args, **kwargs)

        monkeypatch.setattr(loader, "_SHEETS", {})
        monkeypatch.setattr(loader.pd, "read_excel", counting_read)
        copy = tmp_path / loader.WITVLIET_2021_ADULT_PATH.name
        shutil.copyfile(loader.WITVLIET_2021_ADULT_PATH, copy)
        sheet = pd.ExcelFile(copy, engine="openpyxl").sheet_names[0]
        loader._read_sheet(copy, sheet)
        loader._read_sheet(copy, sheet)
        assert parses == [sheet]
        stat = copy.stat()
        os.utime(copy, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        loader._read_sheet(copy, sheet)
        assert parses == [sheet, sheet]
