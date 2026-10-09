"""The chemotaxis reference set: verified entries, their schema, and the loader.

Covers the experiment-tracking requirement "A verified chemotaxis reference set" (every entry is
traceable to its paper's text or figure).
"""

import json
from pathlib import Path

import pytest
from quantumnematode.validation.datasets import (
    ChemotaxisDataset,
    LiteratureSource,
    load_chemotaxis_dataset,
)

_ENTRY = {
    "citation": "Paper (2025). Journal 1:1",
    "attractant": "bacteria",
    "ci_wild_type": 0.9,
    "assay": "an endpoint count",
    "read_from": "text",
    "notes": "stated in the text",
}


def _write(tmp_path: Path, sources: list[dict]) -> Path:
    path = tmp_path / "ci.json"
    path.write_text(json.dumps({"version": "2.0", "sources": sources}))
    return path


class TestTheReferenceSet:
    def test_every_entry_is_traceable(self):
        """Each packaged entry has a citation, an assay and a text or figure value source."""
        dataset = load_chemotaxis_dataset()
        assert dataset.sources
        for source in dataset.sources:
            assert source.citation
            assert source.assay
            assert source.read_from in ("text", "figure")
            assert -1.0 <= source.ci_wild_type <= 1.0

    def test_bacteria_is_a_measured_bacterial_assay(self):
        """The bacteria entry cites a paper that assayed bacteria, with its stated value."""
        source = load_chemotaxis_dataset().get_source_by_attractant("bacteria")
        assert source is not None
        assert "Rodriguez" in source.citation
        assert "OP50" in source.assay
        assert source.read_from == "text"
        assert source.ci_wild_type == pytest.approx(0.9)

    def test_the_misattributed_citations_are_gone(self):
        """None of the citations the check found wrong is listed as a source."""
        citations = " ".join(s.citation for s in load_chemotaxis_dataset().sources)
        for wrong in ("Cell 65(5)", "Neuron 32(2)", "Genetics 175(1)", "J Neurosci 19(21)"):
            assert wrong not in citations


class TestTheLoader:
    def test_a_valid_file_loads(self, tmp_path: Path):
        dataset = load_chemotaxis_dataset(_write(tmp_path, [_ENTRY]))
        assert isinstance(dataset, ChemotaxisDataset)
        assert dataset.version == "2.0"
        assert dataset.sources == [LiteratureSource(**_ENTRY)]

    def test_lookup_is_case_insensitive(self, tmp_path: Path):
        dataset = load_chemotaxis_dataset(_write(tmp_path, [_ENTRY]))
        assert dataset.get_source_by_attractant("BACTERIA") is not None
        assert dataset.get_source_by_attractant("salt") is None

    def test_a_missing_field_is_refused(self, tmp_path: Path):
        entry = {k: v for k, v in _ENTRY.items() if k != "assay"}
        with pytest.raises(ValueError, match="missing required field"):
            load_chemotaxis_dataset(_write(tmp_path, [entry]))

    def test_an_unknown_value_source_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="read_from"):
            load_chemotaxis_dataset(_write(tmp_path, [{**_ENTRY, "read_from": "recalled"}]))

    def test_a_missing_file_is_an_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_chemotaxis_dataset(tmp_path / "absent.json")
