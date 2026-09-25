"""The Wormlight export: its content, its rendering, and its refusal of a dirty tree.

Covers the connectome-substrate requirement "A versioned export of the connectome for Wormlight":
the export carries the loaded connectome, the rendering is stable and parses back, and a dirty
tree is refused unless allowed. Git is never called: the tests that reach ``main`` replace its view
of the tree.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from quantumnematode.connectome.loader import EMMONS_2024_SHA256
from quantumnematode.connectome.muscles import BODY_WALL_MUSCLES
from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION
from quantumnematode.connectome.neurotransmitters import read_atlas_identities, sign_for

_REPO_ROOT = Path(__file__).resolve().parents[4].parent
_scripts = _REPO_ROOT / "scripts"
if not _scripts.is_dir():
    msg = f"could not locate scripts/ from {Path(__file__).resolve()}"
    raise RuntimeError(msg)
sys.path.insert(0, str(_scripts))

import export_wormlight as ew  # noqa: E402  # pyright: ignore[reportMissingImports]

_COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def export() -> dict[str, Any]:
    """Build an export from a clean tree at a fixed commit, once per module."""
    return ew.build_export(commit=_COMMIT, dirty=False)


class TestTheExportCarriesTheConnectome:
    """Scenario: the export carries the loaded connectome."""

    def test_the_schema(self, export: dict[str, Any]) -> None:
        assert export["schema"] == "wormlight.connectome/1"
        assert list(export) == [
            "schema",
            "provenance",
            "neurons",
            "muscles",
            "chemical",
            "gap",
            "neuromuscular",
        ]

    def test_the_counts(self, export: dict[str, Any]) -> None:
        assert len(export["neurons"]) == 302
        assert export["muscles"] == list(BODY_WALL_MUSCLES)
        assert len(export["chemical"]) == 3709
        assert len(export["gap"]) == 1095
        assert len(export["neuromuscular"]) == 956

    def test_the_section_totals(self, export: dict[str, Any]) -> None:
        assert sum(r["sections"] for r in export["chemical"]) == 20965
        assert sum(r["sections"] for r in export["gap"]) == 5864
        assert sum(r["sections"] for r in export["neuromuscular"]) == 5515

    def test_neurons_carry_the_table_and_the_atlas(self, export: dict[str, Any]) -> None:
        """Classes from the table, identities as the atlas reader gives them, primary first."""
        identities = read_atlas_identities()
        names = [neuron["name"] for neuron in export["neurons"]]
        assert names == sorted(NEURON_CLASSIFICATION)
        for neuron in export["neurons"]:
            cell_class, primary = NEURON_CLASSIFICATION[neuron["name"]]
            assert neuron["class"] == cell_class
            assert tuple(neuron["transmitters"]) == identities[neuron["name"]]
            assert neuron["ruleSign"] == sign_for(primary)

    def test_example_records(self, export: dict[str, Any]) -> None:
        neurons = {neuron["name"]: neuron for neuron in export["neurons"]}
        assert neurons["AWCL"]["transmitters"] == ["Glu"]
        assert neurons["AWCL"]["ruleSign"] == 1
        assert neurons["DD1"]["ruleSign"] == -1
        assert neurons["ADFL"]["transmitters"] == ["ACh", "5-HT"]
        assert {"pre": "AWCL", "post": "AIYL", "sections": 22} in export["chemical"]
        assert {"a": "ALA", "b": "CANL", "sections": 401} in export["gap"]
        assert {"pre": "DA9", "muscle": "dBWML24", "sections": 4} in export["neuromuscular"]

    def test_gap_pairs_are_canonical(self, export: dict[str, Any]) -> None:
        pairs = [(r["a"], r["b"]) for r in export["gap"]]
        assert all(a < b for a, b in pairs)
        assert len(set(pairs)) == len(pairs)

    def test_provenance(self, export: dict[str, Any]) -> None:
        provenance = export["provenance"]
        assert provenance["nematodeCommit"] == _COMMIT
        assert provenance["nematodeDirty"] is False
        files = [entry["file"] for entry in provenance["inputs"]]
        assert files == [
            "data/connectome/emmons_2024_s1_connectome_adjacency.xlsx",
            "packages/quantum-nematode/quantumnematode/connectome/neurons.py",
            "data/connectome/elife-95402-supp2-v1.xlsx",
        ]
        for entry in provenance["inputs"]:
            digest = hashlib.sha256((_REPO_ROOT / entry["file"]).read_bytes()).hexdigest()
            assert entry["sha256"] == digest
            assert entry["role"]
        assert provenance["inputs"][0]["sha256"] == EMMONS_2024_SHA256


class TestTheRendering:
    """Scenario: the rendering is stable and parses back."""

    def test_it_parses_back(self, export: dict[str, Any]) -> None:
        assert json.loads(ew.render(export)) == export

    def test_one_record_per_line(self, export: dict[str, Any]) -> None:
        """After the provenance, list openers and closers sit at two spaces, records at four."""
        lines = ew.render(export).splitlines()
        body = lines[lines.index('  "neurons": [') :]
        records = [line for line in body if line.startswith("    ")]
        assert len(records) == sum(len(export[key]) for key in ew._RECORD_LISTS)
        assert '    {"pre": "AWCL", "post": "AIYL", "sections": 22},' in records
        assert '    "dBWML1",' in records

    def test_it_is_stable(self, export: dict[str, Any]) -> None:
        assert ew.render(export) == ew.render(json.loads(ew.render(export)))
        assert ew.render(export).endswith("}\n")

    def test_unknown_or_missing_keys_are_refused(self, export: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="expected"):
            ew.render({**export, "extra": []})
        with pytest.raises(ValueError, match="expected"):
            ew.render({key: value for key, value in export.items() if key != "gap"})


class TestADirtyTree:
    """Scenario: a dirty tree is refused unless allowed."""

    def test_refused_without_allow_dirty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(ew, "git_state", lambda: (_COMMIT, True))
        out = tmp_path / "connectome.v1.json"
        assert ew.main(["--out", str(out)]) == 1
        assert not out.exists()

    def test_allow_dirty_writes_and_records_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        export: dict[str, Any],
    ) -> None:
        monkeypatch.setattr(ew, "git_state", lambda: (_COMMIT, True))
        monkeypatch.setattr(
            ew,
            "build_export",
            lambda *, commit, dirty: {
                **export,
                "provenance": {
                    **export["provenance"],
                    "nematodeCommit": commit,
                    "nematodeDirty": dirty,
                },
            },
        )
        out = tmp_path / "nested" / "connectome.v1.json"
        assert ew.main(["--out", str(out), "--allow-dirty"]) == 0
        written = json.loads(out.read_text())
        assert written["provenance"]["nematodeDirty"] is True
        assert written["provenance"]["nematodeCommit"] == _COMMIT

    def test_a_clean_tree_writes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        export: dict[str, Any],
    ) -> None:
        monkeypatch.setattr(ew, "git_state", lambda: (_COMMIT, False))
        monkeypatch.setattr(ew, "build_export", lambda *, commit, dirty: export)
        out = tmp_path / "connectome.v1.json"
        assert ew.main(["--out", str(out)]) == 0
        assert out.read_text() == ew.render(export)
