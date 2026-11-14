"""The Wormturn export: its signed wiring, its layer, its rendering, and its refusal of dirty trees.

Covers the connectome-substrate requirement "A versioned export of the signed connectome and the
neuromodulator layer for Wormturn": the export carries the signed wiring, the signs are the wiring's
own, the layer round-trips against its files, the rendering is stable and parses back, a dirty tree
is refused unless allowed, and the Wormlight export is unchanged. Git is never called: the tests
that reach ``main`` replace its view of the tree.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from quantumnematode.connectome.loader import (
    EMMONS_2024_SHA256,
    load_cook_2019_hermaphrodite,
    load_emmons_2024_hermaphrodite,
)
from quantumnematode.connectome.muscles import BODY_WALL_MUSCLES
from quantumnematode.connectome.neuromodulator_layer import LAYER_PATH, RECEPTOR_COUPLINGS_PATH
from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION
from quantumnematode.connectome.neurotransmitters import (
    ATLAS_SHA256,
    read_atlas_identities,
    sign_for,
)
from quantumnematode.connectome.signs import (
    FENYVES_S1_SHA256,
    FENYVES_S5_SHA256,
    PHYSIOLOGY_CITATIONS,
    per_connection_signs,
)

_REPO_ROOT = Path(__file__).resolve().parents[4].parent
_scripts = _REPO_ROOT / "scripts"
if not _scripts.is_dir():
    msg = f"could not locate scripts/ from {Path(__file__).resolve()}"
    raise RuntimeError(msg)
sys.path.insert(0, str(_scripts))

import export_wormlight as ew  # noqa: E402  # pyright: ignore[reportMissingImports]
import export_wormturn as wt  # noqa: E402  # pyright: ignore[reportMissingImports]

_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_SOURCES = ("physiology", "expression", "rule", "none")


@pytest.fixture(scope="module")
def export() -> dict[str, Any]:
    """Build an export from a clean tree at a fixed commit, once per module."""
    return wt.build_export(commit=_COMMIT, dirty=False)


@pytest.fixture(scope="module")
def layer() -> dict[str, Any]:
    """Read the vendored layer file directly."""
    return json.loads(LAYER_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def couplings() -> dict[str, Any]:
    """Read the vendored receptor-coupling file directly."""
    return json.loads(RECEPTOR_COUPLINGS_PATH.read_text(encoding="utf-8"))


class TestTheExportCarriesTheSignedWiring:
    """Scenario: the export carries the signed wiring."""

    def test_the_schema(self, export: dict[str, Any]) -> None:
        assert export["schema"] == "wormturn.connectome/1"
        assert list(export) == [
            "schema",
            "provenance",
            "neurons",
            "chemical",
            "gap",
            "signCitations",
            "ligands",
            "couples",
            "links",
            "drives",
            "receptorCouplingRule",
            "receptorCouplings",
        ]

    def test_the_counts(self, export: dict[str, Any]) -> None:
        assert len(export["neurons"]) == 302
        assert len(export["chemical"]) == 3709
        assert len(export["gap"]) == 1095
        assert len(export["signCitations"]) == 9
        assert len(export["ligands"]) == 56
        assert len(export["couples"]) == 121
        assert export["links"] == []
        assert len(export["drives"]) == 1
        assert len(export["receptorCouplingRule"]) == 4
        assert len(export["receptorCouplings"]) == 54

    def test_the_section_totals(self, export: dict[str, Any]) -> None:
        assert sum(r["sections"] for r in export["chemical"]) == 20965
        assert sum(r["sections"] for r in export["gap"]) == 5864

    def test_no_muscles(self, export: dict[str, Any]) -> None:
        """A deliberate scope choice: no muscle list, no neuromuscular list, no muscle anywhere."""
        assert "muscles" not in export
        assert "neuromuscular" not in export
        named = {r["pre"] for r in export["chemical"]} | {r["post"] for r in export["chemical"]}
        named |= {r["a"] for r in export["gap"]} | {r["b"] for r in export["gap"]}
        assert named <= set(NEURON_CLASSIFICATION)
        assert not named & set(BODY_WALL_MUSCLES)

    def test_neurons_carry_the_table_and_the_atlas(self, export: dict[str, Any]) -> None:
        """Classes from the table, identities as the atlas reader gives them, primary first."""
        identities = read_atlas_identities()
        names = [neuron["name"] for neuron in export["neurons"]]
        assert names == sorted(NEURON_CLASSIFICATION)
        for neuron in export["neurons"]:
            assert list(neuron) == ["name", "class", "transmitters", "ruleSign"]
            cell_class, primary = NEURON_CLASSIFICATION[neuron["name"]]
            assert neuron["class"] == cell_class
            assert tuple(neuron["transmitters"]) == identities[neuron["name"]]
            assert neuron["ruleSign"] == sign_for(primary)

    def test_neurons_and_gap_are_the_wormlight_exports(self, export: dict[str, Any]) -> None:
        """The two lists the exports share hold the same records."""
        wormlight = ew.build_export(commit=_COMMIT, dirty=False)
        assert export["neurons"] == wormlight["neurons"]
        assert export["gap"] == wormlight["gap"]
        assert [(r["pre"], r["post"], r["sections"]) for r in export["chemical"]] == [
            (r["pre"], r["post"], r["sections"]) for r in wormlight["chemical"]
        ]

    def test_gap_pairs_are_canonical(self, export: dict[str, Any]) -> None:
        pairs = [(r["a"], r["b"]) for r in export["gap"]]
        assert all(list(r) == ["a", "b", "sections"] for r in export["gap"])
        assert all(a < b for a, b in pairs)
        assert len(set(pairs)) == len(pairs)

    def test_every_edge_has_one_sign_and_one_source(self, export: dict[str, Any]) -> None:
        edges = [(r["pre"], r["post"]) for r in export["chemical"]]
        assert len(set(edges)) == len(edges)
        for record in export["chemical"]:
            assert list(record) == ["pre", "post", "sections", "sign", "signSource", "citation"]
            assert record["signSource"] in _SOURCES
            assert record["sign"] in (1, -1, 0)
            assert (record["sign"] == 0) == (record["signSource"] == "none")
            assert record["sections"] > 0

    def test_the_sign_source_counts(self, export: dict[str, Any]) -> None:
        sources = Counter(r["signSource"] for r in export["chemical"])
        assert sources == {"physiology": 67, "expression": 1676, "rule": 1433, "none": 533}
        signs = Counter((r["signSource"], r["sign"]) for r in export["chemical"])
        assert signs == {
            ("physiology", 1): 19,
            ("physiology", -1): 48,
            ("expression", 1): 1291,
            ("expression", -1): 385,
            ("rule", 1): 1308,
            ("rule", -1): 125,
            ("none", 0): 533,
        }

    def test_a_rule_sign_is_the_presynaptic_neurons(self, export: dict[str, Any]) -> None:
        rule_signs = {neuron["name"]: neuron["ruleSign"] for neuron in export["neurons"]}
        for record in export["chemical"]:
            if record["signSource"] == "rule":
                assert record["sign"] == rule_signs[record["pre"]]
            if record["signSource"] == "none":
                assert rule_signs[record["pre"]] is None

    def test_physiology_signs_cite_the_table(self, export: dict[str, Any]) -> None:
        """Every physiology sign, and no other, carries a key the table lists; every key is used."""
        table = {entry["key"]: entry["citation"] for entry in export["signCitations"]}
        assert table == PHYSIOLOGY_CITATIONS
        assert [entry["key"] for entry in export["signCitations"]] == list(PHYSIOLOGY_CITATIONS)
        assert all(list(entry) == ["key", "citation"] for entry in export["signCitations"])
        used = set()
        for record in export["chemical"]:
            if record["signSource"] == "physiology":
                assert record["citation"] in table
                used.add(record["citation"])
            else:
                assert record["citation"] is None
        assert used == set(table)
        cited = Counter(r["citation"] for r in export["chemical"] if r["citation"])
        assert cited["lopezcruz2019"] == 16

    def test_example_records(self, export: dict[str, Any]) -> None:
        chemical = {(r["pre"], r["post"]): r for r in export["chemical"]}
        assert chemical[("AWCL", "AIYL")] == {
            "pre": "AWCL",
            "post": "AIYL",
            "sections": 22,
            "sign": -1,
            "signSource": "physiology",
            "citation": "chalasani2007",
        }
        assert chemical[("ASEL", "AIAL")]["citation"] == "lopezcruz2019"
        assert chemical[("ADFL", "AIZL")]["sign"] == -1
        assert chemical[("ADFL", "AIZL")]["signSource"] == "expression"
        assert chemical[("ALA", "AVDR")]["sign"] == 1
        assert chemical[("ALA", "AVDR")]["signSource"] == "expression"
        assert chemical[("ADLL", "AIAL")]["signSource"] == "rule"
        assert chemical[("ADEL", "RIGL")]["sign"] == 0
        assert chemical[("ADEL", "RIGL")]["signSource"] == "none"
        assert {"a": "ALA", "b": "CANL", "sections": 401} in export["gap"]


class TestTheSignsAreTheWiringsOwn:
    """Scenario: the signs are the wiring's own."""

    def test_emmons_and_cook_chemical_edges_match(self) -> None:
        emmons = load_emmons_2024_hermaphrodite()
        cook = load_cook_2019_hermaphrodite()
        assert [(s.pre, s.post, s.weight) for s in emmons.chemical_synapses] == [
            (s.pre, s.post, s.weight) for s in cook.chemical_synapses
        ]

    def test_the_signs_equal_either_table(self, export: dict[str, Any]) -> None:
        exported = {
            (r["pre"], r["post"]): (r["sign"], r["signSource"], r["citation"])
            for r in export["chemical"]
        }
        on_emmons = per_connection_signs(load_emmons_2024_hermaphrodite())
        on_cook = per_connection_signs()
        assert on_emmons == on_cook
        assert list(exported) == list(on_emmons)
        assert exported == {
            edge: (sign.sign, sign.source, sign.citation) for edge, sign in on_emmons.items()
        }


class TestTheLayerRoundTrips:
    """Scenario: the layer round-trips against its files."""

    def test_the_ligands(self, export: dict[str, Any], layer: dict[str, Any]) -> None:
        """The same ligands in the file's order, with the same kinds, producers and source tags."""
        assert [r["name"] for r in export["ligands"]] == list(layer["ligands"])
        for record in export["ligands"]:
            assert list(record) == ["name", "kind", "producers"]
            entry = layer["ligands"][record["name"]]
            assert record["kind"] == entry["kind"]
            assert record["producers"] == entry["producers"]
        assert Counter(r["kind"] for r in export["ligands"]) == {"peptide": 52, "monoamine": 4}

    def test_the_couples(self, export: dict[str, Any], layer: dict[str, Any]) -> None:
        """The same couples in the file's order: ids, ligands, receptors, cells, tags and signs."""
        assert export["couples"] == layer["couples"]
        for record, entry in zip(export["couples"], layer["couples"], strict=True):
            assert list(record) == list(entry)
            for key in ("id", "ligand", "receptor", "sign", "sign_class", "within_500nM"):
                assert record[key] == entry[key]
            assert record["cells"] == entry["cells"]
        classes = Counter((r["sign_class"], r["sign"]) for r in export["couples"])
        assert classes == {("fitted", None): 105, ("fixed", 1): 7, ("fixed", -1): 9}

    def test_the_links_and_drives(self, export: dict[str, Any], layer: dict[str, Any]) -> None:
        assert export["links"] == layer["links"]
        assert export["drives"] == layer["drives"]
        assert export["drives"][0]["name"] == "ingestion"
        assert export["drives"][0]["cells"] == ["NSML", "NSMR"]

    def test_the_receptor_couplings(
        self,
        export: dict[str, Any],
        couplings: dict[str, Any],
    ) -> None:
        """Every entry as the file holds it, in the file's order: sign, verified, citation, note."""
        receptors = couplings["receptors"]
        assert [r["receptor"] for r in export["receptorCouplings"]] == list(receptors)
        for record in export["receptorCouplings"]:
            entry = receptors[record["receptor"]]
            assert {key: value for key, value in record.items() if key != "receptor"} == entry
            assert list(record)[1:] == list(entry)
        rule = {r["coupling"]: r["sign"] for r in export["receptorCouplingRule"]}
        assert rule == couplings["rule"]
        assert [r["coupling"] for r in export["receptorCouplingRule"]] == list(couplings["rule"])

    def test_the_files_quirks_are_kept(self, export: dict[str, Any]) -> None:
        """What does not fit cleanly is carried as it is."""
        ligands = {r["name"]: r for r in export["ligands"]}
        assert ligands["flp-24"]["producers"] == {}
        assert any(r["id"] == "flp-24:npr-17" for r in export["couples"])
        verified = [r["verified"] for r in export["receptorCouplings"]]
        assert sum(v is False for v in verified) == 32
        assert sum(isinstance(v, dict) and set(v) == {"date", "source"} for v in verified) == 22
        signed = [r for r in export["receptorCouplings"] if r["sign"] is not None]
        assert len(signed) == 22
        assert all(r["verified"] for r in signed)

    def test_every_layer_cell_is_a_neuron(self, export: dict[str, Any]) -> None:
        cells = {cell for r in export["ligands"] for cell in r["producers"]}
        cells |= {cell for r in export["couples"] for cell in r["cells"]}
        cells |= {cell for r in export["drives"] for cell in r["cells"]}
        assert cells <= {neuron["name"] for neuron in export["neurons"]}

    def test_it_survives_the_rendering(
        self,
        export: dict[str, Any],
        layer: dict[str, Any],
        couplings: dict[str, Any],
    ) -> None:
        """Text outside ASCII is escaped and parses back to the files' strings."""
        parsed = json.loads(wt.render(export))
        assert parsed["couples"] == layer["couples"]
        assert {r.pop("receptor"): r for r in parsed["receptorCouplings"]} == couplings["receptors"]

    def test_a_clashing_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="already have"):
            wt._named({"x": {"name": "y"}}, "name")


class TestTheProvenance:
    """Part of the requirement: every input recorded with its path, digest and role."""

    def test_the_inputs(self, export: dict[str, Any]) -> None:
        provenance = export["provenance"]
        assert provenance["exporter"] == "scripts/export_wormturn.py"
        assert provenance["nematodeCommit"] == _COMMIT
        assert provenance["nematodeDirty"] is False
        files = [entry["file"] for entry in provenance["inputs"]]
        assert files == [
            "data/connectome/emmons_2024_s1_connectome_adjacency.xlsx",
            "packages/quantum-nematode/quantumnematode/connectome/neurons.py",
            "data/connectome/elife-95402-supp2-v1.xlsx",
            "data/connectome/sign_overrides_physiology.csv",
            "data/connectome/fenyves_2020_s1_data.xlsx",
            "data/connectome/fenyves_2020_s5_data.xlsx",
            "packages/quantum-nematode/quantumnematode/connectome/signs.py",
            "data/neuromodulators/layer.json",
            "data/neuromodulators/receptor_couplings.json",
        ]
        for entry in provenance["inputs"]:
            assert list(entry) == ["file", "sha256", "role"]
            digest = hashlib.sha256((_REPO_ROOT / entry["file"]).read_bytes()).hexdigest()
            assert entry["sha256"] == digest
            assert entry["role"]
        digests = {entry["file"]: entry["sha256"] for entry in provenance["inputs"]}
        assert digests[files[0]] == EMMONS_2024_SHA256
        assert digests[files[2]] == ATLAS_SHA256
        assert digests[files[4]] == FENYVES_S1_SHA256
        assert digests[files[5]] == FENYVES_S5_SHA256

    def test_an_lfs_pointer_is_refused(self, tmp_path: Path) -> None:
        pointer = tmp_path / "atlas.xlsx"
        pointer.write_text(
            f"version https://git-lfs.github.com/spec/v1\noid sha256:{ATLAS_SHA256}\nsize 73537\n",
        )
        with pytest.raises(ValueError, match="Git LFS pointer"):
            wt.checked_input(pointer, "release identities")


class TestTheRendering:
    """Scenario: the rendering is stable and parses back."""

    def test_it_parses_back(self, export: dict[str, Any]) -> None:
        assert json.loads(wt.render(export)) == export

    def test_one_record_per_line(self, export: dict[str, Any]) -> None:
        """After the provenance, list openers and closers sit at two spaces, records at four."""
        lines = wt.render(export).splitlines()
        body = lines[lines.index('  "neurons": [') :]
        records = [line for line in body if line.startswith("    ")]
        assert len(records) == sum(len(export[key]) for key in wt._RECORD_LISTS)
        assert (
            '    {"pre": "AWCL", "post": "AIYL", "sections": 22, "sign": -1, '
            '"signSource": "physiology", "citation": "chalasani2007"},'
        ) in records
        assert all(json.loads(line.strip().removesuffix(",")) for line in records)
        assert '  "links": [' in body

    def test_it_is_stable(self, export: dict[str, Any]) -> None:
        assert wt.render(export) == wt.render(json.loads(wt.render(export)))
        assert wt.render(export).endswith("}\n")
        assert wt.render(export).isascii()

    def test_unknown_or_missing_keys_are_refused(self, export: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="expected"):
            wt.render({**export, "muscles": []})
        with pytest.raises(ValueError, match="expected"):
            wt.render({key: value for key, value in export.items() if key != "couples"})


class TestADirtyTree:
    """Scenario: a dirty tree is refused unless allowed."""

    def test_refused_without_allow_dirty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(ew, "git_state", lambda: (_COMMIT, True))
        out = tmp_path / "connectome.v1.json"
        assert wt.main(["--out", str(out)]) == 1
        assert not out.exists()

    def test_allow_dirty_writes_and_records_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        export: dict[str, Any],
    ) -> None:
        monkeypatch.setattr(ew, "git_state", lambda: (_COMMIT, True))
        monkeypatch.setattr(
            wt,
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
        assert wt.main(["--out", str(out), "--allow-dirty"]) == 0
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
        monkeypatch.setattr(wt, "build_export", lambda *, commit, dirty: export)
        out = tmp_path / "connectome.v1.json"
        assert wt.main(["--out", str(out)]) == 0
        assert out.read_text() == wt.render(export)


class TestTheWormlightExportIsUnchanged:
    """Scenario: the Wormlight export is unchanged by sharing its helpers.

    Its own tests pin its records and rendering; this checks that the shared renderer still writes
    its lists, in its order, by default.
    """

    def test_its_default_record_lists(self) -> None:
        assert ew._RECORD_LISTS == ("neurons", "muscles", "chemical", "gap", "neuromuscular")
        lists: dict[str, list[Any]] = {key: [] for key in ew._RECORD_LISTS}
        export = {"schema": ew.SCHEMA, "provenance": {}, **lists}
        assert ew.render(export) == ew.render(export, ew._RECORD_LISTS)
        with pytest.raises(ValueError, match="expected"):
            ew.render(export, wt._RECORD_LISTS)
