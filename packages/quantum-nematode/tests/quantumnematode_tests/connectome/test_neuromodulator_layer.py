"""The vendored neuromodulator layer, the neuron class table and the receptor-placement null.

Covers the neuromodulator-layer spec's scenarios "Every cell names its source", "One gene under two
names", "The potency sensitivity", "Ripoll-Sánchez's couples come from the CC BY networks", "The
class table", "A class of unique size", "Counts are kept exactly" and "The draw is reproducible".
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from quantumnematode.connectome.neuromodulator_layer import (
    LAYER_PATH,
    load_data_layer,
    receptor_placement_null,
)
from quantumnematode.connectome.neuron_classes import class_members, neuron_classes
from quantumnematode.connectome.neurons import NEURON_CLASSIFICATION

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))

import extract_neuromodulator_layer as ex  # noqa: E402  # pyright: ignore[reportMissingImports]

_NAMES = sorted(NEURON_CLASSIFICATION)
_SOURCES = {"bentley2016", "ripollsanchez2023"}


@pytest.fixture(scope="module")
def layer() -> dict:  # type: ignore[type-arg]
    """Return the vendored layer file."""
    return json.loads(LAYER_PATH.read_text())


def test_every_cell_names_its_source(layer: dict) -> None:  # type: ignore[type-arg]
    """Every producer and receptor cell carries at least one known source tag."""
    for ligand in layer["ligands"].values():
        for tags in ligand["producers"].values():
            assert tags
            assert set(tags) <= _SOURCES
    for couple in layer["couples"]:
        for tags in couple["cells"].values():
            assert tags
            assert set(tags) <= _SOURCES


def test_one_gene_under_two_names(layer: dict) -> None:  # type: ignore[type-arg]
    """Bentley's pdf-2 and Ripoll-Sánchez's NLP-37 are one ligand placed by both sources."""
    assert "pdf-2" in layer["ligands"]
    assert "nlp-37" not in {name.lower() for name in layer["ligands"]}
    couple = next(c for c in layer["couples"] if c["id"] == "pdf-2:pdfr-1")
    assert set(couple["sources"]) == _SOURCES
    assert len({c["id"] for c in layer["couples"]}) == len(layer["couples"])


def test_the_potency_sensitivity() -> None:
    """ec50_500nM drops the couples outside Ripoll-Sánchez's rule, pdf-1 → pdfr-1 among them.

    The file marks eight; flp-24, which has no producer in either source, carries no pool, so its
    couple is never loaded and seven are dropped.
    """
    whole = load_data_layer(_NAMES)
    strict = load_data_layer(_NAMES, sensitivity="ec50_500nM")
    names = {r.name for r in whole.rows} - {r.name for r in strict.rows}
    assert "pdf-1:pdfr-1" in names
    assert "pdf-2:pdfr-1" not in names
    assert len(names) == 7
    assert "flp-24" not in whole.ligands


def test_one_source_alone() -> None:
    """Keeping one source keeps only the cells it places."""
    bentley = load_data_layer(_NAMES, sources=["bentley2016"])
    data = json.loads(LAYER_PATH.read_text())
    tagged = {
        (c["id"], cell)
        for c in data["couples"]
        for cell, t in c["cells"].items()
        if "bentley2016" in t
    }
    for row in bentley.rows:
        if row.kind == "couple":
            assert all((row.name, cell) in tagged for cell in row.cells)


def test_the_layer_counts(layer: dict) -> None:  # type: ignore[type-arg]
    """56 ligands (flp-24 with no producer), 121 couples, 16 fixed signs, ingestion onto NSM."""
    assert len(layer["ligands"]) == 56
    assert len(layer["couples"]) == 121
    assert sum(c["sign_class"] == "fixed" for c in layer["couples"]) == 16
    assert layer["drives"][0]["cells"] == ["NSML", "NSMR"]
    assert layer["skipped_names"] == []
    assert [k for k, v in layer["ligands"].items() if not v["producers"]] == ["flp-24"]


def test_a_network_that_is_not_an_outer_product_is_refused(tmp_path: Path) -> None:
    """The extraction raises when a long-range network is not its producers times its receptors."""
    folder = tmp_path / "ripollsanchez2023"
    folder.mkdir()
    (folder / "20230602_Ligand_receptors_interactions_all_EC50_explicit_500nM.csv").write_text(
        "GeneGPCR,GeneLigand\nNPR-1,FLP-18\n",
    )
    names = ["AIYL", "AIYR", "RIAL"]
    rows = ["Row," + ",".join(names), "AIYL,0,1,0", "AIYR,1,0,0", "RIAL,0,0,0"]
    (folder / "Individual_LR08062023_neuropeptide_network001_long_range_model.csv").write_text(
        "\n".join(rows) + "\n",
    )
    with pytest.raises(ValueError, match="not an outer product"):
        ex.ripoll_couples(tmp_path)


@pytest.mark.parametrize(
    ("text", "upper"),
    [
        ("~100 nM", 100.0),
        ("0.4-9 μM", 9000.0),
        ("5 nM-1.2 μM", 1200.0),
        ("1-28nM", 28.0),
        ("114 nM", 114.0),
    ],
)
def test_ec50_upper_bounds(text: str, upper: float) -> None:
    """A range's upper end, in nM, with the unit read from where the source writes it."""
    assert ex.ec50_upper_nm(text) == pytest.approx(upper)


def test_motor_neuron_names_lose_their_padding() -> None:
    """DA01 is DA1; other names pass unchanged."""
    assert ex.neuron_name("DA01") == "DA1"
    assert ex.neuron_name("VD13") == "VD13"
    assert ex.neuron_name("AIYL") == "AIYL"


def test_the_class_table() -> None:
    """The 302 neurons form WormAtlas's 118 classes."""
    classes = neuron_classes()
    members = class_members(classes)
    assert len(classes) == 302
    assert len(members) == 118
    for name, size in (
        ("OLQ", 4),
        ("IL1", 6),
        ("VD", 13),
        ("SAB", 3),
        ("RME", 4),
        ("AVL", 1),
        ("RIV", 2),
    ):
        assert len(members[name]) == size


def _cells_by_gene(spec) -> dict[str, set[str]]:  # type: ignore[no-untyped-def]
    cells: dict[str, set[str]] = defaultdict(set)
    for row in spec.rows:
        if row.kind == "couple":
            cells[row.receptor] |= set(row.cells)
    return cells


def test_counts_are_kept_exactly() -> None:
    """Each receptor gene keeps its cell count and class count, with one placement per gene."""
    spec = load_data_layer(_NAMES)
    null = receptor_placement_null(spec, 7501, _NAMES)
    classes = neuron_classes(_NAMES)
    before, after = _cells_by_gene(spec), _cells_by_gene(null)
    for gene, cells in before.items():
        assert len(after[gene]) == len(cells)
        assert len({classes[c] for c in after[gene]}) == len({classes[c] for c in cells})
    for row in null.rows:
        if row.kind == "couple":
            assert row.receptor is not None
            assert set(row.cells) <= after[row.receptor]
    assert [r.cells for r in null.rows if r.kind == "drive"] == [
        r.cells for r in spec.rows if r.kind == "drive"
    ]
    assert null.producers == spec.producers


def test_a_class_of_unique_size_stays() -> None:
    """VD has 13 cells and no other class does, so a receptor on VD keeps it."""
    spec = load_data_layer(_NAMES)
    null = receptor_placement_null(spec, 7502, _NAMES)
    before, after = _cells_by_gene(spec), _cells_by_gene(null)
    vd = {f"VD{i}" for i in range(1, 14)}
    for gene, cells in before.items():
        assert (cells & vd) == (after[gene] & vd)


def test_the_draw_is_reproducible() -> None:
    """The same seed gives the same layer; another seed moves something."""
    spec = load_data_layer(_NAMES)
    first = receptor_placement_null(spec, 7503, _NAMES)
    assert first == receptor_placement_null(spec, 7503, _NAMES)
    assert first != receptor_placement_null(spec, 7504, _NAMES)
    assert np.mean([a.cells != b.cells for a, b in zip(spec.rows, first.rows, strict=True)]) > 0.5


def test_the_shortcut_term_counts_couples_only() -> None:
    """A link from a sensor-made ligand onto a motor neuron is not a couple, so is not counted."""
    from dataclasses import replace
    from types import SimpleNamespace

    import sensory_motor_hops as hops  # pyright: ignore[reportMissingImports]
    import torch
    from quantumnematode.connectome.neuromodulator_layer import LayerSpec, Row

    topology = SimpleNamespace(
        _idx={name: i for i, name in enumerate(_NAMES)},
        _motor_flat_indices=torch.tensor([_NAMES.index("VA1")]),
    )
    spec = LayerSpec(
        ligands=("x",),
        producers=(("ASEL",),),
        rows=(Row("link", ("VA1",), -1, ligand=0, receptor="r-1", name="x:VA1"),),
        drives=(),
    )
    original = hops.injected_sensors
    hops.injected_sensors = lambda _t: [_NAMES.index("ASEL")]
    try:
        assert hops.modulator_shortcuts(topology, spec) == {"couples": 0, "motor_cells": 0}
        couple = replace(spec, rows=(replace(spec.rows[0], kind="couple"),))
        assert hops.modulator_shortcuts(topology, couple) == {"couples": 1, "motor_cells": 1}
    finally:
        hops.injected_sensors = original
