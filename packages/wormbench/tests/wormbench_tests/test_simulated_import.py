"""Simulated tracks: labels and seeds are required, and a seed is never shared.

Covers the benchmark-package spec's requirement "A command-line grader with the project's output"
(scenario "A simulated track without labels, or with a repeated seed").
"""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

from wormbench import grade as wg
from wormbench import measurement as mm
from wormbench import track_import as ti

if TYPE_CHECKING:
    from pathlib import Path

SECONDS = 2400
METADATA = {"family": "uniform_lawn", "analysis_start_s": 0.0, "lawn": "uniform", "units": "mm"}


def _model() -> mm.TrackerModel:
    return mm.TrackerModel(jitter_mm=0.0)


def _csv(path: Path, animals: dict[str, int | None], *, columns: tuple[str, ...] = ()) -> Path:
    fields = ["animal", "t_s", "x", "y", "primitive", "seed", *columns]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for animal, seed in animals.items():
            for t in range(SECONDS):
                writer.writerow(
                    [animal, float(t), 0.001 * t, 0.0, "run", "" if seed is None else seed],
                )
    path.with_name(path.name + ".json").write_text(json.dumps(METADATA))
    return path


def test_a_measured_animal_without_a_seed_is_refused(tmp_path: Path) -> None:
    """An animal on Scheer's rig with no seed cannot be measured, so it is refused."""
    path = _csv(tmp_path / "a.csv", {"w1": None})
    with pytest.raises(ValueError, match="seed"):
        ti.import_simulated_population(path, _model())


def test_a_track_without_labels_is_refused(tmp_path: Path) -> None:
    """A simulated CSV needs its primitive and seed columns."""
    path = tmp_path / "plain.csv"
    path.write_text("animal,t_s,x,y\nw1,0.0,0.0,0.0\nw1,1.0,0.0,0.0\n")
    path.with_name(path.name + ".json").write_text(json.dumps(METADATA))
    with pytest.raises(ValueError, match="primitive"):
        ti.read_simulated(path)


def test_unknown_labels_are_refused(tmp_path: Path) -> None:
    """Only the primitives' own labels are read."""
    path = tmp_path / "odd.csv"
    path.write_text("animal,t_s,x,y,primitive,seed\nw1,0.0,0.0,0.0,crawl,1\nw1,1.0,0.0,0.0,run,1\n")
    path.with_name(path.name + ".json").write_text(json.dumps(METADATA))
    with pytest.raises(ValueError, match="unknown primitive labels"):
        ti.read_simulated(path)


def test_a_track_not_at_one_second_is_refused(tmp_path: Path) -> None:
    """Labels do not interpolate, so a simulated track must be sampled every second."""
    path = tmp_path / "half.csv"
    path.write_text("animal,t_s,x,y,primitive,seed\nw1,0.0,0.0,0.0,run,1\nw1,0.5,0.0,0.0,run,1\n")
    path.with_name(path.name + ".json").write_text(json.dumps(METADATA))
    with pytest.raises(ValueError, match="every 1 s"):
        ti.read_simulated(path)


def test_a_seed_used_twice_in_a_manifest_is_refused(tmp_path: Path) -> None:
    """Two animals with one seed would draw the same noise, across files as within one."""
    first = _csv(tmp_path / "a.csv", {"w1": 7})
    second = _csv(tmp_path / "b.csv", {"w2": 7})
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"uniform_lawn {first.name}\nuniform_lawn {second.name}\n")
    with pytest.raises(ValueError, match="seed 7 is used twice"):
        wg.simulated_track_populations(manifest, _model())


def test_merged_wcon_records_with_different_seeds_are_refused(tmp_path: Path) -> None:
    """Records merged under one id must carry one seed."""
    document = {
        "units": {"t": "s", "x": "mm", "y": "mm"},
        "data": [
            {
                "id": "w1",
                "t": [0.0, 1.0],
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "@primitive": ["run", "run"],
                "@seed": 1,
            },
            {
                "id": "w1",
                "t": [2.0, 3.0],
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "@primitive": ["run", "run"],
                "@seed": 2,
            },
        ],
        ti.METADATA_KEY: METADATA,
    }
    path = tmp_path / "w.wcon"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="different seeds"):
        ti.read_simulated(path)


def test_local_search_needs_no_seed(tmp_path: Path) -> None:
    """Local search is never measured, so its simulated animals need no seed."""
    metadata = {"family": "local_search", "analysis_start_s": 0.0, "units": "mm"}
    path = tmp_path / "search.csv"
    rows = ["animal,t_s,x,y,primitive,seed"] + [
        f"w1,{float(t)},{0.001 * t},0.0,run," for t in range(3000)
    ]
    path.write_text("\n".join(rows) + "\n")
    path.with_name(path.name + ".json").write_text(json.dumps(metadata))
    family, animals = ti.import_simulated_population(path, _model())
    assert family == "local_search"
    assert [seed for seed, _ in animals] == [None]
