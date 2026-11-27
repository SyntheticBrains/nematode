"""Writing tracks in the formats the grader reads: CSV with its sidecar, and WCON.

Covers the benchmark-package spec's requirement "Tracks written in the formats the grader reads"
(scenarios "A CSV written as WCON grades the same", "Simulated labels and seeds are carried" and
"A track whose grade WCON cannot hold is refused").
"""

from __future__ import annotations

import csv
import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from wormbench import cli, data
from wormbench import measurement as mm
from wormbench import track_export as te
from wormbench import track_import as ti

if TYPE_CHECKING:
    from pathlib import Path

SECONDS = 2700  # local search, the longest family window
LAWNS = {
    "small_lawn": {"centre": [0.5, -0.25], "radii": [1.9, 2.0, 2.1, 2.0]},
    "uniform_lawn": "uniform",
}


def _tracks(seconds: int = SECONDS) -> list[ti.ImportedTrack]:
    """Two animals crossing a lawn's edge, each with a gap and a missing sample."""
    rng = np.random.default_rng(7)
    out = []
    for k, animal in enumerate(("w1", "w2")):
        t = np.arange(seconds, dtype=float)
        t = np.delete(t, np.arange(100 + k, 104 + k))  # a 5-s gap: the importer marks it missing
        x = np.cumsum(rng.normal(0.0, 0.02, len(t))) + 1.5
        y = np.cumsum(rng.normal(0.0, 0.02, len(t)))
        x[10 + k] = math.nan
        out.append(ti.ImportedTrack(animal=animal, t_s=t, x_mm=x, y_mm=y))
    return out


def _metadata(family: str) -> ti.AssayMetadata:
    return ti.parse_metadata({"family": family, "analysis_start_s": 0.0, "lawn": LAWNS.get(family)})


def _steps(population: list[list]) -> list[list[tuple[float, float, bool]]]:
    return [[(s.x, s.y, s.on_lawn) for s in animal] for animal in population]


def _same_steps(a: list[list], b: list[list]) -> bool:
    return np.array_equal(
        np.asarray(_steps(a), dtype=float),
        np.asarray(_steps(b), dtype=float),
        equal_nan=True,
    )


@pytest.mark.parametrize("family", ["small_lawn", "uniform_lawn", "local_search"])
def test_both_formats_read_back_as_written(tmp_path: Path, family: str) -> None:
    """Each writer's file imports to the same samples, metadata and windowed steps."""
    tracks, metadata = _tracks(), _metadata(family)
    csv_path = te.write_csv(tmp_path / "t.csv", tracks, metadata)
    wcon_path = te.write_wcon(tmp_path / "t.wcon", tracks, metadata)
    for path in (csv_path, wcon_path):
        back, back_metadata = ti.read_tracks(path)
        assert back_metadata == metadata
        for a, b in zip(tracks, back, strict=True):
            assert a.animal == b.animal
            for u, v in ((a.t_s, b.t_s), (a.x_mm, b.x_mm), (a.y_mm, b.y_mm)):
                assert np.array_equal(u, v, equal_nan=True)
    found_csv, from_csv = ti.import_population(csv_path)
    found_wcon, from_wcon = ti.import_population(wcon_path)
    assert found_csv == found_wcon == family
    assert _same_steps(from_csv, from_wcon)


def test_the_example_grades_the_same_as_wcon(tmp_path: Path) -> None:
    """Every animal of the worked example, written as WCON by the command, grades identically."""
    data.set_root(None)
    example = tmp_path / "example"
    assert cli.main(["example", "--out", str(example)]) == 0
    lines = (example / "manifest.txt").read_text().splitlines()
    wcon_lines = []
    for line in lines:
        family, name = line.split()
        out = example / name.replace(".csv", ".wcon")
        assert cli.main(["wcon", "--tracks", str(example / name), "--out", str(out)]) == 0
        wcon_lines.append(f"{family} {out.name}")
    (example / "manifest_wcon.txt").write_text("\n".join(wcon_lines) + "\n")
    gradings = []
    for manifest in ("manifest.txt", "manifest_wcon.txt"):
        out = tmp_path / f"{manifest}.json"
        assert cli.main(["grade", "--manifest", str(example / manifest), "--out", str(out)]) == 0
        graded = json.loads(out.read_text())
        graded.pop("manifest")
        gradings.append(graded)
    assert gradings[0] == gradings[1]


def test_lengths_in_pixels_are_written_in_mm(tmp_path: Path) -> None:
    """A CSV in pixels becomes a WCON file in mm whose samples and lawn are the CSV's, in mm."""
    source = tmp_path / "px.csv"
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["animal", "t_s", "x", "y"])
        for t in range(SECONDS):
            writer.writerow(["w", float(t), 100.0 + 0.5 * t, 40.0])
    sidecar = {
        "units": "px",
        "mm_per_px": 0.01,
        "family": "small_lawn",
        "lawn": {"centre": [100.0, 40.0], "radius": 200.0},
    }
    source.with_name("px.csv.json").write_text(json.dumps(sidecar))
    out = te.csv_to_wcon(source, tmp_path / "px.wcon")
    document = json.loads(out.read_text())
    assert document["units"] == {"t": "s", "x": "mm", "y": "mm"}
    assert document[ti.METADATA_KEY]["lawn"]["radii"] == [2.0, 2.0]
    assert _same_steps(ti.import_population(source)[1], ti.import_population(out)[1])


def _simulated_csv(path: Path, *, family: str = "small_lawn") -> Path:
    labels = ("run", "reversal", "omega", "delta")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["animal", "t_s", "x", "y", "primitive", "seed"])
        for k, (animal, seed) in enumerate((("s1", 11), ("s2", 12))):
            for t in range(SECONDS):
                x = "" if t == 50 + k else repr(0.0005 * t * (k + 1))
                writer.writerow([animal, float(t), x, 0.1, labels[(t // 13) % 4], seed])
    metadata = {"family": family, "analysis_start_s": 0.0, "lawn": LAWNS.get(family, "uniform")}
    path.with_name(path.name + ".json").write_text(json.dumps(metadata))
    return path


def test_simulated_labels_and_seeds_are_carried(tmp_path: Path) -> None:
    """A simulated CSV's labels and seeds survive the WCON route, and it is measured the same."""
    source = _simulated_csv(tmp_path / "sim.csv")
    out = te.csv_to_wcon(source, tmp_path / "sim.wcon", simulated=True)
    written, _ = ti.read_simulated_csv(source)
    back, _ = ti.read_simulated_wcon(out)
    assert [(s.primitives, s.seed) for s in written] == [(s.primitives, s.seed) for s in back]
    model = mm.TrackerModel(jitter_mm=0.01)
    _, from_csv = ti.import_simulated_population(source, model)
    _, from_wcon = ti.import_simulated_population(out, model)
    assert [seed for seed, _ in from_csv] == [seed for seed, _ in from_wcon] == [11, 12]
    assert _same_steps([s for _, s in from_csv], [s for _, s in from_wcon])


def test_without_simulated_the_labels_are_not_carried(tmp_path: Path) -> None:
    """A plain conversion writes positions only, so --simulated grading refuses its output."""
    source = _simulated_csv(tmp_path / "sim.csv")
    out = te.csv_to_wcon(source, tmp_path / "plain.wcon")
    assert "@primitive" not in json.loads(out.read_text())["data"][0]
    with pytest.raises(ValueError, match="@primitive"):
        ti.read_simulated_wcon(out)


def test_a_simulated_local_search_track_needs_no_seed(tmp_path: Path) -> None:
    """A simulated animal without a seed is written without ``@seed`` and read back without one."""
    track = ti.SimulatedTrack(
        ti.ImportedTrack("w", np.arange(600.0), np.zeros(600), np.zeros(600)),
        ("run",) * 600,
        None,
    )
    out = te.write_wcon(tmp_path / "ls.wcon", [track], _metadata("local_search"))
    assert "@seed" not in json.loads(out.read_text())["data"][0]
    assert ti.read_simulated_wcon(out)[0][0].seed is None


def test_a_lawn_flag_column_is_refused_as_wcon(tmp_path: Path) -> None:
    """Membership read from an on_lawn column cannot move to the boundary; nothing is written."""
    source = tmp_path / "flag.csv"
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["animal", "t_s", "x", "y", "on_lawn"])
        for t in range(SECONDS):
            writer.writerow(["w", float(t), 0.0, 0.0, t % 2])
    source.with_name("flag.csv.json").write_text(json.dumps({"family": "small_lawn"}))
    out = tmp_path / "flag.wcon"
    with pytest.raises(SystemExit, match="on-lawn flag"):
        cli.main(["wcon", "--tracks", str(source), "--out", str(out)])
    assert not out.exists()
    back = te.write_csv(tmp_path / "again.csv", *ti.read_csv(source))
    assert ti.read_csv(back)[0][0].on_lawn is not None


def test_a_lawn_family_without_a_lawn_is_refused_as_wcon(tmp_path: Path) -> None:
    """A small-lawn WCON file with no boundary could not be graded, so it is not written."""
    metadata = ti.AssayMetadata("small_lawn", 0.0, None)
    with pytest.raises(ValueError, match="needs a lawn boundary"):
        te.write_wcon(tmp_path / "x.wcon", _tracks(), metadata)


@pytest.mark.parametrize(
    ("tracks", "match"),
    [
        ([], "no tracks"),
        (
            [ti.ImportedTrack("w", np.arange(3.0), np.zeros(3), np.zeros(3))] * 2,
            "appear twice",
        ),
        (
            [ti.ImportedTrack("w", np.array([0.0, 2.0, 1.0]), np.zeros(3), np.zeros(3))],
            "must increase",
        ),
        (
            [ti.ImportedTrack("w", np.arange(3.0), np.array([0.0, math.inf, 0.0]), np.zeros(3))],
            "infinite",
        ),
        (
            [ti.ImportedTrack("w", np.arange(3.0), np.zeros(2), np.zeros(3))],
            "differ in length",
        ),
        (
            [
                ti.ImportedTrack("a", np.arange(3.0), np.zeros(3), np.zeros(3)),
                ti.SimulatedTrack(
                    ti.ImportedTrack("b", np.arange(3.0), np.zeros(3), np.zeros(3)),
                    ("run",) * 3,
                    1,
                ),
            ],
            "all simulated or none",
        ),
    ],
)
def test_tracks_the_importer_would_misread_are_refused(
    tmp_path: Path,
    tracks: list,
    match: str,
) -> None:
    """Both writers refuse what the importer would merge, reorder or misread."""
    for write in (te.write_csv, te.write_wcon):
        with pytest.raises(ValueError, match=match):
            write(tmp_path / "x.out", tracks, _metadata("uniform_lawn"))


def test_only_a_csv_is_converted(tmp_path: Path) -> None:
    """The command reads a CSV and its sidecar; any other file is refused by name."""
    source = tmp_path / "tracks.wcon"
    source.write_text("{}")
    with pytest.raises(SystemExit, match="is not a CSV"):
        cli.main(["wcon", "--tracks", str(source), "--out", str(tmp_path / "out.wcon")])


def test_a_file_that_reads_back_differently_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conversion's own check: a WCON file that does not read back as written is deleted."""
    tracks, metadata = _tracks(), _metadata("uniform_lawn")
    source = te.write_csv(tmp_path / "t.csv", tracks, metadata)
    read_wcon = ti.read_wcon

    def shifted(path: Path) -> tuple[list[ti.ImportedTrack], ti.AssayMetadata]:
        back, found = read_wcon(path)
        first = back[0]
        back[0] = ti.ImportedTrack(first.animal, first.t_s, first.x_mm + 1e-12, first.y_mm)
        return back, found

    monkeypatch.setattr(ti, "read_wcon", shifted)
    out = tmp_path / "t.wcon"
    with pytest.raises(ValueError, match="samples read back differ"):
        te.csv_to_wcon(source, out)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.csv", "t.csv.json"]


def test_a_metadata_change_on_reading_back_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conversion compares the metadata too, and keeps a file already at the output."""
    source = te.write_csv(tmp_path / "t.csv", _tracks(), _metadata("small_lawn"))
    read_wcon = ti.read_wcon

    def moved(path: Path) -> tuple[list[ti.ImportedTrack], ti.AssayMetadata]:
        back, found = read_wcon(path)
        return back, ti.AssayMetadata(found.family, found.analysis_start_s + 1.0, found.lawn)

    monkeypatch.setattr(ti, "read_wcon", moved)
    out = tmp_path / "t.wcon"
    out.write_text("an earlier file")
    with pytest.raises(ValueError, match="metadata read back"):
        te.csv_to_wcon(source, out)
    assert out.read_text() == "an earlier file"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".part")]


def test_a_simulated_csv_reads_back_as_written(tmp_path: Path) -> None:
    """``write_csv`` carries each animal's labels and seed as the simulated route reads them."""
    written, metadata = ti.read_simulated_csv(_simulated_csv(tmp_path / "sim.csv"))
    out = te.write_csv(tmp_path / "again.csv", written, metadata)
    back, back_metadata = ti.read_simulated_csv(out)
    assert back_metadata == metadata
    assert [(s.primitives, s.seed) for s in back] == [(s.primitives, s.seed) for s in written]
    for a, b in zip(written, back, strict=True):
        assert np.array_equal(a.track.x_mm, b.track.x_mm, equal_nan=True)


def test_an_on_lawn_column_reads_back_as_written(tmp_path: Path) -> None:
    """Every flag, missing ones included, is written so that the importer reads it back."""
    flags = np.array([1.0, 0.0, math.nan, 1.0])
    track = ti.ImportedTrack("w", np.arange(4.0), np.zeros(4), np.zeros(4), flags)
    out = te.write_csv(tmp_path / "f.csv", [track], ti.AssayMetadata("small_lawn", 0.0, None))
    back = ti.read_csv(out)[0][0]
    assert back.on_lawn is not None
    assert np.array_equal(back.on_lawn, flags, equal_nan=True)


def test_a_flag_on_some_animals_only_is_refused(tmp_path: Path) -> None:
    """An on_lawn column covers every animal or none, as the importer reads one column for all."""
    flagged = ti.ImportedTrack("a", np.arange(3.0), np.zeros(3), np.zeros(3), np.ones(3))
    plain = ti.ImportedTrack("b", np.arange(3.0), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="every animal or none"):
        te.write_csv(tmp_path / "x.csv", [flagged, plain], _metadata("small_lawn"))


def _one(t: np.ndarray, **kwargs: object) -> ti.ImportedTrack:
    n = len(t)
    return ti.ImportedTrack(kwargs.get("animal", "w"), t, np.zeros(n), np.zeros(n))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("track", "match"),
    [
        (ti.SimulatedTrack(_one(np.arange(3.0)), ("run",) * 2, 1), "one primitive label"),
        (ti.SimulatedTrack(_one(np.arange(3.0)), ("run", "forward", "run"), 1), "unknown"),
        (ti.SimulatedTrack(_one(np.arange(3.0) / 2), ("run",) * 3, 1), "every 1 s"),
        (ti.SimulatedTrack(_one(np.arange(3.0)), ("run",) * 3, 7.5), "an integer"),  # type: ignore[arg-type]
        (ti.SimulatedTrack(_one(np.arange(3.0)), ("run",) * 3, seed=True), "an integer"),
        (_one(np.array([])), "at least one sample"),
        (_one(np.array([0.0, math.nan, 2.0])), "finite time"),
    ],
)
def test_a_track_the_importer_would_refuse_is_not_written(
    tmp_path: Path,
    track: ti.ImportedTrack | ti.SimulatedTrack,
    match: str,
) -> None:
    """Labels, cadence, seed and samples are checked as the importer's routes check them."""
    for write in (te.write_csv, te.write_wcon):
        with pytest.raises(ValueError, match=match):
            write(tmp_path / "x.out", [track], _metadata("uniform_lawn"))


def test_ids_are_compared_as_the_importer_reads_them(tmp_path: Path) -> None:
    """Ids 1 and "1" are one animal to the importer, so writing both is refused."""
    tracks = [_one(np.arange(3.0), animal=1), _one(np.arange(3.0), animal="1")]
    with pytest.raises(ValueError, match="appear twice"):
        te.write_wcon(tmp_path / "x.wcon", tracks, _metadata("uniform_lawn"))


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        (ti.AssayMetadata("uniform_lawn", 0.0, None), "uniform well"),
        (ti.AssayMetadata("uniform_lawn", 0.0, ti.Lawn(radii=(2.0, 2.0))), "uniform well"),
        (ti.AssayMetadata("local_search", 0.0, ti.Lawn()), "no lawn"),
        (ti.AssayMetadata("small_lawn", math.nan, None), "finite"),
        (ti.AssayMetadata("chemotaxis", 0.0, None), "family"),
    ],
)
def test_metadata_that_would_read_back_differently_is_refused(
    tmp_path: Path,
    metadata: ti.AssayMetadata,
    match: str,
) -> None:
    """Each family's lawn has one shape the importer reads back; any other is refused."""
    with pytest.raises(ValueError, match=match):
        te.write_csv(tmp_path / "x.csv", _tracks(200), metadata)


@pytest.mark.parametrize(
    ("out", "match"),
    [("t.csv", "overwrite the source"), ("t.csv.json", "overwrite the source"), ("o.csv", ".wcon")],
)
def test_the_output_never_replaces_the_source_or_reads_as_csv(
    tmp_path: Path,
    out: str,
    match: str,
) -> None:
    """The command refuses an output that is its own source, its sidecar, or a .csv name."""
    source = te.write_csv(tmp_path / "t.csv", _tracks(200), _metadata("uniform_lawn"))
    before = source.read_bytes()
    with pytest.raises(SystemExit, match=match):
        cli.main(["wcon", "--tracks", str(source), "--out", str(tmp_path / out)])
    assert source.read_bytes() == before
    assert not (tmp_path / "o.csv").exists()


def test_a_missing_source_is_named(tmp_path: Path) -> None:
    """A missing CSV is reported as itself, not as its sidecar."""
    source = tmp_path / "absent.csv"
    with pytest.raises(SystemExit, match=r"absent\.csv: no such file"):
        cli.main(["wcon", "--tracks", str(source), "--out", str(tmp_path / "out" / "a.wcon")])
    assert not (tmp_path / "out").exists()


def test_a_refused_csv_leaves_both_files_as_they_were(tmp_path: Path) -> None:
    """Metadata are checked before the CSV is opened, so a refusal changes neither file."""
    path = te.write_csv(tmp_path / "t.csv", _tracks(200), _metadata("uniform_lawn"))
    sidecar = tmp_path / "t.csv.json"
    before = (path.read_bytes(), sidecar.read_bytes())
    with pytest.raises(ValueError, match="uniform well"):
        te.write_csv(path, _tracks(150), ti.AssayMetadata("uniform_lawn", 0.0, None))
    assert (path.read_bytes(), sidecar.read_bytes()) == before


def test_the_staged_file_never_opens_an_existing_file(tmp_path: Path) -> None:
    """A file or link already named like a staging file is neither written nor removed."""
    source = te.write_csv(tmp_path / "t.csv", _tracks(200), _metadata("uniform_lawn"))
    plain = tmp_path / "t.wcon.part"
    plain.write_text("mine")
    target = tmp_path / "target.txt"
    target.write_text("kept")
    (tmp_path / ".t.wcon.part").symlink_to(target)
    out = te.csv_to_wcon(source, tmp_path / "t.wcon")
    assert plain.read_text() == "mine"
    assert target.read_text() == "kept"
    assert (tmp_path / ".t.wcon.part").is_symlink()
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == [".t.wcon.part", "t.csv", "t.csv.json", "t.wcon", "t.wcon.part", "target.txt"]
    assert out.stat().st_mode & 0o777 == target.stat().st_mode & 0o777
