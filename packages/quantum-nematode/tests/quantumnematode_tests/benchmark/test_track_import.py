"""Track import: formats, resampling, lawn membership, the CLI, and grading without the simulator.

Covers the track-import spec's scenarios "A skeleton is reduced to its midpoint", "A gap is missing
data", "Pixels without a scale", "A circular lawn", "A held-out animal is refused" and "Grading
with the simulator disabled".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import references as refs
from quantumnematode.benchmark import track_import as ti

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))


def _csv(path: Path, rows: list[tuple], metadata: dict, *, on_lawn: bool = False) -> Path:
    header = "animal,t_s,x,y" + (",on_lawn" if on_lawn else "")
    path.write_text(header + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n")
    path.with_name(path.name + ".json").write_text(json.dumps(metadata))
    return path


def _wcon(path: Path, data: object, metadata: dict, **extra: object) -> Path:
    document = {"units": {"t": "s", "x": "mm", "y": "mm"}, "data": data, ti.METADATA_KEY: metadata}
    path.write_text(json.dumps({**document, **extra}))
    return path


_SEARCH = {"family": "local_search", "analysis_start_s": 0.0}


def test_a_skeleton_is_reduced_to_its_midpoint(tmp_path: Path) -> None:
    """Each time's middle point by index is the track's position."""
    data = {
        "id": "w",
        "t": [0, 1],
        "x": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        "y": [[0, 0, 0], [1, 1, 1]],
    }
    tracks, _ = ti.read_wcon(_wcon(tmp_path / "a.wcon", data, _SEARCH))
    assert tracks[0].x_mm.tolist() == [1.0, 4.0]
    assert tracks[0].y_mm.tolist() == [0.0, 1.0]


def test_a_single_time_skeleton_is_one_sample(tmp_path: Path) -> None:
    """A skeleton at one time, with the time listed or not, gives that time's midpoint."""
    listed = {"id": "a", "t": [0], "x": [[0.0, 1.0, 2.0]], "y": [[0.0, 0.0, 0.0]]}
    single = {"id": "b", "t": 1, "x": [3.0, 4.0, 5.0], "y": [1.0, 1.0, 1.0]}
    tracks, _ = ti.read_wcon(_wcon(tmp_path / "s.wcon", [listed, single], _SEARCH))
    by_id = {track.animal: track for track in tracks}
    assert by_id["a"].x_mm.tolist() == [1.0]
    assert by_id["b"].x_mm.tolist() == [4.0]
    assert by_id["b"].y_mm.tolist() == [1.0]


def test_wcon_offsets_extensions_and_merging(tmp_path: Path) -> None:
    """Origin offsets are added, records sharing an id merge in time order, @ keys are ignored."""
    data = [
        {"id": "w", "t": [2, 3], "x": [1.0, 1.0], "y": [0.0, 0.0], "ox": [10.0, 10.0]},
        {"id": "w", "t": [0, 1], "x": [0.0, 0.0], "y": [0.0, None], "@lab": {"note": 1}},
    ]
    path = _wcon(tmp_path / "b.wcon", data, _SEARCH, **{"@lab": {"rig": 2}})
    tracks, _ = ti.read_wcon(path)
    assert tracks[0].t_s.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert tracks[0].x_mm.tolist() == [0.0, 0.0, 11.0, 11.0]
    assert np.isnan(tracks[0].y_mm[1])


def test_wcon_keys_outside_the_core_are_refused(tmp_path: Path) -> None:
    """A top-level key that is neither core nor an @ extension is named and refused."""
    path = _wcon(
        tmp_path / "c.wcon",
        {"id": "w", "t": [0], "x": [0.0], "y": [0.0]},
        _SEARCH,
        files={},
    )
    with pytest.raises(ValueError, match="files"):
        ti.read_wcon(path)


def test_a_gap_is_missing_data(tmp_path: Path) -> None:
    """Seconds between samples more than 1 s apart are NaN; a 1-s track passes unchanged."""
    rows = [("w", t, float(t), 0.0) for t in (0, 1, 2, 5, 6)]
    tracks, _ = ti.read_csv(_csv(tmp_path / "d.csv", rows, _SEARCH))
    x, _, _ = ti.resample_1s(tracks[0])
    assert x[:3].tolist() == [0.0, 1.0, 2.0]
    assert np.isnan(x[3])
    assert np.isnan(x[4])
    assert x[5:].tolist() == [5.0, 6.0]


def test_a_finer_track_is_interpolated(tmp_path: Path) -> None:
    """Samples at 3 per second are interpolated onto whole seconds."""
    rows = [("w", t / 3, t / 3, 0.0) for t in range(7)]
    tracks, _ = ti.read_csv(_csv(tmp_path / "e.csv", rows, _SEARCH))
    x, _, _ = ti.resample_1s(tracks[0])
    assert x == pytest.approx([0.0, 1.0, 2.0])


def test_pixels_without_a_scale_are_refused(tmp_path: Path) -> None:
    """Pixel units need mm_per_px; with it, positions are scaled."""
    rows = [("w", 0, 10.0, 0.0), ("w", 1, 20.0, 0.0)]
    with pytest.raises(ValueError, match="mm_per_px"):
        ti.read_csv(_csv(tmp_path / "f.csv", rows, {**_SEARCH, "units": "px"}))
    tracks, _ = ti.read_csv(
        _csv(tmp_path / "g.csv", rows, {**_SEARCH, "units": "px", "mm_per_px": 0.01}),
    )
    assert tracks[0].x_mm.tolist() == pytest.approx([0.1, 0.2])


def test_a_circular_lawn() -> None:
    """A sample is on a circular lawn exactly when it lies inside the circle about its centre."""
    lawn = ti.parse_metadata(
        {"family": "small_lawn", "lawn": {"centre": [1.0, 1.0], "radius": 2.0}},
    ).lawn
    x = np.array([1.0, 2.9, 3.1, 1.0])
    y = np.array([1.0, 1.0, 1.0, -1.5])
    assert ti.lawn_membership(x, y, None, lawn).tolist() == [True, True, False, False]


def test_a_polar_lawn_follows_scheers_rule() -> None:
    """A polar boundary gives the same membership as the rule Scheer's tracks are cut by."""
    radii = np.array([2.0, 3.0, 2.0, 1.0])
    lawn = ti.Lawn(centre=(0.0, 0.0), radii=tuple(radii))
    x, y = np.array([0.0, 2.5, 0.0, -1.5]), np.array([2.5, 0.0, -0.5, 0.0])
    expected = refs.boundary_distance(x, y, radii) > 0
    assert ti.lawn_membership(x, y, None, lawn).tolist() == expected.tolist()


def test_the_column_takes_precedence_and_uniform_is_all_on() -> None:
    """An on-lawn column overrides a boundary; a uniform well puts every sample on the lawn."""
    lawn = ti.Lawn(centre=(0.0, 0.0), radii=(1.0, 1.0))
    x = y = np.zeros(3)
    assert ti.lawn_membership(x, y, np.array([0.0, 1.0, np.nan]), lawn).tolist() == [
        False,
        True,
        False,
    ]
    assert ti.lawn_membership(x, y, None, ti.Lawn()).all()


def test_a_held_out_animal_is_refused() -> None:
    """The round trip stops before writing if any row to export is held out."""
    import track_import_check as tic  # pyright: ignore[reportMissingImports]

    tracks = SimpleNamespace(side=np.array([refs.BANDS, refs.HELD_OUT]))
    with pytest.raises(RuntimeError, match="held out"):
        tic.refuse_held_out(tracks, np.array([0, 1]))  # pyright: ignore[reportArgumentType]
    tic.refuse_held_out(tracks, np.array([0]))  # pyright: ignore[reportArgumentType]


def _lawn_track(tmp_path: Path, name: str) -> Path:
    rng = np.random.default_rng(0)
    steps = np.cumsum(rng.normal(0, 0.02, size=(g.LAWN_SAMPLES, 2)), axis=0)
    rows = [("w", t, steps[t, 0], steps[t, 1]) for t in range(g.LAWN_SAMPLES)]
    metadata = {
        "family": "small_lawn",
        "analysis_start_s": 0,
        "lawn": {"centre": [0, 0], "radius": 3.0},
    }
    return _csv(tmp_path / name, rows, metadata)


def test_grading_with_the_simulator_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the evaluator and the simulation function raising, imported tracks are still graded."""
    import benchmark_grade as bg  # pyright: ignore[reportMissingImports]
    import npe_recovery as nr  # pyright: ignore[reportMissingImports]
    from quantumnematode.evaluation.batched import BatchedEvaluator

    def refuse(*_args: object, **_kwargs: object) -> None:
        msg = "the simulator was called"
        raise AssertionError(msg)

    monkeypatch.setattr(BatchedEvaluator, "run", refuse)
    monkeypatch.setattr(nr, "simulate", refuse)
    path = _lawn_track(tmp_path, "h.csv")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"small_lawn {path}\n")
    populations = bg.track_populations(manifest)
    graded = bg.grade_populations(populations, bg.load_bands())
    assert set(graded["families"]) == {"small_lawn"}


def test_a_manifest_family_must_match_the_file(tmp_path: Path) -> None:
    """A line naming a family the file does not declare is refused."""
    import benchmark_grade as bg  # pyright: ignore[reportMissingImports]

    path = _lawn_track(tmp_path, "i.csv")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"uniform_lawn {path}\n")
    with pytest.raises(ValueError, match="declares family"):
        bg.track_populations(manifest)


def test_importing_the_grader_in_a_fresh_interpreter() -> None:
    """The grader and the importer load no torch and no brain, agent or environment module."""
    import subprocess

    code = (
        "import sys\n"
        "import quantumnematode.benchmark.grading, quantumnematode.benchmark.track_import\n"
        "heavy = [m for m in sys.modules if m == 'torch' or m.startswith(('torch.', "
        "'quantumnematode.brain', 'quantumnematode.agent', 'quantumnematode.env'))]\n"
        "print(len(heavy))\n"
        "from quantumnematode.validation import run_step_input\n"
        "print(run_step_input.__name__)\n"
    )
    out = subprocess.run(  # noqa: S603 - this interpreter, fixed code
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out == ["0", "run_step_input"]
