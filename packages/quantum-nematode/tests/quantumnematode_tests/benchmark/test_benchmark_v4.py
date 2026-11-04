"""B v4: the on-lawn activity family graded from positions.

Covers the benchmark-grading spec's scenarios "A worm that does not run forward fails", "A worm
that never pauses fails", "v3 is unchanged" and "Imported tracks carry the activity family", and the
benchmark-controls spec's "v4 becomes the default" (the default chain). The v4 bands here stand in
for the vendored file: v3's, with stand-in activity bands at the planning probe's real values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import measurement as mm
from quantumnematode.benchmark import references as refs
from quantumnematode.report.behaviour import BehaviourStep

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import benchmark_bands as bb  # noqa: E402  # pyright: ignore[reportMissingImports]
import benchmark_grade as bg  # noqa: E402  # pyright: ignore[reportMissingImports]

# Stand-in activity bands at a 64-animal scale, centred on the planning probe's real values.
_ACTIVITY = {
    "pause_share": {"centre": 0.482, "scale": 0.03, "inflated": False},
    "flip_fraction": {"centre": 0.132, "scale": 0.012, "inflated": False},
    "onset_rate_per_min": {"centre": 2.28, "scale": 0.25, "inflated": False},
}


def _undecided() -> dict[str, Any]:
    v3 = bg.load_bands_v3()
    bands = json.loads(json.dumps(v3["bands"]))
    for size in bands.values():
        size["activity"] = _ACTIVITY
    return {
        **v3,
        "version": 4,
        "activity_graded": None,
        "families": None,
        "fit_families": list(bb.V4_FIT_FAMILIES),
        "bars": {**v3["bars"], "wild_type": None},
        "weights": None,
        "fit_weights": None,
        "bands": bands,
        "activity_kappa": dict.fromkeys(g.ACTIVITY_STATISTICS, 1.0),
    }


def _bands_v4(*, onset_graded: bool = True) -> dict[str, Any]:
    return bb.decide_onset(_undecided(), onset_graded)


def _track(xs: np.ndarray, *, on_lawn: bool = True) -> list[BehaviourStep]:
    return [
        BehaviourStep(
            step=i,
            x=float(x),
            y=0.0,
            heading_rad=0.0,
            concentration=0.0,
            dc_dt=0.0,
            grad_dir=0.0,
            grad_strength=0.0,
            on_lawn=on_lawn,
            primitive="run",
        )
        for i, x in enumerate(xs)
    ]


def _reverses_from_standstill(n: int) -> np.ndarray:
    """Return a track that stays put, then backs 0.07 mm/s for 2 s, every 10 s: no forward runs."""
    step = np.zeros(n - 1)
    for start in range(5, n - 1, 10):
        step[start : start + 2] = -0.07
    return np.concatenate(([0.0], np.cumsum(step)))


def test_the_pause_floor_is_the_instruments() -> None:
    """The floor derived from the real tracker's still-worm readings is the graded 0.02 mm."""
    still = bb.still_worm_displacements()
    assert g.pause_floor(still) == g.PAUSE_FLOOR_MM == 0.02
    assert np.mean(still[np.isfinite(still)] < g.PAUSE_FLOOR_MM) >= g.PAUSE_FLOOR_COVERAGE


def test_the_tally_counts_on_lawn_seconds_only() -> None:
    """A second counts only when both its samples are on the lawn; flips need on-lawn pairs."""
    x = np.array([0.0, 0.0, 0.05, 0.10, 0.05, 0.0, 0.0])
    y = np.zeros_like(x)
    on = np.array([True, True, True, True, True, False, False])
    tally = dict(zip(g.ACTIVITY_FIELDS, g.activity_tally(x, y, on), strict=True))
    # Four on-lawn seconds: one still, two forward, one back; the back step after forward flips.
    assert tally["seconds"] == 4
    assert tally["pauses"] == 1
    assert (tally["flips"], tally["pairs"]) == (1, 2)


def test_a_worm_that_does_not_run_forward_fails() -> None:
    """Covers "A worm that does not run forward fails": reversals from standstill flip nothing."""
    populations = {"small_lawn": [_track(_reverses_from_standstill(g.LAWN_SAMPLES))] * 64}
    graded = bg.grade_populations_v4(populations, None, _bands_v4())
    flips = graded["families"]["activity"]["flip_fraction"]
    assert flips["value"] == 0.0
    assert not flips["passes"]


def test_a_worm_that_never_pauses_fails() -> None:
    """Covers "A worm that never pauses fails": a steady walk's pause share is below the band."""
    walk = np.arange(g.LAWN_SAMPLES) * 0.05
    graded = bg.grade_populations_v4({"small_lawn": [_track(walk)] * 64}, None, _bands_v4())
    pauses = graded["families"]["activity"]["pause_share"]
    assert pauses["value"] == 0.0
    assert not pauses["passes"]


def test_v3_is_unchanged() -> None:
    """Covers "v3 is unchanged": v3's grade has no activity family, and v4's v3 families equal it.

    Every v1-v3 statistic, D and presence reading was compared byte for byte against a baseline
    from the code before v4 (`speedup_identity_check.py check`); this pins the surface it reads.
    """
    populations = {"small_lawn": [_track(np.arange(g.LAWN_SAMPLES) * 0.01)] * 64}
    v3 = bg.grade_populations_v3(populations, None, bg.load_bands_v3())
    v4 = bg.grade_populations_v4(populations, None, _bands_v4())
    assert v3["version"] == 3
    assert "activity" not in v3["families"]
    assert v4["version"] == 4
    for family in ("small_lawn", "kinematics"):
        assert bg._finite(v4["families"][family]) == bg._finite(v3["families"][family])
    assert bb.v3_statistics() == 21


def test_the_onset_decision_sets_the_families_and_bar() -> None:
    """The decision fixes 24 statistics at 3.0781 or 23 at 3.0654, and is recorded once."""
    graded, reported = _bands_v4(onset_graded=True), _bands_v4(onset_graded=False)
    assert graded["families"]["activity"] == list(g.ACTIVITY_STATISTICS)
    assert graded["bars"]["wild_type"] == pytest.approx(3.0781, abs=1e-4)
    assert bb.v4_statistics(onset_graded=True) == 24
    assert "onset_rate_per_min" not in reported["families"]["activity"]
    assert reported["bars"]["wild_type"] == pytest.approx(3.0654, abs=1e-4)
    assert bb.v4_statistics(onset_graded=False) == 23
    assert set(graded["fit_families"]) == {"small_lawn", "uniform_lawn", "kinematics", "activity"}
    with pytest.raises(ValueError, match="not yet decided"):
        bb.decide_onset(graded, onset_graded=True)


def test_an_ungraded_onset_rate_is_reported() -> None:
    """With the onset rate reported, it is read beside the grade and not among the graded."""
    populations = {"small_lawn": [_track(_reverses_from_standstill(g.LAWN_SAMPLES))] * 64}
    graded = bg.grade_populations_v4(populations, None, _bands_v4(onset_graded=False))
    assert "onset_rate_per_min" not in graded["families"]["activity"]
    assert "onset_rate_per_min" in graded["activity_reported"]


def test_undecided_bands_are_refused() -> None:
    """No v4 grade reads bands whose onset decision is not recorded."""
    populations = {"small_lawn": [_track(np.zeros(g.LAWN_SAMPLES))] * 64}
    with pytest.raises(ValueError, match="onset rate is graded"):
        bg.grade_populations_v4(populations, None, _undecided())


def test_imported_tracks_carry_the_activity_family(tmp_path: Path) -> None:
    """Covers "Imported tracks carry the activity family": a CSV round trip keeps the tally."""
    import track_import_check as tic  # pyright: ignore[reportMissingImports]
    from quantumnematode.benchmark import track_import as ti

    tracks = refs.load_scheer_tracks()
    row = int(bb.band_rows(tracks, bb.WILD_TYPE)[0])
    native = g.activity_tally(
        tracks.x[row],
        tracks.y[row],
        g.real_on_lawn(tracks.x[row], tracks.y[row], tracks.boundary_radius[row]),
    )
    t = np.arange(tracks.x.shape[1], dtype=float)
    path = tic.write_track(
        tmp_path,
        "animal",
        "csv",
        t,
        tracks.x[row],
        tracks.y[row],
        tic._lawn_metadata("small_lawn", tracks.boundary_radius[row]),
    )
    _, animals = ti.import_population(path)
    steps = animals[0]
    imported = g.activity_tally(
        np.array([s.x for s in steps], dtype=float),
        np.array([s.y for s in steps], dtype=float),
        np.array([bool(s.on_lawn) for s in steps]),
    )
    np.testing.assert_array_equal(imported, native)


def test_v4_is_the_default_only_once_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers "v4 becomes the default": a VALID verdict for v4's very bands makes it the default."""
    bands = tmp_path / "bands_v4.json"
    bands.write_text(json.dumps(_bands_v4(), indent=2) + "\n")
    monkeypatch.setattr(bg, "BANDS_V4_PATH", bands)
    controls = tmp_path / "controls_v4.json"
    monkeypatch.setattr(bg, "CONTROLS_V4_PATH", controls)
    assert bg.v4_is_valid() is False
    controls.write_text(json.dumps({"verdict": "VOID", "bands_sha256": mm.file_sha256(bands)}))
    assert bg.v4_is_valid() is False
    controls.write_text(json.dumps({"verdict": "VALID", "bands_sha256": mm.file_sha256(bands)}))
    assert bg.v4_is_valid() is True
    bands.write_text(bands.read_text() + " ")
    assert bg.v4_is_valid() is False


def test_the_v4_build_refuses_before_part_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no committed launch record the v4 build, and the onset decision, write nothing."""
    monkeypatch.setattr(bb, "V4_LAUNCH_PATH", _REPO / "no-such-dir" / "launch.md")
    out, record = tmp_path / "bands.json", tmp_path / "record.json"
    for extra in ([], ["--onset", "graded"]):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "benchmark_bands.py",
                "--version",
                "v4",
                "--out",
                str(out),
                "--record",
                str(record),
                *extra,
            ],
        )
        assert bb.main() == 1
    assert not out.exists()
    assert not record.exists()
