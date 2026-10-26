#!/usr/bin/env python
"""The track importer's round-trip control: export real tracks, re-import them, match the native path.

- **Lawn families:** the band-cohort Scheer animals of the wild-type small-lawn arm and the
  uniform-well arm (split side ``bands`` only), each written as a CSV with its sidecar and as WCON,
  with its own lawn boundary, re-imported and cut by ``track_import.import_population``. Every
  animal's tally, and the population's statistics, must equal the native path's
  (``real_on_lawn`` and ``animal_windows`` on the same positions).
- **Local search:** Roberts 2016's off-food tracks from the detector's calibration cohorts, written
  and re-imported the same way. Each worm's detector onsets and valid seconds must equal the native
  path's. Roberts' tracks are 10 min, shorter than the local-search window, so the comparison is at
  the detector, the input the family's statistics are computed from.

No held-out animal or cohort is written: the script refuses before writing anything.

Usage::

    uv run python scripts/analysis/track_import_check.py --out <roundtrip.json>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import readouts as ro
from quantumnematode.benchmark import references as refs
from quantumnematode.benchmark import track_import as ti

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_grade as bg
import reorientation_detector_calibration as rdc

DESCRIPTION = "The track importer's round trip against the native grading path."
_REPO = Path(__file__).resolve().parents[2]
OUT = (
    _REPO
    / "docs"
    / "experiments"
    / "logbooks"
    / "supporting"
    / "114-track-import"
    / "roundtrip.json"
)
FORMATS = ("csv", "wcon")
TOLERANCE = 1e-12


def calibration_cohorts() -> list[str]:
    """Return the detector's calibration cohorts, by the calibration script's own split."""
    order = [
        rdc.ROBERTS_COHORTS[i]
        for i in np.random.default_rng(rdc.SPLIT_SEED).permutation(len(rdc.ROBERTS_COHORTS))
    ]
    return order[:3]


def _null(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def write_track(  # noqa: PLR0913 - one animal's file: where, what and its metadata
    directory: Path,
    name: str,
    fmt: str,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    """Write one animal as a CSV with its sidecar, or as WCON with the benchmark's metadata."""
    if fmt == "csv":
        path = directory / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["animal", "t_s", "x", "y"])
            for ti_, xi, yi in zip(t, x, y, strict=True):
                writer.writerow([name, repr(float(ti_)), repr(float(xi)), repr(float(yi))])
        path.with_name(path.name + ".json").write_text(json.dumps({"units": "mm", **metadata}))
        return path
    path = directory / f"{name}.wcon"
    document = {
        "units": {"t": "s", "x": "mm", "y": "mm"},
        "data": [
            {
                "id": name,
                "t": [float(v) for v in t],
                "x": [_null(v) for v in x],
                "y": [_null(v) for v in y],
            },
        ],
        ti.METADATA_KEY: metadata,
    }
    path.write_text(json.dumps(document))
    return path


def _lawn_metadata(family: str, radii: np.ndarray) -> dict[str, Any]:
    if family == "uniform_lawn":
        return {"family": family, "analysis_start_s": 0.0, "lawn": "uniform"}
    return {
        "family": family,
        "analysis_start_s": 0.0,
        "lawn": {"centre": [0.0, 0.0], "radii": [float(r) for r in radii]},
    }


def lawn_round_trip(
    tracks: refs.ScheerTracks,
    arm: str,
    family: str,
    clf: Any,  # noqa: ANN401 - the benchmark's classifier
    directory: Path,
) -> dict[str, Any]:
    """Return one arm's native and imported tallies and statistics, and whether they match."""
    rows = tracks.arm(arm)
    rows = rows[tracks.side[rows] == refs.BANDS]
    refuse_held_out(tracks, rows)
    uniform = family == "uniform_lawn"
    native = []
    imported: dict[str, list[np.ndarray]] = {fmt: [] for fmt in FORMATS}
    t = np.arange(tracks.x.shape[1], dtype=float)
    for i in rows:
        lawn = g.real_on_lawn(tracks.x[i], tracks.y[i], tracks.boundary_radius[i], uniform=uniform)
        native.append(g.tally(g.animal_windows(tracks.x[i], tracks.y[i], lawn, clf)))
        metadata = _lawn_metadata(family, tracks.boundary_radius[i])
        for fmt in FORMATS:
            path = write_track(
                directory,
                f"{family}-{i}",
                fmt,
                t,
                tracks.x[i],
                tracks.y[i],
                metadata,
            )
            found, animals = ti.import_population(path)
            assert found == family  # noqa: S101 - the file was written with this family
            steps = animals[0]
            imported[fmt].append(g.tally(g.simulated_windows(steps, clf, uniform=uniform)))
    native_total = np.sum(native, axis=0)
    result: dict[str, Any] = {
        "arm": arm,
        "animals": len(rows),
        "native_statistics": g.lawn_statistics(native_total),
    }
    for fmt in FORMATS:
        tallies = np.vstack(imported[fmt])
        statistics = g.lawn_statistics(tallies.sum(axis=0))
        same_tallies = bool(np.array_equal(tallies, np.vstack(native)))
        same_statistics = all(
            (np.isnan(a) and np.isnan(b)) or abs(a - b) <= TOLERANCE
            for a, b in (
                (statistics[k], result["native_statistics"][k]) for k in result["native_statistics"]
            )
        )
        result[fmt] = {
            "statistics": statistics,
            "tallies_equal": same_tallies,
            "statistics_equal": same_statistics,
        }
    return result


def search_round_trip(directory: Path) -> dict[str, Any]:
    """Return Roberts' calibration worms' native and imported detector output, and whether they match."""
    tracks = refs.load_search_tracks("roberts2016")
    cohorts = calibration_cohorts()
    refuse_held_out_cohorts(cohorts)
    rows = [i for i, c in enumerate(tracks.cohort) if c in cohorts]
    t = np.arange(tracks.x.shape[1], dtype=float)
    detector = g.OFF_FOOD_DETECTOR
    matches = dict.fromkeys(FORMATS, 0)
    onsets = 0
    for i in rows:
        native_onsets = ro.reversal_onsets(tracks.x[i], tracks.y[i], detector)
        native_valid = ro.valid_seconds(tracks.x[i], tracks.y[i])
        onsets += len(native_onsets)
        metadata = {"family": "local_search", "analysis_start_s": 0.0}
        for fmt in FORMATS:
            path = write_track(directory, f"search-{i}", fmt, t, tracks.x[i], tracks.y[i], metadata)
            imported, _ = ti.read_tracks(path)
            x, y, _ = ti.resample_1s(imported[0])
            found = ro.reversal_onsets(x, y, detector)
            valid = ro.valid_seconds(x, y)
            if np.array_equal(found, native_onsets) and np.array_equal(valid, native_valid):
                matches[fmt] += 1
    return {
        "source": "roberts2016",
        "cohorts": cohorts,
        "worms": len(rows),
        "native_onsets": onsets,
        **{
            fmt: {"worms_matching": matches[fmt], "equal": matches[fmt] == len(rows)}
            for fmt in FORMATS
        },
    }


def refuse_held_out(tracks: refs.ScheerTracks, rows: np.ndarray) -> None:
    """Stop before anything is written if a row about to be exported is not on the band side."""
    held = [int(i) for i in rows if tracks.side[i] != refs.BANDS]
    if held:
        msg = f"rows {held[:5]} are held out; nothing is written"
        raise RuntimeError(msg)


def refuse_held_out_cohorts(cohorts: list[str]) -> None:
    """Stop if a Roberts cohort outside the detector's calibration split is selected."""
    outside = sorted(set(cohorts) - set(calibration_cohorts()))
    if outside:
        msg = f"cohorts {outside} are held out; nothing is written"
        raise RuntimeError(msg)


def main(argv: list[str] | None = None) -> int:
    """CLI: run both round trips and write the record."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    bands = bg.load_bands()
    clf = bg.classifier(bands)
    tracks = refs.load_scheer_tracks()
    arms = {family: bands["arms"][family]["arm"] for family in ("small_lawn", "uniform_lawn")}
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        record: dict[str, Any] = {
            "description": DESCRIPTION,
            "lawn": {
                family: lawn_round_trip(tracks, arm, family, clf, directory)
                for family, arm in arms.items()
            },
            "local_search": search_round_trip(directory),
        }
    passed = all(
        r[fmt]["tallies_equal"] and r[fmt]["statistics_equal"]
        for r in record["lawn"].values()
        for fmt in FORMATS
    ) and all(record["local_search"][fmt]["equal"] for fmt in FORMATS)
    record["passes"] = passed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bg._finite(record), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passes": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
