"""The vendored real-worm reference sets, read back as per-animal tracks.

Each set lives under ``data/benchmark/``, derived from its public deposit by
``scripts/analysis/extract_benchmark_references.py``. Its licence, checksum, units and origin are
in ``data/benchmark/PROVENANCE.md``. Positions are stored as integers: a set's stated unit times a
fixed factor, with the integer type's minimum marking a missing value. This module turns them back
into floats with NaN where missing.

- **Scheer & Bargmann 2023**: on-lawn tracks at 1 s, the authors' per-frame flags at 3 frames/s,
  their 10-second labels and run masks, and every animal's genotype, recording date and cohort side.
- **Calhoun 2014**: off-food search tracks at 1 s, in pixels (no scale is published), with per-frame
  pirouette and omega onsets at 3 frames/s.
- **Roberts 2016**: off-food search tracks at 1 s in mm, with the tracker's per-frame
  forward/reverse/pause state and the authors' four-state decoding at 29.97 frames/s.
- **López-Cruz 2019**: reorientations per 2-min bin after food removal, by genotype.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_DATA = _PROJECT_ROOT / "data" / "benchmark"
SCHEER_TRACKS_PATH = BENCHMARK_DATA / "scheer2023_tracks.npz"
SCHEER_SPLIT_PATH = BENCHMARK_DATA / "scheer2023_split.json"
CALHOUN_TRACKS_PATH = BENCHMARK_DATA / "calhoun2014_tracks.npz"
ROBERTS_TRACKS_PATH = BENCHMARK_DATA / "roberts2016_tracks.npz"
LOPEZCRUZ_PATH = BENCHMARK_DATA / "lopezcruz2019_reorientations.json"

SCHEER_FRAMES_PER_SECOND = 3.0
SCHEER_BIN_SECONDS = 10.0
BANDS = "bands"
HELD_OUT = "held_out"
# Origins of a Scheer animal's positions.
ORIGIN_LAWN_CENTRE = 0
ORIGIN_FIRST_POSITION = 1


def dequantise(values: np.ndarray, per_unit: float) -> np.ndarray:
    """Return stored integers as floats in their unit, NaN where the type's minimum marks a gap."""
    missing = values == np.iinfo(values.dtype).min
    out = values.astype(float) / per_unit
    out[missing] = np.nan
    return out


def unpack_flag(packed: np.ndarray, bit: int) -> np.ndarray:
    """Return one boolean flag from a bit-packed uint8 array."""
    return ((packed >> np.uint8(bit)) & np.uint8(1)).astype(bool)


def boundary_distance(x: np.ndarray, y: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Return the signed distance inside a lawn boundary (positive inside), in the radii's units.

    ``radii`` are the boundary's radius at equally spaced angles from +x about the origin of ``x``
    and ``y``, interpolated periodically in angle.
    """
    angle = np.mod(np.arctan2(y, x), 2 * np.pi)
    targets = np.arange(len(radii)) * (2 * np.pi / len(radii))
    radius = np.interp(angle, targets, radii, period=2 * np.pi)
    return radius - np.hypot(x, y)


@dataclass(frozen=True)
class ScheerTracks:
    """Scheer & Bargmann 2023's animals, one row each.

    Positions are in mm at 1 s from the animal's origin: its lawn centre, or, for an arm recorded
    without a lawn boundary, its first valid position. ``frame_flags`` is bit-packed at 3
    frames/s, one bit per name in ``flag_names``; read one with :meth:`flag`. The run masks and
    labels are per 10-second bin. ``labels`` is 1 roaming, 0 dwelling, and -1 where the authors
    masked the bin.
    """

    x: np.ndarray
    y: np.ndarray
    frame_flags: np.ndarray
    flag_names: tuple[str, ...]
    in_lawn_run: np.ndarray
    out_lawn_run: np.ndarray
    labels: np.ndarray
    boundary_radius: np.ndarray
    genotype: np.ndarray
    genotypes: tuple[str, ...]
    date: np.ndarray
    video: np.ndarray
    videos: tuple[str, ...]
    origin: np.ndarray
    fig1_row: np.ndarray
    side: np.ndarray

    def flag(self, name: str) -> np.ndarray:
        """Return one per-frame flag, ``(animals, frames)`` booleans."""
        return unpack_flag(self.frame_flags, self.flag_names.index(name))

    def arm(self, name: str) -> np.ndarray:
        """Return the row indices of one arm, by its deposit key."""
        return np.flatnonzero(self.genotype == self.genotypes.index(name))


def scheer_split(path: Path = SCHEER_SPLIT_PATH) -> dict[int, str]:
    """Return each recording date's side, ``bands`` or ``held_out``."""
    record = json.loads(path.read_text())
    return {int(date): side for date, side in record["sides"].items()}


def load_scheer_tracks(
    path: Path = SCHEER_TRACKS_PATH,
    split_path: Path = SCHEER_SPLIT_PATH,
) -> ScheerTracks:
    """Load Scheer & Bargmann 2023's tracks with their cohort fields and split side."""
    with np.load(path) as store:
        data = {key: store[key] for key in store.files}
    sides = scheer_split(split_path)
    return ScheerTracks(
        x=dequantise(data["x_um"], 1000.0),
        y=dequantise(data["y_um"], 1000.0),
        frame_flags=data["frame_flags"],
        flag_names=tuple(str(n) for n in data["flag_names"]),
        in_lawn_run=data["in_lawn_run"],
        out_lawn_run=data["out_lawn_run"],
        labels=data["labels"],
        boundary_radius=dequantise(data["boundary_um"], 1000.0),
        genotype=data["genotype"],
        genotypes=tuple(str(g) for g in data["genotypes"]),
        date=data["date"],
        video=data["video"],
        videos=tuple(str(v) for v in data["videos"]),
        origin=data["origin"],
        fig1_row=data["fig1_row"],
        side=np.array([sides[int(d)] for d in data["date"]]),
    )


@dataclass(frozen=True)
class SearchTracks:
    """Off-food search tracks, one row per worm, positions at 1 s in ``units``.

    ``frame_flags`` (Calhoun) holds bit-packed per-frame event onsets, one bit per name in
    ``flag_names``. ``state`` and ``viterbi`` (Roberts) hold per-frame codes named by
    ``state_names`` and ``viterbi_names``, -1 outside the worm's recording.
    """

    source: str
    x: np.ndarray
    y: np.ndarray
    units: str
    frames_per_second: float
    cohort: np.ndarray
    frame_flags: np.ndarray | None = None
    flag_names: tuple[str, ...] = ()
    state: np.ndarray | None = None
    state_names: tuple[str, ...] = ()
    viterbi: np.ndarray | None = None
    viterbi_names: tuple[str, ...] = ()

    def flag(self, name: str) -> np.ndarray:
        """Return one per-frame onset flag, ``(worms, frames)`` booleans."""
        if self.frame_flags is None:
            msg = f"{self.source} has no per-frame flags"
            raise ValueError(msg)
        return unpack_flag(self.frame_flags, self.flag_names.index(name))


def load_search_tracks(source: Literal["calhoun2014", "roberts2016"]) -> SearchTracks:
    """Load one set of off-food search tracks."""
    if source == "calhoun2014":
        with np.load(CALHOUN_TRACKS_PATH) as store:
            n = store["x_cpx"].shape[0]
            return SearchTracks(
                source=source,
                x=dequantise(store["x_cpx"], 100.0),
                y=dequantise(store["y_cpx"], 100.0),
                units=str(store["units"]),
                frames_per_second=float(store["frames_per_second"]),
                cohort=np.full(n, ""),
                frame_flags=store["frame_flags"],
                flag_names=tuple(str(f) for f in store["flag_names"]),
            )
    if source == "roberts2016":
        with np.load(ROBERTS_TRACKS_PATH) as store:
            return SearchTracks(
                source=source,
                x=dequantise(store["x_um"], 1000.0),
                y=dequantise(store["y_um"], 1000.0),
                units="mm",
                frames_per_second=float(store["frames_per_second"]),
                cohort=store["cohort"],
                state=store["state"],
                state_names=tuple(str(s) for s in store["tracker_states"]),
                viterbi=store["viterbi"],
                viterbi_names=tuple(str(s) for s in store["viterbi_states"]),
            )
    msg = f"unknown search source {source!r}"
    raise ValueError(msg)


def load_lopezcruz(path: Path = LOPEZCRUZ_PATH) -> dict[str, Any]:
    """Load López-Cruz 2019's reorientations per 2-min bin, every genotype row of the sheet."""
    return json.loads(path.read_text())
