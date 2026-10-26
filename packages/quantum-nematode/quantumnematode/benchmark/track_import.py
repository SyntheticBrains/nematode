"""Import worm tracks from outside this repository and cut them for the benchmark's grader.

Two formats are read:

- **CSV**, one row per sample, with columns ``animal``, ``t_s``, ``x``, ``y`` and an optional
  ``on_lawn`` (0/1), and a sidecar ``<file>.json`` holding the metadata;
- **WCON 1.0**'s core: ``units``, ``data`` (one record or a list; records sharing an ``id`` are
  merged in time order) and the metadata in a top-level ``@nematode_benchmark`` object, the form
  WCON gives custom fields. A skeleton per time is reduced to its middle point, since the grader
  reads the midbody; origin offsets (``ox``, ``oy``) are added; ``null`` is a missing sample.

The metadata name the ``family`` (``small_lawn``, ``uniform_lawn`` or ``local_search``), the
``analysis_start_s`` at which the family's window begins, the length ``units`` and the ``lawn``:
``{"centre": [x, y], "radius": r}``, ``{"centre": [x, y], "radii": [...]}`` at equally spaced angles
from +x, or ``"uniform"``. Each track is resampled to 1 s, its lawn membership marked by the column,
the boundary or the uniform well, and converted to the per-second steps the grader reads, cut to its
family's window.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark.references import boundary_distance
from quantumnematode.report.behaviour import BehaviourStep

if TYPE_CHECKING:
    from pathlib import Path

LENGTH_MM: dict[str, float] = {
    "mm": 1.0,
    "millimetre": 1.0,
    "millimeter": 1.0,
    "um": 1e-3,
    "µm": 1e-3,
    "μm": 1e-3,
    "micron": 1e-3,
    "microns": 1e-3,
    "cm": 10.0,
    "m": 1000.0,
    "inch": 25.4,
    "in": 25.4,
}
PIXEL_UNITS = ("px", "pixels")
TIME_S: dict[str, float] = {
    "s": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}
FAMILY_SAMPLES = {
    "small_lawn": g.LAWN_SAMPLES,
    "uniform_lawn": g.LAWN_SAMPLES,
    "local_search": g.SEARCH_SAMPLES,
}
WCON_CORE = {"units", "data", "metadata"}
WCON_RECORD = {"id", "t", "x", "y", "ox", "oy", "cx", "cy", "head", "ventral", "px", "py", "ptail"}
MAX_GAP_S = 1.0
METADATA_KEY = "@nematode_benchmark"
# Times closer than this are the same instant; the on-lawn flag reads as on above HALF.
TIME_EPS = 1e-9
HALF = 0.5


@dataclass(frozen=True)
class Lawn:
    """A lawn boundary about ``centre``, or a uniform well when ``radii`` is None."""

    centre: tuple[float, float] = (0.0, 0.0)
    radii: tuple[float, ...] | None = None

    @property
    def uniform(self) -> bool:
        """Return whether the well has no boundary: every sample is on the lawn."""
        return self.radii is None


@dataclass(frozen=True)
class AssayMetadata:
    """What a set of tracks needs to be graded: its family, its window's start and its lawn."""

    family: str
    analysis_start_s: float
    lawn: Lawn | None


@dataclass(frozen=True)
class ImportedTrack:
    """One animal's samples: times in s, positions in mm, and an optional on-lawn flag."""

    animal: str
    t_s: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    on_lawn: np.ndarray | None = None


def _length_scale(units: str, mm_per_px: float | None, *, pixels_allowed: bool) -> float:
    """Return the factor that converts ``units`` to mm."""
    if units in PIXEL_UNITS and pixels_allowed:
        if mm_per_px is None or not mm_per_px > 0:
            msg = f"units {units!r} need a positive mm_per_px"
            raise ValueError(msg)
        return float(mm_per_px)
    if units not in LENGTH_MM:
        msg = f"unknown length unit {units!r}; expected one of {sorted(LENGTH_MM)}"
        raise ValueError(msg)
    return LENGTH_MM[units]


def parse_metadata(raw: dict[str, Any]) -> AssayMetadata:
    """Return the assay metadata from a sidecar or a WCON ``@nematode_benchmark`` object."""
    family = raw.get("family")
    if family not in FAMILY_SAMPLES:
        msg = f"family {family!r}; expected one of {sorted(FAMILY_SAMPLES)}"
        raise ValueError(msg)
    start = float(raw.get("analysis_start_s", 0.0))
    lawn_raw = raw.get("lawn")
    if family == "local_search":
        lawn = None
    elif lawn_raw == "uniform" or family == "uniform_lawn":
        lawn = Lawn()
    elif isinstance(lawn_raw, dict):
        centre = tuple(float(v) for v in lawn_raw.get("centre", (0.0, 0.0)))
        if "radius" in lawn_raw:
            radii = (float(lawn_raw["radius"]),) * 2
        elif "radii" in lawn_raw:
            radii = tuple(float(v) for v in lawn_raw["radii"])
        else:
            msg = "a lawn needs a radius or radii"
            raise ValueError(msg)
        lawn = Lawn(centre=(centre[0], centre[1]), radii=radii)
    else:
        lawn = None
    return AssayMetadata(family=family, analysis_start_s=start, lawn=lawn)


def _scale_lawn(lawn: Lawn | None, scale: float) -> Lawn | None:
    if lawn is None or lawn.uniform:
        return lawn
    assert lawn.radii is not None  # noqa: S101 - a non-uniform lawn has radii
    return Lawn(
        centre=(lawn.centre[0] * scale, lawn.centre[1] * scale),
        radii=tuple(r * scale for r in lawn.radii),
    )


def read_csv(path: Path) -> tuple[list[ImportedTrack], AssayMetadata]:
    """Read a CSV of samples and its sidecar ``<file>.json``; return the tracks in mm and s."""
    sidecar = json.loads(path.with_name(path.name + ".json").read_text())
    scale = _length_scale(
        str(sidecar.get("units", "mm")),
        sidecar.get("mm_per_px"),
        pixels_allowed=True,
    )
    metadata = parse_metadata(sidecar)
    metadata = AssayMetadata(
        metadata.family,
        metadata.analysis_start_s,
        _scale_lawn(metadata.lawn, scale),
    )
    rows: dict[str, list[tuple[float, float, float, float | None]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = {"animal", "t_s", "x", "y"} - fields
        if missing:
            msg = f"{path} lacks columns {sorted(missing)}"
            raise ValueError(msg)
        has_lawn = "on_lawn" in fields
        for row in reader:
            rows.setdefault(row["animal"], []).append(
                (
                    float(row["t_s"]),
                    _number(row["x"]),
                    _number(row["y"]),
                    float(row["on_lawn"]) if has_lawn and row["on_lawn"] != "" else None,
                ),
            )
    tracks = []
    for animal, samples in rows.items():
        samples.sort(key=lambda s: s[0])
        t, x, y, on = zip(*samples, strict=True)
        tracks.append(
            ImportedTrack(
                animal=animal,
                t_s=np.asarray(t, dtype=float),
                x_mm=np.asarray(x, dtype=float) * scale,
                y_mm=np.asarray(y, dtype=float) * scale,
                on_lawn=np.asarray([math.nan if v is None else v for v in on], dtype=float)
                if has_lawn
                else None,
            ),
        )
    return tracks, metadata


def _number(text: str) -> float:
    return math.nan if text.strip().lower() in ("", "nan", "null") else float(text)


def _midpoint(value: Any) -> float:  # noqa: ANN401 - a WCON coordinate: a number, list or null
    """Return a time's coordinate: the value itself, or a skeleton's middle point by index."""
    if value is None:
        return math.nan
    if isinstance(value, list):
        if not value:
            return math.nan
        middle = value[len(value) // 2]
        return math.nan if middle is None else float(middle)
    return float(value)


def read_wcon(path: Path) -> tuple[list[ImportedTrack], AssayMetadata]:
    """Read WCON 1.0's core and the ``@nematode_benchmark`` metadata; return tracks in mm and s."""
    document = json.loads(path.read_text())
    extra = {k for k in document if k not in WCON_CORE and not k.startswith("@")}
    if extra:
        msg = f"{path}: WCON keys outside the core are not read: {sorted(extra)}"
        raise ValueError(msg)
    if METADATA_KEY not in document:
        msg = f"{path} has no {METADATA_KEY} object"
        raise ValueError(msg)
    units = document["units"]
    t_scale = TIME_S.get(str(units.get("t", "s")))
    if t_scale is None:
        msg = f"unknown time unit {units.get('t')!r}; expected one of {sorted(TIME_S)}"
        raise ValueError(msg)
    x_scale = _length_scale(str(units.get("x", "mm")), None, pixels_allowed=False)
    y_scale = _length_scale(str(units.get("y", "mm")), None, pixels_allowed=False)
    metadata = parse_metadata(document[METADATA_KEY])
    metadata = AssayMetadata(
        metadata.family,
        metadata.analysis_start_s,
        _scale_lawn(metadata.lawn, x_scale),
    )
    records = document["data"] if isinstance(document["data"], list) else [document["data"]]
    merged: dict[str, dict[float, tuple[float, float]]] = {}
    for record in records:
        unknown = {k for k in record if k not in WCON_RECORD and not k.startswith("@")}
        if unknown:
            msg = f"{path}: data fields outside the core are not read: {sorted(unknown)}"
            raise ValueError(msg)
        # With a list of times, x and y hold one coordinate (a number or a skeleton) per time; with
        # a single time, the record's coordinate is that time's.
        timed = isinstance(record["t"], list)
        t = record["t"] if timed else [record["t"]]
        x = record["x"] if timed and isinstance(record["x"], list) else [record["x"]]
        y = record["y"] if timed and isinstance(record["y"], list) else [record["y"]]
        ox = record.get("ox", [0.0] * len(t))
        oy = record.get("oy", [0.0] * len(t))
        ox = ox if isinstance(ox, list) else [ox] * len(t)
        oy = oy if isinstance(oy, list) else [oy] * len(t)
        samples = merged.setdefault(str(record["id"]), {})
        for ti, xi, yi, oxi, oyi in zip(t, x, y, ox, oy, strict=True):
            samples[float(ti) * t_scale] = (
                (_midpoint(xi) + float(oxi or 0.0)) * x_scale,
                (_midpoint(yi) + float(oyi or 0.0)) * y_scale,
            )
    tracks = []
    for animal, samples in merged.items():
        times = sorted(samples)
        tracks.append(
            ImportedTrack(
                animal=animal,
                t_s=np.asarray(times, dtype=float),
                x_mm=np.asarray([samples[t][0] for t in times], dtype=float),
                y_mm=np.asarray([samples[t][1] for t in times], dtype=float),
            ),
        )
    return tracks, metadata


def read_tracks(path: Path) -> tuple[list[ImportedTrack], AssayMetadata]:
    """Read a CSV (with its sidecar) or a WCON file, by its suffix."""
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    return read_wcon(path)


def resample_1s(track: ImportedTrack) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Put a track on 1-s steps from its first sample.

    Positions are interpolated linearly where the neighbouring samples lie at most 1 s apart, and
    are NaN otherwise, so the grader's missing-sample rule excludes any window touching a gap. The
    on-lawn flag takes the nearest sample's. A track already at 1 s passes through unchanged.
    """
    t = np.asarray(track.t_s, dtype=float)
    grid = t[0] + np.arange(math.floor(t[-1] - t[0] + TIME_EPS) + 1, dtype=float)
    right = np.minimum(np.searchsorted(t, grid - TIME_EPS, side="left"), len(t) - 1)
    exact = np.abs(t[right] - grid) <= TIME_EPS
    left = np.maximum(right - 1, 0)
    span = t[right] - t[left]
    weight = np.where(span > 0, (grid - t[left]) / np.where(span > 0, span, 1.0), 0.0)
    gap = ~exact & (span > MAX_GAP_S + TIME_EPS)

    def interpolate(values: np.ndarray) -> np.ndarray:
        out = values[left] + weight * (values[right] - values[left])
        out = np.where(exact, values[right], out)
        out[gap] = np.nan
        return out

    x = interpolate(np.asarray(track.x_mm, dtype=float))
    y = interpolate(np.asarray(track.y_mm, dtype=float))
    on = None
    if track.on_lawn is not None:
        flags = np.asarray(track.on_lawn, dtype=float)
        on = np.where(exact | (weight >= HALF), flags[right], flags[left])
        on[gap] = np.nan
    return x, y, on


def lawn_membership(
    x: np.ndarray,
    y: np.ndarray,
    on_column: np.ndarray | None,
    lawn: Lawn | None,
) -> np.ndarray:
    """Return which 1-s samples are on the lawn: by the column, the boundary, or a uniform well.

    A boundary marks a sample on the lawn when its signed distance inside the boundary, about the
    lawn's centre, is positive: the rule the vendored Scheer tracks are cut by.
    """
    if on_column is not None:
        return np.nan_to_num(on_column, nan=0.0) > HALF
    if lawn is None:
        msg = "lawn membership needs an on_lawn column or a lawn in the metadata"
        raise ValueError(msg)
    if lawn.uniform:
        return np.ones(len(x), dtype=bool)
    assert lawn.radii is not None  # noqa: S101 - a non-uniform lawn has radii
    with np.errstate(invalid="ignore"):
        inside = boundary_distance(
            np.asarray(x) - lawn.centre[0],
            np.asarray(y) - lawn.centre[1],
            np.asarray(lawn.radii, dtype=float),
        )
        return inside > 0


def to_steps(x: np.ndarray, y: np.ndarray, on_lawn: np.ndarray | None) -> list[BehaviourStep]:
    """Return one step per second carrying position and lawn membership; the rest is unread."""
    on = np.zeros(len(x), dtype=bool) if on_lawn is None else np.asarray(on_lawn, dtype=bool)
    return [
        BehaviourStep(
            step=i,
            x=float(x[i]),
            y=float(y[i]),
            heading_rad=0.0,
            concentration=0.0,
            dc_dt=0.0,
            grad_dir=0.0,
            grad_strength=0.0,
            on_lawn=bool(on[i]),
        )
        for i in range(len(x))
    ]


def import_population(path: Path) -> tuple[str, list[list[BehaviourStep]]]:
    """Return a file's family and its animals' steps, each cut to the family's window.

    Raises
    ------
    ValueError
        If an animal's track starts after the window, or is shorter than the window.
    """
    tracks, metadata = read_tracks(path)
    samples = FAMILY_SAMPLES[metadata.family]
    animals = []
    for track in tracks:
        x, y, on_column = resample_1s(track)
        offset = metadata.analysis_start_s - float(track.t_s[0])
        if offset < -TIME_EPS:
            msg = f"{track.animal}: its track starts after the analysis window"
            raise ValueError(msg)
        start = round(offset)
        if metadata.family == "local_search":
            lawn = np.zeros(len(x), dtype=bool)
        else:
            lawn = lawn_membership(x, y, on_column, metadata.lawn)
        steps = to_steps(x, y, lawn)
        animals.append(g.analysis_steps(steps, (start, len(steps)), samples))
    return metadata.family, animals
