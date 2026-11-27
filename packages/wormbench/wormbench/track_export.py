"""Write worm tracks in the formats the grader imports: CSV with its sidecar, and WCON 1.0.

The writers are the importer's inverse. A file written here and read back by
:mod:`wormbench.track_import` gives the same samples, in mm and s, and the same assay metadata, so
it grades the same:

- **CSV:** columns ``animal``, ``t_s``, ``x`` and ``y``, and ``on_lawn`` when the tracks carry the
  flag, with the metadata in a sidecar ``<file>.json``. A missing sample is ``nan``.
- **WCON 1.0's core:** ``units`` (``s`` and ``mm``), one ``data`` record per animal, and the
  metadata in a top-level ``@nematode_benchmark`` object. A missing sample is ``null``.

Simulated tracks also carry each second's primitive and the animal's seed: ``primitive`` and
``seed`` columns, or each record's ``@primitive`` and ``@seed``.

WCON's core has no on-lawn flag, so a lawn family's WCON file marks lawn membership by the boundary
in its metadata, or as a uniform well. A track whose membership comes from its ``on_lawn`` column
cannot be written as WCON without changing its grade, and is refused.
"""

from __future__ import annotations

import csv
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from wormbench import track_import as ti

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

WCON_UNITS = {"t": "s", "x": "mm", "y": "mm"}


def metadata_object(metadata: ti.AssayMetadata) -> dict[str, Any]:
    """Return the metadata as the sidecar and ``@nematode_benchmark`` object hold it, in mm.

    Raises
    ------
    ValueError
        If the metadata would read back differently: an unknown family, a start that is not
        finite, a uniform well without a uniform lawn, or local search with a lawn.
    """
    _check_metadata(metadata)
    out: dict[str, Any] = {
        "family": metadata.family,
        "analysis_start_s": float(metadata.analysis_start_s),
    }
    lawn = metadata.lawn
    if metadata.family != "local_search" and lawn is not None:
        if lawn.uniform:
            out["lawn"] = "uniform"
        else:
            assert lawn.radii is not None  # noqa: S101 - a non-uniform lawn has radii
            out["lawn"] = {
                "centre": [float(v) for v in lawn.centre],
                "radii": [float(r) for r in lawn.radii],
            }
    return out


def _check_metadata(metadata: ti.AssayMetadata) -> None:
    family, lawn = metadata.family, metadata.lawn
    if family not in ti.FAMILY_SAMPLES:
        msg = f"family {family!r}; expected one of {sorted(ti.FAMILY_SAMPLES)}"
        raise ValueError(msg)
    if not math.isfinite(float(metadata.analysis_start_s)):
        msg = "analysis_start_s must be finite"
        raise ValueError(msg)
    if family == "uniform_lawn" and (lawn is None or not lawn.uniform):
        msg = "a uniform_lawn file is a uniform well: its lawn must be Lawn(), read as 'uniform'"
        raise ValueError(msg)
    if family == "local_search" and lawn is not None:
        msg = "local search is off food: its metadata hold no lawn"
        raise ValueError(msg)


def _split(
    tracks: Sequence[ti.ImportedTrack | ti.SimulatedTrack],
) -> tuple[list[ti.ImportedTrack], list[ti.SimulatedTrack] | None]:
    """Return the plain tracks, and the simulated ones when every track is simulated."""
    if not tracks:
        msg = "no tracks to write"
        raise ValueError(msg)
    simulated = [t for t in tracks if isinstance(t, ti.SimulatedTrack)]
    if simulated and len(simulated) != len(tracks):
        msg = "a file's tracks are all simulated or none: a simulated file labels every animal"
        raise ValueError(msg)
    plain = [t.track if isinstance(t, ti.SimulatedTrack) else t for t in tracks]
    # The importer reads every animal id as a string, so ids are compared as it will read them.
    animals = [str(t.animal) for t in plain]
    repeated = sorted({a for a in animals if animals.count(a) > 1})
    if repeated:
        msg = f"animals {repeated} appear twice: the importer would merge their samples"
        raise ValueError(msg)
    for track in plain:
        _check_track(track)
    for s in simulated:
        _check_simulated(s)
    return plain, simulated or None


def _check_simulated(simulated: ti.SimulatedTrack) -> None:
    """Refuse a simulated track the importer's simulated route would refuse or misread."""
    animal, t = simulated.track.animal, np.asarray(simulated.track.t_s, dtype=float)
    if len(simulated.primitives) != len(t):
        msg = f"{animal}: needs one primitive label per sample"
        raise ValueError(msg)
    unknown = sorted({str(v) for v in simulated.primitives} - set(ti.PRIMITIVE_LABELS))
    if unknown:
        msg = f"{animal}: unknown primitive labels {unknown}; expected one of {ti.PRIMITIVE_LABELS}"
        raise ValueError(msg)
    if len(t) > 1 and not np.allclose(np.diff(t), 1.0, rtol=0.0, atol=ti.TIME_EPS):
        msg = f"{animal}: a simulated track is sampled every 1 s"
        raise ValueError(msg)
    seed = simulated.seed
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        msg = f"{animal}: a seed is an integer, not {seed!r}"
        raise ValueError(msg)


def _check_track(track: ti.ImportedTrack) -> None:
    t = np.asarray(track.t_s, dtype=float)
    if len(t) == 0:
        msg = f"{track.animal}: a track needs at least one sample"
        raise ValueError(msg)
    if not np.all(np.isfinite(t)):
        msg = f"{track.animal}: every sample needs a finite time"
        raise ValueError(msg)
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        msg = f"{track.animal}: sample times must increase"
        raise ValueError(msg)
    lengths = {len(t), len(track.x_mm), len(track.y_mm)}
    if track.on_lawn is not None:
        lengths.add(len(track.on_lawn))
    if len(lengths) != 1:
        msg = f"{track.animal}: times, positions and flags differ in length"
        raise ValueError(msg)
    for axis in (track.x_mm, track.y_mm):
        if np.any(np.isinf(np.asarray(axis, dtype=float))):
            msg = f"{track.animal}: a position is infinite; write a missing sample as NaN"
            raise ValueError(msg)


def _coordinate(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def write_csv(
    path: Path,
    tracks: Sequence[ti.ImportedTrack | ti.SimulatedTrack],
    metadata: ti.AssayMetadata,
) -> Path:
    """Write the tracks as one CSV, in mm and s, with the metadata in ``<file>.json``."""
    plain, simulated = _split(tracks)
    flagged = [t.on_lawn is not None for t in plain]
    if any(flagged) and not all(flagged):
        msg = "an on_lawn column covers every animal or none"
        raise ValueError(msg)
    columns = ["animal", "t_s", "x", "y"] + (["on_lawn"] if all(flagged) else [])
    columns += ["primitive", "seed"] if simulated else []
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i, track in enumerate(plain):
            for j in range(len(track.t_s)):
                row = [
                    str(track.animal),
                    repr(float(track.t_s[j])),
                    repr(float(track.x_mm[j])),
                    repr(float(track.y_mm[j])),
                ]
                if track.on_lawn is not None:
                    row.append(repr(float(track.on_lawn[j])))
                if simulated:
                    seed = simulated[i].seed
                    row += [simulated[i].primitives[j], "" if seed is None else str(seed)]
                writer.writerow(row)
    sidecar = {"units": "mm", **metadata_object(metadata)}
    path.with_name(path.name + ".json").write_text(json.dumps(sidecar, allow_nan=False))
    return path


def wcon_document(
    tracks: Sequence[ti.ImportedTrack | ti.SimulatedTrack],
    metadata: ti.AssayMetadata,
) -> dict[str, Any]:
    """Return the tracks as a WCON 1.0 document carrying the ``@nematode_benchmark`` metadata."""
    plain, simulated = _split(tracks)
    if metadata.family != "local_search":
        if any(t.on_lawn is not None for t in plain):
            msg = (
                "WCON has no on-lawn flag, and the importer marks lawn membership by an on_lawn "
                "column before the boundary: writing these tracks as WCON would change their grade"
            )
            raise ValueError(msg)
        if metadata.lawn is None:
            msg = f"a {metadata.family} WCON file needs a lawn boundary, or 'uniform', in metadata"
            raise ValueError(msg)
    records = []
    for i, track in enumerate(plain):
        record: dict[str, Any] = {
            "id": str(track.animal),
            "t": [float(v) for v in track.t_s],
            "x": [_coordinate(float(v)) for v in track.x_mm],
            "y": [_coordinate(float(v)) for v in track.y_mm],
        }
        if simulated:
            record["@primitive"] = list(simulated[i].primitives)
            seed = simulated[i].seed
            if seed is not None:
                record["@seed"] = int(seed)
        records.append(record)
    return {"units": dict(WCON_UNITS), "data": records, ti.METADATA_KEY: metadata_object(metadata)}


def write_wcon(
    path: Path,
    tracks: Sequence[ti.ImportedTrack | ti.SimulatedTrack],
    metadata: ti.AssayMetadata,
) -> Path:
    """Write the tracks as one WCON 1.0 file, in mm and s, with the benchmark's metadata."""
    document = wcon_document(tracks, metadata)
    path.write_text(json.dumps(document, allow_nan=False) + "\n")
    return path


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True))


def csv_to_wcon(source: Path, out: Path, *, simulated: bool = False) -> Path:
    """Write a CSV's tracks (with its sidecar) as WCON, then check the WCON reads back the same.

    With ``simulated`` the primitive labels and seeds are carried too. The file is written beside
    ``out`` and read back with the importer, which must give every animal's times and positions
    exactly, its labels and seed, and the metadata. Only then does it replace ``out``; on any
    failure nothing is written to ``out``, and a file already there is left as it was.

    Raises
    ------
    ValueError
        If the source is not a CSV or is missing, ``out`` would overwrite the source or be read as
        a CSV, the tracks cannot be written as WCON, or the check fails.
    """
    if source.suffix.lower() != ".csv":
        msg = f"{source} is not a CSV; its tracks are read from <file>.csv and <file>.csv.json"
        raise ValueError(msg)
    if not source.is_file():
        msg = f"{source}: no such file"
        raise ValueError(msg)
    sidecar = source.with_name(source.name + ".json")
    if out.resolve() in (source.resolve(), sidecar.resolve()):
        msg = f"{out} would overwrite the source; write the WCON file elsewhere"
        raise ValueError(msg)
    if out.suffix.lower() == ".csv":
        msg = f"{out}: the grader reads a .csv file as CSV; name the WCON file .wcon"
        raise ValueError(msg)
    tracks: list[Any]
    if simulated:
        tracks, metadata = ti.read_simulated_csv(source)
    else:
        tracks, metadata = ti.read_csv(source)
    document = wcon_document(tracks, metadata)
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_name(out.name + ".part")
    try:
        staged.write_text(json.dumps(document, allow_nan=False) + "\n")
        back: list[Any]
        if simulated:
            back, back_metadata = ti.read_simulated_wcon(staged)
        else:
            back, back_metadata = ti.read_wcon(staged)
        problem = (
            f"the metadata read back as {back_metadata}, not {metadata}"
            if back_metadata != metadata
            else _difference(tracks, back)
        )
        if problem is not None:
            msg = f"{out}: {problem}; nothing was written"
            raise ValueError(msg)
        staged.replace(out)
    finally:
        staged.unlink(missing_ok=True)
    return out


def _difference(
    written: Sequence[ti.ImportedTrack | ti.SimulatedTrack],
    read: Sequence[ti.ImportedTrack | ti.SimulatedTrack],
) -> str | None:
    """Return how the tracks read back differ from those written, or None if they do not."""
    if [_plain(t).animal for t in read] != [_plain(t).animal for t in written]:
        return "the animals read back differ"
    for a, b in zip(written, read, strict=True):
        pa, pb = _plain(a), _plain(b)
        if not (_same(pa.t_s, pb.t_s) and _same(pa.x_mm, pb.x_mm) and _same(pa.y_mm, pb.y_mm)):
            return f"{pa.animal}'s samples read back differ"
        labels = [(t.primitives, t.seed) for t in (a, b) if isinstance(t, ti.SimulatedTrack)]
        if len(labels) == 1 or (labels and labels[0] != labels[1]):
            return f"{pa.animal}'s labels or seed read back differ"
    return None


def _plain(track: ti.ImportedTrack | ti.SimulatedTrack) -> ti.ImportedTrack:
    return track.track if isinstance(track, ti.SimulatedTrack) else track
