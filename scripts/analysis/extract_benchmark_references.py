#!/usr/bin/env python
r"""Derive the benchmark's vendored real-worm reference sets from their deposits.

One subcommand per source, each reading the original deposit and writing a derived file under
``data/benchmark/``. The deposits themselves are never vendored; ``PROVENANCE.md`` beside the
derived files records each one's licence, checksum and units.

``scheer``
    Scheer & Bargmann 2023 (Zenodo 8310289, CC0). Eight arms of ``GenotypeComparisons_021523.pkl``,
    read through the restricted unpickler: per animal, the midbody position at 1 s in micrometres
    from its lawn centre (or, without a lawn boundary, from its first valid position), the authors'
    per-frame flags at 3 frames/s, bit-packed, their per-bin run masks and labels, the lawn boundary
    as 64 radii, and the animal's genotype, date, video and source row. ``--fig1`` also reads the
    Figure 1 wild type and matches its animals to ``PD1074_od2_Data``.

``split``
    The cohort split, from the store's dates and counts alone: recording dates, a third held out,
    stratified by arm, with each arm's share checked for feasibility first.

``calhoun``
    Calhoun, Chalasani & Sharpee 2014 (Dryad 10.5061/dryad.3j2j9, CC0), ``Wormdata.xlsx``: 47 worms,
    3 frames/s, 30 min, positions in pixels (the paper states no scale) and per-frame onset flags.

``roberts``
    Roberts et al. 2016 (Dryad 10.5061/dryad.35qv6, CC0): the extracted ``wormtracks archive``.
    Igor binary waves, read by a minimal reader here. Positions in mm at 29.97 frames/s, the
    tracker's forward/reverse/pause state per frame, and the authors' four-state Viterbi decoding,
    for the five intact wild-type cohorts, with the dead-worm tracks beside them.

``lopezcruz``
    López-Cruz et al. 2019 (Mendeley 10.17632/prdtw5jgcd.1, CC BY 4.0),
    ``CompleteForagingAssayData.xls``: every row of reorientations per 2-min bin after food removal.
    Needs ``xlrd``, which the project does not depend on: run it with ``uv run --with xlrd``.

Usage::

    uv run python scripts/analysis/extract_benchmark_references.py scheer \
        --genotypes build/scheer2023/GenotypeComparisons_021523.pkl \
        --fig1 build/scheer2023/PD1074_od2_Fig1_021523.pkl \
        --out data/benchmark/scheer2023_tracks.npz \
        --record docs/experiments/logbooks/supporting/105-reference-extraction/scheer_extraction.json
    uv run python scripts/analysis/extract_benchmark_references.py split \
        --tracks data/benchmark/scheer2023_tracks.npz --out data/benchmark/scheer2023_split.json
    uv run python scripts/analysis/extract_benchmark_references.py calhoun \
        --xlsx build/dryad/x/calhoun/Wormdata.xlsx --out data/benchmark/calhoun2014_tracks.npz
    uv run python scripts/analysis/extract_benchmark_references.py roberts \
        --archive build/dryad/x/roberts/arc --out data/benchmark/roberts2016_tracks.npz
    uv run --with xlrd python scripts/analysis/extract_benchmark_references.py lopezcruz \
        --xls build/mendeley/CompleteForagingAssayData.xls \
        --out data/benchmark/lopezcruz2019_reorientations.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

DESCRIPTION = "Derive the benchmark's vendored real-worm reference sets from their deposits."

# ----------------------------------------------------------------------------------------------
# Shared store conventions
# ----------------------------------------------------------------------------------------------

POSITION_STRIDE = 3  # frames per second at 3 frames/s
BOUNDARY_ANGLES = 64

SCHEER_ARMS = (
    "PD1074_od2_Data",
    "PD1074_od1_Data",
    "PD1074_od4_Data",
    "PD1074_od2_uniformlawn_Data",
    # The N2 wild type: the batch-matched control of the N2-background tph-1 strain, recorded only
    # beside it. Kept as its own arm, never pooled with PD1074.
    "N2_od2_Data",
    "MT15434_tph1mg280_od2_Data",
    "CX14295_pdfr1ok3425_od2_Data",
    "CX13078_tax4p678_od2_Data",
)
WILD_TYPE_OD2 = "PD1074_od2_Data"
# The share of Fig. 1 animals that must match the stored wild type for D.1's windows to be checked.
FIG1_MATCH_FLOOR = 0.95
# Bit order of ``frame_flags``: each the authors' per-frame field.
FRAME_FLAGS = (
    ("reversal", "MovingReverse"),
    ("omega", "Omega"),
    ("pause", "Pause"),
    ("head_poke", "HeadPokesAll"),
    ("midbody_in_lawn", "MidbodyInLawn"),
    ("missing", "missingdata"),
)

SPLIT_SEED = 2027
SPLIT_DRAWS = 10_000
HELD_OUT_PROBABILITY = 1.0 / 3.0
ARM_SHARE = (0.20, 0.45)
WILD_TYPE_SHARE = (0.28, 0.38)
_ENUMERATE_MAX_DATES = 20
_SAMPLED_SUBSETS = 100_000


def quantise(values: np.ndarray, per_unit: float, dtype: type[np.signedinteger]) -> np.ndarray:
    """Round ``values * per_unit`` to integers; the dtype's minimum marks a missing value."""
    info = np.iinfo(dtype)
    out = np.full(values.shape, info.min, dtype=dtype)
    finite = np.isfinite(values)
    out[finite] = np.rint(values[finite] * per_unit).astype(dtype)
    return out


def quantise_um(values_mm: np.ndarray, dtype: type[np.signedinteger]) -> np.ndarray:
    """Round positions in mm to whole micrometres; the dtype's minimum marks a missing value."""
    return quantise(values_mm, 1000.0, dtype)


def position_dtype(values_mm: list[np.ndarray]) -> type[np.signedinteger]:
    """int16 if every finite position fits in it, short of its missing marker; int32 otherwise."""
    limit = np.iinfo(np.int16).max / 1000.0
    widest = max((float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 0.0) for v in values_mm)
    return np.int16 if widest <= limit else np.int32


def pack_flags(flags: list[np.ndarray]) -> np.ndarray:
    """Pack up to eight boolean arrays of one shape into a uint8 array, bit i from flags[i]."""
    out = np.zeros(flags[0].shape, dtype=np.uint8)
    for bit, flag in enumerate(flags):
        out |= np.asarray(flag, dtype=bool).astype(np.uint8) << np.uint8(bit)
    return out


# ----------------------------------------------------------------------------------------------
# Scheer & Bargmann 2023
# ----------------------------------------------------------------------------------------------


def _strings(value: Any) -> list[str]:  # noqa: ANN401 - a DataFrame, Series or array of strings
    array = np.asarray(value.to_numpy() if hasattr(value, "to_numpy") else value, dtype=object)
    return [str(v) for v in array.reshape(len(array), -1)[:, 0]]


def _dates(value: Any) -> np.ndarray:  # noqa: ANN401 - a Series of datetime.date
    items = value.tolist() if hasattr(value, "tolist") else list(value)
    return np.array([d.year * 10000 + d.month * 100 + d.day for d in items], dtype=np.int32)


def _per_animal(value: Any, n: int) -> np.ndarray:  # noqa: ANN401 - an array of any per-animal shape
    """One value per animal: a per-frame field is reduced to its per-animal median."""
    array = np.asarray(value, dtype=float).reshape(n, -1)
    if array.shape[1] == 1:
        return array[:, 0]
    with np.errstate(all="ignore"):
        return np.nanmedian(array, axis=1)


def boundary_radii(
    points_x: np.ndarray,
    points_y: np.ndarray,
    centre_x: float,
    centre_y: float,
) -> np.ndarray:
    """Return the boundary's radius at ``BOUNDARY_ANGLES`` equally spaced angles from +x.

    The radii are in the points' units. The polygon is taken in polar coordinates about the centre
    and interpolated periodically in angle.
    """
    keep = np.isfinite(points_x) & np.isfinite(points_y)
    dx, dy = points_x[keep] - centre_x, points_y[keep] - centre_y
    if len(dx) < 3:
        return np.full(BOUNDARY_ANGLES, np.nan)
    angle = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    radius = np.hypot(dx, dy)
    order = np.argsort(angle)
    targets = np.arange(BOUNDARY_ANGLES) * (2 * np.pi / BOUNDARY_ANGLES)
    return np.interp(targets, angle[order], radius[order], period=2 * np.pi)


def _slots(data: dict[str, Any], i: int, centre_x: np.ndarray) -> np.ndarray:
    """Return the background slots animal ``i`` uses: its ``bgvidindex`` values, 1-based."""
    if "bgvidindex" in data:
        index = np.asarray(data["bgvidindex"][i], dtype=float)
        used = np.unique(index[np.isfinite(index)]).astype(int) - 1
        return used[(used >= 0) & (used < centre_x.shape[1])]
    return np.flatnonzero(np.isfinite(centre_x[i]) & (centre_x[i] != 0))


def lawn_geometry(
    data: dict[str, Any],
    n: int,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return each animal's lawn centre (px), boundary radii (mm) and centre spread (µm).

    The lawn is re-detected in each background segment of a recording. ``Center_Point_x/y`` holds one
    column per background slot, and ``Lawn_Boundary_Pts_x/y`` holds a block of rows per animal with
    one polygon per slot; unused slots are zero or NaN. The animal's centre is the median over the
    slots its frames use, and its radii are the median of those slots' radii about that centre.
    """
    centre_x = np.asarray(data["Center_Point_x"], dtype=float).reshape(n, -1)
    centre_y = np.asarray(data["Center_Point_y"], dtype=float).reshape(n, -1)
    slots = centre_x.shape[1]
    points_x = np.asarray(data["Lawn_Boundary_Pts_x"], dtype=float).reshape(n, -1, slots)
    points_y = np.asarray(data["Lawn_Boundary_Pts_y"], dtype=float).reshape(n, -1, slots)
    cx, cy = np.full(n, np.nan), np.full(n, np.nan)
    radii = np.full((n, BOUNDARY_ANGLES), np.nan)
    spread = np.full(n, np.nan)
    for i in range(n):
        used = _slots(data, i, centre_x)
        used = used[np.isfinite(centre_x[i, used]) & (centre_x[i, used] != 0)]
        if len(used) == 0:
            continue
        cx[i], cy[i] = np.median(centre_x[i, used]), np.median(centre_y[i, used])
        spread[i] = 1000.0 * float(
            np.max(np.hypot(centre_x[i, used] - cx[i], centre_y[i, used] - cy[i])) / scale[i],
        )
        per_slot = []
        for c in used:
            px, py = points_x[i, :, c], points_y[i, :, c]
            real = ~((px == 0) & (py == 0))
            per_slot.append(boundary_radii(px[real], py[real], cx[i], cy[i]) / scale[i])
        radii[i] = np.nanmedian(np.stack(per_slot), axis=0)
    return cx, cy, radii, spread


def _boundary_check(  # noqa: PLR0913 - the arm's positions, its geometry and the authors' distance
    data: dict[str, Any],
    x_px: np.ndarray,
    y_px: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    scale: np.ndarray,
    radii: np.ndarray,
) -> dict[str, float]:
    """Compare the store's signed boundary distance with the authors' centroid distance.

    The authors' ``Lawn_Boundary_Dist`` is the head's (it agrees with ``HeadInLawn``), so the midbody
    is compared with ``Centroid_Lawn_Boundary_Dist`` and ``MidbodyInLawn``.
    """
    from quantumnematode.benchmark.references import boundary_distance

    theirs = np.asarray(data["Centroid_Lawn_Boundary_Dist"], dtype=float)[:, ::POSITION_STRIDE]
    x = (x_px[:, ::POSITION_STRIDE] - cx[:, None]) / scale[:, None]
    y = (y_px[:, ::POSITION_STRIDE] - cy[:, None]) / scale[:, None]
    ours = np.stack([boundary_distance(x[i], y[i], radii[i]) for i in range(len(x))])
    in_lawn = np.asarray(data["MidbodyInLawn"], dtype=bool)[:, ::POSITION_STRIDE]
    both = np.isfinite(ours) & np.isfinite(theirs)
    clear = both & (np.abs(theirs) > 0.1)
    return {
        "median_abs_difference_mm": float(np.median(np.abs(ours[both] - theirs[both]))),
        "p95_abs_difference_mm": float(np.percentile(np.abs(ours[both] - theirs[both]), 95)),
        "sign_agreement_clear_of_edge": float(
            np.mean(np.sign(ours[clear]) == np.sign(theirs[clear])),
        ),
        "inside_agrees_with_midbody_in_lawn": float(
            np.mean((ours > 0)[np.isfinite(ours)] == in_lawn[np.isfinite(ours)]),
        ),
    }


def scheer_arm(data: dict[str, Any], arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one arm's per-animal arrays (positions still in mm) and what the arm lacked."""
    x_px = np.asarray(data["Midbody_cent_x"], dtype=float)
    y_px = np.asarray(data["Midbody_cent_y"], dtype=float)
    n, frames = x_px.shape
    scale = _per_animal(data["pixpermm"], n)
    has_boundary = "Center_Point_x" in data and "Lawn_Boundary_Pts_x" in data
    notes: dict[str, Any] = {"animals": int(n), "frames": int(frames), "has_boundary": has_boundary}
    if has_boundary:
        cx, cy, radii, spread = lawn_geometry(data, n, scale)
        notes["centre_spread_um_p99"] = float(np.nanpercentile(spread, 99))
        notes["centre_spread_um_max"] = float(np.nanmax(spread))
        if "Centroid_Lawn_Boundary_Dist" in data and "MidbodyInLawn" in data:
            notes["boundary_check"] = _boundary_check(data, x_px, y_px, cx, cy, scale, radii)
        origin = np.zeros(n, dtype=np.int8)
    else:
        sampled = np.isfinite(x_px) & np.isfinite(y_px)
        first = np.argmax(sampled, axis=1)
        cx, cy = x_px[np.arange(n), first], y_px[np.arange(n), first]
        radii = np.full((n, BOUNDARY_ANGLES), np.nan)
        origin = np.ones(n, dtype=np.int8)
    x_mm = (x_px[:, ::POSITION_STRIDE] - cx[:, None]) / scale[:, None]
    y_mm = (y_px[:, ::POSITION_STRIDE] - cy[:, None]) / scale[:, None]
    absent = [field for _, field in FRAME_FLAGS if field not in data]
    notes["absent_flags"] = absent
    flags = [
        np.asarray(data[field], dtype=bool).reshape(n, frames)
        if field in data
        else np.zeros((n, frames), dtype=bool)
        for _, field in FRAME_FLAGS
    ]
    labels_ma = data["RD_states_Matrix_exog"]
    raw = np.where(np.ma.getdata(labels_ma), 1, 0)
    labels = np.where(np.ma.getmaskarray(labels_ma), -1, raw).astype(np.int8)
    arrays = {
        "x_mm": x_mm,
        "y_mm": y_mm,
        "frame_flags": pack_flags(flags),
        "in_lawn_run": np.asarray(data["InLawnRunMask"], dtype=bool),
        "out_lawn_run": np.asarray(data["OutLawnRunMask"], dtype=bool),
        "labels": labels,
        "boundary_mm": radii,
        "pixpermm": scale.astype(np.float32),
        "videos": _strings(data["videoname"]),
        "matfiles": _strings(data["matfilename"]),
        "date": _dates(data["dates"]),
        "origin": origin,
        "arm": arm,
    }
    return arrays, notes


def match_animals(
    reference: list[tuple[str, str]],
    candidate: list[tuple[str, str]],
) -> list[int | None]:
    """For each reference animal, the candidate row with the same video and source file.

    Repeated (video, file) pairs are matched in order of appearance.
    """
    rows: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(candidate):
        rows.setdefault(key, []).append(i)
    seen: dict[tuple[str, str], int] = {}
    out: list[int | None] = []
    for key in reference:
        k = seen.get(key, 0)
        seen[key] = k + 1
        found = rows.get(key, [])
        out.append(found[k] if k < len(found) else None)
    return out


def _load_pickle(path: Path) -> Any:  # noqa: ANN401 - the deposit's tuple
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _scheer_pickle import SafeUnpickler

    with path.open("rb") as f:
        return SafeUnpickler(f).load()


def extract_scheer(genotypes: Path, fig1: Path | None, out: Path) -> dict[str, Any]:
    """Write the Scheer store; return the extraction record."""
    _bins, bin_size, data = _load_pickle(genotypes)
    expected_bin = POSITION_STRIDE * 10
    if bin_size != expected_bin:
        msg = f"the deposit's bins are {bin_size} frames, not {expected_bin}"
        raise ValueError(msg)
    record: dict[str, Any] = {"source": genotypes.name, "arms": {}}
    parts = []
    for arm in SCHEER_ARMS:
        if arm not in data:
            msg = f"{arm} is not in {genotypes.name}"
            raise KeyError(msg)
        arrays, notes = scheer_arm(data[arm], arm)
        notes["dates"] = len(set(arrays["date"].tolist()))
        record["arms"][arm] = notes
        parts.append(arrays)
        data[arm] = None  # free the arm's source arrays
    wt = parts[SCHEER_ARMS.index(WILD_TYPE_OD2)]
    for p in parts:
        p["fig1_row"] = np.full(len(p["date"]), -1, dtype=np.int32)
    if fig1 is not None:
        record["fig1"] = _match_fig1(fig1, wt)
        if record["fig1"]["matched_fraction"] < FIG1_MATCH_FLOOR:
            msg = (
                f"only {record['fig1']['matched']} of {record['fig1']['fig1_animals']} Fig. 1 "
                f"animals match {WILD_TYPE_OD2}, below the {FIG1_MATCH_FLOOR:.0%} floor; "
                "the windows check could not stand on them"
            )
            raise ValueError(msg)
        wt["fig1_row"][record["fig1"]["store_rows"]] = record["fig1"]["fig1_rows"]
    _write_scheer_store(parts, out)
    record["store_bytes"] = out.stat().st_size
    return record


def _match_fig1(fig1: Path, wt: dict[str, Any]) -> dict[str, Any]:
    _bins, _size, fig = _load_pickle(fig1)
    key = next(iter(fig)) if isinstance(fig, dict) and "Midbody_cent_x" not in fig else None
    figdata = fig[key] if key is not None else fig
    reference = list(
        zip(_strings(figdata["videoname"]), _strings(figdata["matfilename"]), strict=True),
    )
    candidate = list(zip(wt["videos"], wt["matfiles"], strict=True))
    rows = match_animals(reference, candidate)
    matched = [(i, r) for i, r in enumerate(rows) if r is not None]
    fx = np.asarray(figdata["Midbody_cent_x"], dtype=float)
    scale = _per_animal(figdata["pixpermm"], len(fx))
    equal = 0
    for i, r in matched:
        a = fx[i, ::POSITION_STRIDE] / scale[i]
        b = wt["x_mm"][r] + 0.0
        # The store subtracts the centre; compare displacements, which do not depend on it.
        da, db = np.diff(a), np.diff(b)
        if np.allclose(da, db, atol=1e-9, equal_nan=True):
            equal += 1
    return {
        "fig1_animals": len(reference),
        "matched": len(matched),
        "matched_fraction": len(matched) / max(len(reference), 1),
        "positions_equal": equal,
        "store_rows": [r for _, r in matched],
        "fig1_rows": [i for i, _ in matched],
    }


def _write_scheer_store(parts: list[dict[str, Any]], out: Path) -> None:
    dtype = position_dtype([p["x_mm"] for p in parts] + [p["y_mm"] for p in parts])
    videos: list[str] = []
    video_index: dict[str, int] = {}
    for p in parts:
        for v in p["videos"]:
            video_index.setdefault(v, len(videos))
            if video_index[v] == len(videos):
                videos.append(v)
    cat = {
        "x_um": np.concatenate([quantise_um(p["x_mm"], dtype) for p in parts]),
        "y_um": np.concatenate([quantise_um(p["y_mm"], dtype) for p in parts]),
        "frame_flags": np.concatenate([p["frame_flags"] for p in parts]),
        "in_lawn_run": np.concatenate([p["in_lawn_run"] for p in parts]),
        "out_lawn_run": np.concatenate([p["out_lawn_run"] for p in parts]),
        "labels": np.concatenate([p["labels"] for p in parts]),
        "boundary_um": np.concatenate([quantise_um(p["boundary_mm"], np.int16) for p in parts]),
        "pixpermm": np.concatenate([p["pixpermm"] for p in parts]),
        "genotype": np.concatenate(
            [np.full(len(p["date"]), SCHEER_ARMS.index(p["arm"]), dtype=np.int16) for p in parts],
        ),
        "video": np.concatenate(
            [np.array([video_index[v] for v in p["videos"]], dtype=np.int32) for p in parts],
        ),
        "date": np.concatenate([p["date"] for p in parts]),
        "source_row": np.concatenate([np.arange(len(p["date"]), dtype=np.int32) for p in parts]),
        "origin": np.concatenate([p["origin"] for p in parts]),
        "fig1_row": np.concatenate([p["fig1_row"] for p in parts]),
        "genotypes": np.array(SCHEER_ARMS),
        "videos": np.array(videos),
        "flag_names": np.array([name for name, _ in FRAME_FLAGS]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **cat)


# ----------------------------------------------------------------------------------------------
# The cohort split
# ----------------------------------------------------------------------------------------------


def _subset_shares(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Held-out shares of every proper, non-empty subset of an arm's dates (sampled if many)."""
    total = counts.sum()
    k = len(counts)
    if k <= _ENUMERATE_MAX_DATES:
        masks = np.array(list(itertools.product((0, 1), repeat=k)), dtype=bool)
    else:
        masks = rng.random((_SAMPLED_SUBSETS, k)) < HELD_OUT_PROBABILITY
    proper = masks.any(axis=1) & ~masks.all(axis=1)
    return (masks[proper] @ counts) / total


def draw_split(
    dates: np.ndarray,
    arms: np.ndarray,
    wild_type: int,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Draw the cohort split from each animal's recording date and arm alone."""
    unique_dates = np.unique(dates)
    arm_ids = np.unique(arms)
    counts = {
        int(a): np.array([np.sum((arms == a) & (dates == d)) for d in unique_dates])
        for a in arm_ids
    }
    single = [int(a) for a in arm_ids if np.count_nonzero(counts[int(a)]) < 2]
    infeasible = []
    rng = np.random.default_rng(seed)
    for a in arm_ids:
        if int(a) in single:
            continue
        c = counts[int(a)][counts[int(a)] > 0]
        shares = _subset_shares(c, rng)
        if not np.any((shares >= ARM_SHARE[0]) & (shares <= ARM_SHARE[1])):
            infeasible.append(int(a))
    best: tuple[float, int, np.ndarray] | None = None
    for k in range(SPLIT_DRAWS):
        held = np.random.default_rng(seed + k).random(len(unique_dates)) < HELD_OUT_PROBABILITY
        violation = _violation(held, counts, single, infeasible, wild_type)
        if violation == 0.0:
            return _split_record(unique_dates, held, k, counts, single, infeasible, seed)
        if best is None or violation < best[0]:
            best = (violation, k, held)
    if best is None:  # pragma: no cover - SPLIT_DRAWS is positive
        msg = "no draw was made"
        raise RuntimeError(msg)
    msg = (
        f"no draw of {SPLIT_DRAWS} met the split's constraints; the closest was k = {best[1]} "
        f"(violation {best[0]:.3f})"
    )
    raise RuntimeError(msg)


def _violation(
    held: np.ndarray,
    counts: dict[int, np.ndarray],
    single: list[int],
    infeasible: list[int],
    wild_type: int,
) -> float:
    """How far a draw is from meeting every constraint; zero when it meets them all."""
    total = 0.0
    for a, c in counts.items():
        if a in single:
            continue
        on = c > 0
        if not (held & on).any() or not (~held & on).any():
            total += 1.0
            continue
        share = c[held].sum() / c.sum()
        low, high = WILD_TYPE_SHARE if a == wild_type else ARM_SHARE
        if a in infeasible and a != wild_type:
            continue
        total += max(0.0, low - share) + max(0.0, share - high)
    return total


def _split_record(  # noqa: PLR0913 - the split's parts, each recorded
    unique_dates: np.ndarray,
    held: np.ndarray,
    k: int,
    counts: dict[int, np.ndarray],
    single: list[int],
    infeasible: list[int],
    seed: int,
) -> dict[str, Any]:
    arms_record = {}
    for a, c in counts.items():
        arms_record[str(a)] = {
            "dates": int(np.count_nonzero(c)),
            "animals": int(c.sum()),
            "held_out_dates": int(np.count_nonzero(c[held])),
            "held_out_animals": int(c[held].sum()),
            "held_out_share": float(c[held].sum() / c.sum()),
            "single_date": a in single,
            "share_infeasible": a in infeasible,
        }
    return {
        "seed": seed,
        "k": k,
        "draws_allowed": SPLIT_DRAWS,
        "held_out_probability": HELD_OUT_PROBABILITY,
        "arm_share": list(ARM_SHARE),
        "wild_type_share": list(WILD_TYPE_SHARE),
        "sides": {
            str(int(d)): ("held_out" if h else "bands")
            for d, h in zip(unique_dates, held, strict=True)
        },
        "arms": arms_record,
    }


def write_split(tracks: Path, out: Path) -> dict[str, Any]:
    """Draw the split from the store's dates and arms, and write it."""
    with np.load(tracks) as store:
        dates, arms = store["date"], store["genotype"]
        names = [str(g) for g in store["genotypes"]]
    record = draw_split(dates, arms, names.index(WILD_TYPE_OD2))
    record["arm_names"] = {str(i): n for i, n in enumerate(names)}
    record["drawn_from"] = "each animal's recording date and arm; no behavioural field"
    record["instrument_note"] = (
        "Held out of the bands, not unseen by the instrument: the roaming/dwelling classifier was "
        "calibrated on an animal-level split (seed 2026) across every wild-type date, and the "
        "patch-leaving reference used every wild-type animal."
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    return record


# ----------------------------------------------------------------------------------------------
# Calhoun 2014
# ----------------------------------------------------------------------------------------------

CALHOUN_FRAMES = 5400
CALHOUN_FLAGS = ("Pirouettes", "Omega turns", "Non-omega turns")


def _calhoun_number(value: Any) -> float:  # noqa: ANN401 - a spreadsheet cell
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def extract_calhoun(xlsx: Path, out: Path) -> dict[str, Any]:
    """Write Calhoun's tracks: positions at 1 s in pixels and per-frame onset flags."""
    import openpyxl

    book = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    sheets = {ws.title: [list(r) for r in ws.iter_rows(values_only=True)] for ws in book.worksheets}
    malformed: dict[str, int] = {}
    positions = {}
    for axis, name in (("x", "X position"), ("y", "Y position")):
        rows = sheets[name]
        for row in rows:
            for v in row:
                if isinstance(v, str) and v != "NaN":
                    malformed[v] = malformed.get(v, 0) + 1
        positions[axis] = np.array([[_calhoun_number(v) for v in r[:CALHOUN_FRAMES]] for r in rows])
    widths = {name: len(sheets[name][0]) for name in CALHOUN_FLAGS}
    flags = [
        np.array([[bool(v) for v in r[:CALHOUN_FRAMES]] for r in sheets[name]])
        for name in CALHOUN_FLAGS
    ]
    x_px = positions["x"][:, ::POSITION_STRIDE]
    y_px = positions["y"][:, ::POSITION_STRIDE]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x_cpx=quantise(x_px, 100.0, np.int32),
        y_cpx=quantise(y_px, 100.0, np.int32),
        frame_flags=pack_flags(flags),
        flag_names=np.array(["pirouette", "omega", "non_omega"]),
        units=np.array("px"),
        frames_per_second=np.array(3.0),
    )
    return {
        "worms": int(x_px.shape[0]),
        "frames": CALHOUN_FRAMES,
        "sheet_widths": widths,
        "flag_onsets": {name: int(f.sum()) for name, f in zip(CALHOUN_FLAGS, flags, strict=True)},
        "missing_positions": int(np.isnan(positions["x"]).sum() + np.isnan(positions["y"]).sum()),
        "malformed_cells": malformed,
    }


# ----------------------------------------------------------------------------------------------
# Roberts 2016
# ----------------------------------------------------------------------------------------------

_IBW_TYPES = {2: "f4", 4: "f8", 8: "i1", 16: "i2", 32: "i4", 0x48: "u1", 0x50: "u2", 0x60: "u4"}
_IBW_V2_HEADER = 16
_IBW_V2_DATA = 126
_IBW_V5_HEADER = 64
_IBW_V5_DATA = 384
ROBERTS_FPS = 29.97
ROBERTS_WILD_TYPE = ("C15", "C21", "C57", "C162", "C186")
# The authors' Viterbi state indices (``wormtracks.ipf``: stateX = 0, stateR = 1, stateF = 2,
# stateY = 3); X follows forward runs and Y reversals, both paused.
ROBERTS_VITERBI_STATES = ("X", "R", "F", "Y")
# The tracker's state per frame (``wormtracks.ipf``, GetData): 2 forward, 1 reverse, 0 paused.
ROBERTS_TRACKER_STATES = {0: "pause", 1: "reverse", 2: "forward"}
# The authors' program drops two points at each end of every worm before decoding.
ROBERTS_VITERBI_TRIM = 2


def read_ibw(path: Path) -> np.ndarray:
    """Read one numeric Igor binary wave (versions 2 and 5), in either byte order."""
    raw = path.read_bytes()
    little = struct.unpack("<h", raw[:2])[0] in (2, 5)
    order = "<" if little else ">"
    version = struct.unpack(order + "h", raw[:2])[0]
    if version == 2:
        kind = struct.unpack(order + "h", raw[_IBW_V2_HEADER : _IBW_V2_HEADER + 2])[0]
        count = struct.unpack(order + "i", raw[_IBW_V2_HEADER + 42 : _IBW_V2_HEADER + 46])[0]
        offset = _IBW_V2_DATA
    elif version == 5:
        kind = struct.unpack(order + "h", raw[_IBW_V5_HEADER + 16 : _IBW_V5_HEADER + 18])[0]
        count = struct.unpack(order + "i", raw[_IBW_V5_HEADER + 12 : _IBW_V5_HEADER + 16])[0]
        offset = _IBW_V5_DATA
    else:
        msg = f"{path.name}: Igor binary wave version {version} is not read here"
        raise ValueError(msg)
    if kind not in _IBW_TYPES:
        msg = f"{path.name}: wave type {kind} is not a plain numeric type"
        raise ValueError(msg)
    dtype = np.dtype(_IBW_TYPES[kind]).newbyteorder(order)
    return np.frombuffer(raw, dtype=dtype, count=count, offset=offset).astype(
        dtype.newbyteorder("="),
    )


def _seconds(n_frames: int) -> np.ndarray:
    """Return the frame nearest each whole second of a 29.97 frames/s recording."""
    seconds = int(np.floor((n_frames - 1) / ROBERTS_FPS)) + 1
    return np.minimum(np.rint(np.arange(seconds) * ROBERTS_FPS).astype(int), n_frames - 1)


def roberts_cohort(folder: Path, viterbi: Path | None) -> dict[str, Any]:
    """One cohort's worms: positions at 1 s, per-frame tracker states and Viterbi states."""
    xx, yy = read_ibw(folder / "xx.ibw"), read_ibw(folder / "yy.ibw")
    state, worm = read_ibw(folder / "state.ibw"), read_ibw(folder / "worm.ibw")
    vit = read_ibw(viterbi) if viterbi is not None else None
    worms = []
    cursor = 0
    for w in np.unique(worm[np.isfinite(worm)]):
        rows = np.flatnonzero(worm == w)
        if not np.all(np.diff(rows) == 1):
            msg = f"{folder.name}: worm {w} is not contiguous"
            raise ValueError(msg)
        frames = _seconds(len(rows))
        entry = {
            "worm": int(w),
            "x_mm": xx[rows][frames],
            "y_mm": yy[rows][frames],
            "state": np.nan_to_num(state[rows], nan=-1).astype(np.int8),
        }
        if vit is not None:
            kept = len(rows) - 2 * ROBERTS_VITERBI_TRIM
            v = np.full(len(rows), -1, dtype=np.int8)
            v[ROBERTS_VITERBI_TRIM : ROBERTS_VITERBI_TRIM + kept] = vit[cursor : cursor + kept]
            cursor += kept
            entry["viterbi"] = v
        worms.append(entry)
    if vit is not None and cursor != len(vit):
        msg = f"{folder.name}: the Viterbi wave has {len(vit)} points, the worms {cursor}"
        raise ValueError(msg)
    return {"worms": worms, "frames": len(xx)}


def extract_roberts(archive: Path, out: Path) -> dict[str, Any]:
    """Write Roberts's wild-type cohorts and dead worms, each worm as one row of padded arrays."""
    entries: list[dict[str, Any]] = []
    record: dict[str, Any] = {"cohorts": {}}
    for cohort in ROBERTS_WILD_TYPE:
        folder = archive / "Wild type" / cohort
        part = roberts_cohort(folder, archive / "IBW files" / f"{cohort}_Viterbi.ibw")
        for e in part["worms"]:
            e["cohort"] = cohort
        entries += part["worms"]
        record["cohorts"][cohort] = {"worms": len(part["worms"]), "frames": part["frames"]}
    dead = roberts_cohort(archive / "dead worm", None)
    for e in dead["worms"]:
        e["cohort"] = "dead"
    entries += dead["worms"]
    record["cohorts"]["dead"] = {"worms": len(dead["worms"]), "frames": dead["frames"]}
    seconds = max(len(e["x_mm"]) for e in entries)
    frames = max(len(e["state"]) for e in entries)
    n = len(entries)
    x = np.full((n, seconds), np.nan)
    y = np.full((n, seconds), np.nan)
    state = np.full((n, frames), -1, dtype=np.int8)
    vit = np.full((n, frames), -1, dtype=np.int8)
    for i, e in enumerate(entries):
        x[i, : len(e["x_mm"])], y[i, : len(e["y_mm"])] = e["x_mm"], e["y_mm"]
        state[i, : len(e["state"])] = e["state"]
        if "viterbi" in e:
            vit[i, : len(e["viterbi"])] = e["viterbi"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x_um=quantise_um(x, np.int32),
        y_um=quantise_um(y, np.int32),
        state=state,
        viterbi=vit,
        cohort=np.array([e["cohort"] for e in entries]),
        worm=np.array([e["worm"] for e in entries], dtype=np.int16),
        frames=np.array([len(e["state"]) for e in entries], dtype=np.int32),
        frames_per_second=np.array(ROBERTS_FPS),
        tracker_states=np.array(
            [ROBERTS_TRACKER_STATES[k] for k in sorted(ROBERTS_TRACKER_STATES)],
        ),
        viterbi_states=np.array(ROBERTS_VITERBI_STATES),
    )
    record["state_counts"] = {
        str(k): int(np.sum(state == k)) for k in (-1, *sorted(ROBERTS_TRACKER_STATES))
    }
    record["viterbi_counts"] = {str(k): int(np.sum(vit == k)) for k in range(-1, 4)}
    return record


# ----------------------------------------------------------------------------------------------
# López-Cruz 2019
# ----------------------------------------------------------------------------------------------

_NOT_AVAILABLE = {"", "N/A", "NA", "-"}
_MEAN_SEM = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*\+/-\s*([-+]?\d*\.?\d+)")


def parse_mean_sem(cell: Any) -> tuple[float, float] | None:  # noqa: ANN401 - a spreadsheet cell
    """Parse an "m +/- s" cell; ``None`` for an empty or "N/A" one."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    if str(cell).strip().upper() in _NOT_AVAILABLE:
        return None
    match = _MEAN_SEM.match(str(cell))
    if match is None:
        msg = f"cannot read {cell!r} as mean +/- SEM"
        raise ValueError(msg)
    return float(match.group(1)), float(match.group(2))


def extract_lopezcruz(xls: Path, out: Path) -> dict[str, Any]:
    """Write every row of the 2-min-bin sheet, each mutant linked to its same-dates wild type."""
    try:
        import xlrd  # pyright: ignore[reportMissingModuleSource]
    except ImportError as error:  # pragma: no cover - depends on how the script is run
        msg = "xlrd is needed: run with `uv run --with xlrd`"
        raise SystemExit(msg) from error
    sheet = xlrd.open_workbook(str(xls)).sheet_by_name("data_2min_bins")
    header = [str(v) for v in sheet.row_values(2)]
    bins = header[3:-1]
    rows: list[dict[str, Any]] = []
    for r in range(3, sheet.nrows):
        values = sheet.row_values(r)
        genotype = str(values[0]).strip()
        if not genotype:
            continue
        parsed = [parse_mean_sem(v if v != "" else None) for v in values[3:-1]]
        n = values[-1]
        rows.append(
            {
                "row": r,
                "genotype": genotype,
                "figure": str(values[1]).strip(),
                "dates": str(values[2]).strip(),
                "n": int(n) if n != "" else None,
                "mean": [p[0] if p else None for p in parsed],
                "sem": [p[1] if p else None for p in parsed],
            },
        )
    for i, row in enumerate(rows):
        if row["genotype"].lower().startswith("wt"):
            continue
        control = next(
            (
                j
                for j in range(i - 1, -1, -1)
                if rows[j]["genotype"].lower().startswith("wt") and rows[j]["dates"] == row["dates"]
            ),
            None,
        )
        row["control_row"] = rows[control]["row"] if control is not None else None
    absent = [
        g
        for g in ("pdfr-1", "tax-4", "tph-1")
        if not any(r["genotype"].startswith(g) for r in rows)
    ]
    record = {
        "sheet": "data_2min_bins",
        "title": str(sheet.cell_value(0, 3)),
        "bins": bins,
        "rows": rows,
        "absent_d37_mutants": absent,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    return {"rows": len(rows), "absent_d37_mutants": absent}


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------
# Chen 2025
# ----------------------------------------------------------------------------------------------

CHEN_TRACKS_MEMBER = "temp/data4plots/track_learn_all.mat"
CHEN_FOLDERS_MEMBER = "temp/data4plots/learn_folders4.mat"
CHEN_LANDSCAPE_MEMBER = "temp/data4plots/Landscape_low_0623_2.mat"
CHEN_RAW_MEMBER = "temp/track_data/rawTracks_N2nai.mat"
CHEN_NAIVE = 1  # the naive condition's index in track_learn_all, in folder_nai order
CHEN_STANDARD = ("Data_learn/N2/", "_11mM_35_400")
CHEN_LANDSCAPE_STRIDE = 5  # landscape pixels per vendored cell
CHEN_SCALE_MIN_SPEED = 0.05  # mm/s; slower samples are dominated by noise in the speed ratio


def _chen_load(archive: Path, member: str) -> dict[str, Any]:
    import io
    import zipfile

    import scipy.io as sio

    with zipfile.ZipFile(archive) as z:
        return sio.loadmat(io.BytesIO(z.read(member)), squeeze_me=True, struct_as_record=False)


def chen_mm_per_px(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the pixel scale, from the authors' smoothed speeds.

    It is the median, over tracks, of the authors' smoothed speed (mm/s) against the same smoothed
    track's speed in pixels per second.
    """
    ratios = []
    for t in raw["Tracks"]:
        x, y, tt, sp = (np.atleast_1d(v) for v in (t.SmoothX, t.SmoothY, t.Time, t.SmoothSpeed))
        if len(x) < 200:
            continue
        v = np.hypot(np.diff(x), np.diff(y)) / np.diff(tt)
        ok = (sp[1:] > CHEN_SCALE_MIN_SPEED) & np.isfinite(v) & (v > 0)
        if ok.sum() > 50:
            ratios.append(float(np.median(sp[1:][ok] / v[ok])))
    r = np.array(ratios)
    return {
        "mm_per_px": float(np.median(r)),
        "iqr": [float(np.quantile(r, 0.25)), float(np.quantile(r, 0.75))],
        "tracks": len(r),
    }


def resample_fragment(path_px: np.ndarray, time_s: np.ndarray) -> tuple[int, np.ndarray]:
    """Return a fragment's first whole second and its positions at whole seconds, interpolated."""
    first, last = math.ceil(time_s[0] - 1e-9), math.floor(time_s[-1] + 1e-9)
    if last < first:
        return first, np.empty((0, 2))
    grid = np.arange(first, last + 1, dtype=float)
    xy = np.column_stack([np.interp(grid, time_s, path_px[:, k]) for k in (0, 1)])
    return first, xy


def extract_chen(archive: Path, out: Path) -> dict[str, Any]:
    """Write Chen 2025's naive N2 butanone tracks at 1 s in mm, by recording, and the field."""
    raw = _chen_load(archive, CHEN_RAW_MEMBER)
    scale = chen_mm_per_px(raw)
    del raw
    folders = [str(f) for f in _chen_load(archive, CHEN_FOLDERS_MEMBER)["folder_nai"]]
    naive = _chen_load(archive, CHEN_TRACKS_MEMBER)["track_learn_all"][CHEN_NAIVE]
    s = scale["mm_per_px"]
    recording, start, length, xs, ys = [], [], [], [], []
    for r, tracks in enumerate(naive):
        for t in np.atleast_1d(tracks):
            p = np.atleast_2d(t.Path).astype(float)
            first, xy = resample_fragment(p, np.atleast_1d(t.Time).astype(float))
            if len(xy) < 2:
                continue
            recording.append(r)
            start.append(first)
            length.append(len(xy))
            xs.append(xy[:, 0] * s)
            ys.append(xy[:, 1] * s)
    landscape = _chen_load(archive, CHEN_LANDSCAPE_MEMBER)["vq1"]
    # Stored rotated by 180 degrees: index [H - 1 - y, W - 1 - x] gives the track frame's [y, x].
    field = np.rot90(landscape, 2)[::CHEN_LANDSCAPE_STRIDE, ::CHEN_LANDSCAPE_STRIDE]
    dates = [Path(f).parent.name.split("_")[0] for f in folders]
    protocol = [
        "standard" if all(tag in f for tag in CHEN_STANDARD) else "off_protocol" for f in folders
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x_um=quantise_um(np.concatenate(xs), np.int32),
        y_um=quantise_um(np.concatenate(ys), np.int32),
        fragment_recording=np.array(recording, dtype=np.int16),
        fragment_start_s=np.array(start, dtype=np.int32),
        fragment_length_s=np.array(length, dtype=np.int32),
        recording_date=np.array(dates),
        recording_protocol=np.array(protocol),
        recording_folder=np.array([Path(f).parent.name for f in folders]),
        field=field.astype(np.float32),
        field_cell_mm=np.array(CHEN_LANDSCAPE_STRIDE * s),
        mm_per_px=np.array(s),
    )
    return {
        "pixel_scale": scale,
        "recordings": [
            {"date": d, "protocol": p, "fragments": int(sum(1 for x in recording if x == r))}
            for r, (d, p) in enumerate(zip(dates, protocol, strict=True))
        ],
        "fragments": len(recording),
        "seconds": int(sum(length)),
        "field_shape": list(field.shape),
        "field_range": [float(np.nanmin(field)), float(np.nanmax(field))],
    }


def main() -> int:
    """CLI: run one source's extraction and print its record."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest="source", required=True)
    s = sub.add_parser("scheer")
    s.add_argument("--genotypes", type=Path, required=True)
    s.add_argument("--fig1", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--record", type=Path, required=True)
    p = sub.add_parser("split")
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("calhoun")
    c.add_argument("--xlsx", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("roberts")
    r.add_argument("--archive", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    ch = sub.add_parser("chen")
    ch.add_argument("--archive", type=Path, required=True)
    ch.add_argument("--out", type=Path, required=True)
    lc = sub.add_parser("lopezcruz")
    lc.add_argument("--xls", type=Path, required=True)
    lc.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.source == "scheer":
        record = extract_scheer(args.genotypes, args.fig1, args.out)
        fig = record.pop("fig1", {})
        record["fig1"] = {k: v for k, v in fig.items() if k not in ("store_rows", "fig1_rows")}
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n")
    elif args.source == "split":
        record = write_split(args.tracks, args.out)
        record = {k: v for k, v in record.items() if k != "sides"}
    elif args.source == "calhoun":
        record = extract_calhoun(args.xlsx, args.out)
    elif args.source == "roberts":
        record = extract_roberts(args.archive, args.out)
    elif args.source == "chen":
        record = extract_chen(args.archive, args.out)
    else:
        record = extract_lopezcruz(args.xls, args.out)
    print(json.dumps(record, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
