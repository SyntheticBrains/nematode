#!/usr/bin/env python
r"""B v4's controls: the round trip, the reachability oracle, the read and the verdict.

Every stage but ``preflight`` refuses until its part of launch record 124 is committed:

``round-trip``
    After Part A: Scheer's wild-type band animals exported to CSV and WCON and imported again,
    their activity tallies against the native path's. Writes ``round_trip.json`` once.
``oracle-fit``
    After Part A, the bands and the round trip: C.1's 25-parameter model fitted by sep-CMA to the
    three activity statistics alone, on the small lawn, two restarts from the box centre,
    resumable. Writes ``oracle_fit.json``.
``oracle-check``
    After Part B: the fitted answer on 40 fresh populations. The onset rate is read at the bar it
    would be graded at; the pause share and the flip fraction at the bar that reading implies.
    Writes ``oracle_check.json`` once, from which ``benchmark_bands.py --version v4 --onset``
    records the onset decision.
``read``
    After Part C (the onset decision, with the decided bands): fresh N2, the adversary, the
    negatives, C.1's v3 answer and PD1074's held-out cohorts reported, W's minimum and the 3-of-5
    probability. Writes ``read.json`` once.
``verdict``
    VALID only if every registered check holds. Writes ``controls_v4.json`` with the bands'
    SHA-256, which makes v4 the grading default.
``preflight``
    Discarded seeds, before Part A, on bands built in process by the registered rules: the
    adversary on 5 populations and one 60-generation oracle restart, with its answer on 5.

Usage::

    uv run python scripts/analysis/benchmark_v4_controls.py preflight
    uv run python scripts/analysis/benchmark_v4_controls.py round-trip
    uv run python scripts/analysis/benchmark_v4_controls.py oracle-fit
    uv run python scripts/analysis/benchmark_v4_controls.py oracle-check
    uv run python scripts/analysis/benchmark_v4_controls.py read
    uv run python scripts/analysis/benchmark_v4_controls.py verdict
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_bands as bb
import benchmark_controls as bc
import benchmark_grade as bg
import benchmark_v2_controls as v2c
import benchmark_v3_controls as v3c
import c1_v3_fit as c1v3
import npe_recovery as nr
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import measurement as mm
from quantumnematode.benchmark import references as refs

DESCRIPTION = "B v4's controls: the round trip, the reachability oracle, the read and the verdict."
_REPO = Path(__file__).resolve().parents[2]
SUPPORT = bb.V4_SUPPORT_PATH
LAUNCH_PATH = bb.V4_LAUNCH_PATH
BANDS_V4_RECORD = SUPPORT / "bands_v4_record.json"
ROUND_TRIP_PATH = SUPPORT / "round_trip.json"
ORACLE_FIT_PATH = SUPPORT / "oracle_fit.json"
ORACLE_CHECK_PATH = bb.V4_ORACLE_CHECK_PATH
READ_PATH = SUPPORT / "read.json"
PREFLIGHT_PATH = SUPPORT / "preflight.json"
BUILD = _REPO / "build" / "benchmark-v4"
READ_STARTED = BUILD / "read.started"
ORACLE_STATE = BUILD / "oracle"
PREFLIGHT_STATE = BUILD / "preflight-oracle"
PROVISIONAL_BANDS = BUILD / "provisional_bands.json"

ANIMALS = nr.ANIMALS
# The fit's index is 10,000·r + 16·(n - 1) + m, so each restart spans 2,000,000 animal seeds:
# restart 0 610.0-610.32 M, restart 1 612.0-612.32 M, clear of every other block.
ORACLE_FIT_BASE = 610_000_000
ORACLE_CHECK_BASE = 602_000_000
ADVERSARY_BASE = 603_000_000
NEGATIVE_BASE = 604_000_000
C1_V3_BASE = c1v3.GRADE_BASE
PREFLIGHT_ADVERSARY_BASE = 606_000_000
PREFLIGHT_ORACLE_BASE = 607_000_000
ORACLE_CMA_SEEDS = (9431, 9432)
PREFLIGHT_CMA_SEED = 9433
ORACLE_CAP = 100
PREFLIGHT_CAP = 60
# The preflight answer's populations, above every generation's index under its cap (16 x 59 + 15).
PREFLIGHT_ANSWER_OFFSET = 9_000
ORACLE_POPULATIONS = 40
ORACLE_MEAN_Z = 1.0
ORACLE_SHARE = 0.9
GRADE_POPULATIONS = 5
NEEDED = 3
N2_ACTIVITY_SEED = 2047
N2_ALLOWANCE = 1
UNIFORM_FAILURE_RATE = v3c.UNIFORM_FAILURE_RATE
ONSET = bb.ONSET
# The reachability the read stops on; the onset rate's reading decides only whether it is graded.
GATED = tuple(s for s in g.ACTIVITY_STATISTICS if s != ONSET)
BAR_GRADED = g.z_star(bb.v4_statistics(onset_graded=True))
BAR_REPORTED = g.z_star(bb.v4_statistics(onset_graded=False))
NEGATIVE_ORDER = v3c.NEGATIVE_ORDER
SMALL_LAWN = ("small_lawn",)


def population_seeds(base: int, k: int, animals: int = ANIMALS) -> list[int]:
    """Return population k's animal seeds, ``base + 200·k + i``."""
    return [base + ANIMALS * k + i for i in range(animals)]


def onset_key(onset_graded: bool) -> str:  # noqa: FBT001 - the one decision
    """Return the bands record's key for the rule checks at the decided bar."""
    return "graded" if onset_graded else "reported"


def _committed(bound: tuple[Path, ...], part: str) -> bool:
    import cma_recovery as cr

    if not cr.launch_committed(bound=bound, launch_path=LAUNCH_PATH):
        return False
    return part == "A" or f"\n## Part {part}" in LAUNCH_PATH.read_text()


def _part_a() -> bool:
    return _committed((), "A")


def _part_b() -> bool:
    """Return whether Part B is committed, with the bands, their record, the round trip and fit."""
    return _committed((bg.BANDS_V4_PATH, BANDS_V4_RECORD, ROUND_TRIP_PATH, ORACLE_FIT_PATH), "B")


def _part_c() -> bool:
    """Return whether Part C is committed, with the decided bands and the oracle's check."""
    bound = (bg.BANDS_V4_PATH, BANDS_V4_RECORD, ROUND_TRIP_PATH, ORACLE_FIT_PATH, ORACLE_CHECK_PATH)
    if not _committed(bound, "C"):
        return False
    decided = bg.load_bands_v4()["activity_graded"]
    check = json.loads(ORACLE_CHECK_PATH.read_text())
    return decided is not None and (ONSET in decided) == bool(check["onset_graded"])


def _write_once(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        out.write(json.dumps(bg._finite(record), indent=2, allow_nan=False) + "\n")


@functools.cache
def _bands_at(path: str) -> dict[str, Any]:
    return bg.load_bands_v4(Path(path))


# ---------------------------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------------------------


def activity_round_trip(directory: Path) -> dict[str, Any]:
    """Return the wild type's band animals' native and imported activity tallies, and whether equal.

    Each animal is written as a CSV with its sidecar and as WCON, on the small lawn's metadata, and
    read back through the importer.
    """
    import track_import_check as tic
    from quantumnematode.benchmark import track_import as ti

    tracks = refs.load_scheer_tracks()
    rows = bb.band_rows(tracks, bb.WILD_TYPE)
    t = np.arange(tracks.x.shape[1], dtype=float)
    native = []
    imported: dict[str, list[np.ndarray]] = {fmt: [] for fmt in tic.FORMATS}
    for i in rows:
        x, y, radii = tracks.x[i], tracks.y[i], tracks.boundary_radius[i]
        native.append(g.activity_tally(x, y, g.real_on_lawn(x, y, radii)))
        metadata = tic._lawn_metadata("small_lawn", radii)
        for fmt in tic.FORMATS:
            path = tic.write_track(directory, f"activity-{i}", fmt, t, x, y, metadata)
            family, animals = ti.import_population(path)
            if family != "small_lawn":
                msg = f"animal {i} was imported as {family!r}, not the small lawn"
                raise RuntimeError(msg)
            steps = animals[0]
            imported[fmt].append(
                g.activity_tally(
                    np.array([s.x for s in steps], dtype=float),
                    np.array([s.y for s in steps], dtype=float),
                    np.array([bool(s.on_lawn) for s in steps]),
                ),
            )
    result: dict[str, Any] = {
        "arm": bb.WILD_TYPE,
        "animals": len(rows),
        "native_statistics": g.activity_statistics(np.sum(native, axis=0)),
    }
    for fmt in tic.FORMATS:
        tallies = np.vstack(imported[fmt])
        result[fmt] = {
            "statistics": g.activity_statistics(tallies.sum(axis=0)),
            "tallies_equal": bool(np.array_equal(tallies, np.vstack(native))),
        }
    result["passes"] = all(result[fmt]["tallies_equal"] for fmt in tic.FORMATS)
    return result


def _round_trip(_args: argparse.Namespace) -> int:
    if not _part_a():
        print(f"{LAUNCH_PATH} (Part A) is not committed and clean", file=sys.stderr)
        return 1
    if ROUND_TRIP_PATH.exists():
        print(f"{ROUND_TRIP_PATH} exists: the round trip is read once", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as directory:
        trip = activity_round_trip(Path(directory))
    record = {
        "description": "B v4's round trip: Scheer's wild-type band animals through CSV and WCON, "
        "their activity tallies against the native path's.",
        **trip,
    }
    _write_once(ROUND_TRIP_PATH, record)
    print(json.dumps(bg._finite(record), indent=2))
    return 0 if record["passes"] else 1


# ---------------------------------------------------------------------------------------------
# The reachability oracle
# ---------------------------------------------------------------------------------------------


def activity_zs(
    animals: list[list[Any]],
    bands: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Return a small-lawn population's three activity statistics and their capped z's.

    Read on the bands' activity bands at the population's size, whether or not the onset rate is
    graded; an undefined statistic is at the cap.
    """
    values = bg.activity_values(animals)
    band = bg.family_bands_v2(bands, "activity", len(animals))
    return {
        name: {"value": values[name], "z": g.z_score(values[name], band[name])}
        for name in g.ACTIVITY_STATISTICS
    }


def oracle_objective(zs: dict[str, dict[str, float]]) -> float:
    """Return the oracle's objective: the RMS of the three activity statistics' capped z."""
    return float(math.sqrt(np.mean([r["z"] ** 2 for r in zs.values()])))


def _mechanistic_small_lawn(z: list[float], seeds: list[int], bands: dict[str, Any]) -> list[Any]:
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    params = c1v3.parameters(z)
    populations, _ = bg.simulate_v2(
        "mlp",
        lambda _path: BatchedMechanistic(params),
        seeds,
        bands,
        SMALL_LAWN,
    )
    return populations["small_lawn"]


def _oracle_simulate(k: int, z: list[float]) -> float:
    """Return the oracle's objective at a unit-box point on fit simulation k."""
    bands = _bands_at(str(bg.BANDS_V4_PATH))
    animals = _mechanistic_small_lawn(z, population_seeds(ORACLE_FIT_BASE, k), bands)
    return oracle_objective(activity_zs(animals, bands))


def _oracle_preflight_simulate(k: int, z: list[float]) -> float:
    bands = _bands_at(str(PROVISIONAL_BANDS))
    animals = _mechanistic_small_lawn(z, population_seeds(PREFLIGHT_ORACLE_BASE, k), bands)
    return oracle_objective(activity_zs(animals, bands))


def _oracle_population(item: tuple[int, int, list[float], str]) -> dict[str, Any]:
    k, base, z, bands_path = item
    bands = _bands_at(bands_path)
    animals = _mechanistic_small_lawn(z, population_seeds(base, k), bands)
    return {"k": k, "activity": activity_zs(animals, bands)}


def oracle_rule(zs: list[float], bar: float) -> dict[str, Any]:
    """Return one statistic's oracle rule: its mean z in [-1, 1] and |z| <= bar in 90%."""
    mean = float(np.mean(zs))
    share = float(np.mean([abs(z) <= bar for z in zs]))
    return {
        "mean_z": mean,
        "share_within_bar": share,
        "z_sd": float(np.std(zs, ddof=1)),
        "bar": bar,
        "passes": abs(mean) <= ORACLE_MEAN_Z and share >= ORACLE_SHARE,
    }


def oracle_reading(populations: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the oracle's check: the onset decision, then the pause share and the flip fraction.

    The onset rate is read at the bar it would be graded at (24 statistics). If it passes it is
    graded, and the other two are read at that bar; otherwise at the 23-statistic bar.
    """
    zs = {
        name: [float(p["activity"][name]["z"]) for p in populations]
        for name in g.ACTIVITY_STATISTICS
    }
    onset = oracle_rule(zs[ONSET], BAR_GRADED)
    onset_graded = bool(onset["passes"])
    bar = BAR_GRADED if onset_graded else BAR_REPORTED
    rows = {name: oracle_rule(zs[name], bar) for name in GATED}
    return {
        "statistics": {**rows, ONSET: onset},
        "onset_graded": onset_graded,
        "bar": bar,
        "activity_reachable": all(r["passes"] for r in rows.values()),
    }


def _bands_ready() -> str | None:
    if not bg.BANDS_V4_PATH.exists() or not BANDS_V4_RECORD.exists():
        return "v4's bands and their record are not built"
    if not ROUND_TRIP_PATH.exists() or not json.loads(ROUND_TRIP_PATH.read_text())["passes"]:
        return "the round trip has not passed"
    return None


def _oracle_fit(args: argparse.Namespace) -> int:
    import cma_recovery as cr

    if not _part_a():
        print(f"{LAUNCH_PATH} (Part A) is not committed and clean", file=sys.stderr)
        return 1
    refusal = _bands_ready()
    if refusal is not None:
        print(f"{refusal}; the oracle is not fitted", file=sys.stderr)
        return 1
    with ProcessPoolExecutor(max_workers=args.workers, initializer=nr._worker_setup) as pool:
        histories = [
            v2c._cma_restart(
                ORACLE_STATE,
                r,
                seed,
                c1v3.DIMENSION,
                _oracle_simulate,
                ORACLE_CAP,
                pool,
                "oracle",
            )
            for r, seed in enumerate(ORACLE_CMA_SEEDS)
        ]
    best = cr.best_restart([h["medians"] for h in histories])
    answer = histories[best]["answers"][-1]
    record = {
        "description": "B v4's reachability oracle: C.1's 25-parameter model fitted to the three "
        "activity statistics alone (RMS of their capped z), on the small lawn, on animals at 610 M.",
        "bands_sha256": mm.file_sha256(bg.BANDS_V4_PATH),
        "restarts": [
            {
                **{k: h[k] for k in ("restart", "cma_seed", "stop", "medians", "sigmas")},
                "final_objective": cr.final_objective(h["medians"]),
            }
            for h in histories
        ],
        "best_restart": best,
        "answer": answer,
        "parameters": c1v3.parameters(answer),
    }
    ORACLE_FIT_PATH.write_text(json.dumps(bg._finite(record), indent=2, allow_nan=False) + "\n")
    print(json.dumps(bg._finite({k: record[k] for k in ("best_restart", "parameters")}), indent=2))
    return 0


def _oracle_check(args: argparse.Namespace) -> int:
    if not _part_b():
        print(
            f"{LAUNCH_PATH} with Part B, and the files it binds, are not committed and clean; "
            "nothing is read.",
            file=sys.stderr,
        )
        return 1
    if ORACLE_CHECK_PATH.exists():
        print(f"{ORACLE_CHECK_PATH} exists: the oracle's check is read once", file=sys.stderr)
        return 1
    fit = json.loads(ORACLE_FIT_PATH.read_text())
    if fit["bands_sha256"] != mm.file_sha256(bg.BANDS_V4_PATH):
        print("the oracle was fitted on other bands; nothing is read", file=sys.stderr)
        return 1
    items = [
        (p, ORACLE_CHECK_BASE, fit["answer"], str(bg.BANDS_V4_PATH))
        for p in range(ORACLE_POPULATIONS)
    ]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=nr._worker_setup) as pool:
        populations = list(pool.map(_oracle_population, items))
    record = {
        "description": "B v4's oracle check: the fitted answer on 40 populations at 602 M; the "
        "onset decision and the reachability of the pause share and the flip fraction.",
        "bands_sha256": mm.file_sha256(bg.BANDS_V4_PATH),
        "populations": populations,
        **oracle_reading(populations),
    }
    _write_once(ORACLE_CHECK_PATH, record)
    print(json.dumps(bg._finite(record["statistics"]), indent=2))
    print("onset graded:", record["onset_graded"], "; reachable:", record["activity_reachable"])
    # The registered stop: the pause share or the flip fraction is unreachable.
    return 0 if record["activity_reachable"] else 1


# ---------------------------------------------------------------------------------------------
# The positive control: fresh N2
# ---------------------------------------------------------------------------------------------


def n2_activity_control(bands: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Grade the 2 N2 units: v2's halves of N2's unread held-out dates, on N2's own bands.

    Each half is graded on the activity family against bands from N2's band dates at the half's
    size, on ``default_rng([2047, n])`` with PD1074's kappa; the onset rate, if reported, beside.
    """
    tracks = refs.load_scheer_tracks()
    rows = tracks.arm(v2c.N2_ARM)
    band_dates = bb.per_date_activity_tallies(tracks, v2c.N2_ARM)
    unread = v2c.n2_unread_dates(tracks)
    held = rows[np.isin(tracks.date[rows], unread) & (tracks.side[rows] == refs.HELD_OUT)]
    heldout = bb.per_date_activity_tallies(tracks, v2c.N2_ARM, rows=held)
    order = np.random.default_rng(v2c.N2_SPLIT_SEED).permutation(len(unread))
    halves = [
        sorted(unread[k] for k in order[: len(unread) // 2]),
        sorted(unread[k] for k in order[len(unread) // 2 :]),
    ]
    graded = tuple(bands["activity_graded"])
    bar = float(bands["bars"]["wild_type"])
    units = {}
    for h, dates in enumerate(halves):
        tallies = np.vstack([heldout[d] for d in dates])
        n = len(tallies)
        nb = bb.lawn_bands(
            band_dates,
            g.ACTIVITY_STATISTICS,
            n,
            np.random.default_rng([N2_ACTIVITY_SEED, n]),
            kappa=bands["activity_kappa"],
            statistic=g.activity_statistics,
        )
        readings = g.grade(g.activity_statistics(tallies.sum(axis=0)), nb, bar)
        units[f"activity/half{h}"] = {
            "dates": dates,
            "n": n,
            "statistics": bc._reading_dict({k: readings[k] for k in graded}),
            "reported": bc._reading_dict({k: r for k, r in readings.items() if k not in graded}),
            "passes": all(readings[k].passes for k in graded),
        }
    check = record["checks"][onset_key(ONSET in graded)]["activity"]
    rate = 1.0 - check["joint_pass_rate"]
    failed = [u for u, r in units.items() if not r["passes"]]
    return {
        "arm": v2c.N2_ARM,
        "band_dates": len(band_dates),
        "unread_dates": len(unread),
        "split_seed": v2c.N2_SPLIT_SEED,
        "band_seed": N2_ACTIVITY_SEED,
        "units": units,
        "failed": failed,
        "allowance": N2_ALLOWANCE,
        "p_more_than_allowance": bc.allowance_tail([rate] * len(units), N2_ALLOWANCE),
        "passes": len(failed) <= N2_ALLOWANCE,
    }


# ---------------------------------------------------------------------------------------------
# The adversary, the negatives and C.1's v3 answer
# ---------------------------------------------------------------------------------------------


def adversary_answer() -> list[float]:
    """Return C.2's stage-2 answer, the worm that does not run forward."""
    import c2_encoder_probe as p

    return list(json.loads(p.STAGED_PILOT_PATH.read_text())["stage2_extended"]["answer"])


def _adversary_populations(seeds: list[int], bands: dict[str, Any]) -> list[Any]:
    """Run C.2's stage-2 answer on the small lawn, on the wiring it was fitted on."""
    import c2_encoder_probe as p

    p._BRAIN_SETTINGS["sign_override_exclude"] = ["lopezcruz2019"]
    populations, _ = bg.simulate_v2(
        p.ARM,
        p.controller(adversary_answer()),
        seeds,
        bands,
        SMALL_LAWN,
    )
    return populations["small_lawn"]


def _adversary_population(item: tuple[int, int, str]) -> dict[str, Any]:
    k, base, bands_path = item
    bands = _bands_at(bands_path)
    graded = bg.grade_populations_v4(
        {"small_lawn": _adversary_populations(population_seeds(base, k), bands)},
        None,
        bands,
        presence=False,
    )
    activity = bg._finite(graded["families"]["activity"])
    return {
        "k": k,
        "activity": activity,
        "activity_reported": graded.get("activity_reported"),
        "reversal_share": graded["reversal_share"],
        "caught": any(not r["passes"] for r in activity.values()),
    }


def adversary_verdict(populations: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the adversary's populations and whether it is caught in at least 3 of 5."""
    caught = sum(p["caught"] for p in populations)
    return {"populations": populations, "caught_in": caught, "caught_3_of_5": caught >= NEEDED}


def _summary(graded: dict[str, Any]) -> dict[str, Any]:
    return bg._finite(
        {
            "activity": graded["families"]["activity"],
            "activity_reported": graded.get("activity_reported"),
            "family_passes": graded["family_passes"],
            "wild_type_passes": graded["wild_type_passes"],
            "D": graded["D"],
            "D_fit": graded["D_fit"],
            "reversal_share": graded["reversal_share"],
        },
    )


def _negative_population(item: tuple[str, int, int, str]) -> dict[str, Any]:
    name, k, base, bands_path = item
    graded = v3c.negative_graded(
        name,
        population_seeds(base, k),
        _bands_at(bands_path),
        grade=bg.grade_populations_v4,
    )
    return {"name": name, "k": k, **_summary(graded)}


def c1_v3_answer() -> list[float]:
    """Return C.1's v3 answer: its chosen restart's final point."""
    fit = json.loads(c1v3.FIT_PATH.read_text())
    return list(fit["restarts"][fit["chosen_restart"]]["answer"])


def _c1_v3_population(item: tuple[int, str]) -> dict[str, Any]:
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    k, bands_path = item
    bands = _bands_at(bands_path)
    params = c1v3.parameters(c1_v3_answer())
    populations, chemotaxis = bg.simulate_v2(
        "mlp",
        lambda _path: BatchedMechanistic(params),
        population_seeds(C1_V3_BASE, k),
        bands,
    )
    graded = bg.grade_populations_v4(populations, chemotaxis, bands, presence=True)
    return {
        "k": k,
        **_summary(graded),
        "families": bg._finite(graded["families"]),
        "local_search_presence": bg._finite(graded.get("local_search_presence")),
        "chemotaxis_presence": bg._finite(graded.get("chemotaxis_presence")),
    }


def c1_v3_reported(populations: list[dict[str, Any]]) -> dict[str, Any]:
    """Return C.1's v3 answer on v4 beside its v3 grade on the same animals.

    Its v3 families' values on v4 must equal its stored v3 grade's, or the comparison is not like
    for like.
    """
    stored = json.loads(c1v3.FIT_PATH.read_text())["grades"]
    same = all(
        p["families"][family][name]["value"] == reading["value"]
        for p, v3 in zip(populations, stored, strict=True)
        for family, readings in v3["families"].items()
        for name, reading in readings.items()
    )
    passing = sum(p["wild_type_passes"] for p in populations)
    return {
        "base": C1_V3_BASE,
        "populations": populations,
        "v3_values_equal": same,
        "v3_population_passes": [v3["wild_type_passes"] for v3 in stored],
        "v4_population_passes": [p["wild_type_passes"] for p in populations],
        "passes_wild_type_v4": passing >= NEEDED,
        "failing_activity": sorted(
            {name for p in populations for name, r in p["activity"].items() if not r["passes"]},
        ),
    }


def pd1074_held_out(bands: dict[str, Any]) -> dict[str, Any]:
    """Return PD1074's held-out cohorts on the activity family, pooled, as description.

    The planning probe read them, so they are reported, never gated.
    """
    tracks = refs.load_scheer_tracks()
    rows = tracks.arm(bb.WILD_TYPE)
    held = rows[tracks.side[rows] == refs.HELD_OUT]
    per_date = bb.per_date_activity_tallies(tracks, bb.WILD_TYPE, rows=held)
    tallies = np.vstack(list(per_date.values()))
    readings = g.grade(
        g.activity_statistics(tallies.sum(axis=0)),
        bg.family_bands_v2(bands, "activity", len(tallies)),
        float(bands["bars"]["wild_type"]),
    )
    return {
        "animals": len(tallies),
        "dates": len(per_date),
        "statistics": bc._reading_dict(readings),
        "graded": list(bands["activity_graded"]),
    }


# ---------------------------------------------------------------------------------------------
# W's minimum and the 3-of-5 probability
# ---------------------------------------------------------------------------------------------


def w_minimum_v4(bands: dict[str, Any]) -> dict[str, Any]:
    """Return W's minimum on v4's fitted and held-out families, each the 95th percentile.

    v3's rule, with the activity family in the fitted set: the same halves of N2's unread
    held-out dates, the activity difference scaled by PD1074's activity bands at the half's size.
    The held-out set is v3's, unchanged.
    """
    v3_w = v3c.w_minimum_v3(bands)
    family_map = bg._v2_family_map(bands)
    graded = tuple(bands["activity_graded"])
    tracks = refs.load_scheer_tracks()
    clf = bg.classifier(bands)
    rows = tracks.arm(v2c.N2_ARM)
    unread = v2c.n2_unread_dates(tracks)
    held = rows[np.isin(tracks.date[rows], unread) & (tracks.side[rows] == refs.HELD_OUT)]
    held_lawn, held_kin = v2c._tallies_by_date(tracks, held, clf)
    held_act = bb.per_date_activity_tallies(tracks, v2c.N2_ARM, rows=held)
    dates = sorted(held_lawn)
    builder = bg._builder()
    rng = np.random.default_rng(v2c.W_FITTED_SEED)
    fitted = []
    families = ["small_lawn", "kinematics", "activity"]
    for _ in range(v2c.W_SPLITS):
        order = rng.permutation(len(dates))
        sides = [
            [dates[k] for k in order[: len(dates) // 2]],
            [dates[k] for k in order[len(dates) // 2 :]],
        ]
        lawn = [np.vstack([held_lawn[d] for d in side]) for side in sides]
        kin = [np.vstack([held_kin[d] for d in side]) for side in sides]
        act = [np.vstack([held_act[d] for d in side]) for side in sides]
        step = v2c.W_SIZE_STEP
        size = max(step, round(min(len(lawn[0]), len(lawn[1])) / step) * step)
        small, kinematic = builder.small_lawn(size), bg._kinematic_bands_at(size)
        activity = bg.family_bands_v2(bands, "activity", size)
        readings = {
            "small_lawn": v2c._difference(
                g.lawn_statistics(lawn[0].sum(axis=0)),
                g.lawn_statistics(lawn[1].sum(axis=0)),
                {s: small[s].scale for s in g.LAWN_STATISTICS},
            ),
            "kinematics": v2c._difference(
                g.kinematic_statistics(kin[0].sum(axis=0)),
                g.kinematic_statistics(kin[1].sum(axis=0)),
                {s: kinematic[s].scale for s in g.KINEMATIC_STATISTICS},
            ),
            "activity": v2c._difference(
                g.activity_statistics(act[0].sum(axis=0)),
                g.activity_statistics(act[1].sum(axis=0)),
                {s: activity[s].scale for s in graded},
            ),
        }
        fitted.append(g.distance(readings, families, family_map))
    return {
        "fitted": {
            "families": families,
            "source": f"halves of N2's {len(dates)} unread held-out dates, on PD1074's band scales",
            "splits": v2c.W_SPLITS,
            "seed": v2c.W_FITTED_SEED,
            "median": float(np.median(fitted)),
            "minimum_95th": float(np.percentile(fitted, v2c.W_QUANTILE)),
            "v3_minimum_95th": v3_w["fitted"]["minimum_95th"],
        },
        "held_out": v3_w["held_out"],
    }


def three_of_five_v4(record: dict[str, Any], onset_graded: bool) -> dict[str, Any]:  # noqa: FBT001 - the one decision
    """Return a faithful controller's chance of passing the wild type at 3 of 5, by v3's method.

    Each family fails a faithful population at its rate: the rule checks' at v4's decided bar for
    the small lawn, kinematics, local search and activity, B.4's for the uniform well, and the
    chemotaxis leave-one-out's carried from v3.
    """
    checks = record["checks"][onset_key(onset_graded)]
    rates = {
        "small_lawn": 1.0 - checks["small_lawn"]["joint_pass_rate"],
        "uniform_lawn": UNIFORM_FAILURE_RATE,
        "local_search": 1.0 - checks["local_search"]["joint_pass_rate"],
        "kinematics": 1.0 - checks["kinematics"]["joint_pass_rate"],
        "chemotaxis": record["chemotaxis_leave_one_out"]["unit_rate"],
        "activity": 1.0 - checks["activity"]["joint_pass_rate"],
    }
    p = math.prod(1.0 - r for r in rates.values())
    return {
        "failure_rates": rates,
        "population_passes": p,
        "three_of_five": float(scipy_stats.binom.sf(NEEDED - 1, GRADE_POPULATIONS, p)),
    }


# ---------------------------------------------------------------------------------------------
# The read
# ---------------------------------------------------------------------------------------------


def _start_read() -> bool:
    refusal = (
        f"{READ_PATH} or {READ_STARTED} exists: the read is made once, and a started read that "
        "stopped is not repeated."
    )
    if READ_PATH.exists():
        print(refusal, file=sys.stderr)
        return False
    READ_STARTED.parent.mkdir(parents=True, exist_ok=True)
    try:
        with READ_STARTED.open("x") as marker:
            marker.write("the read started\n")
    except FileExistsError:
        print(refusal, file=sys.stderr)
        return False
    return True


def _read(args: argparse.Namespace) -> int:
    if not _part_c():
        print(
            f"{LAUNCH_PATH} with Part C, the decided bands and the files it binds, are not "
            "committed and clean; nothing is read.",
            file=sys.stderr,
        )
        return 1
    if not _start_read():
        return 1
    bands = bg.load_bands_v4()
    path = str(bg.BANDS_V4_PATH)
    record_bands = json.loads(BANDS_V4_RECORD.read_text())
    onset_graded = ONSET in bands["activity_graded"]
    record: dict[str, Any] = {
        "description": "B v4's read: fresh N2, the adversary, the negatives, C.1's v3 answer and "
        "PD1074's held-out cohorts reported, W's minimum and the 3-of-5 probability.",
        "bands_sha256": mm.file_sha256(bg.BANDS_V4_PATH),
        "activity_graded": bands["activity_graded"],
        "bar": bands["bars"]["wild_type"],
        "n2_control": n2_activity_control(bands, record_bands),
    }
    print("N2", record["n2_control"]["failed"], flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=nr._worker_setup) as pool:
        adversary = list(
            pool.map(
                _adversary_population,
                [(k, ADVERSARY_BASE, path) for k in range(GRADE_POPULATIONS)],
            ),
        )
        record["adversary"] = adversary_verdict(adversary)
        print("adversary caught in", record["adversary"]["caught_in"], flush=True)
        jobs = [
            (name, GRADE_POPULATIONS * i + p, NEGATIVE_BASE, path)
            for i, name in enumerate(NEGATIVE_ORDER)
            for p in range(GRADE_POPULATIONS)
        ]
        record["negatives"] = v3c.negative_verdicts(list(pool.map(_negative_population, jobs)))
        print(
            "negatives",
            {n: v["fails_wild_type_in"] for n, v in record["negatives"].items()},
            flush=True,
        )
        c1 = list(pool.map(_c1_v3_population, [(k, path) for k in range(GRADE_POPULATIONS)]))
        record["c1_v3_answer"] = c1_v3_reported(c1)
        print("C.1 v3 on v4", record["c1_v3_answer"]["v4_population_passes"], flush=True)
    record["pd1074_held_out"] = pd1074_held_out(bands)
    record["w_minimum"] = w_minimum_v4(bands)
    record["three_of_five"] = three_of_five_v4(record_bands, onset_graded)
    _write_once(READ_PATH, record)
    return 0


# ---------------------------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------------------------


def verdict(
    record: dict[str, Any],
    round_trip: dict[str, Any],
    oracle_check: dict[str, Any],
    read: dict[str, Any],
) -> dict[str, Any]:
    """Return B v4's validity from its registered readings: VALID only if every check holds."""
    key = onset_key(bool(oracle_check["onset_graded"]))
    negatives = read.get("negatives", {})
    checks = {
        "activity_rule_check": bool(record["checks"][key]["activity"]["passes"]),
        "round_trip": bool(round_trip["passes"]),
        "reachability": bool(oracle_check["activity_reachable"]),
        "n2_control": bool(read.get("n2_control", {}).get("passes", False)),
        "adversary_caught": bool(read.get("adversary", {}).get("caught_3_of_5", False)),
        "negatives_fail_wild_type": all(
            name in negatives and negatives[name]["fails_wild_type_3_of_5"]
            for name in NEGATIVE_ORDER
        ),
    }
    # Beside each family's check the record holds its bar and its count of statistics.
    reported = {
        family: bool(check["passes"])
        for family, check in record["checks"][key].items()
        if isinstance(check, dict) and family != "activity"
    }
    return {
        "checks": checks,
        "reported_rule_checks": reported,
        "verdict": "VALID" if all(checks.values()) else "VOID",
    }


def _verdict(_args: argparse.Namespace) -> int:
    if not _part_c():
        print(
            f"{LAUNCH_PATH} with Part C, and the files it binds, are not committed and clean; no "
            "verdict is recorded.",
            file=sys.stderr,
        )
        return 1
    if not READ_PATH.exists():
        print(f"{READ_PATH} does not exist: the read comes first", file=sys.stderr)
        return 1
    parts = [
        json.loads(p.read_text())
        for p in (BANDS_V4_RECORD, ROUND_TRIP_PATH, ORACLE_CHECK_PATH, READ_PATH)
    ]
    read = parts[3]
    if read["bands_sha256"] != mm.file_sha256(bg.BANDS_V4_PATH):
        print(f"{READ_PATH} was read on other bands; no verdict", file=sys.stderr)
        return 1
    out = {
        "description": "B v4's verdict: the bands record, the round trip, the oracle and the read.",
        "bands_sha256": mm.file_sha256(bg.BANDS_V4_PATH),
        **verdict(*parts),
        "c1_v3_passes_wild_type_v4": read["c1_v3_answer"]["passes_wild_type_v4"],
        "w_minimum": read["w_minimum"],
        "three_of_five": read["three_of_five"],
    }
    bg.CONTROLS_V4_PATH.write_text(json.dumps(bg._finite(out), indent=2, allow_nan=False) + "\n")
    print(json.dumps(out["checks"], indent=2), "\nverdict:", out["verdict"])
    return 0


# ---------------------------------------------------------------------------------------------
# The preflight, on discarded seeds
# ---------------------------------------------------------------------------------------------


def caught_without_onset(population: dict[str, Any]) -> bool:
    """Return whether the pause share or the flip fraction fails at the 23-statistic bar."""
    return any(
        not (r["value"] is not None and abs(float(r["z"])) <= BAR_REPORTED)
        for name, r in population["activity"].items()
        if name in GATED
    )


def _preflight(args: argparse.Namespace) -> int:
    if PREFLIGHT_PATH.exists():
        print(f"{PREFLIGHT_PATH} exists: the preflight is read once", file=sys.stderr)
        return 1
    BUILD.mkdir(parents=True, exist_ok=True)
    start = time.time()
    PROVISIONAL_BANDS.write_text(
        json.dumps(bb.provisional_bands_v4(onset_graded=True), indent=2, allow_nan=False) + "\n",
    )
    record: dict[str, Any] = {
        "description": "B v4's preflight on discarded seeds, before Part A, on bands built in "
        "process by the registered rules: the adversary on 5 populations at 606 M, one "
        "60-generation oracle restart at 607 M (CMA 9433) and its answer on 5 populations.",
        "provisional_bands_sha256": mm.file_sha256(PROVISIONAL_BANDS),
        "bands_seconds": time.time() - start,
    }
    path = str(PROVISIONAL_BANDS)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=nr._worker_setup) as pool:
        start = time.time()
        adversary = list(
            pool.map(
                _adversary_population,
                [(k, PREFLIGHT_ADVERSARY_BASE, path) for k in range(GRADE_POPULATIONS)],
            ),
        )
        record["adversary"] = {
            **adversary_verdict(adversary),
            "caught_without_onset_in": sum(caught_without_onset(p) for p in adversary),
            "seconds": time.time() - start,
        }
        print(
            "adversary caught in",
            record["adversary"]["caught_in"],
            "; without the onset rate",
            record["adversary"]["caught_without_onset_in"],
            flush=True,
        )
        start = time.time()
        history = v2c._cma_restart(
            PREFLIGHT_STATE,
            0,
            PREFLIGHT_CMA_SEED,
            c1v3.DIMENSION,
            _oracle_preflight_simulate,
            PREFLIGHT_CAP,
            pool,
            "preflight oracle",
        )
        record["oracle_restart"] = {
            **{k: history[k] for k in ("cma_seed", "stop", "medians", "sigmas")},
            "answer": history["answers"][-1],
            "seconds": time.time() - start,
            "workers": args.workers,
        }
        populations = list(
            pool.map(
                _oracle_population,
                [
                    (
                        PREFLIGHT_ANSWER_OFFSET + p,
                        PREFLIGHT_ORACLE_BASE,
                        history["answers"][-1],
                        path,
                    )
                    for p in range(GRADE_POPULATIONS)
                ],
            ),
        )
    record["oracle_answer"] = {"populations": populations, **oracle_reading(populations)}
    record["gates"] = {
        "adversary_caught_3_of_5": record["adversary"]["caught_in"] >= NEEDED
        and record["adversary"]["caught_without_onset_in"] >= NEEDED,
        "oracle_reaches_activity": bool(record["oracle_answer"]["activity_reachable"]),
    }
    _write_once(PREFLIGHT_PATH, record)
    print(json.dumps(bg._finite(record["oracle_answer"]["statistics"]), indent=2))
    print(json.dumps(record["gates"]))
    # Either gate failing returns the change to the maintainer before Part A.
    return 0 if all(record["gates"].values()) else 1


def main(argv: list[str] | None = None) -> int:
    """CLI: the preflight, the round trip, the oracle's fit and check, the read and the verdict."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("round-trip")
    for name in ("oracle-fit", "oracle-check", "read", "preflight"):
        sub.add_parser(name).add_argument("--workers", type=int, default=16)
    sub.add_parser("verdict")
    args = parser.parse_args(argv)
    stages = {
        "round-trip": _round_trip,
        "oracle-fit": _oracle_fit,
        "oracle-check": _oracle_check,
        "read": _read,
        "verdict": _verdict,
        "preflight": _preflight,
    }
    return stages[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
