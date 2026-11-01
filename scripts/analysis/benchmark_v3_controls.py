#!/usr/bin/env python
r"""B v3's controls: Chen's round trip, the reachability oracle, the read and the verdict.

Every stage but ``preflight`` refuses until its part of launch record 120 is committed:

``round-trip``
    Chen's standard-protocol fragments, smoothed by the authors and passed through the chemotaxis
    model's jitter, against the raw fragments, on the onset rate and the flip fraction. Writes
    ``round_trip.json`` once.
``oracle-fit``
    Only once the round trip has passed: C.1's model fitted by sep-CMA to the three new
    statistics alone, two restarts from the box centre, resumable. Writes ``oracle_fit.json``.
``read``
    After Part B: the oracle's 40 populations, then, only if they pass, the held-out López-Cruz
    batches, the adversaries, the negatives, W's minimum and the 3-of-5 probability. Writes
    ``read.json`` once.
``verdict``
    VALID only if every registered check holds. Writes ``controls_v3.json`` with the bands'
    SHA-256, which makes v3 the grading default.
``preflight``
    Discarded seeds, before registration: the round trip's jitter, the negatives on two
    populations each and one oracle generation, timed. Reads the bands given.

Usage::

    uv run python scripts/analysis/benchmark_v3_controls.py round-trip \
        --chen-archive build/chen2025/Chen_learning_2024.zip
    uv run python scripts/analysis/benchmark_v3_controls.py oracle-fit
    uv run python scripts/analysis/benchmark_v3_controls.py read
    uv run python scripts/analysis/benchmark_v3_controls.py verdict
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
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
import npe_recovery as nr
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import measurement as mm
from quantumnematode.benchmark import references as refs

DESCRIPTION = (
    "B v3's controls: Chen's round trip, the reachability oracle, the read and the verdict."
)
_REPO = Path(__file__).resolve().parents[2]
SUPPORT = bb.V3_SUPPORT_PATH
LAUNCH_PATH = bb.V3_LAUNCH_PATH
BANDS_V3_RECORD = SUPPORT / "bands_v3_record.json"
ROUND_TRIP_PATH = SUPPORT / "round_trip.json"
ORACLE_FIT_PATH = SUPPORT / "oracle_fit.json"
READ_PATH = SUPPORT / "read.json"
V2_READ_PATH = v2c.READ_PATH
C1B_SUPPORT = _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "119-c1b-fit"
BUILD = _REPO / "build" / "benchmark-v3"
READ_STARTED = BUILD / "read.started"
ORACLE_STATE = BUILD / "oracle"

ANIMALS = nr.ANIMALS
ROUND_TRIP_BASE = 475_000_000
ORACLE_FIT_BASE = 470_000_000
ORACLE_CHECK_BASE = 471_000_000
ADVERSARY_BASE = 472_000_000
NEGATIVE_BASE = 473_000_000
PREFLIGHT_ROUND_TRIP_BASE = 99_970_000
PREFLIGHT_NEGATIVE_BASE = 99_980_000
PREFLIGHT_ORACLE_BASE = 99_990_000
PREFLIGHT_NEGATIVE_POPULATIONS = 2
ORACLE_CMA_SEEDS = (9421, 9422)
ORACLE_CAP = 100
ORACLE_POPULATIONS = 40
ORACLE_MEAN_Z = 1.0
ORACLE_SHARE = 0.9
GRADE_POPULATIONS = 5
NEEDED = 3
# B.4's registered failure rate for the uniform well, whose bands and bar v3 leaves as v2's.
UNIFORM_FAILURE_RATE = 0.05
NEW_STATISTICS = (
    ("local_search", "log_early_late_ratio"),
    ("chemotaxis", "onset_per_min"),
    ("chemotaxis", "flip_fraction"),
)
ORACLE_ASSAYS = ("local_search", "chemotaxis")
ADVERSARY_RESTARTS = (2, 4, 6)
KAPPA_PROBE = "kappa probe"
ANSWERS = ("blind_v2", "c1_v1")
NEGATIVE_ORDER = (*bc.NEGATIVES, *ANSWERS)


def population_seeds(base: int, k: int, animals: int = ANIMALS) -> list[int]:
    """Return population k's animal seeds, ``base + 200·k + i``."""
    return [base + ANIMALS * k + i for i in range(animals)]


def _part_a() -> bool:
    import cma_recovery as cr

    return cr.launch_committed(launch_path=LAUNCH_PATH)


def _part_b() -> bool:
    """Return whether Part B is committed, with the bands, their record, the round trip and fit."""
    import cma_recovery as cr

    bound = (bg.BANDS_V3_PATH, BANDS_V3_RECORD, ROUND_TRIP_PATH, ORACLE_FIT_PATH)
    if not cr.launch_committed(bound=bound, launch_path=LAUNCH_PATH):
        return False
    return "\n## Part B" in LAUNCH_PATH.read_text()


def _write_once(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        out.write(json.dumps(bg._finite(record), indent=2, allow_nan=False) + "\n")


def new_readings(graded: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a grade's readings of v3's three new statistics, keyed ``family/statistic``."""
    return {f"{f}/{s}": graded["families"][f][s] for f, s in NEW_STATISTICS}


# ---------------------------------------------------------------------------------------------
# Chen's round trip
# ---------------------------------------------------------------------------------------------


def chen_round_trip_v3(
    bands: dict[str, Any],
    archive: Path,
    base: int = ROUND_TRIP_BASE,
) -> dict[str, Any]:
    """Return Chen's round trip on v3's five chemotaxis statistics; the two new ones are gated.

    The authors' smoothed fragments gain the chemotaxis model's jitter alone, fragment k on
    ``default_rng([base + k, 2037])``; the fragments' cuts are the real ones.
    """
    from quantumnematode.benchmark import chemotaxis as ch

    chen = refs.load_chen_tracks()
    model = mm.TrackerModel.from_dict(bands["measurement"]["chemotaxis"])
    jitter_only = mm.TrackerModel(jitter_mm=model.jitter_mm, stream_key=model.stream_key)
    fragments = v2c.chen_smoothed(archive)
    raw = [(vx, vy) for _, vx, vy, _, _ in fragments]
    trip = []
    for k, (_, _, _, sx, sy) in enumerate(fragments):
        mx, my = v2c._positions(mm.measure(v2c._as_steps(sx, sy), base + k, jitter_only))
        trip.append((mx, my))

    def pooled(frs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
        return ch.pooled_statistics_v3(ch.fragment_samples(frs, chen.concentration))

    band = bg.family_bands_v2(bands, "chemotaxis", 0)
    gated = {s: band[s] for f, s in NEW_STATISTICS if f == "chemotaxis"}
    rows = v2c._z_rows(pooled(raw), pooled(trip), gated)
    return {
        "fragments": len(fragments),
        "seed_base": base,
        "denoiser": "the authors' SmoothX/SmoothY, resampled to 1 s as the raw paths are",
        "applied": "jitter only; the fragments are the real ones",
        "bar": v2c.ROUND_TRIP_BAR,
        "chemotaxis": rows,
        "passes": all(r["passes"] for r in rows.values() if not r.get("reported")),
    }


def _round_trip(args: argparse.Namespace) -> int:
    if not _part_a():
        print(f"{LAUNCH_PATH} (Part A) is not committed and clean", file=sys.stderr)
        return 1
    if ROUND_TRIP_PATH.exists():
        print(f"{ROUND_TRIP_PATH} exists: the round trip is read once", file=sys.stderr)
        return 1
    record = {
        "description": "B v3's round trip: Chen's smoothed fragments with the model's jitter.",
        "bands_sha256": mm.file_sha256(bg.BANDS_V3_PATH),
        **chen_round_trip_v3(bg.load_bands_v3(), args.chen_archive),
    }
    _write_once(ROUND_TRIP_PATH, record)
    print(json.dumps(bg._finite(record["chemotaxis"]), indent=2), "\npasses:", record["passes"])
    return 0 if record["passes"] else 1


# ---------------------------------------------------------------------------------------------
# The reachability oracle
# ---------------------------------------------------------------------------------------------


def oracle_objective(graded: dict[str, Any]) -> float:
    """Return the oracle's objective: the RMS of the three new statistics' capped z.

    An undefined statistic enters at the cap, as the grading reads it.
    """
    zs = [float(r["z"]) for r in new_readings(graded).values()]
    return float(math.sqrt(np.mean(np.square(zs))))


def _mechanistic_graded(
    params: dict[str, float],
    seeds: list[int],
    bands: dict[str, Any],
    assays: tuple[str, ...] | None = None,
    *,
    presence: bool = True,
) -> dict[str, Any]:
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    populations, chemotaxis = bg.simulate_v2(
        "mlp",
        lambda _path: BatchedMechanistic(params),
        seeds,
        bands,
        assays,
    )
    return bg.grade_populations_v3(populations, chemotaxis, bands, presence=presence)


def _oracle_graded(z: list[float], seeds: list[int], bands: dict[str, Any]) -> dict[str, Any]:
    params = nr.parameters_from_unit(np.asarray(z))
    return _mechanistic_graded(params, seeds, bands, ORACLE_ASSAYS, presence=False)


def _oracle_simulate(k: int, z: list[float]) -> float:
    """Return the oracle's objective at a unit-box point on fit simulation k."""
    graded = _oracle_graded(z, population_seeds(ORACLE_FIT_BASE, k), bg.load_bands_v3())
    return oracle_objective(graded)


def _oracle_fit(args: argparse.Namespace) -> int:
    import cma_recovery as cr

    if not _part_a():
        print(f"{LAUNCH_PATH} (Part A) is not committed and clean", file=sys.stderr)
        return 1
    if not ROUND_TRIP_PATH.exists() or not json.loads(ROUND_TRIP_PATH.read_text())["passes"]:
        print("the round trip has not passed; the oracle is not fitted", file=sys.stderr)
        return 1
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    dimension = len(MECHANISTIC_PARAMETERS)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=nr._worker_setup) as pool:
        histories = [
            v2c._cma_restart(
                ORACLE_STATE,
                r,
                seed,
                dimension,
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
        "description": "B v3's reachability oracle: C.1's model fitted to the three new "
        "statistics alone (RMS of their capped z), on animals at 470 M.",
        "bands_sha256": mm.file_sha256(bg.BANDS_V3_PATH),
        "restarts": [
            {
                **{k: h[k] for k in ("restart", "cma_seed", "stop", "medians", "sigmas")},
                "final_objective": cr.final_objective(h["medians"]),
            }
            for h in histories
        ],
        "best_restart": best,
        "answer": answer,
        "parameters": nr.parameters_from_unit(np.asarray(answer)),
    }
    ORACLE_FIT_PATH.write_text(json.dumps(bg._finite(record), indent=2, allow_nan=False) + "\n")
    print(json.dumps(bg._finite({k: record[k] for k in ("best_restart", "parameters")}), indent=2))
    return 0


def _oracle_check_population(p: int, z: list[float]) -> dict[str, Any]:
    graded = _oracle_graded(z, population_seeds(ORACLE_CHECK_BASE, p), bg.load_bands_v3())
    return bg._finite({"new": new_readings(graded), "reversal_share": graded["reversal_share"]})


def oracle_gate(populations: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the oracle's gate: each new statistic's mean z in [-1, 1] and a 90% share passing.

    Each population holds its new statistics' readings under ``new``. An undefined statistic
    enters at the cap and fails, as the grading reads it.
    """
    rows = {}
    for key in populations[0]["new"]:
        readings = [p["new"][key] for p in populations]
        zs = [g.Z_CAP if r["z"] is None else float(r["z"]) for r in readings]
        mean = float(np.mean(zs))
        share = float(np.mean([bool(r["passes"]) for r in readings]))
        rows[key] = {
            "mean_z": mean,
            "share_within_bar": share,
            "z_sd": float(np.std(zs, ddof=1)),
            "passes": abs(mean) <= ORACLE_MEAN_Z and share >= ORACLE_SHARE,
        }
    return {"statistics": rows, "passes": all(r["passes"] for r in rows.values())}


# ---------------------------------------------------------------------------------------------
# The read: held-out López-Cruz, the adversaries, the negatives, W's minimum, the 3-of-5
# ---------------------------------------------------------------------------------------------


def lopezcruz_held_out(bands: dict[str, Any]) -> dict[str, Any]:
    """Grade each held-out López-Cruz batch's log early/late ratio on v3's band; gamma reported."""
    batches = bb.lopezcruz_batches(refs.load_lopezcruz())
    sides = json.loads(bb.LOPEZCRUZ_SPLIT_PATH.read_text())["sides"]
    band = bg.family_bands_v2(bands, "local_search", 0)
    bar = float(bands["bars"]["wild_type"])
    rows = {}
    for batch, side in sorted(sides.items()):
        if side != refs.HELD_OUT:
            continue
        readings = g.grade(g.log_search_statistics(bb.search_statistics(batches[batch])), band, bar)
        rows[batch] = {
            name: {"value": r.value, "z": r.z, "passes": r.passes} for name, r in readings.items()
        }
    return {
        "batches": rows,
        "count": len(rows),
        "gated": "log_early_late_ratio",
        "passes": all(r["log_early_late_ratio"]["passes"] for r in rows.values()),
    }


def adversary_parameters() -> dict[str, dict[str, float]]:
    """Return C.1b's restarts 2, 4 and 6 and the kappa probe's answers, in the registered order."""
    import c1b_fit as fit

    fitted = json.loads(fit.FIT_PATH.read_text())
    by_restart = {r["restart"]: r for r in fitted["restarts"]}
    out = {
        f"restart {r}": nr.parameters_from_unit(np.asarray(by_restart[r]["answer"]))
        for r in ADVERSARY_RESTARTS
    }
    kappa = json.loads(fit.PROBE_KAPPA_PATH.read_text())
    out[KAPPA_PROBE] = fit.parameters_kappa(np.asarray(kappa["answer"]))
    return out


def _summary(graded: dict[str, Any]) -> dict[str, Any]:
    return bg._finite(
        {
            "new": new_readings(graded),
            "family_passes": graded["family_passes"],
            "wild_type_passes": graded["wild_type_passes"],
            "D": graded["D"],
            "D_fit": graded["D_fit"],
            "reversal_share": graded["reversal_share"],
        },
    )


def _adversary_population(item: tuple[str, dict[str, float], int]) -> dict[str, Any]:
    name, params, k = item
    graded = _mechanistic_graded(params, population_seeds(ADVERSARY_BASE, k), bg.load_bands_v3())
    summary = _summary(graded)
    summary["caught"] = any(not r["passes"] for r in summary["new"].values())
    return {"name": name, "k": k, **summary}


def answer_parameters(name: str) -> dict[str, float]:
    """Return v2's blind-worm answer or C.1's v1 answer, as C.1's parameters."""
    if name == "blind_v2":
        z = json.loads(v2c.BLIND_PATH.read_text())["answer"]
    else:
        fit = json.loads(v2c.C1_FIT.read_text())
        z = fit["restarts"][fit["chosen_restart"]]["answer"]
    return nr.parameters_from_unit(np.asarray(z))


def negative_graded(name: str, seeds: list[int], bands: dict[str, Any]) -> dict[str, Any]:
    """Run one negative, or one of the two answers, on the given animals; grade it on v3.

    The untrained networks are B.4's, each driving an equal block of the animals.
    """
    from quantumnematode.evaluation.controllers import (
        BatchedConnectome,
        BatchedMechanistic,
        BatchedMLP,
    )

    if name in ANSWERS:
        params = answer_parameters(name)
        populations, chemotaxis = bg.simulate_v2(
            "mlp",
            lambda _path: BatchedMechanistic(params),
            seeds,
            bands,
        )
    elif not name.startswith("untrained_"):
        hazards = json.loads(bc.PARAMS_PATH.read_text())
        populations, chemotaxis = bg.simulate_v2(
            "mlp",
            lambda _path: bc.hazard_controller(name, hazards),
            seeds,
            bands,
        )
    else:
        arm = "mlp" if name == "untrained_mlp" else "connectome"
        networks = bc.READ_NETWORKS[name]
        per = len(seeds) // len(networks)
        populations, chemotaxis = {}, []
        for j, network in enumerate(networks):
            block = seeds[j * per : (j + 1) * per]

            def build(path: Path, network: int = network) -> Any:  # noqa: ANN401 - a controller
                brain, _, _ = bc.harness.build_brain(path, network)
                return BatchedMLP(brain) if arm == "mlp" else BatchedConnectome(brain)  # type: ignore[arg-type]

            pops, chem = bg.simulate_v2(arm, build, block, bands)
            for family, part in pops.items():
                populations.setdefault(family, []).extend(part)
            chemotaxis.extend(chem)
    return bg.grade_populations_v3(populations, chemotaxis, bands)


def _negative_population(item: tuple[str, int, int]) -> dict[str, Any]:
    name, k, base = item
    graded = negative_graded(name, population_seeds(base, k), bg.load_bands_v3())
    return {"name": name, "k": k, **_summary(graded)}


def _negative_population_on(item: tuple[str, int, int, str]) -> dict[str, Any]:
    name, k, base, bands_path = item
    graded = negative_graded(name, population_seeds(base, k), bg.load_bands_v3(Path(bands_path)))
    return {"name": name, "k": k, **_summary(graded)}


def _by_name(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["name"], []).append(row)
    return out


def adversary_verdicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return each adversary's populations and whether it is caught in at least 3 of 5."""
    return {
        name: {
            "populations": pops,
            "caught_in": sum(p["caught"] for p in pops),
            "caught_3_of_5": sum(p["caught"] for p in pops) >= NEEDED,
        }
        for name, pops in _by_name(rows).items()
    }


def negative_verdicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return each negative's populations, its wild-type and chemotaxis failures, 3 of 5 each."""
    out = {}
    for name, pops in _by_name(rows).items():
        wild = sum(not p["wild_type_passes"] for p in pops)
        chem = sum(not p["family_passes"]["chemotaxis"] for p in pops)
        out[name] = {
            "populations": pops,
            "fails_wild_type_in": wild,
            "fails_wild_type_3_of_5": wild >= NEEDED,
            "fails_chemotaxis_in": chem,
            "fails_chemotaxis_3_of_5": chem >= NEEDED,
        }
    return out


def w_minimum_v3(bands: dict[str, Any]) -> dict[str, Any]:
    """Return W's minimum on v3's fitted and held-out families, each the 95th percentile.

    v2's rule: the fitted families between halves of N2's unread held-out dates; local search
    between halves of the band batches, crossed with every 3-against-3 split of Chen's recordings.
    """
    from quantumnematode.benchmark import chemotaxis as ch

    family_map = bg._v2_family_map(bands)
    tracks = refs.load_scheer_tracks()
    clf = bg.classifier(bands)
    rows = tracks.arm(v2c.N2_ARM)
    unread = v2c.n2_unread_dates(tracks)
    held = rows[np.isin(tracks.date[rows], unread) & (tracks.side[rows] == refs.HELD_OUT)]
    held_lawn, held_kin = v2c._tallies_by_date(tracks, held, clf)
    dates = sorted(held_lawn)
    builder = bg._builder()
    rng = np.random.default_rng(v2c.W_FITTED_SEED)
    fitted = []
    for _ in range(v2c.W_SPLITS):
        order = rng.permutation(len(dates))
        sides = [
            [dates[k] for k in order[: len(dates) // 2]],
            [dates[k] for k in order[len(dates) // 2 :]],
        ]
        lawn = [np.vstack([held_lawn[d] for d in side]) for side in sides]
        kin = [np.vstack([held_kin[d] for d in side]) for side in sides]
        step = v2c.W_SIZE_STEP
        size = max(step, round(min(len(lawn[0]), len(lawn[1])) / step) * step)
        small, kinematic = builder.small_lawn(size), bg._kinematic_bands_at(size)
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
        }
        fitted.append(g.distance(readings, ["small_lawn", "kinematics"], family_map))
    band_batches = list(bb.search_band_batches_v3(refs.load_lopezcruz()).values())
    search_names = g.LOCAL_SEARCH_STATISTICS_V3
    search_sd = {
        s: g.scale_of(np.array([b[s] for b in band_batches], dtype=float)) for s in search_names
    }
    record = json.loads(BANDS_V3_RECORD.read_text())
    recordings = [
        r["statistics"]
        for r in record["chemotaxis_recordings"].values()
        if r["protocol"] == "standard"
    ]
    chem_names = ch.CHEMOTAXIS_STATISTICS_V3
    chem_sd = {s: g.scale_of(np.array([r[s] for r in recordings], dtype=float)) for s in chem_names}
    k_chem = len(recordings) // 2
    chem_splits = []
    for picked in itertools.combinations(range(len(recordings)), k_chem):
        if 0 not in picked:
            continue  # each 3-against-3 split once
        rest = [j for j in range(len(recordings)) if j not in picked]
        chem_splits.append(
            v2c._difference(
                v2c._mean_of(recordings, picked, chem_names),
                v2c._mean_of(recordings, rest, chem_names),
                {s: v / math.sqrt(k_chem) for s, v in chem_sd.items()},
            ),
        )
    rng = np.random.default_rng(v2c.W_SEARCH_SEED)
    k_search = len(band_batches) // 2
    held_out, search_only = [], []
    for _ in range(v2c.W_SPLITS):
        order = rng.permutation(len(band_batches))
        search = v2c._difference(
            v2c._mean_of(band_batches, order[:k_search], search_names),
            v2c._mean_of(band_batches, order[k_search:], search_names),
            {s: v / math.sqrt(k_search) for s, v in search_sd.items()},
        )
        search_only.append(g.distance({"local_search": search}, ["local_search"], family_map))
        held_out.extend(
            g.distance(
                {"local_search": search, "chemotaxis": chem},
                ["local_search", "chemotaxis"],
                family_map,
            )
            for chem in chem_splits
        )
    chem_only = [g.distance({"chemotaxis": c}, ["chemotaxis"], family_map) for c in chem_splits]
    v2_fitted = json.loads(V2_READ_PATH.read_text())["w_minimum"]["fitted"]["minimum_95th"]
    return {
        "fitted": {
            "families": ["small_lawn", "kinematics"],
            "source": f"halves of N2's {len(dates)} unread held-out dates, on PD1074's band scales",
            "splits": v2c.W_SPLITS,
            "seed": v2c.W_FITTED_SEED,
            "median": float(np.median(fitted)),
            "minimum_95th": float(np.percentile(fitted, v2c.W_QUANTILE)),
            "v2_minimum_95th": v2_fitted,
        },
        "held_out": {
            "families": ["local_search", "chemotaxis"],
            "source": (
                f"halves of López-Cruz's {len(band_batches)} band batches ({v2c.W_SPLITS} splits) "
                f"crossed with every 3-against-3 split of Chen's {len(recordings)} recordings "
                f"({len(chem_splits)})"
            ),
            "seed": v2c.W_SEARCH_SEED,
            "pairs": len(held_out),
            "median": float(np.median(held_out)),
            "minimum_95th": float(np.percentile(held_out, v2c.W_QUANTILE)),
            "reported_local_search_95th": float(np.percentile(search_only, v2c.W_QUANTILE)),
            "reported_chemotaxis_95th": float(np.percentile(chem_only, v2c.W_QUANTILE)),
        },
    }


def three_of_five(record: dict[str, Any]) -> dict[str, Any]:
    """Return a faithful controller's chance of passing the wild type at 3 of 5, by v2's method.

    Each family fails a faithful population at its rate: the small lawn's, kinematics' and local
    search's from their rule checks at v3's bar, the uniform well's B.4's, and chemotaxis's the
    nominal rate for its five statistics.
    """
    checks = record["checks"]
    rates = {
        "small_lawn": 1.0 - checks["small_lawn"]["joint_pass_rate"],
        "uniform_lawn": UNIFORM_FAILURE_RATE,
        "local_search": 1.0 - checks["local_search"]["joint_pass_rate"],
        "kinematics": 1.0 - checks["kinematics"]["joint_pass_rate"],
        "chemotaxis": record["chemotaxis_leave_one_out"]["unit_rate"],
    }
    p = math.prod(1.0 - r for r in rates.values())
    return {
        "failure_rates": rates,
        "population_passes": p,
        "three_of_five": float(scipy_stats.binom.sf(NEEDED - 1, GRADE_POPULATIONS, p)),
    }


def _read(args: argparse.Namespace) -> int:
    if not _part_b():
        print(
            f"{LAUNCH_PATH} with Part B, and the files it binds, are not committed and clean; "
            "nothing is read.",
            file=sys.stderr,
        )
        return 1
    refusal = (
        f"{READ_PATH} or {READ_STARTED} exists: the read is made once, and a started read that "
        "stopped is not repeated."
    )
    if READ_PATH.exists():
        print(refusal, file=sys.stderr)
        return 1
    READ_STARTED.parent.mkdir(parents=True, exist_ok=True)
    try:
        with READ_STARTED.open("x") as marker:
            marker.write("the read started\n")
    except FileExistsError:
        print(refusal, file=sys.stderr)
        return 1
    bands = bg.load_bands_v3()
    answer = json.loads(ORACLE_FIT_PATH.read_text())["answer"]
    record: dict[str, Any] = {
        "description": "B v3's read: the oracle's 40 populations, then held-out López-Cruz, the "
        "adversaries, the negatives, W's minimum and the 3-of-5 probability.",
        "bands_sha256": mm.file_sha256(bg.BANDS_V3_PATH),
    }
    with ProcessPoolExecutor(max_workers=args.workers, initializer=nr._worker_setup) as pool:
        populations = list(
            pool.map(
                _oracle_check_population,
                range(ORACLE_POPULATIONS),
                [answer] * ORACLE_POPULATIONS,
            ),
        )
        record["oracle"] = {"populations": populations, **oracle_gate(populations)}
        print("oracle", json.dumps(record["oracle"]["statistics"], indent=2), flush=True)
        if not record["oracle"]["passes"]:
            # The registered stop: the new statistics are unreachable, and nothing else is read.
            _write_once(READ_PATH, record)
            return 1
        record["lopezcruz_held_out"] = lopezcruz_held_out(bands)
        adversaries = list(adversary_parameters().items())
        jobs = [
            (name, params, GRADE_POPULATIONS * i + p)
            for i, (name, params) in enumerate(adversaries)
            for p in range(GRADE_POPULATIONS)
        ]
        record["adversaries"] = adversary_verdicts(list(pool.map(_adversary_population, jobs)))
        print(
            "adversaries",
            {n: a["caught_in"] for n, a in record["adversaries"].items()},
            flush=True,
        )
        negative_jobs = [
            (name, GRADE_POPULATIONS * i + p, NEGATIVE_BASE)
            for i, name in enumerate(NEGATIVE_ORDER)
            for p in range(GRADE_POPULATIONS)
        ]
        record["negatives"] = negative_verdicts(
            list(pool.map(_negative_population, negative_jobs)),
        )
        print(
            "negatives",
            {
                n: (v["fails_wild_type_in"], v["fails_chemotaxis_in"])
                for n, v in record["negatives"].items()
            },
            flush=True,
        )
    record["w_minimum"] = w_minimum_v3(bands)
    record["three_of_five"] = three_of_five(json.loads(BANDS_V3_RECORD.read_text()))
    _write_once(READ_PATH, record)
    return 0


# ---------------------------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------------------------


def verdict(
    round_trip: dict[str, Any],
    record: dict[str, Any],
    read: dict[str, Any],
    v2_read: dict[str, Any],
) -> dict[str, Any]:
    """Return B v3's validity from its registered readings: VALID only if every check holds."""
    oracle_passes = bool(read["oracle"]["passes"])
    adversaries = read.get("adversaries", {})
    negatives = read.get("negatives", {})
    checks = {
        "rule_checks": all(c["passes"] for c in record["checks"].values()),
        "round_trip": bool(round_trip["passes"]),
        "reachability": oracle_passes,
        "lopezcruz_held_out": bool(read.get("lopezcruz_held_out", {}).get("passes", False)),
        "chemotaxis_leave_one_out": bool(record["chemotaxis_leave_one_out"]["passes"]),
        "adversaries_caught": bool(adversaries)
        and all(a["caught_3_of_5"] for a in adversaries.values()),
        "negatives_fail_wild_type": bool(negatives)
        and all(negatives[n]["fails_wild_type_3_of_5"] for n in NEGATIVE_ORDER),
        "open_loop_fail_chemotaxis": bool(negatives)
        and all(negatives[n]["fails_chemotaxis_3_of_5"] for n in v2c.OPEN_LOOP),
        "lawn_n2_control_carried_from_v2": bool(v2_read["n2_control"]["passes"]),
    }
    return {"checks": checks, "verdict": "VALID" if all(checks.values()) else "VOID"}


def _verdict(_args: argparse.Namespace) -> int:
    if not _part_b():
        print(
            f"{LAUNCH_PATH} with Part B, and the files it binds, are not committed and clean; no "
            "verdict is recorded.",
            file=sys.stderr,
        )
        return 1
    if not READ_PATH.exists():
        print(f"{READ_PATH} does not exist: the read comes first", file=sys.stderr)
        return 1
    parts = [
        json.loads(p.read_text())
        for p in (ROUND_TRIP_PATH, BANDS_V3_RECORD, READ_PATH, V2_READ_PATH)
    ]
    read = parts[2]
    out = {
        "description": "B v3's verdict: the bands record, the round trip, the oracle and the read.",
        "bands_sha256": mm.file_sha256(bg.BANDS_V3_PATH),
        **verdict(*parts),
        "w_minimum": read.get("w_minimum"),
        "three_of_five": read.get("three_of_five"),
    }
    bg.CONTROLS_V3_PATH.write_text(json.dumps(bg._finite(out), indent=2, allow_nan=False) + "\n")
    print(json.dumps(out["checks"], indent=2), "\nverdict:", out["verdict"])
    return 0


# ---------------------------------------------------------------------------------------------
# The preflight, on discarded seeds
# ---------------------------------------------------------------------------------------------


def _oracle_preflight_simulate(item: tuple[int, list[float], str]) -> float:
    k, z, bands_path = item
    bands = bg.load_bands_v3(Path(bands_path))
    return oracle_objective(_oracle_graded(z, population_seeds(PREFLIGHT_ORACLE_BASE, k), bands))


def _preflight(args: argparse.Namespace) -> int:
    bands = bg.load_bands_v3(args.bands)
    record: dict[str, Any] = {
        "description": "Description: B v3's controls preflight on discarded seeds, before Part A.",
        "bands_sha256": mm.file_sha256(args.bands),
        "round_trip": chen_round_trip_v3(bands, args.chen_archive, PREFLIGHT_ROUND_TRIP_BASE),
    }
    print("round trip passes:", record["round_trip"]["passes"], flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=nr._worker_setup) as pool:
        start = time.time()
        from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

        centre = [0.5] * len(MECHANISTIC_PARAMETERS)
        items = [(k, centre, str(args.bands)) for k in range(v2c.BLIND_POPSIZE)]
        objectives = list(pool.map(_oracle_preflight_simulate, items))
        record["oracle_generation"] = {
            "point": "the box centre",
            "objectives": objectives,
            "seconds": time.time() - start,
            "workers": args.workers,
        }
        print(f"one oracle generation: {record['oracle_generation']['seconds']:.0f} s", flush=True)
        jobs = [
            (name, PREFLIGHT_NEGATIVE_POPULATIONS * i + p, PREFLIGHT_NEGATIVE_BASE, str(args.bands))
            for i, name in enumerate(NEGATIVE_ORDER)
            for p in range(PREFLIGHT_NEGATIVE_POPULATIONS)
        ]
        start = time.time()
        rows = list(pool.map(_negative_population_on, jobs))
        record["negatives"] = negative_verdicts(rows)
        record["negatives_seconds"] = time.time() - start
    args.out.write_text(json.dumps(bg._finite(record), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                n: (v["fails_wild_type_in"], v["fails_chemotaxis_in"])
                for n, v in record["negatives"].items()
            },
        ),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI: the round trip, the oracle's fit, the read, the verdict and the preflight."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("round-trip").add_argument("--chen-archive", type=Path, required=True)
    for name in ("oracle-fit", "read"):
        sub.add_parser(name).add_argument("--workers", type=int, default=16)
    sub.add_parser("verdict")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--bands", type=Path, required=True)
    preflight.add_argument("--chen-archive", type=Path, required=True)
    preflight.add_argument("--workers", type=int, default=16)
    preflight.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    stages = {
        "round-trip": _round_trip,
        "oracle-fit": _oracle_fit,
        "read": _read,
        "verdict": _verdict,
        "preflight": _preflight,
    }
    return stages[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
