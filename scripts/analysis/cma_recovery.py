#!/usr/bin/env python
r"""Parameter recovery for C.1's mechanistic baseline by sep-CMA.

The positive control D38 asks of each fitting method: generate the benchmark's statistics at known
parameters, and check the fit gets them back. C.1's model has 20 parameters and the benchmark 11
statistics, so recovery is judged in statistic space: a fit recovers a truth when simulations at its
answer reproduce the truth's statistics about as closely as the truth's own re-simulations do.

**The model, the box and the simulator** are F.0a's (``npe_recovery``): ``BatchedMechanistic`` on
the unit cube mapped through each parameter's range, simulated at 200 fresh animals on the three
wild-type assays and graded.

**The objective** for truth j at z is ``predictive_distance(observation_j, simulate(z))``: D over
the 11 statistics on the bands' n = 200 scales, capped at 10.

**sep-CMA** (``cma``, ``CMA_diagonal``) starts at the box centre with sigma0 0.2, population 16,
bounds [0, 1]. A fit's answer is the distribution's mean mapped into the box (``xfavorite``).
Generations are counted from 1. With m_n generation n's median objective and S_n the mean of m over
generations n-19..n, a fit stops at the first n >= 40 with S_n > S_{n-20} - 0.05, or at its cap.

**The gain fraction** is (prior D - D) / (prior D - floor): 1 at the truth's own noise floor, 0 at
the prior's typical distance.

Subcommands (each refuses until the launch record is committed and unchanged; ``run`` and ``read``
also need its Part B)::

    uv run python scripts/analysis/cma_recovery.py preflight
    uv run python scripts/analysis/cma_recovery.py pilot
    uv run python scripts/analysis/cma_recovery.py run
    uv run python scripts/analysis/cma_recovery.py read
    uv run python scripts/analysis/cma_recovery.py smoke    # execution check, discarded
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import npe_recovery as nr
import npe_region_check as rc

DESCRIPTION = "Parameter recovery for C.1's mechanistic baseline by sep-CMA."
_REPO = Path(__file__).resolve().parents[2]
SUPPORT = _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "112-cma-recovery"
LAUNCH_PATH = SUPPORT / "launch.md"
PREFLIGHT_PATH = SUPPORT / "preflight.json"
PILOT_PATH = SUPPORT / "pilot.json"
RECOVERY_PATH = SUPPORT / "recovery.json"
BUILD = _REPO / "build" / "cma-recovery"

SEED_BASE = 50_000_000
PILOT_SEED_BASE = 80_000_000
SMOKE_SEED_BASE = 99_000_000
CMA_SEED = 5410
PILOT_CMA_SEED = 5418
SMOKE_CMA_SEED = 5991

DIMENSION = 20
POPSIZE = 16
X0 = 0.5
SIGMA0 = 0.2
MAX_GENERATIONS = 300
PLATEAU_WINDOW = 20
PLATEAU_TOLERANCE = 0.05
PLATEAU_FIRST = 40
CAP_FACTOR = 1.5
CAP_FALLBACK = 0.05

TRUTHS = 4
RESTARTS = 2
REPEATS = 20
STABILITY_REPEATS = 10
RESIMULATIONS = 10
REACHABLE = 1.0
PRIOR_REFERENCE = 200
CANDIDATE_BATCH = 8
PROBE_TRUTH = 8207

ORACLE_GATE = 0.85
START_GATE = 0.25
GAIN_GATE = 0.75
PILOT_STOP = 0.75
PROJECTION_HOURS = 20.0

# Simulation indices k; animals are ``base + 200 k + i``.
FIT_BLOCK = 10_000
ANSWER_OFFSET = 5_000
CANDIDATE_BASE_K = 100_000
CENTRE_BASE_K = 101_000
PILOT_ANSWER_OFFSETS = (5_000, 5_100, 5_200)


# ---------------------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------------------


def plateau_reached(medians: list[float]) -> bool:
    """Return whether the plateau rule holds after the generations whose medians are given."""
    n = len(medians)
    if n < PLATEAU_FIRST:
        return False
    recent = float(np.mean(medians[n - PLATEAU_WINDOW : n]))
    before = float(np.mean(medians[n - 2 * PLATEAU_WINDOW : n - PLATEAU_WINDOW]))
    return recent > before - PLATEAU_TOLERANCE


def plateau_generation(medians: list[float]) -> int | None:
    """Return the first generation (counted from 1) at which the plateau rule holds, if any."""
    for n in range(PLATEAU_FIRST, len(medians) + 1):
        if plateau_reached(medians[:n]):
            return n
    return None


def candidate_cap(plateau: int | None) -> int:
    """Return the cap the pilot's plateau generation implies, before the fallback check."""
    if plateau is None:
        return MAX_GENERATIONS
    return min(MAX_GENERATIONS, math.ceil(CAP_FACTOR * plateau))


def registered_cap(plateau: int | None, gain_at_cap: float | None, gain_at_end: float) -> int:
    """Return the registered cap: 300 if the rule never fired or fired before the descent ended."""
    cap = candidate_cap(plateau)
    if cap == MAX_GENERATIONS or gain_at_cap is None:
        return MAX_GENERATIONS
    return MAX_GENERATIONS if gain_at_cap < gain_at_end - CAP_FALLBACK else cap


def final_objective(medians: list[float]) -> float:
    """Return S at a fit's stop: the mean of its last 20 generations' median objectives."""
    return float(np.mean(medians[-PLATEAU_WINDOW:]))


def best_restart(medians_by_restart: list[list[float]]) -> int:
    """Return the restart whose final objective is lowest; the first on a tie."""
    return int(np.argmin([final_objective(m) for m in medians_by_restart]))


def definedness(record: dict[str, Any]) -> tuple[bool, ...]:
    """Return, per statistic in the benchmark's order, whether it is defined."""
    return tuple(record["statistics"][n] is not None for n in nr.statistic_names()[:-1])


def keep_candidate(
    observation: dict[str, Any],
    stability_repeats: list[dict[str, Any]],
    floor: float,
    prior: float,
) -> tuple[bool, str]:
    """Return whether a candidate truth is kept, and why not if it is skipped."""
    reference = definedness(observation)
    flips = sorted(
        {
            name
            for repeat in stability_repeats
            for name, a, b in zip(
                nr.statistic_names()[:-1],
                reference,
                definedness(repeat),
                strict=True,
            )
            if a != b
        },
    )
    if flips:
        return False, "flips: " + ", ".join(flips)
    if prior - floor < REACHABLE:
        return False, f"reachable gain {prior - floor:.2f} < {REACHABLE}"
    return True, "kept"


def answer(strategy: Any) -> np.ndarray:  # noqa: ANN401 - a cma strategy
    """Return a fit's answer: the distribution's mean mapped into the box."""
    return np.clip(np.asarray(strategy.result.xfavorite, dtype=float), 0.0, 1.0)


def new_strategy(seed: int) -> Any:  # noqa: ANN401 - a cma strategy
    """Return sep-CMA at the box centre, as every registered fit starts."""
    import cma

    options = {
        "CMA_diagonal": True,
        "bounds": [0.0, 1.0],
        "popsize": POPSIZE,
        "seed": seed,
        "verbose": -9,
    }
    return cma.CMAEvolutionStrategy([X0] * DIMENSION, SIGMA0, options)


# ---------------------------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------------------------


def fit_k(fit: int, generation: int, member: int) -> int:
    """Return the simulation index of a fit's generation (counted from 1) and member."""
    return FIT_BLOCK * fit + POPSIZE * (generation - 1) + member


def answer_k(fit: int, repeat: int) -> int:
    """Return the simulation index of a registered fit's answer, re-simulated."""
    return FIT_BLOCK * fit + ANSWER_OFFSET + repeat


def candidate_k(candidate: int, repeat: int) -> int:
    """Return the simulation index of a candidate truth's repeat."""
    return CANDIDATE_BASE_K + REPEATS * candidate + repeat


def centre_k(repeat: int) -> int:
    """Return the simulation index of the box centre, re-simulated."""
    return CENTRE_BASE_K + repeat


# ---------------------------------------------------------------------------------------------
# Truths
# ---------------------------------------------------------------------------------------------


def candidate_pool(sims: dict[int, dict[str, Any]]) -> list[int]:
    """Return F.0a's prior simulations eligible as truths, in ascending D to the bands."""
    import benchmark_grade as bg
    from quantumnematode.benchmark import grading as g

    bands = bg.load_bands()["bands"]["200"]
    rows = []
    for k in range(nr.TRAINING):
        record = sims[k]
        presence = record["presence"]
        if k == PROBE_TRUTH or presence is None or presence <= 1.5 or not all(definedness(record)):
            continue
        d = rc.band_distance(record, bands, list(g.FAMILIES))
        if d <= 5.0:
            rows.append((d, k))
    return [k for _, k in sorted(rows)]


def prior_distance(sims: dict[int, dict[str, Any]], k: int, scales: dict[str, float]) -> float:
    """Return the median D from simulation k to F.0a's first 200 other training simulations."""
    others = [i for i in range(nr.TRAINING) if i != k][:PRIOR_REFERENCE]
    return float(np.median([nr.predictive_distance(sims[k], sims[i], scales) for i in others]))


def median_distance(
    observation: dict[str, Any],
    records: list[dict[str, Any]],
    scales: dict[str, float],
) -> float:
    """Return the median D from an observation to each of several simulations."""
    return float(np.median([nr.predictive_distance(observation, r, scales) for r in records]))


# ---------------------------------------------------------------------------------------------
# The launch guard
# ---------------------------------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    """Run git in the repository with fixed arguments."""
    return subprocess.run(  # noqa: S603 - fixed git arguments, no user input
        ["git", *args],  # noqa: S607 - git on the PATH
        cwd=_REPO,
        capture_output=True,
        check=False,
    )


def launch_committed(bound: tuple[Path, ...] = (), launch_path: Path | None = None) -> bool:
    """Return whether the launch record is committed and unchanged, and its bound records with it.

    Each bound record must be byte-identical to its version in the commit that last touched the
    launch record, and unchanged in the working tree. The launch record defaults to this recovery's.
    """
    record = LAUNCH_PATH if launch_path is None else launch_path
    if not record.exists():
        return False
    launch = str(record.relative_to(_REPO))
    commit = _git("log", "-1", "--format=%H", "--", launch).stdout.decode().strip()
    if not commit:
        return False
    paths = [str(p.relative_to(_REPO)) for p in bound]
    if _git("diff", "--quiet", "HEAD", "--", launch, *paths).returncode != 0:
        return False
    for path, rel in zip(bound, paths, strict=True):
        registered = _git("show", f"{commit}:{rel}")
        if (
            registered.returncode != 0
            or not path.exists()
            or registered.stdout != path.read_bytes()
        ):
            return False
    return True


def part_b_committed() -> bool:
    """Return whether Part B is committed, with the preflight and pilot records it binds."""
    return launch_committed((PREFLIGHT_PATH, PILOT_PATH)) and "## Part B" in LAUNCH_PATH.read_text()


# ---------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(nr._finite(record), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def simulate_jobs(
    jobs: list[tuple[int, list[float]]],
    seed_base: int,
    path: Path,
    workers: int,
) -> dict[int, dict[str, Any]]:
    """Run every job not yet stored, and return the stored simulations, by k."""
    nr.run_pool(jobs, workers, path, seed_base)
    return {k: nr._finite(r) for k, r in nr.load_simulations(path).items()}


def run_fit(  # noqa: PLR0913 - keyword-only: one fit's registered settings
    observation: dict[str, Any],
    *,
    fit: int,
    cma_seed: int,
    seed_base: int,
    cap: int,
    use_rule: bool,
    state_dir: Path,
    pool: ProcessPoolExecutor,
    scales: dict[str, float],
    objective: Callable[[dict[str, Any]], float] | None = None,
) -> dict[str, Any]:
    """Run, or resume, one fit; return its history. State is saved after every generation.

    The objective scores one graded simulation; by default it is the distance to ``observation``.

    The strategy's checkpoint and the history each carry the last completed generation; a resume
    whose two markers differ is refused rather than continued from mismatched state.
    """
    state_path = state_dir / f"fit{fit}.pkl"
    history_path = state_dir / f"fit{fit}.json"
    saved = json.loads(history_path.read_text()) if history_path.exists() else None
    if saved is not None and saved["stop"] is not None:
        return saved
    if saved is not None and state_path.exists():
        history = saved
        checkpoint = pickle.loads(state_path.read_bytes())  # noqa: S301 - this run's own state
        if checkpoint["generation"] != history.get("generation"):
            message = (
                f"fit {fit}: checkpoint at generation {checkpoint['generation']}, "
                f"history at {history.get('generation')}"
            )
            raise RuntimeError(message)
        strategy = checkpoint["strategy"]
    else:
        strategy = new_strategy(cma_seed)
        history = {
            "fit": fit,
            "cma_seed": cma_seed,
            "generation": 0,
            "medians": [],
            "sigmas": [],
            "answers": [],
            "seconds": [],
            "stop": None,
        }
    while history["stop"] is None:
        generation = len(history["medians"]) + 1
        start = time.perf_counter()
        population = strategy.ask()
        ks = [fit_k(fit, generation, member) for member in range(POPSIZE)]
        points = [list(map(float, x)) for x in population]
        records = list(pool.map(nr.simulate, ks, points, [seed_base] * POPSIZE))
        scored = [
            nr.predictive_distance(observation, nr._finite(r), scales)
            if objective is None
            else objective(nr._finite(r))
            for r in records
        ]
        strategy.tell(population, scored)
        history["medians"].append(float(np.median(scored)))
        history["sigmas"].append(float(strategy.sigma))
        history["answers"].append(answer(strategy).tolist())
        history["seconds"].append(time.perf_counter() - start)
        if use_rule and plateau_reached(history["medians"]):
            history["stop"] = {"generation": generation, "by": "plateau"}
        elif generation >= cap:
            history["stop"] = {"generation": generation, "by": "cap"}
        history["generation"] = generation
        _write_bytes(state_path, pickle.dumps({"generation": generation, "strategy": strategy}))
        _write_json(history_path, history)
        print(
            f"fit {fit} gen {generation}: median {history['medians'][-1]:.2f}, "
            f"sigma {history['sigmas'][-1]:.3f}",
            flush=True,
        )
    return history


def _gain(prior: float, d: float, floor: float) -> float:
    return rc.gain_fraction(prior, d, floor)


def _preflight(workers: int) -> int:
    """Choose the truths, then score the oracle and the unfitted start against them."""
    sims = nr.load_simulations()
    scales = nr.band_scales()
    pool = candidate_pool(sims)
    path = BUILD / "simulations.jsonl"
    candidates: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for batch_start in range(0, len(pool), CANDIDATE_BATCH):
        batch = list(enumerate(pool))[batch_start : batch_start + CANDIDATE_BATCH]
        jobs = [(candidate_k(c, q), list(sims[k]["z"])) for c, k in batch for q in range(REPEATS)]
        done = simulate_jobs(jobs, SEED_BASE, path, workers)
        for c, k in batch:
            repeats = [done[candidate_k(c, q)] for q in range(REPEATS)]
            floor = median_distance(sims[k], repeats[:STABILITY_REPEATS], scales)
            oracle = median_distance(sims[k], repeats[STABILITY_REPEATS:], scales)
            prior = prior_distance(sims, k, scales)
            keep, reason = keep_candidate(sims[k], repeats[:STABILITY_REPEATS], floor, prior)
            row = {
                "candidate": c,
                "training_simulation": k,
                "floor": floor,
                "oracle_D": oracle,
                "prior_D": prior,
                "kept": keep,
                "reason": reason,
            }
            candidates.append(row)
            if keep and len(kept) < 1 + TRUTHS:
                kept.append(row)
        if len(kept) == 1 + TRUTHS:
            break
    record: dict[str, Any] = {"description": DESCRIPTION, "candidates": candidates}
    if len(kept) < 1 + TRUTHS:
        record["passed"] = False
        record["stop"] = f"only {len(kept)} candidates kept; the run returns to the maintainer"
        _write_json(PREFLIGHT_PATH, record)
        print(record["stop"], file=sys.stderr)
        return 1
    centre = simulate_jobs(
        [(centre_k(q), [X0] * DIMENSION) for q in range(RESIMULATIONS)],
        SEED_BASE,
        path,
        workers,
    )
    centre_runs = [centre[centre_k(q)] for q in range(RESIMULATIONS)]
    truths = kept[1:]
    for row in truths:
        k = row["training_simulation"]
        row["oracle_gain"] = _gain(row["prior_D"], row["oracle_D"], row["floor"])
        row["centre_D"] = median_distance(sims[k], centre_runs, scales)
        row["centre_gain"] = _gain(row["prior_D"], row["centre_D"], row["floor"])
    oracle = float(np.median([r["oracle_gain"] for r in truths]))
    start = float(np.median([r["centre_gain"] for r in truths]))
    record |= {
        "pilot_truth": kept[0],
        "truths": truths,
        "oracle_median_gain": oracle,
        "centre_median_gain": start,
        "oracle_passes": oracle >= ORACLE_GATE,
        "centre_fails": start <= START_GATE,
    }
    record["passed"] = record["oracle_passes"] and record["centre_fails"]
    _write_json(PREFLIGHT_PATH, record)
    print(json.dumps({k: v for k, v in record.items() if k != "candidates"}, indent=2))
    return 0 if record["passed"] else 1


def _resimulate(
    z: list[float],
    ks: list[int],
    seed_base: int,
    path: Path,
    workers: int,
) -> list[dict[str, Any]]:
    done = simulate_jobs([(k, z) for k in ks], seed_base, path, workers)
    return [done[k] for k in ks]


def _pilot(workers: int) -> int:
    """Run the pilot fit to 300 generations; set the cap and project the cost."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if not preflight.get("passed"):
        print("the preflight did not pass; no pilot runs", file=sys.stderr)
        return 1
    sims = nr.load_simulations()
    scales = nr.band_scales()
    truth = preflight["pilot_truth"]
    observation = sims[truth["training_simulation"]]
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = run_fit(
            observation,
            fit=0,
            cma_seed=PILOT_CMA_SEED,
            seed_base=PILOT_SEED_BASE,
            cap=MAX_GENERATIONS,
            use_rule=False,
            state_dir=BUILD / "pilot",
            pool=pool,
            scales=scales,
        )
    plateau = plateau_generation(history["medians"])
    at = {"end": MAX_GENERATIONS}
    if plateau is not None:
        at |= {"plateau": plateau, "cap": candidate_cap(plateau)}
    gains: dict[str, float] = {}
    path = BUILD / "pilot" / "answers.jsonl"
    for (name, generation), offset in zip(
        sorted(at.items(), key=lambda kv: kv[1]),
        PILOT_ANSWER_OFFSETS,
        strict=False,
    ):
        runs = _resimulate(
            history["answers"][generation - 1],
            [offset + q for q in range(RESIMULATIONS)],
            PILOT_SEED_BASE,
            path,
            workers,
        )
        d = median_distance(observation, runs, scales)
        gains[name] = _gain(truth["prior_D"], d, truth["floor"])
    cap = registered_cap(plateau, gains.get("cap"), gains["end"])
    seconds = float(np.median(history["seconds"][100:])) if len(history["seconds"]) > 100 else None
    per_generation = seconds if seconds is not None else float(np.median(history["seconds"]))
    hours = TRUTHS * RESTARTS * (cap + 1) * per_generation / 3600.0
    record = {
        "description": DESCRIPTION,
        "pilot_truth": truth["training_simulation"],
        "plateau_generation": plateau,
        "generations": at,
        "gains": gains,
        "registered_cap": cap,
        "sustained_seconds_per_generation": per_generation,
        "projected_hours": hours,
        "stops": {
            "gain_at_300_below_bar": gains["end"] < PILOT_STOP,
            "projection_over_20_h": hours > PROJECTION_HOURS,
        },
        "medians": history["medians"],
        "sigmas": history["sigmas"],
    }
    _write_json(PILOT_PATH, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("medians", "sigmas")}, indent=2))
    return 1 if record["stops"]["gain_at_300_below_bar"] else 0


def _run(workers: int) -> int:
    """Run the registered fits to the registered cap, then re-simulate each answer."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    pilot = json.loads(PILOT_PATH.read_text())
    if not preflight.get("passed"):
        print("the preflight did not pass; no registered fit runs", file=sys.stderr)
        return 1
    if pilot["stops"]["gain_at_300_below_bar"]:
        print("the pilot's gain at 300 is below the bar; no registered fit runs", file=sys.stderr)
        return 1
    cap = int(pilot["registered_cap"])
    sims = nr.load_simulations()
    scales = nr.band_scales()
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for fit in range(TRUTHS * RESTARTS):
            truth = preflight["truths"][fit // RESTARTS]
            run_fit(
                sims[truth["training_simulation"]],
                fit=fit,
                cma_seed=CMA_SEED + fit,
                seed_base=SEED_BASE,
                cap=cap,
                use_rule=True,
                state_dir=BUILD / "fits",
                pool=pool,
                scales=scales,
            )
    for fit in range(TRUTHS * RESTARTS):
        history = json.loads((BUILD / "fits" / f"fit{fit}.json").read_text())
        _resimulate(
            history["answers"][-1],
            [answer_k(fit, q) for q in range(RESIMULATIONS)],
            SEED_BASE,
            BUILD / "simulations.jsonl",
            workers,
        )
    return 0


def _read() -> int:
    """Read the gate and the reported measures into the recovery record."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    sims = nr.load_simulations()
    scales = nr.band_scales()
    stored = {k: nr._finite(r) for k, r in nr.load_simulations(BUILD / "simulations.jsonl").items()}
    rows = []
    for j, truth in enumerate(preflight["truths"]):
        k = truth["training_simulation"]
        observation = sims[k]
        fits = []
        for r in range(RESTARTS):
            fit = RESTARTS * j + r
            history = json.loads((BUILD / "fits" / f"fit{fit}.json").read_text())
            runs = [stored[answer_k(fit, q)] for q in range(RESIMULATIONS)]
            d = median_distance(observation, runs, scales)
            error = np.abs(np.asarray(history["answers"][-1]) - np.asarray(sims[k]["z"]))
            fits.append(
                {
                    "fit": fit,
                    "final_objective": final_objective(history["medians"]),
                    "stop": history["stop"],
                    "answer_D": d,
                    "gain": _gain(truth["prior_D"], d, truth["floor"]),
                    "parameter_error": error.tolist(),
                    "median_parameter_error": float(np.median(error)),
                    "medians": history["medians"],
                    "sigmas": history["sigmas"],
                    "runs": runs,
                },
            )
        best = best_restart([f["medians"] for f in fits])
        spread = float(
            np.median(
                [
                    nr.predictive_distance(a, b, scales)
                    for a, b in zip(fits[0]["runs"], fits[1]["runs"], strict=True)
                ],
            ),
        )
        for f in fits:
            del f["runs"]
        rows.append(
            {
                "truth": k,
                "floor": truth["floor"],
                "prior_D": truth["prior_D"],
                "best_restart": best,
                "gain": fits[best]["gain"],
                "other_gain": fits[1 - best]["gain"],
                "restart_spread_D": spread,
                "fits": fits,
            },
        )
    median_gain = float(np.median([r["gain"] for r in rows]))
    record = {
        "description": DESCRIPTION,
        "median_gain": median_gain,
        "gate": GAIN_GATE,
        "passes": median_gain >= GAIN_GATE,
        "truths": rows,
    }
    _write_json(RECOVERY_PATH, record)
    print(json.dumps({"median_gain": median_gain, "passes": record["passes"]}, indent=2))
    return 0


def _smoke(workers: int) -> int:
    """Run two generations of one fit and a resume, on discarded seeds."""
    sims = nr.load_simulations()
    scales = nr.band_scales()
    observation = sims[candidate_pool(sims)[0]]
    state = BUILD / "smoke"
    shutil.rmtree(state, ignore_errors=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for cap in (1, 2):
            history = run_fit(
                observation,
                fit=0,
                cma_seed=SMOKE_CMA_SEED,
                seed_base=SMOKE_SEED_BASE,
                cap=cap,
                use_rule=True,
                state_dir=state,
                pool=pool,
                scales=scales,
            )
            if cap == 1:
                history["stop"] = None
                _write_json(state / "fit0.json", history)
    print(json.dumps({k: history[k] for k in ("medians", "sigmas", "stop")}, indent=2))
    ok = len(history["medians"]) == 2 and all(0.0 <= v <= 1.0 for v in history["answers"][-1])
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """CLI: the preflight, the pilot, the registered fits, the reading and a smoke."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("stage", choices=("preflight", "pilot", "run", "read", "smoke"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.stage == "smoke":
        return _smoke(args.workers)
    if args.stage in ("preflight", "pilot") and not launch_committed():
        print("the launch record is not committed, or has changed", file=sys.stderr)
        return 1
    if args.stage in ("run", "read") and not part_b_committed():
        print("Part B, or a record it binds, is not committed or has changed", file=sys.stderr)
        return 1
    stages = {"preflight": _preflight, "pilot": _pilot, "run": _run}
    if args.stage in stages:
        return stages[args.stage](args.workers)
    return _read()


if __name__ == "__main__":
    sys.exit(main())
