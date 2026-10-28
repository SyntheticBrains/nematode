#!/usr/bin/env python
r"""Parameter recovery for C.1's mechanistic baseline by sep-CMA, re-read on benchmark v2.

F.0b-i's rules (``cma_recovery``) on B v2's objective: C.1's model, 200 fresh animals on v2's four
assays through v2's measurement models, scored by the distance over v2's 19 graded statistics on
its n = 200 band scales with its five families weighted equally. What v2 requires otherwise:

- **the truths** are re-selected on v2: the 43 F.0a simulations F.0b-i's rule admits are each
  simulated once on v2; those with every v2 statistic defined, local-search presence above 1.5 and
  chemotaxis present are taken in ascending v2 D to the bands, and kept on F.0b-i's rules;
- **the plateau stop is guarded**: a fit stops on the plateau rule only once sigma <= sigma0;
- **seeds** are new.

Subcommands (``preflight`` and ``pilot`` refuse until the launch record is committed and unchanged;
``run`` and ``read`` also need its Part B and the bound records)::

    uv run python scripts/analysis/cma_recovery_v2.py gate-probe   # discarded seeds; description
    uv run python scripts/analysis/cma_recovery_v2.py preflight
    uv run python scripts/analysis/cma_recovery_v2.py pilot
    uv run python scripts/analysis/cma_recovery_v2.py run
    uv run python scripts/analysis/cma_recovery_v2.py read
    uv run python scripts/analysis/cma_recovery_v2.py smoke        # execution check, discarded
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cma_recovery as cr
import npe_recovery as nr

DESCRIPTION = "Parameter recovery for C.1's model by sep-CMA, re-read on benchmark v2."
_REPO = Path(__file__).resolve().parents[2]
SUPPORT = _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "116-cma-recovery-v2"
LAUNCH_PATH = SUPPORT / "launch.md"
PREFLIGHT_PATH = SUPPORT / "preflight.json"
PILOT_PATH = SUPPORT / "pilot.json"
RECOVERY_PATH = SUPPORT / "recovery.json"
BUILD = _REPO / "build" / "cma-recovery-v2"
PROBE_PATH = BUILD / "gate_probe.json"

FIT_BASE = 170_000_000
PILOT_BASE = 200_000_000
CANDIDATE_BASE = 210_000_000
PROBE_BASE = 99_400_000
SMOKE_BASE = 99_300_000
CMA_SEED = 5420
PILOT_CMA_SEED = 5428
SMOKE_CMA_SEED = 5993

# The candidate scheme's simulation indices, k, at a candidate base.
OBSERVATION_K = 0
REPEAT_K = 1_000
REFERENCE_K = 2_000
CENTRE_K = 2_300
REFERENCES = 201
PRESENCE_FLOOR = 1.5
PROJECTION_HOURS = 30.0
PROBE_MARGIN = 0.05


# ---------------------------------------------------------------------------------------------
# The simulator and the objective
# ---------------------------------------------------------------------------------------------


def _bands() -> dict[str, Any]:
    import benchmark_grade as bg

    return bg.load_bands_v2()


def statistic_keys(bands: dict[str, Any]) -> list[str]:
    """Return v2's graded statistics as ``family/statistic``, in its families' order."""
    return [f"{f}/{s}" for f, stats in bands["families"].items() for s in stats]


def band_scales_v2(bands: dict[str, Any]) -> dict[str, float]:
    """Return each graded statistic's band scale at n = 200."""
    import benchmark_grade as bg

    out = {}
    for family, stats in bands["families"].items():
        family_bands = bg.family_bands_v2(bands, family, nr.ANIMALS)
        for s in stats:
            out[f"{family}/{s}"] = float(family_bands[s].scale)
    return out


def simulate_v2(
    k: int,
    z: list[float],
    seed_base: int,
    *,
    presence: bool = False,
) -> dict[str, Any]:
    """Run simulation k of C.1's model at z on v2; return its graded statistics and D.

    With ``presence``, as the candidates' observations need, it also records the presence tests.
    """
    import benchmark_grade as bg
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    bands = _bands()
    seeds = [seed_base + nr.ANIMALS * k + i for i in range(nr.ANIMALS)]
    params = nr.parameters_from_unit(np.asarray(z))
    start = time.perf_counter()
    populations, chemotaxis = bg.simulate_v2(
        "mlp",
        lambda _path: BatchedMechanistic(params),
        seeds,
        bands,
    )
    graded = bg.grade_populations_v2(populations, chemotaxis, bands, presence=presence)
    statistics = {
        f"{family}/{name}": reading["value"]
        for family, readings in graded["families"].items()
        for name, reading in readings.items()
    }
    record = {
        "k": k,
        "z": list(map(float, z)),
        "statistics": statistics,
        "D": graded["D"],
        "D_fit": graded["D_fit"],
        "seconds": time.perf_counter() - start,
    }
    if presence:
        record["presence"] = {
            "local_search": graded["local_search_presence"]["lower_bound"],
            "chemotaxis": bool(graded["chemotaxis_presence"]["present"]),
        }
    return nr._finite(record)


def simulate_v2_presence(k: int, z: list[float], seed_base: int) -> dict[str, Any]:
    """Run :func:`simulate_v2` with presence; a picklable entry for the pool."""
    return simulate_v2(k, z, seed_base, presence=True)


def predictive_distance_v2(
    truth: dict[str, Any],
    draw: dict[str, Any],
    scales: dict[str, float],
    family_map: dict[str, tuple[str, ...]],
) -> float:
    """Return the distance between two v2 records' graded statistics on the bands' scales.

    Families weigh equally, split equally within each; z is capped at 10. A statistic undefined in
    both counts 0, and one undefined in only one counts the cap. A reported, ungraded statistic
    takes no part, since only the family map's statistics are read.
    """
    from quantumnematode.benchmark import grading as g

    readings: dict[str, dict[str, Any]] = {f: {} for f in family_map}
    for family, stats in family_map.items():
        for s in stats:
            key = f"{family}/{s}"
            a, b = truth["statistics"].get(key), draw["statistics"].get(key)
            if a is None and b is None:
                z = 0.0
            elif a is None or b is None:
                z = g.Z_CAP
            else:
                z = g.z_score(float(b) - float(a), g.Band(0.0, scales[key]))
            readings[family][s] = g.Reading(0.0, g.Band(0.0, scales[key]), z, passes=True)
    return g.distance(readings, list(family_map), family_map)


class Objective:
    """The distance from one observation to a simulation, picklable and with its scales loaded."""

    def __init__(self, observation: dict[str, Any]) -> None:
        bands = _bands()
        self.observation = observation
        self.scales = band_scales_v2(bands)
        self.family_map = {f: tuple(st) for f, st in bands["families"].items()}

    def __call__(self, record: dict[str, Any]) -> float:
        """Return the distance from the observation to one simulation's record."""
        return predictive_distance_v2(self.observation, record, self.scales, self.family_map)


def median_distance_v2(observation: dict[str, Any], records: list[dict[str, Any]]) -> float:
    """Return the median distance from an observation to each of several simulations."""
    objective = Objective(observation)
    return float(np.median([objective(r) for r in records]))


def guarded_stop(medians: list[float], sigma: float) -> bool:
    """Return whether the guarded plateau rule stops a fit: it holds and sigma <= sigma0."""
    return cr.plateau_reached(medians) and sigma <= cr.SIGMA0


def guarded_plateau_generation(medians: list[float], sigmas: list[float]) -> int | None:
    """Return the first generation (from 1) at which the guarded plateau rule holds, if any."""
    for n in range(cr.PLATEAU_FIRST, len(medians) + 1):
        if guarded_stop(medians[:n], sigmas[n - 1]):
            return n
    return None


# ---------------------------------------------------------------------------------------------
# Running simulations
# ---------------------------------------------------------------------------------------------


def run_jobs(
    jobs: list[tuple[int, list[float]]],
    seed_base: int,
    path: Path,
    workers: int,
    *,
    presence: bool = False,
) -> dict[int, dict[str, Any]]:
    """Run the jobs not yet stored at ``path`` (one JSON line each); return every stored record."""
    done = (
        {json.loads(line)["k"] for line in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    todo = [(k, z) for k, z in jobs if k not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = simulate_v2_presence if presence else simulate_v2
    if todo:
        print(f"{len(done)} stored, {len(todo)} to run on {workers} workers", flush=True)
        with (
            ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool,
            path.open("a") as out,
        ):
            futures = [pool.submit(entry, k, z, seed_base) for k, z in todo]
            for future in as_completed(futures):
                out.write(json.dumps(future.result(), allow_nan=False) + "\n")
                out.flush()
    return {r["k"]: r for r in (json.loads(line) for line in path.read_text().splitlines())}


# ---------------------------------------------------------------------------------------------
# Truths
# ---------------------------------------------------------------------------------------------


def eligible(record: dict[str, Any]) -> bool:
    """Return whether a candidate's v2 observation is eligible to be a truth."""
    presence = record.get("presence") or {}
    return (
        all(v is not None for v in record["statistics"].values())
        and (presence.get("local_search") or 0.0) > PRESENCE_FLOOR
        and bool(presence.get("chemotaxis"))
    )


def definedness_v2(record: dict[str, Any], keys: list[str]) -> tuple[bool, ...]:
    """Return, per graded statistic, whether it is defined."""
    return tuple(record["statistics"].get(k) is not None for k in keys)


def keep_candidate_v2(
    observation: dict[str, Any],
    repeats: list[dict[str, Any]],
    floor: float,
    prior: float,
    keys: list[str],
) -> tuple[bool, str]:
    """Return whether a candidate is kept on F.0b-i's rules, and why not if it is skipped."""
    reference = definedness_v2(observation, keys)
    flips = sorted(
        {
            key
            for r in repeats
            for key, a, b in zip(keys, reference, definedness_v2(r, keys), strict=True)
            if a != b
        },
    )
    if flips:
        return False, "flips: " + ", ".join(flips)
    if prior - floor < cr.REACHABLE:
        return False, f"prior - floor {prior - floor:.2f} below {cr.REACHABLE}"
    return True, ""


def select_truths(base: int, path: Path, workers: int) -> dict[str, Any]:
    """Return the preflight's record at a candidate base: candidates, kept truths and the gates."""
    bands = _bands()
    keys = statistic_keys(bands)
    sims = nr.load_simulations()
    pool = cr.candidate_pool(sims)
    observations = run_jobs(
        [(OBSERVATION_K + c, list(sims[k]["z"])) for c, k in enumerate(pool)],
        base,
        path,
        workers,
        presence=True,
    )
    references = run_jobs(
        [(REFERENCE_K + i, list(sims[i]["z"])) for i in range(REFERENCES)],
        base,
        path,
        workers,
    )
    order = sorted(
        (c for c in range(len(pool)) if eligible(observations[OBSERVATION_K + c])),
        key=lambda c: observations[OBSERVATION_K + c]["D"],
    )
    candidates: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for start in range(0, len(order), cr.CANDIDATE_BATCH):
        batch = order[start : start + cr.CANDIDATE_BATCH]
        done = run_jobs(
            [
                (REPEAT_K + cr.REPEATS * c + q, list(sims[pool[c]]["z"]))
                for c in batch
                for q in range(cr.REPEATS)
            ],
            base,
            path,
            workers,
        )
        for c in batch:
            k = pool[c]
            observation = observations[OBSERVATION_K + c]
            repeats = [done[REPEAT_K + cr.REPEATS * c + q] for q in range(cr.REPEATS)]
            others = [references[REFERENCE_K + i] for i in range(REFERENCES) if i != k][:200]
            floor = median_distance_v2(observation, repeats[: cr.STABILITY_REPEATS])
            oracle = median_distance_v2(observation, repeats[cr.STABILITY_REPEATS :])
            prior = median_distance_v2(observation, others)
            keep, reason = keep_candidate_v2(
                observation,
                repeats[: cr.STABILITY_REPEATS],
                floor,
                prior,
                keys,
            )
            row = {
                "candidate": c,
                "training_simulation": k,
                "v2_D": observation["D"],
                "floor": floor,
                "oracle_D": oracle,
                "prior_D": prior,
                "kept": keep,
                "reason": reason,
            }
            candidates.append(row)
            if keep and len(kept) < 1 + cr.TRUTHS:
                kept.append(row)
        if len(kept) == 1 + cr.TRUTHS:
            break
    record: dict[str, Any] = {
        "description": DESCRIPTION,
        "base": base,
        "pool": len(pool),
        "eligible": len(order),
        "ineligible": [
            {"candidate": c, "training_simulation": pool[c]}
            for c in range(len(pool))
            if c not in order
        ],
        "candidates": candidates,
    }
    if len(kept) < 1 + cr.TRUTHS:
        record["passed"] = False
        record["stop"] = f"only {len(kept)} candidates kept; the run returns to the maintainer"
        return record
    centre = run_jobs(
        [(CENTRE_K + q, [cr.X0] * cr.DIMENSION) for q in range(cr.RESIMULATIONS)],
        base,
        path,
        workers,
    )
    centre_runs = [centre[CENTRE_K + q] for q in range(cr.RESIMULATIONS)]
    truths = kept[1:]
    for row in truths:
        observation = observations[OBSERVATION_K + row["candidate"]]
        row["oracle_gain"] = cr._gain(row["prior_D"], row["oracle_D"], row["floor"])
        row["centre_D"] = median_distance_v2(observation, centre_runs)
        row["centre_gain"] = cr._gain(row["prior_D"], row["centre_D"], row["floor"])
    oracle_gain = float(np.median([r["oracle_gain"] for r in truths]))
    start_gain = float(np.median([r["centre_gain"] for r in truths]))
    record |= {
        "pilot_truth": kept[0],
        "truths": truths,
        "oracle_median_gain": oracle_gain,
        "centre_median_gain": start_gain,
        "oracle_passes": oracle_gain >= cr.ORACLE_GATE,
        "centre_fails": start_gain <= cr.START_GATE,
    }
    record["passed"] = record["oracle_passes"] and record["centre_fails"]
    return record


def observation_of(row: dict[str, Any], path: Path) -> dict[str, Any]:
    """Return a truth's stored v2 observation from the candidates' simulations."""
    stored = {r["k"]: r for r in (json.loads(line) for line in path.read_text().splitlines())}
    return stored[OBSERVATION_K + row["candidate"]]


# ---------------------------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------------------------


def _write(path: Path, record: dict[str, Any]) -> None:
    cr._write_json(path, record)


def _gate_probe(workers: int) -> int:
    """Read the gates' oracle on discarded seeds before registration; description only."""
    record = select_truths(PROBE_BASE, BUILD / "probe" / "simulations.jsonl", workers)
    near = {}
    if "oracle_median_gain" in record:
        near = {
            "oracle": record["oracle_median_gain"] < cr.ORACLE_GATE + PROBE_MARGIN,
            "centre": record["centre_median_gain"] > cr.START_GATE - PROBE_MARGIN,
        }
    record["near_or_past_bar"] = near
    record["returns_to_maintainer"] = (not record.get("passed")) or any(near.values())
    _write(PROBE_PATH, record)
    print(json.dumps({k: v for k, v in record.items() if k != "candidates"}, indent=2))
    return 1 if record["returns_to_maintainer"] else 0


def _preflight(workers: int) -> int:
    record = select_truths(CANDIDATE_BASE, BUILD / "candidates.jsonl", workers)
    _write(PREFLIGHT_PATH, record)
    print(json.dumps({k: v for k, v in record.items() if k != "candidates"}, indent=2))
    return 0 if record.get("passed") else 1


def _resimulate(z: list[float], ks: list[int], seed_base: int, path: Path, workers: int) -> list:
    done = run_jobs([(k, z) for k in ks], seed_base, path, workers)
    return [done[k] for k in ks]


def _pilot(workers: int) -> int:
    """Run the pilot fit to 300 generations; set the cap by the guarded rule and project the cost."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if not preflight.get("passed"):
        print("the preflight did not pass; no pilot runs", file=sys.stderr)
        return 1
    truth = preflight["pilot_truth"]
    observation = observation_of(truth, BUILD / "candidates.jsonl")
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = cr.run_fit(
            observation,
            fit=0,
            cma_seed=PILOT_CMA_SEED,
            seed_base=PILOT_BASE,
            cap=cr.MAX_GENERATIONS,
            use_rule=False,
            state_dir=BUILD / "pilot",
            pool=pool,
            scales={},
            objective=Objective(observation),
            simulate=simulate_v2,
            stop=guarded_stop,
        )
    plateau = guarded_plateau_generation(history["medians"], history["sigmas"])
    at = {"end": cr.MAX_GENERATIONS}
    if plateau is not None:
        at |= {"plateau": plateau, "cap": cr.candidate_cap(plateau)}
    gains: dict[str, float] = {}
    path = BUILD / "pilot" / "answers.jsonl"
    for (name, generation), offset in zip(
        sorted(at.items(), key=lambda kv: kv[1]),
        cr.PILOT_ANSWER_OFFSETS,
        strict=False,
    ):
        runs = _resimulate(
            history["answers"][generation - 1],
            [offset + q for q in range(cr.RESIMULATIONS)],
            PILOT_BASE,
            path,
            workers,
        )
        d = median_distance_v2(observation, runs)
        gains[name] = cr._gain(truth["prior_D"], d, truth["floor"])
    cap = cr.registered_cap(plateau, gains.get("cap"), gains["end"])
    seconds = history["seconds"]
    per_generation = (
        float(np.median(seconds[100:])) if len(seconds) > 100 else float(np.median(seconds))
    )
    hours = cr.TRUTHS * cr.RESTARTS * (cap + 1) * per_generation / 3600.0
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
            "gain_at_300_below_bar": gains["end"] < cr.PILOT_STOP,
            "projection_over_30_h": hours > PROJECTION_HOURS,
        },
        "medians": history["medians"],
        "sigmas": history["sigmas"],
    }
    _write(PILOT_PATH, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("medians", "sigmas")}, indent=2))
    return 1 if record["stops"]["gain_at_300_below_bar"] else 0


def _run(workers: int) -> int:
    """Run the registered fits to the registered cap, then re-simulate each answer."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    pilot = json.loads(PILOT_PATH.read_text())
    if not preflight.get("passed") or pilot["stops"]["gain_at_300_below_bar"]:
        print("the preflight or the pilot stopped the run; no registered fit runs", file=sys.stderr)
        return 1
    cap = int(pilot["registered_cap"])
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for fit in range(cr.TRUTHS * cr.RESTARTS):
            truth = preflight["truths"][fit // cr.RESTARTS]
            observation = observation_of(truth, BUILD / "candidates.jsonl")
            cr.run_fit(
                observation,
                fit=fit,
                cma_seed=CMA_SEED + fit,
                seed_base=FIT_BASE,
                cap=cap,
                use_rule=True,
                state_dir=BUILD / "fits",
                pool=pool,
                scales={},
                objective=Objective(observation),
                simulate=simulate_v2,
                stop=guarded_stop,
            )
    for fit in range(cr.TRUTHS * cr.RESTARTS):
        history = json.loads((BUILD / "fits" / f"fit{fit}.json").read_text())
        _resimulate(
            history["answers"][-1],
            [cr.answer_k(fit, q) for q in range(cr.RESIMULATIONS)],
            FIT_BASE,
            BUILD / "answers.jsonl",
            workers,
        )
    return 0


def _read() -> int:
    """Read the gate and the reported measures into the recovery record."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    stored = {
        r["k"]: r
        for r in (json.loads(x) for x in (BUILD / "answers.jsonl").read_text().splitlines())
    }
    sims = nr.load_simulations()
    rows = []
    for j, truth in enumerate(preflight["truths"]):
        observation = observation_of(truth, BUILD / "candidates.jsonl")
        objective = Objective(observation)
        fits = []
        for r in range(cr.RESTARTS):
            fit = cr.RESTARTS * j + r
            history = json.loads((BUILD / "fits" / f"fit{fit}.json").read_text())
            runs = [stored[cr.answer_k(fit, q)] for q in range(cr.RESIMULATIONS)]
            d = float(np.median([objective(x) for x in runs]))
            truth_z = np.asarray(sims[truth["training_simulation"]]["z"])
            error = np.abs(np.asarray(history["answers"][-1]) - truth_z)
            fits.append(
                {
                    "fit": fit,
                    "final_objective": cr.final_objective(history["medians"]),
                    "stop": history["stop"],
                    "answer_D": d,
                    "gain": cr._gain(truth["prior_D"], d, truth["floor"]),
                    "parameter_error": error.tolist(),
                    "median_parameter_error": float(np.median(error)),
                    "medians": history["medians"],
                    "sigmas": history["sigmas"],
                    "runs": runs,
                },
            )
        best = cr.best_restart([f["medians"] for f in fits])
        spread = float(
            np.median(
                [
                    predictive_distance_v2(a, b, objective.scales, objective.family_map)
                    for a, b in zip(fits[0]["runs"], fits[1]["runs"], strict=True)
                ],
            ),
        )
        for f in fits:
            del f["runs"]
        rows.append(
            {
                "truth": truth["training_simulation"],
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
        "gate": cr.GAIN_GATE,
        "passes": median_gain >= cr.GAIN_GATE,
        "truths": rows,
    }
    _write(RECOVERY_PATH, record)
    print(json.dumps({"median_gain": median_gain, "passes": record["passes"]}, indent=2))
    return 0


def _smoke(workers: int) -> int:
    """Run two generations of one fit and a resume, on discarded seeds."""
    sims = nr.load_simulations()
    z = list(sims[cr.candidate_pool(sims)[0]]["z"])
    observation = simulate_v2(0, z, SMOKE_BASE)
    state = BUILD / "smoke"
    shutil.rmtree(state, ignore_errors=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for cap in (1, 2):
            history = cr.run_fit(
                observation,
                fit=0,
                cma_seed=SMOKE_CMA_SEED,
                seed_base=SMOKE_BASE,
                cap=cap,
                use_rule=True,
                state_dir=state,
                pool=pool,
                scales={},
                objective=Objective(observation),
                simulate=simulate_v2,
                stop=guarded_stop,
            )
            if cap == 1:
                history["stop"] = None
                cr._write_json(state / "fit0.json", history)
    print(json.dumps({k: history[k] for k in ("medians", "sigmas", "stop", "seconds")}, indent=2))
    ok = len(history["medians"]) == 2 and all(0.0 <= v <= 1.0 for v in history["answers"][-1])
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """CLI: the gates' probe, the preflight, the pilot, the fits, the reading and a smoke."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "stage",
        choices=("gate-probe", "preflight", "pilot", "run", "read", "smoke"),
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.stage == "smoke":
        return _smoke(args.workers)
    if args.stage == "gate-probe":
        return _gate_probe(args.workers)
    if args.stage in ("preflight", "pilot") and not cr.launch_committed(launch_path=LAUNCH_PATH):
        print(f"{LAUNCH_PATH} is not committed and clean; nothing is run.", file=sys.stderr)
        return 1
    if args.stage in ("run", "read") and not cr.launch_committed(
        bound=(PREFLIGHT_PATH, PILOT_PATH),
        launch_path=LAUNCH_PATH,
    ):
        print(
            f"{LAUNCH_PATH} with Part B and its records is not committed and clean.",
            file=sys.stderr,
        )
        return 1
    stages = {"preflight": _preflight, "pilot": _pilot, "run": _run}
    if args.stage == "read":
        return _read()
    return stages[args.stage](args.workers)


if __name__ == "__main__":
    sys.exit(main())
