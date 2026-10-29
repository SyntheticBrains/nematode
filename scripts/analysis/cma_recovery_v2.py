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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cma_recovery as cr
import npe_recovery as nr

DESCRIPTION = "Parameter recovery for C.1's model by sep-CMA, re-read on benchmark v2."
_REPO = Path(__file__).resolve().parents[2]
SUPPORT = _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "116-cma-recovery-v2"
LAUNCH_PATH = SUPPORT / "launch.md"
PREFLIGHT_PATH = SUPPORT / "preflight.json"
PILOT_PATH = SUPPORT / "pilot.json"
# Amendment 1: the pilot re-run on the uncapped fit objective, on its own seeds and record.
PILOT_A1_PATH = SUPPORT / "pilot_amendment1.json"
RECOVERY_PATH = SUPPORT / "recovery.json"
BUILD = _REPO / "build" / "cma-recovery-v2"
PROBE_PATH = BUILD / "gate_probe.json"
CANDIDATES_PATH = BUILD / "candidates.jsonl"

FIT_BASE = 170_000_000
PILOT_BASE = 200_000_000
CANDIDATE_BASE = 210_000_000
PROBE_BASE = 99_400_000
SMOKE_BASE = 99_300_000
CMA_SEED = 5420
PILOT_CMA_SEED = 5428
PILOT_A1_BASE = 220_000_000
PILOT_A1_CMA_SEED = 5429
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
    *,
    capped: bool = True,
) -> float:
    """Return the distance between two v2 records' graded statistics on the bands' scales.

    Families weigh equally, split equally within each. A statistic undefined in both counts 0, and
    one undefined in only one counts the cap of 10. A reported, ungraded statistic takes no part,
    since only the family map's statistics are read.

    With ``capped`` (the registered D) each z is capped at 10. Without it, as a fit's objective,
    defined statistics keep their full z, so a statistic far from the truth still ranks candidates
    by how far; an undefined one still counts the cap.
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
            elif capped:
                z = g.z_score(float(b) - float(a), g.Band(0.0, scales[key]))
            else:
                z = (float(b) - float(a)) / scales[key]
            readings[family][s] = g.Reading(0.0, g.Band(0.0, scales[key]), z, passes=True)
    return g.distance(readings, list(family_map), family_map)


UNDEFINED_RANK = 1_000.0
COMPRESSION = 10.0


def ranking_score(
    truth: dict[str, Any],
    draw: dict[str, Any],
    scales: dict[str, float],
    family_map: dict[str, tuple[str, ...]],
) -> float:
    """Return a fit's ranking score: definedness first, then a compressed distance.

    The score is 1,000 x u + D_c. u counts the graded statistics undefined in exactly one of the two
    records; D_c is the weighted RMS (families equal, split equally within each) of
    10·asinh(z/10) over the statistics defined in both, for z uncapped on the bands' scales, a
    statistic undefined in both counting 0. D_c stays below 1,000 for any physically possible
    record, so fewer undefined statistics always rank ahead; near the truth each term is z to
    within 1.5% for |z| <= 3.
    """
    from quantumnematode.benchmark import grading as g

    readings: dict[str, dict[str, Any]] = {f: {} for f in family_map}
    undefined = 0
    for family, stats in family_map.items():
        for s in stats:
            key = f"{family}/{s}"
            a, b = truth["statistics"].get(key), draw["statistics"].get(key)
            if (a is None) != (b is None):
                undefined += 1
                term = 0.0
            elif a is None:
                term = 0.0
            else:
                term = COMPRESSION * float(
                    np.arcsinh((float(b) - float(a)) / scales[key] / COMPRESSION),
                )
            readings[family][s] = g.Reading(0.0, g.Band(0.0, scales[key]), term, passes=True)
    return UNDEFINED_RANK * undefined + g.distance(readings, list(family_map), family_map)


class Objective:
    """The distance from one observation to a simulation, picklable and with its scales loaded.

    ``capped`` is the registered D. A fit ranks its candidates on the uncapped distance, or with
    ``ranking`` on :func:`ranking_score`.
    """

    def __init__(
        self,
        observation: dict[str, Any],
        *,
        capped: bool = True,
        ranking: bool = False,
    ) -> None:
        bands = _bands()
        self.capped = capped
        self.ranking = ranking
        self.observation = observation
        self.scales = band_scales_v2(bands)
        self.family_map = {f: tuple(st) for f, st in bands["families"].items()}

    def __call__(self, record: dict[str, Any]) -> float:
        """Return the distance from the observation to one simulation's record."""
        if self.ranking:
            return ranking_score(self.observation, record, self.scales, self.family_map)
        return predictive_distance_v2(
            self.observation,
            record,
            self.scales,
            self.family_map,
            capped=self.capped,
        )


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


def read_records(path: Path) -> dict[int, dict[str, Any]]:
    """Return a JSONL store's complete records by k; an unterminated last line is ignored.

    A line is complete when it ends in a newline. A run killed mid-write leaves its last line torn,
    and that job stays eligible to run again.
    """
    if not path.exists():
        return {}
    text = path.read_text()
    lines = text.split("\n")
    complete = lines[:-1]  # the piece after the last newline is empty, or a torn line
    return {r["k"]: r for r in (json.loads(line) for line in complete if line)}


def _drop_torn_line(path: Path) -> None:
    """Truncate a store to its last complete line, so appended records start on a fresh line."""
    if not path.exists():
        return
    text = path.read_text()
    if text and not text.endswith("\n"):
        path.write_text(text[: text.rfind("\n") + 1])


def run_jobs(
    jobs: list[tuple[int, list[float]]],
    seed_base: int,
    path: Path,
    workers: int,
    *,
    presence: bool = False,
) -> dict[int, dict[str, Any]]:
    """Run the jobs not yet stored at ``path`` (one JSON line each); return every stored record.

    A stored record is reused only for the same parameters and, where it records one, the same
    seed base; any other match on k is refused, so a store is never silently mixed.

    Raises
    ------
    ValueError
        If a stored record for a requested k has other parameters or another seed base.
    """
    _drop_torn_line(path)
    stored = read_records(path)
    for k, z in jobs:
        record = stored.get(k)
        if record is None:
            continue
        if list(map(float, record["z"])) != list(map(float, z)):
            msg = f"{path}: stored k = {k} has other parameters than requested"
            raise ValueError(msg)
        if record.get("seed_base", seed_base) != seed_base:
            msg = f"{path}: stored k = {k} was run on seed base {record['seed_base']}, not {seed_base}"
            raise ValueError(msg)
    todo = [(k, z) for k, z in jobs if k not in stored]
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = simulate_v2_presence if presence else simulate_v2
    if todo:
        print(f"{len(stored)} stored, {len(todo)} to run on {workers} workers", flush=True)
        with (
            ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool,
            path.open("a") as out,
        ):
            futures = [pool.submit(entry, k, z, seed_base) for k, z in todo]
            for future in as_completed(futures):
                record = {**future.result(), "seed_base": seed_base}
                out.write(json.dumps(record, allow_nan=False) + "\n")
                out.flush()
    return read_records(path)


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
    return read_records(path)[OBSERVATION_K + row["candidate"]]


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


OBJECTIVE_PROBE_BASE = 99_900_000
OBJECTIVE_PROBE_CMA_SEED = 5994
OBJECTIVE_PROBE_GENERATIONS = 60


def _objective_probe(workers: int) -> int:
    """Fit the pilot truth for 60 generations on the uncapped objective; discarded seeds."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    truth = preflight["pilot_truth"]
    observation = observation_of(truth, BUILD / "candidates.jsonl")
    state = BUILD / "objective_probe"
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = cr.run_fit(
            observation,
            fit=0,
            cma_seed=OBJECTIVE_PROBE_CMA_SEED,
            seed_base=OBJECTIVE_PROBE_BASE,
            cap=OBJECTIVE_PROBE_GENERATIONS,
            use_rule=False,
            state_dir=state,
            pool=pool,
            scales={},
            objective=Objective(observation, capped=False),
            simulate=simulate_v2,
            stop=guarded_stop,
        )
    runs = _resimulate(
        history["answers"][-1],
        [cr.ANSWER_OFFSET + q for q in range(cr.RESIMULATIONS)],
        OBJECTIVE_PROBE_BASE,
        state / "answers.jsonl",
        workers,
    )
    d = median_distance_v2(observation, runs)
    record = {
        "description": "Description: the pilot truth fitted on the uncapped objective, discarded seeds.",
        "generations": OBJECTIVE_PROBE_GENERATIONS,
        "answer_capped_D": d,
        "gain": cr._gain(truth["prior_D"], d, truth["floor"]),
        "medians": history["medians"],
        "sigmas": history["sigmas"],
    }
    _write(state / "probe.json", record)
    print(json.dumps({k: record[k] for k in ("answer_capped_D", "gain")}, indent=2))
    return 0


def _preflight(workers: int) -> int:
    record = select_truths(CANDIDATE_BASE, BUILD / "candidates.jsonl", workers)
    _write(PREFLIGHT_PATH, record)
    print(json.dumps({k: v for k, v in record.items() if k != "candidates"}, indent=2))
    return 0 if record.get("passed") else 1


def _resimulate(z: list[float], ks: list[int], seed_base: int, path: Path, workers: int) -> list:
    done = run_jobs([(k, z) for k in ks], seed_base, path, workers)
    return [done[k] for k in ks]


def _pilot(workers: int) -> int:
    """Run the pilot to 300 generations on the uncapped objective; set the cap and project the cost.

    The registered pilot on the capped objective stopped the run (``pilot.json``); amendment 1's
    pilot runs on its own seeds and writes its own record.
    """
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
            cma_seed=PILOT_A1_CMA_SEED,
            seed_base=PILOT_A1_BASE,
            cap=cr.MAX_GENERATIONS,
            use_rule=False,
            state_dir=BUILD / "pilot_amendment1",
            pool=pool,
            scales={},
            objective=Objective(observation, capped=False),
            simulate=simulate_v2,
            stop=guarded_stop,
        )
    plateau = guarded_plateau_generation(history["medians"], history["sigmas"])
    at = {"end": cr.MAX_GENERATIONS}
    if plateau is not None:
        at |= {"plateau": plateau, "cap": cr.candidate_cap(plateau)}
    gains: dict[str, float] = {}
    path = BUILD / "pilot_amendment1" / "answers.jsonl"
    for (name, generation), offset in zip(
        sorted(at.items(), key=lambda kv: kv[1]),
        cr.PILOT_ANSWER_OFFSETS,
        strict=False,
    ):
        runs = _resimulate(
            history["answers"][generation - 1],
            [offset + q for q in range(cr.RESIMULATIONS)],
            PILOT_A1_BASE,
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
    _write(PILOT_A1_PATH, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("medians", "sigmas")}, indent=2))
    return 1 if record["stops"]["gain_at_300_below_bar"] else 0


def _run(workers: int) -> int:
    """Run the registered fits to the registered cap, then re-simulate each answer."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    pilot = json.loads(PILOT_A1_PATH.read_text())
    if not preflight.get("passed") or pilot["stops"]["gain_at_300_below_bar"]:
        print("the preflight or the pilot stopped the run; no registered fit runs", file=sys.stderr)
        return 1
    _run_fits(
        preflight,
        cap=int(pilot["registered_cap"]),
        cma_seed=CMA_SEED,
        seed_base=FIT_BASE,
        build=BUILD,
        objective=lambda observation: Objective(observation, capped=False),
        workers=workers,
    )
    return 0


def _run_fits(  # noqa: PLR0913 - keyword-only: one attempt's registered settings
    preflight: dict[str, Any],
    *,
    cap: int,
    cma_seed: int,
    seed_base: int,
    build: Path,
    objective: Callable[[dict[str, Any]], Objective],
    workers: int,
) -> None:
    """Run 4 truths x 2 restarts to the guarded plateau or ``cap``, then re-simulate each answer.

    The fits' state goes to ``build/fits`` and the answers' simulations to
    ``build/answers.jsonl``; ``objective`` builds the ranking objective from an observation.
    """
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for fit in range(cr.TRUTHS * cr.RESTARTS):
            truth = preflight["truths"][fit // cr.RESTARTS]
            observation = observation_of(truth, CANDIDATES_PATH)
            cr.run_fit(
                observation,
                fit=fit,
                cma_seed=cma_seed + fit,
                seed_base=seed_base,
                cap=cap,
                use_rule=True,
                state_dir=build / "fits",
                pool=pool,
                scales={},
                objective=objective(observation),
                simulate=simulate_v2,
                stop=guarded_stop,
            )
    for fit in range(cr.TRUTHS * cr.RESTARTS):
        history = json.loads((build / "fits" / f"fit{fit}.json").read_text())
        _resimulate(
            history["answers"][-1],
            [cr.answer_k(fit, q) for q in range(cr.RESIMULATIONS)],
            seed_base,
            build / "answers.jsonl",
            workers,
        )


def _read() -> int:
    """Read the gate and the reported measures into the recovery record."""
    return _read_fits(BUILD, RECOVERY_PATH, DESCRIPTION)


def _read_fits(build: Path, out: Path, description: str) -> int:
    """Read the gate on the registered, capped D from one attempt's fits into ``out``."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    stored = read_records(build / "answers.jsonl")
    sims = nr.load_simulations()
    rows = []
    for j, truth in enumerate(preflight["truths"]):
        observation = observation_of(truth, CANDIDATES_PATH)
        objective = Objective(observation)
        fits = []
        for r in range(cr.RESTARTS):
            fit = cr.RESTARTS * j + r
            history = json.loads((build / "fits" / f"fit{fit}.json").read_text())
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
        "description": description,
        "median_gain": median_gain,
        "gate": cr.GAIN_GATE,
        "passes": median_gain >= cr.GAIN_GATE,
        "truths": rows,
    }
    _write(out, record)
    print(json.dumps({"median_gain": median_gain, "passes": record["passes"]}, indent=2))
    return 0


DIAGNOSIS_PATH = SUPPORT / "diagnosis.json"


def _decompose(
    observation: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return each graded statistic's weight, the observation's value, and each group's median |z|.

    z is uncapped; a statistic defined in the observation and undefined in a record reads None, and
    each group also reports how many of its records leave the statistic undefined.
    """
    from quantumnematode.benchmark import grading as g

    bands = _bands()
    scales = band_scales_v2(bands)
    family_map = {f: tuple(st) for f, st in bands["families"].items()}
    weights = g.weights(list(family_map), family_map)
    rows = []
    for (family, stat), weight in weights.items():
        key = f"{family}/{stat}"
        a = observation["statistics"].get(key)
        row: dict[str, Any] = {"statistic": key, "weight": weight, "observation": a}
        for name, records in groups.items():
            values = [r["statistics"].get(key) for r in records]
            undefined = sum(v is None for v in values)
            defined = [abs(v - a) / scales[key] for v in values if v is not None and a is not None]
            row[name] = {
                "median_abs_z": float(np.median(defined)) if defined else None,
                "undefined": undefined,
            }
        rows.append(row)
    return rows


def _diagnose() -> int:
    """Record the per-statistic decompositions behind the pilot's and the failed truths' readings."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    stored = read_records(BUILD / "candidates.jsonl")
    pilot_answers = read_records(BUILD / "pilot" / "answers.jsonl")
    answers = read_records(BUILD / "answers.jsonl")

    def repeats(row: dict[str, Any]) -> list[dict[str, Any]]:
        c = row["candidate"]
        return [stored[REPEAT_K + cr.REPEATS * c + q] for q in range(cr.STABILITY_REPEATS)]

    pilot_truth = preflight["pilot_truth"]
    record: dict[str, Any] = {
        "description": "Description: per-statistic decompositions, uncapped z on v2's n = 200 scales.",
        "capped_pilot": {
            "truth": pilot_truth["training_simulation"],
            "statistics": _decompose(
                stored[OBSERVATION_K + pilot_truth["candidate"]],
                {
                    "own_repeats": repeats(pilot_truth),
                    "answer_at_300": [pilot_answers[5_200 + q] for q in range(cr.RESIMULATIONS)],
                },
            ),
        },
        "truths": [],
    }
    for j, truth in enumerate(preflight["truths"]):
        groups = {"own_repeats": repeats(truth)}
        for r in range(cr.RESTARTS):
            fit = cr.RESTARTS * j + r
            groups[f"fit_{fit}"] = [answers[cr.answer_k(fit, q)] for q in range(cr.RESIMULATIONS)]
        record["truths"].append(
            {
                "truth": truth["training_simulation"],
                "floor": truth["floor"],
                "statistics": _decompose(stored[OBSERVATION_K + truth["candidate"]], groups),
            },
        )
    _write(DIAGNOSIS_PATH, record)
    print(f"wrote {DIAGNOSIS_PATH}")
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
                objective=Objective(observation, capped=False),
                simulate=simulate_v2,
                stop=guarded_stop,
            )
            if cap == 1:
                history["stop"] = None
                cr._write_json(state / "fit0.json", history)
    print(json.dumps({k: history[k] for k in ("medians", "sigmas", "stop", "seconds")}, indent=2))
    ok = len(history["medians"]) == 2 and all(0.0 <= v <= 1.0 for v in history["answers"][-1])
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------
# Attempt 2: the ranking score on the registered truths
# ---------------------------------------------------------------------------------------------

SUPPORT_B = _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "117-cma-recovery-v2b"
LAUNCH_B_PATH = SUPPORT_B / "launch.md"
PROBE_B_PATH = SUPPORT_B / "probe.json"
RECOVERY_B_PATH = SUPPORT_B / "recovery.json"
BUILD_B = BUILD / "attempt2"
# The reused truths' observations live off-repo; their file is pinned by this digest.
CANDIDATES_SHA256 = "7728d7f0e7c950be49103adf41cd281a95ad8846011c30ac2b92b6a0ecbc7c2e"
PROBE_B_BASE = 300_000_000
PROBE_B_CMA_SEED = 5995
PROBE_B_TRUTHS = ((1164, 200, True), (6985, 200, True), (9105, 80, False))
PROBE_B_BAR = 0.80
FIT_B_BASE = 230_000_000
CMA_B_SEED = 5430
CAP_B = 272


def candidates_pinned() -> bool:
    """Return whether the reused observations' file still matches its registered digest."""
    from quantumnematode.benchmark.measurement import file_sha256

    return CANDIDATES_PATH.exists() and file_sha256(CANDIDATES_PATH) == CANDIDATES_SHA256


def _truth(preflight: dict[str, Any], simulation: int) -> dict[str, Any]:
    return next(t for t in preflight["truths"] if t["training_simulation"] == simulation)


def _fit_b(  # noqa: PLR0913 - one attempt-2 fit's settings
    truth: dict[str, Any],
    *,
    fit: int,
    cma_seed: int,
    seed_base: int,
    cap: int,
    use_rule: bool,
    state_dir: Path,
    pool: Any,  # noqa: ANN401 - a pool
) -> dict[str, Any]:
    observation = observation_of(truth, CANDIDATES_PATH)
    return cr.run_fit(
        observation,
        fit=fit,
        cma_seed=cma_seed,
        seed_base=seed_base,
        cap=cap,
        use_rule=use_rule,
        state_dir=state_dir,
        pool=pool,
        scales={},
        objective=Objective(observation, ranking=True),
        simulate=simulate_v2,
        stop=guarded_stop,
    )


def _answer_gain(  # noqa: PLR0913 - one answer's re-simulation settings
    truth: dict[str, Any],
    history: dict[str, Any],
    fit: int,
    base: int,
    path: Path,
    workers: int,
) -> tuple[float, float]:
    runs = _resimulate(
        history["answers"][-1],
        [cr.answer_k(fit, q) for q in range(cr.RESIMULATIONS)],
        base,
        path,
        workers,
    )
    d = median_distance_v2(observation_of(truth, CANDIDATES_PATH), runs)
    return d, cr._gain(truth["prior_D"], d, truth["floor"])


def _probe_b(workers: int) -> int:
    """Fit the failed truths and one recovered on the ranking score; discarded seeds."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for fit, (simulation, cap, use_rule) in enumerate(PROBE_B_TRUTHS):
            truth = _truth(preflight, simulation)
            history = _fit_b(
                truth,
                fit=fit,
                cma_seed=PROBE_B_CMA_SEED + fit,
                seed_base=PROBE_B_BASE,
                cap=cap,
                use_rule=use_rule,
                state_dir=BUILD_B / "probe",
                pool=pool,
            )
            d, gain = _answer_gain(
                truth,
                history,
                fit,
                PROBE_B_BASE,
                BUILD_B / "probe" / "answers.jsonl",
                workers,
            )
            rows.append(
                {
                    "truth": simulation,
                    "floor": truth["floor"],
                    "prior_D": truth["prior_D"],
                    "stop": history["stop"],
                    "answer_D": d,
                    "gain": gain,
                    "medians": history["medians"],
                    "sigmas": history["sigmas"],
                },
            )
            print(f"probe truth {simulation}: gain {gain:.3f}, stop {history['stop']}", flush=True)
    record = {
        "description": "Description: the ranking score on discarded seeds, before registration.",
        "bar": PROBE_B_BAR,
        "truths": rows,
        "returns_to_maintainer": any(r["gain"] < PROBE_B_BAR for r in rows),
    }
    _write(PROBE_B_PATH, record)
    return 1 if record["returns_to_maintainer"] else 0


def _run_b(workers: int) -> int:
    """Run attempt 2's registered fits on the ranking score, then re-simulate each answer."""
    _run_fits(
        json.loads(PREFLIGHT_PATH.read_text()),
        cap=CAP_B,
        cma_seed=CMA_B_SEED,
        seed_base=FIT_B_BASE,
        build=BUILD_B,
        objective=lambda observation: Objective(observation, ranking=True),
        workers=workers,
    )
    return 0


def _read_b() -> int:
    """Read attempt 2's gate, on the registered capped D, into its recovery record."""
    return _read_fits(
        BUILD_B,
        RECOVERY_B_PATH,
        "Attempt 2: sep-CMA ranking on definedness then the compressed z, on attempt 1's truths.",
    )


PROBE_B_DIAGNOSIS_PATH = SUPPORT_B / "probe_diagnosis.json"
PROBE_B2_PATH = SUPPORT_B / "probe_restart.json"
# The second restart of the truth the first probe left short, on the probe's discarded seeds.
PROBE_B2_TRUTH = 1164
PROBE_B2_FIT = len(PROBE_B_TRUTHS)


def _probe_b2(workers: int) -> int:
    """Fit a second restart of the short truth and read the better restart as the gate selects it."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    probe = json.loads(PROBE_B_PATH.read_text())
    first = next(r for r in probe["truths"] if r["truth"] == PROBE_B2_TRUTH)
    _, cap, use_rule = next(t for t in PROBE_B_TRUTHS if t[0] == PROBE_B2_TRUTH)
    truth = _truth(preflight, PROBE_B2_TRUTH)
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = _fit_b(
            truth,
            fit=PROBE_B2_FIT,
            cma_seed=PROBE_B_CMA_SEED + PROBE_B2_FIT,
            seed_base=PROBE_B_BASE,
            cap=cap,
            use_rule=use_rule,
            state_dir=BUILD_B / "probe",
            pool=pool,
        )
    d, gain = _answer_gain(
        truth,
        history,
        PROBE_B2_FIT,
        PROBE_B_BASE,
        BUILD_B / "probe" / "answers.jsonl",
        workers,
    )
    restarts = [
        {"fit": 0, "gain": first["gain"], "stop": first["stop"], "medians": first["medians"]},
        {"fit": PROBE_B2_FIT, "gain": gain, "stop": history["stop"], "medians": history["medians"]},
    ]
    best = cr.best_restart([r["medians"] for r in restarts])
    record = {
        "description": "Description: a second restart of the probe's short truth, on discarded "
        "seeds, with the better restart selected by its final objective as the gate selects it.",
        "truth": PROBE_B2_TRUTH,
        "bar": PROBE_B_BAR,
        "second_restart": {"answer_D": d, "gain": gain, "sigmas": history["sigmas"]}
        | {k: restarts[1][k] for k in ("stop", "medians")},
        "final_objectives": [cr.final_objective(r["medians"]) for r in restarts],
        "selected_restart": restarts[best]["fit"],
        "selected_gain": restarts[best]["gain"],
        "returns_to_maintainer": restarts[best]["gain"] < PROBE_B_BAR,
    }
    _write(PROBE_B2_PATH, record)
    print(
        f"second restart: gain {gain:.3f}, stop {history['stop']}; "
        f"selected fit {record['selected_restart']}, gain {record['selected_gain']:.3f}",
        flush=True,
    )
    return 1 if record["returns_to_maintainer"] else 0


def _diagnose_b() -> int:
    """Record each probed truth's per-statistic decomposition: its own repeats against its answer."""
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    stored = read_records(CANDIDATES_PATH)
    answers = read_records(BUILD_B / "probe" / "answers.jsonl")
    truths = []
    for fit, (simulation, _, _) in enumerate(PROBE_B_TRUTHS):
        truth = _truth(preflight, simulation)
        c = truth["candidate"]
        truths.append(
            {
                "truth": simulation,
                "floor": truth["floor"],
                "statistics": _decompose(
                    stored[OBSERVATION_K + c],
                    {
                        "own_repeats": [
                            stored[REPEAT_K + cr.REPEATS * c + q]
                            for q in range(cr.STABILITY_REPEATS)
                        ],
                        "probe_answer": [
                            answers[cr.answer_k(fit, q)] for q in range(cr.RESIMULATIONS)
                        ],
                    },
                ),
            },
        )
    if PROBE_B2_PATH.exists():
        row = next(t for t in truths if t["truth"] == PROBE_B2_TRUTH)
        c = _truth(preflight, PROBE_B2_TRUTH)["candidate"]
        second = _decompose(
            stored[OBSERVATION_K + c],
            {
                "second_restart": [
                    answers[cr.answer_k(PROBE_B2_FIT, q)] for q in range(cr.RESIMULATIONS)
                ],
            },
        )
        for statistic, extra in zip(row["statistics"], second, strict=True):
            statistic["second_restart"] = extra["second_restart"]
    record = {
        "description": "Description: the probe's per-statistic decompositions, uncapped z on v2's "
        "n = 200 scales.",
        "truths": truths,
    }
    _write(PROBE_B_DIAGNOSIS_PATH, record)
    print(f"wrote {PROBE_B_DIAGNOSIS_PATH}")
    return 0


def attempt_b(stage: str, workers: int) -> int:
    """Run an attempt-2 stage behind its guards: the pinned observations, then the launch record."""
    if not candidates_pinned():
        print(
            f"{CANDIDATES_PATH} does not match its registered SHA-256; nothing runs.",
            file=sys.stderr,
        )
        return 1
    probes = {"probe-b": _probe_b, "diagnose-b": lambda _: _diagnose_b(), "probe-b2": _probe_b2}
    if stage in probes:
        return probes[stage](workers)
    if not probe_cleared():
        print(f"{PROBE_B_PATH} is missing or holds a gain below the bar.", file=sys.stderr)
        return 1
    if not cr.launch_committed(
        bound=(PREFLIGHT_PATH, PILOT_A1_PATH, PROBE_B_PATH, PROBE_B2_PATH),
        launch_path=LAUNCH_B_PATH,
    ):
        print(f"{LAUNCH_B_PATH} is not committed and clean; nothing runs.", file=sys.stderr)
        return 1
    return _run_b(workers) if stage == "run-b" else _read_b()


def probe_cleared() -> bool:
    """Return whether the probe ran and every probed truth reached the bar.

    Where the short truth's second restart has run, that truth reads the restart the gate would
    select, and the probe clears only if that one reaches the bar.
    """
    if not PROBE_B_PATH.exists():
        return False
    gains = {r["truth"]: r["gain"] for r in json.loads(PROBE_B_PATH.read_text())["truths"]}
    if PROBE_B2_PATH.exists():
        gains[PROBE_B2_TRUTH] = json.loads(PROBE_B2_PATH.read_text())["selected_gain"]
    return all(g >= PROBE_B_BAR for g in gains.values())


def main(argv: list[str] | None = None) -> int:
    """CLI: the gates' probe, the preflight, the pilot, the fits, the reading and a smoke."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "stage",
        choices=(
            "gate-probe",
            "objective-probe",
            "preflight",
            "pilot",
            "run",
            "read",
            "diagnose",
            "smoke",
            "probe-b",
            "diagnose-b",
            "probe-b2",
            "run-b",
            "read-b",
        ),
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.stage in ("probe-b", "diagnose-b", "probe-b2", "run-b", "read-b"):
        return attempt_b(args.stage, args.workers)
    unguarded = {"smoke": _smoke, "gate-probe": _gate_probe, "objective-probe": _objective_probe}
    if args.stage in unguarded:
        return unguarded[args.stage](args.workers)
    if args.stage in ("preflight", "pilot") and not cr.launch_committed(launch_path=LAUNCH_PATH):
        print(f"{LAUNCH_PATH} is not committed and clean; nothing is run.", file=sys.stderr)
        return 1
    if args.stage in ("run", "read") and not cr.launch_committed(
        bound=(PREFLIGHT_PATH, PILOT_PATH, PILOT_A1_PATH),
        launch_path=LAUNCH_PATH,
    ):
        print(
            f"{LAUNCH_PATH} with Part B and its records is not committed and clean.",
            file=sys.stderr,
        )
        return 1
    stages = {
        "preflight": _preflight,
        "pilot": _pilot,
        "run": _run,
        "read": lambda _: _read(),
        "diagnose": lambda _: _diagnose(),
    }
    return stages[args.stage](args.workers)


if __name__ == "__main__":
    sys.exit(main())
