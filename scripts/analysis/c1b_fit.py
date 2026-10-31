#!/usr/bin/env python
r"""C.1's mechanistic baseline refitted to the real wild type on benchmark v2, and its verdict.

**The model** is ``BatchedMechanistic`` on F.0a's box. **The objective** ranks each candidate by
definedness first, then a compressed distance (``cma_recovery_v2.ranking_score``), against the
bands' centres and scales at n = 200, for 200 fresh animals on v2's four assays through its
measurement models. **The fit** is sep-CMA from the box centre, 8 restarts, each stopping on the
plateau rule once sigma <= sigma0 and its last 40 generations hold no median with an undefined
statistic, or at the cap.

**Selection** is apart from the verdict: every plateaued restart is graded with presence on 3
selection populations, and the one passing most is chosen (ties: the lower median D, then the
lower restart). **The verdict** reads the chosen restart on 5 registered populations, graded with
presence: it passes the wild type when at least 3 pass. Every restart is graded there too.

**The pilot**, on discarded seeds before the fits, runs one restart 300 generations to set the cap,
project the cost, and grade its answer's reachability on 5 discarded populations.

Subcommands (``pilot`` refuses until Part A of the launch record is committed and unchanged; the
rest also need Part B, the pilot's record and the widened-box probe's; each binds the mutant
encoding table; ``probe-wide`` and ``smoke`` are unguarded and discarded)::

    uv run python scripts/analysis/c1b_fit.py pilot
    uv run python scripts/analysis/c1b_fit.py run
    uv run python scripts/analysis/c1b_fit.py select
    uv run python scripts/analysis/c1b_fit.py grade
    uv run python scripts/analysis/c1b_fit.py read
    uv run python scripts/analysis/c1b_fit.py smoke    # execution check, discarded
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cma_recovery as cr
import cma_recovery_v2 as v2
import npe_recovery as nr

DESCRIPTION = "C.1's mechanistic baseline refitted to the wild type on benchmark v2."
_REPO = Path(__file__).resolve().parents[2]
_LOGBOOKS = _REPO / "docs" / "experiments" / "logbooks" / "supporting"
SUPPORT = _LOGBOOKS / "119-c1b-fit"
LAUNCH_PATH = SUPPORT / "launch.md"
PILOT_PATH = SUPPORT / "pilot.json"
FIT_PATH = SUPPORT / "fit.json"
ENCODINGS_PATH = _LOGBOOKS / "113-c1-fit" / "encodings.md"
BUILD = _REPO / "build" / "c1b-fit"

RESTARTS = 8
CMA_SEED = 5749
PILOT_CMA_SEED = 5740
FIT_BASE = 400_000_000
GRADE_BASE = 420_000_000
SELECT_BASE = 421_000_000
PILOT_BASE = 430_000_000
PILOT_GRADE_BASE = 432_000_000
SMOKE_CMA_SEED = 5999
SMOKE_BASE = 99_050_000
SMOKE_GRADE_BASE = 99_060_000
GRADE_POPULATIONS = 5
SELECT_POPULATIONS = 3
PASS_NEEDED = 3
RESIMULATIONS = 10
# F.0b-i's cap fallback (0.05 of gain) on a D scale: times the median prior - floor of its v2
# truths, 5.42.
CAP_TOLERANCE_D = 0.27
PROJECTION_HOURS = 20.0
# Set only by a maintainer's decision registered in Part B: run although the pilot's answer did
# not reach the wild type. Set by the decision of 2026-11-01, registered in Part B.
REACHABILITY_ACCEPTED = True
RECOVERED = ("kappa_weathervane", "search_tau_s")
# The widened-box probe: two parameters the pilot pinned at F.0a's upper bounds, with their upper
# bounds raised; neither range uses search data.
WIDE_UPPER = {"edge": 10.0, "a_intake": 10.0}
PROBE_WIDE_PATH = SUPPORT / "probe_wide.json"
PROBE_WIDE_BASE = 440_000_000
PROBE_WIDE_GRADE_BASE = 442_000_000


def parameters_from_box(z: np.ndarray, upper: dict[str, float] | None = None) -> dict[str, float]:
    """Map a point of the unit box to C.1's parameters, with some upper bounds replaced."""
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    out = {}
    for value, name in zip(np.asarray(z, dtype=float), MECHANISTIC_PARAMETERS, strict=True):
        kind, low, high = nr.PRIOR[name]
        high = (upper or {}).get(name, high)
        if kind == "log":
            out[name] = float(math.exp(math.log(low) + value * (math.log(high) - math.log(low))))
        else:
            out[name] = float(low + value * (high - low))
    return out


def simulate_wide(k: int, z: list[float], seed_base: int) -> dict[str, Any]:
    """Run simulation k at z on v2, z mapped through the widened box."""
    return v2.simulate_v2(
        k,
        z,
        seed_base,
        parameters=parameters_from_box(np.asarray(z), WIDE_UPPER),
    )


# ---------------------------------------------------------------------------------------------
# The objective and the stop rule
# ---------------------------------------------------------------------------------------------


class BandObjective:
    """The ranking score of a simulation against the real bands' centres, picklable."""

    def __init__(self) -> None:
        import benchmark_grade as bg

        bands = v2._bands()
        self.family_map = {f: tuple(st) for f, st in bands["families"].items()}
        self.scales = v2.band_scales_v2(bands)
        centres = {}
        for family, stats in self.family_map.items():
            family_bands = bg.family_bands_v2(bands, family, nr.ANIMALS)
            for s in stats:
                centres[f"{family}/{s}"] = float(family_bands[s].centre)
        self.truth = {"statistics": centres}

    def __call__(self, record: dict[str, Any]) -> float:
        """Return the ranking score of one simulation's record against the bands."""
        return v2.ranking_score(self.truth, record, self.scales, self.family_map)


def defined_stop(medians: list[float], sigma: float) -> bool:
    """Return whether a restart stops: the guarded plateau holds over a window with no undefined.

    Both windows the plateau rule compares, the last 40 generations, must hold no median of
    ``UNDEFINED_RANK`` or more, so a restart still bringing statistics into definition goes on.
    """
    window = medians[-2 * cr.PLATEAU_WINDOW :]
    return v2.guarded_stop(medians, sigma) and max(window) < v2.UNDEFINED_RANK


def defined_plateau_generation(medians: list[float], sigmas: list[float]) -> int | None:
    """Return the first generation (from 1) at which the doubly guarded rule holds, if any."""
    for n in range(cr.PLATEAU_FIRST, len(medians) + 1):
        if defined_stop(medians[:n], sigmas[n - 1]):
            return n
    return None


def pilot_cap(plateau: int | None, d_at_cap: float | None, d_at_end: float) -> int:
    """Return the cap: F.0b-i's rule, with its fallback on the registered D."""
    cap = cr.candidate_cap(plateau)
    if plateau is None or d_at_cap is None:
        return cr.MAX_GENERATIONS
    return cr.MAX_GENERATIONS if d_at_cap - d_at_end > CAP_TOLERANCE_D else cap


# ---------------------------------------------------------------------------------------------
# Selection and the verdict
# ---------------------------------------------------------------------------------------------


def _median_d(records: list[dict[str, Any]]) -> float:
    """Return the median D of graded records; an incomplete grade has no D and counts the cap."""
    from quantumnematode.benchmark import grading as g

    return float(np.median([g.Z_CAP if r["D"] is None else r["D"] for r in records]))


def chosen_restart(selection: dict[int, list[dict[str, Any]]]) -> int | None:
    """Return the selected restart: most selection populations passing, then lower D, then r.

    ``selection`` holds each plateaued restart's selection grades; None when it is empty.
    """
    if not selection:
        return None
    return min(
        selection,
        key=lambda r: (
            -sum(g["wild_type_passes"] for g in selection[r]),
            _median_d(selection[r]),
            r,
        ),
    )


def wild_type_verdict(grades: list[dict[str, Any]]) -> bool:
    """Return whether a restart passes the wild type: at least 3 of its 5 populations pass."""
    return sum(g["wild_type_passes"] for g in grades) >= PASS_NEEDED


def failing(grades: list[dict[str, Any]]) -> list[str]:
    """Return the statistics and presence tests failing in any of a restart's populations."""
    out = set()
    for g in grades:
        for family, readings in g["families"].items():
            out |= {f"{family}/{name}" for name, r in readings.items() if not r["passes"]}
        out |= {
            f"{family}/presence"
            for family, key in (
                ("local_search", "local_search_presence"),
                ("chemotaxis", "chemotaxis_presence"),
            )
            if g.get(key) is not None and not g[key]["present"]
        }
    return sorted(out)


# ---------------------------------------------------------------------------------------------
# Simulation and grading
# ---------------------------------------------------------------------------------------------


def grade_answer(k: int, z: list[float], base: int, wide: bool = False) -> dict[str, Any]:  # noqa: FBT001, FBT002 - a pool's positional argument
    """Grade an answer on population k at a base, with presence: every family and the pass.

    With ``wide``, z is mapped through the widened box.
    """
    import benchmark_grade as bg
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    bands = v2._bands()
    seeds = [base + nr.ANIMALS * k + i for i in range(nr.ANIMALS)]
    params = parameters_from_box(np.asarray(z), WIDE_UPPER if wide else None)
    populations, chemotaxis = bg.simulate_v2(
        "mlp",
        lambda _path: BatchedMechanistic(params),
        seeds,
        bands,
    )
    graded = bg.grade_populations_v2(populations, chemotaxis, bands, presence=True)
    return nr._finite(
        {
            "k": k,
            "base": base,
            "families": {
                family: {
                    name: {"value": r["value"], "z": r["z"], "passes": r["passes"]}
                    for name, r in readings.items()
                }
                for family, readings in graded["families"].items()
            },
            "family_passes": graded["family_passes"],
            "local_search_presence": graded.get("local_search_presence"),
            "chemotaxis_presence": graded.get("chemotaxis_presence"),
            "wild_type_passes": bool(graded["wild_type_passes"]),
            "D": graded["D"],
            "D_fit": graded["D_fit"],
        },
    )


def _grade_all(jobs: list[tuple[Any, ...]], workers: int) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        return list(pool.map(grade_answer, *zip(*jobs, strict=True)))


def _fit(  # noqa: PLR0913 - one restart's settings
    fit: int,
    *,
    cma_seed: int,
    seed_base: int,
    cap: int,
    use_rule: bool,
    state_dir: Path,
    pool: Any,  # noqa: ANN401 - a pool
    simulate: Any = None,  # noqa: ANN401 - a picklable simulator
) -> dict[str, Any]:
    return cr.run_fit(
        {},
        fit=fit,
        cma_seed=cma_seed,
        seed_base=seed_base,
        cap=cap,
        use_rule=use_rule,
        state_dir=state_dir,
        pool=pool,
        scales={},
        objective=BandObjective(),
        simulate=simulate or v2.simulate_v2,
        stop=defined_stop,
    )


def _median_answer_d(  # noqa: PLR0913 - one answer's re-simulation settings
    z: list[float],
    ks: list[int],
    base: int,
    path: Path,
    workers: int,
    simulate: Any = None,  # noqa: ANN401 - a picklable simulator
) -> float:
    if simulate is None:
        return _median_d(v2._resimulate(z, ks, base, path, workers))
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        stored = v2.read_records(path)
        todo = [k for k in ks if k not in stored]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as out:
            for record in pool.map(simulate, todo, [z] * len(todo), [base] * len(todo)):
                out.write(json.dumps({**record, "seed_base": base}, allow_nan=False) + "\n")
    stored = v2.read_records(path)
    return _median_d([stored[k] for k in ks])


# ---------------------------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------------------------


def _pilot(workers: int) -> int:
    """Run one restart 300 generations on discarded seeds: the cap, the cost, reachability."""
    return _pilot_run(workers, wide=False)


def _probe_wide(workers: int) -> int:
    """Run the pilot again on the widened box, on its own discarded seeds; description only."""
    return _pilot_run(workers, wide=True)


def _pilot_run(workers: int, *, wide: bool) -> int:
    state = BUILD / ("probe_wide" if wide else "pilot")
    base = PROBE_WIDE_BASE if wide else PILOT_BASE
    grade_base = PROBE_WIDE_GRADE_BASE if wide else PILOT_GRADE_BASE
    simulate = simulate_wide if wide else None
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = _fit(
            0,
            cma_seed=PILOT_CMA_SEED,
            seed_base=base,
            cap=cr.MAX_GENERATIONS,
            use_rule=False,
            state_dir=state,
            pool=pool,
            simulate=simulate,
        )
    plateau = defined_plateau_generation(history["medians"], history["sigmas"])
    at = {"end": cr.MAX_GENERATIONS}
    if plateau is not None:
        at |= {"plateau": plateau, "cap": cr.candidate_cap(plateau)}
    d: dict[str, float] = {}
    for (name, generation), offset in zip(
        sorted(at.items(), key=lambda kv: kv[1]),
        cr.PILOT_ANSWER_OFFSETS,
        strict=False,
    ):
        d[name] = _median_answer_d(
            history["answers"][generation - 1],
            [offset + q for q in range(RESIMULATIONS)],
            base,
            state / "answers.jsonl",
            workers,
            simulate,
        )
    cap = pilot_cap(plateau, d.get("cap"), d["end"])
    seconds = history["seconds"]
    per_generation = (
        float(np.median(seconds[100:])) if len(seconds) > 100 else float(np.median(seconds))
    )
    hours = RESTARTS * (cap + 1) * per_generation / 3600.0
    answer = history["answers"][-1]
    grades = _grade_all(
        [(g, answer, grade_base, wide) for g in range(GRADE_POPULATIONS)],
        workers,
    )
    passing = sum(g["wild_type_passes"] for g in grades)
    record: dict[str, Any] = {
        "description": "The reachability pilot on discarded seeds: the cap, the cost and whether "
        "its answer reaches the wild type.",
        "plateau_generation": plateau,
        "generations": at,
        "median_D": d,
        "registered_cap": cap,
        "sustained_seconds_per_generation": per_generation,
        "projected_hours": hours,
        "reachability": {
            "populations_passing": passing,
            "reaches": passing >= PASS_NEEDED,
            "failing": failing(grades),
            "grades": grades,
        },
        "returns_to_maintainer": passing < PASS_NEEDED or hours > PROJECTION_HOURS,
        "medians": history["medians"],
        "sigmas": history["sigmas"],
    }
    if wide:
        record["description"] = (
            "Description: the pilot re-run on a box with edge's and a_intake's upper bounds "
            "raised to 10, on its own discarded seeds, before any registered fit."
        )
        record["upper_bounds"] = WIDE_UPPER
        record["parameters"] = parameters_from_box(np.asarray(answer), WIDE_UPPER)
        record["answer"] = answer
    cr._write_json(PROBE_WIDE_PATH if wide else PILOT_PATH, record)
    summary = {
        k: record[k]
        for k in ("plateau_generation", "median_D", "registered_cap", "projected_hours")
    }
    print(json.dumps(summary | {"populations_passing": passing}, indent=2))
    return 1 if record["returns_to_maintainer"] else 0


def _histories() -> list[dict[str, Any]]:
    return [json.loads((BUILD / "fits" / f"fit{r}.json").read_text()) for r in range(RESTARTS)]


def _plateaued(histories: list[dict[str, Any]]) -> list[int]:
    return [r for r, h in enumerate(histories) if h["stop"] and h["stop"]["by"] == "plateau"]


def _run(workers: int) -> int:
    """Run the 8 restarts to the pilot's cap, then re-simulate each answer 10 times."""
    pilot = json.loads(PILOT_PATH.read_text())
    if not pilot["reachability"]["reaches"] and not REACHABILITY_ACCEPTED:
        print("the pilot's answer did not reach the wild type; no fit runs", file=sys.stderr)
        return 1
    cap = int(pilot["registered_cap"])
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for r in range(RESTARTS):
            _fit(
                r,
                cma_seed=CMA_SEED + r,
                seed_base=FIT_BASE,
                cap=cap,
                use_rule=True,
                state_dir=BUILD / "fits",
                pool=pool,
            )
    for r, history in enumerate(_histories()):
        v2._resimulate(
            history["answers"][-1],
            [cr.answer_k(r, q) for q in range(RESIMULATIONS)],
            FIT_BASE,
            BUILD / "answers.jsonl",
            workers,
        )
    return 0


def select_k(restart: int, population: int) -> int:
    """Return the simulation index of a restart's selection population (animals at SELECT_BASE)."""
    return SELECT_POPULATIONS * restart + population


def _select(workers: int) -> int:
    """Grade every plateaued restart on its 3 selection populations, with presence."""
    histories = _histories()
    plateaued = _plateaued(histories)
    jobs = [
        (select_k(r, s), histories[r]["answers"][-1], SELECT_BASE)
        for r in plateaued
        for s in range(SELECT_POPULATIONS)
    ]
    grades = _grade_all(jobs, workers) if jobs else []
    by_restart = {
        str(r): grades[i * SELECT_POPULATIONS : (i + 1) * SELECT_POPULATIONS]
        for i, r in enumerate(plateaued)
    }
    cr._write_json(BUILD / "selection.json", by_restart)
    return 0


def _grade(workers: int) -> int:
    """Grade every restart on the 5 registered populations, with presence."""
    histories = _histories()
    jobs = [(g, h["answers"][-1], GRADE_BASE) for h in histories for g in range(GRADE_POPULATIONS)]
    grades = _grade_all(jobs, workers)
    by_restart = [
        grades[r * GRADE_POPULATIONS : (r + 1) * GRADE_POPULATIONS] for r in range(RESTARTS)
    ]
    (BUILD / "grades.json").write_text(json.dumps(by_restart, indent=2, allow_nan=False) + "\n")
    return 0


def _spread(plateaued: list[int], runs: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    objective = BandObjective()
    pairs = [
        {
            "restarts": [i, j],
            "D": float(
                np.median(
                    [
                        v2.predictive_distance_v2(a, b, objective.scales, objective.family_map)
                        for a, b in zip(runs[i], runs[j], strict=True)
                    ],
                ),
            ),
        }
        for i, j in itertools.combinations(plateaued, 2)
    ]
    spread = [p["D"] for p in pairs]
    return {
        "pairs": len(spread),
        "median": float(np.median(spread)) if spread else None,
        "range": [min(spread), max(spread)] if spread else None,
        "each_pair": pairs,
    }


def _read() -> int:
    """Choose the restart on its selection populations and read its verdict on the registered."""
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    histories = _histories()
    plateaued = _plateaued(histories)
    selection = {int(r): g for r, g in json.loads((BUILD / "selection.json").read_text()).items()}
    grades = json.loads((BUILD / "grades.json").read_text())
    stored = v2.read_records(BUILD / "answers.jsonl")
    runs = {r: [stored[cr.answer_k(r, q)] for q in range(RESIMULATIONS)] for r in range(RESTARTS)}
    objective = BandObjective()
    restarts = [
        {
            "restart": r,
            "cma_seed": CMA_SEED + r,
            "stop": h["stop"],
            "final_score": cr.final_objective(h["medians"]),
            "answer_D_median": _median_d(runs[r]),
            "answer_score_median": float(np.median([objective(x) for x in runs[r]])),
            "selection_passing": (
                sum(g["wild_type_passes"] for g in selection[r]) if r in selection else None
            ),
            "population_passes": [g["wild_type_passes"] for g in grades[r]],
            "passes_wild_type": wild_type_verdict(grades[r]),
            "population_D": [g["D"] for g in grades[r]],
            "answer": h["answers"][-1],
            "medians": h["medians"],
            "sigmas": h["sigmas"],
        }
        for r, h in enumerate(histories)
    ]
    chosen = chosen_restart({r: selection[r] for r in plateaued})
    record: dict[str, Any] = {
        "description": DESCRIPTION,
        "chosen_restart": chosen,
        "restarts": restarts,
        "restart_spread_D": _spread(plateaued, runs),
    }
    if chosen is None:
        record["stop"] = "no restart plateaued; the run returns to the maintainer"
        cr._write_json(FIT_PATH, record)
        print(record["stop"], file=sys.stderr)
        return 1
    parameters = nr.parameters_from_unit(np.asarray(histories[chosen]["answers"][-1]))
    record |= {
        "passes_wild_type": wild_type_verdict(grades[chosen]),
        "populations_passing": int(sum(g["wild_type_passes"] for g in grades[chosen])),
        "diagnosis": failing(grades[chosen]),
        "parameters": {
            name: {"value": parameters[name], "recovered_in_F0b": name in RECOVERED}
            for name in MECHANISTIC_PARAMETERS
        },
        "grades": grades[chosen],
    }
    cr._write_json(FIT_PATH, record)
    keys = ("chosen_restart", "passes_wild_type", "populations_passing", "diagnosis")
    print(json.dumps({k: record[k] for k in keys}, indent=2))
    return 0


def _smoke(workers: int) -> int:
    """Run two generations of one restart and grade its answer once, on discarded seeds."""
    state = BUILD / "smoke"
    shutil.rmtree(state, ignore_errors=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = _fit(
            0,
            cma_seed=SMOKE_CMA_SEED,
            seed_base=SMOKE_BASE,
            cap=2,
            use_rule=True,
            state_dir=state,
            pool=pool,
        )
    graded = grade_answer(0, history["answers"][-1], SMOKE_GRADE_BASE)
    print(
        json.dumps({"medians": history["medians"], "wild_type_passes": graded["wild_type_passes"]}),
    )
    return 0 if len(history["medians"]) == 2 and graded["D"] is not None else 1


def _guarded(stage: str) -> bool:
    if stage == "pilot":
        bound: tuple[Path, ...] = (ENCODINGS_PATH,)
    else:
        bound = (ENCODINGS_PATH, PILOT_PATH, PROBE_WIDE_PATH)
    return cr.launch_committed(bound, launch_path=LAUNCH_PATH)


def main(argv: list[str] | None = None) -> int:
    """CLI: the pilot, the fits, selection, grading, the reading and a smoke."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "stage",
        choices=("pilot", "probe-wide", "run", "select", "grade", "read", "smoke"),
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.stage == "smoke":
        return _smoke(args.workers)
    if args.stage == "probe-wide":
        return _probe_wide(args.workers)
    if not _guarded(args.stage):
        print(
            "the launch record, its records or the encoding table is uncommitted or changed",
            file=sys.stderr,
        )
        return 1
    stages = {
        "pilot": _pilot,
        "run": _run,
        "select": _select,
        "grade": _grade,
        "read": lambda _: _read(),
    }
    return stages[args.stage](args.workers)


if __name__ == "__main__":
    sys.exit(main())
