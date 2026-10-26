#!/usr/bin/env python
r"""C.1's mechanistic baseline fitted to the wild type by sep-CMA, and its registered verdict.

**The model** is ``BatchedMechanistic`` on F.0a's box (``npe_recovery.PRIOR``). **The objective**
is the benchmark's own D on the 11 wild-type statistics, against the band cohorts at n = 200, for
200 fresh animals per evaluation. **The fit** is F.0b-i's loop (``cma_recovery.run_fit``): 8
restarts of sep-CMA from the box centre, each stopping at the plateau rule or the cap of 210. A
restart stopped by the cap has not met the plateau criterion and cannot be chosen; the chosen
restart is the plateaued one with the lowest mean median objective over its last 20 generations.

**The verdict.** Every restart's answer is graded on five registered 200-animal populations. A
population passes when every family passes (each statistic at |z| <= 2.84; local search only where
its presence test holds), and a restart passes the wild type when at least 3 of its 5 do.

Subcommands (``run``, ``grade`` and ``read`` refuse until the launch record is committed and
unchanged, with the encoding table byte-identical to its version there)::

    uv run python scripts/analysis/c1_fit.py run
    uv run python scripts/analysis/c1_fit.py grade
    uv run python scripts/analysis/c1_fit.py read
    uv run python scripts/analysis/c1_fit.py smoke    # execution check, discarded
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cma_recovery as cr
import npe_recovery as nr

DESCRIPTION = "C.1's mechanistic baseline fitted to the wild type by sep-CMA."
_REPO = Path(__file__).resolve().parents[2]
SUPPORT = _REPO / "docs" / "experiments" / "logbooks" / "supporting" / "113-c1-fit"
LAUNCH_PATH = SUPPORT / "launch.md"
ENCODINGS_PATH = SUPPORT / "encodings.md"
FIT_PATH = SUPPORT / "fit.json"
BUILD = _REPO / "build" / "c1-fit"

RESTARTS = 8
CMA_SEED = 5741
CAP = 210
SEED_BASE = 110_000_000
GRADE_BASE = 126_000_000
GRADE_POPULATIONS = 5
PASS_NEEDED = 3
RESIMULATIONS = 10
SMOKE_CMA_SEED = 5992
SMOKE_SEED_BASE = 99_100_000
SMOKE_GRADE_BASE = 99_200_000
RECOVERED = ("kappa_weathervane", "search_tau_s")


# ---------------------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------------------


def objective(record: dict[str, Any]) -> float:
    """Return a simulation's benchmark D to the bands; an incomplete grade counts the cap."""
    from quantumnematode.benchmark import grading as g

    d = record.get("D")
    return float(g.Z_CAP) if d is None else float(d)


def chosen_restart(histories: list[dict[str, Any]]) -> int | None:
    """Return the plateaued restart with the lowest final objective, or None if none plateaued."""
    plateaued = [
        (cr.final_objective(h["medians"]), r)
        for r, h in enumerate(histories)
        if h["stop"] is not None and h["stop"]["by"] == "plateau"
    ]
    return min(plateaued)[1] if plateaued else None


def population_passes(family_passes: dict[str, bool]) -> bool:
    """Return whether one graded population passes the wild type: every family passes."""
    return all(family_passes.values())


def wild_type_verdict(passes: list[bool]) -> bool:
    """Return whether a restart passes the wild type: at least 3 of its 5 populations pass."""
    return sum(passes) >= PASS_NEEDED


def mutant_verdict(passing_restarts: list[bool], *, chosen: bool) -> str:
    """Return a mutant's verdict under the ``unidentified`` rule.

    ``passing_restarts`` holds the mutant's verdict for each restart that passes the wild type. If
    they disagree the prediction is ``unidentified``; with none, the chosen restart's verdict is
    used, and the caller reports the wild-type failure beside it.
    """
    if not passing_restarts:
        return "pass" if chosen else "fail"
    if len(set(passing_restarts)) > 1:
        return "unidentified"
    return "pass" if passing_restarts[0] else "fail"


def grade_k(population: int) -> int:
    """Return the simulation index of a registered grading population (animals at GRADE_BASE)."""
    return population


# ---------------------------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------------------------


def grade_answer(population: int, z: list[float], base: int = GRADE_BASE) -> dict[str, Any]:
    """Grade an answer on one population; return every family's readings and passes.

    The registered populations are at ``GRADE_BASE``; a smoke passes its own discarded base.
    """
    from quantumnematode.evaluation.batched import BatchedEvaluator
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    shared = nr._shared()
    bg = shared["bg"]
    seeds = [base + nr.ANIMALS * grade_k(population) + i for i in range(nr.ANIMALS)]
    populations = {}
    for family, config in shared["configs"].items():
        controller = BatchedMechanistic(nr.parameters_from_unit(np.asarray(z)))
        trajectories = BatchedEvaluator(config, controller, seeds).run()
        assert trajectories.assay is not None  # noqa: S101 - every assay config records its timing
        analysis = tuple(trajectories.assay["analysis_steps"])  # type: ignore[arg-type]
        populations[family] = [
            bg.window(family, trajectories.to_behaviour_steps(i), analysis)
            for i in range(nr.ANIMALS)
        ]
    graded = bg.grade_populations(populations, shared["bands"])
    return nr._finite(
        {
            "population": population,
            "families": {
                family: {
                    name: {"value": r["value"], "z": r["z"], "passes": r["passes"]}
                    for name, r in readings.items()
                }
                for family, readings in graded["families"].items()
            },
            "family_passes": graded["family_passes"],
            "presence": graded["local_search_presence"],
            "D": graded["D"],
            "D_fit": graded["D_fit"],
        },
    )


def _guarded() -> bool:
    return cr.launch_committed((ENCODINGS_PATH,), launch_path=LAUNCH_PATH)


def _histories() -> list[dict[str, Any]]:
    return [json.loads((BUILD / "fits" / f"fit{r}.json").read_text()) for r in range(RESTARTS)]


def _run(workers: int) -> int:
    """Run the 8 restarts, then re-simulate each answer 10 times."""
    scales = nr.band_scales()
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for r in range(RESTARTS):
            cr.run_fit(
                {},
                fit=r,
                cma_seed=CMA_SEED + r,
                seed_base=SEED_BASE,
                cap=CAP,
                use_rule=True,
                state_dir=BUILD / "fits",
                pool=pool,
                scales=scales,
                objective=objective,
            )
    for r, history in enumerate(_histories()):
        cr.simulate_jobs(
            [(cr.answer_k(r, q), history["answers"][-1]) for q in range(RESIMULATIONS)],
            SEED_BASE,
            BUILD / "simulations.jsonl",
            workers,
        )
    return 0


def _grade(workers: int) -> int:
    """Grade every restart's answer on the five registered populations."""
    histories = _histories()
    jobs = [(g, h["answers"][-1]) for h in histories for g in range(GRADE_POPULATIONS)]
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        grades = list(pool.map(grade_answer, [g for g, _ in jobs], [z for _, z in jobs]))
    by_restart = [
        grades[r * GRADE_POPULATIONS : (r + 1) * GRADE_POPULATIONS] for r in range(RESTARTS)
    ]
    (BUILD / "grades.json").write_text(json.dumps(by_restart, indent=2, allow_nan=False) + "\n")
    return 0


def _read() -> int:
    """Choose the restart, read the verdict, and write the fit record."""
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    histories = _histories()
    grades = json.loads((BUILD / "grades.json").read_text())
    stored = nr.load_simulations(BUILD / "simulations.jsonl")
    scales = nr.band_scales()
    runs = {
        r: [nr._finite(stored[cr.answer_k(r, q)]) for q in range(RESIMULATIONS)]
        for r in range(RESTARTS)
    }
    restarts = []
    for r, history in enumerate(histories):
        passes = [population_passes(p["family_passes"]) for p in grades[r]]
        restarts.append(
            {
                "restart": r,
                "cma_seed": CMA_SEED + r,
                "stop": history["stop"],
                "final_objective": cr.final_objective(history["medians"]),
                "answer_D_median": float(np.median([objective(x) for x in runs[r]])),
                "population_passes": passes,
                "passes_wild_type": wild_type_verdict(passes),
                "population_D": [p["D"] for p in grades[r]],
                "population_D_fit": [p["D_fit"] for p in grades[r]],
                "answer": history["answers"][-1],
                "medians": history["medians"],
                "sigmas": history["sigmas"],
            },
        )
    chosen = chosen_restart(histories)
    plateaued = [r for r in range(RESTARTS) if histories[r]["stop"]["by"] == "plateau"]
    pairs = [
        {
            "restarts": [i, j],
            "D": float(
                np.median(
                    [
                        nr.predictive_distance(a, b, scales)
                        for a, b in zip(runs[i], runs[j], strict=True)
                    ],
                ),
            ),
        }
        for i, j in itertools.combinations(plateaued, 2)
    ]
    spread = [pair["D"] for pair in pairs]
    record: dict[str, Any] = {
        "description": DESCRIPTION,
        "chosen_restart": chosen,
        "restarts": restarts,
        "restart_spread_D": {
            "pairs": len(spread),
            "median": float(np.median(spread)) if spread else None,
            "range": [min(spread), max(spread)] if spread else None,
            "each_pair": pairs,
        },
    }
    if chosen is None:
        record["stop"] = "no restart plateaued; the run returns to the maintainer"
        cr._write_json(FIT_PATH, record)
        print(record["stop"], file=sys.stderr)
        return 1
    best = restarts[chosen]
    failing = sorted(
        {
            f"{family}/{name}"
            for population in grades[chosen]
            if not population_passes(population["family_passes"])
            for family, readings in population["families"].items()
            for name, reading in readings.items()
            if not reading["passes"]
        }
        | {
            "local_search/presence"
            for population in grades[chosen]
            if not population["presence"]["present"]
        },
    )
    parameters = nr.parameters_from_unit(np.asarray(best["answer"]))
    record |= {
        "passes_wild_type": best["passes_wild_type"],
        "populations_passing": int(sum(best["population_passes"])),
        "diagnosis": failing,
        "parameters": {
            name: {"value": parameters[name], "recovered_in_F0b": name in RECOVERED}
            for name in MECHANISTIC_PARAMETERS
        },
        "grades": grades[chosen],
    }
    cr._write_json(FIT_PATH, record)
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "chosen_restart",
                    "passes_wild_type",
                    "populations_passing",
                    "diagnosis",
                    "restart_spread_D",
                )
            },
            indent=2,
        ),
    )
    return 0


def _smoke(workers: int) -> int:
    """Run two generations of one restart and grade its answer once, on discarded seeds."""
    state = BUILD / "smoke"
    shutil.rmtree(state, ignore_errors=True)
    scales = nr.band_scales()
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = cr.run_fit(
            {},
            fit=0,
            cma_seed=SMOKE_CMA_SEED,
            seed_base=SMOKE_SEED_BASE,
            cap=2,
            use_rule=True,
            state_dir=state,
            pool=pool,
            scales=scales,
            objective=objective,
        )
    graded = grade_answer(0, history["answers"][-1], base=SMOKE_GRADE_BASE)
    print(json.dumps({"medians": history["medians"], "family_passes": graded["family_passes"]}))
    return 0 if len(history["medians"]) == 2 and graded["D"] is not None else 1


def main(argv: list[str] | None = None) -> int:
    """CLI: the fit, the grading, the reading and a smoke."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("stage", choices=("run", "grade", "read", "smoke"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.stage == "smoke":
        return _smoke(args.workers)
    if not _guarded():
        print("the launch record or the encoding table is uncommitted or changed", file=sys.stderr)
        return 1
    if args.stage == "run":
        return _run(args.workers)
    if args.stage == "grade":
        return _grade(args.workers)
    return _read()


if __name__ == "__main__":
    sys.exit(main())
