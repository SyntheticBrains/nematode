#!/usr/bin/env python
r"""C.1 refitted to the real wild type on benchmark v3, with its 25-parameter model, and its verdict.

**The model** is ``BatchedMechanistic`` with C.1's 20 parameters on F.0a's box, plus an off-food
run speed and curvature noise, a per-state speed spread and an off-food weathervane gain
(:data:`EXTRA`), each on a range set without search data.

**The recovery**, before any fit: sep-CMA recovers the model at two known truths, the 25-parameter
probes' passing answers. Each truth's observation, floor, oracle, prior D and the box centre's D are
read first (``recovery-preflight``); then each truth is fitted twice from the box centre on its
observation, and the better restart's gain fraction is read (``recovery``). The fit runs only if the
median gain over the truths is at least 0.75.

**The fit** is sep-CMA from the box centre, 12 restarts, on the definedness-first ranking score
against v3's band centres at n = 200, each stopping on the doubly guarded plateau or at the cap of
330. Every plateaued restart is graded with presence on 5 selection populations, and the one passing
most is chosen; the verdict reads the chosen restart on 5 registered populations: it passes the wild
type when at least 3 pass.

Subcommands (``recovery-preflight`` and ``recovery`` refuse until Part A of launch record 121 is
committed and unchanged; the rest also need Part B with the recovery's records and a passing
recovery; each binds the mutant encoding table; ``smoke`` is unguarded and discarded)::

    uv run python scripts/analysis/c1_v3_fit.py recovery-preflight
    uv run python scripts/analysis/c1_v3_fit.py recovery
    uv run python scripts/analysis/c1_v3_fit.py run
    uv run python scripts/analysis/c1_v3_fit.py select
    uv run python scripts/analysis/c1_v3_fit.py grade
    uv run python scripts/analysis/c1_v3_fit.py read
    uv run python scripts/analysis/c1_v3_fit.py smoke    # execution check, discarded
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1b_fit as c1b
import cma_recovery as cr
import cma_recovery_v2 as v2
import npe_recovery as nr
import npe_region_check as rc

DESCRIPTION = "C.1 refitted to the wild type on benchmark v3, with its 25-parameter model."
_REPO = Path(__file__).resolve().parents[2]
_LOGBOOKS = _REPO / "docs" / "experiments" / "logbooks" / "supporting"
SUPPORT = _LOGBOOKS / "121-c1-v3-fit"
LAUNCH_PATH = SUPPORT / "launch.md"
PROBES = SUPPORT / "probes"
TRUTH_PATHS = (PROBES / "basin_warm.json", PROBES / "basin_boxA.json")
RECOVERY_PREFLIGHT_PATH = SUPPORT / "recovery_preflight.json"
RECOVERY_PATH = SUPPORT / "recovery.json"
FIT_PATH = SUPPORT / "fit.json"
ENCODINGS_PATH = _LOGBOOKS / "113-c1-fit" / "encodings.md"
BUILD = _REPO / "build" / "c1-v3-fit"

# The five parameters beyond C.1's 20, each uniform on its range: the state speeds' prior, the
# curvature priors', v2's reachability spread box, and the weathervane gain's prior.
EXTRA = {
    "v_off_food": (0.005, 0.20),
    "sigma_off_food": (5.0, 60.0),
    "speed_log_sd_dwell": (0.0, 1.5),
    "speed_log_sd_roam": (0.0, 1.5),
    "kappa_off_food": (-50.0, 50.0),
}
DIMENSION = 20 + len(EXTRA)
# The two states can swap roles; a parameter error is read in the closer of the two orders.
SWAP_PAIRS = (
    ("a0", "b0"),
    ("a_intake", "b_intake"),
    ("a_satiety", "b_satiety"),
    ("v_dwell", "v_roam"),
    ("sigma_dwell", "sigma_roam"),
    ("r_dwell", "r_roam"),
    ("o_dwell", "o_roam"),
    ("speed_log_sd_dwell", "speed_log_sd_roam"),
)

RESTARTS = 12
CMA_SEED = 5769
RECOVERY_CMA_SEED = 5782
PRIOR_SEED = 5781
CAP = 330
FIT_BLOCK = 10_000
ANSWER_OFFSET = 8_000  # above every generation's index under the cap, 16 x 329 + 15 = 5,279
FIT_BASE = 500_000_000
GRADE_BASE = 525_000_000
SELECT_BASE = 526_000_000
RECOVERY_BASE = 530_000_000
TRUTH_BASE = 540_000_000
PRIOR_BASE = 541_000_000
CENTRE_BASE = 542_000_000
SMOKE_CMA_SEED = 5990
SMOKE_BASE = 86_000_000
SMOKE_GRADE_BASE = 86_970_000
GRADE_POPULATIONS = 5
SELECT_POPULATIONS = 5
PASS_NEEDED = 3
RESIMULATIONS = 10
TRUTH_REPEATS = 20  # 1-10 the floor, 11-20 the oracle
TRUTH_STRIDE = 100
PRIOR_DRAWS = 200
RECOVERY_RESTARTS = 2
ORACLE_GATE = 0.85
START_GATE = 0.25
SEPARATION = 1.0
GAIN_GATE = 0.75
ROBERTS_REVERSE_SHARE = 0.173


# ---------------------------------------------------------------------------------------------
# The model and its mutants
# ---------------------------------------------------------------------------------------------


def parameters(z: np.ndarray | list[float]) -> dict[str, float]:
    """Map a point of the 25-parameter unit box to C.1 (v3)'s parameters."""
    z = np.asarray(z, dtype=float)
    if len(z) != DIMENSION:
        msg = f"C.1 (v3) has {DIMENSION} dimensions, not {len(z)}"
        raise ValueError(msg)
    out = nr.parameters_from_unit(z[:20])
    for value, (name, (low, high)) in zip(z[20:], EXTRA.items(), strict=True):
        out[name] = float(low + value * (high - low))
    return out


def mutant_parameters(params: dict[str, float], mutant: str) -> dict[str, float] | None:
    """Return a mutant's parameters under the registered encodings; None where not encodable.

    tph-1 removes switching's dependence on intake; tax-4 removes both weathervane gains and the
    dC/dt responses; pdfr-1 has no counterpart. The kinematic parameters are kept by every mutant.
    """
    if mutant == "tph-1":
        return {**params, "a_intake": 0.0, "b_intake": 0.0}
    if mutant == "tax-4":
        zeroed = ("kappa_weathervane", "kappa_off_food", "beta_reversal", "beta_omega")
        return {**params, **dict.fromkeys(zeroed, 0.0)}
    if mutant == "pdfr-1":
        return None
    msg = f"unknown mutant {mutant!r}"
    raise ValueError(msg)


def unit_errors(answer: list[float], truth: list[float]) -> dict[str, Any]:
    """Return each parameter's unit-box error, in the closer of the two state orders."""
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    names = [*MECHANISTIC_PARAMETERS, *EXTRA]
    a = dict(zip(names, map(float, answer), strict=True))
    t = dict(zip(names, map(float, truth), strict=True))
    swapped = dict(a)
    for first, second in SWAP_PAIRS:
        swapped[first], swapped[second] = a[second], a[first]

    def errors(x: dict[str, float]) -> dict[str, float]:
        return {n: abs(x[n] - t[n]) for n in names}

    plain, other = errors(a), errors(swapped)
    use_swap = sum(other.values()) < sum(plain.values())
    return {"swapped": use_swap, "errors": other if use_swap else plain}


# ---------------------------------------------------------------------------------------------
# Simulation, grading and the objectives
# ---------------------------------------------------------------------------------------------


def _bands() -> dict[str, Any]:
    import benchmark_grade as bg

    return bg.load_bands_v3()


def _scales_and_map(bands: dict[str, Any]) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
    return v2.band_scales_v2(bands), {f: tuple(st) for f, st in bands["families"].items()}


def simulate(k: int, z: list[float], seed_base: int) -> dict[str, Any]:
    """Run simulation k of C.1 (v3) at z on v3's four assays; return its graded statistics."""
    import benchmark_grade as bg
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    bands = _bands()
    seeds = [seed_base + nr.ANIMALS * k + i for i in range(nr.ANIMALS)]
    params = parameters(z)
    start = time.perf_counter()
    populations, chemotaxis = bg.simulate_v2(
        "mlp",
        lambda _path: BatchedMechanistic(params),
        seeds,
        bands,
    )
    graded = bg.grade_populations_v3(populations, chemotaxis, bands, presence=False)
    statistics = {
        f"{family}/{name}": reading["value"]
        for family, readings in graded["families"].items()
        for name, reading in readings.items()
    }
    share = graded["reversal_share"]
    return nr._finite(
        {
            "k": k,
            "z": list(map(float, z)),
            "statistics": statistics,
            "D": graded["D"],
            "D_fit": graded["D_fit"],
            "reversal_share": None if share is None else share["value"],
            "seconds": time.perf_counter() - start,
        },
    )


def grade_answer(k: int, z: list[float], base: int) -> dict[str, Any]:
    """Grade an answer on population k at a base on v3, with presence: every family and the pass."""
    import benchmark_grade as bg
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    bands = _bands()
    seeds = [base + nr.ANIMALS * k + i for i in range(nr.ANIMALS)]
    params = parameters(z)
    populations, chemotaxis = bg.simulate_v2(
        "mlp",
        lambda _path: BatchedMechanistic(params),
        seeds,
        bands,
    )
    graded = bg.grade_populations_v3(populations, chemotaxis, bands, presence=True)
    share = graded["reversal_share"]
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
            "reversal_share": None if share is None else share["value"],
        },
    )


class BandObjective:
    """The ranking score of a simulation against v3's band centres at n = 200, picklable."""

    def __init__(self) -> None:
        import benchmark_grade as bg

        bands = _bands()
        self.scales, self.family_map = _scales_and_map(bands)
        centres = {}
        for family, stats in self.family_map.items():
            family_bands = bg.family_bands_v2(bands, family, nr.ANIMALS)
            for s in stats:
                centres[f"{family}/{s}"] = float(family_bands[s].centre)
        self.truth = {"statistics": centres}

    def __call__(self, record: dict[str, Any]) -> float:
        """Return the ranking score of one simulation's record against the bands."""
        return v2.ranking_score(self.truth, record, self.scales, self.family_map)


class TruthObjective:
    """The ranking score of a simulation against a truth's observation, on v3's scales; picklable."""

    def __init__(self, observation: dict[str, Any]) -> None:
        self.observation = observation
        self.scales, self.family_map = _scales_and_map(_bands())

    def __call__(self, record: dict[str, Any]) -> float:
        """Return the ranking score of one simulation's record against the observation."""
        return v2.ranking_score(self.observation, record, self.scales, self.family_map)


def capped_d(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Return the registered, capped D between two records on v3's scales and family map."""
    scales, family_map = _scales_and_map(_bands())
    return v2.predictive_distance_v2(a, b, scales, family_map, capped=True)


def start_centre(seed: int) -> Any:  # noqa: ANN401 - a cma strategy
    """Return sep-CMA at the 25-parameter box's centre, sigma0 0.2."""
    import cma

    options = {
        "CMA_diagonal": True,
        "bounds": [0.0, 1.0],
        "popsize": cr.POPSIZE,
        "seed": seed,
        "verbose": -9,
    }
    return cma.CMAEvolutionStrategy([cr.X0] * DIMENSION, cr.SIGMA0, options)


def answer_k(fit: int, repeat: int) -> int:
    """Return the simulation index of a fit's answer, re-simulated: above every generation's."""
    return FIT_BLOCK * fit + ANSWER_OFFSET + repeat


def truth_k(truth: int, repeat: int) -> int:
    """Return the simulation index of a truth's observation (0) or re-simulation (1-20)."""
    return TRUTH_STRIDE * truth + repeat


def select_k(restart: int, population: int) -> int:
    """Return the simulation index of a restart's selection population."""
    return SELECT_POPULATIONS * restart + population


def run_jobs(
    jobs: list[tuple[int, list[float]]],
    seed_base: int,
    path: Path,
    workers: int,
) -> dict[int, dict[str, Any]]:
    """Run each (k, z) not already in the store at a seed base; return every record by k."""
    stored = v2.read_records(path)
    todo = [(k, z) for k, z in jobs if k not in stored]
    if todo:
        v2._drop_torn_line(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with (
            ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool,
            path.open("a") as out,
        ):
            for record in pool.map(
                simulate,
                [k for k, _ in todo],
                [z for _, z in todo],
                [seed_base] * len(todo),
            ):
                out.write(json.dumps(record, allow_nan=False) + "\n")
                out.flush()
                stored[record["k"]] = record
    return stored


def _grade_all(jobs: list[tuple[int, list[float], int]], workers: int) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        return list(pool.map(grade_answer, *zip(*jobs, strict=True)))


# ---------------------------------------------------------------------------------------------
# The recovery
# ---------------------------------------------------------------------------------------------


def truths() -> list[list[float]]:
    """Return the two truths: the 25-parameter probes' passing answers, warm then box A."""
    return [json.loads(p.read_text())["answer"] for p in TRUTH_PATHS]


def prior_points() -> np.ndarray:
    """Return the 200 registered prior draws of the 25-parameter unit box."""
    return np.random.default_rng(PRIOR_SEED).random((PRIOR_DRAWS, DIMENSION))


def gain(prior: float, d: float, floor: float) -> float:
    """Return the gain fraction, (prior D - D) / (prior D - floor)."""
    return rc.gain_fraction(prior, d, floor)


def preflight_record(
    observations: list[dict[str, Any]],
    repeats: list[list[dict[str, Any]]],
    prior_runs: list[dict[str, Any]],
    centre_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the recovery's pre-fit checks from each truth's observation and re-simulations.

    ``repeats[t]`` holds truth t's re-simulations 1-20, in order: 1-10 make the floor, 11-20 the
    oracle. Every D is the capped v3 D.
    """
    rows = []
    for observation, runs in zip(observations, repeats, strict=True):
        floor = float(np.median([capped_d(observation, r) for r in runs[:10]]))
        oracle = float(np.median([capped_d(observation, r) for r in runs[10:20]]))
        prior = float(np.median([capped_d(observation, r) for r in prior_runs]))
        centre = float(np.median([capped_d(observation, r) for r in centre_runs]))
        rows.append(
            {
                "floor": floor,
                "oracle_D": oracle,
                "prior_D": prior,
                "centre_D": centre,
                "separated": prior - floor >= SEPARATION,
                "oracle_gain": gain(prior, oracle, floor),
                "centre_gain": gain(prior, centre, floor),
            },
        )
    oracle_gain = float(np.median([r["oracle_gain"] for r in rows]))
    centre_gain = float(np.median([r["centre_gain"] for r in rows]))
    return {
        "truths": rows,
        "oracle_median_gain": oracle_gain,
        "centre_median_gain": centre_gain,
        "passed": bool(
            all(r["separated"] for r in rows)
            and oracle_gain >= ORACLE_GATE
            and centre_gain <= START_GATE,
        ),
    }


def recovery_gate(gains: list[float]) -> dict[str, Any]:
    """Return the recovery's gate: the median best-restart gain over the truths, at least 0.75."""
    median = float(np.median(gains))
    return {"gains": gains, "median_gain": median, "passes": median >= GAIN_GATE}


def _recovery_preflight(workers: int) -> int:
    if RECOVERY_PREFLIGHT_PATH.exists():
        print(f"{RECOVERY_PREFLIGHT_PATH} exists: the preflight is read once", file=sys.stderr)
        return 1
    zs = truths()
    truth_jobs = [(truth_k(t, q), z) for t, z in enumerate(zs) for q in range(TRUTH_REPEATS + 1)]
    truth_runs = run_jobs(truth_jobs, TRUTH_BASE, BUILD / "truths.jsonl", workers)
    prior = prior_points()
    prior_runs = run_jobs(
        [(j, list(map(float, prior[j]))) for j in range(PRIOR_DRAWS)],
        PRIOR_BASE,
        BUILD / "prior.jsonl",
        workers,
    )
    centre_runs = run_jobs(
        [(q, [cr.X0] * DIMENSION) for q in range(RESIMULATIONS)],
        CENTRE_BASE,
        BUILD / "centre.jsonl",
        workers,
    )
    record = {
        "description": "C.1 (v3)'s recovery: each truth's observation, floor, oracle, prior D and "
        "the box centre's D, with the pre-fit checks.",
        **preflight_record(
            [truth_runs[truth_k(t, 0)] for t in range(len(zs))],
            [
                [truth_runs[truth_k(t, q)] for q in range(1, TRUTH_REPEATS + 1)]
                for t in range(len(zs))
            ],
            [prior_runs[j] for j in range(PRIOR_DRAWS)],
            [centre_runs[q] for q in range(RESIMULATIONS)],
        ),
    }
    cr._write_json(RECOVERY_PREFLIGHT_PATH, record)
    print(json.dumps(nr._finite(record), indent=2))
    return 0 if record["passed"] else 1


def _recovery(workers: int) -> int:
    if (
        not RECOVERY_PREFLIGHT_PATH.exists()
        or not json.loads(
            RECOVERY_PREFLIGHT_PATH.read_text(),
        )["passed"]
    ):
        print("the recovery's preflight has not passed; no recovery fit runs", file=sys.stderr)
        return 1
    preflight = json.loads(RECOVERY_PREFLIGHT_PATH.read_text())
    zs = truths()
    truth_runs = v2.read_records(BUILD / "truths.jsonl")
    observations = [truth_runs[truth_k(t, 0)] for t in range(len(zs))]
    histories: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for t, observation in enumerate(observations):
            for j in range(RECOVERY_RESTARTS):
                fit = RECOVERY_RESTARTS * t + j
                histories[fit] = cr.run_fit(
                    observation,
                    fit=fit,
                    cma_seed=RECOVERY_CMA_SEED + fit,
                    seed_base=RECOVERY_BASE,
                    cap=CAP,
                    use_rule=True,
                    state_dir=BUILD / "recovery",
                    pool=pool,
                    scales={},
                    objective=TruthObjective(observation),
                    simulate=simulate,
                    stop=c1b.defined_stop,
                    strategy_factory=start_centre,
                )
    answers = run_jobs(
        [
            (answer_k(fit, q), h["answers"][-1])
            for fit, h in histories.items()
            for q in range(RESIMULATIONS)
        ],
        RECOVERY_BASE,
        BUILD / "recovery_answers.jsonl",
        workers,
    )
    rows, best_gains = [], []
    for t, observation in enumerate(observations):
        row = preflight["truths"][t]
        fits = [RECOVERY_RESTARTS * t + j for j in range(RECOVERY_RESTARTS)]
        best = fits[cr.best_restart([histories[f]["medians"] for f in fits])]
        restarts = []
        for fit in fits:
            d = float(
                np.median(
                    [
                        capped_d(observation, answers[answer_k(fit, q)])
                        for q in range(RESIMULATIONS)
                    ],
                ),
            )
            restarts.append(
                {
                    "fit": fit,
                    "cma_seed": RECOVERY_CMA_SEED + fit,
                    "stop": histories[fit]["stop"],
                    "final_objective": cr.final_objective(histories[fit]["medians"]),
                    "D": d,
                    "gain": gain(row["prior_D"], d, row["floor"]),
                    "best": fit == best,
                    "parameter_errors": unit_errors(histories[fit]["answers"][-1], zs[t]),
                },
            )
        best_gains.append(next(r["gain"] for r in restarts if r["best"]))
        rows.append({"truth": t, "source": TRUTH_PATHS[t].name, "restarts": restarts})
    record = {
        "description": "C.1 (v3)'s reduced recovery: 2 truths x 2 restarts, the gate on the "
        "median best-restart gain.",
        "truths": rows,
        **recovery_gate(best_gains),
    }
    cr._write_json(RECOVERY_PATH, record)
    print(json.dumps(nr._finite({k: record[k] for k in ("gains", "median_gain", "passes")})))
    return 0 if record["passes"] else 1


# ---------------------------------------------------------------------------------------------
# The fit, selection, grading and the verdict
# ---------------------------------------------------------------------------------------------


def _histories() -> list[dict[str, Any]]:
    return [json.loads((BUILD / "fits" / f"fit{r}.json").read_text()) for r in range(RESTARTS)]


def _run(workers: int) -> int:
    """Run the 12 restarts to the cap, then re-simulate each answer 10 times."""
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for r in range(RESTARTS):
            cr.run_fit(
                {},
                fit=r,
                cma_seed=CMA_SEED + r,
                seed_base=FIT_BASE,
                cap=CAP,
                use_rule=True,
                state_dir=BUILD / "fits",
                pool=pool,
                scales={},
                objective=BandObjective(),
                simulate=simulate,
                stop=c1b.defined_stop,
                strategy_factory=start_centre,
            )
    run_jobs(
        [
            (answer_k(r, q), h["answers"][-1])
            for r, h in enumerate(_histories())
            for q in range(RESIMULATIONS)
        ],
        FIT_BASE,
        BUILD / "answers.jsonl",
        workers,
    )
    return 0


def _select(workers: int) -> int:
    """Grade every plateaued restart on its 5 selection populations, with presence."""
    histories = _histories()
    plateaued = c1b._plateaued(histories)
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
    pairs = [
        {
            "restarts": [i, j],
            "D": float(
                np.median([capped_d(a, b) for a, b in zip(runs[i], runs[j], strict=True)]),
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
    histories = _histories()
    plateaued = c1b._plateaued(histories)
    selection = {int(r): g for r, g in json.loads((BUILD / "selection.json").read_text()).items()}
    grades = json.loads((BUILD / "grades.json").read_text())
    stored = v2.read_records(BUILD / "answers.jsonl")
    runs = {r: [stored[answer_k(r, q)] for q in range(RESIMULATIONS)] for r in range(RESTARTS)}
    objective = BandObjective()
    restarts = [
        {
            "restart": r,
            "cma_seed": CMA_SEED + r,
            "stop": h["stop"],
            "final_score": cr.final_objective(h["medians"]),
            "answer_D_median": c1b._median_d(runs[r]),
            "answer_score_median": float(np.median([objective(x) for x in runs[r]])),
            "selection_passing": (
                sum(g["wild_type_passes"] for g in selection[r]) if r in selection else None
            ),
            "population_passes": [g["wild_type_passes"] for g in grades[r]],
            "passes_wild_type": c1b.wild_type_verdict(grades[r]),
            "population_D": [g["D"] for g in grades[r]],
            "reversal_share": [g["reversal_share"] for g in grades[r]],
            "answer": h["answers"][-1],
            "medians": h["medians"],
            "sigmas": h["sigmas"],
        }
        for r, h in enumerate(histories)
    ]
    chosen = c1b.chosen_restart({r: selection[r] for r in plateaued})
    record: dict[str, Any] = {
        "description": DESCRIPTION,
        "chosen_restart": chosen,
        "restarts": restarts,
        "restart_spread_D": _spread(plateaued, runs),
        "reversal_share_reference": ROBERTS_REVERSE_SHARE,
    }
    if chosen is None:
        record["stop"] = "no restart plateaued; the run returns to the maintainer"
        cr._write_json(FIT_PATH, record)
        print(record["stop"], file=sys.stderr)
        return 1
    record |= {
        "passes_wild_type": c1b.wild_type_verdict(grades[chosen]),
        "populations_passing": int(sum(g["wild_type_passes"] for g in grades[chosen])),
        "diagnosis": c1b.failing(grades[chosen]),
        "parameters": parameters(histories[chosen]["answers"][-1]),
        "grades": grades[chosen],
    }
    cr._write_json(FIT_PATH, record)
    keys = ("chosen_restart", "passes_wild_type", "populations_passing", "diagnosis")
    print(json.dumps({k: record[k] for k in keys}, indent=2))
    return 0


# ---------------------------------------------------------------------------------------------
# The smoke and the guards
# ---------------------------------------------------------------------------------------------


def _smoke(workers: int) -> int:
    """Two generations of a band fit and of a truth fit, a preflight's arithmetic, one grade.

    Every seed is discarded: the band fit at 86,000,000, the truth fit at 86,500,000, the truth's
    and prior's simulations at 86,600,000, and the grade at 86,970,000.
    """
    state = BUILD / "smoke"
    shutil.rmtree(state, ignore_errors=True)
    truth = truths()[0]
    store = state / "sims.jsonl"
    sims = run_jobs(
        [(q, truth) for q in range(3)] + [(100, [cr.X0] * DIMENSION)],
        SMOKE_BASE + 600_000,
        store,
        workers,
    )
    pre = preflight_record([sims[0]], [[sims[1], sims[2]] * 10], [sims[100]], [sims[100]])
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        band = cr.run_fit(
            {},
            fit=0,
            cma_seed=SMOKE_CMA_SEED,
            seed_base=SMOKE_BASE,
            cap=2,
            use_rule=True,
            state_dir=state / "band",
            pool=pool,
            scales={},
            objective=BandObjective(),
            simulate=simulate,
            stop=c1b.defined_stop,
            strategy_factory=start_centre,
        )
        recovered = cr.run_fit(
            sims[0],
            fit=0,
            cma_seed=SMOKE_CMA_SEED,
            seed_base=SMOKE_BASE + 500_000,
            cap=2,
            use_rule=True,
            state_dir=state / "truth",
            pool=pool,
            scales={},
            objective=TruthObjective(sims[0]),
            simulate=simulate,
            stop=c1b.defined_stop,
            strategy_factory=start_centre,
        )
    graded = grade_answer(0, band["answers"][-1], SMOKE_GRADE_BASE)
    print(
        json.dumps(
            nr._finite(
                {
                    "band_medians": band["medians"],
                    "truth_medians": recovered["medians"],
                    "preflight_floor": pre["truths"][0]["floor"],
                    "wild_type_passes": graded["wild_type_passes"],
                    "D": graded["D"],
                },
            ),
        ),
    )
    ok = len(band["medians"]) == len(recovered["medians"]) == 2 and graded["D"] is not None
    return 0 if ok else 1


def _guarded(stage: str) -> bool:
    bound: tuple[Path, ...] = (ENCODINGS_PATH, *TRUTH_PATHS)
    if stage in ("recovery-preflight", "recovery"):
        return cr.launch_committed(bound, launch_path=LAUNCH_PATH)
    bound = (*bound, RECOVERY_PREFLIGHT_PATH, RECOVERY_PATH)
    has_part_b = LAUNCH_PATH.exists() and "\n## Part B" in LAUNCH_PATH.read_text()
    if not (has_part_b and cr.launch_committed(bound, launch_path=LAUNCH_PATH)):
        return False
    return bool(json.loads(RECOVERY_PATH.read_text())["passes"])


def main(argv: list[str] | None = None) -> int:
    """CLI: the recovery, the fit, selection, grading, the reading and a smoke."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "stage",
        choices=("recovery-preflight", "recovery", "run", "select", "grade", "read", "smoke"),
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.stage == "smoke":
        return _smoke(args.workers)
    if not _guarded(args.stage):
        print(
            "the launch record, its bound records, the encoding table or the recovery's pass is "
            "missing, uncommitted or changed",
            file=sys.stderr,
        )
        return 1
    stages = {
        "recovery-preflight": _recovery_preflight,
        "recovery": _recovery,
        "run": _run,
        "select": _select,
        "grade": _grade,
        "read": lambda _: _read(),
    }
    return stages[args.stage](args.workers)


if __name__ == "__main__":
    sys.exit(main())
