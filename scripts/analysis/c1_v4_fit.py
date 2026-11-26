#!/usr/bin/env python
r"""C.1 refitted on benchmark v4 from C.1 (v3)'s answer, with its encodable mutants, and its reads.

**The model** is ``BatchedMechanistic`` with C.1 (v3)'s 25 parameters on their boxes plus
``r_dwell_on_food``, a dwelling animal's reversal rate on food, on ``r_dwell``'s prior
(:data:`ON_FOOD`). It runs on the speed-following arm, where an on-food reversal backs at a speed
that follows the worm's recent run speed.

**The start** is C.1 (v3)'s registered answer, read from its fit record, with the new coordinate at
its box's centre; sep-CMA runs from it at sigma0 0.1.

**The objective** is the definedness-first ranking score against v4's band centres at n = 200,
with tph-1's and tax-4's small-lawn differences, simulated on the same animals, together half the
objective. A restart stops on the plateau once sigma is at most sigma0, every median in its last
40 generations is below 100, or at the cap.

**The stages**, in order:

- ``pilot``: one restart on discarded seeds, graded on 5 populations; its go rule needs the wild
  type, tph-1 and tax-4 each in 3 of 5. It also checks the recovery's preflight on its own answer
  on discarded seeds, projects the registered compute against a 48-h bar, and sets the cap.
- ``recovery-preflight`` and ``recovery``: the refit's positive control. The pilot's answer is the
  truth, fitted twice from the registered start; the gain is measured from the start, and the fit
  runs only if the better restart's gain is at least 0.75.
- ``run``, ``select``, ``grade`` and ``read``: 12 restarts, selection on 5 separate populations
  per plateaued restart, every restart graded on 15 registered populations, and the verdict on the
  chosen restart's first 5: the wild type, tph-1 and tax-4 each in at least 3. ``read`` adds each
  mutant's row under the encoding table's ``unidentified`` rule and the held-out reading on
  Scheer's held-out recording dates, both as description.
- ``loo``: one restart with tax-4 out of the objective, graded on 15 populations, as description.

Every stage but ``smoke`` refuses unless its part of the launch record is committed, its bound
records are unchanged and the fit's code is Part A's commit's, and none overwrites its own record::

    uv run python scripts/analysis/c1_v4_fit.py pilot
    uv run python scripts/analysis/c1_v4_fit.py recovery-preflight
    uv run python scripts/analysis/c1_v4_fit.py recovery
    uv run python scripts/analysis/c1_v4_fit.py run
    uv run python scripts/analysis/c1_v4_fit.py select
    uv run python scripts/analysis/c1_v4_fit.py grade
    uv run python scripts/analysis/c1_v4_fit.py read
    uv run python scripts/analysis/c1_v4_fit.py loo
    uv run python scripts/analysis/c1_v4_fit.py smoke    # execution check, discarded
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_grade as bg
import c1_v3_fit as c1v3
import cma_recovery as cr
import cma_recovery_v2 as v2
import npe_recovery as nr
import npe_region_check as rc

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

DESCRIPTION = "C.1 refitted on benchmark v4 from C.1 (v3)'s answer, with its encodable mutants."
_REPO = Path(__file__).resolve().parents[2]
_LOGBOOKS = _REPO / "docs" / "experiments" / "logbooks" / "supporting"
SUPPORT = _LOGBOOKS / "136-c1-v4-fit"
LAUNCH_PATH = SUPPORT / "launch.md"
START_PATH = _LOGBOOKS / "121-c1-v3-fit" / "fit.json"
PROBE_TRUTH_PATH = _LOGBOOKS / "125-c1-v4-fit" / "probes" / "mechanism_ondwell.json"
FOLLOWING_CONFIGS: tuple[Path, ...] = tuple(
    _REPO
    / "configs"
    / "scenarios"
    / "foraging"
    / f"{bg.ARMS['mlp_following']}_assay_{protocol}.yml"
    for protocol in (
        "chemotaxis_chen",
        "food_removal_calhoun",
        "food_removal_lopezcruz",
        "scheer_lawn",
        "uniform_lawn_flavell",
        "uniform_lawn_scheer",
    )
)
BANDS_PATH = _REPO / "data" / "benchmark" / "bands_v4.json"
PRIMITIVES_PATH = _REPO / "data" / "locomotion" / "primitives.json"
ENCODINGS_PATH = c1v3.ENCODINGS_PATH
PILOT_PATH = SUPPORT / "pilot.json"
RECOVERY_PREFLIGHT_PATH = SUPPORT / "recovery_preflight.json"
RECOVERY_PATH = SUPPORT / "recovery.json"
FIT_PATH = SUPPORT / "fit.json"
LOO_PATH = SUPPORT / "loo.json"
BUILD = _REPO / "build" / "c1-v4-fit"
ARM = "mlp_following"

# The added parameter: its name and log box, r_dwell's prior.
ON_FOOD = ("r_dwell_on_food", 1e-3, 0.5)
DIMENSION = c1v3.DIMENSION + 1
NEW_X0 = 0.5
SIGMA0 = 0.1
ANIMALS = nr.ANIMALS
MUTANTS: tuple[str, ...] = ("tph-1", "tax-4")
LOO_MUTANTS: tuple[str, ...] = ("tph-1",)
# The encoding table's graded statistics; the other lawn difference is recorded and reported.
GRADED: dict[str, tuple[str, ...]] = {
    "tph-1": ("roaming_fraction",),
    "tax-4": ("roaming_fraction", "exits_per_on_lawn_hour"),
}
LAWN_DIFFERENCES = ("roaming_fraction", "exits_per_on_lawn_hour")

# The stop: the plateau, sigma at most sigma0, and every median of the last 40 below this.
TIGHT = 100.0
PILOT_CAP = 400
CAP_FLOOR = 300
CAP_CEILING = 500
CAP_STEP = 10
CAP_FACTOR = 1.5
# The pilot's cost bar: the projected registered compute, in hours.
COST_BAR_HOURS = 48.0

RESTARTS = 12
RECOVERY_RESTARTS = 2
CMA_SEED = 9451
RECOVERY_CMA_SEED = 9463  # 9463-9464; 9465 and 9466 unused
PILOT_CMA_SEED = 9467
LOO_CMA_SEED = 9468
SMOKE_CMA_SEED = 9469
PRIOR_SEED = 9470

FIT_BLOCK = cr.FIT_BLOCK
ANSWER_OFFSET = 8_000  # above every generation's index under the ceiling, 16 x 499 + 15 = 7,999
FIT_BASE = 650_000_000
GRADE_BASE = 675_000_000
SELECT_BASE = 676_000_000
LOO_GRADE_BASE = 677_000_000
RECOVERY_BASE = 680_000_000
TRUTH_BASE = 690_000_000
PRIOR_BASE = 691_000_000
START_BASE = 692_000_000
LOO_BASE = 694_000_000
PILOT_BASE = 696_000_000
PILOT_GRADE_BASE = 698_000_000
PILOT_TRUTH_BASE = 698_100_000
PILOT_START_BASE = 698_200_000
SMOKE_BASE = 699_000_000
PROBE_BASE = 647_000_000  # the separation probe before Part A, discarded

GRADE_POPULATIONS = 15
VERDICT_POPULATIONS = 5
SELECT_POPULATIONS = 5
PILOT_POPULATIONS = 5
PASS_NEEDED = 3
RESIMULATIONS = 10
TRUTH_REPEATS = 20  # 1-10 the floor, 11-20 the oracle
TRUTH_STRIDE = 100
PRIOR_DRAWS = 200
TRUTHS = 1
ORACLE_GATE = 0.85
SEPARATION = 1.0
GAIN_GATE = 0.75
ROBERTS_REVERSE_SHARE = 0.173
# The recovery reads every check and its gate on both distances.
DISTANCE_NAMES: tuple[str, ...] = ("combined", "wild_type")
# The heading, a line of its own, that an amended Part A carries to run the fit despite a failed
# recovery gate; a mention of it inside a sentence does not count.
DECIDED_MARKER = "### Decision: run despite the recovery's gate"


# ---------------------------------------------------------------------------------------------
# The model, its mutants and its start
# ---------------------------------------------------------------------------------------------


def parameters(z: Sequence[float]) -> dict[str, float]:
    """Map a point of the 26-parameter unit box to C.1 (v4)'s parameters."""
    point = np.asarray(z, dtype=float)
    if len(point) != DIMENSION:
        msg = f"C.1 (v4) has {DIMENSION} dimensions, not {len(point)}"
        raise ValueError(msg)
    name, low, high = ON_FOOD
    out = c1v3.parameters(point[: c1v3.DIMENSION])
    out[name] = float(math.exp(math.log(low) + point[c1v3.DIMENSION] * math.log(high / low)))
    return out


def mutant_parameters(params: dict[str, float], mutant: str) -> dict[str, float] | None:
    """Return a mutant's parameters under C.1 (v3)'s registered encodings.

    Every mutant keeps the on-food dwelling rate, a locomotor rate that no encoding reads.
    """
    return c1v3.mutant_parameters(params, mutant)


def start_point() -> list[float]:
    """Return the registered start: C.1 (v3)'s chosen answer, the new coordinate at 0.5."""
    record = json.loads(START_PATH.read_text())
    answer = record["restarts"][record["chosen_restart"]]["answer"]
    if len(answer) != c1v3.DIMENSION:
        msg = f"{START_PATH}: the chosen answer has {len(answer)} coordinates"
        raise ValueError(msg)
    return [*map(float, answer), NEW_X0]


def start_warm(seed: int) -> Any:  # noqa: ANN401 - a cma strategy
    """Return sep-CMA at the registered start, sigma0 0.1."""
    import cma

    options = {
        "CMA_diagonal": True,
        "bounds": [0.0, 1.0],
        "popsize": cr.POPSIZE,
        "seed": seed,
        "verbose": -9,
    }
    return cma.CMAEvolutionStrategy(start_point(), SIGMA0, options)


def probe_truth() -> list[float]:
    """Return the earlier probe's answer, on this model's 26 coordinates."""
    answer = json.loads(PROBE_TRUTH_PATH.read_text())["answer"]
    if len(answer) != DIMENSION:
        msg = f"{PROBE_TRUTH_PATH}: {len(answer)} coordinates, not {DIMENSION}"
        raise ValueError(msg)
    return [float(v) for v in answer]


def unit_errors(answer: Sequence[float], truth: Sequence[float]) -> dict[str, Any]:
    """Return each parameter's unit-box error, the 25 in the closer of the two state orders."""
    errors = c1v3.unit_errors(list(answer[: c1v3.DIMENSION]), list(truth[: c1v3.DIMENSION]))
    errors["errors"][ON_FOOD[0]] = abs(float(answer[-1]) - float(truth[-1]))
    return errors


# ---------------------------------------------------------------------------------------------
# Simulation and grading
# ---------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _bands() -> dict[str, Any]:
    """Return v4's bands, read once per process; callers do not modify them."""
    return bg.load_bands_v4()


def _difference(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return None
    return float(a - b)


def _controller(params: dict[str, float]) -> Callable[[Path], Any]:
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    return lambda _path: BatchedMechanistic(params)


def evaluate(
    z: Sequence[float],
    seeds: list[int],
    *,
    presence: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Simulate a point on these animals: v4's grading of the wild type, and each mutant's Δ.

    Each mutant runs the small lawn on the same animals, and its differences from that wild type
    are recorded on both lawn statistics.
    """
    bands = _bands()
    params = parameters(z)
    populations, chemotaxis = bg.simulate_v2(ARM, _controller(params), seeds, bands)
    graded = bg.grade_populations_v4(populations, chemotaxis, bands, presence=presence)
    clf = bg.classifier(bands)
    wild = bg.family_values("small_lawn", populations["small_lawn"], clf)
    delta: dict[str, dict[str, float | None]] = {}
    for mutant in MUTANTS:
        mutated = mutant_parameters(params, mutant)
        if mutated is None:
            msg = f"{mutant} has no encoding in C.1's table"
            raise ValueError(msg)
        mutant_populations, _ = bg.simulate_v2(
            ARM,
            _controller(mutated),
            seeds,
            bands,
            assays=("small_lawn",),
        )
        values = bg.family_values("small_lawn", mutant_populations["small_lawn"], clf)
        delta[mutant] = {s: _difference(values.get(s), wild.get(s)) for s in LAWN_DIFFERENCES}
    return graded, delta


def _statistics(graded: dict[str, Any]) -> dict[str, float | None]:
    return {
        f"{family}/{name}": reading["value"]
        for family, readings in graded["families"].items()
        for name, reading in readings.items()
    }


def animal_seeds(base: int, k: int) -> list[int]:
    """Return simulation k's 200 animal seeds at a base: base + 200·k + i."""
    return [base + ANIMALS * k + i for i in range(ANIMALS)]


def simulate(k: int, z: list[float], seed_base: int) -> dict[str, Any]:
    """Run simulation k at z: the wild type's graded statistics without presence, and each Δ."""
    seeds = animal_seeds(seed_base, k)
    start = time.perf_counter()
    graded, delta = evaluate(z, seeds, presence=False)
    share = graded["reversal_share"]
    return nr._finite(
        {
            "k": k,
            "z": list(map(float, z)),
            "statistics": _statistics(graded),
            "delta": delta,
            "D": graded["D"],
            "D_fit": graded["D_fit"],
            "reversal_share": None if share is None else share["value"],
            "seconds": time.perf_counter() - start,
        },
    )


def mutant_reading(
    mutant: str,
    delta: dict[str, float | None],
    bands: dict[str, Any],
) -> dict[str, Any]:
    """Grade a mutant's differences on the encoding table's statistics, the other one reported."""
    from quantumnematode.benchmark import grading as g

    mutant_bands = bg.mutant_bands(bands, mutant, ANIMALS)
    values = {s: math.nan if (d := delta.get(s)) is None else float(d) for s in LAWN_DIFFERENCES}
    graded = g.grade(
        {s: values[s] for s in GRADED[mutant]},
        {s: mutant_bands[s] for s in GRADED[mutant]},
        float(bands["bars"][mutant]),
        directions=True,
    )
    statistics = {}
    for s in LAWN_DIFFERENCES:
        statistics[s] = {
            "delta": delta.get(s),
            "z": g.z_score(values[s], mutant_bands[s]),
            "graded": s in GRADED[mutant],
            "passes": graded[s].passes if s in graded else None,
        }
    return {"statistics": statistics, "passes": all(r.passes for r in graded.values())}


def grade_answer(k: int, z: list[float], base: int) -> dict[str, Any]:
    """Grade an answer on population k at a base, with presence: every family and each mutant."""
    bands = _bands()
    seeds = animal_seeds(base, k)
    graded, delta = evaluate(z, seeds, presence=True)
    share = graded["reversal_share"]
    keyed = {
        f"{family}/{name}": reading
        for family, readings in graded["families"].items()
        for name, reading in readings.items()
    }
    return nr._finite(
        {
            "k": k,
            "base": base,
            "statistics": {key: r["value"] for key, r in keyed.items()},
            "z": {key: r["z"] for key, r in keyed.items()},
            "passes": {key: r["passes"] for key, r in keyed.items()},
            "family_passes": graded["family_passes"],
            "local_search_presence": graded.get("local_search_presence"),
            "chemotaxis_presence": graded.get("chemotaxis_presence"),
            "wild_type_passes": bool(graded["wild_type_passes"]),
            "D": graded["D"],
            "D_fit": graded["D_fit"],
            "reversal_share": None if share is None else share["value"],
            "delta": delta,
            "mutants": {m: mutant_reading(m, delta[m], bands) for m in MUTANTS},
        },
    )


# ---------------------------------------------------------------------------------------------
# The objectives and the recovery's distance
# ---------------------------------------------------------------------------------------------


def band_truth(bands: dict[str, Any]) -> dict[str, Any]:
    """Return the bands' centres as a truth: every wild-type statistic and each mutant's Δ."""
    statistics = {}
    for family, stats in bands["families"].items():
        family_bands = bg.family_bands_v2(bands, family, ANIMALS)
        for s in stats:
            statistics[f"{family}/{s}"] = float(family_bands[s].centre)
    delta = {
        m: {s: float(band.centre) for s, band in bg.mutant_bands(bands, m, ANIMALS).items()}
        for m in MUTANTS
    }
    return {"statistics": statistics, "delta": delta}


def mutant_scales(bands: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return each mutant's difference bands' scales at n = 200."""
    return {
        m: {s: float(band.scale) for s, band in bg.mutant_bands(bands, m, ANIMALS).items()}
        for m in MUTANTS
    }


def mutant_z(truth: dict[str, Any], record: dict[str, Any], scale: float, m: str, s: str) -> float:
    """Return a mutant difference's |z| against the truth's, capped at 10; undefined at the cap."""
    from quantumnematode.benchmark import grading as g

    value = record.get("delta", {}).get(m, {}).get(s)
    centre = truth["delta"][m].get(s)
    if value is None or centre is None:
        return g.Z_CAP
    return abs(g.z_score(float(value), g.Band(float(centre), scale)))


def combine(wild: float, terms: Sequence[float], families: int) -> float:
    """Return the families' distance and the mutants' terms combined, the mutants at half."""
    if not terms:
        return wild
    w = families / len(terms)
    return math.sqrt(
        (families * wild**2 + w * sum(t**2 for t in terms)) / (families + w * len(terms)),
    )


class Objective:
    """The ranking score against a truth, with the fitted mutants weighed in at half; picklable."""

    def __init__(
        self,
        truth: dict[str, Any],
        bands: dict[str, Any],
        mutants: tuple[str, ...] = MUTANTS,
    ) -> None:
        self.truth = truth
        self.mutants = mutants
        self.scales = v2.band_scales_v2(bands)
        self.mscales = mutant_scales(bands)
        self.family_map = {f: tuple(s) for f, s in bands["families"].items()}

    def __call__(self, record: dict[str, Any]) -> float:
        """Return one record's score: the undefined count first, then the combined distance."""
        score = v2.ranking_score(self.truth, record, self.scales, self.family_map)
        if not self.mutants:
            return score
        undefined = math.floor(score / v2.UNDEFINED_RANK)
        dc = score - v2.UNDEFINED_RANK * undefined
        terms = []
        for m in self.mutants:
            zs = [mutant_z(self.truth, record, self.mscales[m][s], m, s) for s in GRADED[m]]
            compressed = [v2.COMPRESSION * math.asinh(z / v2.COMPRESSION) for z in zs]
            terms.append(math.sqrt(float(np.mean(np.square(compressed)))))
        return v2.UNDEFINED_RANK * undefined + combine(dc, terms, len(self.family_map))


def band_objective(mutants: tuple[str, ...] = MUTANTS) -> Objective:
    """Return the fit's objective: the score against v4's band centres at n = 200."""
    bands = _bands()
    return Objective(band_truth(bands), bands, mutants)


def truth_objective(observation: dict[str, Any]) -> Objective:
    """Return a recovery fit's objective: the score against a truth's observation."""
    return Objective(observation, _bands())


@functools.lru_cache(maxsize=1)
def _distance_parts() -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
    bands = _bands()
    family_map = {f: tuple(s) for f, s in bands["families"].items()}
    return v2.band_scales_v2(bands), mutant_scales(bands), family_map


def recovery_distance(truth: dict[str, Any], record: dict[str, Any]) -> float:
    """Return the capped D on the six families, the mutants in at half on their capped z."""
    scales, mscales, family_map = _distance_parts()
    wild = v2.predictive_distance_v2(truth, record, scales, family_map, capped=True)
    terms = [
        math.sqrt(
            float(np.mean([mutant_z(truth, record, mscales[m][s], m, s) ** 2 for s in GRADED[m]])),
        )
        for m in MUTANTS
    ]
    return combine(wild, terms, len(family_map))


def wild_distance(truth: dict[str, Any], record: dict[str, Any]) -> float:
    """Return the registered capped D on the six families alone, the mutants left out."""
    scales, _, family_map = _distance_parts()
    return v2.predictive_distance_v2(truth, record, scales, family_map, capped=True)


def distances() -> dict[str, Callable[[dict[str, Any], dict[str, Any]], float]]:
    """Return the recovery's two distances by name, looked up when called."""
    return {"combined": recovery_distance, "wild_type": wild_distance}


# ---------------------------------------------------------------------------------------------
# The stop, the cap and the rules
# ---------------------------------------------------------------------------------------------


def stop(medians: list[float], sigma: float) -> bool:
    """Return whether a restart stops: the plateau, sigma <= sigma0, and the last 40 below 100.

    A median of 1,000 or more marks an undefined statistic, so the last condition also keeps a
    restart running while any median in its window is undefined.
    """
    if len(medians) < cr.PLATEAU_FIRST:
        return False
    window = medians[-2 * cr.PLATEAU_WINDOW :]
    return (
        cr.plateau_reached(medians)
        and sigma <= SIGMA0
        and max(window) < v2.UNDEFINED_RANK
        and max(window) < TIGHT
    )


def registered_cap(stop_generation: int) -> int:
    """Return the cap from the pilot's stop: 1.5 x it, up to a multiple of 10, in [300, 500]."""
    raw = math.ceil(CAP_FACTOR * stop_generation)
    rounded = CAP_STEP * math.ceil(raw / CAP_STEP)
    return int(min(CAP_CEILING, max(CAP_FLOOR, rounded)))


def population_passes(grade: dict[str, Any], mutants: tuple[str, ...] = MUTANTS) -> bool:
    """Return whether a population passes the wild type and every mutant."""
    return bool(grade["wild_type_passes"]) and all(grade["mutants"][m]["passes"] for m in mutants)


def counts(grades: list[dict[str, Any]]) -> dict[str, int]:
    """Return how many populations pass the wild type and each mutant."""
    return {
        "wild_type": sum(bool(g["wild_type_passes"]) for g in grades),
        **{m: sum(bool(g["mutants"][m]["passes"]) for g in grades) for m in MUTANTS},
    }


def three_of_five(grades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the rule on the first five: the wild type, tph-1 and tax-4 each in at least 3."""
    used = grades[:VERDICT_POPULATIONS]
    passing = counts(used)
    return {
        "populations": len(used),
        "passing": passing,
        "passes": bool(len(used) == VERDICT_POPULATIONS)
        and all(n >= PASS_NEEDED for n in passing.values()),
    }


def diagnosis(grades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return each failing statistic's z per population, presence's failures and each mutant's."""
    statistics: dict[str, list[float | None]] = {}
    for i, record in enumerate(grades):
        for key, ok in record["passes"].items():
            if ok is False:
                statistics.setdefault(key, [None] * len(grades))[i] = record["z"][key]
    presence = {
        name: sum(record.get(name) is not None and not record[name]["present"] for record in grades)
        for name in ("local_search_presence", "chemotaxis_presence")
    }
    mutants = {m: sum(not record["mutants"][m]["passes"] for record in grades) for m in MUTANTS}
    return {"statistics": statistics, "presence_failures": presence, "mutant_failures": mutants}


def pass_rates(grades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the wild type's and each mutant's pass share over every graded population."""
    passing = counts(grades)
    return {"populations": len(grades), **{k: v / len(grades) for k, v in passing.items()}}


def chosen_restart(
    selection: dict[int, list[dict[str, Any]]],
    scores: dict[int, float],
) -> int | None:
    """Return the restart passing the full criterion most; ties to the lower score, then lower r.

    ``selection`` must hold plateaued restarts only (:func:`choose` filters them).
    """
    if not selection:
        return None
    return min(
        selection,
        key=lambda r: (-sum(population_passes(g) for g in selection[r]), scores[r], r),
    )


def choose(
    histories: list[dict[str, Any]],
    selection: dict[int, list[dict[str, Any]]],
    scores: dict[int, float],
) -> int | None:
    """Return the chosen restart among the plateaued ones; a capped restart is never chosen."""
    plateaued = set(_plateaued(histories))
    return chosen_restart({r: g for r, g in selection.items() if r in plateaued}, scores)


def mutant_verdict(passing_restarts: list[bool], *, chosen: bool) -> str:
    """Return a mutant's row under the encoding table's ``unidentified`` rule.

    ``passing_restarts`` holds the mutant's verdict for each restart that passes the wild type. If
    they disagree the row is ``unidentified``; with none, the chosen restart's verdict is used, and
    the wild-type failure is stated beside it.
    """
    if not passing_restarts:
        return "pass" if chosen else "fail"
    if len(set(passing_restarts)) > 1:
        return "unidentified"
    return "pass" if passing_restarts[0] else "fail"


def mutant_rows(grades: list[list[dict[str, Any]]], chosen: int) -> dict[str, Any]:
    """Return each mutant's row across the restarts whose first five pass the wild type in 3."""
    first = [counts(g[:VERDICT_POPULATIONS]) for g in grades]
    wild = [r for r, c in enumerate(first) if c["wild_type"] >= PASS_NEEDED]
    rows: dict[str, Any] = {"wild_type_restarts": wild, "wild_type_failure_stated": not wild}
    for m in MUTANTS:
        verdicts = [first[r][m] >= PASS_NEEDED for r in wild]
        rows[m] = {
            "row": mutant_verdict(verdicts, chosen=first[chosen][m] >= PASS_NEEDED),
            "restart_verdicts": dict(zip(map(str, wild), verdicts, strict=True)),
        }
    return rows


def gain_from_start(start: float, d: float, floor: float) -> float:
    """Return the gain measured from the start: (start D - D) / (start D - floor)."""
    return rc.gain_fraction(start, d, floor)


# ---------------------------------------------------------------------------------------------
# Running simulations
# ---------------------------------------------------------------------------------------------


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
    """Run each (k, z) not already stored at a seed base; return every record by k.

    A stored record is reused only for the point and seed base it was run at.
    """
    stored = v2.read_records(path)
    for k, z in jobs:
        if k not in stored:
            continue
        if [float(v) for v in stored[k]["z"]] != [float(v) for v in z]:
            msg = f"{path}: simulation {k} is stored at another point; remove the store to rerun"
            raise ValueError(msg)
        if stored[k].get("seed_base") != seed_base:
            msg = f"{path}: simulation {k} is stored at another seed base"
            raise ValueError(msg)
    todo = [(k, z) for k, z in jobs if k not in stored]
    if todo:
        v2._drop_torn_line(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with (
            ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool,
            path.open("a") as out,
        ):
            for result in pool.map(
                simulate,
                [k for k, _ in todo],
                [z for _, z in todo],
                [seed_base] * len(todo),
            ):
                record = {**result, "seed_base": seed_base}
                out.write(json.dumps(record, allow_nan=False) + "\n")
                out.flush()
                stored[record["k"]] = record
    return stored


def grade_all(jobs: list[tuple[int, list[float], int]], workers: int) -> list[dict[str, Any]]:
    """Grade each (k, z, base) with presence and both mutants."""
    if not jobs:
        return []
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        return list(pool.map(grade_answer, *zip(*jobs, strict=True)))


def fit_restart(  # noqa: PLR0913 - keyword-only: one restart's registered settings
    *,
    fit: int,
    cma_seed: int,
    seed_base: int,
    cap: int,
    objective: Objective,
    state_dir: Path,
    pool: ProcessPoolExecutor,
) -> dict[str, Any]:
    """Run, or resume, one restart from the registered start with the registered stop."""
    return cr.run_fit(
        {},
        fit=fit,
        cma_seed=cma_seed,
        seed_base=seed_base,
        cap=cap,
        use_rule=True,
        state_dir=state_dir,
        pool=pool,
        scales={},
        objective=objective,
        simulate=simulate,
        stop=stop,
        strategy_factory=start_warm,
    )


def _write_once(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        msg = f"{path} exists: each record is written once"
        raise FileExistsError(msg)
    cr._write_json(path, nr._finite(record))


# ---------------------------------------------------------------------------------------------
# The recovery's preflight
# ---------------------------------------------------------------------------------------------


def distance_reading(  # noqa: PLR0913 - the truth's runs, the start's and the prior's
    observation: dict[str, Any],
    runs: list[dict[str, Any]],
    start_runs: list[dict[str, Any]],
    prior_runs: list[dict[str, Any]] | None,
    reference: dict[str, Any],
    distance: Callable[[dict[str, Any], dict[str, Any]], float],
) -> dict[str, Any]:
    """Return one distance's floor, oracle, start D, checks, band reference and gate line.

    ``runs`` are the truth's re-simulations 1-20, in order: 1-10 make the floor, 11-20 the oracle.
    """
    floor = float(np.median([distance(observation, r) for r in runs[:10]]))
    oracle = float(np.median([distance(observation, r) for r in runs[10:20]]))
    start = float(np.median([distance(observation, r) for r in start_runs]))
    band = distance(observation, reference)
    band_gain = gain_from_start(start, band, floor)
    out: dict[str, Any] = {
        "floor": floor,
        "oracle_D": oracle,
        "start_D": start,
        "separated": start - floor >= SEPARATION,
        "oracle_gain": gain_from_start(start, oracle, floor),
        "band_reference_D": band,
        "band_reference_gain": band_gain,
        "cannot_discriminate": band_gain >= GAIN_GATE,
        "gate_line_D": floor + (1.0 - GAIN_GATE) * (start - floor),
    }
    if prior_runs is not None:
        prior = float(np.median([distance(observation, r) for r in prior_runs]))
        out |= {
            "prior_D": prior,
            "prior_oracle_gain": rc.gain_fraction(prior, oracle, floor),
            "prior_start_gain": rc.gain_fraction(prior, start, floor),
        }
    return out


def preflight_record(
    observations: list[dict[str, Any]],
    repeats: list[list[dict[str, Any]]],
    start_runs: list[dict[str, Any]],
    prior_runs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Return the recovery's pre-fit checks for each truth, on both distances.

    ``repeats[t]`` holds truth t's re-simulations 1-20. Each check is read on the combined distance
    and on the wild type's alone: the start at least 1.0 above the floor, and the oracle's gain
    from the start at least 0.85. No statistic's definedness may flip between the observation and
    the floor's re-simulations. Each truth's observation is kept, so the recovery fits against it.
    """
    reference = band_truth(_bands())
    rows = []
    for observation, runs in zip(observations, repeats, strict=True):
        flips = definedness_flips(observation, runs[:10])
        readings = {
            name: distance_reading(observation, runs, start_runs, prior_runs, reference, fn)
            for name, fn in distances().items()
        }
        rows.append(
            {
                **readings,
                "definedness_flips": flips,
                "stable": not flips,
                "observation": observation,
            },
        )
    passed = all(
        row["stable"]
        and all(
            row[name]["separated"] and row[name]["oracle_gain"] >= ORACLE_GATE
            for name in DISTANCE_NAMES
        )
        for row in rows
    )
    return {"truths": rows, "passed": bool(passed)}


def definedness_flips(observation: dict[str, Any], repeats: list[dict[str, Any]]) -> list[str]:
    """Return the statistics whose definedness differs between the observation and any repeat."""

    def defined(record: dict[str, Any]) -> set[str]:
        wild = {k for k, v in record["statistics"].items() if v is not None}
        mutant = {
            f"{m}/{s}" for m, d in record["delta"].items() for s, v in d.items() if v is not None
        }
        return wild | mutant

    seen = defined(observation)
    return sorted(set().union(*(seen ^ defined(r) for r in repeats)))


def restart_reading(
    row: dict[str, Any],
    observation: dict[str, Any],
    answers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a recovery restart's median D and gain from the start, on both distances."""
    out = {}
    for name, distance in distances().items():
        d = float(np.median([distance(observation, a) for a in answers]))
        reading = row[name]
        out[name] = {
            "D": d,
            "gain": gain_from_start(reading["start_D"], d, reading["floor"]),
            "prior_gain": (
                rc.gain_fraction(reading["prior_D"], d, reading["floor"])
                if "prior_D" in reading
                else None
            ),
        }
    return out


def best_fit(fits: list[int], medians: dict[int, list[float]]) -> int:
    """Return the recovery restart with the lower final objective; the first on a tie."""
    return fits[cr.best_restart([medians[f] for f in fits])]


def recovery_gate(best: dict[str, Any]) -> dict[str, Any]:
    """Return the recovery's gate: the better restart's gain from the start >= 0.75 on both."""
    gains = {name: float(best[name]["gain"]) for name in DISTANCE_NAMES}
    return {"gains": gains, "passes": all(g >= GAIN_GATE for g in gains.values())}


def prior_points() -> np.ndarray:
    """Return the 200 registered prior draws of the 26-parameter unit box."""
    return np.random.default_rng(PRIOR_SEED).random((PRIOR_DRAWS, DIMENSION))


def _truth_runs(
    zs: list[list[float]],
    base: int,
    store: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    jobs = [(truth_k(t, q), z) for t, z in enumerate(zs) for q in range(TRUTH_REPEATS + 1)]
    runs = run_jobs(jobs, base, store, workers)
    observations = [runs[truth_k(t, 0)] for t in range(len(zs))]
    repeats = [[runs[truth_k(t, q)] for q in range(1, TRUTH_REPEATS + 1)] for t in range(len(zs))]
    return observations, repeats


def _start_runs(base: int, store: Path, workers: int) -> list[dict[str, Any]]:
    runs = run_jobs([(q, start_point()) for q in range(RESIMULATIONS)], base, store, workers)
    return [runs[q] for q in range(RESIMULATIONS)]


# ---------------------------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------------------------


def _pilot(workers: int) -> int:
    """One restart on discarded seeds, its 5 grades, the go rule, the preflight's check, the cap."""
    if PILOT_PATH.exists():
        print(f"{PILOT_PATH} exists: the pilot is read once", file=sys.stderr)
        return 1
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = fit_restart(
            fit=0,
            cma_seed=PILOT_CMA_SEED,
            seed_base=PILOT_BASE,
            cap=PILOT_CAP,
            objective=band_objective(),
            state_dir=BUILD / "pilot",
            pool=pool,
        )
    answer = history["answers"][-1]
    grades = grade_all([(g, answer, PILOT_GRADE_BASE) for g in range(PILOT_POPULATIONS)], workers)
    stopped = history["stop"]["by"] == "plateau"
    rule = three_of_five(grades)
    observations, repeats = _truth_runs(
        [answer],
        PILOT_TRUTH_BASE,
        BUILD / "pilot_truths.jsonl",
        workers,
    )
    starts = _start_runs(PILOT_START_BASE, BUILD / "pilot_start.jsonl", workers)
    check = preflight_record(observations, repeats, starts, None)
    generation = history["stop"]["generation"]
    seconds = float(np.mean(history["seconds"]))
    cap = registered_cap(generation) if stopped else None
    projected = projected_hours(generation, seconds, cap)
    decision = go_decision(history["stop"]["by"], rule, check, projected)
    record = {
        "description": "C.1 (v4)'s pilot on discarded seeds: one restart from the registered "
        "start, its go rule, the recovery preflight's check and the cap.",
        "cma_seed": PILOT_CMA_SEED,
        "stop": history["stop"],
        "medians": history["medians"],
        "sigmas": history["sigmas"],
        "seconds": history["seconds"],
        "answer": answer,
        "parameters": parameters(answer),
        "grades": grades,
        "go_rule": rule,
        "diagnosis": diagnosis(grades),
        "preflight_check": check,
        "cap": cap,
        "seconds_per_generation": seconds,
        "projected_hours": projected,
        "cost_bar_hours": COST_BAR_HOURS,
        "go_decision": decision,
        "go": decision["go"],
    }
    _write_once(PILOT_PATH, record)
    keys = ("stop", "go_rule", "cap", "projected_hours", "go_decision", "seconds_per_generation")
    print(json.dumps(nr._finite({k: record[k] for k in keys}), indent=2))
    print("preflight check:", json.dumps(nr._finite(check), indent=2))
    return 0 if record["go"] else 1


def projected_hours(generation: int, seconds: float, cap: int | None = None) -> dict[str, Any]:
    """Return the projected registered compute from the pilot's stop and seconds per generation.

    The bar reads ``registered``, every registered restart stopping where the pilot did; at the
    cap, where one is given, the upper bound is reported beside it.
    """
    restarts = TRUTHS * RECOVERY_RESTARTS + RESTARTS + 1
    hours = generation * seconds / 3600
    out: dict[str, Any] = {
        "recovery": TRUTHS * RECOVERY_RESTARTS * hours,
        "fit": RESTARTS * hours,
        "loo": hours,
        "registered": restarts * hours,
    }
    if cap is not None:
        out["registered_at_cap"] = restarts * cap * seconds / 3600
    return out


def go_decision(
    stop_by: str,
    rule: dict[str, Any],
    check: dict[str, Any],
    projected: dict[str, Any],
) -> dict[str, Any]:
    """Return the pilot's go: it stopped on the plateau, the go rule, the check, the cost bar."""
    parts = {
        "stopped": stop_by == "plateau",
        "go_rule": bool(rule["passes"]),
        "preflight_check": bool(check["passed"]),
        "within_cost": projected["registered"] <= COST_BAR_HOURS,
    }
    return {**parts, "go": all(parts.values())}


def truths() -> list[list[float]]:
    """Return the recovery's one truth: the pilot's answer, bound by Part B."""
    return [json.loads(PILOT_PATH.read_text())["answer"]]


def _cap() -> int:
    return int(json.loads(PILOT_PATH.read_text())["cap"])


def _recovery_preflight(workers: int) -> int:
    if RECOVERY_PREFLIGHT_PATH.exists():
        print(f"{RECOVERY_PREFLIGHT_PATH} exists: the preflight is read once", file=sys.stderr)
        return 1
    observations, repeats = _truth_runs(truths(), TRUTH_BASE, BUILD / "truths.jsonl", workers)
    starts = _start_runs(START_BASE, BUILD / "start.jsonl", workers)
    prior = prior_points()
    prior_store = run_jobs(
        [(j, list(map(float, prior[j]))) for j in range(PRIOR_DRAWS)],
        PRIOR_BASE,
        BUILD / "prior.jsonl",
        workers,
    )
    record = {
        "description": "C.1 (v4)'s recovery preflight: the truth's floor, oracle, start D, prior "
        "D, band reference and gate line, with the pre-fit checks.",
        **preflight_record(
            observations,
            repeats,
            starts,
            [prior_store[j] for j in range(PRIOR_DRAWS)],
        ),
    }
    _write_once(RECOVERY_PREFLIGHT_PATH, record)
    print(json.dumps(nr._finite(record), indent=2))
    return 0 if record["passed"] else 1


def _recovery(workers: int) -> int:
    if RECOVERY_PATH.exists():
        print(f"{RECOVERY_PATH} exists: the recovery is read once", file=sys.stderr)
        return 1
    if (
        not RECOVERY_PREFLIGHT_PATH.exists()
        or not json.loads(RECOVERY_PREFLIGHT_PATH.read_text())["passed"]
    ):
        print("the recovery's preflight has not passed; no recovery fit runs", file=sys.stderr)
        return 1
    preflight = json.loads(RECOVERY_PREFLIGHT_PATH.read_text())
    zs = truths()
    observations = [row["observation"] for row in preflight["truths"]]
    cap = _cap()
    histories: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for t, observation in enumerate(observations):
            for j in range(RECOVERY_RESTARTS):
                fit = RECOVERY_RESTARTS * t + j
                histories[fit] = fit_restart(
                    fit=fit,
                    cma_seed=RECOVERY_CMA_SEED + fit,
                    seed_base=RECOVERY_BASE,
                    cap=cap,
                    objective=truth_objective(observation),
                    state_dir=BUILD / "recovery",
                    pool=pool,
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
    rows, best_readings = [], []
    for t, observation in enumerate(observations):
        row = preflight["truths"][t]
        fits = [RECOVERY_RESTARTS * t + j for j in range(RECOVERY_RESTARTS)]
        best = best_fit(fits, {f: histories[f]["medians"] for f in fits})
        restarts = []
        for fit in fits:
            reading = restart_reading(
                row,
                observation,
                [answers[answer_k(fit, q)] for q in range(RESIMULATIONS)],
            )
            restarts.append(
                {
                    "fit": fit,
                    "cma_seed": RECOVERY_CMA_SEED + fit,
                    "stop": histories[fit]["stop"],
                    "final_objective": cr.final_objective(histories[fit]["medians"]),
                    **reading,
                    "best": fit == best,
                    "parameter_errors": unit_errors(histories[fit]["answers"][-1], zs[t]),
                    "medians": histories[fit]["medians"],
                    "sigmas": histories[fit]["sigmas"],
                },
            )
        best_readings.append(next(r for r in restarts if r["best"]))
        rows.append({"truth": t, "restarts": restarts})
    record = {
        "description": "C.1 (v4)'s recovery: the pilot's answer fitted twice from the registered "
        "start, the gate on the better restart's gain from the start, on both distances.",
        "truths": rows,
        **recovery_gate(best_readings[0]),
    }
    _write_once(RECOVERY_PATH, record)
    print(json.dumps(nr._finite({k: record[k] for k in ("gains", "passes")})))
    return 0 if record["passes"] else 1


def _histories() -> list[dict[str, Any]]:
    """Return the 12 restarts' histories; refused while any restart has not stopped."""
    histories = [json.loads((BUILD / "fits" / f"fit{r}.json").read_text()) for r in range(RESTARTS)]
    running = [r for r, h in enumerate(histories) if h["stop"] is None]
    if running:
        msg = f"restarts {running} have not stopped; run the fit to its end first"
        raise RuntimeError(msg)
    return histories


def _plateaued(histories: list[dict[str, Any]]) -> list[int]:
    return [r for r, h in enumerate(histories) if h["stop"]["by"] == "plateau"]


def _run(workers: int) -> int:
    """Run the 12 restarts, then re-simulate each answer 10 times."""
    cap = _cap()
    objective = band_objective()
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for r in range(RESTARTS):
            fit_restart(
                fit=r,
                cma_seed=CMA_SEED + r,
                seed_base=FIT_BASE,
                cap=cap,
                objective=objective,
                state_dir=BUILD / "fits",
                pool=pool,
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
    """Grade every plateaued restart on its 5 selection populations."""
    histories = _histories()
    plateaued = _plateaued(histories)
    jobs = [
        (select_k(r, s), histories[r]["answers"][-1], SELECT_BASE)
        for r in plateaued
        for s in range(SELECT_POPULATIONS)
    ]
    grades = grade_all(jobs, workers)
    by_restart = {
        str(r): grades[i * SELECT_POPULATIONS : (i + 1) * SELECT_POPULATIONS]
        for i, r in enumerate(plateaued)
    }
    cr._write_json(BUILD / "selection.json", nr._finite(by_restart))
    return 0


def _grade(workers: int) -> int:
    """Grade every restart on the 15 registered populations."""
    histories = _histories()
    jobs = [(g, h["answers"][-1], GRADE_BASE) for h in histories for g in range(GRADE_POPULATIONS)]
    grades = grade_all(jobs, workers)
    by_restart = [
        grades[r * GRADE_POPULATIONS : (r + 1) * GRADE_POPULATIONS] for r in range(RESTARTS)
    ]
    cr._write_json(BUILD / "grades.json", nr._finite(by_restart))
    return 0


def restart_spread(plateaued: list[int], runs: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """Return the pairwise median capped D between plateaued restarts' re-simulations."""
    bands = _bands()
    scales = v2.band_scales_v2(bands)
    family_map = {f: tuple(s) for f, s in bands["families"].items()}
    pairs = [
        {
            "restarts": [i, j],
            "D": float(
                np.median(
                    [
                        v2.predictive_distance_v2(a, b, scales, family_map, capped=True)
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


def at_bounds(z: Sequence[float], tolerance: float = 0.01) -> list[str]:
    """Return the parameters whose unit coordinate lies within the tolerance of a bound."""
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    names = [*MECHANISTIC_PARAMETERS, *c1v3.EXTRA, ON_FOOD[0]]
    return [n for n, v in zip(names, z, strict=True) if v <= tolerance or v >= 1 - tolerance]


def held_out(grades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return C.2's held-out reading on these graded populations, as description.

    It reads N2's and PD1074's held-out recording dates, which only the registered read may.
    """
    import c2_fit

    return c2_fit.held_out_reading(grades)


def held_out_cached(grades: list[dict[str, Any]], chosen: int) -> dict[str, Any]:
    """Return the held-out reading, kept on disk so a retried read loads no held-out row again."""
    path = BUILD / "held_out.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["chosen_restart"] == chosen:
            return cached["reading"]
    reading = held_out(grades)
    cr._write_json(path, nr._finite({"chosen_restart": chosen, "reading": reading}))
    return reading


def smoke_sets() -> dict[str, np.ndarray]:
    """Return band-cohort stand-ins for the held-out sets, so a smoke reads no held-out row."""
    import benchmark_bands as bb
    import benchmark_v2_controls as v2c
    from quantumnematode.benchmark import references as refs
    from wormbench import builder

    tracks = refs.load_scheer_tracks()
    return {
        "N2_bands": builder.band_rows(tracks, v2c.N2_ARM),
        "PD1074_bands": builder.band_rows(tracks, bb.WILD_TYPE),
    }


def _read() -> int:
    """Choose the restart on its selection populations and read its verdict on the registered."""
    if FIT_PATH.exists():
        print(f"{FIT_PATH} exists: the verdict is read once", file=sys.stderr)
        return 1
    histories = _histories()
    plateaued = _plateaued(histories)
    selection = {int(r): g for r, g in json.loads((BUILD / "selection.json").read_text()).items()}
    grades = json.loads((BUILD / "grades.json").read_text())
    stored = v2.read_records(BUILD / "answers.jsonl")
    runs = {r: [stored[answer_k(r, q)] for q in range(RESIMULATIONS)] for r in range(RESTARTS)}
    objective = band_objective()
    scores = {r: float(np.median([objective(x) for x in runs[r]])) for r in range(RESTARTS)}
    restarts = [
        {
            "restart": r,
            "cma_seed": CMA_SEED + r,
            "stop": h["stop"],
            "final_score": cr.final_objective(h["medians"]),
            "answer_score_median": scores[r],
            "answer_D_median": float(np.median([x["D"] for x in runs[r]])),
            "selection_passing": (
                sum(population_passes(g) for g in selection[r]) if r in selection else None
            ),
            "registered": three_of_five(grades[r]),
            "pass_rates": pass_rates(grades[r]),
            "population_D": [g["D"] for g in grades[r]],
            "reversal_share": [g["reversal_share"] for g in grades[r]],
            "answer": h["answers"][-1],
            "at_bounds": at_bounds(h["answers"][-1]),
            "medians": h["medians"],
            "sigmas": h["sigmas"],
            "seconds": h["seconds"],
        }
        for r, h in enumerate(histories)
    ]
    chosen = choose(histories, selection, scores)
    record: dict[str, Any] = {
        "description": DESCRIPTION,
        "cap": _cap(),
        "chosen_restart": chosen,
        "restarts": restarts,
        "restart_spread_D": restart_spread(plateaued, runs),
        "reversal_share_reference": ROBERTS_REVERSE_SHARE,
    }
    if chosen is None:
        record["stop"] = "no restart plateaued; the refit returns to the maintainer"
        _write_once(FIT_PATH, record)
        print(record["stop"], file=sys.stderr)
        return 1
    verdict = three_of_five(grades[chosen])
    record |= {
        "verdict": verdict,
        "passes": verdict["passes"],
        "diagnosis": diagnosis(grades[chosen][:VERDICT_POPULATIONS]),
        "pass_rates": pass_rates(grades[chosen]),
        "parameters": parameters(histories[chosen]["answers"][-1]),
        "grades": grades[chosen],
        "mutant_rows": mutant_rows(grades, chosen),
        "held_out": held_out_cached(grades[chosen], chosen),
    }
    _write_once(FIT_PATH, record)
    keys = ("chosen_restart", "verdict", "pass_rates", "mutant_rows")
    print(json.dumps(nr._finite({k: record[k] for k in keys}), indent=2))
    return 0


def loo_prediction(tax4_passing: int) -> str:
    """Return the leave-one-out's tax-4 reading on its first five populations."""
    return "prediction passed" if tax4_passing >= PASS_NEEDED else "prediction failed"


def _loo(workers: int) -> int:
    """Run the tax-4 leave-one-out: one restart with tax-4 out of the objective, graded on 15."""
    if LOO_PATH.exists():
        print(f"{LOO_PATH} exists: the leave-one-out is read once", file=sys.stderr)
        return 1
    if not FIT_PATH.exists():
        print("the registered fit has not been read; the leave-one-out follows it", file=sys.stderr)
        return 1
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        history = fit_restart(
            fit=0,
            cma_seed=LOO_CMA_SEED,
            seed_base=LOO_BASE,
            cap=_cap(),
            objective=band_objective(LOO_MUTANTS),
            state_dir=BUILD / "loo",
            pool=pool,
        )
    answer = history["answers"][-1]
    grades = grade_all([(g, answer, LOO_GRADE_BASE) for g in range(GRADE_POPULATIONS)], workers)
    fit = json.loads(FIT_PATH.read_text())
    first = three_of_five(grades)
    record = {
        "description": "C.1 (v4)'s tax-4 leave-one-out, as description: one restart with tax-4 "
        "out of the objective, its answer graded on 15 populations.",
        "cma_seed": LOO_CMA_SEED,
        "stop": history["stop"],
        "medians": history["medians"],
        "sigmas": history["sigmas"],
        "answer": answer,
        "parameters": parameters(answer),
        "grades": grades,
        "first_five": first,
        "prediction": loo_prediction(first["passing"]["tax-4"]),
        "capped": history["stop"]["by"] == "cap",
        "wild_type_fails": first["passing"]["wild_type"] < PASS_NEEDED,
        "pass_rates": pass_rates(grades),
        "registered_pass_rates": fit.get("pass_rates"),
    }
    _write_once(LOO_PATH, record)
    keys = ("stop", "prediction", "capped", "wild_type_fails", "first_five")
    print(json.dumps(nr._finite({k: record[k] for k in keys}), indent=2))
    return 0


def _smoke(workers: int) -> int:
    """Run an execution check, on discarded seeds at 699,000,000 and above.

    Two generations of a band fit, a truth fit and a leave-one-out fit; a preflight's arithmetic;
    one grade with both mutants; the held-out reading's code on it, with band-cohort dates in place
    of the held-out sets, so the smoke reads no held-out row.
    """
    state = BUILD / "smoke"
    shutil.rmtree(state, ignore_errors=True)
    store = state / "sims.jsonl"
    sims = run_jobs(
        [(q, start_point()) for q in range(4)],
        SMOKE_BASE + 300_000,
        store,
        workers,
    )
    pre = preflight_record([sims[3]], [[sims[1], sims[2]] * 10], [sims[0]], [sims[0]])
    histories = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        for name, base, objective in (
            ("band", SMOKE_BASE, band_objective()),
            ("truth", SMOKE_BASE + 100_000, truth_objective(sims[3])),
            ("loo", SMOKE_BASE + 200_000, band_objective(LOO_MUTANTS)),
        ):
            histories[name] = fit_restart(
                fit=0,
                cma_seed=SMOKE_CMA_SEED,
                seed_base=base,
                cap=2,
                objective=objective,
                state_dir=state / name,
                pool=pool,
            )
    graded = grade_all([(0, histories["band"]["answers"][-1], SMOKE_BASE + 400_000)], workers)[0]
    import c2_fit

    reading = c2_fit.held_out_reading(
        [graded],
        sets=smoke_sets(),
        splits=3,
        reference_populations=1,
    )
    print(
        json.dumps(
            nr._finite(
                {
                    "medians": {n: h["medians"] for n, h in histories.items()},
                    "preflight_floors": {
                        name: pre["truths"][0][name]["floor"] for name in DISTANCE_NAMES
                    },
                    "preflight_passed": pre["passed"],
                    "wild_type_passes": graded["wild_type_passes"],
                    "mutants": {m: graded["mutants"][m]["passes"] for m in MUTANTS},
                    "D": graded["D"],
                    "held_out": {k: v["answer"] for k, v in reading.items()},
                },
            ),
            indent=2,
        ),
    )
    ok = all(len(h["medians"]) == 2 for h in histories.values()) and graded["D"] is not None
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------
# The guards
# ---------------------------------------------------------------------------------------------


PART_A_BOUND: tuple[Path, ...] = (
    START_PATH,
    ENCODINGS_PATH,
    *FOLLOWING_CONFIGS,
    BANDS_PATH,
    PRIMITIVES_PATH,
)
PART_B_BOUND: tuple[Path, ...] = (*PART_A_BOUND, PILOT_PATH)
PART_C_BOUND: tuple[Path, ...] = (*PART_B_BOUND, RECOVERY_PREFLIGHT_PATH, RECOVERY_PATH)


def _has_part(name: str) -> bool:
    return LAUNCH_PATH.exists() and f"\n## {name}" in LAUNCH_PATH.read_text()


# The code the fit runs: the package sources and the scripts it calls, held at Part A's commit.
CODE_PATHS: tuple[Path, ...] = (
    _REPO / "packages" / "quantum-nematode" / "quantumnematode",
    _REPO / "packages" / "wormbench" / "wormbench",
    _REPO / "uv.lock",
    _REPO / "pyproject.toml",
    _REPO / "packages" / "quantum-nematode" / "pyproject.toml",
    _REPO / "packages" / "wormbench" / "pyproject.toml",
    *(
        _REPO / "scripts" / "analysis" / f"{name}.py"
        for name in (
            "c1_v4_fit",
            "c1_v3_fit",
            "c2_fit",
            "cma_recovery",
            "cma_recovery_v2",
            "npe_recovery",
            "npe_region_check",
            "benchmark_grade",
            "benchmark_bands",
            "benchmark_v2_controls",
        )
    ),
)


def part_a_text(text: str) -> str:
    """Return a launch record's Part A: everything before its Part B."""
    return text.split("\n## Part B", 1)[0]


def part_a_commit() -> str | None:
    """Return Part A's commit: the last commit that changed Part A's own text.

    Part B and Part C leave Part A unchanged, so their commits do not move it; a dated amendment of
    Part A does, and re-pins the code there. None before the launch record is committed.
    """
    rel = str(LAUNCH_PATH.relative_to(_REPO))
    commits = cr._git("log", "--format=%H", "--", rel).stdout.decode().split()[::-1]
    pin, previous = None, None
    for commit in commits:
        shown = cr._git("show", f"{commit}:{rel}")
        if shown.returncode != 0:
            continue
        text = part_a_text(shown.stdout.decode())
        if text != previous:
            pin, previous = commit, text
    return pin


def code_unchanged() -> bool:
    """Return whether the fit's code is Part A's commit's: no change, and no untracked file."""
    commit = part_a_commit()
    if commit is None:
        return False
    paths = [str(p.relative_to(_REPO)) for p in CODE_PATHS]
    if cr._git("diff", "--quiet", commit, "--", *paths).returncode != 0:
        return False
    status = cr._git("status", "--porcelain", "--untracked-files=all", "--", *paths)
    return status.returncode == 0 and not status.stdout.strip()


def _part_a() -> bool:
    return cr.launch_committed(PART_A_BOUND, launch_path=LAUNCH_PATH) and code_unchanged()


def _part_b() -> bool:
    return (
        _has_part("Part B")
        and cr.launch_committed(PART_B_BOUND, launch_path=LAUNCH_PATH)
        and bool(json.loads(PILOT_PATH.read_text())["go"])
    )


def decided_despite_gate() -> bool:
    """Return whether an amended Part A carries the decision heading as a line of its own."""
    if not LAUNCH_PATH.exists():
        return False
    lines = part_a_text(LAUNCH_PATH.read_text()).splitlines()
    return any(line.strip() == DECIDED_MARKER for line in lines)


def _part_c() -> bool:
    return (
        _has_part("Part C")
        and cr.launch_committed(PART_C_BOUND, launch_path=LAUNCH_PATH)
        and (bool(json.loads(RECOVERY_PATH.read_text())["passes"]) or decided_despite_gate())
    )


# Each stage's parts: Part A alone for the pilot; with Part B and the pilot's go for the recovery;
# with Part C and the recovery's pass for the fit and everything after it.
REQUIRED: dict[str, tuple[str, ...]] = {
    "pilot": ("A",),
    "recovery-preflight": ("A", "B"),
    "recovery": ("A", "B"),
    "run": ("A", "B", "C"),
    "select": ("A", "B", "C"),
    "grade": ("A", "B", "C"),
    "read": ("A", "B", "C"),
    "loo": ("A", "B", "C"),
}


def guarded(stage: str) -> bool:
    """Return whether a stage may run: its parts committed, their bound records and code unchanged."""
    checks = {"A": _part_a, "B": _part_b, "C": _part_c}
    return all(checks[part]() for part in REQUIRED[stage])


STAGES: dict[str, Callable[[int], int]] = {
    "pilot": _pilot,
    "recovery-preflight": _recovery_preflight,
    "recovery": _recovery,
    "run": _run,
    "select": _select,
    "grade": _grade,
    "read": lambda _: _read(),
    "loo": _loo,
}


def main(argv: list[str] | None = None) -> int:
    """CLI: the pilot, the recovery, the fit, selection, grading, the reading and a smoke."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("stage", choices=(*STAGES, "smoke"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.stage == "smoke":
        return _smoke(args.workers)
    if not guarded(args.stage):
        print(
            "the launch record's part for this stage, its bound records or the gate it needs is "
            "missing, uncommitted, changed or failing",
            file=sys.stderr,
        )
        return 1
    return STAGES[args.stage](args.workers)


if __name__ == "__main__":
    sys.exit(main())
