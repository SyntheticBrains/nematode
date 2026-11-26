#!/usr/bin/env python3
"""The fidelity table: C.1 (v3)'s predicted mutants graded once, and the table assembled.

C.1 (v3) was fitted to the wild type alone, so its tph-1 and tax-4 rows are predictions under the
registered encoding table. ``grade`` reads them once:

- restarts 0, 4 and 6, the three whose answers pass v3's wild type, each on its own registered
  populations, on the arm it was fitted on;
- first the wild type alone, which must reproduce its committed v3 distances and passes, and the
  chosen restart its committed v4 grade; no mutant is simulated unless it does;
- then the wild type on every v4 assay and each mutant on the small lawn, on the same animals, with
  each mutant's differences from that wild type graded on v4's mutant bands at n = 200.

``table`` assembles the fidelity table from the committed records and the grade record. It copies
each recorded value and checks every count it can recompute against its record.

Stages::

    uv run python scripts/analysis/fidelity_table.py grade
    uv run python scripts/analysis/fidelity_table.py table
    uv run python scripts/analysis/fidelity_table.py smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_grade as bg
import c1_fit
import c1_v3_fit as c1v3
import c1_v4_fit as c1v4
import cma_recovery as cr
import npe_recovery as nr

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_REPO = Path(__file__).resolve().parents[2]
_LOGBOOKS = _REPO / "docs" / "experiments" / "logbooks" / "supporting"
SUPPORT = _LOGBOOKS / "137-fidelity-table"
LAUNCH_PATH = SUPPORT / "launch.md"
GRADE_PATH = SUPPORT / "grade.json"
TABLE_PATH = SUPPORT / "fidelity_table.json"
TABLE_MD_PATH = SUPPORT / "fidelity_table.md"
BUILD = _REPO / "build" / "fidelity-table"

C1_FIT_PATH = _LOGBOOKS / "121-c1-v3-fit" / "fit.json"
C1_RECOVERY_PATH = _LOGBOOKS / "121-c1-v3-fit" / "recovery.json"
C2_FIT_PATH = _LOGBOOKS / "133-c2-fit" / "fit.json"
B4_READ_PATH = _LOGBOOKS / "124-benchmark-v4" / "read.json"
B4_CONTROLS_PATH = _LOGBOOKS / "110-benchmark-controls" / "controls.json"
PILOT_PATH = _LOGBOOKS / "136-c1-v4-fit" / "pilot.json"
ENCODINGS_PATH = _LOGBOOKS / "113-c1-fit" / "encodings.md"
BANDS_PATH = _REPO / "data" / "benchmark" / "bands_v4.json"
BANDS_V3_PATH = _REPO / "data" / "benchmark" / "bands_v3.json"
CLASSIFIER_PATH = _REPO / "data" / "benchmark" / "classifier.json"
PRIMITIVES_PATH = _REPO / "data" / "locomotion" / "primitives.json"

ARM = "mlp"
MLP_CONFIGS: tuple[Path, ...] = tuple(
    sorted((_REPO / "configs" / "scenarios" / "foraging").glob(f"{bg.ARMS[ARM]}_assay_*.yml")),
)

RESTARTS = (0, 4, 6)
CHOSEN = 6
MUTANTS = ("tph-1", "tax-4")
NOT_ENCODABLE = ("pdfr-1",)
POPULATIONS = 5
THRESHOLD = 3
GRADE_BASE = 525_000_000
MORE_BASE = 555_000_000
MORE_POPULATIONS = 10
SMOKE_BASE = 99_870_000
SMOKE_POPULATIONS = 2
ANIMALS = 200
D_TOLERANCE = 1e-9
WORKERS = 16

BOUND: tuple[Path, ...] = (
    C1_FIT_PATH,
    C1_RECOVERY_PATH,
    C2_FIT_PATH,
    B4_READ_PATH,
    B4_CONTROLS_PATH,
    PILOT_PATH,
    ENCODINGS_PATH,
    BANDS_PATH,
    BANDS_V3_PATH,
    CLASSIFIER_PATH,
    PRIMITIVES_PATH,
    *MLP_CONFIGS,
)

# The code the grade runs: the package sources and the analysis scripts it imports, held at Part
# A's commit.
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
            "fidelity_table",
            "c1_fit",
            "c1b_fit",
            "c1_v3_fit",
            "c1_v4_fit",
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


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    """Return a file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def answer(restart: int) -> list[float]:
    """Return a C.1 (v3) restart's answer, its unit coordinates from the committed fit record."""
    record = _read(C1_FIT_PATH)["restarts"][restart]
    if record["restart"] != restart:
        msg = f"restart {restart} is not at its index in {C1_FIT_PATH}"
        raise ValueError(msg)
    return list(map(float, record["answer"]))


def population_seeds(base: int, g: int) -> list[int]:
    """Return population g's animal seeds at a base: base + 200·g + i."""
    return [base + ANIMALS * g + i for i in range(ANIMALS)]


# ---------------------------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------------------------


def _controller(params: dict[str, float]) -> Callable[[Path], Any]:
    from quantumnematode.evaluation.controllers import BatchedMechanistic

    return lambda _path: BatchedMechanistic(params)


def grade_population(job: tuple[int, int, int], *, mutants: bool = True) -> dict[str, Any]:
    """Grade one restart on one population: the wild type on v3 and v4, and each mutant's Δ.

    With ``mutants`` false only the wild type is simulated, and the record holds no mutant.
    """
    restart, g, base = job
    bands = c1v4._bands()
    bands_v3 = c1v3._bands()
    seeds = population_seeds(base, g)
    params = c1v3.parameters(answer(restart))
    populations, chemotaxis = bg.simulate_v2(ARM, _controller(params), seeds, bands)
    v4 = bg.grade_populations_v4(populations, chemotaxis, bands, presence=True)
    v3 = bg.grade_populations_v3(populations, chemotaxis, bands_v3, presence=True)
    share = v4["reversal_share"]
    record: dict[str, Any] = {
        "restart": restart,
        "g": g,
        "base": base,
        "v3": {"D": v3["D"], "wild_type_passes": bool(v3["wild_type_passes"])},
        "v4": {
            "D": v4["D"],
            "D_fit": v4["D_fit"],
            "wild_type_passes": bool(v4["wild_type_passes"]),
            "family_passes": v4["family_passes"],
            "reversal_share": None if share is None else share["value"],
        },
    }
    if not mutants:
        return nr._finite(record)
    clf = bg.classifier(bands)
    wild = bg.family_values("small_lawn", populations["small_lawn"], clf)
    delta: dict[str, dict[str, float | None]] = {}
    readings: dict[str, Any] = {}
    for mutant in MUTANTS:
        mutated = c1v3.mutant_parameters(params, mutant)
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
        delta[mutant] = {
            s: c1v4._difference(values.get(s), wild.get(s)) for s in c1v4.LAWN_DIFFERENCES
        }
        readings[mutant] = c1v4.mutant_reading(mutant, delta[mutant], bands)
    return nr._finite({**record, "delta": delta, "mutants": readings})


def run_jobs(
    jobs: Sequence[tuple[int, int, int]],
    workers: int,
    *,
    mutants: bool = True,
) -> list[dict[str, Any]]:
    """Grade every (restart, population, base) job on a process pool, in job order."""
    with ProcessPoolExecutor(max_workers=workers, initializer=nr._worker_setup) as pool:
        return list(pool.map(partial(grade_population, mutants=mutants), jobs))


def wild_jobs() -> list[tuple[int, int, int]]:
    """Return the reproduction check's jobs: each restart on its 5 registered populations."""
    return [(r, g, GRADE_BASE) for r in RESTARTS for g in range(POPULATIONS)]


def registered_jobs() -> list[tuple[int, int, int]]:
    """Return the registered jobs: each restart on its 5 populations, then the chosen on 10 more."""
    return [*wild_jobs(), *((CHOSEN, g, MORE_BASE) for g in range(MORE_POPULATIONS))]


# ---------------------------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------------------------


def _same(got: float | None, committed: float | None) -> bool:
    if got is None or committed is None:
        return False
    return math.isclose(got, committed, rel_tol=0.0, abs_tol=D_TOLERANCE)


def reproduction(
    records: Sequence[dict[str, Any]],
    c1_fit_record: dict[str, Any],
    b4_read: dict[str, Any],
) -> dict[str, Any]:
    """Compare the registered populations' wild type with its committed grades.

    Each of restarts 0, 4 and 6 must appear once on each of its 5 registered populations. Every
    v3 D must equal its committed ``population_D`` within the tolerance and every v3 pass its
    ``population_passes``; the chosen restart's v4 D and pass must equal B v4's read of it.
    """
    committed = {r["restart"]: r for r in c1_fit_record["restarts"]}
    answer_read = b4_read["c1_v3_answer"]
    v4_committed = {p["k"]: p for p in answer_read["populations"]}
    mismatches: list[dict[str, Any]] = []
    if answer_read["base"] != GRADE_BASE:
        mismatches.append(
            {"what": "B v4's base", "got": answer_read["base"], "committed": GRADE_BASE},
        )
    registered = [rec for rec in records if rec["base"] == GRADE_BASE]
    pairs = sorted((rec["restart"], rec["g"]) for rec in registered)
    expected = sorted((r, g) for r in RESTARTS for g in range(POPULATIONS))
    if pairs != expected:
        mismatches.append({"what": "populations", "got": pairs, "committed": expected})

    def check(rec: dict[str, Any], what: str, got: Any, value: Any, *, exact: bool) -> None:  # noqa: ANN401
        ok = got == value if exact else _same(got, value)
        if not ok:
            mismatches.append(
                {
                    "restart": rec["restart"],
                    "g": rec["g"],
                    "what": what,
                    "got": got,
                    "committed": value,
                },
            )

    for rec in registered:
        r, g = rec["restart"], rec["g"]
        check(rec, "v3 D", rec["v3"]["D"], committed[r]["population_D"][g], exact=False)
        check(
            rec,
            "v3 pass",
            rec["v3"]["wild_type_passes"],
            committed[r]["population_passes"][g],
            exact=True,
        )
        if r == CHOSEN:
            check(rec, "v4 D", rec["v4"]["D"], v4_committed[g]["D"], exact=False)
            check(
                rec,
                "v4 pass",
                rec["v4"]["wild_type_passes"],
                bool(v4_committed[g]["wild_type_passes"]),
                exact=True,
            )
    return {"reproduced": not mismatches, "mismatches": mismatches}


def repeat_identical(wild: Sequence[dict[str, Any]], full: Sequence[dict[str, Any]]) -> bool:
    """Return whether the full run's registered wild type equals the reproduction check's."""
    first = {(r["restart"], r["g"]): (r["v3"], r["v4"]) for r in wild}
    second = {(r["restart"], r["g"]): (r["v3"], r["v4"]) for r in full if r["base"] == GRADE_BASE}
    return first == second


def restart_passes(records: Sequence[dict[str, Any]], restart: int, mutant: str) -> list[bool]:
    """Return one restart's mutant passes on its registered populations, in population order.

    A population passes a mutant when all of the mutant's graded statistics pass in it.
    """
    rows = sorted(
        (r for r in records if r["restart"] == restart and r["base"] == GRADE_BASE),
        key=lambda r: r["g"],
    )
    return [bool(r["mutants"][mutant]["passes"]) for r in rows]


def restart_verdict(passes: Sequence[bool]) -> bool:
    """Return whether a restart passes a mutant: in at least 3 of its 5 registered populations."""
    if len(passes) != POPULATIONS:
        msg = f"a restart's verdict needs {POPULATIONS} populations, got {len(passes)}"
        raise ValueError(msg)
    return sum(passes) >= THRESHOLD


def prediction(records: Sequence[dict[str, Any]], mutant: str) -> dict[str, Any]:
    """Return a mutant's prediction under the ``unidentified`` rule across restarts 0, 4 and 6."""
    passes = {r: restart_passes(records, r, mutant) for r in RESTARTS}
    verdicts = {r: restart_verdict(p) for r, p in passes.items()}
    rule = c1_fit.mutant_verdict([verdicts[r] for r in RESTARTS], chosen=verdicts[CHOSEN])
    return {
        "passes": {str(r): sum(p) for r, p in passes.items()},
        "by_population": {str(r): p for r, p in passes.items()},
        "restart_passes": {str(r): v for r, v in verdicts.items()},
        "prediction": rule,
    }


def more_populations(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the chosen restart's pass rates over its 5 registered and 10 more populations."""
    rows = [r for r in records if r["restart"] == CHOSEN]
    out: dict[str, Any] = {"populations": len(rows)}
    out["wild_type_v4"] = sum(r["v4"]["wild_type_passes"] for r in rows) / len(rows)
    for m in MUTANTS:
        out[m] = sum(bool(r["mutants"][m]["passes"]) for r in rows) / len(rows)
    more = [r for r in rows if r["base"] == MORE_BASE]
    out["more"] = {
        "populations": len(more),
        "wild_type_v4": sum(r["v4"]["wild_type_passes"] for r in more),
        **{m: sum(bool(r["mutants"][m]["passes"]) for r in more) for m in MUTANTS},
    }
    return out


def _header() -> dict[str, Any]:
    return {
        "description": (
            "C.1 (v3)'s predicted tph-1 and tax-4 under the registered encoding table, on its "
            "wild-type-passing restarts' registered populations, with the chosen restart on 10 more."
        ),
        "arm": ARM,
        "restarts": list(RESTARTS),
        "chosen": CHOSEN,
        "threshold": f"{THRESHOLD} of {POPULATIONS}",
        "bound": {_rel(p): sha256(p) for p in BOUND},
    }


def failed_record(wild: list[dict[str, Any]], check: dict[str, Any]) -> dict[str, Any]:
    """Return the grade record when the wild type does not reproduce: no mutant was simulated."""
    return {
        **_header(),
        "reproduction": check,
        "wild_type": wild,
        "populations": None,
        "predictions": None,
    }


def grade_record(
    wild: list[dict[str, Any]],
    check: dict[str, Any],
    full: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the grade record: the reproduction check, the populations and the predictions."""
    same = repeat_identical(wild, full)
    record: dict[str, Any] = {
        **_header(),
        "reproduction": {**check, "repeat_identical": same},
        "wild_type": wild,
        "populations": full,
    }
    if not (check["reproduced"] and same):
        record["predictions"] = None
        return record
    record["predictions"] = {m: prediction(full, m) for m in MUTANTS}
    record["more_populations"] = more_populations(full)
    return record


# ---------------------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------------------


def _median(values: Sequence[float]) -> float:
    return float(st.median(values))


def _family_counts(family_passes: Sequence[dict[str, bool]]) -> dict[str, int]:
    families = family_passes[0].keys()
    return {f: sum(bool(p[f]) for p in family_passes) for f in families}


def _agree(name: str, recorded: Any, recomputed: Any) -> Any:  # noqa: ANN401
    """Return a recorded value, refusing if a recount from the same record disagrees."""
    if recorded != recomputed:
        msg = f"{name}: the record holds {recorded!r} and its populations give {recomputed!r}"
        raise ValueError(msg)
    return recorded


def tph1_references(controls: dict[str, Any]) -> dict[str, Any]:
    """Return tph-1's leaving references: its band centre and B.4's single held-out reading."""
    reading = controls["positive_controls"]["units"]["mutant/tph-1"]["statistics"][
        "exits_per_on_lawn_hour"
    ]
    return {
        "band_centre": reading["centre"],
        "held_out_value": reading["value"],
        "source": _rel(B4_CONTROLS_PATH),
    }


def c1_row(
    c1fit: dict[str, Any],
    recovery: dict[str, Any],
    b4: dict[str, Any],
    controls: dict[str, Any],
    grade: dict[str, Any],
) -> dict[str, Any]:
    """Return C.1 (v3)'s row: its v4 wild type as Logbook 124 read it, and its predicted mutants."""
    answer_read = b4["c1_v3_answer"]
    pops = answer_read["populations"]
    by_population = [bool(p["wild_type_passes"]) for p in pops]
    _agree("C.1 (v3)'s v4 passes", list(answer_read["v4_population_passes"]), by_population)
    tph1_exits = [
        r["delta"]["tph-1"]["exits_per_on_lawn_hour"]
        for r in sorted(grade["populations"], key=lambda r: r["g"])
        if r["restart"] == CHOSEN and r["base"] == GRADE_BASE
    ]
    mutants: dict[str, Any] = {m: {"kind": "predicted", **grade["predictions"][m]} for m in MUTANTS}
    for m in NOT_ENCODABLE:
        mutants[m] = {"kind": "not encodable", "prediction": None}
    return {
        "controller": "C.1 (v3)",
        "answer": f"restart {c1fit['chosen_restart']} of {_rel(C1_FIT_PATH)}",
        "body": f"benchmark_grade.ARMS[{ARM!r}]: the pooled on-food reversal",
        "fitted": "the wild type on v3's five families; no mutant",
        "recovery": {
            "control": "C.1 (v3)'s reduced recovery",
            "median_gain": recovery["median_gain"],
            "passes": recovery["passes"],
        },
        "wild_type_v3": {
            "populations": len(c1fit["grades"]),
            "passing": c1fit["populations_passing"],
        },
        "wild_type_v4": {
            "populations": len(pops),
            "passing": sum(by_population),
            "by_population": by_population,
            "family_passes": _family_counts([p["family_passes"] for p in pops]),
            "D": [p["D"] for p in pops],
            "D_fit": [p["D_fit"] for p in pops],
            "reversal_share": [p["reversal_share"]["value"] for p in pops],
            "failing_activity": answer_read["failing_activity"],
        },
        "mutants": mutants,
        "tph1_leaving_reported": {
            "exits_per_on_lawn_hour_delta": tph1_exits,
            **tph1_references(controls),
        },
        "more_populations": grade["more_populations"],
        "held_out": "not read",
    }


def c2_row(c2: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    """Return C.2's row from its registered fit record."""
    grades = sorted(c2["grades"], key=lambda g: g["population"])
    first = [g for g in grades if g["population"] < POPULATIONS]
    verdict = c2["verdict"]
    by_population = [bool(g["wild_type_passes"]) for g in first]
    _agree("C.2's wild-type passes", verdict["wild_type_passes"], sum(by_population))
    mutants = {}
    for m in ("tph-1", "pdfr-1", "tax-4"):
        passes = _agree(
            f"C.2's {m} passes",
            verdict["mutant_passes"][m],
            sum(bool(g["mutants"][m]["passes"]) for g in first),
        )
        mutants[m] = {"kind": "fitted", "passes": passes}
    held = {
        name: {
            "dates": h["dates"],
            "answer_median": h["answer"]["median"],
            "real_halves_p95": h["real_halves"]["p95"],
            "band_reference_p95": h["band_reference"]["p95"],
        }
        for name, h in c2["held_out"].items()
    }
    return {
        "controller": "C.2",
        "answer": f"the registered answer of {_rel(C2_FIT_PATH)}",
        "body": (
            "benchmark_grade.ARMS['connectome_c2_following']: the speed-following on-food reversal"
        ),
        "fitted": "all six families with tph-1, pdfr-1 and tax-4 at half",
        "recovery": {
            "control": "F.0b-ii",
            "gain": c2["recovery_gate"]["gain"],
            "passes": c2["recovery_gate"]["passes"],
        },
        "verdict": verdict,
        "wild_type_v4": {
            "populations": len(first),
            "passing": verdict["wild_type_passes"],
            "by_population": by_population,
            "family_passes": _family_counts([g["family_passes"] for g in first]),
            "D": [g["D"] for g in first],
            "D_fit": [g["D_fit"] for g in first],
            "reversal_share": [g["reversal_share"] for g in first],
        },
        "pass_rates": c2["pass_rates"],
        "mutants": mutants,
        "tph1_leaving_reported": {
            "exits_per_on_lawn_hour_delta": [
                g["delta"]["tph-1"]["exits_per_on_lawn_hour"] for g in first
            ],
            **tph1_references(controls),
        },
        "all_D": [g["D"] for g in grades],
        "held_out": held,
    }


def negative_rows(b4: dict[str, Any]) -> list[dict[str, Any]]:
    """Return B v4's negatives, each as read in its controls."""
    rows = []
    for name, neg in b4["negatives"].items():
        pops = neg["populations"]
        failing = _agree(
            f"{name}'s failures",
            neg["fails_wild_type_in"],
            sum(not p["wild_type_passes"] for p in pops),
        )
        rows.append(
            {
                "controller": name,
                "populations": len(pops),
                "fails_wild_type_in": failing,
                "family_passes": _family_counts([p["family_passes"] for p in pops]),
                "D_median": _median([p["D"] for p in pops]),
                "D_fit_median": _median([p["D_fit"] for p in pops]),
            },
        )
    return rows


def reference_rows(b4: dict[str, Any]) -> dict[str, Any]:
    """Return the real-worm controls and W's minimum, as B v4 read them."""
    n2 = b4["n2_control"]
    pd = b4["pd1074_held_out"]
    return {
        "n2_unread_dates_activity": {
            "dates": n2["unread_dates"],
            "units": sorted(n2["units"]),
            "failed": n2["failed"],
            "passes": n2["passes"],
        },
        "pd1074_held_out_activity": {
            "dates": pd["dates"],
            "animals": pd["animals"],
            "passes": {s: pd["statistics"][s]["passes"] for s in pd["graded"]},
        },
        "w_minimum": b4["w_minimum"],
    }


def pilot_row(pilot: dict[str, Any]) -> dict[str, Any]:
    """Return C.1's v4 pilot answer, as description on discarded seeds."""
    return {
        "controller": "C.1 (v4) pilot answer (description, discarded seeds)",
        "stop": pilot["stop"],
        "passing": pilot["go_rule"]["passing"],
        "populations": pilot["go_rule"]["populations"],
        "D": [g["D"] for g in pilot["grades"]],
    }


def contrast(c1: dict[str, Any], c2: dict[str, Any]) -> dict[str, Any]:
    """Return C.2 against C.1, as description with its conditions and no verdict."""
    return {
        "reading": "description; no verdict on which is closer",
        "wild_type_v4_passing": {
            "C.2": f"{c2['wild_type_v4']['passing']} of {c2['wild_type_v4']['populations']}",
            "C.1 (v3)": f"{c1['wild_type_v4']['passing']} of {c1['wild_type_v4']['populations']}",
        },
        "median_D": {
            "C.2": _median(c2["wild_type_v4"]["D"]),
            "C.1 (v3)": _median(c1["wild_type_v4"]["D"]),
        },
        "median_D_fit": {
            "C.2": _median(c2["wild_type_v4"]["D_fit"]),
            "C.1 (v3)": _median(c1["wild_type_v4"]["D_fit"]),
        },
        "conditions": [
            "C.2's recovery control, F.0b-ii, failed its gate",
            "the bodies differ: C.2 on the speed-following on-food reversal, C.1 (v3) on the pooled",
            "C.2's mutants are fitted and in-sample; C.1's are predicted",
            "each is graded on its own registered populations, so the populations are not paired",
        ],
    }


def build_table(grade: dict[str, Any]) -> dict[str, Any]:
    """Return the fidelity table from the committed records and the grade record."""
    if not grade.get("predictions"):
        msg = "the grade record holds no prediction: the wild type did not reproduce"
        raise ValueError(msg)
    b4 = _read(B4_READ_PATH)
    controls = _read(B4_CONTROLS_PATH)
    c1 = c1_row(_read(C1_FIT_PATH), _read(C1_RECOVERY_PATH), b4, controls, grade)
    c2 = c2_row(_read(C2_FIT_PATH), controls)
    sources = {_rel(p): sha256(p) for p in BOUND}
    if GRADE_PATH.exists():
        sources[_rel(GRADE_PATH)] = sha256(GRADE_PATH)
    return {
        "description": (
            "C.4, the fidelity table: Phase 10's controllers, the negatives and the real-worm "
            "controls on benchmark v4."
        ),
        "benchmark": "v4",
        "bands_sha256": sha256(BANDS_PATH),
        "sources": sources,
        "controllers": [
            c1,
            c2,
            {"controller": "C.3", "status": "not run in Phase 10"},
        ],
        "negatives": negative_rows(b4),
        "references": reference_rows(b4),
        "description_rows": [pilot_row(_read(PILOT_PATH))],
        "contrast": contrast(c1, c2),
    }


def _passed(passes: bool) -> str:  # noqa: FBT001 - a record's boolean, rendered
    return "passed" if passes else "failed"


def _verdict_cell(m: dict[str, Any], wild_type: dict[str, Any]) -> str:
    if m["kind"] == "not encodable":
        return "not encodable"
    if m["kind"] == "fitted":
        return f"fitted, {m['passes']} of {POPULATIONS}"
    by_restart = ", ".join(f"r{r} {n}" for r, n in m["passes"].items())
    return (
        f"predicted: **{m['prediction']}** ({by_restart} of {POPULATIONS}); the wild type passes "
        f"v4 in {wild_type['passing']} of {wild_type['populations']}"
    )


def _held_cell(held: dict[str, Any]) -> str:
    return "; ".join(
        f"{name}: median {h['answer_median']:.2f}, against real halves' 95th percentile "
        f"{h['real_halves_p95']:.2f} and the band reference's {h['band_reference_p95']:.2f}"
        for name, h in held.items()
    )


def markdown(table: dict[str, Any]) -> str:
    """Return the table rendered as Markdown, every figure from the JSON."""
    c1, c2, c3 = table["controllers"]
    w1, w2 = c1["wild_type_v4"], c2["wild_type_v4"]
    lines = [
        "# The fidelity table (C.4)",
        "",
        "Benchmark v4. Every figure is from `fidelity_table.json`.",
        "",
        "| | C.1 (v3) | C.2 |",
        "|---|---|---|",
        f"| Body | {c1['body']} | {c2['body']} |",
        f"| Fitted | {c1['fitted']} | {c2['fitted']} |",
        (
            f"| Recovery control | {c1['recovery']['control']}: gain "
            f"{c1['recovery']['median_gain']:.3f}, {_passed(c1['recovery']['passes'])} "
            f"| {c2['recovery']['control']}: gain {c2['recovery']['gain']:.3f}, "
            f"{_passed(c2['recovery']['passes'])} |"
        ),
        (
            f"| Wild type on v4 | {w1['passing']} of {w1['populations']} | {w2['passing']} of "
            f"{w2['populations']} ({c2['pass_rates']['wild_type']:.0%} of "
            f"{c2['pass_rates']['populations']}) |"
        ),
    ]
    lines.extend(
        (
            f"| … {family} | {w1['family_passes'][family]} of {w1['populations']} "
            f"| {w2['family_passes'][family]} of {w2['populations']} |"
        )
        for family in w1["family_passes"]
    )
    lines.append(
        f"| Median D (D_fit) | {_median(w1['D']):.2f} ({_median(w1['D_fit']):.2f}) "
        f"| {_median(w2['D']):.2f} ({_median(w2['D_fit']):.2f}) |",
    )
    lines.append(
        f"| Median reversal share (real worms 17.3%) | {_median(w1['reversal_share']):.1%} "
        f"| {_median(w2['reversal_share']):.1%} |",
    )
    lines.extend(
        f"| {m} | {_verdict_cell(c1['mutants'][m], w1)} | {_verdict_cell(c2['mutants'][m], w2)} |"
        for m in ("tph-1", "pdfr-1", "tax-4")
    )
    ref1, ref2 = c1["tph1_leaving_reported"], c2["tph1_leaving_reported"]
    lines.append(
        f"| tph-1's leaving, Δ exits per on-lawn hour (reported) "
        f"| median {_median(ref1['exits_per_on_lawn_hour_delta']):+.2f} "
        f"| median {_median(ref2['exits_per_on_lawn_hour_delta']):+.2f} |",
    )
    lines += [f"| Held-out dates | {c1['held_out']} | {_held_cell(c2['held_out'])} |", ""]
    lines += [
        (
            f"tph-1's leaving is reported against its band centre ({ref1['band_centre']:+.2f} "
            f"per on-lawn hour) and B.4's single held-out reading ({ref1['held_out_value']:+.2f})."
        ),
        "",
        f"C.3: {c3['status']}.",
        "",
        "## The negatives (B v4's controls)",
        "",
        "| Negative | Fails the wild type in | Median D |",
        "|---|---|---|",
    ]
    lines += [
        f"| {n['controller']} | {n['fails_wild_type_in']} of {n['populations']} "
        f"| {n['D_median']:.2f} |"
        for n in table["negatives"]
    ]
    ref = table["references"]
    n2, pd = ref["n2_unread_dates_activity"], ref["pd1074_held_out_activity"]
    w = ref["w_minimum"]
    lines += [
        "",
        "## The real-worm controls (B v4)",
        "",
        (
            f"- N2's {n2['dates']} unread dates, on activity in {len(n2['units'])} halves: "
            f"{'pass' if n2['passes'] else 'fail'}."
        ),
        (
            f"- PD1074's {pd['dates']} held-out dates, on activity: "
            + ", ".join(f"{s} {'passes' if p else 'fails'}" for s, p in pd["passes"].items())
            + "."
        ),
        (
            f"- W's minimum, the distance between two halves of real dates (95th percentile): "
            f"{w['fitted']['minimum_95th']:.2f} on the fitted families, "
            f"{w['held_out']['minimum_95th']:.2f} on local search and chemotaxis."
        ),
        "",
        "## C.2 against C.1 (description)",
        "",
    ]
    lines += [f"- {c}." for c in table["contrast"]["conditions"]]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# The guards and the stages
# ---------------------------------------------------------------------------------------------


def part_a_text(text: str) -> str:
    """Return a launch record's Part A: everything before its Part B."""
    return text.split("\n## Part B", 1)[0]


def part_a_commit() -> str | None:
    """Return Part A's commit: the last commit that changed Part A's own text."""
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
    """Return whether the code is Part A's commit's: no change, and no untracked file."""
    commit = part_a_commit()
    if commit is None:
        return False
    paths = [str(p.relative_to(_REPO)) for p in CODE_PATHS]
    if cr._git("diff", "--quiet", commit, "--", *paths).returncode != 0:
        return False
    status = cr._git("status", "--porcelain", "--untracked-files=all", "--", *paths)
    return status.returncode == 0 and not status.stdout.strip()


def part_a() -> bool:
    """Return whether Part A is committed with its bound records and code unchanged."""
    return cr.launch_committed(BOUND, launch_path=LAUNCH_PATH) and code_unchanged()


def _write_once(path: Path, text: str) -> None:
    with path.open("x") as handle:
        handle.write(text)


def _grade(workers: int) -> int:
    if GRADE_PATH.exists():
        print(f"{GRADE_PATH} exists: the grade is read once", file=sys.stderr)
        return 1
    if not part_a():
        print(
            "refusing: launch record 137's Part A is not committed with its records and code",
            file=sys.stderr,
        )
        return 1
    wild = run_jobs(wild_jobs(), workers, mutants=False)
    check = reproduction(wild, _read(C1_FIT_PATH), _read(B4_READ_PATH))
    print(f"reproduced: {check['reproduced']} ({len(check['mismatches'])} mismatches)")
    if not check["reproduced"]:
        _write_once(GRADE_PATH, json.dumps(failed_record(wild, check), indent=1) + "\n")
        print("no mutant was simulated; the reading returns to the maintainer", file=sys.stderr)
        return 1
    full = run_jobs(registered_jobs(), workers)
    record = grade_record(wild, check, full)
    _write_once(GRADE_PATH, json.dumps(record, indent=1) + "\n")
    if record["predictions"] is None:
        print("refusing: the full run's wild type differs from the check's", file=sys.stderr)
        return 1
    for m, p in record["predictions"].items():
        print(f"{m}: {p['prediction']} (passes by restart {p['passes']})")
    return 0


def _table() -> int:
    if TABLE_PATH.exists() or TABLE_MD_PATH.exists():
        print("the table exists: it is written once", file=sys.stderr)
        return 1
    if not cr.launch_committed(BOUND, launch_path=LAUNCH_PATH) or not GRADE_PATH.exists():
        print("refusing: Part A's records changed, or the grade has not run", file=sys.stderr)
        return 1
    grade = _read(GRADE_PATH)
    if not grade.get("predictions"):
        print(
            "refusing: the wild type did not reproduce, so no prediction is read",
            file=sys.stderr,
        )
        return 1
    table = build_table(grade)
    text = markdown(table)
    _write_once(TABLE_PATH, json.dumps(table, indent=1) + "\n")
    _write_once(TABLE_MD_PATH, text)
    print(text)
    return 0


def _smoke(workers: int) -> int:
    """Run the grade path on discarded seeds: the chosen restart on two populations."""
    BUILD.mkdir(parents=True, exist_ok=True)
    jobs = [(CHOSEN, g, SMOKE_BASE) for g in range(SMOKE_POPULATIONS)]
    wild = run_jobs(jobs, min(workers, len(jobs)), mutants=False)
    full = run_jobs(jobs, min(workers, len(jobs)))
    same = repeat_identical(
        [{**r, "base": GRADE_BASE} for r in wild],
        [{**r, "base": GRADE_BASE} for r in full],
    )
    record = {**_header(), "smoke": True, "repeat_identical": same, "populations": full}
    (BUILD / "smoke.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"the wild type repeats identically: {same}")
    return 0 if same else 1


STAGES = ("grade", "table", "smoke")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one stage."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)
    if args.stage == "grade":
        return _grade(args.workers)
    if args.stage == "table":
        return _table()
    return _smoke(args.workers)


if __name__ == "__main__":
    sys.exit(main())
