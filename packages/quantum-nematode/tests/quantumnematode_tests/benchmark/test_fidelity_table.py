"""Tests for C.4's fidelity table (``scripts/analysis/fidelity_table.py``).

Each test names the ``controller-fit`` spec scenario it covers, from change ``add-fidelity-table``.
"""

from __future__ import annotations

import copy
import itertools
import json
import statistics as st
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS = Path(__file__).resolve().parents[5] / "scripts" / "analysis"
sys.path.insert(0, str(_SCRIPTS))
import fidelity_table as ft  # noqa: E402  # pyright: ignore[reportMissingImports]


def _d3(r: int, g: int) -> float:
    return 1.0 + 0.1 * r + 0.01 * g


def _d4(r: int, g: int) -> float:
    return 2.0 + 0.1 * r + 0.01 * g


def _wild(restart: int, g: int, base: int) -> dict[str, Any]:
    return {
        "restart": restart,
        "g": g,
        "base": base,
        "v3": {"D": _d3(restart, g), "wild_type_passes": (g + restart) % 2 == 0},
        "v4": {
            "D": _d4(restart, g),
            "D_fit": 2.5,
            "wild_type_passes": g % 2 == 1,
            "family_passes": {"small_lawn": False},
            "reversal_share": 0.03,
        },
    }


def _population(restart: int, g: int, base: int, passes: dict[str, bool]) -> dict[str, Any]:
    return {
        **_wild(restart, g, base),
        "delta": {
            m: {"roaming_fraction": -0.1, "exits_per_on_lawn_hour": 1.0 + 0.1 * g}
            for m in ft.MUTANTS
        },
        "mutants": {m: {"passes": passes[m], "statistics": {}} for m in ft.MUTANTS},
    }


def _records(passing: dict[int, dict[str, int]]) -> list[dict[str, Any]]:
    """Return registered populations where restart r passes mutant m in its first n populations."""
    records = [
        _population(r, g, ft.GRADE_BASE, {m: g < passing.get(r, {}).get(m, 0) for m in ft.MUTANTS})
        for r in ft.RESTARTS
        for g in range(ft.POPULATIONS)
    ]
    records.extend(
        _population(ft.CHOSEN, g, ft.MORE_BASE, dict.fromkeys(ft.MUTANTS, False))
        for g in range(ft.MORE_POPULATIONS)
    )
    return records


def _wild_records() -> list[dict[str, Any]]:
    return [_wild(r, g, ft.GRADE_BASE) for r in ft.RESTARTS for g in range(ft.POPULATIONS)]


def _committed() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a fit record and a B v4 read that the synthetic wild type reproduces exactly."""
    restarts = [
        {
            "restart": r,
            "population_D": [_d3(r, g) for g in range(ft.POPULATIONS)],
            "population_passes": [(g + r) % 2 == 0 for g in range(ft.POPULATIONS)],
        }
        for r in range(12)
    ]
    b4 = {
        "c1_v3_answer": {
            "base": ft.GRADE_BASE,
            "populations": [
                {"k": g, "D": _d4(ft.CHOSEN, g), "wild_type_passes": g % 2 == 1}
                for g in range(ft.POPULATIONS)
            ],
        },
    }
    return {"restarts": restarts}, b4


# --- The wild type does not reproduce -------------------------------------------------------


def test_the_wild_type_reproduces() -> None:
    """Scenario: The wild type does not reproduce (the control: equal values reproduce)."""
    fit, b4 = _committed()
    assert ft.reproduction(_wild_records(), fit, b4) == {"reproduced": True, "mismatches": []}


@pytest.mark.parametrize(
    ("restart", "g", "what"),
    [(0, 3, "v3 D"), (4, 2, "v3 pass"), (6, 4, "v4 D"), (6, 1, "v4 pass"), (4, 0, "v3 D")],
)
def test_the_wild_type_does_not_reproduce(restart: int, g: int, what: str) -> None:
    """Scenario: The wild type does not reproduce.

    One value off on one population, on any restart, is a mismatch at that population.
    """
    fit, b4 = _committed()
    records = _wild_records()
    rec = next(r for r in records if r["restart"] == restart and r["g"] == g)
    if what == "v3 D":
        rec["v3"]["D"] += 2e-9
    elif what == "v3 pass":
        rec["v3"]["wild_type_passes"] = not rec["v3"]["wild_type_passes"]
    elif what == "v4 D":
        rec["v4"]["D"] += 2e-9
    else:
        rec["v4"]["wild_type_passes"] = not rec["v4"]["wild_type_passes"]
    check = ft.reproduction(records, fit, b4)
    assert not check["reproduced"]
    assert [(m["restart"], m["g"], m["what"]) for m in check["mismatches"]] == [(restart, g, what)]


def test_a_difference_inside_the_tolerance_reproduces() -> None:
    """Scenario: The wild type does not reproduce (a difference below 1e-9 is not a mismatch)."""
    fit, b4 = _committed()
    records = _wild_records()
    records[7]["v3"]["D"] += 5e-10
    assert ft.reproduction(records, fit, b4)["reproduced"]


def test_a_missing_or_undefined_population_does_not_reproduce() -> None:
    """Scenario: The wild type does not reproduce (a missing or undefined D is a mismatch)."""
    fit, b4 = _committed()
    assert not ft.reproduction(_wild_records()[:-1], fit, b4)["reproduced"]
    assert not ft.reproduction([], fit, b4)["reproduced"]
    records = _wild_records()
    records[2]["v3"]["D"] = None
    check = ft.reproduction(records, fit, b4)
    assert [m["what"] for m in check["mismatches"]] == ["v3 D"]
    b4["c1_v3_answer"]["base"] = 1
    assert not ft.reproduction(_wild_records(), fit, b4)["reproduced"]


def test_the_description_populations_are_not_checked() -> None:
    """Scenario: The wild type does not reproduce (only the registered populations are compared)."""
    fit, b4 = _committed()
    records = [*_wild_records(), {**_wild(ft.CHOSEN, 0, ft.MORE_BASE), "v3": {"D": 99.0}}]
    assert ft.reproduction(records, fit, b4)["reproduced"]


def test_a_failed_reproduction_simulates_no_mutant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Scenario: The wild type does not reproduce.

    The grade runs the wild type alone, records the mismatch and no mutant, and ``table`` refuses.
    """
    fit, b4 = _committed()
    calls: list[bool] = []

    def run_jobs(_jobs: list[Any], _workers: int, *, mutants: bool = True) -> list[dict[str, Any]]:
        calls.append(mutants)
        records = _wild_records()
        records[5]["v3"]["D"] += 1.0
        return records

    monkeypatch.setattr(ft, "GRADE_PATH", tmp_path / "grade.json")
    monkeypatch.setattr(ft, "part_a", lambda: True)
    monkeypatch.setattr(ft, "run_jobs", run_jobs)
    monkeypatch.setattr(ft, "_read", lambda path: fit if path == ft.C1_FIT_PATH else b4)
    assert ft.main(["grade"]) == 1
    assert calls == [False]
    record = json.loads((tmp_path / "grade.json").read_text())
    assert record["predictions"] is None
    assert record["populations"] is None
    assert all("mutants" not in r and "delta" not in r for r in record["wild_type"])
    with pytest.raises(ValueError, match="no prediction"):
        ft.build_table(record)


def test_the_grade_runs_the_mutants_after_the_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Scenario: The wild type does not reproduce (the control: a reproduced check reads on)."""
    fit, b4 = _committed()
    calls: list[tuple[int, bool]] = []

    def run_jobs(jobs: list[Any], _workers: int, *, mutants: bool = True) -> list[dict[str, Any]]:
        calls.append((len(jobs), mutants))
        return _records({ft.CHOSEN: {"tph-1": 3}}) if mutants else _wild_records()

    monkeypatch.setattr(ft, "GRADE_PATH", tmp_path / "grade.json")
    monkeypatch.setattr(ft, "part_a", lambda: True)
    monkeypatch.setattr(ft, "run_jobs", run_jobs)
    monkeypatch.setattr(ft, "_read", lambda path: fit if path == ft.C1_FIT_PATH else b4)
    assert ft.main(["grade"]) == 0
    assert calls == [(15, False), (25, True)]
    record = json.loads((tmp_path / "grade.json").read_text())
    assert record["reproduction"]["repeat_identical"] is True
    assert record["predictions"]["tph-1"]["prediction"] == "unidentified"


def test_a_full_run_whose_wild_type_differs_reads_no_prediction() -> None:
    """Scenario: The wild type does not reproduce (the full run must repeat the check's)."""
    full = _records({})
    full[3]["v4"]["D"] += 1e-12
    record = ft.grade_record(_wild_records(), {"reproduced": True, "mismatches": []}, full)
    assert record["reproduction"]["repeat_identical"] is False
    assert record["predictions"] is None


# --- The simulation -------------------------------------------------------------------------


def test_each_mutant_runs_on_the_wild_types_animals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario: A mutant passes in three populations (what a population's reading is made of).

    The wild type runs every assay, each mutant the small lawn alone, all on the same animals, with
    the registered encodings, and each Δ is the mutant's value minus the wild type's, graded on
    v4's bands.
    """
    from quantumnematode.benchmark import references as refs

    def refuse(*_a: Any, **_k: Any) -> None:
        msg = "a population loaded Scheer's tracks"
        raise AssertionError(msg)

    monkeypatch.setattr(refs, "load_scheer_tracks", refuse)
    bands4, bands3 = {"v": 4}, {"v": 3}
    calls: list[dict[str, Any]] = []
    graded: dict[str, Any] = {"D": 1.0, "D_fit": 1.1, "wild_type_passes": True}
    graded |= {"family_passes": {}, "reversal_share": {"value": 0.04}}
    values = iter(
        [
            {"roaming_fraction": 0.20, "exits_per_on_lawn_hour": 0.5},
            {"roaming_fraction": 0.30, "exits_per_on_lawn_hour": 0.6},
            {"roaming_fraction": 0.05, "exits_per_on_lawn_hour": 9.5},
        ],
    )

    def simulate_v2(
        arm: str,
        controller: Any,
        seeds: list[int],
        bands: Any,
        assays: Any = None,
    ) -> tuple[dict[str, Any], None]:
        calls.append(
            {"arm": arm, "params": controller, "seeds": seeds, "bands": bands, "assays": assays},
        )
        return {"small_lawn": len(calls)}, None

    readings: list[tuple[str, Any]] = []

    def mutant_reading(m: str, _d: dict[str, Any], b: Any) -> dict[str, Any]:
        readings.append((m, b))
        return {"passes": True}

    monkeypatch.setattr(ft.c1v4, "_bands", lambda: bands4)
    monkeypatch.setattr(ft.c1v3, "_bands", lambda: bands3)
    monkeypatch.setattr(ft, "_controller", lambda params: params)
    monkeypatch.setattr(ft.bg, "simulate_v2", simulate_v2)
    monkeypatch.setattr(ft.bg, "grade_populations_v4", lambda *_a, **_k: graded)
    monkeypatch.setattr(ft.bg, "grade_populations_v3", lambda *_a, **_k: graded)
    monkeypatch.setattr(ft.bg, "classifier", lambda b: ("clf", b["v"]))
    monkeypatch.setattr(ft.bg, "family_values", lambda *_a: next(values))
    monkeypatch.setattr(ft.c1v4, "mutant_reading", mutant_reading)
    rec = ft.grade_population((4, 2, ft.GRADE_BASE))
    params = ft.c1v3.parameters(ft.answer(4))
    seeds = ft.population_seeds(ft.GRADE_BASE, 2)
    assert [c["arm"] for c in calls] == ["mlp"] * 3
    assert [c["assays"] for c in calls] == [None, ("small_lawn",), ("small_lawn",)]
    assert all(c["seeds"] == seeds for c in calls)
    assert all(c["bands"] is bands4 for c in calls)
    assert calls[0]["params"] == params
    assert calls[1]["params"] == ft.c1v3.mutant_parameters(params, "tph-1")
    assert calls[2]["params"] == ft.c1v3.mutant_parameters(params, "tax-4")
    assert rec["delta"]["tph-1"] == pytest.approx(
        {"roaming_fraction": 0.10, "exits_per_on_lawn_hour": 0.1},
    )
    assert rec["delta"]["tax-4"] == pytest.approx(
        {"roaming_fraction": -0.15, "exits_per_on_lawn_hour": 9.0},
    )
    assert readings == [("tph-1", bands4), ("tax-4", bands4)]
    assert rec["v4"]["reversal_share"] == 0.04


def test_the_wild_type_alone_simulates_no_mutant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario: The wild type does not reproduce (the check's populations hold no mutant)."""
    calls: list[Any] = []
    graded: dict[str, Any] = {"D": 1.0, "D_fit": 1.1, "wild_type_passes": True}
    graded |= {"family_passes": {}, "reversal_share": None}

    def simulate_v2(*a: Any, **k: Any) -> tuple[dict[str, Any], None]:
        calls.append((a, k))
        return {"small_lawn": None}, None

    monkeypatch.setattr(ft.c1v4, "_bands", dict)
    monkeypatch.setattr(ft.c1v3, "_bands", dict)
    monkeypatch.setattr(ft, "_controller", lambda params: params)
    monkeypatch.setattr(ft.bg, "simulate_v2", simulate_v2)
    monkeypatch.setattr(ft.bg, "grade_populations_v4", lambda *_a, **_k: graded)
    monkeypatch.setattr(ft.bg, "grade_populations_v3", lambda *_a, **_k: graded)
    rec = ft.grade_population((6, 0, ft.GRADE_BASE), mutants=False)
    assert len(calls) == 1
    assert "mutants" not in rec
    assert "delta" not in rec


def test_the_body_is_the_fitted_one() -> None:
    """Scenario: A mutant passes in three populations (on the body C.1 (v3) was fitted on)."""
    assert ft.ARM == "mlp"
    names = sorted(p.name for p in ft.MLP_CONFIGS)
    stem = "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1_primitives_assay_"
    assays = (
        "chemotaxis_chen",
        "food_removal_calhoun",
        "food_removal_lopezcruz",
        "scheer_lawn",
        "uniform_lawn_flavell",
        "uniform_lawn_scheer",
    )
    assert names == sorted(f"{stem}{a}.yml" for a in assays)


# --- The rule -------------------------------------------------------------------------------


def test_restarts_disagree_on_a_mutant() -> None:
    """Scenario: Restarts disagree on a mutant."""
    records = _records({0: {"tph-1": 3}, 4: {"tph-1": 3}, 6: {"tph-1": 2}})
    p = ft.prediction(records, "tph-1")
    assert p["restart_passes"] == {"0": True, "4": True, "6": False}
    assert p["prediction"] == "unidentified"


def test_a_mutant_passes_in_three_populations() -> None:
    """Scenario: A mutant passes in three populations."""
    records = _records({0: {"tax-4": 3}, 4: {"tax-4": 4}, 6: {"tax-4": 3}})
    p = ft.prediction(records, "tax-4")
    assert p["restart_passes"] == {"0": True, "4": True, "6": True}
    assert p["passes"] == {"0": 3, "4": 4, "6": 3}
    assert p["prediction"] == "pass"


def test_every_restart_failing_predicts_a_fail() -> None:
    """Scenario: A mutant passes in three populations (two of five is a failing restart)."""
    records = _records({0: {"tax-4": 2}, 4: {"tax-4": 0}, 6: {"tax-4": 1}})
    assert ft.prediction(records, "tax-4")["prediction"] == "fail"


def test_the_description_populations_do_not_enter_the_rule() -> None:
    """Scenario: A mutant passes in three populations (only the registered five are counted)."""
    records = _records({})
    for r in records:
        if r["base"] == ft.MORE_BASE:
            r["mutants"]["tph-1"]["passes"] = True
    assert ft.prediction(records, "tph-1")["prediction"] == "fail"
    more = ft.more_populations(records)
    assert more["more"]["tph-1"] == ft.MORE_POPULATIONS
    assert more["populations"] == 15
    assert more["tph-1"] == pytest.approx(10 / 15)


def test_a_restart_verdict_needs_five_populations() -> None:
    """Scenario: A mutant passes in three populations (fewer than five populations is refused)."""
    assert ft.restart_verdict([True, True, True, False, False])
    assert not ft.restart_verdict([True, True, False, False, False])
    with pytest.raises(ValueError, match="5 populations"):
        ft.restart_verdict([True, True, True])


def test_the_rule_reads_restarts_zero_four_and_six() -> None:
    """Scenario: Restarts disagree on a mutant (the rule's restarts are v3's wild-type passers)."""
    fit = json.loads(ft.C1_FIT_PATH.read_text())
    passing = tuple(r["restart"] for r in fit["restarts"] if r["passes_wild_type"])
    assert passing == ft.RESTARTS
    assert fit["chosen_restart"] == ft.CHOSEN


def test_the_answer_is_the_committed_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario: Restarts disagree on a mutant (each answer is read from the fit record)."""
    fit = json.loads(ft.C1_FIT_PATH.read_text())
    assert ft.answer(6) == fit["restarts"][6]["answer"]
    broken = copy.deepcopy(fit)
    broken["restarts"][6]["restart"] = 5
    monkeypatch.setattr(ft, "_read", lambda _path: broken)
    with pytest.raises(ValueError, match="not at its index"):
        ft.answer(6)


# --- The table ------------------------------------------------------------------------------


def _grade_with_predictions() -> dict[str, Any]:
    records = _records({ft.CHOSEN: {"tph-1": 1}})
    return {
        "populations": records,
        "reproduction": {"reproduced": True, "mismatches": []},
        "predictions": {m: ft.prediction(records, m) for m in ft.MUTANTS},
        "more_populations": ft.more_populations(records),
    }


def _table_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ft, "TABLE_PATH", tmp_path / "table.json")
    monkeypatch.setattr(ft, "TABLE_MD_PATH", tmp_path / "table.md")
    monkeypatch.setattr(ft, "GRADE_PATH", tmp_path / "grade.json")


def test_c1s_held_out_cell_is_not_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Scenario: C.1's held-out cell.

    The table reads no Scheer track, C.1's cell is "not read", and C.2's is copied from its fit.
    """
    from quantumnematode.benchmark import references as refs

    def refuse(*_a: Any, **_k: Any) -> None:
        msg = "the table loaded Scheer's tracks"
        raise AssertionError(msg)

    monkeypatch.setattr(refs, "load_scheer_tracks", refuse)
    _table_paths(monkeypatch, tmp_path)
    (tmp_path / "grade.json").write_text(json.dumps(_grade_with_predictions()))
    monkeypatch.setattr(ft.cr, "launch_committed", lambda *_a, **_k: True)
    monkeypatch.setattr(ft, "part_a_commit", lambda: "a1")
    assert ft.main(["table"]) == 0
    table = json.loads((tmp_path / "table.json").read_text())
    c1, c2, c3 = table["controllers"]
    assert c1["held_out"] == "not read"
    c2_fit = json.loads(ft.C2_FIT_PATH.read_text())
    for name, h in c2_fit["held_out"].items():
        assert c2["held_out"][name] == {
            "dates": h["dates"],
            "answer_median": h["answer"]["median"],
            "real_halves_p95": h["real_halves"]["p95"],
            "band_reference_p95": h["band_reference"]["p95"],
        }
    assert c3 == {"controller": "C.3", "status": "not run in Phase 10"}
    assert (tmp_path / "table.md").read_text() == ft.markdown(table)


def test_the_table_rows_and_kinds() -> None:
    """Scenario: C.1's held-out cell (the rows and each mutant's kind)."""
    table = ft.build_table(_grade_with_predictions())
    c1, c2, _ = table["controllers"]
    assert {m: c1["mutants"][m]["kind"] for m in ("tph-1", "pdfr-1", "tax-4")} == {
        "tph-1": "predicted",
        "pdfr-1": "not encodable",
        "tax-4": "predicted",
    }
    assert c2["mutants"] == {
        "tph-1": {"kind": "fitted", "passes": 0},
        "pdfr-1": {"kind": "fitted", "passes": 5},
        "tax-4": {"kind": "fitted", "passes": 5},
    }
    assert c2["wild_type_v4"]["passing"] == 5
    assert c1["wild_type_v4"]["passing"] == 0
    assert len(table["negatives"]) == 8
    assert all(n["fails_wild_type_in"] == 5 for n in table["negatives"])
    assert table["contrast"]["reading"].startswith("description")
    assert table["description_rows"][0]["passing"] == {"wild_type": 0, "tph-1": 5, "tax-4": 0}
    text = ft.markdown(table)
    cell = "| tph-1 | predicted: **fail** (r0 0, r4 0, r6 1 of 5); the wild type passes v4 in 0"
    assert cell in text
    pilot = table["description_rows"][0]
    line = (
        f"- {pilot['controller']}: stopped at generation {pilot['stop']['generation']} by "
        f"{pilot['stop']['by']}; passes the wild type in 0 of 5, tph-1 in 5 of 5, tax-4 in 0 of 5; "
        f"D by population {', '.join(f'{v:.2f}' for v in pilot['D'])}."
    )
    assert line in text
    assert text.index("## Description rows") < text.index(line)
    assert text.index(line) < text.index("## C.2 against C.1 (description)")


def test_the_table_copies_each_recorded_cell() -> None:
    """Scenario: C.1's held-out cell (each cell is its record's value)."""
    grade = _grade_with_predictions()
    table = ft.build_table(grade)
    b4 = json.loads(ft.B4_READ_PATH.read_text())
    c2_fit = json.loads(ft.C2_FIT_PATH.read_text())
    controls = json.loads(ft.B4_CONTROLS_PATH.read_text())
    c1, c2, _ = table["controllers"]
    pops = b4["c1_v3_answer"]["populations"]
    first = sorted(
        (g for g in c2_fit["grades"] if g["population"] < 5),
        key=lambda g: g["population"],
    )
    w1, w2 = c1["wild_type_v4"], c2["wild_type_v4"]
    assert w1["D"] == [p["D"] for p in pops]
    assert w1["D_fit"] == [p["D_fit"] for p in pops]
    assert w1["reversal_share"] == [p["reversal_share"]["value"] for p in pops]
    assert w1["by_population"] == b4["c1_v3_answer"]["v4_population_passes"]
    for family in pops[0]["family_passes"]:
        assert w1["family_passes"][family] == sum(p["family_passes"][family] for p in pops)
        assert w2["family_passes"][family] == sum(g["family_passes"][family] for g in first)
    assert w2["D"] == [g["D"] for g in first]
    assert w2["D_fit"] == [g["D_fit"] for g in first]
    assert c2["verdict"] == c2_fit["verdict"]
    assert c2["pass_rates"] == c2_fit["pass_rates"]
    assert c2["recovery"]["gain"] == c2_fit["recovery_gate"]["gain"]
    leaving = c1["tph1_leaving_reported"]
    assert leaving["exits_per_on_lawn_hour_delta"] == [1.0 + 0.1 * g for g in range(5)]
    reading = controls["positive_controls"]["units"]["mutant/tph-1"]["statistics"]
    assert leaving["band_centre"] == reading["exits_per_on_lawn_hour"]["centre"]
    assert leaving["held_out_value"] == reading["exits_per_on_lawn_hour"]["value"]
    assert c1["more_populations"] == grade["more_populations"]
    for neg, row in zip(b4["negatives"].values(), table["negatives"], strict=True):
        assert row["D_median"] == st.median(p["D"] for p in neg["populations"])
    assert table["contrast"]["median_D"] == {
        "C.2": st.median(g["D"] for g in first),
        "C.1 (v3)": st.median(p["D"] for p in pops),
    }
    assert table["references"]["w_minimum"] == b4["w_minimum"]
    text = ft.markdown(table)
    median_line = (
        f"| Median D (D_fit) | {st.median(p['D'] for p in pops):.2f} "
        f"({st.median(p['D_fit'] for p in pops):.2f}) | {st.median(g['D'] for g in first):.2f} "
        f"({st.median(g['D_fit'] for g in first):.2f}) |"
    )
    assert median_line in text
    assert f"{b4['w_minimum']['fitted']['minimum_95th']:.2f} on the fitted families" in text
    assert "N2's 26 unread dates, on activity in 2 halves: pass." in text


def test_a_recount_that_disagrees_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario: A bound record changed (a record whose counts disagree is refused)."""
    c2_fit = json.loads(ft.C2_FIT_PATH.read_text())
    c2_fit["verdict"]["mutant_passes"]["tax-4"] = 4
    real = ft._read
    monkeypatch.setattr(ft, "_read", lambda p: c2_fit if p == ft.C2_FIT_PATH else real(p))
    with pytest.raises(ValueError, match=r"C\.2's tax-4 passes"):
        ft.build_table(_grade_with_predictions())


# --- The guards -----------------------------------------------------------------------------


def test_the_grade_runs_before_part_a(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Scenario: The grade runs before Part A."""
    monkeypatch.setattr(ft, "GRADE_PATH", tmp_path / "grade.json")
    monkeypatch.setattr(ft, "LAUNCH_PATH", tmp_path / "launch.md")
    monkeypatch.setattr(ft, "run_jobs", lambda *_a, **_k: pytest.fail("the grade simulated"))
    assert ft.main(["grade"]) == 1
    assert not (tmp_path / "grade.json").exists()


def test_the_grade_refuses_changed_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Scenario: The grade runs before Part A (changed code refuses like an uncommitted record)."""
    monkeypatch.setattr(ft, "GRADE_PATH", tmp_path / "grade.json")
    monkeypatch.setattr(ft.cr, "launch_committed", lambda *_a, **_k: True)
    monkeypatch.setattr(ft, "part_a_commit", lambda: "a1")
    monkeypatch.setattr(ft, "code_unchanged", lambda: False)
    monkeypatch.setattr(ft, "run_jobs", lambda *_a, **_k: pytest.fail("the grade simulated"))
    assert ft.main(["grade"]) == 1


def test_the_grade_record_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Scenario: The grade record exists."""
    path = tmp_path / "grade.json"
    path.write_text("{}")
    monkeypatch.setattr(ft, "GRADE_PATH", path)
    monkeypatch.setattr(ft, "part_a", lambda: True)
    monkeypatch.setattr(ft, "run_jobs", lambda *_a, **_k: pytest.fail("the grade simulated"))
    assert ft.main(["grade"]) == 1
    assert path.read_text() == "{}"


def test_a_bound_record_changed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Scenario: A bound record changed."""
    _table_paths(monkeypatch, tmp_path)
    (tmp_path / "grade.json").write_text(json.dumps(_grade_with_predictions()))
    monkeypatch.setattr(ft.cr, "launch_committed", lambda *_a, **_k: False)
    monkeypatch.setattr(ft, "part_a_commit", lambda: "a1")
    monkeypatch.setattr(ft, "code_unchanged", lambda: True)
    monkeypatch.setattr(ft, "build_table", lambda _g: pytest.fail("the table was built"))
    assert ft.main(["table"]) == 1
    assert ft.part_a() is False
    assert not (tmp_path / "table.json").exists()
    assert not (tmp_path / "table.md").exists()


def test_the_table_refuses_a_grade_without_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Scenario: The wild type does not reproduce (``table`` refuses the record)."""
    grade = _grade_with_predictions()
    grade["predictions"] = None
    _table_paths(monkeypatch, tmp_path)
    (tmp_path / "grade.json").write_text(json.dumps(grade))
    monkeypatch.setattr(ft.cr, "launch_committed", lambda *_a, **_k: True)
    monkeypatch.setattr(ft, "part_a_commit", lambda: "a1")
    assert ft.main(["table"]) == 1
    assert not (tmp_path / "table.json").exists()
    assert not (tmp_path / "table.md").exists()


def test_the_table_is_written_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Scenario: The grade record exists (the table, like the grade, is written once)."""
    _table_paths(monkeypatch, tmp_path)
    (tmp_path / "grade.json").write_text(json.dumps(_grade_with_predictions()))
    (tmp_path / "table.md").write_text("earlier")
    monkeypatch.setattr(ft.cr, "launch_committed", lambda *_a, **_k: True)
    monkeypatch.setattr(ft, "part_a_commit", lambda: "a1")
    assert ft.main(["table"]) == 1
    assert not (tmp_path / "table.json").exists()
    assert (tmp_path / "table.md").read_text() == "earlier"
    path = tmp_path / "once.json"
    ft._write_once(path, "a")
    with pytest.raises(FileExistsError):
        ft._write_once(path, "b")
    assert path.read_text() == "a"


class _Git:
    """A fake ``git`` for the pin and the code check."""

    def __init__(self, history: list[tuple[str, str]], *, diff: int = 0, status: str = "") -> None:
        self.history = history
        self.diff = diff
        self.status = status
        self.diffed: list[str] = []

    def __call__(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        if args[0] == "log":
            out = "\n".join(c for c, _ in reversed(self.history))
            return subprocess.CompletedProcess(args, 0, out.encode(), b"")
        if args[0] == "show":
            commit = args[1].split(":", 1)[0]
            text = dict(self.history)[commit]
            return subprocess.CompletedProcess(args, 0, text.encode(), b"")
        if args[0] == "diff":
            self.diffed.append(args[2])
            return subprocess.CompletedProcess(args, self.diff, b"", b"")
        if args[0] == "status":
            return subprocess.CompletedProcess(args, 0, self.status.encode(), b"")
        raise AssertionError(args)


def test_the_code_is_checked_at_part_as_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario: The grade runs before Part A (the code is compared with Part A's commit)."""
    git = _Git([("a1", "# Part A"), ("b2", "# Part A\n## Part B: later")])
    monkeypatch.setattr(ft.cr, "_git", git)
    assert ft.part_a_commit() == "a1"
    assert ft.code_unchanged()
    assert git.diffed == ["a1"]
    monkeypatch.setattr(ft.cr, "_git", _Git([("a1", "# Part A"), ("c3", "# Part A, amended")]))
    assert ft.part_a_commit() == "c3"


def test_the_bound_records_are_checked_at_part_as_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Scenario: A bound record changed (the records are compared at the code check's pin)."""
    pins: list[str | None] = []

    def committed(*_a: object, commit: str | None = None, **_k: object) -> bool:
        pins.append(commit)
        return True

    monkeypatch.setattr(ft.cr, "launch_committed", committed)
    monkeypatch.setattr(ft, "part_a_commit", lambda: "a1")
    monkeypatch.setattr(ft, "code_unchanged", lambda: True)
    assert ft.part_a()
    _table_paths(monkeypatch, tmp_path)
    assert ft.main(["table"]) == 1  # no grade record yet
    assert pins == ["a1", "a1"]
    monkeypatch.setattr(ft, "part_a_commit", lambda: None)
    assert not ft.records_unchanged()
    assert pins == ["a1", "a1"]


@pytest.mark.parametrize(("diff", "status"), [(1, ""), (0, "?? scripts/analysis/extra.py\n")])
def test_changed_or_untracked_code_refuses(
    monkeypatch: pytest.MonkeyPatch,
    diff: int,
    status: str,
) -> None:
    """Scenario: The grade runs before Part A (a diff or an untracked file refuses)."""
    monkeypatch.setattr(ft.cr, "_git", _Git([("a1", "# Part A")], diff=diff, status=status))
    assert not ft.code_unchanged()
    monkeypatch.setattr(ft.cr, "launch_committed", lambda *_a, **_k: True)
    assert ft.part_a() is False


def test_the_guarded_code_includes_the_script_and_its_imports() -> None:
    """Scenario: The grade runs before Part A (the script and its imports are guarded)."""
    names = {p.name for p in ft.CODE_PATHS}
    for name in ("fidelity_table.py", "c1_v3_fit.py", "c1_v4_fit.py", "c1_fit.py", "c1b_fit.py"):
        assert name in names
    assert all(p.exists() for p in ft.CODE_PATHS)


def test_the_bound_records_include_every_source() -> None:
    """Scenario: A bound record changed (every record the table reads is bound)."""
    bound = set(ft.BOUND)
    for path in (
        ft.C1_FIT_PATH,
        ft.C1_RECOVERY_PATH,
        ft.C2_FIT_PATH,
        ft.B4_READ_PATH,
        ft.B4_CONTROLS_PATH,
        ft.PILOT_PATH,
        ft.ENCODINGS_PATH,
        ft.BANDS_PATH,
        ft.BANDS_V3_PATH,
        ft.CLASSIFIER_PATH,
        ft.PRIMITIVES_PATH,
    ):
        assert path in bound
        assert path.exists()
    assert len(ft.MLP_CONFIGS) == 6


# --- Seeds ----------------------------------------------------------------------------------

# The registered and probe blocks of C.1 (v3), C.1's v4 refit and C.2, from the tracker's seed
# table and the fit scripts: [start, end) animal-seed intervals.
OTHER_BLOCKS = [
    (500_000_000, 524_000_000),  # C.1 (v3)'s fit, answers at 10,000·r + 8,000 + q, r < 12
    (526_000_000, 526_012_000),  # C.1 (v3)'s selection, 5r + s < 60
    (530_000_000, 538_000_000),  # C.1 (v3)'s recovery fits
    (540_000_000, 543_000_000),  # C.1 (v3)'s truth, prior draws and centre
    (641_000_000, 641_003_200),  # Logbook 136's tax-4 probe
    (647_000_000, 700_000_000),  # C.1's v4 refit and its probes (Logbook 136)
    (700_000_000, 1_600_000_000),  # C.2's probes and registered fit (Logbooks 126-133)
    (99_840_000, 99_870_000),  # earlier smokes and checks in the 99.8 M range
]


def test_every_block_is_enumerated() -> None:
    """Scenario: Every block is enumerated."""
    ours = {
        "registered": (ft.GRADE_BASE, ft.GRADE_BASE + ft.ANIMALS * ft.POPULATIONS),
        "more": (ft.MORE_BASE, ft.MORE_BASE + ft.ANIMALS * ft.MORE_POPULATIONS),
        "smoke": (ft.SMOKE_BASE, ft.SMOKE_BASE + ft.ANIMALS * ft.SMOKE_POPULATIONS),
    }
    assert ours["registered"] == (525_000_000, 525_001_000)
    for lo, hi in ours.values():
        for start, end in OTHER_BLOCKS:
            assert hi <= start or lo >= end, (lo, start)
    blocks = sorted(ours.values())
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(blocks))


def test_the_jobs_are_the_registered_ones() -> None:
    """Scenario: Every block is enumerated (each restart on its five, the chosen on ten more)."""
    assert ft.wild_jobs() == [(r, g, ft.GRADE_BASE) for r in (0, 4, 6) for g in range(5)]
    jobs = ft.registered_jobs()
    assert jobs[:15] == ft.wild_jobs()
    assert jobs[15:] == [(6, g, ft.MORE_BASE) for g in range(10)]
    assert ft.population_seeds(ft.GRADE_BASE, 1)[:2] == [525_000_200, 525_000_201]
    assert len(ft.population_seeds(ft.MORE_BASE, 9)) == 200


def test_a_restarts_statistics_must_pass_in_the_same_population() -> None:
    """Scenario: A mutant passes in three populations (both statistics pass in the same one)."""
    bands = ft.c1v4._bands()
    centre = ft.c1v4.band_truth(bands)["delta"]["tax-4"]
    good = dict(centre)
    roaming_only = {**centre, "exits_per_on_lawn_hour": centre["exits_per_on_lawn_hour"] + 14.0}
    exits_only = {**centre, "roaming_fraction": centre["roaming_fraction"] + 0.5}
    assert ft.c1v4.mutant_reading("tax-4", good, bands)["passes"]
    split = [roaming_only, roaming_only, good, exits_only, exits_only]
    records = _records({})
    for rec in records:
        if rec["restart"] == ft.CHOSEN and rec["base"] == ft.GRADE_BASE:
            rec["mutants"]["tax-4"] = ft.c1v4.mutant_reading("tax-4", split[rec["g"]], bands)
    for delta in (roaming_only, exits_only):
        reading = ft.c1v4.mutant_reading("tax-4", delta, bands)
        assert not reading["passes"]
        assert any(s["passes"] for s in reading["statistics"].values() if s["graded"])
    assert ft.restart_passes(records, ft.CHOSEN, "tax-4") == [False, False, True, False, False]
    assert ft.prediction(records, "tax-4")["restart_passes"]["6"] is False
