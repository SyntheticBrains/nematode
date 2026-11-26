"""C.1 refitted on benchmark v4 from a warm start: its model, objective, rules and seeds.

Covers the controller-fit spec's scenarios:

- "The start reproduces C.1 (v3)'s answer";
- "A restart above the tightened level does not stop";
- "The sigma guard reads the registered sigma0";
- "The formula is C.2's on two mutants";
- "tax-4 keeps the on-food dwelling rate";
- "The go rule fails", "The pilot does not stop", "The projected cost is too high" and "The cap
  from a short pilot";
- "A mutant fails the verdict", "A capped restart is not chosen" and "Selection counts the full
  criterion";
- "The leave-one-out gates nothing";
- "Restarts disagree on a mutant";
- "The fit refuses before the recovery passes", "The code changes after Part A", "Part B does not
  move the pin" and "The maintainer decides to run despite the gate";
- "Every block is enumerated";

the fit-recovery spec's "The gain is read from the start", "The start is too close to the truth",
"A statistic's definedness flips", "The gate passes on the mutants alone" and "The recovery falls
short"; and the benchmark-controls
spec's "A registered fit reads the held-out dates once", for the smoke's stand-ins.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import c1_v3_fit as c1v3  # noqa: E402  # pyright: ignore[reportMissingImports]
import c1_v4_fit as fit  # noqa: E402  # pyright: ignore[reportMissingImports]
import cma_recovery_v2 as v2  # noqa: E402  # pyright: ignore[reportMissingImports]


@pytest.fixture(scope="module")
def bands() -> dict[str, Any]:
    """Return v4's bands."""
    return fit._bands()


def _grade(*, wild: bool, tph1: bool = True, tax4: bool = True) -> dict[str, Any]:
    return {
        "wild_type_passes": wild,
        "mutants": {"tph-1": {"passes": tph1}, "tax-4": {"passes": tax4}},
    }


# ---------------------------------------------------------------------------------------------
# The model and the start
# ---------------------------------------------------------------------------------------------


def test_the_model_has_twenty_six_parameters() -> None:
    """The unit box maps to C.1 (v3)'s 25 parameters and the on-food dwelling rate, log-scaled."""
    params = fit.parameters(np.full(fit.DIMENSION, 0.5))
    assert fit.DIMENSION == 26
    assert len(params) == 26
    assert params["r_dwell_on_food"] == pytest.approx(math.sqrt(1e-3 * 0.5))
    assert fit.parameters(np.zeros(26))["r_dwell_on_food"] == pytest.approx(1e-3)
    assert fit.parameters(np.ones(26))["r_dwell_on_food"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="26 dimensions"):
        fit.parameters(np.full(25, 0.5))


def test_the_start_reproduces_c1_v3s_answer() -> None:
    """Covers "The start reproduces C.1 (v3)'s answer": 25 coordinates as fitted, then 0.5."""
    record = json.loads(fit.START_PATH.read_text())
    answer = record["restarts"][record["chosen_restart"]]["answer"]
    start = fit.start_point()
    assert start[:25] == [float(v) for v in answer]
    assert start[25] == 0.5
    v3 = c1v3.parameters(answer)
    v4 = fit.parameters(start)
    assert {k: v4[k] for k in v3} == v3
    strategy = fit.start_warm(1)
    assert list(strategy.x0) == start
    assert strategy.sigma0 == fit.SIGMA0 == 0.1


def test_tax4_keeps_the_on_food_dwelling_rate() -> None:
    """Covers "tax-4 keeps the on-food dwelling rate"; tph-1 keeps it too."""
    params = fit.parameters(np.full(26, 0.7))
    tax4 = fit.mutant_parameters(params, "tax-4")
    assert tax4 is not None
    for name in ("kappa_weathervane", "kappa_off_food", "beta_reversal", "beta_omega"):
        assert tax4[name] == 0.0
    assert tax4["r_dwell_on_food"] == params["r_dwell_on_food"]
    tph1 = fit.mutant_parameters(params, "tph-1")
    assert tph1 is not None
    assert tph1["a_intake"] == tph1["b_intake"] == 0.0
    assert tph1["r_dwell_on_food"] == params["r_dwell_on_food"]


def test_the_probe_truth_is_on_this_parameterisation() -> None:
    """Logbook 125's answer, read by the separation probe, is on this model's 26 coordinates."""
    record = json.loads(fit.PROBE_TRUTH_PATH.read_text())
    assert record["reaches"] is True
    assert fit.probe_truth() == [float(v) for v in record["answer"]]
    probe = record["parameters"]
    mapped = fit.parameters(fit.probe_truth())
    assert mapped == pytest.approx(probe)


# ---------------------------------------------------------------------------------------------
# The stop and the cap
# ---------------------------------------------------------------------------------------------


def test_a_restart_above_the_tightened_level_does_not_stop() -> None:
    """Covers "A restart above the tightened level does not stop"."""
    flat = [5.0] * 60
    assert fit.stop(flat, 0.05) is True
    raised = [*flat[:30], 150.0, *flat[31:]]
    assert fit.stop(raised, 0.05) is False
    undefined = [*flat[:30], 1_003.0, *flat[31:]]
    assert fit.stop(undefined, 0.05) is False
    assert fit.stop([5.0] * 39, 0.05) is False


def test_the_sigma_guard_reads_the_registered_sigma0() -> None:
    """Covers "The sigma guard reads the registered sigma0": 0.15 keeps it running, 0.1 not."""
    flat = [5.0] * 60
    assert fit.stop(flat, 0.15) is False
    assert fit.stop(flat, 0.1) is True
    falling = [10.0 - 0.1 * n for n in range(60)]
    assert fit.stop(falling, 0.05) is False


def test_the_cap_from_a_short_pilot() -> None:
    """Covers "The cap from a short pilot": 1.5 x the stop, up to 10, within [300, 500]."""
    assert fit.registered_cap(104) == 300
    assert fit.registered_cap(201) == 310
    assert fit.registered_cap(221) == 340
    assert fit.registered_cap(400) == 500
    # The answers sit above every generation's index under the largest cap.
    assert fit.ANSWER_OFFSET > 16 * (fit.CAP_CEILING - 1) + 15


# ---------------------------------------------------------------------------------------------
# The objective and the recovery's distance
# ---------------------------------------------------------------------------------------------


def _record(bands: dict[str, Any], shift: float, delta: dict[str, Any]) -> dict[str, Any]:
    truth = fit.band_truth(bands)
    statistics = {
        k: v + shift * (1 + i % 3) for i, (k, v) in enumerate(truth["statistics"].items())
    }
    return {"statistics": statistics, "delta": delta}


def test_the_formula_is_c2s_on_two_mutants(bands: dict[str, Any]) -> None:
    """Covers "The formula is C.2's on two mutants": the score equals C.2's on tph-1 and tax-4."""
    import c2_fit  # pyright: ignore[reportMissingImports]

    truth = fit.band_truth(bands)
    c2_truth = c2_fit.band_truth(bands)
    for shift, delta in (
        (0.0, truth["delta"]),
        (0.01, {"tph-1": {"roaming_fraction": 0.2}, "tax-4": {"roaming_fraction": -0.1}}),
        (0.02, {"tph-1": {"roaming_fraction": None}, "tax-4": {"exits_per_on_lawn_hour": 3.0}}),
    ):
        record = _record(bands, shift, delta)
        ours = fit.Objective(truth, bands)(record)
        theirs = c2_fit.Objective(c2_truth, bands, families=None, mutants=fit.MUTANTS).distance(
            record,
        )
        assert ours == pytest.approx(theirs)
    assert fit.Objective(truth, bands)({**truth}) == pytest.approx(0.0)


def test_an_undefined_mutant_difference_counts_at_the_cap(bands: dict[str, Any]) -> None:
    """A missing or undefined Δ scores at the z cap, so it never undercuts a defined one."""
    truth = fit.band_truth(bands)
    scale = fit.mutant_scales(bands)["tph-1"]["roaming_fraction"]
    record = {"delta": {"tph-1": {"roaming_fraction": None}}}
    assert fit.mutant_z(truth, record, scale, "tph-1", "roaming_fraction") == 10.0
    assert fit.mutant_z(truth, {}, scale, "tph-1", "roaming_fraction") == 10.0


def test_the_recovery_distance_weighs_the_mutants_at_half(bands: dict[str, Any]) -> None:
    """The recovery's distance is the capped D and each mutant's RMS capped z, combined at half."""
    truth = fit.band_truth(bands)
    record = _record(bands, 0.01, {"tph-1": {"roaming_fraction": 0.2}, "tax-4": {}})
    scales = v2.band_scales_v2(bands)
    family_map = {f: tuple(s) for f, s in bands["families"].items()}
    wild = v2.predictive_distance_v2(truth, record, scales, family_map, capped=True)
    mscales = fit.mutant_scales(bands)
    tph1 = fit.mutant_z(
        truth,
        record,
        mscales["tph-1"]["roaming_fraction"],
        "tph-1",
        "roaming_fraction",
    )
    tax4 = math.sqrt((10.0**2 + 10.0**2) / 2)
    expected = math.sqrt((6 * wild**2 + 3 * (tph1**2 + tax4**2)) / (6 + 3 * 2))
    assert fit.recovery_distance(truth, record) == pytest.approx(expected)


def test_the_mutant_reading_is_c2s(bands: dict[str, Any]) -> None:
    """A mutant's grade on the encoding table's statistics is C.2's, statistic for statistic."""
    import c2_fit  # pyright: ignore[reportMissingImports]

    for mutant in fit.MUTANTS:
        centres = {s: b.centre for s, b in fit.bg.mutant_bands(bands, mutant, 200).items()}
        for delta in (centres, {s: v + 0.5 for s, v in centres.items()}, {}):
            assert fit.mutant_reading(mutant, delta, bands) == c2_fit.mutant_reading(
                mutant,
                delta,
                bands,
            )


# ---------------------------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------------------------


def test_the_projected_cost_is_too_high() -> None:
    """Covers "The projected cost is too high": 15 restarts x 300 x 40 s is 50 h, above 48."""
    projected = fit.projected_hours(300, 40.0)
    assert projected["registered"] == pytest.approx(50.0)
    assert projected["registered"] > fit.COST_BAR_HOURS
    assert fit.projected_hours(200, 38.0)["registered"] == pytest.approx(15 * 200 * 38 / 3600)


def test_the_go_rule_fails() -> None:
    """Covers "The go rule fails": tph-1 in 2 of 5 stops the refit, as does a short record."""
    grades = [_grade(wild=True, tph1=n < 2) for n in range(5)]
    assert fit.three_of_five(grades)["passes"] is False
    assert fit.three_of_five([_grade(wild=True)] * 5)["passes"] is True
    assert fit.three_of_five([_grade(wild=True)] * 4)["passes"] is False


def test_a_mutant_fails_the_verdict() -> None:
    """Covers "A mutant fails the verdict": the wild type in 4 of 5 and tax-4 in 2 of 5 fail."""
    grades = [_grade(wild=n < 4, tax4=n < 2) for n in range(5)] + [_grade(wild=True)] * 10
    verdict = fit.three_of_five(grades)
    assert verdict["passing"] == {"wild_type": 4, "tph-1": 5, "tax-4": 2}
    assert verdict["passes"] is False
    assert fit.pass_rates(grades)["wild_type"] == pytest.approx(14 / 15)


def test_selection_counts_the_full_criterion() -> None:
    """Covers "Selection counts the full criterion" and "A capped restart is not chosen"."""
    a = [_grade(wild=True, tph1=n >= 3) for n in range(5)]
    b = [_grade(wild=True)] * 3 + [_grade(wild=False)] * 2
    scores = {0: 1.0, 1: 2.0, 2: 0.5}
    assert fit.chosen_restart({0: a, 1: b}, scores) == 1
    assert fit.chosen_restart({0: b, 1: b}, scores) == 0
    assert fit.chosen_restart({1: b, 2: b}, scores) == 2
    assert fit.chosen_restart({}, scores) is None
    histories = [{"stop": {"by": "cap"}}, {"stop": {"by": "plateau"}}]
    assert fit._plateaued(histories) == [1]


def test_restarts_disagree_on_a_mutant() -> None:
    """Covers "Restarts disagree on a mutant": the row is unidentified, whatever the chosen's."""
    import c1_fit  # pyright: ignore[reportMissingImports]

    agree = [_grade(wild=True)] * 5
    fail_tph1 = [_grade(wild=True, tph1=False)] * 5
    fail_wild = [_grade(wild=False)] * 5
    rows = fit.mutant_rows([agree, agree, fail_tph1, fail_wild], chosen=0)
    assert rows["wild_type_restarts"] == [0, 1, 2]
    assert rows["tph-1"]["row"] == "unidentified"
    assert rows["tax-4"]["row"] == "pass"
    none = fit.mutant_rows([fail_wild, fail_wild], chosen=1)
    assert none["wild_type_failure_stated"] is True
    assert none["tph-1"]["row"] == "pass"
    for verdicts, chosen in (([], True), ([], False), ([True, False], True), ([False], True)):
        assert fit.mutant_verdict(verdicts, chosen=chosen) == c1_fit.mutant_verdict(
            verdicts,
            chosen=chosen,
        )


def test_the_leave_one_out_is_labelled_both_ways() -> None:
    """The leave-one-out's tax-4 reads "prediction passed" at 3 of 5, "prediction failed" below."""
    assert fit.loo_prediction(3) == "prediction passed"
    assert fit.loo_prediction(2) == "prediction failed"


def test_the_gain_is_read_from_the_start() -> None:
    """Covers "The gain is read from the start"; the gate is the median over the truths."""
    assert fit.gain_from_start(3.0, 1.4, 1.0) == pytest.approx(0.8)
    assert fit.gain_from_start(3.0, 3.0, 1.0) == pytest.approx(0.0)


def test_the_leave_one_out_gates_nothing(bands: dict[str, Any]) -> None:
    """Covers "The leave-one-out gates nothing": its objective drops tax-4, and no gate reads it."""
    truth = fit.band_truth(bands)
    loo = fit.Objective(truth, bands, fit.LOO_MUTANTS)
    record = _record(bands, 0.0, {"tph-1": truth["delta"]["tph-1"], "tax-4": {}})
    assert loo(record) == pytest.approx(0.0)
    assert fit.Objective(truth, bands)(record) > 0.0
    assert fit.LOO_PATH not in fit.PART_C_BOUND
    # With one mutant left, it carries half: w = 6.
    assert fit.combine(1.0, [2.0], 6) == pytest.approx(math.sqrt((6 * 1.0 + 6 * 4.0) / 12))


# ---------------------------------------------------------------------------------------------
# The seeds and the guards
# ---------------------------------------------------------------------------------------------


def _blocks() -> dict[str, tuple[int, int]]:
    """Return each use's animal-seed range, [first, last], at the largest cap."""
    animals = 200
    last_gen = 16 * (fit.CAP_CEILING - 1) + 15

    def span(base: int, k_low: int, k_high: int) -> tuple[int, int]:
        return base + animals * k_low, base + animals * k_high + animals - 1

    blocks: dict[str, tuple[int, int]] = {}
    for name, base, fits in (
        ("fit", fit.FIT_BASE, fit.RESTARTS),
        ("recovery", fit.RECOVERY_BASE, fit.TRUTHS * fit.RECOVERY_RESTARTS),
        ("loo", fit.LOO_BASE, 1),
        ("pilot", fit.PILOT_BASE, 1),
    ):
        for f in range(fits):
            blocks[f"{name} {f}"] = span(base, fit.FIT_BLOCK * f, fit.FIT_BLOCK * f + last_gen)
            blocks[f"{name} answers {f}"] = span(
                base,
                fit.answer_k(f, 0),
                fit.answer_k(f, fit.RESIMULATIONS - 1),
            )
    blocks["registered"] = span(fit.GRADE_BASE, 0, fit.GRADE_POPULATIONS - 1)
    blocks["selection"] = span(
        fit.SELECT_BASE,
        0,
        fit.select_k(fit.RESTARTS - 1, fit.SELECT_POPULATIONS - 1),
    )
    blocks["loo grades"] = span(fit.LOO_GRADE_BASE, 0, fit.GRADE_POPULATIONS - 1)
    blocks["truths"] = span(fit.TRUTH_BASE, 0, fit.truth_k(fit.TRUTHS - 1, fit.TRUTH_REPEATS))
    blocks["prior"] = span(fit.PRIOR_BASE, 0, fit.PRIOR_DRAWS - 1)
    blocks["start"] = span(fit.START_BASE, 0, fit.RESIMULATIONS - 1)
    blocks["pilot grades"] = span(fit.PILOT_GRADE_BASE, 0, fit.PILOT_POPULATIONS - 1)
    blocks["pilot truths"] = span(
        fit.PILOT_TRUTH_BASE,
        0,
        fit.truth_k(fit.TRUTHS - 1, fit.TRUTH_REPEATS),
    )
    blocks["pilot start"] = span(fit.PILOT_START_BASE, 0, fit.RESIMULATIONS - 1)
    blocks["smoke"] = span(fit.SMOKE_BASE, 0, 400_000 // animals)
    blocks["separation probe"] = span(fit.PROBE_BASE, 0, 110)
    return blocks


def test_every_block_is_enumerated() -> None:
    """Covers "Every block is enumerated": no animal seed is in two uses at the largest cap."""
    ranges = sorted(_blocks().items(), key=lambda kv: kv[1][0])
    for (a, (_, a_end)), (b, (b_start, _)) in itertools.pairwise(ranges):
        assert a_end < b_start, f"{a} overlaps {b}"
    # Every block sits between Logbook 125's probes (below 647 M) and C.2's work (700 M up).
    assert ranges[0][1][0] >= 647_000_000
    assert ranges[-1][1][1] < 700_000_000


def test_every_cma_seed_is_its_own() -> None:
    """The restarts, the recovery, the pilot, the leave-one-out, the smoke and the prior differ."""
    seeds = [fit.CMA_SEED + r for r in range(fit.RESTARTS)]
    seeds += [fit.RECOVERY_CMA_SEED + f for f in range(fit.TRUTHS * fit.RECOVERY_RESTARTS)]
    seeds += [fit.PILOT_CMA_SEED, fit.LOO_CMA_SEED, fit.SMOKE_CMA_SEED, fit.PRIOR_SEED]
    assert len(set(seeds)) == len(seeds)
    assert min(seeds) == 9451
    assert max(seeds) == 9470


@pytest.mark.parametrize("stage", list(fit.STAGES))
def test_every_registered_stage_refuses_before_part_a(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no committed launch record, no registered stage runs."""
    monkeypatch.setattr(fit, "LAUNCH_PATH", _REPO / "no-such-dir" / "launch.md")
    assert fit.main([stage]) == 1


def test_the_fit_refuses_before_the_recovery_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers "The fit refuses before the recovery passes": Part C alone does not open the fit."""
    monkeypatch.setattr(fit, "_part_a", lambda: True)
    monkeypatch.setattr(fit, "_part_b", lambda: True)
    monkeypatch.setattr(fit, "_part_c", lambda: False)
    for stage in ("run", "select", "grade", "read", "loo"):
        assert fit.guarded(stage) is False
    assert fit.guarded("recovery") is True
    assert fit.guarded("pilot") is True


def test_part_c_needs_the_recovery_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Part C reads the recovery's own pass: a committed record at a gain of 0.70 refuses."""
    recovery = tmp_path / "recovery.json"
    recovery.write_text(json.dumps({"median_gain": 0.70, "passes": False}))
    monkeypatch.setattr(fit, "RECOVERY_PATH", recovery)
    monkeypatch.setattr(fit, "_has_part", lambda _name: True)
    monkeypatch.setattr(fit.cr, "launch_committed", lambda *_a, **_k: True)
    assert fit._part_c() is False
    recovery.write_text(json.dumps({"median_gain": 0.80, "passes": True}))
    assert fit._part_c() is True


def test_no_stage_overwrites_its_record(tmp_path: Path) -> None:
    """A stage's record is written once; a second write is refused."""
    path = tmp_path / "record.json"
    fit._write_once(path, {"a": 1})
    with pytest.raises(FileExistsError):
        fit._write_once(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_a_stored_record_is_reused_only_at_its_point_and_base(tmp_path: Path) -> None:
    """A store's record for k is reused at its own z and seed base, and refused at another."""
    store = tmp_path / "sims.jsonl"
    z = [0.5] * fit.DIMENSION
    store.write_text(json.dumps({"k": 3, "z": z, "statistics": {}, "seed_base": 7}) + "\n")
    assert fit.run_jobs([(3, z)], 7, store, workers=1)[3]["z"] == z
    with pytest.raises(ValueError, match="another point"):
        fit.run_jobs([(3, [0.4] * fit.DIMENSION)], 7, store, workers=1)
    with pytest.raises(ValueError, match="another seed base"):
        fit.run_jobs([(3, z)], 8, store, workers=1)


def test_the_smoke_reads_no_held_out_row() -> None:
    """The smoke's stand-in sets hold band-cohort rows only; the held-out sets are not touched."""
    from quantumnematode.benchmark import references as refs

    tracks = refs.load_scheer_tracks()
    sets = fit.smoke_sets()
    assert set(sets) == {"N2_bands", "PD1074_bands"}
    for rows in sets.values():
        assert len(rows) > 0
        assert set(tracks.side[rows]) == {refs.BANDS}


def test_every_stage_simulates_on_the_following_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every evaluation simulates the wild type and both mutants on ``mlp_following``."""
    arms: list[str] = []

    def simulate_v2(arm: str, *_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], None]:
        arms.append(arm)
        return {"small_lawn": []}, None

    graded = {"families": {}, "reversal_share": None, "D": 1.0, "D_fit": 1.0}
    monkeypatch.setattr(fit.bg, "simulate_v2", simulate_v2)
    monkeypatch.setattr(fit.bg, "grade_populations_v4", lambda *_a, **_k: graded)
    monkeypatch.setattr(fit.bg, "classifier", lambda _bands: None)
    monkeypatch.setattr(
        fit.bg,
        "family_values",
        lambda *_a: {"roaming_fraction": 0.5, "exits_per_on_lawn_hour": 1.0},
    )
    record = fit.simulate(0, fit.start_point(), 0)
    assert arms == ["mlp_following"] * 3
    assert record["delta"] == {m: dict.fromkeys(fit.LAWN_DIFFERENCES, 0.0) for m in fit.MUTANTS}


def _kinds(monkeypatch: pytest.MonkeyPatch, **tables: dict[str, float]) -> None:
    """Make each recovery distance read a record's ``kind`` from its own table."""
    monkeypatch.setattr(
        fit,
        "distances",
        lambda: {
            name: (lambda _a, b, table=table: table.get(b.get("kind"), 9.0))
            for name, table in tables.items()
        },
    )
    monkeypatch.setattr(fit, "band_truth", lambda _bands: {"kind": "band"})


def _preflight(**extra: dict[str, float]) -> dict[str, Any]:
    plain = {"statistics": {"x": 1.0}, "delta": {"tph-1": {"roaming_fraction": 0.1}}}
    runs = [{"kind": "floor", **plain}] * 10 + [{"kind": "oracle", **plain}] * 10
    return fit.preflight_record([plain], [runs], [{"kind": "start"}], None)


def test_the_start_is_too_close_to_the_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers "The start is too close to the truth", on either distance; 1.0 itself is enough."""
    good = {"floor": 1.0, "oracle": 1.05, "start": 3.0, "band": 2.9}
    _kinds(monkeypatch, combined=good, wild_type=dict(good))
    record = _preflight()
    assert record["passed"] is True
    assert record["truths"][0]["combined"]["oracle_gain"] == pytest.approx(0.975)
    assert record["truths"][0]["combined"]["gate_line_D"] == pytest.approx(1.5)
    assert record["truths"][0]["combined"]["cannot_discriminate"] is False
    _kinds(monkeypatch, combined=good, wild_type={**good, "start": 1.6})
    assert _preflight()["passed"] is False
    _kinds(monkeypatch, combined={**good, "start": 1.6}, wild_type=good)
    assert _preflight()["passed"] is False
    _kinds(monkeypatch, combined=good, wild_type={**good, "start": 2.0, "oracle": 1.0})
    assert _preflight()["truths"][0]["wild_type"]["separated"] is True


def test_the_oracle_is_read_on_both_distances(monkeypatch: pytest.MonkeyPatch) -> None:
    """An oracle below 0.85 on the wild type's distance fails the preflight on its own."""
    good = {"floor": 1.0, "oracle": 1.05, "start": 3.0}
    _kinds(monkeypatch, combined=good, wild_type={**good, "oracle": 1.5})
    record = _preflight()
    assert record["truths"][0]["wild_type"]["oracle_gain"] == pytest.approx(0.75)
    assert record["passed"] is False


def test_a_statistics_definedness_flips(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers "A statistic's definedness flips": a mutant Δ undefined in one of 1-10 fails it."""
    good = {"floor": 1.0, "oracle": 1.05, "start": 3.0}
    _kinds(monkeypatch, combined=good, wild_type=good)
    defined = {"statistics": {"x": 1.0}, "delta": {"tph-1": {"roaming_fraction": 0.1}}}
    flipped = {"statistics": {"x": 1.0}, "delta": {"tph-1": {"roaming_fraction": None}}}
    late = [{"kind": "floor", **defined}] * 10 + [{"kind": "oracle", **flipped}] * 10
    assert fit.preflight_record([defined], [late], [{"kind": "start"}], None)["passed"] is True
    runs = [{"kind": "floor", **defined}] * 9 + [{"kind": "floor", **flipped}]
    runs += [{"kind": "oracle", **defined}] * 10
    record = fit.preflight_record([defined], [runs], [{"kind": "start"}], None)
    assert record["truths"][0]["definedness_flips"] == ["tph-1/roaming_fraction"]
    assert record["passed"] is False
    assert record["truths"][0]["observation"] == defined


def test_the_floor_reads_re_simulations_one_to_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    """The floor is the median over re-simulations 1-10, the oracle over 11-20."""
    table = {f"r{q}": float(q) for q in range(1, 21)} | {"start": 50.0}
    _kinds(monkeypatch, combined=table, wild_type=dict(table))
    plain = {"statistics": {}, "delta": {}}
    runs = [{"kind": f"r{q}", **plain} for q in range(1, 21)]
    row = fit.preflight_record([plain], [runs], [{"kind": "start"}], None)["truths"][0]
    assert row["combined"]["floor"] == 5.5
    assert row["combined"]["oracle_D"] == 15.5


def test_the_gain_is_read_from_the_start_not_the_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers "The gain is read from the start": a restart's gain uses the start D."""
    _kinds(monkeypatch, combined={"answer": 2.0}, wild_type={"answer": 2.0})
    reading = {"floor": 1.0, "start_D": 6.0, "prior_D": 10.0}
    row = {"combined": reading, "wild_type": dict(reading)}
    out = fit.restart_reading(row, {}, [{"kind": "answer"}] * 10)
    assert out["combined"]["gain"] == pytest.approx(0.8)
    assert out["combined"]["prior_gain"] == pytest.approx(8 / 9)
    assert fit.gain_from_start(3.0, 1.4, 1.0) == pytest.approx(0.8)


def test_the_recovery_falls_short() -> None:
    """Covers "The recovery falls short": 0.70 on either distance fails the gate."""
    assert (
        fit.recovery_gate({"combined": {"gain": 0.70}, "wild_type": {"gain": 0.90}})["passes"]
        is False
    )
    assert (
        fit.recovery_gate({"combined": {"gain": 0.90}, "wild_type": {"gain": 0.70}})["passes"]
        is False
    )
    assert (
        fit.recovery_gate({"combined": {"gain": 0.75}, "wild_type": {"gain": 0.75}})["passes"]
        is True
    )


def test_the_gate_passes_on_the_mutants_alone() -> None:
    """Covers "The gate passes on the mutants alone": 0.86 combined, 0.40 on the wild type."""
    gate = fit.recovery_gate({"combined": {"gain": 0.86}, "wild_type": {"gain": 0.40}})
    assert gate["gains"] == {"combined": 0.86, "wild_type": 0.40}
    assert gate["passes"] is False


def test_the_better_restart_is_the_lower_final_objective() -> None:
    """The gate reads the restart with the lower mean of its last 20 medians."""
    assert fit.best_fit([0, 1], {0: [5.0] * 20, 1: [3.0] * 20}) == 1
    assert fit.best_fit([0, 1], {0: [3.0] * 20, 1: [5.0] * 20}) == 0


def test_the_recovery_stage_gates_on_the_better_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recovery fits against the preflight's observation and gates on its better restart."""
    observation = {"kind": "observation"}
    reading = {"floor": 1.0, "start_D": 3.0, "prior_D": 10.0}
    preflight = {
        "passed": True,
        "truths": [{"combined": reading, "wild_type": dict(reading), "observation": observation}],
    }
    pilot = {"answer": [0.5] * fit.DIMENSION, "cap": 300}
    for name, record in (("RECOVERY_PREFLIGHT_PATH", preflight), ("PILOT_PATH", pilot)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(record))
        monkeypatch.setattr(fit, name, path)
    monkeypatch.setattr(fit, "RECOVERY_PATH", tmp_path / "recovery.json")
    monkeypatch.setattr(fit, "BUILD", tmp_path)
    seen: list[dict[str, Any]] = []

    def fit_restart(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        good = kwargs["fit"] == 1
        return {
            "medians": [2.0 if good else 4.0] * 20,
            "sigmas": [0.05] * 20,
            "answers": [[0.5] * fit.DIMENSION],
            "stop": {"by": "plateau", "generation": 60},
        }

    monkeypatch.setattr(fit, "fit_restart", fit_restart)
    monkeypatch.setattr(
        fit,
        "run_jobs",
        lambda jobs, *_a: {k: {"kind": "good" if k >= fit.FIT_BLOCK else "bad"} for k, _ in jobs},
    )
    _kinds(monkeypatch, combined={"good": 1.4, "bad": 2.8}, wild_type={"good": 1.4, "bad": 2.8})
    assert fit._recovery(1) == 0
    record = json.loads((tmp_path / "recovery.json").read_text())
    assert record["gains"] == {"combined": pytest.approx(0.8), "wild_type": pytest.approx(0.8)}
    assert record["passes"] is True
    assert [s["cma_seed"] for s in seen] == [fit.RECOVERY_CMA_SEED, fit.RECOVERY_CMA_SEED + 1]
    assert all(s["seed_base"] == fit.RECOVERY_BASE and s["cap"] == 300 for s in seen)
    assert all(s["objective"].truth == observation for s in seen)


# ---------------------------------------------------------------------------------------------
# The pilot's go
# ---------------------------------------------------------------------------------------------


def _go(**parts: Any) -> dict[str, Any]:
    defaults = {
        "stop_by": "plateau",
        "rule": {"passes": True},
        "check": {"passed": True},
        "projected": {"registered": 30.0},
    }
    return fit.go_decision(**{**defaults, **parts})


def test_the_go_needs_every_part() -> None:
    """Covers "The pilot does not stop" and "The projected cost is too high", part by part."""
    assert _go()["go"] is True
    assert _go(stop_by="cap")["go"] is False
    assert _go(rule={"passes": False})["go"] is False
    assert _go(check={"passed": False})["go"] is False
    assert _go(projected={"registered": 48.5})["go"] is False
    assert _go(projected={"registered": 48.0})["go"] is True


def test_the_pilot_stage_reads_its_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pilot runs once at its own seeds and writes its go; a capped pilot does not go."""
    monkeypatch.setattr(fit, "PILOT_PATH", tmp_path / "pilot.json")
    monkeypatch.setattr(fit, "BUILD", tmp_path)
    seen: dict[str, Any] = {}

    def fit_restart(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {
            "medians": [5.0] * 400,
            "sigmas": [0.05] * 400,
            "seconds": [38.0] * 400,
            "answers": [[0.5] * fit.DIMENSION],
            "stop": {"by": "cap", "generation": 400},
        }

    monkeypatch.setattr(fit, "fit_restart", fit_restart)
    monkeypatch.setattr(
        fit,
        "grade_all",
        lambda jobs, _w: [_grade(wild=True) | {"passes": {}, "z": {}} for _ in jobs],
    )
    monkeypatch.setattr(fit, "_truth_runs", lambda *_a: ([{}], [[{}] * 20]))
    monkeypatch.setattr(fit, "_start_runs", lambda *_a: [{}] * 10)
    monkeypatch.setattr(fit, "preflight_record", lambda *_a: {"passed": True, "truths": []})
    monkeypatch.setattr(fit, "diagnosis", lambda _g: {})
    assert fit._pilot(1) == 1
    record = json.loads((tmp_path / "pilot.json").read_text())
    assert record["go"] is False
    assert record["go_decision"]["stopped"] is False
    assert record["cap"] is None
    assert seen["cma_seed"] == fit.PILOT_CMA_SEED
    assert seen["seed_base"] == fit.PILOT_BASE
    assert seen["cap"] == fit.PILOT_CAP
    assert fit._pilot(1) == 1  # written once


# ---------------------------------------------------------------------------------------------
# The guards, through guarded()
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "needs"),
    [
        ("pilot", "A"),
        ("recovery-preflight", "AB"),
        ("recovery", "AB"),
        ("run", "ABC"),
        ("select", "ABC"),
        ("grade", "ABC"),
        ("read", "ABC"),
        ("loo", "ABC"),
    ],
)
def test_each_stage_needs_its_parts(
    stage: str,
    needs: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage opens with its parts and refuses when any one of them is missing."""
    state = dict.fromkeys("ABC", True)
    for part in "ABC":
        monkeypatch.setattr(fit, f"_part_{part.lower()}", lambda part=part: state[part])
    assert fit.guarded(stage) is True
    for part in "ABC":
        state[part] = False
        assert fit.guarded(stage) is (part not in needs)
        state[part] = True


def test_part_a_needs_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers "The code changes after Part A" through the guard: Part A refuses a changed tree."""
    monkeypatch.setattr(fit.cr, "launch_committed", lambda *_a, **_k: True)
    monkeypatch.setattr(fit, "code_unchanged", lambda: False)
    assert fit._part_a() is False
    assert fit.guarded("pilot") is False
    monkeypatch.setattr(fit, "code_unchanged", lambda: True)
    assert fit._part_a() is True


def test_part_b_needs_the_pilots_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Part B opens the recovery only when the committed pilot record says go."""
    pilot = tmp_path / "pilot.json"
    monkeypatch.setattr(fit, "PILOT_PATH", pilot)
    monkeypatch.setattr(fit, "_has_part", lambda _name: True)
    monkeypatch.setattr(fit.cr, "launch_committed", lambda *_a, **_k: True)
    pilot.write_text(json.dumps({"go": False}))
    assert fit._part_b() is False
    pilot.write_text(json.dumps({"go": True}))
    assert fit._part_b() is True
    monkeypatch.setattr(fit, "_has_part", lambda _name: False)
    assert fit._part_b() is False


def test_part_c_needs_the_pass_or_a_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover "The fit refuses before the recovery passes" and "The maintainer decides to run".

    Only the decision heading, on a line of its own inside Part A, opens a failed gate.
    """
    recovery, launch = tmp_path / "recovery.json", tmp_path / "launch.md"
    monkeypatch.setattr(fit, "RECOVERY_PATH", recovery)
    monkeypatch.setattr(fit, "LAUNCH_PATH", launch)
    monkeypatch.setattr(fit, "_has_part", lambda _name: True)
    monkeypatch.setattr(fit.cr, "launch_committed", lambda *_a, **_k: True)
    recovery.write_text(json.dumps({"passes": False}))
    launch.write_text("# Launch\n\n## Part A\n\nrules\n\n## Part B\n\n## Part C\n")
    assert fit._part_c() is False
    launch.write_text(f"# Launch\n\n## Part B\n\n{fit.DECIDED_MARKER}\n\n## Part C\n")
    assert fit._part_c() is False
    launch.write_text(f"# Launch\n\n## Part A\n\nAn amendment adds {fit.DECIDED_MARKER}.\n")
    assert fit._part_c() is False
    launch.write_text(f"# Launch\n\n## Part A\n\n{fit.DECIDED_MARKER}\n\n## Part B\n")
    assert fit._part_c() is True
    recovery.write_text(json.dumps({"passes": True}))
    launch.write_text("# Launch\n\n## Part A\n\n## Part B\n")
    assert fit._part_c() is True


class _Done:
    def __init__(self, code: int = 0, out: bytes = b"") -> None:
        self.returncode, self.stdout = code, out


def test_part_a_is_pinned_at_its_last_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers "Part B does not move the pin": only a change to Part A's own text moves it."""
    texts = {
        "c1": "# L\n\n## Part A\n\nrules\n",
        "c2": "# L\n\n## Part A\n\nrules\n\n## Part B\n\npilot\n",
        "c3": "# L\n\n## Part A\n\nrules, amended\n\n## Part B\n\npilot\n",
    }
    history = ["c1", "c2"]

    def git(*args: str) -> _Done:
        if args[0] == "log":
            newest_first = history[::-1]
            return _Done(out=" ".join(newest_first[:1] if "-1" in args else newest_first).encode())
        return _Done(out=texts[args[1].split(":")[0]].encode())

    monkeypatch.setattr(fit.cr, "_git", git)
    assert fit.part_a_commit() == "c1"
    history.append("c3")
    assert fit.part_a_commit() == "c3"


def test_the_code_changes_after_part_a(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers "The code changes after Part A": a changed or untracked file, or no Part A."""
    calls: list[tuple[str, ...]] = []
    results = {"diff": _Done(0), "status": _Done(0, b"")}

    def git(*args: str) -> _Done:
        calls.append(args)
        return results[args[0]]

    monkeypatch.setattr(fit, "part_a_commit", lambda: "abc123")
    monkeypatch.setattr(fit.cr, "_git", git)
    assert fit.code_unchanged() is True
    assert calls[0][:3] == ("diff", "--quiet", "abc123")
    assert "--untracked-files=all" in calls[1]
    results["diff"] = _Done(1)
    assert fit.code_unchanged() is False
    results |= {"diff": _Done(0), "status": _Done(0, b"?? packages/x.py\n")}
    assert fit.code_unchanged() is False
    monkeypatch.setattr(fit, "part_a_commit", lambda: None)
    assert fit.code_unchanged() is False
    paths = {str(p.relative_to(_REPO)) for p in fit.CODE_PATHS}
    for path in (
        "packages/quantum-nematode/quantumnematode",
        "packages/wormbench/wormbench",
        "scripts/analysis/c1_v4_fit.py",
        "uv.lock",
        "pyproject.toml",
    ):
        assert path in paths


# ---------------------------------------------------------------------------------------------
# The seeds each stage uses, the objective's undefined count, and the readings
# ---------------------------------------------------------------------------------------------


def _fake_graded() -> dict[str, Any]:
    return {
        "families": {"small_lawn": {"x": {"value": 1.0, "z": 0.0, "passes": True}}},
        "family_passes": {"small_lawn": True},
        "local_search_presence": None,
        "chemotaxis_presence": None,
        "wild_type_passes": True,
        "reversal_share": None,
        "D": 1.0,
        "D_fit": 1.0,
    }


def test_simulations_and_grades_use_their_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulations and grades run base + 200·k + i, for i < 200."""
    seen: list[list[int]] = []

    def evaluate(_z: Any, seeds: list[int], *, presence: bool) -> tuple[dict, dict]:
        seen.append(seeds)
        return _fake_graded(), {m: {} for m in fit.MUTANTS}

    monkeypatch.setattr(fit, "evaluate", evaluate)
    fit.simulate(3, fit.start_point(), 1_000_000)
    fit.grade_answer(4, fit.start_point(), 2_000_000)
    assert seen[0] == [1_000_000 + 600 + i for i in range(200)]
    assert seen[1] == [2_000_000 + 800 + i for i in range(200)]


@pytest.mark.parametrize(("stage", "base"), [("select", "SELECT_BASE"), ("grade", "GRADE_BASE")])
def test_selection_and_grading_use_their_bases(
    stage: str,
    base: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection grades plateaued restarts at the selection base; grading every one at its own."""
    histories = [
        {"stop": {"by": "cap" if r == 2 else "plateau"}, "answers": [[r / 20] * fit.DIMENSION]}
        for r in range(fit.RESTARTS)
    ]
    monkeypatch.setattr(fit, "_histories", lambda: histories)
    monkeypatch.setattr(fit, "BUILD", tmp_path)
    jobs: list[tuple[int, list[float], int]] = []
    monkeypatch.setattr(fit, "grade_all", lambda j, _w: jobs.extend(j) or [{} for _ in j])
    getattr(fit, f"_{stage}")(1)
    assert {b for _, _, b in jobs} == {getattr(fit, base)}
    if stage == "select":
        restarts = [r for r in range(fit.RESTARTS) if r != 2]
        assert [k for k, _, _ in jobs] == [fit.select_k(r, s) for r in restarts for s in range(5)]
    else:
        assert [k for k, _, _ in jobs] == list(range(fit.GRADE_POPULATIONS)) * fit.RESTARTS


def test_an_undefined_wild_type_statistic_ranks_first(bands: dict[str, Any]) -> None:
    """A record with an undefined wild-type statistic scores 1,000 or more, as C.2's does."""
    import c2_fit  # pyright: ignore[reportMissingImports]

    truth = fit.band_truth(bands)
    key = next(iter(truth["statistics"]))
    record = {"statistics": {**truth["statistics"], key: None}, "delta": truth["delta"]}
    ours = fit.Objective(truth, bands)(record)
    assert ours >= v2.UNDEFINED_RANK
    theirs = c2_fit.Objective(
        c2_fit.band_truth(bands),
        bands,
        families=None,
        mutants=fit.MUTANTS,
    ).distance(record)
    assert ours == pytest.approx(theirs)


def test_mutant_rows_read_the_first_five_at_three() -> None:
    """The rule reads each restart's first five at three of five, wild type and mutants alike."""
    three = [_grade(wild=n < 3, tph1=n < 3) for n in range(5)]
    two = [_grade(wild=n < 2) for n in range(5)]
    late = [_grade(wild=True, tph1=False)] * 5 + [_grade(wild=True)] * 10
    rows = fit.mutant_rows([three, two, late], chosen=0)
    assert rows["wild_type_restarts"] == [0, 2]
    assert rows["tph-1"]["restart_verdicts"] == {"0": True, "2": False}
    assert rows["tph-1"]["row"] == "unidentified"
    only_late = fit.mutant_rows([late], chosen=0)
    assert only_late["tph-1"]["row"] == "fail"


def test_a_mutant_on_the_wrong_side_fails_within_the_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Δ of the wrong sign fails though its |z| is inside the bar, where the sign is resolved.

    At v4's bands every wrong-signed resolved Δ already lies past the bar, so a band is set here.
    """
    from quantumnematode.benchmark import grading as g

    bands = {
        "roaming_fraction": g.Band(0.02, 0.05, direction=1),
        "exits_per_on_lawn_hour": g.Band(0.3, 0.36),
    }
    monkeypatch.setattr(fit.bg, "mutant_bands", lambda *_a: bands)
    reading = fit.mutant_reading(
        "tax-4",
        {"roaming_fraction": -0.01, "exits_per_on_lawn_hour": 0.3},
        {"bars": {"tax-4": 2.2414}},
    )
    assert abs(reading["statistics"]["roaming_fraction"]["z"]) < 2.2414
    assert reading["statistics"]["roaming_fraction"]["passes"] is False
    assert reading["passes"] is False


def test_a_capped_restart_is_not_chosen_even_if_it_passes_most() -> None:
    """Covers "A capped restart is not chosen": choose() drops it before counting."""
    histories = [{"stop": {"by": "cap"}}, {"stop": {"by": "plateau"}}]
    selection = {0: [_grade(wild=True)] * 5, 1: [_grade(wild=True)] * 3 + [_grade(wild=False)] * 2}
    assert fit.choose(histories, selection, {0: 1.0, 1: 2.0}) == 1


def test_unit_errors_carry_the_on_food_rate() -> None:
    """The parameter errors include the new rate, unpaired."""
    truth = [0.5] * fit.DIMENSION
    answer = [*truth[:25], 0.8]
    errors = fit.unit_errors(answer, truth)["errors"]
    assert errors["r_dwell_on_food"] == pytest.approx(0.3)


def test_the_leave_one_out_runs_on_its_own_terms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leave-one-out scores tph-1 alone, at its own seeds, and labels its tax-4."""
    (tmp_path / "fit.json").write_text(json.dumps({"pass_rates": {}}))
    (tmp_path / "pilot.json").write_text(json.dumps({"cap": 300}))
    monkeypatch.setattr(fit, "FIT_PATH", tmp_path / "fit.json")
    monkeypatch.setattr(fit, "PILOT_PATH", tmp_path / "pilot.json")
    monkeypatch.setattr(fit, "LOO_PATH", tmp_path / "loo.json")
    monkeypatch.setattr(fit, "BUILD", tmp_path)
    seen: dict[str, Any] = {}

    def fit_restart(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {
            "medians": [5.0],
            "sigmas": [0.05],
            "answers": [[0.5] * fit.DIMENSION],
            "stop": {"by": "plateau", "generation": 60},
        }

    jobs: list[tuple[int, list[float], int]] = []

    def grade_all(j: list[tuple[int, list[float], int]], _w: int) -> list[dict[str, Any]]:
        jobs.extend(j)
        return [_grade(wild=True, tax4=n < 2) for n in range(len(j))]

    monkeypatch.setattr(fit, "fit_restart", fit_restart)
    monkeypatch.setattr(fit, "grade_all", grade_all)
    assert fit._loo(1) == 0
    assert seen["objective"].mutants == ("tph-1",)
    assert (seen["cma_seed"], seen["seed_base"], seen["cap"]) == (
        fit.LOO_CMA_SEED,
        fit.LOO_BASE,
        300,
    )
    assert {b for _, _, b in jobs} == {fit.LOO_GRADE_BASE}
    record = json.loads((tmp_path / "loo.json").read_text())
    assert record["prediction"] == "prediction failed"
    assert record["capped"] is False


def test_the_launch_record_decides_nothing() -> None:
    """The committed launch record's Part A carries no decision heading: its gate binds."""
    if not fit.LAUNCH_PATH.exists():
        pytest.skip("no launch record yet")
    assert fit.decided_despite_gate() is False
