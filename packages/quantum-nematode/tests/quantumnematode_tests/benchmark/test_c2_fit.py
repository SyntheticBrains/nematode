"""C.2's registered fit: its objective, procedure, verdict, recovery, seeds and guards.

Covers the controller-fit spec's scenarios "A later stage keeps the search's steps", "Stage 1 fits
only its subset", "A plateau does not stop a stage", "The mutants weigh half", "A statistic at its
bar", "tph-1's leaving is not scored", "A statistic undefined in most noise populations", "A resume
with changed settings", "A fitted mutant fails the verdict", "pdfr-1 does not decide", "tax-4 is
never scored", "The fit refuses before the recovery is recorded", "A changed config stops a run", "A
failed pilot blocks the recovery", "The longest stage stays inside its sub-block", "The real halves
reproduce W's minimum", "The smoke reads no held-out track", "A start that does not hold the lawn
is dropped", "A start that misses 2a's descent or bar is dropped", "A registered run misses a
checkpoint", "A start's 2a ends above the bar", "No start holds the lawn", "The lawn-holding
reading is a mean", "The descent reading is a mean", "An interrupted drop resumes to the same
answer", "Each start has its own animals", "A registered fit with no answer fails" and "The lawn
guard"; and the fit-recovery spec's "The preflight fails", "The recovery falls short", "A truth
whose statistic flips", "The preflight runs once" and "A recovery with no answer does not pass".
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import benchmark_grade as bg  # noqa: E402  # pyright: ignore[reportMissingImports]
import c2_fit as fit  # noqa: E402  # pyright: ignore[reportMissingImports]
import cma_recovery_v2 as v2  # noqa: E402  # pyright: ignore[reportMissingImports]

# The script's own strategies, which the fake runs wrap.
_FRESH, _CARRIED = fit.fresh, fit.carried


@pytest.fixture(scope="module")
def bands() -> dict[str, Any]:
    """Return benchmark v4's bands."""
    return bg.load_bands_v4()


def _record(truth: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return a record at the truth, responsive and unsaturated, with overrides applied."""
    record = {
        "statistics": dict(truth["statistics"]),
        "delta": {m: dict(d) for m, d in truth["delta"].items()},
        "saturated": 0.0,
        "spread": {"speed": 1.0, "reversal": 1.0},
    }
    record.update(overrides)
    return record


def _at_z(truth: dict[str, Any], bands: dict[str, Any], key: str, z: float) -> float:
    return truth["statistics"][key] + z * v2.band_scales_v2(bands)[key]


# ---------------------------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------------------------


def test_the_mutants_weigh_half(bands: dict[str, Any]) -> None:
    """Covers "The mutants weigh half": six families at 1 and three mutants at 3 give sqrt(5)."""
    truth = fit.band_truth(bands)
    unit = 10 * math.sinh(0.1)  # compresses to exactly 1
    three = 10 * math.sinh(0.3)  # compresses to exactly 3
    record = _record(truth)
    for key in truth["statistics"]:
        record["statistics"][key] = _at_z(truth, bands, key, unit)
    scales = fit.mutant_scales(bands)
    for m in fit.MUTANTS:
        for s in fit.GRADED[m]:
            record["delta"][m][s] = truth["delta"][m][s] + three * scales[m][s]
    objective = fit.Objective(truth, bands, families=None, mutants=fit.MUTANTS)
    assert objective.distance(record) == pytest.approx(math.sqrt(5.0), rel=1e-9)
    assert objective(record) == pytest.approx(math.sqrt(5.0), rel=1e-9)


def test_a_statistic_at_its_bar(bands: dict[str, Any]) -> None:
    """Covers "A statistic at its bar": its failure term is 10 x 0.5 = 5."""
    truth = fit.band_truth(bands)
    sd = dict.fromkeys(truth["statistics"], fit.SD_FLOOR)
    objective = fit.Objective(truth, bands, families=None, mutants=(), sd=sd)
    key = "activity/onset_rate_per_min"
    record = _record(truth)
    record["statistics"][key] = _at_z(truth, bands, key, float(bands["bars"]["wild_type"]))
    assert objective.failure(record) == pytest.approx(5.0, abs=1e-9)
    assert fit.phi(0.0) == 0.5


def test_tph1_leaving_is_not_scored(bands: dict[str, Any]) -> None:
    """Covers "tph-1's leaving is not scored": its exits difference moves no term."""
    truth = fit.band_truth(bands)
    sd = {
        **dict.fromkeys(truth["statistics"], 1.0),
        **{f"{m}/{s}": 1.0 for m in fit.MUTANTS for s in fit.GRADED[m]},
    }
    objective = fit.Objective(truth, bands, families=None, mutants=fit.MUTANTS, sd=sd)
    near = _record(truth)
    far = _record(truth)
    far["delta"]["tph-1"]["exits_per_on_lawn_hour"] += 50.0
    assert objective(far) == objective(near)
    assert fit.GRADED["tph-1"] == ("roaming_fraction",)
    reading = fit.mutant_reading("tph-1", far["delta"]["tph-1"], bands)
    assert reading["statistics"]["exits_per_on_lawn_hour"]["graded"] is False
    assert reading["statistics"]["exits_per_on_lawn_hour"]["passes"] is None
    assert reading["passes"] is True


def test_tax4_is_never_scored_in_the_leave_one_out(bands: dict[str, Any]) -> None:
    """Covers "tax-4 is never scored": a leave-one-out score ignores tax-4's differences."""
    truth = fit.band_truth(bands)
    sd = {
        **dict.fromkeys(truth["statistics"], 1.0),
        **{f"{m}/{s}": 1.0 for m in fit.MUTANTS for s in fit.GRADED[m]},
    }
    mutants = fit.RUNS["loo"].mutants
    assert "tax-4" not in mutants
    objective = fit.Objective(truth, bands, families=None, mutants=mutants, sd=sd)
    near, far = _record(truth), _record(truth)
    far["delta"]["tax-4"] = dict.fromkeys(fit.LAWN_DIFFERENCES, 10.0)
    assert objective(far) == objective(near)


def test_an_undefined_statistic_and_the_guards(bands: dict[str, Any]) -> None:
    """An undefined statistic ranks a thousand behind; saturation and a pinned output are paid."""
    truth = fit.band_truth(bands)
    objective = fit.Objective(truth, bands, families=fit.FIT_FAMILIES, mutants=())
    record = _record(truth)
    record["statistics"]["kinematics/dwelling_speed_p90"] = None
    assert objective(record) == pytest.approx(1000.0)
    pinned = _record(truth, saturated=0.6, spread={"speed": 0.25, "reversal": 1.0})
    assert objective(pinned) == pytest.approx(500.0 + 300.0 * 0.5)


@pytest.mark.parametrize("stage", fit.STAGES, ids=lambda s: s.name)
def test_the_lawn_guard(bands: dict[str, Any], stage: Any) -> None:
    """Covers "The lawn guard": 0 at 0.95 and above, 150 at 0.475, 300 undefined, in every stage."""
    truth = fit.band_truth(bands)
    sd = {
        **dict.fromkeys(truth["statistics"], 1.0),
        **{f"{m}/{s}": 1.0 for m in fit.MUTANTS for s in fit.GRADED[m]},
    }
    objective = fit.Objective(
        truth,
        bands,
        families=stage.families,
        mutants=fit.MUTANTS if stage.mutants else (),
        sd=sd if stage.failure else None,
    )
    assert truth["statistics"][fit.LAWN_STATISTIC] > fit.LAWN_GUARD_BELOW
    for lawn, paid in ((None, 300.0), (0.0, 300.0), (0.475, 150.0), (0.95, 0.0), (0.99, 0.0)):
        record = _record(truth)
        record["statistics"][fit.LAWN_STATISTIC] = lawn
        assert fit.lawn_guard(record) == pytest.approx(paid, abs=1e-9)
        assert fit.Objective.guards(record) == pytest.approx(paid, abs=1e-9)
        rest = objective.distance(record) + objective.failure(record)
        assert objective(record) - rest == pytest.approx(paid, abs=1e-9)
    # It adds to the other guards.
    pinned = _record(truth, saturated=0.6)
    pinned["statistics"][fit.LAWN_STATISTIC] = 0.475
    assert fit.Objective.guards(pinned) == pytest.approx(500.0 + 150.0)


def test_a_statistic_undefined_in_most_noise_populations(bands: dict[str, Any]) -> None:
    """Covers "A statistic undefined in most noise populations": its SD is the floor."""
    truth = fit.band_truth(bands)
    key = "small_lawn/roaming_fraction"
    scale = v2.band_scales_v2(bands)[key]
    populations = []
    for p in range(30):
        record = _record(truth)
        record["statistics"][key] = truth["statistics"][key] + (p % 2) * 2.0 * scale
        record["statistics"]["small_lawn/exits_per_on_lawn_hour"] = None if p else 0.5
        populations.append(record)
    sd = fit.noise_sd(populations, truth, bands, fit.MUTANTS)
    assert sd[key] == pytest.approx(float(np.std([(p % 2) * 2.0 for p in range(30)], ddof=1)))
    assert sd["small_lawn/exits_per_on_lawn_hour"] == fit.SD_FLOOR
    assert sd["small_lawn/time_on_lawn"] == fit.SD_FLOOR  # no spread at all
    assert set(sd) >= {f"{m}/{s}" for m in fit.MUTANTS for s in fit.GRADED[m]}


# ---------------------------------------------------------------------------------------------
# The procedure, on fake simulations
# ---------------------------------------------------------------------------------------------


def _fake_entries() -> list[Any]:
    blocks = ["wiring", "sensory", "readout", "noise", "modulator", "timescale", "readout"]
    names = ["w", "s", "r", "n", "m", "timescale.speed_noise", "readout.centre.VB"]
    return [
        SimpleNamespace(name=n, block=b, unit=lambda _v: 0.123)
        for n, b in zip(names, blocks, strict=True)
    ]


def _fake(  # noqa: C901, PLR0913 - the patches, the run's shape and what its starts read
    monkeypatch: pytest.MonkeyPatch,
    build: Path,
    bands: dict[str, Any],
    *,
    generations: tuple[int, ...] = (3, 3, 3, 45),
    hold: int = 3,
    hold_bars: tuple[float, ...] = (0.5,),
    descent: int | None = None,
    descent_bars: tuple[float, ...] | None = None,
    reach_bars: tuple[float, ...] | None = None,
    readings: tuple[tuple[int, ...], ...] | None = None,
    at: dict[tuple[int, int], dict[str, float]] | None = None,
    lawn: tuple[float, ...] = (0.9,),
    reversal: tuple[float, ...] = (1.0,),
    d_fit: tuple[float, ...] = (1.0,),
    kill_rules: bool = False,
    interrupt: tuple[int, int, int] | None = None,
    candidate_lawn: float | None = None,
) -> dict:
    """Run the procedure on fake simulations: every candidate and readout seen, every hand-off.

    ``lawn``, ``reversal`` and ``d_fit`` give each start's time on the lawn, reversal spread and
    D_fit at every readout, the last value serving every later start. 2a reads the lawn at
    ``hold``, the descent at ``descent`` (2a's end unless given) and the bar at 2a's end, each
    against the registered bar unless given; ``readings``, when given, are the three readings'
    readouts. ``at`` overrides a readout's values at (start, 2a's generation). ``interrupt`` stops
    the run as (stage, generation, start) begins to simulate, as a process dying there would.
    ``candidate_lawn``, when given, is every candidate's time on the small lawn.
    """
    seen: dict[str, Any] = {"candidates": {}, "carried": [], "fresh": [], "stores": [], "start": 0}
    truth = fit.band_truth(bands)
    saved = build / "fake" / "state.pkl"
    if saved.exists():
        # A resumed run is inside the start its saved state names.
        seen["start"] = pickle.loads(saved.read_bytes())["start"]  # noqa: S301 - the test's own

    def generation(_pool, _run, stage, g, candidates, path, first_half_start=0):
        if interrupt == (stage, g, first_half_start):
            raise KeyboardInterrupt
        seen["candidates"].setdefault(stage, []).extend(candidates)
        seen["stores"].append((stage, g, first_half_start, path.name))
        # The readout that follows a generation reads its stage and generation.
        seen["at"] = (stage, g)
        out = []
        for z in candidates:
            record = _record(truth)
            key = "small_lawn/roaming_fraction"
            record["statistics"][key] = truth["statistics"][key] + 0.01 * float(np.sum(z))
            if candidate_lawn is not None:
                record["statistics"][fit.LAWN_STATISTIC] = candidate_lawn
            out.append(record)
        return out

    def of(values: tuple[float, ...]) -> float:
        return values[min(seen["start"], len(values) - 1)]

    def readout(_pool, z, _seeds, _truth):
        seen.setdefault("readouts", []).append(list(z))
        out = {
            "spread": {"speed": 1.0, "reversal": of(reversal)},
            "time_on_lawn": of(lawn),
            "D_fit": of(d_fit),
        }
        stage, g = seen.get("at", (None, None))
        if at is not None and stage == 1:
            out |= at.get((seen["start"], g), {})
        return out

    def fresh(x0, sigma0, seed, popsize):
        seen["fresh"].append({"x0": list(x0), "sigma0": sigma0, "seed": seed, "popsize": popsize})
        if sigma0 == fit.STAGES[0].sigma0:
            # Stage 1 opens a start, which its CMA seed names: 1, then 11, 13, ...
            seen["start"] = 0 if seed == 1 else (seed - 11) // 2 + 1
        return _FRESH(x0, sigma0, seed, popsize)

    def carried(strategy, x0, seed, popsize):
        new = _CARRIED(strategy, x0, seed, popsize)
        seen["carried"].append(
            {
                "sigma": float(strategy.sigma),
                "stds": np.asarray(strategy.stds).copy(),
                "x0": list(x0),
                "new_sigma": float(new.sigma),
                "new_stds": np.asarray(new.stds).copy(),
                "new_mean": np.asarray(new.result.xfavorite).copy(),
            },
        )
        return new

    monkeypatch.setattr(fit, "BUILD", build)
    monkeypatch.setattr(fit, "entries", _fake_entries)
    monkeypatch.setattr(fit, "_generation", generation)
    monkeypatch.setattr(fit, "readout", readout)
    monkeypatch.setattr(fit, "noise_step", lambda *_a: [_record(truth), _record(truth)])
    monkeypatch.setattr(fit, "carried", carried)
    monkeypatch.setattr(fit, "fresh", fresh)
    monkeypatch.setattr(fit, "head", lambda: "head")
    monkeypatch.setattr(fit, "manifest_digest", lambda: "digest")
    sizes = {"animals": 1, "popsize": 4, "first_animals": 1, "first_popsize": 3}
    layout = fit.Layout(1, **sizes)
    starts = len(hold_bars)
    run = fit.Run(
        "fake",
        layout,
        (1, 2, 3, 4),
        11,
        generations=generations,
        hold_readouts=readings[0] if readings else (hold,),
        hold_bars=hold_bars,
        descent_readouts=readings[1] if readings else (descent or generations[1],),
        descent_bars=descent_bars or (fit.DESCENT_BAR,) * starts,
        reach_readouts=readings[2] if readings else (generations[1],),
        reach_bars=reach_bars or (fit.REACH_BAR,) * starts,
        kill_rules=kill_rules,
    )
    record = fit.procedure(run, truth, pool=None)
    return {"seen": seen, "record": record, "run": run, "truth": truth}


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bands: dict[str, Any]) -> dict:
    """Run the procedure on fake simulations, its one start holding the lawn."""
    return _fake(monkeypatch, tmp_path, bands)


def test_stage_one_fits_only_its_subset(fake_run: dict) -> None:
    """Covers "Stage 1 fits only its subset": every other parameter stays at 0.5."""
    subset = fit.subset_indices(_fake_entries())
    assert subset == [0, 1, 2, 3, 6]
    for z in fake_run["seen"]["candidates"][0]:
        assert [z[i] for i in range(7) if i not in subset] == [0.5, 0.5]
    # 2a starts fresh from stage 1's answer, the timescale set to 1 s (the fake entry's 0.123).
    answer = fake_run["record"]["answers"]["1"]
    assert answer[5] == 0.5
    starts = fake_run["seen"]["fresh"]
    assert [s["sigma0"] for s in starts] == [0.2, 0.1, 0.05]
    assert starts[0]["x0"] == [0.5] * 5
    assert starts[1]["x0"] == [*answer[:5], 0.123, answer[6]]


def test_2b_restarts_the_search_at_2as_answer(fake_run: dict) -> None:
    """Covers "2b restarts the search at 2a's answer": isotropic at 0.05, on the third seed."""
    answers = fake_run["record"]["answers"]
    run = fake_run["run"]
    two_b = fake_run["seen"]["fresh"][2]
    assert two_b == {
        "x0": answers["2a"],
        "sigma0": 0.05,
        "seed": run.cma_seeds[2],
        "popsize": run.layout.popsize,
    }
    assert fit.STAGES[2].name == "2b"
    assert fit.STAGES[2].sigma0 == 0.05
    # The new strategy is isotropic: every per-parameter step is sigma0, whatever 2a's were.
    es = _FRESH(two_b["x0"], two_b["sigma0"], two_b["seed"], two_b["popsize"])
    assert es.sigma == pytest.approx(0.05)
    assert np.allclose(es.stds, 0.05)
    assert np.allclose(es.result.xfavorite, answers["2a"])
    # 2a's search state is handed to no stage.
    assert not [h for h in fake_run["seen"]["carried"] if np.allclose(h["new_mean"], answers["2a"])]


def test_each_stage_reads_its_start_point_first(fake_run: dict) -> None:
    """Each stage's first readout reads its start point in the box, not CMA's internal mean."""
    answers = fake_run["record"]["answers"]
    readouts = fake_run["seen"]["readouts"]
    # Stage 1 reads 0, 3; 2a reads 0, 3; 2b reads 0, 3; the final stage reads 0, 10, ..., 40, 45.
    starts = [readouts[0], readouts[2], readouts[4]]
    assert starts[0] == [0.5] * 7
    assert np.allclose(starts[2], answers["2a"], atol=1e-12)
    assert min(answers["2a"]) < 0.05  # near a bound, where the two differ


def test_a_later_stage_keeps_the_search_steps(fake_run: dict) -> None:
    """Covers "The final stage keeps 2b's search steps": sigma, steps and mean carried once."""
    hands = fake_run["seen"]["carried"]
    assert len(hands) == 1  # 2b -> final; 2b starts its own search
    record = fake_run["record"]
    answers = record["answers"]
    for hand, previous in zip(hands, ("2b",), strict=True):
        assert hand["new_sigma"] == pytest.approx(hand["sigma"])
        assert np.allclose(hand["new_stds"], hand["stds"])
        assert np.allclose(hand["new_mean"], answers[previous])
        # The strategy handed on is 2b's own at its end, not 2a's.
        assert hand["sigma"] == pytest.approx(record["histories"][previous]["sigmas"][-1])
        assert hand["sigma"] != pytest.approx(record["histories"]["2a"]["sigmas"][-1])


def test_a_plateau_does_not_stop_a_stage(fake_run: dict) -> None:
    """Covers "A plateau does not stop a stage": the final stage runs its count, plateau read."""
    final = fake_run["record"]["histories"]["final"]["medians"]
    assert len(final) == 45
    assert fake_run["record"]["plateau_generation"] is not None
    assert fake_run["record"]["plateau_generation"] <= 45
    assert fake_run["record"]["noise"]["populations"] == 2


@pytest.mark.parametrize(("lawn", "share"), [(None, 0.0), (0.5, 1.0), (0.95, 0.0)])
def test_each_generation_records_who_pays_the_lawn_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bands: dict[str, Any],
    lawn: float | None,
    share: float,
) -> None:
    """Every stage records, each generation, the share of its candidates paying the lawn guard."""
    done = _fake(monkeypatch, tmp_path, bands, candidate_lawn=lawn)
    histories = done["record"]["histories"]
    assert set(histories) == {s.name for s in fit.STAGES}
    for name, n in zip(histories, done["run"].generations, strict=True):
        assert histories[name]["lawn_guarded"] == [share] * n


def test_a_resume_with_changed_settings(fake_run: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers "A resume with changed settings": the saved state is refused."""
    run = fake_run["run"]
    changed = dataclasses.replace(run, generations=(3, 3, 3, 46))
    with pytest.raises(RuntimeError, match="other settings"):
        fit.procedure(changed, fake_run["truth"], pool=None)
    # A changed bar or readout of any of 2a's readings, or other restart seeds, is a changed run.
    for other in (
        {"hold_bars": (0.6,)},
        {"descent_bars": (4.0,)},
        {"descent_readouts": (2,)},
        {"reach_bars": (2.4,)},
        {"reach_readouts": (2,)},
        {"restart_seed": 13},
    ):
        with pytest.raises(RuntimeError, match="other settings"):
            fit.procedure(dataclasses.replace(run, **other), fake_run["truth"], pool=None)
    # So is a changed lawn guard, its bar or its weight.
    for name, value in (("LAWN_GUARD_BELOW", 0.9), ("LAWN_PENALTY", 200.0)):
        with monkeypatch.context() as patch:
            patch.setattr(fit, name, value)
            with pytest.raises(RuntimeError, match="other settings"):
                fit.procedure(run, fake_run["truth"], pool=None)
    # So is a stage's start: 2b carrying 2a's search state, or at another sigma0.
    for sigma0 in (None, 0.1):
        stages = list(fit.STAGES)
        stages[2] = dataclasses.replace(stages[2], sigma0=sigma0)
        with monkeypatch.context() as patch:
            patch.setattr(fit, "STAGES", tuple(stages))
            with pytest.raises(RuntimeError, match="other settings"):
                fit.procedure(run, fake_run["truth"], pool=None)
    # Resumed with its own settings, the finished run returns its answer unchanged.
    again = fit.procedure(run, fake_run["truth"], pool=None)
    assert again["answer"] == fake_run["record"]["answer"]


def test_the_pilot_kills_on_the_mean_of_its_last_readouts() -> None:
    """Stage 1's kill rule reads the mean of its last three readouts, not the last one alone."""
    stage1, stage2a = fit.STAGES[0], fit.STAGES[1]
    good = {"spread": {"speed": 1.0, "reversal": 1.0}, "time_on_lawn": 0.85, "D_fit": 2.0}
    dip = {**good, "time_on_lawn": 0.60}
    # A last readout below the bar with a mean above it (0.767) does not kill; the reverse does.
    assert fit._kill(stage1, {120: dip, 130: good, 140: good, 150: dip}) is None
    assert fit._kill(stage1, {130: dip, 140: dip, 150: {**good, "time_on_lawn": 0.95}}) is not None
    pinned = {**good, "spread": {"speed": 1.0, "reversal": 0.2}}
    assert fit._kill(stage1, {130: pinned, 140: pinned, 150: good}) is not None
    # 2a's bar is a reading of every run's starts, so no kill rule reads 2a, or any later stage.
    high = {**good, "D_fit": 9.0}
    assert fit._kill(stage2a, {280: high, 290: high, 300: high}) is None
    assert fit._kill(fit.STAGES[2], {200: high}) is None


# ---------------------------------------------------------------------------------------------
# The first half's starts
# ---------------------------------------------------------------------------------------------

LONG = {"generations": (3, 20, 3, 45), "hold": 10}


def test_a_start_that_does_not_hold_the_lawn_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bands: dict[str, Any],
) -> None:
    """Covers "A start that does not hold the lawn is dropped": the next starts from the centre."""
    out = _fake(monkeypatch, tmp_path, bands, **LONG, hold_bars=(0.95,) * 3, lawn=(0.9, 0.97))
    record, seen = out["record"], out["seen"]
    assert record["kept"] == 1
    assert record["killed"] is None
    assert record["answer"] is not None
    dropped, kept = record["starts"]
    assert (dropped["start"], dropped["kept"], kept["start"], kept["kept"]) == (0, False, 1, True)
    assert dropped["holding"] == {
        "name": "lawn",
        "generations": [10],
        "time_on_lawn": 0.9,
        "bar": 0.95,
        "holds": False,
        "passes": False,
    }
    assert dropped["readings"] == {"lawn": dropped["holding"]}
    assert "on the lawn 0.900" in dropped["dropped"]
    assert kept["holding"]["holds"] is True
    assert kept["dropped"] is None
    # The kept start passed all three of 2a's readings.
    assert [(r["name"], r["passes"]) for r in kept["readings"].values()] == [
        ("lawn", True),
        ("descent", True),
        ("reach", True),
    ]
    # The dropped start stopped at its reading, with everything it produced; the kept one ran on.
    assert len(dropped["histories"]["1"]["medians"]) == 3
    assert len(dropped["histories"]["2a"]["medians"]) == 10
    assert sorted(dropped["readouts"]["2a"]) == ["0", "10"]
    assert len(dropped["mean_at_drop"]) == 7
    assert dropped["answers"]["1"] != record["answers"]["1"]
    assert len(record["histories"]["2a"]["medians"]) == 20
    assert set(record["histories"]) == {"1", "2a", "2b", "final"}
    # Each start is fresh from the box's centre, on its own CMA seeds and at stage 1's own size;
    # 2b then starts its own search, on the run's seed whichever start was kept.
    fresh = seen["fresh"]
    assert [f["sigma0"] for f in fresh] == [0.2, 0.1, 0.2, 0.1, 0.05]
    assert [f["seed"] for f in fresh] == [1, 2, 11, 12, 3]
    assert [f["popsize"] for f in fresh] == [3, 4, 3, 4, 4]
    assert fresh[4]["x0"] == record["answers"]["2a"]
    assert fresh[2]["x0"] == [0.5] * 5
    assert (dropped["cma_seeds"], kept["cma_seeds"]) == ([1, 2], [11, 12])
    # Every start's 2a opens with the speed noise's timescale at 1 s (the fake entry's 0.123).
    assert fresh[1]["x0"][5] == fresh[3]["x0"][5] == 0.123
    # The dropped start's kept mean is the one its last readout read (stage 1: 0, 3; 2a: 0, 10).
    assert dropped["mean_at_drop"] == seen["readouts"][3]
    # Each start's candidates go to its own store and seeds; the later stages are no start's.
    stores = {(stage, a, name) for stage, _g, a, name in seen["stores"]}
    assert stores == {
        (0, 0, "start_0_stage_1.jsonl"),
        (1, 0, "start_0_stage_2a.jsonl"),
        (0, 1, "start_1_stage_1.jsonl"),
        (1, 1, "start_1_stage_2a.jsonl"),
        (2, 0, "stage_2b.jsonl"),
        (3, 0, "stage_final.jsonl"),
    }
    assert max(g for stage, g, a, _n in seen["stores"] if (stage, a) == (1, 0)) == 10


def test_each_start_is_read_against_its_own_bar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bands: dict[str, Any],
) -> None:
    """A start's reading is held against that start's bar, not the first one's."""
    later = {**LONG, "hold_bars": (0.95, 0.5), "lawn": (0.9, 0.6)}
    assert _fake(monkeypatch, tmp_path / "a", bands, **later)["record"]["kept"] == 1
    first = {**LONG, "hold_bars": (0.5, 0.95), "lawn": (0.6, 0.9)}
    record = _fake(monkeypatch, tmp_path / "b", bands, **first)["record"]
    assert record["kept"] == 0
    assert len(record["starts"]) == 1


def test_the_checkpoints_stop_only_the_pilot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bands: dict[str, Any],
) -> None:
    """Covers "A registered run misses a checkpoint" and "A start's 2a ends above the bar"."""
    bars = {**LONG, "hold_bars": (0.95, 0.95, 0.95), "lawn": (0.9, 0.97)}
    # 2a ending at D_fit 2.8 in the second start: dropped at its bar in every run, the pilot's too,
    # and the third start, at D_fit 1.0, kept.
    late = {**bars, "d_fit": (1.0, 2.8, 1.0)}
    for name, kill in (("registered", False), ("pilot", True)):
        record = _fake(monkeypatch, tmp_path / name, bands, **late, kill_rules=kill)["record"]
        assert (record["killed"], record["kept"]) == (None, 2)
        assert record["answer"] is not None
        assert record["checkpoints"]["2a"]["kill"] is None
        assert [a["dropped"] is None for a in record["starts"]] == [False, False, True]
        assert record["starts"][1]["dropped"] == "2a: D_fit 2.80 at generations [20] (bar 2.5)"
    # Stage 1 below its bar: no start is dropped for it, and the pilot stops at whichever start.
    for n, spreads in enumerate(((0.2,), (1.0, 0.2))):
        low = {**bars, "reversal": spreads}
        registered = _fake(monkeypatch, tmp_path / f"registered{n}", bands, **low)["record"]
        assert registered["answer"] is not None
        assert registered["checkpoints"]["1"]["kill"].startswith("stage 1")
        pilot = _fake(monkeypatch, tmp_path / f"pilot{n}", bands, **low, kill_rules=True)["record"]
        assert pilot["killed"].startswith("stage 1")
        assert (pilot["start"], pilot["kept"], pilot["answer"]) == (n, None, None)
        assert [a["kept"] for a in pilot["starts"]] == [False] * n
        assert set(pilot["histories"]) == {"1"}


def test_no_start_holds_the_lawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bands: dict[str, Any],
) -> None:
    """Covers "No start holds the lawn": the run has no answer, and no later stage runs."""
    out = _fake(monkeypatch, tmp_path, bands, **LONG, hold_bars=(0.95, 0.95), lawn=(0.9,))
    record = out["record"]
    assert record["killed"] == "no start of 2 held the lawn and reached 2a's bar"
    assert record["kept"] is None
    assert record["answer"] is None
    assert [a["kept"] for a in record["starts"]] == [False, False]
    assert record["histories"] == {}
    assert set(out["seen"]["candidates"]) == {0, 1}
    # Read again, the finished run starts nothing more.
    again = fit.procedure(out["run"], out["truth"], pool=None)
    assert again["starts"] == record["starts"]
    assert len(out["seen"]["fresh"]) == 4


def test_the_lawn_holding_reading_is_a_mean() -> None:
    """Covers "The lawn-holding reading is a mean": of its three readouts, and of no other."""
    readouts = {g: {"time_on_lawn": 0.2} for g in (0, 10, 20, 30, 70)}
    for g, t in zip((40, 50, 60), (0.93, 0.95, 0.98), strict=True):
        readouts[g] = {"time_on_lawn": t}
    reading = fit.lawn_holding(readouts, (40, 50, 60), 0.95)
    assert reading["time_on_lawn"] == pytest.approx(0.9533333)
    assert reading["holds"] is True
    # A last readout above the bar does not carry a mean below it, and an undefined one counts as 0.
    low = {40: {"time_on_lawn": 0.90}, 50: {"time_on_lawn": 0.94}, 60: {"time_on_lawn": 0.99}}
    assert fit.lawn_holding(low, (40, 50, 60), 0.95)["holds"] is False
    undefined = {40: {"time_on_lawn": None}, 50: {"time_on_lawn": 1.0}, 60: {"time_on_lawn": 1.0}}
    assert fit.lawn_holding(undefined, (40, 50, 60), 0.95)["time_on_lawn"] == pytest.approx(2 / 3)
    exact = {g: {"time_on_lawn": 0.95} for g in (40, 50, 60)}
    assert fit.lawn_holding(exact, (40, 50, 60), 0.95)["holds"] is True


# 2a reads the lawn at 10, the descent at 20 and the bar at 30.
THREE = {"generations": (3, 30, 3, 45), "hold": 10, "descent": 20, "hold_bars": (0.5,) * 3}


def test_a_start_that_misses_the_descent_or_the_bar_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bands: dict[str, Any],
) -> None:
    """Covers "A start that misses 2a's descent or bar is dropped", each at its own reading."""
    # Start 0 at D_fit 5.0 misses the descent (4.25); start 1 at 3.0 passes it and misses the bar
    # (2.5); start 2 at 2.0 passes both.
    out = _fake(monkeypatch, tmp_path, bands, **THREE, d_fit=(5.0, 3.0, 2.0))
    record, seen = out["record"], out["seen"]
    assert (record["killed"], record["kept"]) == (None, 2)
    descent, bar, kept = record["starts"]
    assert descent["dropped"] == "2a: D_fit 5.00 at generations [20] (bar 4.25)"
    assert bar["dropped"] == "2a: D_fit 3.00 at generations [30] (bar 2.5)"
    assert kept["dropped"] is None
    # Each dropped start holds every reading it took, and stopped at the one it failed.
    assert [(n, r["passes"]) for n, r in descent["readings"].items()] == [
        ("lawn", True),
        ("descent", False),
    ]
    assert descent["readings"]["descent"] == {
        "name": "descent",
        "generations": [20],
        "D_fit": 5.0,
        "bar": 4.25,
        "passes": False,
    }
    assert [(n, r["passes"]) for n, r in bar["readings"].items()] == [
        ("lawn", True),
        ("descent", True),
        ("reach", False),
    ]
    assert [len(a["histories"]["2a"]["medians"]) for a in (descent, bar)] == [20, 30]
    assert sorted(bar["readouts"]["2a"], key=int) == ["0", "10", "20", "30"]
    assert all(r["passes"] for r in kept["readings"].values())
    # The kept start's 2a is the record's; the dropped ones ran no later stage.
    assert len(record["histories"]["2a"]["medians"]) == 30
    assert set(record["histories"]) == {"1", "2a", "2b", "final"}
    assert [f["seed"] for f in seen["fresh"]] == [1, 2, 11, 12, 13, 14, 3]
    assert all(f["x0"] == [0.5] * 5 for f in seen["fresh"][:6:2])
    assert seen["fresh"][6]["x0"] == record["answers"]["2a"]
    # A start is read against its own bars: with start 1's bar at 3.0, it is kept.
    again = _fake(
        monkeypatch,
        tmp_path / "own",
        bands,
        **THREE,
        d_fit=(5.0, 3.0, 2.0),
        reach_bars=(2.5, 3.0, 2.5),
    )
    assert again["record"]["kept"] == 1
    # And with start 1's descent and bar at 6.0, start 1 at D_fit 5.0 passes both and is kept.
    own_descent = _fake(
        monkeypatch,
        tmp_path / "own_descent",
        bands,
        **THREE,
        d_fit=(5.0, 5.0, 2.0),
        descent_bars=(4.25, 6.0, 4.25),
        reach_bars=(2.5, 6.0, 2.5),
    )
    assert own_descent["record"]["kept"] == 1
    # Every start missing the bar: no answer, and no later stage runs.
    none = _fake(monkeypatch, tmp_path / "none", bands, **THREE, d_fit=(3.0,))["record"]
    assert (none["killed"], none["kept"], none["answer"]) == (
        fit.NO_START_KEPT.format(3),
        None,
        None,
    )
    assert none["histories"] == {}
    assert [a["readings"]["reach"]["passes"] for a in none["starts"]] == [False] * 3


def test_each_reading_in_a_run_is_a_mean_of_its_readouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bands: dict[str, Any],
) -> None:
    """In a run, each of 2a's readings takes the mean of all its readouts, at its last one."""
    # 2a of 60 generations reads the lawn at 10 and 20, the descent at 30 and 40, the bar at 50, 60.
    shape = {
        "generations": (3, 60, 3, 45),
        "readings": ((10, 20), (30, 40), (50, 60)),
        "hold_bars": (0.9,) * 4,
        "lawn": (0.95,),
    }
    # Each start's last readout passes its reading and the mean of its two does not: start 0 at the
    # lawn (0.6, then 1.0), start 1 at the descent (8.0, then 2.0), start 2 at the bar (5.0, 1.0).
    at = {
        (0, 10): {"time_on_lawn": 0.6},
        (0, 20): {"time_on_lawn": 1.0},
        (1, 30): {"D_fit": 8.0},
        (1, 40): {"D_fit": 2.0},
        (2, 50): {"D_fit": 5.0},
        (2, 60): {"D_fit": 1.0},
    }
    record = _fake(monkeypatch, tmp_path, bands, **shape, at=at)["record"]
    assert record["kept"] == 3
    lawn, descent, bar, kept = record["starts"]
    assert lawn["readings"]["lawn"]["time_on_lawn"] == pytest.approx(0.8)
    assert descent["readings"]["descent"]["D_fit"] == pytest.approx(5.0)
    assert bar["readings"]["reach"]["D_fit"] == pytest.approx(3.0)
    assert [r["generations"] for r in kept["readings"].values()] == [[10, 20], [30, 40], [50, 60]]
    # Each was taken at its last readout, and the start dropped there.
    assert [len(a["histories"]["2a"]["medians"]) for a in (lawn, descent, bar)] == [20, 40, 60]


def test_the_descent_reading_is_a_mean() -> None:
    """Covers "The descent reading is a mean": of its three readouts, at most its bar."""
    readouts = {g: {"D_fit": 9.0} for g in (0, 120, 160)}
    for g, d in zip((130, 140, 150), (4.0, 4.5, 4.2), strict=True):
        readouts[g] = {"D_fit": d}
    reading = fit.descent_reading(readouts, (130, 140, 150), 4.25)
    assert reading["D_fit"] == pytest.approx(4.2333333)
    assert reading["passes"] is True
    # A last readout under the bar does not carry a mean above it.
    high = {130: {"D_fit": 4.6}, 140: {"D_fit": 4.5}, 150: {"D_fit": 3.9}}
    assert fit.descent_reading(high, (130, 140, 150), 4.25)["passes"] is False
    # An undefined D_fit is infinitely far, and readouts all at the bar pass.
    undefined = {130: {"D_fit": None}, 140: {"D_fit": 1.0}, 150: {"D_fit": 1.0}}
    assert fit.descent_reading(undefined, (130, 140, 150), 4.25)["D_fit"] == math.inf
    assert fit.descent_reading(undefined, (130, 140, 150), 4.25)["passes"] is False
    assert fit.descent_reading(undefined, (130, 140, 150), math.inf)["passes"] is True
    exact = {g: {"D_fit": 2.5} for g in (280, 290, 300)}
    assert fit.descent_reading(exact, (280, 290, 300), 2.5)["passes"] is True
    # The registered readings, in order, on every registered run.
    for run in (fit.RUNS[n] for n in ("pilot", "recovery", "fit", "loo")):
        assert run.readings(5) == [
            ("lawn", (40, 50, 60), 0.95),
            ("descent", (130, 140, 150), 4.25),
            ("reach", (280, 290, 300), 2.5),
        ]


@pytest.mark.parametrize(
    "interrupt",
    [
        (1, 15, 0),  # between the lawn and the descent, in the start that misses the descent
        (0, 1, 1),  # the descent's drop saved, the next start opened
        (1, 25, 1),  # between the descent and the bar, in the start that misses the bar
        (1, 5, 2),  # the bar's drop saved, inside the kept start's 2a
        (2, 2, 0),  # past the first half
    ],
)
def test_a_run_resumes_across_the_descent_and_the_bar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bands: dict[str, Any],
    interrupt: tuple[int, int, int],
) -> None:
    """A run stopped around a descent or bar drop resumes to an uninterrupted run's answer."""
    shape = {**THREE, "d_fit": (5.0, 3.0, 2.0)}
    whole = _fake(monkeypatch, tmp_path / "whole", bands, **shape)["record"]
    with pytest.raises(KeyboardInterrupt):
        _fake(monkeypatch, tmp_path / "cut", bands, **shape, interrupt=interrupt)
    again = _fake(monkeypatch, tmp_path / "cut", bands, **shape)
    resumed = again["record"]
    assert resumed["answer"] == whole["answer"]
    assert resumed["histories"] == whole["histories"]
    assert _timeless(resumed["starts"]) == _timeless(whole["starts"])
    # It resumed, not restarted: simulation began at the interrupted generation, and no earlier
    # start's first half ran again.
    stage, generation, start = interrupt
    stores = again["seen"]["stores"]
    assert min((s, g) for s, g, a, _n in stores if a == start or s >= 2) == (stage, generation)
    assert not [x for x in stores if x[0] < 2 and x[2] < start]
    assert stage >= 2 or not [x for x in stores if x[0] < 2 and x[2] == start and x[0] < stage]


def _timeless(starts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in a.items() if k != "seconds"} for a in starts]


RESUMED = {**LONG, "hold_bars": (0.95, 0.95), "lawn": (0.9, 0.97)}


@pytest.mark.parametrize(
    "shape",
    [
        RESUMED,  # dropped at the lawn
        {**THREE, "d_fit": (5.0, 2.0)},  # dropped at the descent
        {**THREE, "d_fit": (3.0, 2.0)},  # dropped at the bar
    ],
    ids=["lawn", "descent", "bar"],
)
def test_an_interrupted_drop_resumes_to_the_same_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bands: dict[str, Any],
    shape: dict[str, Any],
) -> None:
    """Covers "An interrupted drop resumes to the same answer", at each of 2a's readings.

    The drop is made again from the reading the saved state holds.
    """
    whole = _fake(monkeypatch, tmp_path / "whole", bands, **shape)["record"]
    drop = fit._drop
    calls = []

    def interrupted(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        drop(*args, **kwargs)

    monkeypatch.setattr(fit, "_drop", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _fake(monkeypatch, tmp_path / "cut", bands, **shape)
    again = _fake(monkeypatch, tmp_path / "cut", bands, **shape)
    resumed = again["record"]
    assert len(calls) == 2
    assert resumed["answer"] == whole["answer"]
    assert resumed["histories"] == whole["histories"]
    assert _timeless(resumed["starts"]) == _timeless(whole["starts"])
    # It resumed: nothing of the first start ran again, and only the second start was opened
    # (then 2b's own search).
    assert [f["seed"] for f in again["seen"]["fresh"]] == [11, 12, 3]
    assert not [x for x in again["seen"]["stores"] if x[2] == 0 and x[0] < 2]


@pytest.mark.parametrize(
    "interrupt",
    [
        (0, 1, 1),  # the drop saved, the next start opened
        (1, 5, 1),  # inside the next start's 2a, before its reading
        (1, 11, 1),  # the start kept, 2a running on
        (2, 2, 0),  # past the first half
    ],
)
def test_a_run_resumes_inside_any_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bands: dict[str, Any],
    interrupt: tuple[int, int, int],
) -> None:
    """A run stopped anywhere after a drop resumes to an uninterrupted run's starts and answer."""
    whole = _fake(monkeypatch, tmp_path / "whole", bands, **RESUMED)["record"]
    with pytest.raises(KeyboardInterrupt):
        _fake(monkeypatch, tmp_path / "cut", bands, **RESUMED, interrupt=interrupt)
    again = _fake(monkeypatch, tmp_path / "cut", bands, **RESUMED)
    resumed = again["record"]
    assert resumed["answer"] == whole["answer"]
    assert resumed["histories"] == whole["histories"]
    assert _timeless(resumed["starts"]) == _timeless(whole["starts"])
    assert [a["kept"] for a in resumed["starts"]] == [False, True]
    # Nothing saved ran again: the resumed call simulated only from the interrupted generation on.
    stage, generation, _start = interrupt
    assert min((s, g) for s, g, _a, _n in again["seen"]["stores"]) == (stage, generation)


def test_a_starts_candidates_are_simulated_on_its_own_animals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Covers "Each start has its own animals": the start reaches the simulation's seeds."""
    seen = []

    def simulate(z, seeds, *, assays, mutants):
        seen.append({"seeds": list(seeds), "assays": assays, "mutants": mutants})
        return {"statistics": {}}

    monkeypatch.setattr(fit, "simulate_point", simulate)
    pool = SimpleNamespace(map=lambda job, items: [job(i) for i in items])
    run = fit.RUNS["fit"]
    z = [0.5] * 3
    for stage, generation, start in ((0, 2, 3), (1, 60, 3), (1, 61, 3)):
        path = tmp_path / f"{stage}_{generation}.jsonl"
        records = fit._generation(pool, run, stage, generation, [z, z], path, start)
        assert [r["member"] for r in records] == [0, 1]
        assert seen[-1]["seeds"] == run.layout.candidate_seeds(stage, generation, 1, start)
        assert seen[-1]["assays"] == fit.FIT_ASSAYS
    assert seen[0]["seeds"][0] == 1_280_000_000 + 250_000 * 3 + 64 * 24
    # A stored candidate is reused at the same vector, and refused at another.
    path = tmp_path / "0_2.jsonl"
    count = len(seen)
    assert fit._generation(pool, run, 0, 2, [z, z], path, 3) == json.loads(
        "[" + ",".join(path.read_text().splitlines()) + "]",
    )
    assert len(seen) == count
    with pytest.raises(ValueError, match="another vector"):
        fit._generation(pool, run, 0, 2, [[0.4] * 3, z], path, 3)
    # Every generation of 2a is the start's own: before generation 60's reading and after it.
    own = 1_280_000_000 + 20_000_000 + 2_880_000 * 3
    assert seen[2]["seeds"][0] == own + 200 * 48 * 59
    assert seen[4]["seeds"][0] == own + 200 * 48 * 60


def test_a_reading_without_a_readout_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bands: dict[str, Any],
) -> None:
    """A reading without a readout, out of order or past 2a, or 2a past its seeds, is refused."""
    with pytest.raises(ValueError, match="has no readout"):
        _fake(monkeypatch, tmp_path / "a", bands, generations=(3, 20, 3, 45), hold=7)
    with pytest.raises(ValueError, match="has no readout"):
        _fake(monkeypatch, tmp_path / "b", bands, generations=(3, 20, 3, 45), hold=10, descent=15)
    with pytest.raises(ValueError, match="out of order"):
        _fake(monkeypatch, tmp_path / "c", bands, generations=(3, 30, 3, 45), hold=20, descent=10)
    truth = fit.band_truth(bands)
    layout = fit.Layout(1, animals=1, popsize=4)
    run = fit.Run("fake", layout, (1, 2, 3, 4), 11, generations=(3, 30, 3, 45))
    shapes = (
        # The registered readings' generations, past this run's 2a of 30.
        ({}, "past 2a's end"),
        ({"hold_readouts": (10,), "descent_readouts": (20,), "reach_readouts": (40,)}, "past 2a"),
        # Bars for a different number of starts.
        ({"hold_readouts": (10,), "descent_readouts": (20,), "reach_readouts": (30,)}, "bars for"),
    )
    for change, match in shapes:
        bars = {"descent_bars": (4.25,)} if match == "bars for" else {}
        with pytest.raises(ValueError, match=match):
            fit.procedure(dataclasses.replace(run, **change, **bars), truth, pool=None)
    # 2a's generations past a start's slice of seeds.
    fitting = {"hold_readouts": (10,), "descent_readouts": (20,), "reach_readouts": (30,)}
    tight = dataclasses.replace(layout, two_a_stride=4 * 30 - 1)
    with pytest.raises(ValueError, match="do not fit"):
        fit.procedure(dataclasses.replace(run, layout=tight, **fitting), truth, pool=None)


def test_a_registered_run_with_no_answer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Covers "A registered fit with no answer fails" and the recovery's "no answer" scenario."""
    reason = fit.NO_START_KEPT.format(6)
    killed = {"killed": reason, "answer": None}
    monkeypatch.setattr(fit, "BUILD", tmp_path)
    for name in ("FIT", "RECOVERY", "PILOT", "LOO"):
        monkeypatch.setattr(fit, f"{name}_PATH", tmp_path / f"{name.lower()}.json")
    # The fit: nothing to grade, which does not stop the chain, and the read is a fail.
    (tmp_path / "fit_procedure.json").write_text(json.dumps(killed))
    assert fit._grade(1) == 0
    assert not (tmp_path / "fit_grades.json").exists()
    assert fit._read() == 0
    reading = json.loads((tmp_path / "fit.json").read_text())["verdict"]
    assert reading["passes"] is False
    assert reading["no_answer"] == reason
    # A fit still running has no answer either, and its grade refuses.
    (tmp_path / "fit_procedure.json").write_text(json.dumps({"killed": None, "answer": None}))
    assert fit._grade(1) == 1

    class Pool:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_a: object) -> None:
            return None

    def no_grade(*_a: object) -> None:
        msg = "a run with no answer is not graded"
        raise AssertionError(msg)

    observation = {"statistics": {"a": 1.0}, "delta": {}}
    (tmp_path / "preflight.json").write_text(json.dumps({"observation": observation}))
    monkeypatch.setattr(fit, "RECOVERY_PREFLIGHT_PATH", tmp_path / "preflight.json")
    monkeypatch.setattr(fit, "_pool", lambda _w: Pool())
    monkeypatch.setattr(fit, "procedure", lambda *_a: dict(killed))
    monkeypatch.setattr(fit, "grade", no_grade)
    # The recovery: its gate is not met.
    assert fit._recovery(1) == 1
    gate = json.loads((tmp_path / "recovery.json").read_text())["gate"]
    assert gate["passes"] is False
    assert gate["no_answer"] == reason
    # The pilot: no go. The leave-one-out: no prediction. The smoke: a failure.
    assert fit._pilot(1) == 1
    pilot = json.loads((tmp_path / "pilot.json").read_text())
    assert (pilot["go"], pilot["killed"], pilot["grades"]) == (False, reason, [])
    assert fit._loo(1) == 0
    assert json.loads((tmp_path / "loo.json").read_text())["tax4_predicted"] is None
    assert fit._smoke(1) == 1


# ---------------------------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------------------------


def _grade(*, wild: bool, **mutants: bool) -> dict[str, Any]:
    passes = {"tph-1": True, "pdfr-1": True, "tax-4": True, **mutants}
    return {
        "wild_type_passes": wild,
        "passes": {"activity/onset_rate_per_min": wild},
        "z": {"activity/onset_rate_per_min": -2.0 if wild else -3.3},
        "statistics": {},
        "mutants": {m: {"passes": p} for m, p in passes.items()},
    }


def test_a_fitted_mutant_fails_the_verdict() -> None:
    """Covers "A fitted mutant fails the verdict": the wild type 4 of 5, tax-4 2 of 5."""
    taxes = [True, True, False, False, False]
    grades = [_grade(wild=g < 4, **{"tax-4": t}) for g, t in enumerate(taxes)]
    reading = fit.verdict(grades)
    assert reading["wild_type_passes"] == 4
    assert reading["passes"] is False
    assert reading["diagnosis"]["mutant_failures"]["tax-4"] == 3
    assert reading["diagnosis"]["statistics"]["activity/onset_rate_per_min"][4] == -3.3


def test_pdfr1_does_not_decide() -> None:
    """Covers "pdfr-1 does not decide": pdfr-1 in 1 of 5 leaves a pass."""
    grades = [_grade(wild=True, **{"pdfr-1": g == 0}) for g in range(5)]
    reading = fit.verdict(grades)
    assert reading["passes"] is True
    assert reading["mutant_passes"]["pdfr-1"] == 1
    # Only the first five populations enter the verdict.
    assert fit.verdict([*grades, *[_grade(wild=False)] * 10])["passes"] is True


# ---------------------------------------------------------------------------------------------
# F.0b-ii
# ---------------------------------------------------------------------------------------------


def _preflight(
    monkeypatch: pytest.MonkeyPatch,
    distances: dict[str, Any],
    bands: dict,
) -> dict:
    monkeypatch.setattr(
        fit,
        "recovery_distance",
        lambda _o, r, _b: distances[r.get("kind", "reference")],
    )

    def run(kind: str) -> dict[str, Any]:
        statistics = {"a": 1.0, fit.LAWN_STATISTIC: distances.get("lawn", 0.98)}
        return {
            "kind": kind,
            "statistics": statistics,
            "delta": {"tph-1": {"s": 0.1}},
            "D_fit": distances.get("d_fit", 1.8),
        }

    repeats = [run("floor")] * 10 + [run("oracle")] * 10
    return fit.preflight_record(run("floor"), repeats, [run("prior")], [run("centre")], bands)


def test_the_preflight_checks(monkeypatch: pytest.MonkeyPatch, bands: dict[str, Any]) -> None:
    """Covers "The preflight fails": an oracle gain of 0.80 stops the recovery."""
    good = {"floor": 0.36, "oracle": 0.37, "prior": 6.4, "centre": 5.95, "reference": 1.13}
    record = _preflight(monkeypatch, good, bands)
    assert record["passed"] is True
    assert record["wild_type_reference_gain"] == pytest.approx((6.4 - 1.13) / (6.4 - 0.36))
    assert record["gate_line_D"] == pytest.approx(6.4 - 0.75 * (6.4 - 0.36))
    oracle = 6.4 - 0.80 * (6.4 - 0.36)
    assert _preflight(monkeypatch, {**good, "oracle": oracle}, bands)["checks"]["oracle"] is False
    centre = 6.4 - 0.30 * (6.4 - 0.36)
    assert _preflight(monkeypatch, {**good, "centre": centre}, bands)["passed"] is False
    # Covers "A truth that does not hold the lawn": no start recovering it could be kept.
    assert record["checks"]["holds_lawn"] is True
    off = _preflight(monkeypatch, {**good, "lawn": 0.94}, bands)
    assert (off["checks"]["holds_lawn"], off["passed"]) == (False, False)
    assert off["observation_time_on_lawn"] == 0.94
    # Covers "A truth past 2a's bar": no start recovering it could be kept, and at the bar it can.
    assert (record["checks"]["reaches_bar"], record["observation_D_fit"]) == (True, 1.8)
    for d_fit, reaches in ((2.5, True), (2.6, False), (None, False)):
        far = _preflight(monkeypatch, {**good, "d_fit": d_fit}, bands)
        assert (far["checks"]["reaches_bar"], far["passed"]) == (reaches, reaches)


def test_a_truth_whose_statistic_flips() -> None:
    """Covers "A truth whose statistic flips": a repeat losing a statistic is named."""
    observation = {"statistics": {"a": 1.0, "b": 2.0}, "delta": {"tph-1": {"s": 0.1}}}
    flipped = {"statistics": {"a": 1.0, "b": None}, "delta": {"tph-1": {"s": 0.1}}}
    assert fit.definedness_flips(observation, [observation, flipped]) == ["b"]
    assert fit.definedness_flips(observation, [observation]) == []


def test_the_recovery_falls_short() -> None:
    """Covers "The recovery falls short": a gain of 0.70 does not clear the gate."""
    preflight = {"prior_D": 6.4, "floor": 0.4}
    assert fit.recovery_gate(6.4 - 0.70 * 6.0, preflight)["passes"] is False
    assert fit.recovery_gate(6.4 - 0.80 * 6.0, preflight)["passes"] is True


def test_the_preflight_runs_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Covers "The preflight runs once": an existing record is refused."""
    path = tmp_path / "recovery_preflight.json"
    path.write_text("{}\n")
    monkeypatch.setattr(fit, "RECOVERY_PREFLIGHT_PATH", path)
    assert fit._recovery_preflight(1) == 1


def test_the_recovery_distance_at_the_truth_is_zero(bands: dict[str, Any]) -> None:
    """The recovery distance is 0 at the truth, and a mutant enters at half its weight."""
    truth = fit.band_truth(bands)
    assert fit.recovery_distance(truth, _record(truth), bands) == pytest.approx(0.0)
    moved = _record(truth)
    scale = fit.mutant_scales(bands)["tax-4"]["roaming_fraction"]
    moved["delta"]["tax-4"]["roaming_fraction"] += 2.0 * scale
    # tax-4 grades two statistics: its term is sqrt(4 / 2); weight 2 of 12.
    assert fit.recovery_distance(truth, moved, bands) == pytest.approx(math.sqrt(2 * 2.0 / 12))


# ---------------------------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------------------------

# Every earlier registered or probe block at or above 100,000,000 (the seed table).
EARLIER = (
    (110_000_000, 126_001_000),
    (400_000_000, 442_001_000),
    (470_000_000, 475_100_000),
    (500_000_000, 542_002_000),
    (600_000_000, 646_001_000),
    (700_000_000, 795_001_000),
    (800_000_000, 878_003_000),
    (880_000_000, 880_002_000),
    # The first readiness pilot, stopped at 2a, and the runs that diagnosed it.
    (890_000_000, 890_000_199),
    (900_000_000, 916_000_199),
    (1_010_000_000, 1_055_000_199),
    # The second readiness pilot, which kept no start, and the runs that diagnosed it.
    (1_060_000_000, 1_065_000_199),
    (1_090_000_000, 1_095_000_199),
    (1_100_000_000, 1_118_002_999),
    (1_130_000_000, 1_135_000_199),
    # The third readiness pilot, whose kept start missed 2a's bar, and the check that diagnosed it.
    (1_140_000_000, 1_158_002_999),
    (1_170_000_000, 1_170_003_199),
    # The fourth readiness pilot, whose answer failed its go rule (every start's 2a reserved), the
    # 2b screen that diagnosed it (its readouts reused the pilot's readout animals), and a timing
    # check of the simulation's kernels.
    (1_200_000_000, 1_237_279_999),
    (1_400_000_000, 1_400_575_999),
    (1_410_000_000, 1_410_575_999),
    (1_420_000_000, 1_420_000_207),
    # The fit-first investigation's reserved animals.
    (1_500_000_000, 1_501_000_199),
)
# The CMA seeds the seed tables give earlier pilots, probes and tests: C.2's pilots and probes,
# the speed-following reversals' tests (9601-9605), the sign-scale calibration (9601-9616), the
# fit-first investigation's generators and reserved seeds (9701-9722, 9731, 9732, 9741-9759), and
# the neural readout's ceiling's generators (9770-9779, 9781-9799, 9801-9805).
EARLIER_CMA = frozenset(
    [
        *range(9504, 9508),
        *range(9516, 9526),
        *range(9531, 9535),
        *range(9541, 9551),
        9571,
        9574,
        9581,
        9582,
        *range(9591, 9595),
        *range(9601, 9617),
        *range(9651, 9661),
        *range(9661, 9665),
        *range(9671, 9681),
        *range(9701, 9723),
        9731,
        9732,
        *range(9741, 9760),
        *range(9770, 9780),
        *range(9781, 9800),
        *range(9801, 9806),
        9922,
        9923,
    ],
)


def _blocks() -> dict[str, tuple[int, int]]:
    blocks = {}
    for name, run in fit.RUNS.items():
        for use, span in run.layout.spans(run.generations, run.starts).items():
            blocks[f"{name} {use}"] = span
    n = fit.ANIMALS
    blocks["truth"] = (fit.TRUTH_BASE, fit.TRUTH_BASE + n * (fit.TRUTH_REPEATS + 1) - 1)
    blocks["prior"] = (fit.PRIOR_BASE, fit.PRIOR_BASE + n * fit.PRIOR_DRAWS - 1)
    blocks["centre"] = (fit.CENTRE_BASE, fit.CENTRE_BASE + n * fit.CENTRE_REPEATS - 1)
    blocks["speed noise"] = (fit.SPEED_NOISE_BASE, fit.SPEED_NOISE_BASE + n)
    blocks["stand-in"] = (885_000_000, 887_100_000 + 4 * n - 1)
    return blocks


def test_seed_blocks_do_not_overlap() -> None:
    """Every block is disjoint from the others and from every earlier block in the seed table."""
    ranges = sorted(_blocks().items(), key=lambda kv: kv[1][0])
    for (a, (_, a_end)), (b, (b_start, _)) in itertools.pairwise(ranges):
        assert a_end < b_start, f"{a} overlaps {b}"
    for name, (start, end) in _blocks().items():
        for first, last in EARLIER:
            assert end < first or start > last, f"{name} overlaps an earlier block"


def test_the_longest_stage_stays_inside_its_sub_block() -> None:
    """Covers "The longest stage stays inside its sub-block": 2a's last candidate, every start's."""
    base = 1_280_000_000
    layout = fit.Layout(base)
    for a in range(fit.STARTS):
        last = layout.candidate_seeds(1, 300, fit.POPSIZE - 1, a)[-1]
        assert last == base + 20_000_000 + 2_880_000 * (a + 1) - 1
        if a + 1 < fit.STARTS:
            assert last < layout.candidate_seeds(1, 1, 0, a + 1)[0]
    # The last start's 2a ends inside the run's 40,000,000, before the next run's block.
    last = layout.candidate_seeds(1, 300, fit.POPSIZE - 1, fit.STARTS - 1)[-1]
    assert last < base + 40_000_000 == fit.RUNS["loo"].layout.base
    # The pilot, on its fifth base, follows the leave-one-out's block.
    bases = [fit.RUNS[n].layout.base for n in ("recovery", "fit", "loo", "pilot")]
    assert [b - a for a, b in itertools.pairwise(bases)] == [40_000_000] * 3
    smoke = fit.RUNS["smoke"]
    for start, end in smoke.layout.spans(smoke.generations, smoke.starts).values():
        assert 99_880_000 <= start <= end <= 99_889_999


def test_each_start_has_its_own_animals() -> None:
    """Covers "Each start has its own animals": stage 1 and every generation of 2a, per start."""
    base = 1_280_000_000
    layout = fit.Layout(base)
    run = fit.RUNS["fit"]
    assert (run.starts, run.hold_readouts, set(run.hold_bars)) == (6, (40, 50, 60), {0.95})
    assert (run.descent_readouts, set(run.descent_bars)) == ((130, 140, 150), {4.25})
    assert (run.reach_readouts, set(run.reach_bars)) == ((280, 290, 300), {2.5})
    assert len(run.descent_bars) == len(run.reach_bars) == run.starts
    assert (layout.popsize_of(0), layout.popsize_of(1), layout.popsize_of(3)) == (24, 48, 48)
    assert len(layout.candidate_seeds(0, 1, 0)) == 64
    assert len(layout.candidate_seeds(1, 1, 0)) == 200
    # Stage 1: each start's 150 generations inside its own slice of stage 1's sub-block.
    assert layout.candidate_seeds(0, 2, 5, 3)[0] == base + 250_000 * 3 + 64 * (24 + 5)
    grade_last = layout.grade_seeds(layout.grade_populations - 1)[-1]
    spans = layout.spans(run.generations, run.starts)
    for a in range(run.starts):
        first, last = spans[f"stage 0 start {a}"]
        assert base + 250_000 * a == first <= last < base + 250_000 * (a + 1) <= base + 4_000_000
        # 2a, all 300 generations: each start's own, after the grade's animals.
        first, last = spans[f"stage 1 start {a}"]
        own = base + 20_000_000 + 2_880_000 * a
        assert grade_last < own == first <= last == own + 2_880_000 - 1
        assert layout.candidate_seeds(1, 300, 47, a)[-1] == last
        # Past the lawn's and the descent's readings, a start's animals are still its own.
        for g in (61, 151):
            later = layout.candidate_seeds(1, g, 0, a)
            assert later[0] == own + 200 * 48 * (g - 1)
            assert a == 0 or later != layout.candidate_seeds(1, g, 0, 0)
    # The second half's stages are the kept start's, whichever it is.
    assert spans["stage 2"] == (base + 8_000_000, base + 8_000_000 + 200 * 9_600 - 1)
    assert spans["stage 3"] == (base + 12_000_000, base + 12_000_000 + 200 * 2_400 - 1)
    assert "stage 1" not in spans
    # The first half's CMA seeds are each start's own; the later stages' are the run's.
    assert [run.cma_seed(0, a) for a in range(6)] == [5811, 5841, 5843, 5845, 5847, 5849]
    assert [run.cma_seed(1, a) for a in range(6)] == [5812, 5842, 5844, 5846, 5848, 5850]
    assert (run.cma_seed(2, 4), run.cma_seed(3, 4)) == (5813, 5814)
    # No CMA seed serves twice, and the registered runs' stay in C.2's band, beside the prior's.
    seeds = {
        n: [r.cma_seed(s, a) for s in range(2) for a in range(r.starts)]
        for n, r in fit.RUNS.items()
    }
    for name, run_ in fit.RUNS.items():
        seeds[name] += [run_.cma_seeds[s] for s in (2, 3)]
    flat = [seed for group in seeds.values() for seed in group] + [fit.PRIOR_SEED]
    assert len(flat) == len(set(flat))
    assert not EARLIER_CMA & set(flat), "a run reuses an earlier pilot's or probe's CMA seed"
    for name in ("recovery", "fit", "loo"):
        assert all(5801 <= seed <= 5864 for seed in seeds[name])


def test_only_the_pilot_has_kill_rules() -> None:
    """Covers "A registered run misses a checkpoint": the pilot alone stops on a kill rule."""
    assert fit.RUNS["pilot"].kill_rules is True
    assert [n for n, r in fit.RUNS.items() if r.kill_rules] == ["pilot"]


def test_the_neural_readings_generators() -> None:
    """The paired test's two bootstraps are their own, clear of every earlier generator."""
    bootstraps = {fit.NEURAL_CLASS_BOOTSTRAP_SEED, fit.NEURAL_NEURON_BOOTSTRAP_SEED}
    assert len(bootstraps) == 2
    assert not bootstraps & EARLIER_CMA, "a bootstrap reuses an earlier generator"
    assert 9781 not in bootstraps  # the ceiling's first null rewiring
    # Chance's relabellings reuse the first reading's generator, so the two compare.
    assert fit.NEURAL_CHANCE_SEED == 9731
    runs = {r.cma_seed(s, a) for r in fit.RUNS.values() for s in range(4) for a in range(r.starts)}
    assert not bootstraps & (runs | {fit.PRIOR_SEED})


# ---------------------------------------------------------------------------------------------
# The guards
# ---------------------------------------------------------------------------------------------


def test_a_changed_config_stops_a_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Covers "A changed config stops a run": a manifest file that changes is refused."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "x.py").write_text("x = 1\n")
    (tmp_path / "a.yml").write_text("a: 1\n")
    monkeypatch.setattr(fit, "MANIFEST_FILES", ("a.yml",))
    monkeypatch.setattr(fit, "PACKAGE", "pkg")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(fit.manifest(tmp_path)))
    assert fit.manifest_matches(path, tmp_path)
    (tmp_path / "a.yml").write_text("a: 2\n")
    assert not fit.manifest_matches(path, tmp_path)
    (tmp_path / "a.yml").write_text("a: 1\n")
    (tmp_path / "pkg" / "x.py").write_text("x = 2\n")
    assert not fit.manifest_matches(path, tmp_path)


def _guards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **state: Any) -> None:
    parts = state.get("parts", "")
    launch = tmp_path / "launch.md"
    launch.write_text("# Launch\n" + "".join(f"\n## Part {p}\n" for p in parts))
    records = {
        "speed": state.get("speed", {"passes": True}),
        "pilot": state.get("pilot", {"go": True}),
        "preflight": state.get("preflight", {"passed": True}),
        "recovery": state.get("recovery", {"gate": {"passes": True}}),
    }
    paths = {}
    for name, record in records.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(record))
    monkeypatch.setattr(fit, "LAUNCH_PATH", launch)
    monkeypatch.setattr(fit, "SPEED_NOISE_PATH", paths["speed"])
    monkeypatch.setattr(fit, "PILOT_PATH", paths["pilot"])
    monkeypatch.setattr(fit, "RECOVERY_PREFLIGHT_PATH", paths["preflight"])
    monkeypatch.setattr(fit, "RECOVERY_PATH", paths["recovery"])
    monkeypatch.setattr(fit, "manifest_matches", lambda *_a: state.get("manifest", True))
    monkeypatch.setattr(fit.bg, "v4_is_valid", lambda: True)
    monkeypatch.setattr(fit.cr, "launch_committed", lambda *_a, **_k: state.get("committed", True))
    monkeypatch.setattr(fit, "committed", lambda _p: state.get("preflight_committed", True))


def test_every_stage_refuses_before_its_part(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Covers "The fit refuses before the recovery is recorded" and the earlier parts' guards."""
    _guards(monkeypatch, tmp_path, manifest=False, parts="ABC")
    assert "manifest" in fit.guard("pilot")
    _guards(monkeypatch, tmp_path, committed=False, parts="ABC")
    assert "Part A" in fit.guard("pilot")
    _guards(monkeypatch, tmp_path, speed={"passes": False}, parts="ABC")
    assert "speed noise" in fit.guard("pilot")
    _guards(monkeypatch, tmp_path, parts="A")
    assert fit.guard("pilot") is None
    assert "Part B" in fit.guard("recovery-preflight")
    _guards(monkeypatch, tmp_path, parts="AB")
    assert fit.guard("recovery-preflight") is None
    assert fit.guard("recovery") is None
    for stage in ("run", "grade", "read", "loo"):
        assert "Part C" in fit.guard(stage)
    _guards(monkeypatch, tmp_path, parts="AB", preflight_committed=False)
    assert "uncommitted" in fit.guard("recovery")
    _guards(monkeypatch, tmp_path, parts="AB", preflight={"passed": False})
    assert "not passed" in fit.guard("recovery")
    _guards(monkeypatch, tmp_path, parts="ABC", recovery={"gate": {"passes": False}})
    assert "gate" in fit.guard("run")
    _guards(monkeypatch, tmp_path, parts="ABC")
    assert all(fit.guard(s) is None for s in ("run", "grade", "read", "loo"))


def test_a_failed_pilot_blocks_the_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Covers "A failed pilot blocks the recovery": the go rule's failure refuses the preflight."""
    _guards(monkeypatch, tmp_path, parts="AB", pilot={"go": False})
    assert "go rule" in fit.guard("recovery-preflight")


# ---------------------------------------------------------------------------------------------
# The held-out reading
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_the_real_halves_reproduce_w_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers "The real halves reproduce W's minimum" and "The smoke reads no held-out track"."""
    from quantumnematode.benchmark import references as refs

    bands = bg.load_bands_v4()
    clf = bg.classifier(bands)
    tracks = refs.load_scheer_tracks()
    sets = fit.date_sets(tracks)
    per_date = fit.tallies(tracks, sets["N2_unread"], clf)
    reading = fit.held_out_distances(per_date, [], [], bands)
    recorded = json.loads(fit.W_MINIMUM_PATH.read_text())["w_minimum"]["fitted"]
    assert reading["dates"] == 26
    assert reading["real_halves"]["p95"] == pytest.approx(recorded["minimum_95th"], rel=1e-12)
    assert reading["real_halves"]["median"] == pytest.approx(recorded["median"], rel=1e-12)
    assert len(sets["PD1074_held_out"]) == 573
    # The band reference draws band-cohort animals only.
    picked: list[np.ndarray] = []
    original = fit.tallies
    monkeypatch.setattr(
        fit,
        "tallies",
        lambda t, rows, c: picked.append(rows) or original(t, rows, c),
    )
    reference = fit.band_reference(tracks, clf, populations=2)
    assert all((tracks.side[rows] == refs.BANDS).all() for rows in picked)
    assert len(reference) == 2
    # With its own date sets, the reading never asks for the held-out ones.
    monkeypatch.setattr(fit, "date_sets", lambda _t: pytest.fail("read held-out dates"))
    band_rows = tracks.arm("PD1074_od2_Data")
    band_rows = band_rows[tracks.side[band_rows] == refs.BANDS]
    out = fit.held_out_reading(
        reference,
        sets={"bands": band_rows},
        splits=2,
        reference_populations=1,
    )
    assert out["bands"]["answer"]["n"] == 2 * 2 * 2


def test_the_restart_spread_is_zero_between_equal_answers(bands: dict[str, Any]) -> None:
    """The restart spread reads 0 between identical populations, on W's families and all six."""
    truth = fit.band_truth(bands)
    spread = fit.restart_spread([_record(truth)] * 3, [_record(truth)] * 3, bands)
    assert spread == {"populations": 3, "w_families": 0.0, "all_six": 0.0}


def test_the_noise_free_control_point_drops_the_timescale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the OU noise the timescale leaves the vector, and the centres keep their place."""
    monkeypatch.setattr(fit, "entries", _fake_entries)
    assert fit._without_noise([0, 1, 2, 3, 4, 5, 6]) == [0, 1, 2, 3, 4, 6]
