"""C.1 refitted on benchmark v3: the 25-parameter model, its mutants, the recovery and the seeds.

Covers the controller-fit spec's scenarios "A capped restart is not chosen", "Seed blocks do not
overlap", "The recovery falls short", "The cap comes from the probes" and "tax-4 removes both
weathervane gains".
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import c1_v3_fit as fit  # noqa: E402  # pyright: ignore[reportMissingImports]
import c1b_fit as c1b  # noqa: E402  # pyright: ignore[reportMissingImports]
import cma_recovery_v2 as v2  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_the_model_has_twenty_five_parameters() -> None:
    """The unit box maps to C.1's 20 parameters and the five added ones, each in its range."""
    params = fit.parameters(np.full(fit.DIMENSION, 0.5))
    assert fit.DIMENSION == 25
    assert len(params) == 25
    for name, (low, high) in fit.EXTRA.items():
        assert params[name] == pytest.approx((low + high) / 2)
    assert fit.parameters(np.zeros(25))["kappa_off_food"] == pytest.approx(-50.0)
    with pytest.raises(ValueError, match="25 dimensions"):
        fit.parameters(np.full(20, 0.5))


def test_tax4_removes_both_weathervane_gains() -> None:
    """Covers "tax-4 removes both weathervane gains": the kinematic parameters are kept."""
    params = fit.parameters(np.full(25, 0.7))
    tax4 = fit.mutant_parameters(params, "tax-4")
    assert tax4 is not None
    for name in ("kappa_weathervane", "kappa_off_food", "beta_reversal", "beta_omega"):
        assert tax4[name] == 0.0
    for name in ("v_off_food", "sigma_off_food", "speed_log_sd_dwell", "speed_log_sd_roam", "edge"):
        assert tax4[name] == params[name]
    tph1 = fit.mutant_parameters(params, "tph-1")
    assert tph1 is not None
    assert tph1["a_intake"] == tph1["b_intake"] == 0.0
    assert tph1["kappa_off_food"] == params["kappa_off_food"]
    assert fit.mutant_parameters(params, "pdfr-1") is None


def _blocks() -> dict[str, tuple[int, int]]:
    """Return each registered use's animal-seed range, [first, last]."""
    animals = 200
    last_gen = 16 * (fit.CAP - 1) + 15

    def span(base: int, k_low: int, k_high: int) -> tuple[int, int]:
        return base + animals * k_low, base + animals * k_high + animals - 1

    blocks = {}
    for r in range(fit.RESTARTS):
        blocks[f"fit {r}"] = span(fit.FIT_BASE, fit.FIT_BLOCK * r, fit.FIT_BLOCK * r + last_gen)
        blocks[f"answers {r}"] = span(
            fit.FIT_BASE,
            fit.answer_k(r, 0),
            fit.answer_k(r, fit.RESIMULATIONS - 1),
        )
    for f in range(2 * fit.RECOVERY_RESTARTS):
        blocks[f"recovery fit {f}"] = span(
            fit.RECOVERY_BASE,
            fit.FIT_BLOCK * f,
            fit.FIT_BLOCK * f + last_gen,
        )
        blocks[f"recovery answers {f}"] = span(
            fit.RECOVERY_BASE,
            fit.answer_k(f, 0),
            fit.answer_k(f, fit.RESIMULATIONS - 1),
        )
    blocks["registered"] = span(fit.GRADE_BASE, 0, fit.GRADE_POPULATIONS - 1)
    blocks["selection"] = span(
        fit.SELECT_BASE,
        0,
        fit.select_k(fit.RESTARTS - 1, fit.SELECT_POPULATIONS - 1),
    )
    blocks["truths"] = span(fit.TRUTH_BASE, 0, fit.truth_k(1, fit.TRUTH_REPEATS))
    blocks["prior"] = span(fit.PRIOR_BASE, 0, fit.PRIOR_DRAWS - 1)
    blocks["centre"] = span(fit.CENTRE_BASE, 0, fit.RESIMULATIONS - 1)
    return blocks


def test_seed_blocks_do_not_overlap() -> None:
    """Covers "Seed blocks do not overlap": every registered use's animals are its own."""
    ranges = sorted(_blocks().items(), key=lambda kv: kv[1][0])
    for (a, (_, a_end)), (b, (b_start, _)) in itertools.pairwise(ranges):
        assert a_end < b_start, f"{a} overlaps {b}"
    # The answers sit above every generation's index under the cap.
    assert fit.ANSWER_OFFSET > 16 * (fit.CAP - 1) + 15
    # The probes and the smoke are discarded and below every registered block.
    assert min(start for start, _ in _blocks().values()) > fit.SMOKE_GRADE_BASE + 200 * 5


def test_the_recovery_falls_short() -> None:
    """Covers "The recovery falls short": a median gain of 0.70 does not clear the gate."""
    assert fit.recovery_gate([0.70, 0.70])["passes"] is False
    assert fit.recovery_gate([0.90, 0.60])["passes"] is True
    assert fit.recovery_gate([0.90, 0.60])["median_gain"] == pytest.approx(0.75)
    assert fit.gain(7.0, 1.0, 0.4) == pytest.approx(6.0 / 6.6)


def test_the_preflight_needs_every_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-fit checks: prior D separated from the floor, the oracle high, the centre low."""
    distances = {"floor": 0.4, "oracle": 0.45, "prior": 7.0, "centre": 6.9}
    monkeypatch.setattr(fit, "capped_d", lambda _a, b: distances[b["kind"]])
    runs = [{"kind": "floor"}] * 10 + [{"kind": "oracle"}] * 10
    record = fit.preflight_record([{}], [runs], [{"kind": "prior"}], [{"kind": "centre"}])
    assert record["passed"] is True
    assert record["oracle_median_gain"] > fit.ORACLE_GATE
    distances["centre"] = 2.0
    assert (
        fit.preflight_record([{}], [runs], [{"kind": "prior"}], [{"kind": "centre"}])["passed"]
        is False
    )
    distances |= {"centre": 6.9, "prior": 1.2}
    assert (
        fit.preflight_record([{}], [runs], [{"kind": "prior"}], [{"kind": "centre"}])["passed"]
        is False
    )


def test_parameter_errors_allow_a_state_swap() -> None:
    """An answer whose two states swap roles is read in the swapped order, with no error."""
    from quantumnematode.evaluation.controllers import MECHANISTIC_PARAMETERS

    names = [*MECHANISTIC_PARAMETERS, *fit.EXTRA]
    rng = np.random.default_rng(1)
    truth = dict(zip(names, rng.random(25), strict=True))
    swapped = dict(truth)
    for first, second in fit.SWAP_PAIRS:
        swapped[first], swapped[second] = truth[second], truth[first]
    out = fit.unit_errors([swapped[n] for n in names], [truth[n] for n in names])
    assert out["swapped"] is True
    assert max(out["errors"].values()) == pytest.approx(0.0)
    plain = fit.unit_errors([truth[n] for n in names], [truth[n] for n in names])
    assert plain["swapped"] is False


def test_a_capped_restart_is_not_chosen() -> None:
    """Covers "A capped restart is not chosen": only plateaued restarts enter selection."""
    histories = [
        {"stop": {"generation": 330, "by": "cap"}},
        {"stop": {"generation": 180, "by": "plateau"}},
    ]
    assert c1b._plateaued(histories) == [1]
    passing = {"wild_type_passes": True, "D": 1.0}
    failing = {"wild_type_passes": False, "D": 2.0}
    selection = {1: [passing] * 3 + [failing] * 2, 4: [passing] * 4 + [failing]}
    assert c1b.chosen_restart(selection) == 4


def test_the_cap_comes_from_the_probes() -> None:
    """Covers "The cap comes from the probes": at least 1.5 x the longest 25-parameter plateau."""
    probes = ("probe_full25.json", "basin_warm.json", "basin_boxA.json", "basin_boxB.json")
    plateaus = [
        json.loads((fit.PROBES / name).read_text())["stop"]["generation"] for name in probes
    ]
    assert max(plateaus) == 218
    assert 1.5 * max(plateaus) <= fit.CAP


def test_the_truths_are_the_passing_probes() -> None:
    """The recovery's truths are the two 25-parameter probe answers that reach the wild type."""
    for path in fit.TRUTH_PATHS:
        record = json.loads(path.read_text())
        assert record["reaches"] is True
        assert len(record["answer"]) == fit.DIMENSION


def test_the_objectives_score_their_targets() -> None:
    """The band objective is 0 at the band centres; the truth objective at its observation."""
    band = fit.BandObjective()
    centres = {"statistics": dict(band.truth["statistics"])}
    assert band(centres) == pytest.approx(0.0)
    assert len(centres["statistics"]) == 21
    truth = fit.TruthObjective(centres)
    assert truth(centres) == pytest.approx(0.0)
    key = next(iter(centres["statistics"]))
    assert truth({"statistics": {**centres["statistics"], key: None}}) >= v2.UNDEFINED_RANK


@pytest.mark.parametrize(
    "stage",
    ["recovery-preflight", "recovery", "run", "select", "grade", "read"],
)
def test_every_registered_stage_refuses_before_part_a(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no committed launch record no registered stage runs."""
    monkeypatch.setattr(fit, "LAUNCH_PATH", _REPO / "no-such-dir" / "launch.md")
    assert fit.main([stage]) == 1


def test_a_stored_record_is_reused_only_at_its_point(tmp_path: Path) -> None:
    """A store's record for k is reused at the same z, and refused at another, before any run."""
    store = tmp_path / "sims.jsonl"
    z = [0.5] * fit.DIMENSION
    store.write_text(json.dumps({"k": 3, "z": z, "statistics": {}}) + "\n")
    assert fit.run_jobs([(3, z)], 0, store, workers=1)[3]["z"] == z
    with pytest.raises(ValueError, match="another point"):
        fit.run_jobs([(3, [0.4] * fit.DIMENSION)], 0, store, workers=1)


def test_a_stored_record_is_reused_only_at_its_seed_base(tmp_path: Path) -> None:
    """A record run at one seed base is refused at another; a record carrying none is reused."""
    store = tmp_path / "sims.jsonl"
    z = [0.5] * fit.DIMENSION
    store.write_text(
        json.dumps({"k": 3, "z": z, "statistics": {}, "seed_base": 500_000_000})
        + "\n"
        + json.dumps({"k": 4, "z": z, "statistics": {}})
        + "\n",
    )
    assert fit.run_jobs([(3, z)], 500_000_000, store, workers=1)[3]["seed_base"] == 500_000_000
    with pytest.raises(ValueError, match="another seed base"):
        fit.run_jobs([(3, z)], 530_000_000, store, workers=1)
    assert fit.run_jobs([(4, z)], 530_000_000, store, workers=1)[4]["z"] == z


def test_selection_refuses_an_unfinished_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection, grading and the read need every restart stopped."""
    monkeypatch.setattr(fit, "BUILD", tmp_path)
    (tmp_path / "fits").mkdir()
    for r in range(fit.RESTARTS):
        stop = None if r == 5 else {"generation": 100, "by": "plateau"}
        (tmp_path / "fits" / f"fit{r}.json").write_text(json.dumps({"stop": stop}))
    with pytest.raises(RuntimeError, match=r"\[5\] have not stopped"):
        fit._histories()
