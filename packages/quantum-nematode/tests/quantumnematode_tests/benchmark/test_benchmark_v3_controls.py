"""B v3's controls: the oracle's gate, the adversaries' and negatives' 3-of-5 rules, the verdict.

Covers the benchmark-controls spec's scenarios "A reversal-dominated answer is caught", "The new
statistics are unreachable" and "v3 becomes the default" (the verdict and its guards).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import benchmark_v3_controls as v3c  # noqa: E402  # pyright: ignore[reportMissingImports]

_KEYS = [f"{f}/{s}" for f, s in v3c.NEW_STATISTICS]


def _population(z: float, *, passes: bool = True) -> dict[str, Any]:
    return {"new": {key: {"z": z, "passes": passes} for key in _KEYS}, "reversal_share": None}


def test_the_oracle_passes_inside_its_mean_and_share() -> None:
    """Mean z within [-1, 1] and at least 90% passing, on every new statistic, passes."""
    populations = [_population(0.5) for _ in range(36)] + [
        _population(3.5, passes=False) for _ in range(4)
    ]
    gate = v3c.oracle_gate(populations)
    assert gate["passes"] is True
    assert gate["statistics"][_KEYS[0]]["share_within_bar"] == pytest.approx(0.9)


def test_the_new_statistics_are_unreachable() -> None:
    """Covers "The new statistics are unreachable": a mean beyond 1, or a share under 90%, fails."""
    shifted = [_population(1.2) for _ in range(40)]
    assert v3c.oracle_gate(shifted)["passes"] is False
    spread = [_population(0.0) for _ in range(35)] + [
        _population(0.0, passes=False) for _ in range(5)
    ]
    assert v3c.oracle_gate(spread)["passes"] is False


def _adversary(name: str, caught: list[bool]) -> list[dict[str, Any]]:
    return [{"name": name, "k": k, "caught": c} for k, c in enumerate(caught)]


def test_a_reversal_dominated_answer_is_caught() -> None:
    """Covers "A reversal-dominated answer is caught": 3 of 5 populations catch it; 2 do not."""
    rows = _adversary("restart 2", [True, True, True, False, False]) + _adversary(
        "restart 4",
        [True, True, False, False, False],
    )
    verdicts = v3c.adversary_verdicts(rows)
    assert verdicts["restart 2"]["caught_3_of_5"] is True
    assert verdicts["restart 4"]["caught_3_of_5"] is False
    assert verdicts["restart 4"]["caught_in"] == 2


def test_a_negative_must_fail_in_three_of_five() -> None:
    """A negative fails the wild type, and chemotaxis, by counting its failing populations."""
    rows = [
        {
            "name": "hmm",
            "wild_type_passes": k >= 3,
            "family_passes": {"chemotaxis": k >= 2},
        }
        for k in range(5)
    ]
    out = v3c.negative_verdicts(rows)["hmm"]
    assert out["fails_wild_type_in"] == 3
    assert out["fails_wild_type_3_of_5"] is True
    assert out["fails_chemotaxis_in"] == 2
    assert out["fails_chemotaxis_3_of_5"] is False


def _valid_parts() -> tuple[dict[str, Any], ...]:
    checks = {c: {"passes": True} for c in ("small_lawn", "kinematics", "local_search")}
    record = {"checks": checks, "chemotaxis_leave_one_out": {"passes": True}}
    negatives = {
        n: {"fails_wild_type_3_of_5": True, "fails_chemotaxis_3_of_5": True}
        for n in v3c.NEGATIVE_ORDER
    }
    read = {
        "oracle": {"passes": True},
        "lopezcruz_held_out": {"passes": True},
        "adversaries": {"restart 2": {"caught_3_of_5": True}},
        "negatives": negatives,
    }
    return {"passes": True}, record, read, {"n2_control": {"passes": True}}


def test_the_verdict_is_valid_only_if_every_check_holds() -> None:
    """Every check passing gives VALID; any one failing, or a stopped read, gives VOID."""
    round_trip, record, read, v2_read = _valid_parts()
    assert v3c.verdict(round_trip, record, read, v2_read)["verdict"] == "VALID"
    read["adversaries"]["restart 4"] = {"caught_3_of_5": False}
    assert v3c.verdict(round_trip, record, read, v2_read)["verdict"] == "VOID"
    stopped = {"oracle": {"passes": False}}
    out = v3c.verdict(round_trip, record, stopped, v2_read)
    assert out["verdict"] == "VOID"
    assert out["checks"]["negatives_fail_wild_type"] is False


def test_three_of_five_follows_the_family_rates() -> None:
    """The faithful 3-of-5 chance is the binomial tail of the five families' joint pass rate."""
    record = {
        "checks": {
            "small_lawn": {"joint_pass_rate": 1.0},
            "kinematics": {"joint_pass_rate": 1.0},
            "local_search": {"joint_pass_rate": 1.0},
        },
        "chemotaxis_leave_one_out": {"unit_rate": 0.0},
    }
    out = v3c.three_of_five(record)
    assert out["population_passes"] == pytest.approx(0.95)
    p = 0.95
    expected = sum(
        (1 if k in (0, 5) else 5 if k in (1, 4) else 10) * p**k * (1 - p) ** (5 - k)
        for k in (3, 4, 5)
    )
    assert out["three_of_five"] == pytest.approx(expected)


@pytest.mark.parametrize("stage", ["round-trip", "oracle-fit", "read", "verdict"])
def test_every_registered_stage_refuses_before_part_a(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no committed launch record no registered stage reads anything."""
    monkeypatch.setattr(v3c, "LAUNCH_PATH", _REPO / "no-such-dir" / "launch.md")
    argv = [stage, "--chen-archive", "none.zip"] if stage == "round-trip" else [stage]
    assert v3c.main(argv) == 1
