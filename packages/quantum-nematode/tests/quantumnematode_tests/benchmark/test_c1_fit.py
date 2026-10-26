"""C.1's fit: the objective, the chosen restart, the 3-of-5 verdict, the unidentified rule, seeds.

Covers the controller-fit spec's scenarios "The fit is guarded", "No restart plateaus", "C.1 fails
the wild type", "One unlucky population does not fail a faithful controller" and "Two passing
restarts disagree".
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import c1_fit as c1  # noqa: E402  # pyright: ignore[reportMissingImports]
import cma_recovery as cr  # noqa: E402  # pyright: ignore[reportMissingImports]
import npe_recovery as nr  # noqa: E402  # pyright: ignore[reportMissingImports]


def _history(by: str, medians: list[float]) -> dict[str, Any]:
    return {"stop": {"generation": len(medians), "by": by}, "medians": medians}


def test_the_objective_is_the_benchmark_distance() -> None:
    """D is the objective; an incomplete grade counts the cap."""
    assert c1.objective({"D": 1.25}) == pytest.approx(1.25)
    assert c1.objective({"D": None}) == pytest.approx(10.0)


def test_a_capped_restart_is_not_chosen() -> None:
    """The lowest final objective wins only among plateaued restarts."""
    histories = [
        _history("cap", [0.1] * 40),
        _history("plateau", [2.0] * 40),
        _history("plateau", [1.0] * 40),
    ]
    assert c1.chosen_restart(histories) == 2


def test_no_restart_plateaus() -> None:
    """With every restart at the cap, no restart is chosen and no verdict is read."""
    assert c1.chosen_restart([_history("cap", [1.0] * 40)] * 3) is None


def test_a_population_passes_when_every_family_passes() -> None:
    """One failing family fails the population."""
    assert c1.population_passes({"small_lawn": True, "uniform_lawn": True, "local_search": True})
    assert not c1.population_passes(
        {"small_lawn": True, "uniform_lawn": False, "local_search": True},
    )


def test_one_unlucky_population_does_not_fail_a_faithful_controller() -> None:
    """Four of five passes the wild type; two of five does not."""
    assert c1.wild_type_verdict([True, True, False, True, True])
    assert c1.wild_type_verdict([True, False, True, False, True])
    assert not c1.wild_type_verdict([True, False, False, False, True])


def test_two_passing_restarts_that_disagree_are_unidentified() -> None:
    """Disagreement among passing restarts is unidentified; agreement is their verdict."""
    assert c1.mutant_verdict([True, False], chosen=True) == "unidentified"
    assert c1.mutant_verdict([False, False], chosen=True) == "fail"
    assert c1.mutant_verdict([], chosen=True) == "pass"


def test_the_seed_layout_is_disjoint() -> None:
    """Fit animals stay below the grading populations, which stay clear of every other scheme."""
    top_fit = max(
        cr.fit_k(c1.RESTARTS - 1, c1.CAP, cr.POPSIZE - 1),
        cr.answer_k(c1.RESTARTS - 1, c1.RESIMULATIONS - 1),
    )
    assert c1.SEED_BASE + nr.ANIMALS * (top_fit + 1) <= c1.GRADE_BASE
    top_grade = c1.GRADE_BASE + nr.ANIMALS * c1.GRADE_POPULATIONS
    assert top_grade < 200_000_000
    assert c1.SMOKE_SEED_BASE + nr.ANIMALS * 32 < c1.SEED_BASE
    smoke_fit_top = c1.SMOKE_SEED_BASE + nr.ANIMALS * 32
    assert smoke_fit_top <= c1.SMOKE_GRADE_BASE
    assert c1.SMOKE_GRADE_BASE + nr.ANIMALS < c1.SEED_BASE


class _Pool:
    """Stands in for the worker pool: returns given records instead of simulating."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def map(self, _function: Any, ks: list[int], *_rest: Any) -> list[dict[str, Any]]:
        return [self.records[i % len(self.records)] for i in range(len(ks))]


def _records() -> list[dict[str, Any]]:
    names = nr.statistic_names()[:-1]
    return [{"statistics": dict.fromkeys(names, 0.2 + 0.01 * j), "D": 1.0 + j} for j in range(16)]


def test_run_fit_uses_a_given_objective(tmp_path: Path) -> None:
    """With an objective, a generation's median is the median of its scores."""
    history = cr.run_fit(
        {},
        fit=0,
        cma_seed=1,
        seed_base=0,
        cap=1,
        use_rule=True,
        state_dir=tmp_path,
        pool=_Pool(_records()),  # pyright: ignore[reportArgumentType]
        scales={},
        objective=c1.objective,
    )
    assert history["medians"] == [pytest.approx(8.5)]


def test_run_fit_defaults_to_the_distance_to_the_observation(tmp_path: Path) -> None:
    """Without an objective, the score is the distance to the observation, as in recovery."""
    names = nr.statistic_names()[:-1]
    observation = {"statistics": dict.fromkeys(names, 0.2)}
    scales = dict.fromkeys(names, 0.01)
    records = _records()
    history = cr.run_fit(
        observation,
        fit=0,
        cma_seed=1,
        seed_base=0,
        cap=1,
        use_rule=True,
        state_dir=tmp_path,
        pool=_Pool(records),  # pyright: ignore[reportArgumentType]
        scales=scales,
    )
    expected = sorted(nr.predictive_distance(observation, r, scales) for r in records)
    assert history["medians"] == [pytest.approx((expected[7] + expected[8]) / 2)]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 - fixed git arguments in a temporary repository
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],  # noqa: S607 - git on the PATH
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_the_fit_is_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The encoding table is bound to the launch record's commit."""
    repo = tmp_path / "repo"
    support = repo / "support"
    support.mkdir(parents=True)
    _git(repo, "init", "-q")
    launch, encodings = support / "launch.md", support / "encodings.md"
    monkeypatch.setattr(cr, "_REPO", repo)
    monkeypatch.setattr(c1, "LAUNCH_PATH", launch)
    monkeypatch.setattr(c1, "ENCODINGS_PATH", encodings)
    launch.write_text("# Launch\n")
    encodings.write_text("| tph-1 | a_intake = b_intake = 0 |\n")
    assert not c1._guarded()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "register")
    assert c1._guarded()
    encodings.write_text("| tph-1 | a_intake = 0 |\n")
    assert not c1._guarded()
    assert c1.main(["read"]) == 1
