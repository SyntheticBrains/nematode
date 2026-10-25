"""sep-CMA recovery's rules: the plateau, the cap, the truths, the best restart and the seeds.

Covers the fit-recovery spec's scenarios "A truth whose statistics flip", "A fit that never moves",
"The best restart is chosen by the fit's own objective", "The answer stays in the box", "An early
plateau does not shorten the cap" and "The read is guarded", and a refused resume from mismatched
checkpoints.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import cma_recovery as cr  # noqa: E402  # pyright: ignore[reportMissingImports]
import npe_recovery as nr  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_the_plateau_rule_fires_on_a_flat_series_from_generation_40() -> None:
    """A flat objective plateaus at the first generation the rule is read, and not before."""
    flat = [3.0] * 60
    assert not cr.plateau_reached(flat[:39])
    assert cr.plateau_generation(flat) == 40


def test_the_plateau_rule_does_not_fire_during_a_descent() -> None:
    """Falling by 0.01 a generation is 0.2 over 20 generations, well past the tolerance."""
    descent = [6.0 - 0.01 * n for n in range(300)]
    assert cr.plateau_generation(descent) is None


def test_the_cap_follows_the_pilot_plateau() -> None:
    """The cap is 1.5 times the plateau generation, at most 300, and 300 without a plateau."""
    assert cr.candidate_cap(50) == 75
    assert cr.candidate_cap(250) == 300
    assert cr.registered_cap(None, None, 0.9) == 300
    assert cr.registered_cap(50, 0.88, 0.90) == 75


def test_an_early_plateau_does_not_shorten_the_cap() -> None:
    """A pilot still gaining more than 0.05 after 1.5 times its plateau keeps the cap at 300."""
    assert cr.registered_cap(50, 0.70, 0.80) == 300


def _record(*, defined: bool = True) -> dict:
    names = nr.statistic_names()[:-1]
    stats: dict[str, float | None] = dict.fromkeys(names, 0.5)
    if not defined:
        stats["local_search/gamma_per_min"] = None
    return {"statistics": stats}


def test_a_truth_whose_statistics_flip_is_skipped() -> None:
    """One stability repeat leaving gamma undefined skips the candidate, naming the statistic."""
    keep, reason = cr.keep_candidate(_record(), [_record(), _record(defined=False)], 0.5, 6.0)
    assert not keep
    assert "gamma_per_min" in reason


def test_a_truth_with_no_reachable_gain_is_skipped() -> None:
    """A prior D within 1.0 of the floor leaves the gain fraction unstable."""
    keep, reason = cr.keep_candidate(_record(), [_record()], 3.5, 4.0)
    assert not keep
    assert "reachable" in reason
    assert cr.keep_candidate(_record(), [_record()], 0.5, 6.0) == (True, "kept")


def test_a_fit_that_never_moves_gains_nothing() -> None:
    """Scored at the prior's typical distance, the gain is 0; at the floor, 1."""
    assert cr._gain(6.9, 6.9, 0.6) == pytest.approx(0.0)
    assert cr._gain(6.9, 0.6, 0.6) == pytest.approx(1.0)


def test_the_best_restart_is_chosen_by_its_own_objective() -> None:
    """The restart with the lower mean of its last 20 medians wins, whatever else it did."""
    a = [5.0] * 30 + [2.0] * 20
    b = [1.0] * 30 + [2.5] * 20
    assert cr.best_restart([a, b]) == 0
    assert cr.final_objective(a) == pytest.approx(2.0)


def test_the_answer_stays_in_the_box() -> None:
    """With the optimum outside the box, cma's mean leaves it; the answer does not."""
    strategy = cr.new_strategy(1)
    for _ in range(25):
        population = strategy.ask()
        strategy.tell(population, [float(np.sum((np.asarray(x) + 0.5) ** 2)) for x in population])
    assert np.any(np.asarray(strategy.mean) < 0.0)
    a = cr.answer(strategy)
    assert np.all((a >= 0.0) & (a <= 1.0))


def test_the_seed_layout_is_disjoint() -> None:
    """Every registered index is unique, and animals stay below the pilot and smoke bases."""
    fits = {
        cr.fit_k(f, g, m)
        for f in range(cr.TRUTHS * cr.RESTARTS)
        for g in range(1, cr.MAX_GENERATIONS + 1)
        for m in range(cr.POPSIZE)
    }
    answers = {cr.answer_k(f, q) for f in range(cr.TRUTHS * cr.RESTARTS) for q in range(10)}
    candidates = {cr.candidate_k(c, q) for c in range(43) for q in range(cr.REPEATS)}
    centre = {cr.centre_k(q) for q in range(10)}
    groups = [fits, answers, candidates, centre]
    assert sum(map(len, groups)) == len(set().union(*groups))
    top = max(set().union(*groups))
    assert cr.SEED_BASE + nr.ANIMALS * top + nr.ANIMALS < cr.PILOT_SEED_BASE
    pilot_top = max(cr.PILOT_ANSWER_OFFSETS) + 10
    assert cr.PILOT_SEED_BASE + nr.ANIMALS * pilot_top < cr.SMOKE_SEED_BASE


def _git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 - fixed git arguments in a temporary repository
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],  # noqa: S607 - git on the PATH
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_the_read_is_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard passes only when the launch record and the records it binds match its commit."""
    repo = tmp_path / "repo"
    support = repo / "support"
    support.mkdir(parents=True)
    _git(repo, "init", "-q")
    launch, preflight, pilot = (support / n for n in ("launch.md", "preflight.json", "pilot.json"))
    monkeypatch.setattr(cr, "_REPO", repo)
    monkeypatch.setattr(cr, "LAUNCH_PATH", launch)
    monkeypatch.setattr(cr, "PREFLIGHT_PATH", preflight)
    monkeypatch.setattr(cr, "PILOT_PATH", pilot)
    launch.write_text("# Part A\n\n## Part B\n")
    preflight.write_text("{}\n")
    pilot.write_text('{"registered_cap": 210}\n')

    # An untracked launch record is not committed.
    assert not cr.launch_committed()

    # Matching committed records pass.
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "register")
    assert cr.launch_committed()
    assert cr.part_b_committed()

    # A preflight record changed in the working tree fails.
    preflight.write_text('{"passed": true}\n')
    assert not cr.part_b_committed()
    preflight.write_text("{}\n")
    assert cr.part_b_committed()

    # A pilot record committed later, with the launch record unchanged, fails the byte comparison.
    pilot.write_text('{"registered_cap": 300}\n')
    _git(repo, "commit", "-q", "-am", "later pilot")
    assert cr.launch_committed()
    assert not cr.part_b_committed()


def test_a_resume_with_mismatched_markers_is_refused(tmp_path: Path) -> None:
    """A checkpoint a generation ahead of its history is refused, not continued."""
    (tmp_path / "fit0.json").write_text(
        json.dumps({"generation": 1, "stop": None, "medians": [1.0]}),
    )
    checkpoint = {"generation": 2, "strategy": None}
    (tmp_path / "fit0.pkl").write_bytes(pickle.dumps(checkpoint))
    with pytest.raises(RuntimeError, match="generation 2"):
        cr.run_fit(
            {},
            fit=0,
            cma_seed=1,
            seed_base=0,
            cap=10,
            use_rule=True,
            state_dir=tmp_path,
            pool=None,  # pyright: ignore[reportArgumentType]
            scales={},
        )
