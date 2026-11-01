"""C.1 refitted on benchmark v2: the objective, the stop rule, selection, the cap and the seeds.

Covers the controller-fit spec's scenarios "The objective is the bands'", "An undefined median
holds a restart", "Selection and verdict share no animals", "The most passing restart is chosen"
and "The pilot cannot reach the wild type".
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import c1b_fit as fit  # noqa: E402  # pyright: ignore[reportMissingImports]
import cma_recovery as cr  # noqa: E402  # pyright: ignore[reportMissingImports]
import cma_recovery_v2 as v2  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_the_objective_is_the_bands() -> None:
    """Covers "The objective is the bands'": zero at the centres, definedness first."""
    objective = fit.BandObjective()
    at_centres = {"statistics": dict(objective.truth["statistics"])}
    assert objective(at_centres) == pytest.approx(0.0)
    key = next(iter(at_centres["statistics"]))
    undefined = {"statistics": {**at_centres["statistics"], key: None}}
    assert objective(undefined) == pytest.approx(v2.UNDEFINED_RANK)
    far = {
        "statistics": {
            k: c + 1_700 * objective.scales[k] for k, c in at_centres["statistics"].items()
        },
    }
    assert objective(far) < objective(undefined)


def test_an_undefined_median_holds_a_restart() -> None:
    """Covers "An undefined median holds a restart": the plateau waits for a defined window."""
    flat = [5.0] * 60
    assert fit.defined_stop(flat, cr.SIGMA0)
    held = [5.0] * 30 + [1005.0] + [5.0] * 29
    assert not fit.defined_stop(held, cr.SIGMA0)
    assert not fit.defined_stop(flat, 0.3)
    sigmas = [0.1] * 100
    late = [1005.0] * 10 + [5.0] * 90
    assert fit.defined_plateau_generation(late, sigmas) == 50


def _pass(d: float | None) -> dict[str, object]:
    return {"wild_type_passes": True, "D": d}


def _fail(d: float | None) -> dict[str, object]:
    return {"wild_type_passes": False, "D": d}


def test_the_most_passing_restart_is_chosen() -> None:
    """Covers "The most passing restart is chosen": passes first, then lower median D, then r."""
    selection = {
        2: [_pass(2.0), _pass(2.0), _pass(2.5)],
        5: [_pass(1.0), _pass(1.0), _fail(9.0)],
    }
    assert fit.chosen_restart(selection) == 2
    tied = {
        4: [_pass(2.0), _fail(2.2), _pass(2.1)],
        1: [_pass(1.5), _pass(1.9), _fail(3.0)],
        6: [_pass(1.5), _pass(1.9), _fail(3.0)],
    }
    assert fit.chosen_restart(tied) == 1
    assert fit.chosen_restart({}) is None
    incomplete = {0: [_fail(None)] * 3, 3: [_fail(9.0)] * 3}
    assert fit.chosen_restart(incomplete) == 3


def test_the_verdict_needs_three_of_five() -> None:
    """One unlucky population does not fail a faithful controller; three failing do."""
    assert fit.wild_type_verdict([_pass(1.0)] * 4 + [_fail(4.0)])
    assert not fit.wild_type_verdict([_pass(1.0)] * 2 + [_fail(4.0)] * 3)
    assert fit.wild_type_verdict([_pass(1.0)] * 3 + [_fail(4.0)] * 2)


def test_the_cap_follows_the_pilot() -> None:
    """F.0b-i's cap rule, its fallback on the registered D."""
    assert fit.pilot_cap(None, None, 3.0) == cr.MAX_GENERATIONS
    assert fit.pilot_cap(120, 3.1, 3.0) == 180
    assert fit.pilot_cap(120, 3.0 + fit.CAP_TOLERANCE_D + 0.01, 3.0) == cr.MAX_GENERATIONS


def test_selection_and_verdict_share_no_animals() -> None:
    """Covers "Selection and verdict share no animals", and every registered range is disjoint."""
    animals = 200
    fit_max = fit.FIT_BASE + animals * (cr.FIT_BLOCK * (fit.RESTARTS - 1) + cr.ANSWER_OFFSET + 9)
    ranges = [
        (fit.FIT_BASE, fit_max + animals),
        (fit.GRADE_BASE, fit.GRADE_BASE + animals * fit.GRADE_POPULATIONS),
        (
            fit.SELECT_BASE,
            fit.SELECT_BASE
            + animals * (fit.select_k(fit.RESTARTS - 1, fit.SELECT_POPULATIONS - 1) + 1),
        ),
        (fit.PILOT_BASE, fit.PILOT_BASE + animals * (max(cr.PILOT_ANSWER_OFFSETS) + 10)),
        (fit.PILOT_GRADE_BASE, fit.PILOT_GRADE_BASE + animals * fit.GRADE_POPULATIONS),
        (fit.SMOKE_BASE, fit.SMOKE_BASE + animals * 32),
        (fit.SMOKE_GRADE_BASE, fit.SMOKE_GRADE_BASE + animals),
    ]
    ranges.sort()
    for (_, end), (start, _) in itertools.pairwise(ranges):
        assert end <= start
    assert v2.PROBE_B_BASE + 8 * animals * cr.FIT_BLOCK <= fit.FIT_BASE


def test_the_pilot_cannot_reach_the_wild_type_stops_the_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers "The pilot cannot reach the wild type": no registered fit runs."""
    import json

    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"reachability": {"reaches": False}, "registered_cap": 200}))
    monkeypatch.setattr(fit, "PILOT_PATH", pilot)

    class RanError(Exception):
        pass

    def ran(*_: object, **__: object) -> dict[str, object]:
        raise RanError

    monkeypatch.setattr(fit, "_fit", ran)
    monkeypatch.setattr(fit, "REACHABILITY_ACCEPTED", False)
    assert fit._run(1) == 1
    # Only the maintainer's registered acceptance lets the fits run after a failed pilot.
    monkeypatch.setattr(fit, "REACHABILITY_ACCEPTED", True)
    with pytest.raises(RanError):
        fit._run(1)


def test_failing_names_statistics_and_presence() -> None:
    """The diagnosis names each failing statistic and each failed presence test."""
    grade = {
        "families": {
            "kinematics": {"roaming_speed_median": {"passes": False}, "x": {"passes": True}},
        },
        "local_search_presence": {"present": False},
        "chemotaxis_presence": {"present": True},
    }
    assert fit.failing([grade]) == ["kinematics/roaming_speed_median", "local_search/presence"]


def test_the_fits_need_part_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every stage after the pilot needs the launch record's Part B; the pilot needs Part A only."""
    launch = tmp_path / "launch.md"
    launch.write_text("# Launch record\n\n## Part A\n\nrules\n")
    monkeypatch.setattr(fit, "LAUNCH_PATH", launch)
    monkeypatch.setattr(fit.cr, "launch_committed", lambda *_, **__: True)
    assert fit._guarded("pilot")
    assert not fit._guarded("run")
    launch.write_text("# Launch record\n\n## Part A\n\nrules\n\n## Part B\n\nthe cap\n")
    assert fit._guarded("run")


def test_the_kappa_probe_maps_one_more_dimension() -> None:
    """The 21st dimension is the off-food gain on the on-food range; the rest is F.0a's box."""
    import npe_recovery as nr  # pyright: ignore[reportMissingImports]
    import numpy as np

    z = np.random.default_rng(1).random(21)
    params = fit.parameters_kappa(z)
    assert {k: v for k, v in params.items() if k != "kappa_off_food"} == nr.parameters_from_unit(
        z[:20],
    )
    low, high = nr.PRIOR["kappa_weathervane"][1:]
    assert params["kappa_off_food"] == pytest.approx(low + z[20] * (high - low))
    assert fit.parameters_kappa(np.r_[z[:20], 0.5])["kappa_off_food"] == pytest.approx(0.0)
