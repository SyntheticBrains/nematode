"""The measurement model: noise, lost seconds, fragmentation and per-animal streams.

Covers the benchmark-grading spec's scenario "An omega second can go missing".
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from quantumnematode.benchmark import measurement as mm
from quantumnematode.report.behaviour import BehaviourStep


def _steps(n: int, kinds: list[str] | None = None) -> list[BehaviourStep]:
    kinds = kinds or ["run"] * n
    return [
        BehaviourStep(
            step=i,
            x=float(i),
            y=0.0,
            heading_rad=0.0,
            concentration=0.0,
            dc_dt=0.0,
            grad_dir=0.0,
            grad_strength=0.0,
            on_lawn=True,
            primitive=kinds[i],
        )
        for i in range(n)
    ]


def test_noise_has_the_stated_sd() -> None:
    """Positions gain normal noise of jitter_mm per axis; nothing else changes."""
    model = mm.TrackerModel(jitter_mm=0.01)
    out = mm.measure(_steps(20_000), seed=3, model=model)
    dx = np.array([o.x - i for i, o in enumerate(out)])
    assert np.std(dx) == pytest.approx(0.01, rel=0.03)
    assert all(o.on_lawn for o in out)


def test_an_omega_second_can_go_missing() -> None:
    """A loss starts in omega seconds at p_omega and lasts a run drawn from the omega runs."""
    model = mm.TrackerModel(jitter_mm=0.0, p_omega=1.0, p_other=0.0, runs_omega_s=(3,))
    kinds = ["run"] * 10 + ["omega"] + ["run"] * 10
    out = mm.measure(_steps(21, kinds), seed=1, model=model)
    lost = [math.isnan(o.x) for o in out]
    assert lost == [False] * 10 + [True] * 3 + [False] * 8


def test_without_runs_a_loss_is_one_second() -> None:
    """Without measured runs, a start loses that second alone."""
    rng = np.random.default_rng(0)
    model = mm.TrackerModel(jitter_mm=0.0, p_omega=0.0, p_other=1.0)
    lost = mm.lost_seconds(["run"] * 5, model, rng)
    assert lost.tolist() == [True] * 5


def test_an_animal_is_measured_the_same_alone_or_again() -> None:
    """The animal's own stream makes the measurement reproducible from its seed."""
    model = mm.TrackerModel(jitter_mm=0.01, p_other=0.01, runs_other_s=(5, 10))
    a = mm.measure(_steps(500), seed=42, model=model)
    b = mm.measure(_steps(500), seed=42, model=model)
    c = mm.measure(_steps(500), seed=43, model=model)
    assert [(s.x, s.y) for s in a] == [(s.x, s.y) for s in b]
    assert [(s.x, s.y) for s in a] != [(s.x, s.y) for s in c]


def test_fragments_are_cut_with_one_lost_second() -> None:
    """A fragmenting tracker cuts the track at drawn lengths, one lost second at each cut."""
    model = mm.TrackerModel(jitter_mm=0.0, fragment_lengths_s=(10,))
    out = mm.measure(_steps(35), seed=0, model=model)
    lost = [i for i, o in enumerate(out) if math.isnan(o.x)]
    assert lost == [10, 21, 32]


def test_jitter_from_steps_inverts_the_still_worm() -> None:
    """Steps between noisy readings of one point give back the per-axis SD."""
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 0.005, size=(2, 100_000, 2))
    steps = np.hypot(*(a - b).T)
    assert mm.jitter_from_steps(steps) == pytest.approx(0.005, rel=0.02)


def test_the_model_round_trips_through_its_record() -> None:
    """A recorded model reads back as the same model."""
    model = mm.TrackerModel(
        jitter_mm=0.004,
        p_omega=0.05,
        runs_omega_s=(16, 20),
        fragment_lengths_s=(39,),
    )
    assert mm.TrackerModel.from_dict(model.to_dict()) == model
