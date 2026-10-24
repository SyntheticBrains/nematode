"""The benchmark's negative controllers: an open-loop two-state worm and curvature walks.

Covers the benchmark-controls spec's scenarios "The HMM worm ignores food" and "Every negative is
graded at n = 200" (each animal's randomness from its own generator).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from quantumnematode.evaluation.batched import BatchedSenses
from quantumnematode.evaluation.controllers import BatchedCurvatureWalk, BatchedStateHazards

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))

_TRANSITIONS = np.array([[0.979, 0.021], [0.102, 0.898]])


def _senses(n: int, concentration: float = 0.0, contact: float = 0.0) -> BatchedSenses:
    zeros = np.zeros(n)
    return BatchedSenses(
        concentration=np.full(n, concentration),
        lateral=zeros,
        derivative=zeros,
        food_contact=np.full(n, contact),
        satiety_fraction=zeros,
        locomotor=np.zeros(n, dtype=np.int64),
        stam=np.zeros((n, 1)),
        moved=1.0,
        intake=zeros,
        max_intake=None,
        lateral_scale=1.0,
        derivative_scale=1.0,
    )


def _generators(seeds: list[int]) -> list[torch.Generator]:
    return [torch.Generator().manual_seed(s) for s in seeds]


def _hmm() -> BatchedStateHazards:
    return BatchedStateHazards(
        _TRANSITIONS,
        (0.022, 0.067),
        (61.0, 35.0),
        np.array([[0.088, 1e-4, 1e-4], [0.080, 1e-3, 1e-4]]),
        window_steps=10,
    )


def _run(controller: object, seeds: list[int], steps: int, **senses: float) -> np.ndarray:
    controller.reset(len(seeds))  # type: ignore[attr-defined]
    gens = _generators(seeds)
    return np.stack(
        [controller.act(_senses(len(seeds), **senses), gens) for _ in range(steps)],  # type: ignore[attr-defined]
    )


def test_the_chain_roams_at_its_stationary_share() -> None:
    """Over many animals and windows, the roaming share matches the transitions' stationary one."""
    controller = _hmm()
    raws = _run(controller, list(range(400)), 600)
    roaming = raws[:, :, 0] > np.median(raws[:, :, 0])
    expected = 0.021 / (0.021 + 0.102)
    assert roaming.mean() == pytest.approx(expected, abs=0.03)


def test_the_hmm_worm_ignores_food() -> None:
    """The same animals act identically on and off food: nothing sensed enters the action."""
    on = _run(_hmm(), [1, 2, 3], 120, concentration=1.0, contact=1.0)
    off = _run(_hmm(), [1, 2, 3], 120)
    assert np.array_equal(on, off)


def test_an_animal_is_the_same_alone_and_in_a_batch() -> None:
    """Each animal draws only from its own generator."""
    for make in (_hmm, lambda: BatchedCurvatureWalk(0.03, (1e-4,) * 3, "ar1", 50.0, 0.5)):
        alone = _run(make(), [7], 100)
        crowd = _run(make(), [3, 7, 11], 100)
        assert np.array_equal(alone[:, 0], crowd[:, 1])


def test_the_random_walk_turns_uniformly() -> None:
    """Uniform mode draws each second's curvature uniformly over the whole raw range."""
    raws = _run(BatchedCurvatureWalk(0.03, (1e-4,) * 3, "uniform"), list(range(200)), 100)
    curvature = raws[:, :, 1].ravel()
    assert curvature.min() >= -1.0
    assert curvature.max() <= 1.0
    assert np.histogram(curvature, bins=10, range=(-1, 1))[0].min() > 0.08 * len(curvature)


def test_the_correlated_walk_keeps_its_persistence() -> None:
    """AR(1) curvature has the lag-1 autocorrelation it was given (well inside the clip)."""
    raws = _run(BatchedCurvatureWalk(0.03, (1e-4,) * 3, "ar1", 10.0, 0.6), list(range(100)), 400)
    c = raws[:, :, 1]
    lag1 = np.corrcoef(c[1:].ravel(), c[:-1].ravel())[0, 1]
    assert lag1 == pytest.approx(0.6, abs=0.03)


def test_mean_action_is_refused() -> None:
    """These controllers need each animal's generator."""
    with pytest.raises(ValueError, match="noise is None"):
        _hmm().act(_senses(2), None)


def test_the_negatives_parameters_read_band_dates_only() -> None:
    """The matched parameters' only row selector returns band-side animals."""
    import benchmark_negatives_params as bnp  # pyright: ignore[reportMissingImports]

    tracks = bnp.refs.load_scheer_tracks()
    rows = bnp.band_rows(tracks)
    assert len(rows)
    assert (tracks.side[rows] == bnp.refs.BANDS).all()


def test_the_reading_refuses_without_a_committed_launch_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No committed launch record, no read: the script exits before loading held-out rows."""
    import benchmark_controls as bc  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(bc, "launch_committed", lambda: False)
    monkeypatch.setattr(bc.bb, "BandBuilder", lambda *a, **k: pytest.fail("held-out read"))
    assert bc.main(["read", "--out", str(tmp_path / "out.json")]) == 1
    assert not (tmp_path / "out.json").exists()


def test_one_untrained_seed_is_one_network_across_assays() -> None:
    """The arm's brain config is the same on every assay, so one seed gives the same weights."""
    import benchmark_controls as bc  # pyright: ignore[reportMissingImports]

    for arm in ("mlp", "connectome"):
        weights = []
        for family in bc.bg.PROTOCOLS:
            brain, _, _ = bc.harness.build_brain(bc._assay_path(arm, family), 5301)
            module = brain.actor if arm == "mlp" else brain.topology  # type: ignore[attr-defined]
            weights.append(torch.cat([p.detach().flatten() for p in module.parameters()]))
        assert all(torch.equal(weights[0], w) for w in weights[1:])
