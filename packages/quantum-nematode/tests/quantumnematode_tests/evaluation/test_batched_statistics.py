"""The batched evaluator against the per-step runner on full assays: graded statistics agree.

Covers the batched-evaluation spec's parity requirement at the statistic level. On seeds 5001-5032,
over full Scheer and Flavell assays, each statistic's paired difference has a 95% bootstrap interval
containing zero: no bias between the two runners is detected. Over a full hour the closed sensory
loop decorrelates some pairs, as float32 rounding grows near a steep lawn edge. Their differences
are then those of independent animals, so the check is for bias, not identity.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import pytest
from quantumnematode.validation import roaming_dwelling as rd

from . import _parity as p

sys.path.insert(0, str(p.REPO / "scripts" / "analysis"))
import lawn_states as ls  # pyright: ignore[reportMissingImports]

_SEEDS = list(range(5001, 5033))
_BOUTS = {"reversal", "omega", "delta"}
# Differences this small are float rounding: the statistic is the same.
_IDENTICAL = 1e-9


def _statistics(steps: list, intake_fraction: float, max_satiety: float) -> dict[str, float]:  # type: ignore[type-arg]
    """Return one animal's graded statistics from its captured steps."""
    classifier = rd.GaussianClassifier(hmm=rd.load_calibrated_hmm())
    episode = ls.episode_windows(steps, classifier, intake_fraction, 1.0, step_seconds=1.0)
    on_food = episode.states != rd.OFF_FOOD
    roaming = float(np.mean(episode.states[on_food] == rd.ROAMING)) if on_food.any() else 0.0
    starts = sum(
        1 for a, b in itertools.pairwise(steps) if a.primitive == "run" and b.primitive in _BOUTS
    )
    return {
        "on_lawn": float(np.mean([bool(s.on_lawn) for s in steps])),
        "roaming": roaming,
        "bouts_per_min": starts / (len(steps) / 60.0),
        "satiety": float(steps[-1].satiety or 0.0) / max_satiety,
        "intake": float(sum(s.intake or 0.0 for s in steps)),
    }


def _bootstrap_interval(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(5001)
    means = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@pytest.mark.slow
@pytest.mark.parametrize("protocol", ["scheer_lawn", "uniform_lawn_flavell"])
@pytest.mark.parametrize("controller", ["mlp", "pirouette"])
def test_graded_statistics_agree(controller: str, protocol: str) -> None:
    """Paired per-animal statistics: each difference's 95% interval contains zero."""
    config, single, batched = p.build(controller, protocol)
    lawns = config.environment.foraging.lawns
    max_satiety = config.satiety.initial_satiety
    trajectories = p.BatchedEvaluator(config, batched, _SEEDS).run()
    refs = [p.run_reference(config, single, seed) for seed in _SEEDS]
    reference = [_statistics(r.behaviour, lawns.intake_fraction, max_satiety) for r in refs]
    batched_stats = [
        _statistics(trajectories.to_behaviour_steps(i), lawns.intake_fraction, max_satiety)
        for i in range(len(_SEEDS))
    ]
    scale = max(np.mean([r["intake"] for r in reference]), 1e-12)
    for name in reference[0]:
        a = np.array([r[name] for r in reference])
        b = np.array([s[name] for s in batched_stats])
        diff = (b - a) / scale if name == "intake" else b - a
        if np.all(np.abs(diff) < _IDENTICAL):
            continue
        low, high = _bootstrap_interval(diff)
        assert low <= 0.0 <= high, (name, low, high)
