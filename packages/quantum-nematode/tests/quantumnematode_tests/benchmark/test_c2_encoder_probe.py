"""C.2's encoder probes: the readiness stop, the seed blocks, the hand-set vector and one grading.

Covers the connectome-encoder spec's scenario "A restart with half its candidates undefined does not
stop", and grades one encoded population on every family and mutant.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import c2_encoder_probe as probe  # noqa: E402  # pyright: ignore[reportMissingImports]
import cma_recovery as cr  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_a_restart_with_half_its_candidates_undefined_does_not_stop() -> None:
    """Medians near 500 for 40 generations at sigma <= sigma0 do not stop a restart."""
    flat = [500.0 + 0.01 * (i % 2) for i in range(60)]
    assert cr.plateau_reached(flat)
    assert not probe.readiness_stop(flat, sigma=0.1)
    defined = [5.0 + 0.01 * (i % 2) for i in range(60)]
    assert probe.readiness_stop(defined, sigma=0.1)
    assert not probe.readiness_stop(defined, sigma=0.3)
    one_high = [*defined[:-30], 150.0, *defined[-29:]]
    assert not probe.readiness_stop(one_high, sigma=0.1)


def test_the_seed_blocks_do_not_overlap() -> None:
    """The restarts, warm restart, grading, cost pilot, feasibility and smoke are disjoint."""
    last_k = cr.POPSIZE * probe.CAP - 1
    blocks = [
        (
            probe.READINESS_BASE + probe.ANIMALS * probe.RESTART_BLOCK * r,
            probe.READINESS_BASE + probe.ANIMALS * (probe.RESTART_BLOCK * r + last_k + 1),
        )
        for r in range(probe.RESTARTS)
    ]
    blocks.append((probe.WARM_BASE, probe.WARM_BASE + probe.ANIMALS * (last_k + 1)))
    grades = probe.GRADE_POPULATIONS * (probe.WARM_RESTART + 1)
    blocks.append((probe.GRADE_BASE, probe.GRADE_BASE + probe.ANIMALS * grades))
    blocks.append((probe.COST_BASE, probe.COST_BASE + probe.ANIMALS * 2_002))
    blocks.append((probe.FEASIBILITY_BASE, probe.FEASIBILITY_BASE + probe.ANIMALS))
    trials = probe.ANIMALS * probe.FEASIBILITY_TRIALS
    blocks.append((probe.FEASIBILITY_TRIAL_BASE, probe.FEASIBILITY_TRIAL_BASE + trials))
    blocks.append((probe.SMOKE_BASE, probe.SMOKE_BASE + probe.ANIMALS * 101))
    assert last_k < probe.RESTART_BLOCK
    ordered = sorted(blocks)
    for (_, end), (start, _) in itertools.pairwise(ordered):
        assert end <= start


def test_the_hand_set_vector_names_its_mechanism() -> None:
    """Every named value lands on its parameter, and the rest sit at zero or their floor."""
    z = probe.feasibility_vector()
    table = probe.entries()
    values = {e.name: e.value(u) for e, u in zip(table, z, strict=True)}
    for name, value in probe.FEASIBILITY_VALUES.items():
        assert values[name] == pytest.approx(value, rel=1e-9)
    assert values["wiring.excitatory"] == 0.0
    assert values["wiring.gap"] == 0.0
    assert values["sensory.AWC.concentration"] == pytest.approx(0.0, abs=1e-12)
    assert values["modulator.serotonin:mod-1"] == pytest.approx(0.001)
    with pytest.raises(KeyError, match="no encoder parameters"):
        probe.feasibility_vector({"modulator.nothing": 1.0})


@pytest.mark.smoke
def test_an_encoded_population_is_graded() -> None:
    """One encoded population of 64 is graded with presence on all five families and the mutants."""
    z = np.clip(0.5 + 0.1 * np.random.default_rng(1).standard_normal(len(probe.entries())), 0, 1)
    graded = probe.grade_seeds(z, [probe.SMOKE_BASE + 100_000 + i for i in range(64)])
    assert set(graded["families"]) == {
        "small_lawn",
        "uniform_lawn",
        "local_search",
        "kinematics",
        "chemotaxis",
    }
    assert sum(len(r) for r in graded["families"].values()) == 21
    assert graded["D_fit"] is not None
    assert set(graded["mutants"]) == set(probe.MUTANTS)
    for reading in graded["mutants"].values():
        assert set(reading["statistics"]) == {"roaming_fraction", "exits_per_on_lawn_hour"}
