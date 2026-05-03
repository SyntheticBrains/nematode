"""Unit tests for the ``early_stop_on_saturation`` loop flag.

The loop tracks ``best_fitness`` per generation and exits early if the
strict-greater comparison fails for ``early_stop_on_saturation``
consecutive generations.  Counter persists across resume.
"""

from __future__ import annotations

import csv
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from quantumnematode.evolution.encoders import MLPPPOEncoder
from quantumnematode.evolution.loop import EvolutionLoop
from quantumnematode.optimizers.evolutionary import CMAESOptimizer
from quantumnematode.utils.config_loader import (
    EvolutionConfig,
    load_simulation_config,
)

if TYPE_CHECKING:
    from quantumnematode.evolution.encoders import GenomeEncoder
    from quantumnematode.evolution.fitness import FitnessFunction
    from quantumnematode.evolution.genome import Genome
    from quantumnematode.utils.config_loader import SimulationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MLPPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/mlpppo_small_oracle.yml"


class ScriptedFitness:
    """Fitness function that returns a scripted per-generation fitness.

    Returns ``trajectory[generation]`` for every child within a
    generation, so ``max(fitnesses) == trajectory[generation]`` (the
    early-stop counter compares ``current_best`` to the previous
    generation's best).  Cycles to the last value if the loop runs
    longer than the trajectory.
    """

    def __init__(self, trajectory: list[float], population_size: int) -> None:
        self.trajectory = trajectory
        self.population_size = population_size
        self._call_count = 0

    def evaluate(
        self,
        genome: Genome,
        sim_config: SimulationConfig,
        encoder: GenomeEncoder,
        *,
        episodes: int,
        seed: int,
    ) -> float:
        """Return the scripted fitness for the current call's generation index."""
        # Map the call count to a generation index (population_size calls per gen).
        gen_idx = self._call_count // self.population_size
        self._call_count += 1
        if gen_idx >= len(self.trajectory):
            gen_idx = len(self.trajectory) - 1
        return self.trajectory[gen_idx]


def _make_loop(  # noqa: PLR0913
    output_dir: Path,
    *,
    fitness: FitnessFunction,
    generations: int,
    population_size: int = 4,
    early_stop_on_saturation: int | None = None,
    checkpoint_every: int = 100,
) -> EvolutionLoop:
    """Build a loop with a scripted fitness for early-stop testing."""
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    encoder = MLPPPOEncoder()
    ecfg = EvolutionConfig(
        algorithm="cmaes",
        population_size=population_size,
        generations=generations,
        episodes_per_eval=1,
        parallel_workers=1,
        checkpoint_every=checkpoint_every,
        early_stop_on_saturation=early_stop_on_saturation,
    )
    dim = encoder.genome_dim(sim_config)
    optimizer = CMAESOptimizer(
        num_params=dim,
        population_size=population_size,
        sigma0=ecfg.sigma0,
        seed=42,
    )
    rng = np.random.default_rng(42)
    return EvolutionLoop(
        optimizer=optimizer,
        encoder=encoder,
        fitness=fitness,
        sim_config=sim_config,
        evolution_config=ecfg,
        output_dir=output_dir,
        rng=rng,
        log_level=logging.WARNING,
    )


# ---------------------------------------------------------------------------
# Counter trajectory tests
# ---------------------------------------------------------------------------


def test_early_stop_flat_trajectory_fires(tmp_path: Path) -> None:
    """Flat trajectory ``[0.5, 0.5, 0.5, 0.5]`` with N=3 → fires after gen 4.

    Walk:
    - gen 1: counter=0 (no previous; bootstrap)
    - gen 2 (0.5 > 0.5 false): counter=1
    - gen 3: counter=2
    - gen 4: counter=3 → fire after gen 4

    Lineage SHALL contain exactly 4 generations x population_size rows.
    """
    pop = 4
    fitness = ScriptedFitness([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], pop)
    loop = _make_loop(
        tmp_path,
        fitness=fitness,
        generations=10,
        population_size=pop,
        early_stop_on_saturation=3,
    )
    loop.run()

    # Lineage should have 4 generations x pop rows + 1 header row.
    with (tmp_path / "lineage.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1 + 4 * pop, f"Expected {1 + 4 * pop} rows, got {len(rows)}"


def test_early_stop_monotonic_improvement_never_fires(tmp_path: Path) -> None:
    """Monotonic improvement ``[0.1, 0.2, 0.3, 0.4, 0.5]`` → loop runs full budget.

    Counter stays 0 throughout because every generation strictly
    improves over the previous.
    """
    pop = 4
    fitness = ScriptedFitness([0.1, 0.2, 0.3, 0.4, 0.5], pop)
    loop = _make_loop(
        tmp_path,
        fitness=fitness,
        generations=5,
        population_size=pop,
        early_stop_on_saturation=3,
    )
    loop.run()

    # Lineage should have 5 generations x pop rows + header (no early-stop).
    with (tmp_path / "lineage.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1 + 5 * pop


def test_early_stop_counter_resets_on_strict_improvement(tmp_path: Path) -> None:
    """Trajectory ``[0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]`` with N=5 → never fires.

    Walk per the spec scenario "Counter resets on any strict improvement":
    - counter=0 at gen 1 (no previous)
    - counter=0 at gen 2 (strict improvement)
    - counter=1, 2 at gens 3, 4
    - counter=0 at gen 5 (strict improvement, reset)
    - counter=1, 2, 3, 4 at gens 6-9
    - never reaches 5

    Loop runs the full ``generations: 9`` budget.
    """
    pop = 4
    trajectory = [0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]
    fitness = ScriptedFitness(trajectory, pop)
    loop = _make_loop(
        tmp_path,
        fitness=fitness,
        generations=9,
        population_size=pop,
        early_stop_on_saturation=5,
    )
    loop.run()

    # Lineage should have 9 generations x pop rows + header.
    with (tmp_path / "lineage.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1 + 9 * pop


def test_early_stop_compares_to_prev_gen_not_running_max(tmp_path: Path) -> None:
    """Trajectory ``[0.3, 0.5, 0.4, 0.45, 0.46, 0.47, 0.48, 0.49]`` with N=3 → never fires.

    The trajectory regresses from 0.5 to 0.4 at gen 3 then recovers
    monotonically (0.45 > 0.4, 0.46 > 0.45, ...).  Under the
    spec-correct prev-gen comparison, gens 4-8 are all strict
    improvements over the immediately-prior generation, so the counter
    resets each time and never reaches N=3.

    Under the (rejected) running-max comparison, every gen 3+ would be
    counted as non-improving because none surpass the all-time peak of
    0.5, and the counter would fire at gen 5.  This test pins the
    correct semantics and would have caught the original bug.
    """
    pop = 4
    trajectory = [0.3, 0.5, 0.4, 0.45, 0.46, 0.47, 0.48, 0.49]
    fitness = ScriptedFitness(trajectory, pop)
    loop = _make_loop(
        tmp_path,
        fitness=fitness,
        generations=8,
        population_size=pop,
        early_stop_on_saturation=3,
    )
    loop.run()

    # Lineage should have 8 generations x pop rows + header (no early-stop).
    with (tmp_path / "lineage.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1 + 8 * pop


# ---------------------------------------------------------------------------
# Resume preserves the counter
# ---------------------------------------------------------------------------


def test_resume_preserves_early_stop_counter(tmp_path: Path) -> None:
    """Resume SHALL restore both ``_gens_without_improvement`` and ``_last_best_fitness``.

    Trajectory: `[0.5, 0.5, 0.5]` with N=3, generations=3.  Run 2 gens
    (counter=0, then 1), save checkpoint, resume, run gen 3 (counter
    bumps to 2 — does NOT trigger early-stop because N=3 not 2).  The
    test asserts the counter and last_best_fitness are preserved
    correctly across the save/load boundary.
    """
    pop = 4
    fitness1 = ScriptedFitness([0.5, 0.5], pop)
    loop1 = _make_loop(
        tmp_path,
        fitness=fitness1,
        generations=2,
        population_size=pop,
        early_stop_on_saturation=3,
        checkpoint_every=1,  # save after every generation
    )
    loop1.run()

    # After 2 gens with flat trajectory: counter=1, last_best_fitness=0.5.
    with (tmp_path / "checkpoint.pkl").open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local file
    assert payload["gens_without_improvement"] == 1
    assert payload["last_best_fitness"] == pytest.approx(0.5)

    # Resume with a fresh loop instance + new scripted fitness.
    fitness2 = ScriptedFitness([0.5, 0.5], pop)
    loop2 = _make_loop(
        tmp_path,
        fitness=fitness2,
        generations=3,
        population_size=pop,
        early_stop_on_saturation=3,
        checkpoint_every=1,
    )
    loop2.run(resume_from=tmp_path / "checkpoint.pkl")

    # After resume + 1 more gen: counter=2 (still flat, but N=3 not yet hit).
    with (tmp_path / "checkpoint.pkl").open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local file
    assert payload["gens_without_improvement"] == 2
    assert payload["last_best_fitness"] == pytest.approx(0.5)
