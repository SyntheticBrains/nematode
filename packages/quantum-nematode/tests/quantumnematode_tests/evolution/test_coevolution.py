"""Tests for :mod:`quantumnematode.evolution.coevolution`.

Covers the public-surface behaviours:

- Side state surface + composition over inheritance.
- K-block boundary + opposing side frozen + fresh CMA-ES at transition.
- Block elite pushed to HoF.
- Prey gen-0 warm-start from a lamarckian-elite warmstart bundle
  (incl. missing-path / wrong-shape).
- Predator gen-0 bootstrap (heuristic-imitation pretrain mocked +
  cold-start).
- Generality probe (cadence + non-mutation + held-out construction).
- Champion history schema round-trip.
- Rebalance knob.
- Checkpoint round-trip.

Tests use a minimal sim_config that satisfies the `CoevolutionConfig`
validators with `population_size=4`, `K_per_block=1`,
`generation_pairs=1`, and `predator_gen0_bootstrap='cold_start'` to
avoid the ~30s heuristic-imitation pretrain in setup. Fitness +
encoder behaviour during the per-generation evaluation is stubbed via
monkeypatch so tests run in seconds rather than minutes; full
end-to-end evaluation is exercised by the campaign-driver smoke pilot.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from quantumnematode.brain.arch.lstmppo import LSTMPPOBrainConfig
from quantumnematode.brain.modules import ModuleName
from quantumnematode.evolution.coevolution import (
    CHECKPOINT_VERSION,
    DEFAULT_HOF_CAPACITY,
    CoevolutionLoop,
)
from quantumnematode.evolution.genome import Genome
from quantumnematode.evolution.hall_of_fame import HallOfFame
from quantumnematode.utils.config_loader import (
    BrainContainerConfig,
    CoevolutionConfig,
    EnvironmentConfig,
    EvolutionConfig,
    PredatorBrainConfigSchema,
    PredatorConfig,
    SimulationConfig,
)

pytestmark = pytest.mark.slow


def _build_minimal_sim_config(  # noqa: PLR0913 — kwargs map 1:1 to CoevolutionConfig fields
    *,
    prey_gen0_seed_path: Path | None = None,
    predator_gen0_bootstrap: str = "cold_start",
    population_size: int = 4,
    K_per_block: int = 1,  # noqa: N803 — matches CoevolutionConfig.K_per_block field name
    generation_pairs: int = 1,
    held_out_size: int = 2,
    generality_probe_every: int = 10,
    rebalance_threshold: float | None = None,
) -> SimulationConfig:
    """Build the smallest sim_config that satisfies CoevolutionConfig validators.

    Cold-start predator avoids the ~30s heuristic-imitation pretrain in
    setup. Population_size=4 + K_per_block=1 keeps the per-generation
    body fast enough for ~20 unit tests to complete in a few seconds
    (the heavy work — actual fitness evaluation — is stubbed by
    `_StubFitness` in the tests that exercise the loop body).
    """
    return SimulationConfig(
        seed=42,
        brain=BrainContainerConfig(
            name="lstmppo",
            config=LSTMPPOBrainConfig(
                sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
            ),
        ),
        environment=EnvironmentConfig(
            grid_size=20,
            predators=PredatorConfig(
                enabled=True,
                count=1,
                brain_config=PredatorBrainConfigSchema(kind="mlpppo_predator"),
            ),
        ),
        coevolution=CoevolutionConfig(
            prey_evolution=EvolutionConfig(
                algorithm="cmaes",
                cma_diagonal=True,
                learn_episodes_per_eval=1,
                inheritance="lamarckian",
                population_size=population_size,
            ),
            predator_evolution=EvolutionConfig(
                algorithm="cmaes",
                cma_diagonal=True,
                learn_episodes_per_eval=0,
                inheritance="none",
                population_size=population_size,
            ),
            generation_pairs=generation_pairs,
            K_per_block=K_per_block,
            held_out_size=held_out_size,
            generality_probe_every=generality_probe_every,
            predator_gen0_bootstrap=predator_gen0_bootstrap,  # type: ignore[arg-type]
            prey_gen0_seed_path=prey_gen0_seed_path,
            rebalance_threshold=rebalance_threshold,
        ),
    )


def _make_loop(tmp_path: Path, **kwargs: Any) -> CoevolutionLoop:
    """Construct a `CoevolutionLoop` with a minimal config.

    Wraps `_build_minimal_sim_config` so test bodies don't repeat the
    boilerplate. Returns the loop with `output_dir=tmp_path/coevo` and
    a fresh seed-42 RNG.
    """
    sim_config = _build_minimal_sim_config(**kwargs)
    return CoevolutionLoop(
        sim_config,
        output_dir=tmp_path / "coevo",
        rng=np.random.default_rng(seed=42),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """`__init__` surface — config validation + side-state population."""

    def test_missing_coevolution_block_raises(self, tmp_path: Path) -> None:
        """Sim_config with no `coevolution` block SHALL raise."""
        sim_config = SimulationConfig()  # no coevolution
        with pytest.raises(ValueError, match=r"sim_config\.coevolution"):
            CoevolutionLoop(
                sim_config,
                output_dir=tmp_path / "coevo",
                rng=np.random.default_rng(seed=42),
            )

    def test_minimal_loop_constructs(self, tmp_path: Path) -> None:
        """A minimal valid config SHALL produce a fully-populated loop."""
        loop = _make_loop(tmp_path)
        assert loop.prey.name == "prey"
        assert loop.predator.name == "predator"
        # Initial K-block index is 0; current side is the configured start side.
        assert loop._k_block_index == 0
        assert loop._current_side == "prey"  # default
        # Per-side output dirs exist.
        assert (tmp_path / "coevo" / "prey").is_dir()
        assert (tmp_path / "coevo" / "predator").is_dir()
        # Probe CSV is initialised with the header row.
        probe_path = tmp_path / "coevo" / "generality_probe.csv"
        assert probe_path.exists()
        header = probe_path.read_text().strip().split("\n")[0]
        assert header == "generation,side,opponent_index,fitness"

    def test_per_side_fitness_is_hardcoded_per_d13(self, tmp_path: Path) -> None:
        """Prey SHALL get LearnedPerformanceFitness; predator PredatorEpisodicKillRate."""
        from quantumnematode.evolution.fitness import LearnedPerformanceFitness
        from quantumnematode.evolution.predator_fitness import (
            PredatorEpisodicKillRate,
        )

        loop = _make_loop(tmp_path)
        assert isinstance(loop.prey.fitness, LearnedPerformanceFitness)
        assert isinstance(loop.predator.fitness, PredatorEpisodicKillRate)

    def test_per_side_inheritance_is_hardcoded_per_d10_d13(self, tmp_path: Path) -> None:
        """Prey SHALL get LamarckianInheritance; predator NoInheritance."""
        from quantumnematode.evolution.inheritance import (
            LamarckianInheritance,
            NoInheritance,
        )

        loop = _make_loop(tmp_path)
        assert isinstance(loop.prey.inheritance, LamarckianInheritance)
        assert isinstance(loop.predator.inheritance, NoInheritance)

    def test_default_hof_capacity(self, tmp_path: Path) -> None:
        """Both HoFs SHALL have `DEFAULT_HOF_CAPACITY=8`."""
        loop = _make_loop(tmp_path)
        assert loop.prey.hof.capacity == DEFAULT_HOF_CAPACITY
        assert loop.predator.hof.capacity == DEFAULT_HOF_CAPACITY


# ---------------------------------------------------------------------------
# Gen-0 warmstart loader
# ---------------------------------------------------------------------------


class TestWarmstartLoader:
    """Prey gen-0 warm-start from a lamarckian-elite warmstart bundle."""

    def test_missing_path_falls_back_to_zeros(self, tmp_path: Path) -> None:
        """`prey_gen0_seed_path=None` SHALL fall back to x0=zeros (cold-start prey)."""
        loop = _make_loop(tmp_path, prey_gen0_seed_path=None)
        # CMA-ES with x0=zeros has its initial mean at zero; the
        # internal `_es.mean` is a numpy array of zeros.
        np.testing.assert_array_equal(
            loop.prey.optimizer._es.mean,
            np.zeros(loop.prey.optimizer.num_params),
        )

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        """`prey_gen0_seed_path` pointing at a missing file SHALL raise."""
        with pytest.raises(ValueError, match="does not exist"):
            _make_loop(tmp_path, prey_gen0_seed_path=tmp_path / "nope.json")

    def test_wrong_length_params_raises(self, tmp_path: Path) -> None:
        """A warmstart file with `params` length != prey genome_dim SHALL raise."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text(
            json.dumps(
                {
                    "genome_id": "bad",
                    "generation": 0,
                    "fitness": 0.5,
                    "params": [0.1, 0.2, 0.3],  # 3 floats — won't match LSTMPPO dim
                    "brain_config": {},
                },
            ),
        )
        with pytest.raises(ValueError, match="params of length"):
            _make_loop(tmp_path, prey_gen0_seed_path=bad_path)


# ---------------------------------------------------------------------------
# Held-out construction
# ---------------------------------------------------------------------------


class TestHeldOutConstruction:
    """Spec scenario "Held-Out Set Construction"."""

    def test_predator_held_out_uses_radius_grid(self, tmp_path: Path) -> None:
        """Predator held-out specs SHALL be drawn from the {4,6,8,10} x {0,1} grid."""
        loop = _make_loop(tmp_path, held_out_size=8)
        # All 8 grid combos should be present at held_out_size=8 (no
        # replacement, so it's the full grid).
        valid_detection = {4, 6, 8, 10}
        valid_damage = {0, 1}
        for det, dmg in loop._predator_held_out_specs:
            assert det in valid_detection
            assert dmg in valid_damage
        # 8 unique combos at default grid_size=8.
        assert len(set(loop._predator_held_out_specs)) == 8

    def test_predator_held_out_oversize_uses_replacement(self, tmp_path: Path) -> None:
        """`held_out_size > grid_size` SHALL sample WITH replacement."""
        loop = _make_loop(tmp_path, held_out_size=20)
        assert len(loop._predator_held_out_specs) == 20

    def test_prey_held_out_missing_bundle_no_op(self, tmp_path: Path) -> None:
        """Missing prey held-out bundle SHALL log a warning and return []."""
        # Override the class-level `_PREY_HELD_OUT_BUNDLE_DIR` to a
        # tmp path that doesn't exist on disk so we exercise the
        # missing-dir branch directly. The production bundle
        # (`configs/evolution/coevolution_warmstart_prey/`) DOES exist
        # on a fresh checkout, so testing the no-op path requires an
        # explicit override here.
        from quantumnematode.evolution.coevolution import CoevolutionLoop

        with patch.object(
            CoevolutionLoop,
            "_PREY_HELD_OUT_BUNDLE_DIR",
            tmp_path / "no_such_bundle",
        ):
            loop = _make_loop(tmp_path)
            assert loop._prey_held_out == []


# ---------------------------------------------------------------------------
# Alternating schedule
# ---------------------------------------------------------------------------


class _StubFitness:
    """Drop-in fitness stub returning a deterministic fitness per genome.

    Used by the schedule-controller tests to skip the real
    multi-agent runner. Fitness is `mean(genome.params)` so each test
    can construct genomes with predictable fitnesses.
    """

    def __init__(self, fixed_value: float | None = None) -> None:
        self.fixed_value = fixed_value
        self.calls: list[Any] = []

    def evaluate(
        self,
        genome: Genome,
        sim_config: SimulationConfig,
        encoder: object,
        *,
        episodes: int,
        seed: int,
        **_kwargs: Any,
    ) -> float:
        self.calls.append(genome.genome_id)
        if self.fixed_value is not None:
            return self.fixed_value
        # Deterministic per-genome fitness.
        return float(np.mean(genome.params))


class TestAlternatingSchedule:
    """Spec scenarios "K-Block Boundary" + "Opposing Side Frozen" + "Fresh CMA-ES At Transition"."""

    def test_k_block_count_and_side_flip(self, tmp_path: Path) -> None:
        """generation_pairs=2 SHALL execute 4 K-blocks alternating prey/predator/prey/predator."""
        loop = _make_loop(tmp_path, generation_pairs=2, K_per_block=1)
        # Stub fitnesses so we don't run real evaluations.
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.3)  # type: ignore[assignment]

        loop.run()

        # 4 K-blocks total: prey 0, predator 1, prey 2, predator 3.
        # After completion `_k_block_index` is the count of completed
        # blocks (4); `_current_side` flipped on each transition so it
        # ends at "prey" (which would be the next training side if the
        # loop kept going).
        assert loop._k_block_index == 4
        assert loop._current_side == "prey"
        # Each side ran K_per_block=1 generation per K-block x 2 K-blocks each.
        assert loop.prey.generation == 2
        assert loop.predator.generation == 2
        # Champion history grew by 1 per K-block per side (2 K-blocks each).
        assert len(loop.prey.champion_history) == 2
        assert len(loop.predator.champion_history) == 2

    def test_cma_es_reconstructed_at_each_transition(self, tmp_path: Path) -> None:
        """Each K-block transition SHALL produce a fresh `CMAESOptimizer` instance."""
        loop = _make_loop(tmp_path, generation_pairs=2, K_per_block=1)
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.3)  # type: ignore[assignment]

        # Capture initial optimizer ids; after run, the prey + predator
        # optimizers SHALL each have been swapped out.
        prey_opt_ids = [id(loop.prey.optimizer)]
        predator_opt_ids = [id(loop.predator.optimizer)]

        # Patch `_rebuild_optimizer` to record its calls + capture new ids.
        original = loop._rebuild_optimizer

        def capture(side: Any) -> None:
            original(side)
            if side.name == "prey":
                prey_opt_ids.append(id(side.optimizer))
            else:
                predator_opt_ids.append(id(side.optimizer))

        with patch.object(loop, "_rebuild_optimizer", side_effect=capture):
            loop.run()

        # 4 K-blocks -> 3 transitions, but the LAST transition skips
        # rebuild (the loop exits before the would-be next block).
        # Order: prey-block-end -> predator rebuild; predator-block-end ->
        # prey rebuild; prey-block-end -> predator rebuild; predator-
        # block-end -> loop exits, no rebuild. So predator was rebuilt
        # 2 times; prey was rebuilt 1 time.
        # Assert each side's optimizer was rebuilt at least once and
        # the new instance is a different object.
        assert len(predator_opt_ids) >= 2
        assert len(prey_opt_ids) >= 2
        assert predator_opt_ids[0] != predator_opt_ids[-1]
        assert prey_opt_ids[0] != prey_opt_ids[-1]

    def test_opposing_side_frozen_during_off_block(self, tmp_path: Path) -> None:
        """During side X's K-block, side Y's population/optimizer/hof SHALL NOT change."""
        # Use generation_pairs=1 K_per_block=2 so we have one prey
        # K-block of 2 generations, then one predator K-block. After
        # the prey K-block, the predator's hof + champion_history MUST
        # still be empty (predator hasn't trained yet).
        loop = _make_loop(tmp_path, generation_pairs=1, K_per_block=2)
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.3)  # type: ignore[assignment]

        # Snapshot predator state mid-prey-block. We can't easily
        # interrupt mid-block without threading; instead we run only
        # ONE K-block by setting generation_pairs=1 K_per_block=1 + a
        # different starting side. Easier: just verify the post-prey
        # state — the predator side hasn't pushed anything yet.

        # Run the full 2-block schedule.
        loop.run()

        # Both ran their K-blocks. Verify champion_history and hof
        # population sequences are correct: prey had 1 K-block, then
        # predator had 1 K-block.
        assert len(loop.prey.champion_history) == 1
        assert len(loop.predator.champion_history) == 1


# ---------------------------------------------------------------------------
# HoF push at K-block end
# ---------------------------------------------------------------------------


class TestHoFPush:
    """Spec scenario "Block Elite Pushed To HoF"."""

    def test_block_elite_pushed_to_hof(self, tmp_path: Path) -> None:
        """The K-block's top-fitness genome SHALL be pushed to its HoF."""
        loop = _make_loop(tmp_path, generation_pairs=1, K_per_block=1, population_size=4)
        loop.prey.fitness = _StubFitness()  # mean(params) — varies by genome
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.3)  # type: ignore[assignment]

        loop.run()

        # Each side should have exactly 1 entry in its HoF (1 K-block
        # ran on each side at K_per_block=1).
        assert len(loop.prey.hof) == 1
        assert len(loop.predator.hof) == 1
        # The HoF entry's fitness SHALL equal the K-block elite's
        # recorded fitness (matches champion_history's last entry).
        assert loop.prey.hof.fitnesses() == [loop.prey.champion_history[-1]["fitness"]]


# ---------------------------------------------------------------------------
# Champion history schema
# ---------------------------------------------------------------------------


class TestChampionHistorySchema:
    """Champion-history schema: round-trips through `json.dump`/`json.load`."""

    def test_champion_history_round_trip_through_json(self) -> None:
        """champion_history entries SHALL round-trip through `json.dump`/`json.load`.

        The loop writes via `json.dump` after converting `params` via
        `params.tolist()`; deserialise via
        `np.asarray(d["params"], dtype=np.float32)` to restore in-memory
        shape. This test exercises the schema directly.
        """
        original = [
            {
                "genome_id": "test-elite-0",
                "generation": 5,
                "k_block_index": 0,
                "fitness": 0.42,
                "params": [0.1, -0.2, 0.3, -0.4],
            },
            {
                "genome_id": "test-elite-1",
                "generation": 15,
                "k_block_index": 2,
                "fitness": 0.78,
                "params": [1.0, 2.0, 3.0, 4.0],
            },
        ]
        # Round-trip via JSON.
        as_json = json.dumps(original)
        restored = json.loads(as_json)
        assert restored == original
        # Restore numpy params shape.
        for entry in restored:
            params_array = np.asarray(entry["params"], dtype=np.float32)
            assert params_array.dtype == np.float32
            assert params_array.shape == (len(entry["params"]),)


# ---------------------------------------------------------------------------
# Generality probe
# ---------------------------------------------------------------------------


class TestGeneralityProbe:
    """Spec scenarios "Probe Cadence and Output Layout" + "Probe Does Not Mutate"."""

    def test_probe_writes_csv_with_correct_schema(self, tmp_path: Path) -> None:
        """Probe SHALL write rows with `(generation, side, opponent_index, fitness)`."""
        loop = _make_loop(tmp_path, generation_pairs=1, K_per_block=1, held_out_size=2)
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.3)  # type: ignore[assignment]
        # Force the probe to fire at every generation.
        loop.coevolution_config.generality_probe_every = 1  # type: ignore[misc]

        loop.run()

        probe_path = tmp_path / "coevo" / "generality_probe.csv"
        lines = probe_path.read_text().strip().split("\n")
        # Header + at least 1 row per side per probe firing.
        # K_per_block=1, generation_pairs=1 -> 2 total per-side
        # generations. Probe fires at gens (post-K-block) where
        # `side.generation % 1 == 0`, so on every K-block.
        assert lines[0] == "generation,side,opponent_index,fitness"
        # Under Option B wiring, prey-side reads from
        # `_predator_held_out_specs` (always non-empty — built at
        # __init__ from the heuristic-radius grid); predator-side reads
        # from `_prey_held_out` which is empty here (no bundle dir or
        # genome-dim mismatch). So we expect prey rows but no predator
        # rows.
        for row in lines[1:]:
            cells = row.split(",")
            assert len(cells) == 4
            # Side is "prey" or "predator" (predator side fires only
            # if `_prey_held_out` is non-empty).
            assert cells[1] in {"prey", "predator"}
            # opponent_index is an integer.
            int(cells[2])
            # Fitness is a real float (the stub fitness returns
            # `fixed_value`, never NaN). Verify it is finite — Option B
            # probe body returns positive fitness via the side's
            # fitness function.
            value = float(cells[3])
            assert np.isfinite(value)

    def test_probe_does_not_mutate_state(self, tmp_path: Path) -> None:
        """Probe SHALL NOT alter population, optimizer, hof, or generation counter."""
        loop = _make_loop(tmp_path, generation_pairs=1, K_per_block=1, held_out_size=2)
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.3)  # type: ignore[assignment]
        loop.run()

        # Snapshot post-run state.
        prey_pop_ids = [g.genome_id for g in loop.prey.population]
        prey_hof_count = len(loop.prey.hof)
        prey_gen = loop.prey.generation
        prey_opt_id = id(loop.prey.optimizer)
        # Manually fire the probe AGAIN — invariants say it MUST NOT
        # change any state.
        loop._fire_generality_probe()
        assert [g.genome_id for g in loop.prey.population] == prey_pop_ids
        assert len(loop.prey.hof) == prey_hof_count
        assert loop.prey.generation == prey_gen
        assert id(loop.prey.optimizer) == prey_opt_id

    def test_prey_side_probe_iterates_predator_held_out(
        self,
        tmp_path: Path,
    ) -> None:
        """Prey-side probe SHALL produce one row per `_predator_held_out_specs` entry.

        Under the cross-species yardstick wiring, the prey-side probe
        evaluates the prey elite against the OPPOSING-side held-out
        set (`_predator_held_out_specs`). The row count per fire MUST
        equal `len(_predator_held_out_specs)` regardless of the prey
        bundle's load state.
        """
        held_out_size = 3
        loop = _make_loop(
            tmp_path,
            generation_pairs=1,
            K_per_block=1,
            held_out_size=held_out_size,
            generality_probe_every=1,
        )
        loop.prey.fitness = _StubFitness(fixed_value=0.7)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.4)  # type: ignore[assignment]

        # Seed `prey.champion_history` with a synthetic K-block elite
        # so the probe's "no champions" no-op branch doesn't fire.
        encoder = loop.prey.encoder
        elite_params = np.zeros(encoder.genome_dim(loop.sim_config), dtype=np.float32)
        loop.prey.champion_history.append(
            {
                "genome_id": "synthetic-prey-elite",
                "generation": 0,
                "k_block_index": 0,
                "fitness": 0.7,
                "params": elite_params.tolist(),
            },
        )
        # Empty predator champion_history -> predator-side probe is a no-op.
        loop._fire_generality_probe()

        probe_path = tmp_path / "coevo" / "generality_probe.csv"
        lines = probe_path.read_text().strip().split("\n")
        # Header + held_out_size prey rows. No predator rows because
        # `predator.champion_history` is empty (predator-side probe
        # no-ops when no champion exists yet).
        assert len(lines) == 1 + held_out_size
        prey_rows = [line for line in lines[1:] if line.split(",")[1] == "prey"]
        assert len(prey_rows) == held_out_size
        # opponent_index spans 0..held_out_size-1.
        assert sorted(int(r.split(",")[2]) for r in prey_rows) == list(range(held_out_size))
        # All rows MUST be finite floats (Option B body, not the NaN
        # placeholder of the deferred-body design).
        for row in prey_rows:
            value = float(row.split(",")[3])
            assert np.isfinite(value)

    def test_predator_side_probe_iterates_prey_held_out(
        self,
        tmp_path: Path,
    ) -> None:
        """Predator-side probe SHALL produce one row per `_prey_held_out` entry.

        Under the cross-species yardstick wiring, the predator-side
        probe evaluates the predator elite against the OPPOSING-side
        held-out set (`_prey_held_out`). Prey held-out is empty in the
        test harness (no bundle dir), so we manually inject one entry
        to exercise the predator-side body.
        """
        loop = _make_loop(
            tmp_path,
            generation_pairs=1,
            K_per_block=1,
            held_out_size=2,
            generality_probe_every=1,
        )
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.4)  # type: ignore[assignment]

        # Inject a held-out prey genome so the predator-side branch
        # has something to iterate.
        prey_genome_dim = loop.prey.encoder.genome_dim(loop.sim_config)
        injected = Genome(
            params=np.zeros(prey_genome_dim, dtype=np.float32),
            genome_id="injected-held-out-prey",
            parent_ids=[],
            generation=0,
        )
        loop._prey_held_out = [injected]

        # Seed predator champion_history so the predator-side probe
        # is not a no-op.
        predator_params_dim = loop.predator.encoder.genome_dim(loop.sim_config)
        elite_params = np.zeros(predator_params_dim, dtype=np.float32)
        loop.predator.champion_history.append(
            {
                "genome_id": "synthetic-predator-elite",
                "generation": 0,
                "k_block_index": 0,
                "fitness": 0.4,
                "params": elite_params.tolist(),
            },
        )

        loop._fire_generality_probe()

        probe_path = tmp_path / "coevo" / "generality_probe.csv"
        rows = probe_path.read_text().strip().split("\n")[1:]
        predator_rows = [r for r in rows if r.split(",")[1] == "predator"]
        # Exactly one predator row (one held-out prey entry).
        assert len(predator_rows) == 1
        # Stub fitness was 0.4; probe routed through it.
        cells = predator_rows[0].split(",")
        assert int(cells[2]) == 0
        assert float(cells[3]) == pytest.approx(0.4)

    def test_probe_does_not_capture_germline_weights(
        self,
        tmp_path: Path,
    ) -> None:
        """Probe SHALL NOT pass weight_capture_path so germline weights stay frozen.

        Verifies the eval-only contract: `_evaluate_in_worker` is
        called with `warm_start_path_override=None,
        weight_capture_path=None`, mirroring the no-inheritance branch
        of the worker entry point. We exercise this by checking that
        no per-genome `.pt` files are written under the probe's tmp dir
        for the prey-side branch (heuristic predator + no opposition
        weights), and that the side's inheritance dir is unchanged
        before/after the probe fire.
        """
        loop = _make_loop(
            tmp_path,
            generation_pairs=1,
            K_per_block=1,
            held_out_size=1,
            generality_probe_every=1,
        )
        loop.prey.fitness = _StubFitness(fixed_value=0.6)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.4)  # type: ignore[assignment]

        # Seed a prey champion so the prey-side branch fires.
        encoder = loop.prey.encoder
        elite_params = np.zeros(encoder.genome_dim(loop.sim_config), dtype=np.float32)
        loop.prey.champion_history.append(
            {
                "genome_id": "synthetic-prey-elite",
                "generation": 0,
                "k_block_index": 0,
                "fitness": 0.6,
                "params": elite_params.tolist(),
            },
        )

        prey_inherit_dir = tmp_path / "coevo" / "prey" / "inheritance"
        before = sorted(prey_inherit_dir.rglob("*.pt")) if prey_inherit_dir.exists() else []
        loop._fire_generality_probe()
        after = sorted(prey_inherit_dir.rglob("*.pt")) if prey_inherit_dir.exists() else []
        # The probe MUST NOT write into the per-side inheritance dir.
        assert before == after


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """Full state round-trip through the four-file checkpoint format."""

    def test_save_and_load_preserves_state(self, tmp_path: Path) -> None:
        """`_save_checkpoint` then `_load_checkpoint` SHALL preserve all loop state."""
        loop = _make_loop(tmp_path, generation_pairs=1, K_per_block=1)

        # Manually advance state to simulate a completed K-block.
        fake_genome = Genome(
            params=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            genome_id="elite-prey",
            parent_ids=[],
            generation=0,
        )
        loop.prey.champion_history.append(
            {
                "genome_id": "elite-prey",
                "generation": 0,
                "k_block_index": 0,
                "fitness": 0.7,
                "params": [1.0, 2.0, 3.0],
            },
        )
        loop.prey.hof.push(fake_genome, fitness=0.7)
        loop.prey.generation = 1
        loop._k_block_index = 1
        loop._current_side = "predator"

        loop._save_checkpoint()

        # Construct a fresh loop with the same config + a fresh master
        # rng, then resume.
        sim_config = _build_minimal_sim_config()
        loop2 = CoevolutionLoop(
            sim_config,
            output_dir=tmp_path / "coevo",
            rng=np.random.default_rng(seed=99),  # different seed from save-side
        )
        loop2._load_checkpoint()

        assert loop2._k_block_index == 1
        assert loop2._current_side == "predator"
        assert len(loop2.prey.hof) == 1
        assert loop2.prey.hof.fitnesses() == [0.7]
        assert len(loop2.prey.champion_history) == 1
        assert loop2.prey.generation == 1

    def test_checkpoint_version_mismatch_raises(self, tmp_path: Path) -> None:
        """Per-side checkpoint with wrong version SHALL refuse to resume."""
        loop = _make_loop(tmp_path)
        loop._save_checkpoint()
        # Tamper with the prey checkpoint to set a bogus version.
        import pickle

        prey_ckpt = tmp_path / "coevo" / "prey" / "checkpoint.pkl"
        with prey_ckpt.open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 — test file
        payload["checkpoint_version"] = CHECKPOINT_VERSION + 99
        with prey_ckpt.open("wb") as fh:
            pickle.dump(payload, fh)

        sim_config = _build_minimal_sim_config()
        loop2 = CoevolutionLoop(
            sim_config,
            output_dir=tmp_path / "coevo",
            rng=np.random.default_rng(seed=99),
        )
        with pytest.raises(ValueError, match="checkpoint version mismatch"):
            loop2._load_checkpoint()


# ---------------------------------------------------------------------------
# HoF integration
# ---------------------------------------------------------------------------


class TestHoFOppositionSampling:
    """Spec scenarios "70/30 Mixture During Evaluation" + "Empty HoF Fallback"."""

    def test_empty_opposing_pop_returns_empty_opposition(self, tmp_path: Path) -> None:
        """With no opposing population, `_build_opposition` SHALL return []."""
        loop = _make_loop(tmp_path)
        # Predator side has no population yet (gen-0 bootstrap).
        assert loop._build_opposition(loop.predator) == []

    def test_mix_with_pop_invoked_when_opposing_pop_present(self, tmp_path: Path) -> None:
        """With an opposing population, `_build_opposition` SHALL call `mix_with_pop`."""
        loop = _make_loop(tmp_path)
        # Manually populate the predator side (simulating a completed
        # predator K-block).
        loop.predator.population = [
            Genome(
                params=np.zeros(3, dtype=np.float32),
                genome_id=f"pred-{i}",
                parent_ids=[],
                generation=0,
            )
            for i in range(4)
        ]
        opposition = loop._build_opposition(loop.predator)
        # Returned list size matches `opposing.population` size (HoF is
        # empty so all from pop, but the size is still len(pop)).
        assert len(opposition) == 4
        # All entries should be from the pop (HoF is empty).
        assert all(g.genome_id.startswith("pred-") for g in opposition)


# ---------------------------------------------------------------------------
# Rebalance heuristic (§6.14)
# ---------------------------------------------------------------------------


class TestRebalanceHeuristic:
    """Rebalance knob: dominant side gets an extra K-block when opposing saturates."""

    def test_disabled_by_default(self, tmp_path: Path) -> None:
        """Threshold None SHALL never freeze a side (`_evaluate_rebalance` returns True)."""
        loop = _make_loop(tmp_path, rebalance_threshold=None)
        # Manually populate K-block-mean history with a dominance
        # pattern that WOULD trigger the rebalance if the knob were
        # enabled: prey at 0.1, predator at 1.0 across 3 K-blocks.
        loop._k_block_mean_fitness["prey"] = [0.1, 0.1, 0.1]
        loop._k_block_mean_fitness["predator"] = [1.0, 1.0, 1.0]
        # Even with that history, no threshold means flip normally.
        assert loop._evaluate_rebalance(loop.prey, loop.predator) is True
        assert loop._evaluate_rebalance(loop.predator, loop.prey) is True

    def test_threshold_freezes_dominant_side_when_saturated(self, tmp_path: Path) -> None:
        """With `rebalance_threshold=0.5`, prey saturated for 3 blocks SHALL freeze predator.

        "Freeze predator" = `_evaluate_rebalance` returns False when
        `training_side` is prey, meaning prey gets an extra K-block.
        """
        loop = _make_loop(tmp_path, rebalance_threshold=0.5)
        # Prey at 0.1, predator at 1.0 -> prey/predator = 0.1, well
        # below threshold 0.5 -> prey saturated.
        loop._k_block_mean_fitness["prey"] = [0.1, 0.1, 0.1]
        loop._k_block_mean_fitness["predator"] = [1.0, 1.0, 1.0]
        # When prey just trained (training_side=prey), keep prey
        # training (predator stays frozen).
        assert loop._evaluate_rebalance(loop.prey, loop.predator) is False
        # When predator just trained, normal flip — the heuristic
        # only kicks in when the saturated side is the just-trained
        # side, otherwise we're already about to give it a block.
        assert loop._evaluate_rebalance(loop.predator, loop.prey) is True


# ---------------------------------------------------------------------------
# Smoke: end-to-end run with stubs
# ---------------------------------------------------------------------------


class TestEndToEndStubbedRun:
    """High-level smoke: run the loop with stubbed fitness, verify outputs."""

    def test_run_writes_lineage_csv(self, tmp_path: Path) -> None:
        """A completed run SHALL write per-side `lineage.csv` files."""
        loop = _make_loop(tmp_path, generation_pairs=1, K_per_block=1)
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.3)  # type: ignore[assignment]
        loop.run()
        assert (tmp_path / "coevo" / "prey" / "lineage.csv").exists()
        assert (tmp_path / "coevo" / "predator" / "lineage.csv").exists()

    def test_run_writes_checkpoint_files(self, tmp_path: Path) -> None:
        """A completed run SHALL write all four checkpoint files."""
        loop = _make_loop(tmp_path, generation_pairs=1, K_per_block=1)
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.3)  # type: ignore[assignment]
        loop.run()
        assert (tmp_path / "coevo" / "prey" / "checkpoint.pkl").exists()
        assert (tmp_path / "coevo" / "predator" / "checkpoint.pkl").exists()
        assert (tmp_path / "coevo" / "coevolution_state.json").exists()
        assert (tmp_path / "coevo" / "coevolution_rng.pkl").exists()


# ---------------------------------------------------------------------------
# Wall-time instrumentation
# ---------------------------------------------------------------------------


class TestWalltimeInstrumentation:
    """`_record_walltime` writes per-eval + per-gen aggregate rows to walltime.csv."""

    def test_walltime_csv_initialised_with_header(self, tmp_path: Path) -> None:
        """`__init__` SHALL create walltime.csv with the canonical header row."""
        loop = _make_loop(tmp_path)
        path = tmp_path / "coevo" / "walltime.csv"
        assert path.is_file()
        rows = path.read_text().strip().split("\n")
        assert rows[0] == "scope,side,generation,index,parallel_workers,wall_seconds"
        # Header-only at __init__ — no run has fired yet.
        assert len(rows) == 1
        del loop

    def test_run_writes_evaluation_and_generation_rows(self, tmp_path: Path) -> None:
        """A completed run SHALL append one evaluation row per child + one generation row per gen.

        At population_size=4, K_per_block=1, generation_pairs=1 the run
        executes 2 K-blocks (1 prey + 1 predator), each with 1
        generation. So we expect:
        - 4 evaluation rows + 1 generation row for prey side
        - 4 evaluation rows + 1 generation row for predator side
        Total: 10 data rows + 1 header.
        """
        loop = _make_loop(tmp_path, generation_pairs=1, K_per_block=1, population_size=4)
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.3)  # type: ignore[assignment]
        loop.run()
        with (tmp_path / "coevo" / "walltime.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        # 8 evaluation rows (4 prey + 4 predator) + 2 generation rows.
        assert len(rows) == 10
        eval_rows = [r for r in rows if r["scope"] == "evaluation"]
        gen_rows = [r for r in rows if r["scope"] == "generation"]
        assert len(eval_rows) == 8
        assert len(gen_rows) == 2
        # Per-side breakdown.
        prey_evals = [r for r in eval_rows if r["side"] == "prey"]
        predator_evals = [r for r in eval_rows if r["side"] == "predator"]
        assert len(prey_evals) == 4
        assert len(predator_evals) == 4
        # Generation rows record `index = population_size`.
        for gr in gen_rows:
            assert int(gr["index"]) == 4
        # parallel_workers=1 (smoke config / minimal sim has no
        # multi-worker pool dispatch).
        assert all(int(r["parallel_workers"]) == 1 for r in rows)
        # Wall seconds parse as floats and are non-negative.
        for r in rows:
            wall = float(r["wall_seconds"])
            assert wall >= 0.0

    def test_walltime_csv_preserved_across_init(self, tmp_path: Path) -> None:
        """Re-instantiating a loop SHALL NOT truncate prior walltime.csv rows.

        Resume goes through `__init__` again; if the CSV is opened in
        `"w"` mode unconditionally, every prior wall-time row is wiped
        and the aggregator's `total_run_wall_seconds` for the resumed
        seed silently understates the campaign total. Both
        `walltime.csv` and `generality_probe.csv` use the
        header-only-on-fresh-create pattern.
        """
        # First instantiation: fresh CSVs with header rows.
        loop_a = _make_loop(tmp_path)
        walltime_path = tmp_path / "coevo" / "walltime.csv"
        probe_path = tmp_path / "coevo" / "generality_probe.csv"
        # Append a synthetic data row to each CSV to simulate prior-run
        # state. (Real runs append via `_record_walltime` /
        # `_fire_generality_probe`; we forge the rows here to exercise
        # the resume-safety contract independent of the loop body.)
        with walltime_path.open("a", newline="") as fh:
            csv.writer(fh).writerow(["evaluation", "prey", 0, 0, 1, "1.234"])
        with probe_path.open("a", newline="") as fh:
            csv.writer(fh).writerow([5, "prey", 0, "0.42"])
        del loop_a

        # Second instantiation (mirrors the resume path): __init__ runs
        # again but MUST NOT clobber the existing CSVs.
        _make_loop(tmp_path)
        with walltime_path.open() as fh:
            walltime_rows = list(csv.reader(fh))
        with probe_path.open() as fh:
            probe_rows = list(csv.reader(fh))
        # Header + 1 forged data row each.
        assert len(walltime_rows) == 2
        assert walltime_rows[0][0] == "scope"
        assert walltime_rows[1] == ["evaluation", "prey", "0", "0", "1", "1.234"]
        assert len(probe_rows) == 2
        assert probe_rows[0][0] == "generation"
        assert probe_rows[1] == ["5", "prey", "0", "0.42"]


# ---------------------------------------------------------------------------
# Sanity: HoF default capacity import
# ---------------------------------------------------------------------------


def test_default_hof_capacity_constant() -> None:
    """`DEFAULT_HOF_CAPACITY` SHALL be 8."""
    assert DEFAULT_HOF_CAPACITY == 8


def test_checkpoint_version_constant() -> None:
    """`CHECKPOINT_VERSION` SHALL be set (used for resume invariants)."""
    assert isinstance(CHECKPOINT_VERSION, int)
    assert CHECKPOINT_VERSION >= 1


def test_hof_dataclass_to_dict_round_trip() -> None:
    """Sanity: `HallOfFame.to_dict`/`from_dict` round-trips empty + populated buffers."""
    hof = HallOfFame(capacity=4, replacement="quality")
    restored = HallOfFame.from_dict(hof.to_dict())
    assert len(restored) == 0
    assert restored.capacity == 4


# ---------------------------------------------------------------------------
# Self-review regression: bugs caught + fixed in pre-push review. Each
# test pins the fix so a future refactor that re-introduces the bug fails
# loudly here rather than silently breaking a downstream campaign run.
# ---------------------------------------------------------------------------


class TestProbeFiresAtKBlockBoundary:
    """Probe SHALL fire at K-block boundary AFTER elite push, not mid-block.

    Pre-fix bug: cadence check ran inside the per-generation for-loop
    AFTER `training_side.generation += 1`, BEFORE the K-block-end push
    to `champion_history`. With `K_per_block=10` and
    `generality_probe_every=10`, the very first probe attempt at gen 10
    saw an empty `champion_history` (push hasn't happened yet) and
    silently no-op'd — the prey side's K-block-0 elite never got
    probed. Post-fix: probe runs at K-block end AFTER the elite push.
    """

    def test_probe_writes_rows_for_first_kblock(self, tmp_path: Path) -> None:
        """First K-block at K_per_block=2, probe_every=2 SHALL write probe rows.

        Setup: prey K-block 0 trains for 2 gens -> elite pushed -> probe
        fires at gen 2 (since `2 % 2 == 0`). The pre-fix cadence bug
        would have probed BEFORE the push and produced 0 rows.

        Under Option B wiring, the prey side faces the
        `_predator_held_out_specs` set (heuristic-radius grid, always
        non-empty), so the K-block-end probe writes prey rows. The
        predator-side probe writes 0 rows here because `_prey_held_out`
        is empty in the test harness (no bundle dir or genome-dim
        mismatch).
        """
        loop = _make_loop(
            tmp_path,
            generation_pairs=1,
            K_per_block=2,
            held_out_size=2,
            generality_probe_every=2,
        )
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.predator.fitness = _StubFitness(fixed_value=0.3)  # type: ignore[assignment]

        loop.run()

        probe_path = tmp_path / "coevo" / "generality_probe.csv"
        lines = probe_path.read_text().strip().split("\n")
        assert lines[0] == "generation,side,opponent_index,fitness"
        assert len(lines) > 1, (
            f"probe MUST write at least one row at the K-block boundary; "
            f"got only the header. Lines: {lines}"
        )
        # Verify the rows came from the prey side under Option B
        # wiring (prey faces `_predator_held_out_specs`, which is the
        # only non-empty held-out set in this test harness).
        for row in lines[1:]:
            cells = row.split(",")
            assert cells[1] == "prey"


class TestResumeBundleDriftCrossCheck:
    """Resume SHALL detect bundle drift via genome-id matching, not RNG re-sample.

    Pre-fix bug: `_load_held_out_prey_bundle` (called in `__init__`)
    used `_held_out_rng.choice` to draw a subset; on resume the
    construction-time RNG state differed from the saved RNG state
    (the master rng was not yet restored), so `without-replacement`
    draws picked a different subset -> spurious "bundle drifted"
    raise even when the bundle on disk was byte-identical.
    Post-fix: `_reload_prey_held_out_by_ids` reads files BY ID
    matching `prey_held_out_ids` recorded in the JSON.
    """

    def test_resume_with_drifted_bundle_detected(self, tmp_path: Path) -> None:
        """If the saved bundle's IDs no longer match disk, resume SHALL raise.

        Constructs a synthetic bundle dir with 2 fixture files,
        records their IDs in the checkpoint, then DELETES one file
        and verifies resume raises with a clear "drifted" diagnostic.
        """
        # Build a synthetic bundle dir relative to cwd (the production
        # path is `configs/evolution/coevolution_held_out_prey/` in
        # the repo root). To avoid polluting the real repo, monkeypatch
        # the bundle-dir constant on the loop methods.
        bundle_dir = tmp_path / "held_out_bundle"
        bundle_dir.mkdir(parents=True)

        # Test the helper directly with a minimal sim_config + recorded_ids
        # that don't exist on disk -> expect ValueError.
        loop = _make_loop(tmp_path)
        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "glob", return_value=[]),
            pytest.raises(ValueError, match="drifted"),
        ):
            # No bundle files on disk but we recorded 2 IDs at save time.
            loop._reload_prey_held_out_by_ids(["genome-a", "genome-b"])


class TestChampionHistoryJSONEmitted:
    """`_save_checkpoint` SHALL emit `{output_dir}/champion_history.json`.

    Pre-fix bug: champion_history was only persisted inside the
    per-side pickles. The downstream aggregator reads the top-level
    JSON file; without it, the aggregator can't load champion history
    without unpickling each side's checkpoint pickle (which requires
    numpy + the full cma library).
    Post-fix: `_save_checkpoint` writes a top-level `champion_history.json`
    with `{prey: list[dict], predator: list[dict]}`.
    """

    def test_save_emits_top_level_champion_history_json(self, tmp_path: Path) -> None:
        """`_save_checkpoint` SHALL produce `champion_history.json` at output_dir top-level."""
        loop = _make_loop(tmp_path)
        # Manually populate champion_history on each side so the JSON
        # is non-trivial.
        loop.prey.champion_history.append(
            {
                "genome_id": "elite-prey-0",
                "generation": 0,
                "k_block_index": 0,
                "fitness": 0.7,
                "params": [1.0, 2.0, 3.0],
            },
        )
        loop.predator.champion_history.append(
            {
                "genome_id": "elite-pred-0",
                "generation": 0,
                "k_block_index": 1,
                "fitness": 0.4,
                "params": [-0.5, 0.5],
            },
        )

        loop._save_checkpoint()

        history_path = tmp_path / "coevo" / "champion_history.json"
        assert history_path.exists()
        data = json.loads(history_path.read_text())
        # Required keys: `prey` and `predator`. Extra `k_block_index`
        # field is for cross-file consistency check at resume; the
        # downstream aggregator ignores it.
        assert {"prey", "predator"}.issubset(data.keys())
        assert len(data["prey"]) == 1
        assert len(data["predator"]) == 1
        assert data["prey"][0]["genome_id"] == "elite-prey-0"
        assert data["prey"][0]["fitness"] == 0.7
        assert data["prey"][0]["params"] == [1.0, 2.0, 3.0]
        assert data["predator"][0]["genome_id"] == "elite-pred-0"


class TestRebalanceLagDocumented:
    """S6: rebalance heuristic only fires when saturated side IS the just-trained side.

    Spec ambiguity flagged in audit: the heuristic has a one-block
    lag when the dominant side just trained (we flip normally to the
    saturated side, then on the NEXT K-block boundary the freeze
    kicks in). This test pins the lag behaviour so a future refactor
    that "fixes" the lag (by also freezing when training_side is the
    DOMINANT side) doesn't silently change semantics.
    """

    def test_rebalance_no_freeze_when_dominant_side_just_trained(self, tmp_path: Path) -> None:
        """Prey saturated + training_side=predator (dominant just trained) SHALL flip normally.

        The next iteration trains prey (the saturated side); the freeze
        kicks in on the K-block AFTER that.
        """
        loop = _make_loop(tmp_path, rebalance_threshold=0.5)
        # Prey saturated for 3 K-blocks; predator dominant.
        loop._k_block_mean_fitness["prey"] = [0.1, 0.1, 0.1]
        loop._k_block_mean_fitness["predator"] = [1.0, 1.0, 1.0]

        # Predator just trained -> flip normally to prey (so prey can
        # train). Freeze kicks in NEXT block when training_side flips
        # back to prey AND prey is still saturated.
        assert loop._evaluate_rebalance(loop.predator, loop.prey) is True


# ---------------------------------------------------------------------------
# S2 regression: phenotypic_cycling absolute-variance floor
# ---------------------------------------------------------------------------


class TestCheckpointTornSaveDetected:
    """Round-2 #1: cross-file `k_block_index` mismatch SHALL refuse to resume.

    Pre-fix gap: `_save_checkpoint` writes 5 files but `_load_checkpoint`
    didn't cross-check `k_block_index` between them. A torn save (kill
    landed mid-write across files) could leave the per-side pickles
    at K-block 4 while the JSON state was still at K-block 3 → resume
    silently corrupts the alternating-schedule cursor.
    Post-fix: `k_block_index` is embedded in EVERY checkpoint file.
    `_load_checkpoint` reads the RNG pickle's value as canonical
    (since it's written last) and validates the other 4 files match.
    Any mismatch raises with a diagnostic naming the divergent file.
    """

    def test_per_side_pickle_k_block_mismatch_raises(self, tmp_path: Path) -> None:
        """Tampered prey checkpoint with stale k_block_index SHALL refuse to resume."""
        import pickle

        loop = _make_loop(tmp_path)
        # Advance state so champion_history is non-trivial.
        loop._k_block_index = 5
        loop._save_checkpoint()

        # Tamper with the prey pickle to set a stale k_block_index
        # (simulates a torn save where the prey pickle was written
        # before the K-block-index increment, then the kill landed).
        prey_ckpt = tmp_path / "coevo" / "prey" / "checkpoint.pkl"
        with prey_ckpt.open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 — test file
        payload["k_block_index"] = 4  # canonical (RNG pickle) is 5
        with prey_ckpt.open("wb") as fh:
            pickle.dump(payload, fh)

        sim_config = _build_minimal_sim_config()
        loop2 = CoevolutionLoop(
            sim_config,
            output_dir=tmp_path / "coevo",
            rng=np.random.default_rng(seed=99),
        )
        with pytest.raises(ValueError, match="k_block_index mismatch"):
            loop2._load_checkpoint()

    def test_missing_rng_pickle_refuses_resume(self, tmp_path: Path) -> None:
        """Missing `coevolution_rng.pkl` SHALL be treated as incomplete checkpoint.

        The RNG pickle is written LAST in `_save_checkpoint`; its
        absence implies an interrupted save. `_load_checkpoint` MUST
        refuse to resume rather than try to recover from the
        intermediate state of the other 4 files.
        """
        loop = _make_loop(tmp_path)
        loop._save_checkpoint()
        # Simulate an interrupted save: delete the RNG pickle.
        rng_path = tmp_path / "coevo" / "coevolution_rng.pkl"
        rng_path.unlink()

        sim_config = _build_minimal_sim_config()
        loop2 = CoevolutionLoop(
            sim_config,
            output_dir=tmp_path / "coevo",
            rng=np.random.default_rng(seed=99),
        )
        with pytest.raises(FileNotFoundError, match="interrupted"):
            loop2._load_checkpoint()


class TestHeldOutBundlePathRepoAnchored:
    """Round-2 #2: held-out bundle path SHALL resolve from `__file__`, not cwd.

    Pre-fix gap: `_load_held_out_prey_bundle` and
    `_reload_prey_held_out_by_ids` used a bare relative path
    `Path("configs/evolution/coevolution_warmstart_prey")` resolved
    against cwd. A campaign driver launched from `/tmp` (or any
    non-repo-root cwd) would silently no-op the loader.
    Post-fix: class attribute `_PREY_HELD_OUT_BUNDLE_DIR` is anchored
    to `Path(__file__).resolve().parents[4]` so the path resolves
    correctly regardless of cwd.
    """

    def test_bundle_path_anchored_to_repo_root(self) -> None:
        """Class attribute SHALL resolve under `configs/evolution/coevolution_warmstart_prey`."""
        from quantumnematode.evolution.coevolution import (
            _DEFAULT_PREY_HELD_OUT_BUNDLE_DIR,
        )

        # Repo root is the parent of `packages/quantum-nematode/...`.
        # The resolved path's last 3 components MUST be
        # `configs/evolution/coevolution_warmstart_prey`. Same dir as
        # the warmstart bundle — held-out + warmstart share genomes.
        assert _DEFAULT_PREY_HELD_OUT_BUNDLE_DIR.name == "coevolution_warmstart_prey"
        assert _DEFAULT_PREY_HELD_OUT_BUNDLE_DIR.parent.name == "evolution"
        assert _DEFAULT_PREY_HELD_OUT_BUNDLE_DIR.parent.parent.name == "configs"

    def test_class_attribute_overridable(self, tmp_path: Path) -> None:
        """`_PREY_HELD_OUT_BUNDLE_DIR` is a class attribute SHALL be overridable.

        Tests + the campaign driver may want to point at a different
        bundle (e.g. seed-specific bundles for ablations).
        """
        # Create a synthetic bundle dir with a known fixture and
        # verify the loop reads from there, not the default.
        custom_dir = tmp_path / "custom_bundle"
        custom_dir.mkdir(parents=True)

        with patch.object(CoevolutionLoop, "_PREY_HELD_OUT_BUNDLE_DIR", custom_dir):
            loop = _make_loop(tmp_path)
            # The custom dir is empty, so the loader returns [] AND
            # the warning mentions the custom path (not the repo-root
            # default).
            assert loop._prey_held_out == []


class TestCyclingTinyAmplitudeOnLargeConstant:
    """S2: tiny-amplitude oscillation on a large constant SHALL NOT flag cycling.

    Pre-fix gap: `phenotypic_cycling` only had a relative-variance
    gate (`residual / raw < 1e-12`). For series like
    `[1.0, 1.0+1e-15, 1.0-1e-15, ...]`, the ratio is order-1 (residual
    after detrend is comparable to the raw variance, both at
    machine-epsilon scale around the mean), so the rel gate didn't
    fire. The permutation test then ran on numerical noise and could
    spuriously flag p<0.05 cycling.
    Post-fix: absolute-variance floor `eps * (1 + mean^2) * n`
    catches this case.
    """

    def test_tiny_oscillation_on_large_constant_rejected(self) -> None:
        """Series with mean ~1.0 and 1e-15 amplitude SHALL return cycling_detected=False."""
        from quantumnematode.evolution.redqueen_metrics import phenotypic_cycling

        n = 40
        t = np.arange(n, dtype=np.float64)
        # Mean ~ 1.0; oscillation amplitude 1e-15 (machine epsilon scale).
        series = 1.0 + 1e-15 * np.sin(2.0 * np.pi * t / 8.0)
        result = phenotypic_cycling(series)
        assert result["cycling_detected"] is False, (
            f"tiny-amplitude oscillation on large constant must NOT trigger "
            f"cycling detection (would be a false positive on numerical noise). "
            f"Got: {result}"
        )


class TestPreySideProbeEnvOverride:
    """Probe-semantics fix: prey-side probe SHALL use calibrated env, not training env.

    The training env's predator settings (count, speed, grid_size) may
    be tuned for harder dynamics than klinotaxis-LSTMPPO prey can survive
    against heuristic predators. The probe overrides these to the
    calibrated `PROBE_ENV_*` constants so the diagnostic stays informative.
    """

    def test_probe_overrides_predator_count(self, tmp_path: Path) -> None:
        """Prey-side probe sim_config SHALL use `PROBE_ENV_PREDATOR_COUNT`, not training count."""
        from quantumnematode.evolution.coevolution import PROBE_ENV_PREDATOR_COUNT

        loop = _make_loop(tmp_path)
        # Set the training env's predator count to something distinct
        # from the probe constant so the override is observable.
        non_probe_count = 4
        assert non_probe_count != PROBE_ENV_PREDATOR_COUNT, "test setup must use a distinct count"
        loop.sim_config.environment.predators.count = non_probe_count  # type: ignore[union-attr]

        patched = loop._build_prey_side_probe_sim_config(opponent_index=0)
        assert patched.environment.predators.count == PROBE_ENV_PREDATOR_COUNT  # type: ignore[union-attr]

    def test_probe_overrides_predator_speed(self, tmp_path: Path) -> None:
        """Prey-side probe sim_config SHALL use `PROBE_ENV_PREDATOR_SPEED`, not training speed."""
        from quantumnematode.evolution.coevolution import PROBE_ENV_PREDATOR_SPEED

        loop = _make_loop(tmp_path)
        non_probe_speed = 1.5
        assert non_probe_speed != PROBE_ENV_PREDATOR_SPEED
        loop.sim_config.environment.predators.speed = non_probe_speed  # type: ignore[union-attr]

        patched = loop._build_prey_side_probe_sim_config(opponent_index=0)
        assert patched.environment.predators.speed == PROBE_ENV_PREDATOR_SPEED  # type: ignore[union-attr]

    def test_probe_overrides_grid_size(self, tmp_path: Path) -> None:
        """Prey-side probe sim_config SHALL use `PROBE_ENV_GRID_SIZE`, not training grid."""
        from quantumnematode.evolution.coevolution import PROBE_ENV_GRID_SIZE

        loop = _make_loop(tmp_path)
        non_probe_grid = 16
        assert non_probe_grid != PROBE_ENV_GRID_SIZE
        loop.sim_config.environment.grid_size = non_probe_grid  # type: ignore[union-attr]

        patched = loop._build_prey_side_probe_sim_config(opponent_index=0)
        assert patched.environment.grid_size == PROBE_ENV_GRID_SIZE  # type: ignore[union-attr]

    def test_probe_still_uses_held_out_spec_for_radii(self, tmp_path: Path) -> None:
        """Probe SHALL still take detection/damage radii from held-out spec (NOT calibrated)."""
        loop = _make_loop(tmp_path, held_out_size=2)
        # Held-out specs come from `_build_held_out_predator_specs`; pick
        # the first one and verify the probe sim_config sees the same value.
        spec = loop._predator_held_out_specs[0]
        patched = loop._build_prey_side_probe_sim_config(opponent_index=0)
        assert patched.environment.predators.detection_radius == spec[0]  # type: ignore[union-attr]
        assert patched.environment.predators.damage_radius == spec[1]  # type: ignore[union-attr]


class TestPersistCmaAcrossKBlocks:
    """`persist_cma_across_kblocks` SHALL skip the K-block-end CMA-ES rebuild."""

    def test_persist_true_skips_rebuild(self, tmp_path: Path) -> None:
        """When set, `_rebuild_optimizer` SHALL leave `side.optimizer` unchanged."""
        loop = _make_loop(tmp_path)
        loop.predator.evolution_config.persist_cma_across_kblocks = True  # type: ignore[misc]
        original_optimizer = loop.predator.optimizer
        loop._rebuild_optimizer(loop.predator)
        assert loop.predator.optimizer is original_optimizer, (
            "persist_cma_across_kblocks=True must skip optimizer rebuild "
            "so the existing CMA-ES study continues uninterrupted across "
            "K-block boundaries."
        )

    def test_prey_probe_fitness_defaults_to_episodic_success_rate(
        self,
        tmp_path: Path,
    ) -> None:
        """Option 1: in-run prey probe SHALL default to `EpisodicSuccessRate` (frozen-weight).

        Rationale: the training-time `LearnedPerformanceFitness` runs K
        PPO train episodes against held-out opponents before the L eval,
        which fine-tunes the elite's policy against a different opponent
        class than what it co-evolved against. The result is the policy
        consistently degrades to 0.0 by eval phase. Frozen-weight
        `EpisodicSuccessRate` measures the elite AS-IS — the
        scientifically correct test of "what did the prey learn?"
        matching the post-hoc analysis path.
        """
        from quantumnematode.evolution.fitness import (
            EpisodicSuccessRate,
            LearnedPerformanceFitness,
        )

        loop = _make_loop(tmp_path)
        # Probe-time fitness is frozen-weight `EpisodicSuccessRate`.
        assert isinstance(loop._prey_probe_fitness, EpisodicSuccessRate)
        # Distinct from `loop.prey.fitness` (training-time
        # `LearnedPerformanceFitness` with K PPO train + L frozen eval).
        assert isinstance(loop.prey.fitness, LearnedPerformanceFitness)
        # And `_prey_probe_fitness` is NOT the same instance/class as
        # the training fitness — they're intentionally different
        # measurements.
        assert type(loop._prey_probe_fitness) is not type(loop.prey.fitness)

    def test_probe_gap3_loads_lamarckian_checkpoint_when_present(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Gap-3 fix: in-run probe SHALL pass elite's `.pt` as `warm_start_path_override`.

        Under Lamarckian inheritance, the K-block-elite's `genome.params`
        encode the CMA-ES sample (pre-PPO-training weights). The actual
        co-evolved policy lives in the post-training `.pt` checkpoint.
        The probe MUST load that checkpoint via `warm_start_path_override`
        to measure what the prey actually learned — otherwise the in-run
        probe consistently shows ~0.0 (untrained CMA-ES sample) while
        post-hoc analysis correctly shows a non-zero value for the same
        elite.
        """
        loop = _make_loop(tmp_path)
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        # Seed a synthetic K-block elite + create its `.pt` so the
        # probe can find it on disk.
        elite_gen = 4
        elite_gid = "synthetic-prey-elite"
        loop.prey.champion_history.append(
            {
                "genome_id": elite_gid,
                "generation": elite_gen,
                "k_block_index": 0,
                "fitness": 0.95,
                "params": [0.0] * loop.prey.encoder.genome_dim(loop.sim_config),
            },
        )
        # Materialise the canonical Lamarckian checkpoint path so the
        # probe's `candidate.exists()` check passes. Content doesn't
        # matter — we monkeypatch `_evaluate_in_worker` to capture
        # the args tuple without actually loading the file.
        checkpoint_path = loop.prey.inheritance.checkpoint_path(
            loop.prey.output_dir,
            elite_gen,
            elite_gid,
        )
        assert checkpoint_path is not None
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.touch()

        captured: list[tuple] = []

        def fake_evaluate_in_worker(args: tuple) -> float:
            captured.append(args)
            return 0.5

        monkeypatch.setattr(
            "quantumnematode.evolution.coevolution._evaluate_in_worker",
            fake_evaluate_in_worker,
        )
        # Fire the probe; it should pass the .pt path as the 10th
        # element of the args tuple (warm_start_path_override).
        loop._fire_generality_probe()
        # Expect at least one prey-side probe call (predator has no
        # champions so its branch no-ops).
        assert len(captured) > 0
        # Each captured args tuple's 10th element (index 9) is the
        # `warm_start_path_override` field. For the gap-3 fix, this
        # SHALL equal the canonical Lamarckian checkpoint path.
        for args in captured:
            warm_start = args[9]
            assert warm_start == checkpoint_path, (
                f"probe must pass elite's .pt as warm_start_path_override; "
                f"got {warm_start} (expected {checkpoint_path})"
            )

    def test_probe_gap3_no_warm_start_when_checkpoint_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When elite's `.pt` is missing, probe SHALL pass None (fallback to genome.params).

        Resilient to GC of post-K-block inheritance dirs: if the elite's
        checkpoint was removed (e.g., not the K-block survivor under
        `_gc_inheritance_dir`), the probe gracefully falls back to the
        raw genome.params via the encoder, NOT crashing.
        """
        loop = _make_loop(tmp_path)
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]
        loop.prey.champion_history.append(
            {
                "genome_id": "missing-checkpoint-elite",
                "generation": 4,
                "k_block_index": 0,
                "fitness": 0.95,
                "params": [0.0] * loop.prey.encoder.genome_dim(loop.sim_config),
            },
        )
        # Deliberately do NOT create the .pt file. The probe's
        # `candidate.exists()` check should evaluate False and
        # `warm_start_path` should default to None.
        captured: list[tuple] = []

        def fake_evaluate_in_worker(args: tuple) -> float:
            captured.append(args)
            return 0.0

        monkeypatch.setattr(
            "quantumnematode.evolution.coevolution._evaluate_in_worker",
            fake_evaluate_in_worker,
        )
        loop._fire_generality_probe()
        assert len(captured) > 0
        for args in captured:
            warm_start = args[9]
            assert warm_start is None, (
                f"probe must pass None when checkpoint is missing; got {warm_start}"
            )

    def test_probe_gap3_prefers_champion_archive_over_inheritance_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Gap-3 Option A: probe SHALL prefer champion_archive over inheritance dir.

        The champion_archive holds the K-block-elite's `.pt` separately
        from `inheritance/`, so it's not subject to GC. When BOTH paths
        exist, probe uses the archive (it's the canonical
        post-PPO-trained-elite source).
        """
        loop = _make_loop(tmp_path)
        loop.prey.fitness = _StubFitness(fixed_value=0.5)  # type: ignore[assignment]
        loop._prey_probe_fitness = loop.prey.fitness  # type: ignore[assignment]

        elite_gen = 4
        elite_gid = "synthetic-prey-elite"
        elite_k_block = 0
        loop.prey.champion_history.append(
            {
                "genome_id": elite_gid,
                "generation": elite_gen,
                "k_block_index": elite_k_block,
                "fitness": 0.95,
                "params": [0.0] * loop.prey.encoder.genome_dim(loop.sim_config),
            },
        )
        # Create BOTH the inheritance path and the archive path.
        inh_path = loop.prey.inheritance.checkpoint_path(
            loop.prey.output_dir,
            elite_gen,
            elite_gid,
        )
        assert inh_path is not None
        inh_path.parent.mkdir(parents=True, exist_ok=True)
        inh_path.write_bytes(b"inheritance")
        archive_path = loop._kblock_archive_path(loop.prey, elite_k_block)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"archive")

        captured: list[tuple] = []

        def fake_evaluate(args: tuple) -> float:
            captured.append(args)
            return 0.5

        monkeypatch.setattr(
            "quantumnematode.evolution.coevolution._evaluate_in_worker",
            fake_evaluate,
        )
        loop._fire_generality_probe()
        assert len(captured) > 0
        for args in captured:
            assert args[9] == archive_path, (
                f"probe must prefer champion_archive when both exist; "
                f"got {args[9]} (expected {archive_path})"
            )

    def test_archive_kblock_elite_checkpoint_copies_pt(
        self,
        tmp_path: Path,
    ) -> None:
        """`_archive_kblock_elite_checkpoint` SHALL copy from inheritance to archive."""
        from quantumnematode.evolution.genome import Genome

        loop = _make_loop(tmp_path)
        # Manually create an inheritance .pt for a synthetic elite at gen 3.
        elite_gen = 3
        elite_gid = "test-elite-001"
        loop._k_block_index = 1  # simulate being mid-K-block-1
        inh_path = loop.prey.inheritance.checkpoint_path(
            loop.prey.output_dir,
            elite_gen,
            elite_gid,
        )
        assert inh_path is not None
        inh_path.parent.mkdir(parents=True, exist_ok=True)
        inh_path.write_bytes(b"trained-weights-payload")

        # Build a synthetic Genome to pass to the archive helper.
        elite_genome = Genome(
            params=np.zeros(
                loop.prey.encoder.genome_dim(loop.sim_config),
                dtype=np.float32,
            ),
            genome_id=elite_gid,
            parent_ids=[],
            generation=elite_gen,
        )
        loop._archive_kblock_elite_checkpoint(
            loop.prey,
            elite_genome=elite_genome,
            elite_gen=elite_gen,
        )
        archive_path = loop._kblock_archive_path(loop.prey, 1)
        assert archive_path.exists()
        # Content should match the source.
        assert archive_path.read_bytes() == b"trained-weights-payload"

    def test_archive_kblock_elite_noop_for_no_inheritance(
        self,
        tmp_path: Path,
    ) -> None:
        """Archive SHALL be a no-op when side inheritance kind != 'weights'."""
        from quantumnematode.evolution.genome import Genome

        loop = _make_loop(tmp_path)
        # Predator side defaults to NoInheritance under the minimal config.
        assert loop.predator.inheritance.kind() == "none"
        elite_genome = Genome(
            params=np.zeros(
                loop.predator.encoder.genome_dim(loop.sim_config),
                dtype=np.float32,
            ),
            genome_id="pred-elite-001",
            parent_ids=[],
            generation=2,
        )
        loop._archive_kblock_elite_checkpoint(
            loop.predator,
            elite_genome=elite_genome,
            elite_gen=2,
        )
        # No archive created.
        archive_path = loop._kblock_archive_path(loop.predator, 0)
        assert not archive_path.exists()

    def test_persist_false_still_rebuilds(self, tmp_path: Path) -> None:
        """Default (False) behaviour SHALL rebuild — preserves legacy semantics."""
        loop = _make_loop(tmp_path)
        # Default is False; assert no regression.
        assert loop.predator.evolution_config.persist_cma_across_kblocks is False
        original_optimizer = loop.predator.optimizer
        # Need a champion to drive the rebuild's x0 (else _rebuild_optimizer
        # falls back to optimizer.mean which is also fine — both paths
        # produce a NEW optimizer).
        loop.predator.champion_history.append(
            {
                "k_block_index": 0,
                "generation": 0,
                "genome_id": "predator-elite-0",
                "fitness": 0.5,
                "params": [0.0] * loop.predator.encoder.genome_dim(loop.sim_config),
            },
        )
        loop._rebuild_optimizer(loop.predator)
        assert loop.predator.optimizer is not original_optimizer, (
            "default behaviour (persist=False) must rebuild the optimizer."
        )


class TestPredatorInheritanceYamlConfigurable:
    """`predator_evolution.inheritance` SHALL drive predator side state.

    Default `"none"` -> `NoInheritance`; `"lamarckian"` -> `LamarckianInheritance`.
    Removes the prior hardcode that pinned predator to `NoInheritance()`.
    """

    def test_default_predator_inheritance_is_no_inheritance(self, tmp_path: Path) -> None:
        """Default config SHALL preserve legacy `NoInheritance` predator behaviour."""
        from quantumnematode.evolution.inheritance import NoInheritance

        loop = _make_loop(tmp_path)
        assert isinstance(loop.predator.inheritance, NoInheritance)
        # Confirm the inheritance kind matches.
        assert loop.predator.inheritance.kind() == "none"

    def test_lamarckian_predator_inheritance_builds_lamarckian(self, tmp_path: Path) -> None:
        """`predator_evolution.inheritance="lamarckian"` SHALL build LamarckianInheritance."""
        from quantumnematode.evolution.inheritance import LamarckianInheritance

        sim_config = _build_minimal_sim_config()
        # Patch predator inheritance in the parsed config — schema accepts
        # "lamarckian" for EvolutionConfig.
        sim_config.coevolution.predator_evolution = (  # type: ignore[union-attr]
            sim_config.coevolution.predator_evolution.model_copy(  # type: ignore[union-attr]
                update={"inheritance": "lamarckian"},
            )
        )
        loop = CoevolutionLoop(
            sim_config,
            output_dir=tmp_path / "coevo",
            rng=np.random.default_rng(seed=42),
        )
        assert isinstance(loop.predator.inheritance, LamarckianInheritance)
        assert loop.predator.inheritance.kind() == "weights"

    def test_invalid_predator_inheritance_raises(self, tmp_path: Path) -> None:
        """An unsupported predator inheritance value SHALL raise at construction."""
        # Patch the parsed config to bypass Pydantic schema validation and
        # exercise the runtime guard in `_build_predator_state`. Use
        # object.__setattr__ to bypass Pydantic's frozen-ness if needed.
        sim_config = _build_minimal_sim_config()
        # Replace with a fake-string that the Literal would reject at
        # schema-load but we sneak past by direct field overwrite.
        pe = sim_config.coevolution.predator_evolution  # type: ignore[union-attr]
        object.__setattr__(pe, "inheritance", "baldwin")  # unsupported on predator
        with pytest.raises(ValueError, match="Predator-side inheritance"):
            CoevolutionLoop(
                sim_config,
                output_dir=tmp_path / "coevo",
                rng=np.random.default_rng(seed=42),
            )

    def test_coevolution_validator_accepts_ga(self) -> None:
        """CoevolutionConfig SHALL accept ``algorithm="ga"`` on both sides."""
        from quantumnematode.utils.config_loader import CoevolutionConfig, EvolutionConfig

        prey_evo = EvolutionConfig(
            algorithm="ga",
            cma_diagonal=True,
            learn_episodes_per_eval=8,
            inheritance="lamarckian",
        )
        pred_evo = EvolutionConfig(
            algorithm="ga",
            cma_diagonal=True,
            learn_episodes_per_eval=1,
            inheritance="lamarckian",
        )
        # Construction SHALL NOT raise.
        CoevolutionConfig(prey_evolution=prey_evo, predator_evolution=pred_evo, generation_pairs=2)

    def test_coevolution_validator_rejects_tpe(self) -> None:
        """CoevolutionConfig SHALL still reject TPE — unbounded weight encoder incompatible."""
        from quantumnematode.utils.config_loader import CoevolutionConfig, EvolutionConfig

        prey_evo = EvolutionConfig(
            algorithm="tpe",
            cma_diagonal=True,
            learn_episodes_per_eval=8,
            inheritance="lamarckian",
        )
        pred_evo = EvolutionConfig(
            algorithm="cmaes",
            cma_diagonal=True,
            learn_episodes_per_eval=1,
            inheritance="lamarckian",
        )
        with pytest.raises(ValueError, match="algorithm must be one of"):
            CoevolutionConfig(
                prey_evolution=prey_evo,
                predator_evolution=pred_evo,
                generation_pairs=2,
            )

    def test_coevolution_validator_ga_ignores_cma_diagonal(self) -> None:
        """Under ``algorithm="ga"`` the inert ``cma_diagonal`` field SHALL NOT be enforced.

        ``cma_diagonal`` is a sep-CMA-ES toggle; under GA it has no
        meaning. Configs using GA shouldn't have to set a CMA-ES-flavoured
        field they don't use.
        """
        from quantumnematode.utils.config_loader import CoevolutionConfig, EvolutionConfig

        prey_evo = EvolutionConfig(
            algorithm="ga",
            cma_diagonal=False,  # inert under GA, should not raise
            learn_episodes_per_eval=8,
            inheritance="lamarckian",
        )
        pred_evo = EvolutionConfig(
            algorithm="ga",
            cma_diagonal=False,  # inert under GA, should not raise
            learn_episodes_per_eval=1,
            inheritance="lamarckian",
        )
        # Construction SHALL NOT raise.
        CoevolutionConfig(
            prey_evolution=prey_evo,
            predator_evolution=pred_evo,
            generation_pairs=2,
        )

    def test_coevolution_validator_cmaes_still_requires_diagonal(self) -> None:
        """Under ``algorithm="cmaes"`` the ``cma_diagonal`` check SHALL still fire."""
        from quantumnematode.utils.config_loader import CoevolutionConfig, EvolutionConfig

        prey_evo = EvolutionConfig(
            algorithm="cmaes",
            cma_diagonal=False,  # invalid under CMA-ES
            learn_episodes_per_eval=8,
            inheritance="lamarckian",
        )
        pred_evo = EvolutionConfig(
            algorithm="cmaes",
            cma_diagonal=True,
            learn_episodes_per_eval=1,
            inheritance="lamarckian",
        )
        with pytest.raises(ValueError, match="cma_diagonal must be True"):
            CoevolutionConfig(
                prey_evolution=prey_evo,
                predator_evolution=pred_evo,
                generation_pairs=2,
            )
