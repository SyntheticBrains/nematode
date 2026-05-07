"""Tests for :mod:`quantumnematode.evolution.predator_encoders` (task 3.5).

Covers spec scenarios from `evolution-framework/spec.md`:

- Registry surface: `PREDATOR_ENCODER_REGISTRY` exposes the
  `mlpppo_predator` entry; lookup-only (no agent-side fallback).
- `MLPPPOPredatorEncoder` overrides `initial_genome` / `decode` /
  `genome_dim` to call the predator factory; `decode` returns
  `MLPPPOPredatorBrain` (NOT an agent brain).
- Round-trip: weights flatten → unflatten match action behaviour on a
  fixed test set.
- Initial genome reproducibility under fixed seed.
- Genome dim matches the brain's `WeightPersistence` parameter count.
- Decode works under `CMAESOptimizer(diagonal=True)` flat-vector
  samples — verifies that unbounded weight encoding + sep-CMA-ES
  produces a valid genome (no bounds rejection).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from quantumnematode.env.env import PredatorType
from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain
from quantumnematode.env.predator_brain import PredatorBrainParams
from quantumnematode.evolution._predator_brain_factory import (
    instantiate_predator_brain_from_sim_config,
)
from quantumnematode.evolution.encoders import GenomeEncoder
from quantumnematode.evolution.predator_encoders import (
    PREDATOR_ENCODER_REGISTRY,
    MLPPPOPredatorEncoder,
    get_predator_encoder,
)
from quantumnematode.utils.config_loader import (
    EnvironmentConfig,
    PredatorBrainConfigSchema,
    PredatorConfig,
    SimulationConfig,
)


def _make_sim_config() -> SimulationConfig:
    """Minimal sim_config carrying an `mlpppo_predator` brain block.

    Predator is the only configured side; agent block is intentionally
    None — predator encoder must not depend on agent config (different
    sub-tree than the agent encoder reads).
    """
    return SimulationConfig(
        environment=EnvironmentConfig(
            grid_size=20,
            predators=PredatorConfig(
                enabled=True,
                count=1,
                brain_config=PredatorBrainConfigSchema(kind="mlpppo_predator"),
            ),
        ),
    )


def _make_test_params(predator_pos: tuple[int, int] = (5, 5)) -> PredatorBrainParams:
    """Synthesise a `PredatorBrainParams` for action-equivalence checks."""
    return PredatorBrainParams(
        predator_id="test",
        predator_position=predator_pos,
        predator_type=PredatorType.PURSUIT,
        detection_radius=8,
        damage_radius=1,
        agent_positions=((10, 10),),
        chase_target=(10, 10),
        is_pursuing=True,
        grid_size=20,
        rng=np.random.default_rng(seed=0),
        step_index=0,
    )


class TestRegistry:
    """Spec scenario "Registry Surface"."""

    def test_registry_includes_mlpppo_predator(self) -> None:
        """The registry SHALL contain the `mlpppo_predator -> MLPPPOPredatorEncoder` entry."""
        assert "mlpppo_predator" in PREDATOR_ENCODER_REGISTRY
        assert PREDATOR_ENCODER_REGISTRY["mlpppo_predator"] is MLPPPOPredatorEncoder

    def test_registry_lookup_only_no_agent_fallback(self) -> None:
        """Looking up an agent brain name SHALL raise ValueError (no fallback)."""
        with pytest.raises(ValueError, match="No predator encoder"):
            get_predator_encoder("mlpppo")  # agent-side brain name

    def test_registry_unknown_kind_raises(self) -> None:
        """Looking up an unknown predator kind SHALL raise ValueError."""
        with pytest.raises(ValueError, match="No predator encoder"):
            get_predator_encoder("nonexistent_predator_kind")


class TestPredatorBrainFactoryErrors:
    """Spec scenario "Predator Brain Factory Surface" — direct error-path coverage.

    The encoder tests below exercise the factory's success path indirectly.
    These tests pin the three explicit `raise ValueError` arms in
    :func:`instantiate_predator_brain_from_sim_config` so a future refactor
    that swallows or reorders them fails loudly.
    """

    def test_missing_environment_raises(self) -> None:
        """`sim_config.environment is None` SHALL raise `ValueError`."""
        sim_config = SimulationConfig()  # environment defaults to None
        with pytest.raises(ValueError, match="environment"):
            instantiate_predator_brain_from_sim_config(sim_config)

    def test_missing_predator_brain_config_raises(self) -> None:
        """`environment.predators is None` (or `brain_config is None`) SHALL raise `ValueError`."""
        # No `predators` block at all.
        sim_config_no_predators = SimulationConfig(
            environment=EnvironmentConfig(grid_size=20),
        )
        with pytest.raises(ValueError, match="brain_config"):
            instantiate_predator_brain_from_sim_config(sim_config_no_predators)

        # `predators` block present but `brain_config` omitted (the default).
        sim_config_no_brain = SimulationConfig(
            environment=EnvironmentConfig(
                grid_size=20,
                predators=PredatorConfig(enabled=True, count=1),
            ),
        )
        with pytest.raises(ValueError, match="brain_config"):
            instantiate_predator_brain_from_sim_config(sim_config_no_brain)

    def test_unsupported_kind_raises(self) -> None:
        """`kind` other than `"mlpppo_predator"` SHALL raise `ValueError`.

        The heuristic kind has no encoder counterpart — there is nothing
        to evolve. The factory is an evolution-side helper; the env-side
        dispatcher (`_build_predator_brain`) is what handles `"heuristic"`
        for the env path.
        """
        sim_config = SimulationConfig(
            environment=EnvironmentConfig(
                grid_size=20,
                predators=PredatorConfig(
                    enabled=True,
                    count=1,
                    brain_config=PredatorBrainConfigSchema(kind="heuristic"),
                ),
            ),
        )
        with pytest.raises(ValueError, match="mlpppo_predator"):
            instantiate_predator_brain_from_sim_config(sim_config)


class TestEncoderProtocolConformance:
    """Spec scenario "MLPPPOPredatorEncoder Brain Factory Override"."""

    def test_encoder_satisfies_genome_encoder_protocol(self) -> None:
        """`MLPPPOPredatorEncoder` SHALL satisfy the runtime-checkable Protocol."""
        encoder = MLPPPOPredatorEncoder()
        assert isinstance(encoder, GenomeEncoder)

    def test_brain_name_is_pinned(self) -> None:
        """`brain_name` SHALL be `"mlpppo_predator"`."""
        encoder = MLPPPOPredatorEncoder()
        assert encoder.brain_name == "mlpppo_predator"

    def test_decode_returns_mlpppo_predator_brain(self) -> None:
        """`decode` SHALL return an `MLPPPOPredatorBrain`, NOT an agent brain."""
        sim_config = _make_sim_config()
        encoder = MLPPPOPredatorEncoder()
        rng = np.random.default_rng(seed=42)
        genome = encoder.initial_genome(sim_config, rng=rng)
        brain = encoder.decode(genome, sim_config, seed=42)
        assert isinstance(brain, MLPPPOPredatorBrain)


class TestRoundTrip:
    """Spec scenario "MLPPPOPredatorEncoder Round-Trip"."""

    def test_decoded_brain_actions_match_after_round_trip(self) -> None:
        """Decoded weights SHALL produce identical actions on a fixed test set."""
        sim_config = _make_sim_config()
        encoder = MLPPPOPredatorEncoder()
        rng = np.random.default_rng(seed=42)
        genome = encoder.initial_genome(sim_config, rng=rng)

        # Two independent decodes from the same genome — different
        # post-decode init seeds, so any leak from the post-load init
        # values into action behaviour would surface as a mismatch.
        brain_a = encoder.decode(genome, sim_config, seed=42)
        brain_b = encoder.decode(genome, sim_config, seed=999)

        # Fixed test set spanning a few predator positions so a single
        # accidentally-correct argmax doesn't pass for the wrong reason.
        for predator_pos in [(5, 5), (10, 12), (1, 18), (15, 3), (8, 8)]:
            params = _make_test_params(predator_pos=predator_pos)
            assert brain_a.run_brain(params) == brain_b.run_brain(params), (
                f"action mismatch at predator_pos={predator_pos}"
            )

    def test_genome_dim_matches_weight_persistence_count(self) -> None:
        """`genome_dim` SHALL match the brain's flattened weight count."""
        sim_config = _make_sim_config()
        encoder = MLPPPOPredatorEncoder()
        rng = np.random.default_rng(seed=42)
        genome = encoder.initial_genome(sim_config, rng=rng)
        # Both must agree — the flatten produces exactly `genome_dim` floats.
        assert genome.params.size == encoder.genome_dim(sim_config)

    def test_decode_works_with_cmaes_diagonal_flat_sample(self) -> None:
        """Decode SHALL accept a flat genome sampled by `CMAESOptimizer(diagonal=True)`.

        The optimiser produces unbounded float vectors with no
        `birth_metadata.shape_map` (the optimiser has no template to
        preserve). The encoder's `decode` SHALL re-derive the shape_map
        from a fresh template.
        """
        sim_config = _make_sim_config()
        encoder = MLPPPOPredatorEncoder()
        rng = np.random.default_rng(seed=42)
        # Reference initial genome to learn the dim.
        ref = encoder.initial_genome(sim_config, rng=rng)
        # Synthesise an "optimiser-sampled" genome with empty
        # birth_metadata, mimicking what `CMAESOptimizer.sample()`
        # produces (a flat float vector wrapped in a Genome with no
        # extra metadata).
        from quantumnematode.evolution.genome import Genome

        flat_sample = rng.normal(0.0, 1.0, size=ref.params.size).astype(np.float32)
        sampled_genome = Genome(
            params=flat_sample,
            genome_id="cmaes_sample",
            parent_ids=[],
            generation=0,
            birth_metadata={},  # explicitly empty — matches optimiser output
        )
        # SHALL NOT raise; the encoder re-derives shape_map from a
        # fresh template via the parent's fallback path.
        brain = encoder.decode(sampled_genome, sim_config, seed=42)
        assert isinstance(brain, MLPPPOPredatorBrain)


class TestInitialGenomeReproducibility:
    """Spec scenario "Initial Genome Reproducibility"."""

    def test_initial_genome_reproducible_under_fixed_seed(self) -> None:
        """Two `initial_genome` calls with same seed SHALL return identical params.

        Reproducibility comes from torch's global generator (the
        predator brain's orthogonal-init draws from torch globals),
        which the brain seeds via its `seed=` kwarg. The encoder's
        factory wraps that — same factory seed → same orthogonal init
        → same flattened genome.
        """
        sim_config = _make_sim_config()
        encoder = MLPPPOPredatorEncoder()

        # Pre-seed torch globals so two `initial_genome` calls run from
        # the same starting state. (The encoder doesn't pass a seed
        # through `initial_genome` — that's by design; the parent's
        # `_ClassicalPPOEncoder.initial_genome` similarly doesn't seed
        # the brain factory. Reproducibility here means
        # "deterministic given identical pre-call torch state", which
        # is what loop callers ensure via `set_global_seed`.)
        torch.manual_seed(42)
        g_a = encoder.initial_genome(sim_config, rng=np.random.default_rng(0))
        torch.manual_seed(42)
        g_b = encoder.initial_genome(sim_config, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(g_a.params, g_b.params)


class TestEncoderUnboundedness:
    """`genome_stds` and `genome_bounds` inherit None — appropriate for sep-CMA-ES."""

    def test_genome_stds_returns_none(self) -> None:
        """Weight encoders use uniform sigma (None signals optimiser default)."""
        sim_config = _make_sim_config()
        encoder = MLPPPOPredatorEncoder()
        assert encoder.genome_stds(sim_config) is None

    def test_genome_bounds_returns_none(self) -> None:
        """Weight encoders are unbounded (None signals no schema bounds)."""
        sim_config = _make_sim_config()
        encoder = MLPPPOPredatorEncoder()
        assert encoder.genome_bounds(sim_config) is None
