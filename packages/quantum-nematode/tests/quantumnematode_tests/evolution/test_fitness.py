"""Tests for :mod:`quantumnematode.evolution.fitness`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain
from quantumnematode.evolution.encoders import MLPPPOEncoder
from quantumnematode.evolution.fitness import EpisodicSuccessRate, FrozenEvalRunner
from quantumnematode.report.dtypes import TerminationReason
from quantumnematode.utils.config_loader import SimulationConfig, load_simulation_config

if TYPE_CHECKING:
    from quantumnematode.evolution.genome import Genome

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MLPPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/mlpppo_small_oracle.yml"


# ---------------------------------------------------------------------------
# EpisodicSuccessRate basic contract
# ---------------------------------------------------------------------------


def _make_genome_for(
    config_path: Path,
) -> tuple[SimulationConfig, MLPPPOEncoder, Genome]:
    """Build ``(sim_config, encoder, genome)`` from a YAML config path."""
    sim_config = load_simulation_config(str(config_path))
    encoder = MLPPPOEncoder()
    genome = encoder.initial_genome(sim_config, rng=np.random.default_rng(0))
    return sim_config, encoder, genome


def test_episodic_success_rate_returns_float_in_unit_interval() -> None:
    """Fitness SHALL be a finite float in [0.0, 1.0] for an arbitrary genome."""
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    fitness = EpisodicSuccessRate()
    score = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=42)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_episodic_success_rate_deterministic_for_seeded_genome() -> None:
    """Same genome + same seed SHALL produce byte-identical fitness."""
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    fitness = EpisodicSuccessRate()
    a = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=42)
    b = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=42)
    assert a == b


def test_episodic_success_rate_rejects_zero_episodes() -> None:
    """Zero or negative episodes SHALL raise ValueError."""
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    fitness = EpisodicSuccessRate()
    with pytest.raises(ValueError, match="episodes must be positive"):
        fitness.evaluate(genome, sim_config, encoder, episodes=0, seed=42)
    with pytest.raises(ValueError, match="episodes must be positive"):
        fitness.evaluate(genome, sim_config, encoder, episodes=-1, seed=42)


# ---------------------------------------------------------------------------
# Frozen-weight contract
# ---------------------------------------------------------------------------


def test_frozen_eval_runner_never_calls_learn() -> None:
    """``FrozenEvalRunner`` SHALL NOT call ``brain.learn`` during episodes.

    Patches ``MLPPPOBrain.learn`` and ``MLPPPOBrain.update_memory`` and runs
    a short fitness eval; both methods SHALL be untouched.
    """
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    fitness = EpisodicSuccessRate()

    with (
        patch.object(MLPPPOBrain, "learn", autospec=True) as mock_learn,
        patch.object(MLPPPOBrain, "update_memory", autospec=True) as mock_um,
    ):
        fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=42)

    assert mock_learn.call_count == 0
    assert mock_um.call_count == 0


def test_episodic_success_rate_uses_termination_reason_for_success() -> None:
    """Success SHALL be defined as ``COMPLETED_ALL_FOOD`` termination.

    Patch ``StandardEpisodeRunner.run`` to return controlled termination
    reasons; verify the fitness function counts only ``COMPLETED_ALL_FOOD``
    as success.
    """
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    fitness = EpisodicSuccessRate()

    from quantumnematode.agent.runners import EpisodeResult

    completed = EpisodeResult(
        agent_path=[(0, 0)],
        termination_reason=TerminationReason.COMPLETED_ALL_FOOD,
        food_history=None,
    )
    starved = EpisodeResult(
        agent_path=[(0, 0)],
        termination_reason=TerminationReason.STARVED,
        food_history=None,
    )

    # 2 successes, 1 failure → 2/3 fitness
    sequence = [completed, starved, completed]
    with patch.object(FrozenEvalRunner, "run", side_effect=sequence):
        score = fitness.evaluate(genome, sim_config, encoder, episodes=3, seed=42)

    assert score == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# Seed propagation
# ---------------------------------------------------------------------------


def test_evaluate_passes_seed_to_encoder_decode() -> None:
    """``evaluate(seed=X)`` SHALL forward ``X`` to BOTH the encoder and the env factory.

    Specifically, ``encoder.decode`` and ``create_env_from_config`` MUST both
    receive the same ``seed`` value the fitness function was invoked with.
    """
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    fitness = EpisodicSuccessRate()

    captured_decode: dict[str, int | None] = {}
    captured_env: dict[str, int | None] = {}
    real_decode = encoder.decode
    from quantumnematode.evolution import fitness as fitness_module

    real_env_factory = fitness_module.create_env_from_config

    def spy_decode(g, cfg, *, seed=None):
        captured_decode["seed"] = seed
        return real_decode(g, cfg, seed=seed)

    def spy_env_factory(env_config, *, seed=None):
        captured_env["seed"] = seed
        return real_env_factory(env_config, seed=seed)

    with (
        patch.object(MLPPPOEncoder, "decode", side_effect=spy_decode),
        patch.object(fitness_module, "create_env_from_config", side_effect=spy_env_factory),
    ):
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=99)

    assert captured_decode["seed"] == 99
    assert captured_env["seed"] == 99


def test_evaluate_fitness_seed_overrides_brain_config_seed() -> None:
    """Fitness ``seed`` overrides ``brain.config.seed`` and reaches the runtime RNG.

    Sets ``brain.config.seed = 0`` in sim_config, then runs evaluate with
    seed=1 and seed=2.  The two results MAY differ but aren't guaranteed to
    — random brains can both score 0.0.  What we DO guarantee, and assert
    here, is (a) seed=1 twice produces byte-identical fitness, and (b) a
    different seed runs without error.  Combined with
    ``test_evaluate_passes_seed_to_encoder_decode``, this proves the fitness
    seed (not the YAML BrainConfig.seed) controls per-evaluation RNG.
    """
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    # Pin the YAML brain config seed to 0; the fitness function's seed kwarg
    # overrides this via encoder.decode → instantiate_brain_from_sim_config.
    assert sim_config.brain is not None  # narrows Optional for type checker
    bogus_brain_cfg = sim_config.brain.config.model_copy(update={"seed": 0})
    sim_config = sim_config.model_copy(
        update={
            "brain": sim_config.brain.model_copy(update={"config": bogus_brain_cfg}),
        },
    )

    fitness = EpisodicSuccessRate()
    a = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=1)
    b = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=2)
    # We can't guarantee they differ — random brains can score 0.0 for both —
    # but we CAN guarantee that with seed=1 twice we get the same value, and
    # that the seed kwarg actually reaches the encoder.  Combined with
    # test_evaluate_passes_seed_to_encoder_decode, this gives us the contract.
    a_again = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=1)
    assert a == a_again
    # And b is computed without error for a different seed.
    assert isinstance(b, float)


# ---------------------------------------------------------------------------
# food_history sentinel (B9)
# ---------------------------------------------------------------------------


def test_frozen_eval_runner_preserves_food_history_sentinel() -> None:
    """``FrozenEvalRunner._terminate_episode`` SHALL pass kwargs through unchanged.

    The override only injects ``learn=False, update_memory=False`` and
    forwards everything else.  In particular, the parent's
    ``food_history=...`` Ellipsis sentinel is preserved (not converted to
    ``None``), so the parent correctly falls back to ``agent.food_history``.

    We verify this by calling the override directly with a stand-in agent,
    capturing what gets forwarded to ``super()._terminate_episode``.
    """
    runner = FrozenEvalRunner()

    # Capture what the parent's _terminate_episode receives.
    captured: dict[str, object] = {}

    def fake_super(self, agent, params, reward, **kwargs):
        captured.update(kwargs)

    with patch.object(
        FrozenEvalRunner.__bases__[0],
        "_terminate_episode",
        autospec=True,
        side_effect=fake_super,
    ):
        runner._terminate_episode(  # type: ignore[arg-type]
            agent=None,  # type: ignore[arg-type]
            params=None,
            reward=0.0,
            success=True,
            termination_reason=TerminationReason.COMPLETED_ALL_FOOD,
        )

    # learn and update_memory were forced to False
    assert captured["learn"] is False
    assert captured["update_memory"] is False
    # success and termination_reason passed through
    assert captured["success"] is True
    assert captured["termination_reason"] == TerminationReason.COMPLETED_ALL_FOOD
    # food_history was NOT injected by the override (sentinel preserved by
    # virtue of not being in kwargs at all)
    assert "food_history" not in captured
