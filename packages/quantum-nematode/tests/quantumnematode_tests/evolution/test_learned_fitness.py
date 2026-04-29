"""Tests for :class:`LearnedPerformanceFitness`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from quantumnematode.evolution.encoders import HyperparameterEncoder
from quantumnematode.evolution.fitness import (
    FrozenEvalRunner,
    LearnedPerformanceFitness,
)
from quantumnematode.evolution.genome import Genome
from quantumnematode.report.dtypes import TerminationReason
from quantumnematode.utils.config_loader import (
    EvolutionConfig,
    ParamSchemaEntry,
    SimulationConfig,
    load_simulation_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MLPPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/mlpppo_small_oracle.yml"


def _make_sim_config_with_schema(
    learn_eps: int = 2,
    eval_eps: int | None = 1,
) -> SimulationConfig:
    """Build a SimulationConfig with hyperparam_schema and the K/L knobs."""
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    return sim_config.model_copy(
        update={
            "hyperparam_schema": [
                ParamSchemaEntry(
                    name="learning_rate",
                    type="float",
                    bounds=(1e-5, 1e-2),
                ),
            ],
            "evolution": EvolutionConfig(
                algorithm="cmaes",
                population_size=2,
                generations=1,
                episodes_per_eval=1,
                learn_episodes_per_eval=learn_eps,
                eval_episodes_per_eval=eval_eps,
            ),
        },
    )


def _make_genome(sim_config: SimulationConfig) -> Genome:
    """Build a single hyperparam genome via build_birth_metadata."""
    from quantumnematode.evolution.encoders import build_birth_metadata

    return Genome(
        params=np.array([0.001], dtype=np.float32),
        genome_id="test",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_learned_performance_smoke_k2_l1() -> None:
    """K=2 train + L=1 eval SHALL return a float in [0, 1]."""
    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=1)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    score = fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Defensive guards
# ---------------------------------------------------------------------------


def test_no_evolution_block_raises() -> None:
    """Missing evolution: block SHALL raise with clear message."""
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))  # no evolution
    assert sim_config.evolution is None
    encoder = HyperparameterEncoder()
    genome = Genome(
        params=np.array([0.001], dtype=np.float32),
        genome_id="t",
        parent_ids=[],
        generation=0,
        birth_metadata={
            "param_schema": [{"name": "learning_rate", "type": "float", "bounds": [1e-5, 1e-2]}],
        },
    )
    fitness = LearnedPerformanceFitness()
    with pytest.raises(ValueError, match="evolution:"):
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)


def test_no_environment_raises() -> None:
    """Missing environment: block SHALL raise (mirrors EpisodicSuccessRate)."""
    sim_config = _make_sim_config_with_schema(learn_eps=2)
    sim_config = sim_config.model_copy(update={"environment": None})
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    with pytest.raises(ValueError, match="environment"):
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)


def test_no_reward_raises() -> None:
    """Missing reward: block SHALL raise (mirrors EpisodicSuccessRate)."""
    sim_config = _make_sim_config_with_schema(learn_eps=2)
    sim_config = sim_config.model_copy(update={"reward": None})
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    with pytest.raises(ValueError, match="reward"):
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)


def test_k0_raises_with_episodic_success_rate_hint() -> None:
    """learn_episodes_per_eval=0 SHALL raise with EpisodicSuccessRate hint."""
    sim_config = _make_sim_config_with_schema(learn_eps=0, eval_eps=1)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    with pytest.raises(ValueError, match="EpisodicSuccessRate"):
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)


# ---------------------------------------------------------------------------
# Train/eval phase invariants
# ---------------------------------------------------------------------------


def test_eval_env_is_fresh_create_env_called_twice() -> None:
    """``create_env_from_config`` SHALL be called exactly twice per evaluate.

    Once for train env, once for eval env.  Locks the
    fresh-env-on-eval invariant in.
    """
    sim_config = _make_sim_config_with_schema(learn_eps=1, eval_eps=1)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    with patch(
        "quantumnematode.evolution.fitness.create_env_from_config",
        wraps=__import__(
            "quantumnematode.evolution.fitness",
            fromlist=["create_env_from_config"],
        ).create_env_from_config,
    ) as mock_create:
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)
    assert mock_create.call_count == 2


def test_eval_phase_does_not_call_learn() -> None:
    """During L eval episodes, brain.learn() SHALL NOT be called.

    Mirrors the test_frozen_eval_runner_never_calls_learn approach.
    """
    sim_config = _make_sim_config_with_schema(learn_eps=1, eval_eps=2)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    # Spy on FrozenEvalRunner.run to confirm it's used for the eval phase
    # (the L eval episodes go through this class, which neuters brain.learn).
    eval_run_count = 0
    original_run = FrozenEvalRunner.run

    def _counting_run(self, *args, **kwargs):
        nonlocal eval_run_count
        eval_run_count += 1
        return original_run(self, *args, **kwargs)

    with patch.object(FrozenEvalRunner, "run", _counting_run):
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)
    # 2 eval episodes → 2 calls to FrozenEvalRunner.run.  This proves the
    # eval phase actually goes through the frozen runner (where
    # brain.learn is a no-op), as opposed to the standard runner.
    assert eval_run_count == 2


# ---------------------------------------------------------------------------
# Eval-episode count resolution (L)
# ---------------------------------------------------------------------------


def test_eval_episodes_falls_back_to_kwarg_when_yaml_none() -> None:
    """eval_episodes_per_eval=None SHALL fall back to the ``episodes`` kwarg."""
    sim_config = _make_sim_config_with_schema(learn_eps=1, eval_eps=None)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    eval_run_count = 0
    original_run = FrozenEvalRunner.run

    def _counting_run(self, *args, **kwargs):
        nonlocal eval_run_count
        eval_run_count += 1
        return original_run(self, *args, **kwargs)

    with patch.object(FrozenEvalRunner, "run", _counting_run):
        fitness.evaluate(genome, sim_config, encoder, episodes=5, seed=42)
    assert eval_run_count == 5  # kwarg wins


def test_eval_episodes_yaml_overrides_kwarg_when_set() -> None:
    """eval_episodes_per_eval IN YAML SHALL win over ``episodes`` kwarg."""
    sim_config = _make_sim_config_with_schema(learn_eps=1, eval_eps=3)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    eval_run_count = 0
    original_run = FrozenEvalRunner.run

    def _counting_run(self, *args, **kwargs):
        nonlocal eval_run_count
        eval_run_count += 1
        return original_run(self, *args, **kwargs)

    with patch.object(FrozenEvalRunner, "run", _counting_run):
        fitness.evaluate(genome, sim_config, encoder, episodes=99, seed=42)
    assert eval_run_count == 3  # YAML wins


# ---------------------------------------------------------------------------
# Score correctness
# ---------------------------------------------------------------------------


def test_score_uses_termination_reason_for_success() -> None:
    """Success counted iff result.termination_reason == COMPLETED_ALL_FOOD."""
    sim_config = _make_sim_config_with_schema(learn_eps=1, eval_eps=4)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    # Build mock results: 3 success, 1 failure.  Use a SimpleNamespace
    # rather than constructing real EpisodeResult instances (the real
    # type takes more args than we need to mock; the score path only
    # reads result.termination_reason).
    from types import SimpleNamespace

    mock_results = [
        SimpleNamespace(termination_reason=TerminationReason.COMPLETED_ALL_FOOD),
        SimpleNamespace(termination_reason=TerminationReason.MAX_STEPS),
        SimpleNamespace(termination_reason=TerminationReason.COMPLETED_ALL_FOOD),
        SimpleNamespace(termination_reason=TerminationReason.COMPLETED_ALL_FOOD),
    ]
    call_idx = 0

    def _mock_run(self, *args, **kwargs):
        nonlocal call_idx
        result = mock_results[call_idx]
        call_idx += 1
        return result

    with patch.object(FrozenEvalRunner, "run", _mock_run):
        score = fitness.evaluate(genome, sim_config, encoder, episodes=99, seed=42)
    # 3 successes / 4 eval episodes
    assert score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Train phase actually runs
# ---------------------------------------------------------------------------


def test_train_phase_runs_k_episodes() -> None:
    """StandardEpisodeRunner.run SHALL be called K times during the train phase."""
    sim_config = _make_sim_config_with_schema(learn_eps=3, eval_eps=1)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    train_run_count = 0
    from quantumnematode.agent.runners import StandardEpisodeRunner

    original_run = StandardEpisodeRunner.run

    def _counting_run(self, *args, **kwargs):
        nonlocal train_run_count
        # Distinguish train (StandardEpisodeRunner) from eval (FrozenEvalRunner
        # which subclasses StandardEpisodeRunner — count only when it's the
        # base class instance).
        if type(self).__name__ == "StandardEpisodeRunner":
            train_run_count += 1
        return original_run(self, *args, **kwargs)

    with patch.object(StandardEpisodeRunner, "run", _counting_run):
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)

    # 3 train episodes + 1 eval episode (FrozenEvalRunner subclasses
    # StandardEpisodeRunner so its dispatch goes through the patch too,
    # but the type-check filters it).
    assert train_run_count == 3


# ---------------------------------------------------------------------------
# Env reset between episodes (both train and eval phases)
# ---------------------------------------------------------------------------


def test_env_resets_between_train_episodes() -> None:
    """Train phase SHALL call ``agent.reset_environment()`` between episodes.

    Without per-episode reset, a failed episode (e.g. starvation) leaves
    the env in a degraded state that subsequent episodes inherit.  The
    brain weights persist across episodes (that's training); the env
    state must NOT.
    """
    from quantumnematode.agent.agent import QuantumNematodeAgent

    sim_config = _make_sim_config_with_schema(learn_eps=3, eval_eps=1)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    reset_call_count = 0
    original_reset = QuantumNematodeAgent.reset_environment

    def _counting_reset(self) -> None:
        nonlocal reset_call_count
        reset_call_count += 1
        return original_reset(self)

    with patch.object(QuantumNematodeAgent, "reset_environment", _counting_reset):
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)

    # 3 train episodes → 2 resets (between ep 0→1 and ep 1→2).
    # 1 eval episode → 0 additional resets.
    # Total: 2 (train transitions) + 0 (eval is single ep) = 2.
    expected_train_resets = 2
    assert reset_call_count == expected_train_resets


def test_env_resets_between_eval_episodes() -> None:
    """Eval phase SHALL also reset env between episodes."""
    from quantumnematode.agent.agent import QuantumNematodeAgent

    sim_config = _make_sim_config_with_schema(learn_eps=1, eval_eps=4)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    reset_call_count = 0
    original_reset = QuantumNematodeAgent.reset_environment

    def _counting_reset(self) -> None:
        nonlocal reset_call_count
        reset_call_count += 1
        return original_reset(self)

    with patch.object(QuantumNematodeAgent, "reset_environment", _counting_reset):
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)

    # 1 train ep → 0 train transitions.  4 eval eps → 3 eval transitions.
    expected_eval_resets = 3
    assert reset_call_count == expected_eval_resets


def test_eval_count_zero_raises() -> None:
    """``eval_count <= 0`` SHALL raise ValueError before the eval loop runs.

    This is reachable when a programmatic caller passes ``episodes=0``
    AND ``evolution.eval_episodes_per_eval`` is None — Pydantic's
    ge=1 constraint on the field can't catch the protocol kwarg.
    """
    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=None)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    with pytest.raises(ValueError, match="eval_count must be > 0"):
        fitness.evaluate(genome, sim_config, encoder, episodes=0, seed=42)


# ---------------------------------------------------------------------------
# Warm-start
# ---------------------------------------------------------------------------


def test_warm_start_loads_weights_before_train(tmp_path: Path) -> None:
    """``warm_start_path`` set → ``load_weights`` called with that path.

    Mocks ``load_weights`` so the test stays unit-scoped (no real checkpoint
    on disk).  Asserts the load is invoked with the brain (decoded from the
    genome) and the configured warm-start path.
    """
    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=1)
    fake_path = tmp_path / "fake_checkpoint.pt"
    assert sim_config.evolution is not None
    sim_config = sim_config.model_copy(
        update={
            "evolution": sim_config.evolution.model_copy(
                update={"warm_start_path": fake_path},
            ),
        },
    )
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    with patch("quantumnematode.evolution.fitness.load_weights") as mock_load:
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)

    mock_load.assert_called_once()
    # ``load_weights(brain, path)`` — second positional is the path.
    assert mock_load.call_args.args[1] == fake_path


def test_warm_start_unset_skips_load() -> None:
    """``warm_start_path is None`` (default) → ``load_weights`` NOT called.

    Preserves M2 part-1 behaviour: fresh-init weights from
    ``encoder.decode``, no extra step.
    """
    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=1)
    assert sim_config.evolution is not None
    assert sim_config.evolution.warm_start_path is None
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    with patch("quantumnematode.evolution.fitness.load_weights") as mock_load:
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)

    mock_load.assert_not_called()


def test_warm_start_missing_path_raises(tmp_path: Path) -> None:
    """``warm_start_path`` pointing at a missing file → ``FileNotFoundError``.

    Error originates in ``brain.weights.load_weights`` (NOT mocked here)
    and must surface to the caller — fitness.evaluate must not swallow
    or remap the exception.
    """
    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=1)
    missing_path = tmp_path / "does_not_exist.pt"
    assert sim_config.evolution is not None
    sim_config = sim_config.model_copy(
        update={
            "evolution": sim_config.evolution.model_copy(
                update={"warm_start_path": missing_path},
            ),
        },
    )
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    with pytest.raises(FileNotFoundError, match="Weight file not found"):
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)
