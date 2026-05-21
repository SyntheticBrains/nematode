"""Tests for :class:`LearnedPerformanceFitness`."""

from __future__ import annotations

import re
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
    """``warm_start_path`` set → ``load_weights`` called BEFORE the K train episodes run.

    Mocks ``load_weights`` (no real checkpoint on disk needed) AND patches
    ``StandardEpisodeRunner.run`` so we can verify the call ordering: the
    warm-start load MUST happen before the first train-phase ``run()``
    call, otherwise the K train episodes would be running against
    fresh-init weights instead of the checkpoint.  Both mocks attach to
    a single ``MagicMock`` parent so their interleaved call order is
    captured in ``parent.mock_calls`` for direct inspection.
    """
    from unittest.mock import MagicMock

    from quantumnematode.agent.runners import StandardEpisodeRunner

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

    # Build a parent MagicMock so both children record into a single
    # ordered ``mock_calls`` list — that's what we assert ordering on.
    parent = MagicMock()
    # Eval phase still needs a real return so we don't crash there;
    # ``side_effect=Exception`` would short-circuit but obscure the
    # ordering assertion.  The simplest path is to let the eval
    # ``FrozenEvalRunner.run`` pass through unmocked since this test is
    # only about train-phase ordering.
    with (
        patch("quantumnematode.evolution.fitness.load_weights", parent.load) as mock_load,
        patch.object(StandardEpisodeRunner, "run", parent.train_run),
    ):
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)

    mock_load.assert_called_once()
    # ``load_weights(brain, path)`` — second positional is the path.
    assert mock_load.call_args.args[1] == fake_path

    # Ordering check: in the parent's ``mock_calls`` list, the first call
    # SHALL be ``parent.load(...)`` (the warm-start load) and any
    # ``parent.train_run(...)`` calls SHALL come strictly after.
    call_names = [call[0] for call in parent.mock_calls]
    assert call_names, "no calls recorded — both mocks were skipped"
    assert call_names[0] == "load", (
        f"warm-start load must be the FIRST recorded call; got order: {call_names}"
    )
    assert "train_run" in call_names, (
        "no train-phase StandardEpisodeRunner.run was recorded; the train loop didn't execute"
    )
    assert call_names.index("load") < call_names.index("train_run"), (
        f"warm-start load ({call_names.index('load')}) must come strictly "
        f"before the first train_run ({call_names.index('train_run')}); "
        f"got order: {call_names}"
    )


def test_warm_start_unset_skips_load() -> None:
    """``warm_start_path is None`` (default) → ``load_weights`` NOT called.

    The default code path uses fresh-init weights from
    ``encoder.decode`` with no extra load step.
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


def test_tei_prior_source_corrupted_substrate_raises_operator_friendly(
    tmp_path: Path,
) -> None:
    """A corrupted ``.tei.pt`` SHALL surface a RuntimeError naming the path.

    Without the wrapping at ``LearnedPerformanceFitness.evaluate``'s
    substrate-load site, a raw torch unpickle exception escapes inside
    the worker — hard to correlate with the source artifact.
    """
    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=1)
    corrupted_path = tmp_path / "elite_g0.tei.pt"
    # Write junk bytes — torch.load will fail to unpickle.
    corrupted_path.write_bytes(b"\x00\x01\x02not a torch tensor")
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    # Match both the generic prefix AND the failing path — the whole
    # point of the wrapping is that operators can correlate the error
    # back to the source artifact.
    expected_match = r"Failed to load transgenerational substrate from .*" + re.escape(
        str(corrupted_path),
    )
    with pytest.raises(RuntimeError, match=expected_match):
        fitness.evaluate(
            genome,
            sim_config,
            encoder,
            episodes=1,
            seed=42,
            tei_prior_source=(corrupted_path, 0.6, 1),
        )


# ---------------------------------------------------------------------------
# fitness_metric dispatch — primary-metric selector
# ---------------------------------------------------------------------------


def _patched_evolution_config(
    sim_config: SimulationConfig,
    *,
    fitness_metric: str = "composite",
    fitness_survival_weight: float = 0.0,
    learn_eps: int = 2,
    eval_eps: int | None = 1,
) -> SimulationConfig:
    """Rebuild the sim_config with a specific fitness_metric + weight."""
    assert sim_config.evolution is not None
    return sim_config.model_copy(
        update={
            "evolution": sim_config.evolution.model_copy(
                update={
                    "fitness_metric": fitness_metric,
                    "fitness_survival_weight": fitness_survival_weight,
                    "learn_episodes_per_eval": learn_eps,
                    "eval_episodes_per_eval": eval_eps,
                },
            ),
        },
    )


def _patch_runs(success_count: int, death_count: int, eval_count: int):
    """Patch ``FrozenEvalRunner.run`` to return deterministic termination_reasons.

    First ``success_count`` calls return COMPLETED_ALL_FOOD; next ``death_count``
    return HEALTH_DEPLETED; the rest return MAX_STEPS (neither survive nor
    succeed). Caller is responsible for ensuring ``success_count + death_count
    <= eval_count``.
    """
    sequence: list[TerminationReason] = (
        [TerminationReason.COMPLETED_ALL_FOOD] * success_count
        + [TerminationReason.HEALTH_DEPLETED] * death_count
        + [TerminationReason.MAX_STEPS] * (eval_count - success_count - death_count)
    )
    iterator = iter(sequence)

    class _StubResult:
        def __init__(self, reason: TerminationReason) -> None:
            self.termination_reason = reason

    def _fake_run(self, agent, reward_config, max_steps):
        return _StubResult(next(iterator))

    return patch.object(FrozenEvalRunner, "run", _fake_run)


def test_fitness_metric_default_composite_is_byte_equivalent() -> None:
    """``fitness_metric`` defaults to ``composite``; preserves the legacy composite formula."""
    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=4)
    sim_config = _patched_evolution_config(
        sim_config,
        fitness_metric="composite",
        fitness_survival_weight=1.0,
        eval_eps=4,
    )
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    # 2 successes, 2 deaths → success_rate=0.5, death_rate=0.5
    # composite = 0.5 * (1 - 1.0 * 0.5) = 0.25
    with _patch_runs(success_count=2, death_count=2, eval_count=4):
        score = fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)
    assert score == pytest.approx(0.25, rel=1e-6)


def test_fitness_metric_survival_rate_returns_survival_fraction() -> None:
    """``fitness_metric: survival_rate`` SHALL return ``1 - deaths/eval_count``."""
    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=4)
    sim_config = _patched_evolution_config(
        sim_config,
        fitness_metric="survival_rate",
        fitness_survival_weight=1.0,
        eval_eps=4,
    )
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    # 1 success, 1 death, 2 max_steps → survival_rate = 1 - 1/4 = 0.75
    with _patch_runs(success_count=1, death_count=1, eval_count=4):
        score = fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)
    assert score == pytest.approx(0.75, rel=1e-6)


def test_fitness_metric_survival_rate_ignores_survival_weight() -> None:
    """Under ``survival_rate`` dispatch, ``fitness_survival_weight`` SHALL NOT affect the return."""
    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=4)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    # Same termination distribution, two different fitness_survival_weight values.
    sim_w0 = _patched_evolution_config(
        sim_config,
        fitness_metric="survival_rate",
        fitness_survival_weight=0.0,
        eval_eps=4,
    )
    sim_w1 = _patched_evolution_config(
        sim_config,
        fitness_metric="survival_rate",
        fitness_survival_weight=1.0,
        eval_eps=4,
    )
    # 4 deaths → survival_rate = 0.0 regardless of weight.
    with _patch_runs(success_count=0, death_count=4, eval_count=4):
        score_w0 = fitness.evaluate(genome, sim_w0, encoder, episodes=1, seed=42)
    with _patch_runs(success_count=0, death_count=4, eval_count=4):
        score_w1 = fitness.evaluate(genome, sim_w1, encoder, episodes=1, seed=42)
    assert score_w0 == pytest.approx(0.0, abs=1e-6)
    assert score_w1 == pytest.approx(0.0, abs=1e-6)


def test_fitness_metric_success_rate_returns_foraging_fraction() -> None:
    """``fitness_metric: success_rate`` SHALL return ``successes/eval_count`` ignoring deaths."""
    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=4)
    sim_config = _patched_evolution_config(
        sim_config,
        fitness_metric="success_rate",
        fitness_survival_weight=1.0,  # ignored under success_rate
        eval_eps=4,
    )
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    # 3 successes, 1 death → success_rate = 3/4 = 0.75 (deaths ignored)
    with _patch_runs(success_count=3, death_count=1, eval_count=4):
        score = fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)
    assert score == pytest.approx(0.75, rel=1e-6)


def test_fitness_metric_invalid_value_rejected_at_yaml_load() -> None:
    """Pydantic SHALL reject ``fitness_metric`` values outside the Literal."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EvolutionConfig(
            algorithm="cmaes",
            population_size=2,
            generations=1,
            episodes_per_eval=1,
            fitness_metric="fastest_food",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# eval_diagnostics.jsonl side-channel writer
# ---------------------------------------------------------------------------


def test_eval_diagnostics_writes_jsonl_row_with_all_metrics(tmp_path: Path) -> None:
    """``diagnostics_path`` SHALL emit one JSONL row with all metric variants."""
    import json

    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=4)
    sim_config = _patched_evolution_config(
        sim_config,
        fitness_metric="survival_rate",
        fitness_survival_weight=1.0,
        eval_eps=4,
    )
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    diag_path = tmp_path / "eval_diagnostics.jsonl"
    with _patch_runs(success_count=2, death_count=1, eval_count=4):
        score = fitness.evaluate(
            genome,
            sim_config,
            encoder,
            episodes=1,
            seed=42,
            diagnostics_path=diag_path,
        )
    assert diag_path.exists()
    rows = [json.loads(line) for line in diag_path.read_text().strip().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["genome_id"] == "test"
    assert row["eval_count"] == 4
    assert row["successes"] == 2
    assert row["deaths"] == 1
    assert row["success_rate"] == pytest.approx(0.50)
    assert row["survival_rate"] == pytest.approx(0.75)  # 1 - 1/4
    assert row["composite"] == pytest.approx(0.50 * (1.0 - 1.0 * 0.25))
    assert row["fitness"] == pytest.approx(score)
    assert row["fitness_metric"] == "survival_rate"
    assert row["fitness_survival_weight"] == pytest.approx(1.0)


def test_eval_diagnostics_none_preserves_no_side_effects(tmp_path: Path) -> None:
    """When ``diagnostics_path`` is None, no file is written + return value is unchanged."""
    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=4)
    sim_config = _patched_evolution_config(
        sim_config,
        fitness_metric="composite",
        fitness_survival_weight=0.5,
        eval_eps=4,
    )
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    # Run once WITHOUT diagnostics_path.
    with _patch_runs(success_count=2, death_count=1, eval_count=4):
        score_no_diag = fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)
    # Run again WITH diagnostics_path on a fresh path.
    diag_path = tmp_path / "diag.jsonl"
    with _patch_runs(success_count=2, death_count=1, eval_count=4):
        score_with_diag = fitness.evaluate(
            genome,
            sim_config,
            encoder,
            episodes=1,
            seed=42,
            diagnostics_path=diag_path,
        )
    # The side-channel is observation-only.
    assert score_no_diag == pytest.approx(score_with_diag, rel=1e-9)
    # No file is written when diagnostics_path is None.
    assert not (tmp_path / "eval_diagnostics.jsonl").exists()
    assert diag_path.exists()


def test_eval_diagnostics_appends_across_multiple_calls(tmp_path: Path) -> None:
    """Multiple ``evaluate`` calls to the same ``diagnostics_path`` SHALL append, not overwrite."""
    import json

    sim_config = _make_sim_config_with_schema(learn_eps=2, eval_eps=4)
    sim_config = _patched_evolution_config(
        sim_config,
        fitness_metric="composite",
        eval_eps=4,
    )
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    diag_path = tmp_path / "eval_diagnostics.jsonl"
    for _ in range(3):
        with _patch_runs(success_count=1, death_count=1, eval_count=4):
            fitness.evaluate(
                genome,
                sim_config,
                encoder,
                episodes=1,
                seed=42,
                diagnostics_path=diag_path,
            )
    rows = [json.loads(line) for line in diag_path.read_text().strip().splitlines()]
    assert len(rows) == 3
