"""Tests for the per-genome warm-start override + weight-capture kwargs.

Both kwargs were added in M3 to plumb Lamarckian inheritance through the
existing ``LearnedPerformanceFitness.evaluate`` signature without
mutating run-wide config or changing the ``FitnessFunction`` Protocol.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from quantumnematode.evolution.encoders import HyperparameterEncoder, build_birth_metadata
from quantumnematode.evolution.fitness import LearnedPerformanceFitness
from quantumnematode.evolution.genome import Genome
from quantumnematode.utils.config_loader import (
    EvolutionConfig,
    ParamSchemaEntry,
    SimulationConfig,
    load_simulation_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MLPPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/mlpppo_small_oracle.yml"


def _make_sim_config(learn_eps: int = 2, eval_eps: int | None = 1) -> SimulationConfig:
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
    return Genome(
        params=np.array([0.001], dtype=np.float32),
        genome_id="test-genome",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )


# ---------------------------------------------------------------------------
# weight_capture_path
# ---------------------------------------------------------------------------


def test_learned_performance_writes_capture_file_when_path_set(tmp_path: Path) -> None:
    """``weight_capture_path`` SHALL trigger ``save_weights`` after train, before eval.

    Spec scenario "weight_capture_path captures post-train weights before eval".
    Mocked to verify the call without depending on torch.save's exact behaviour.
    """
    sim_config = _make_sim_config(learn_eps=2, eval_eps=1)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    capture_path = tmp_path / "captured" / "genome.pt"

    with patch("quantumnematode.evolution.fitness.save_weights") as mock_save:
        score = fitness.evaluate(
            genome,
            sim_config,
            encoder,
            episodes=1,
            seed=42,
            weight_capture_path=capture_path,
        )

    assert 0.0 <= score <= 1.0
    # save_weights was called exactly once, with a brain instance and the capture path.
    mock_save.assert_called_once()
    _brain_arg, path_arg = mock_save.call_args.args
    assert path_arg == capture_path
    # Parent directory was created (mkdir parents=True, exist_ok=True).
    assert capture_path.parent.exists()


def test_learned_performance_no_capture_when_path_none(tmp_path: Path) -> None:
    """When ``weight_capture_path=None`` (default), no save call SHALL happen."""
    sim_config = _make_sim_config(learn_eps=2, eval_eps=1)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    with patch("quantumnematode.evolution.fitness.save_weights") as mock_save:
        # Implicit None for weight_capture_path
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)
    mock_save.assert_not_called()
    assert not (tmp_path / "captured").exists()


# ---------------------------------------------------------------------------
# warm_start_path_override
# ---------------------------------------------------------------------------


def test_learned_performance_loads_warm_start_override(tmp_path: Path) -> None:
    """``warm_start_path_override`` SHALL be passed to ``load_weights`` per call.

    Spec scenario "warm_start_path_override takes precedence per-genome".
    Mocked load_weights to verify the override path is the one used.
    """
    sim_config = _make_sim_config(learn_eps=2, eval_eps=1)
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()
    override_path = tmp_path / "override" / "parent.pt"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.touch()  # exists check is implicit in load_weights

    with patch("quantumnematode.evolution.fitness.load_weights") as mock_load:
        fitness.evaluate(
            genome,
            sim_config,
            encoder,
            episodes=1,
            seed=42,
            warm_start_path_override=override_path,
        )

    mock_load.assert_called_once()
    _brain_arg, path_arg = mock_load.call_args.args
    assert path_arg == override_path


def test_learned_performance_override_wins_over_evolution_warm_start(tmp_path: Path) -> None:
    """When both override AND evolution.warm_start_path are set, override SHALL win.

    The validator on EvolutionConfig forbids the combination at YAML
    load time, so this only tests the override-precedence rule at the
    fitness level — exercising the documented contract via direct
    construction (skipping the validator).
    """
    sim_config = _make_sim_config(learn_eps=2, eval_eps=1)
    # Bypass EvolutionConfig._validate_inheritance by post-update — testing
    # the fitness function's resolution rule, not the validator.
    sim_config.evolution.warm_start_path = tmp_path / "static.pt"
    (tmp_path / "static.pt").touch()
    override_path = tmp_path / "override.pt"
    override_path.touch()

    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    with patch("quantumnematode.evolution.fitness.load_weights") as mock_load:
        fitness.evaluate(
            genome,
            sim_config,
            encoder,
            episodes=1,
            seed=42,
            warm_start_path_override=override_path,
        )
    _brain_arg, path_arg = mock_load.call_args.args
    assert path_arg == override_path  # override wins


def test_learned_performance_no_load_when_both_none(tmp_path: Path) -> None:
    """When neither override nor static path is set, no load SHALL happen."""
    sim_config = _make_sim_config(learn_eps=2, eval_eps=1)
    assert sim_config.evolution.warm_start_path is None
    encoder = HyperparameterEncoder()
    genome = _make_genome(sim_config)
    fitness = LearnedPerformanceFitness()

    with patch("quantumnematode.evolution.fitness.load_weights") as mock_load:
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=42)
    mock_load.assert_not_called()
    # tmp_path used to satisfy fixture ordering only
    assert tmp_path.exists()
