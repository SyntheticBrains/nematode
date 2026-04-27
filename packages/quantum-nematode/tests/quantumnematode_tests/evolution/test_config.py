"""Tests for the ``evolution:`` block on :class:`SimulationConfig`."""

from __future__ import annotations

import math
from pathlib import Path

import yaml
from quantumnematode.utils.config_loader import (
    EvolutionConfig,
    SimulationConfig,
    load_simulation_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MLPPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/mlpppo_small_oracle.yml"


def test_existing_scenario_config_loads_without_evolution_block() -> None:
    """Configs with no ``evolution:`` block SHALL load with ``evolution=None``."""
    cfg = load_simulation_config(str(MLPPPO_CONFIG))
    assert cfg.evolution is None


def test_evolution_block_parses_into_populated_config(tmp_path: Path) -> None:
    """A YAML with an ``evolution:`` block SHALL populate ``SimulationConfig.evolution``.

    Unspecified fields fall back to ``EvolutionConfig`` defaults.
    """
    yaml_content = {
        "max_steps": 100,
        "evolution": {
            "algorithm": "cmaes",
            "population_size": 8,
            "generations": 10,
            "episodes_per_eval": 3,
        },
    }
    yaml_path = tmp_path / "ev.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))

    cfg = load_simulation_config(str(yaml_path))
    assert cfg.evolution is not None
    assert cfg.evolution.algorithm == "cmaes"
    assert cfg.evolution.population_size == 8
    assert cfg.evolution.generations == 10
    assert cfg.evolution.episodes_per_eval == 3
    # Unspecified fields use defaults
    assert cfg.evolution.sigma0 == math.pi / 2
    assert cfg.evolution.parallel_workers == 1
    assert cfg.evolution.checkpoint_every == 10


def test_evolution_config_defaults() -> None:
    """``EvolutionConfig()`` SHALL produce documented defaults."""
    ec = EvolutionConfig()
    assert ec.algorithm == "cmaes"
    assert ec.population_size == 20
    assert ec.generations == 50
    assert ec.episodes_per_eval == 15
    assert ec.sigma0 == math.pi / 2
    assert ec.elite_fraction == 0.2
    assert ec.mutation_rate == 0.1
    assert ec.crossover_rate == 0.8
    assert ec.parallel_workers == 1
    assert ec.checkpoint_every == 10
    assert ec.cma_diagonal is False  # Off by default for back-compat


def test_evolution_config_algorithm_literal() -> None:
    """``algorithm`` SHALL accept ``"cmaes"`` and ``"ga"``; reject others."""
    EvolutionConfig(algorithm="cmaes")
    EvolutionConfig(algorithm="ga")
    # Pydantic raises ValidationError on bad literal
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EvolutionConfig(algorithm="invalid")  # type: ignore[arg-type]


def test_simulation_config_extra_unknown_keys_ignored(tmp_path: Path) -> None:
    """Unknown top-level keys SHALL be silently ignored (Pydantic v2 default).

    This is what allows adding the ``evolution`` field without breaking
    older scenario configs that don't know about it.
    """
    yaml_content = {
        "max_steps": 100,
        "some_future_field": "ignored",
    }
    yaml_path = tmp_path / "ev.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))

    # Should not raise.
    cfg = load_simulation_config(str(yaml_path))
    assert isinstance(cfg, SimulationConfig)
    assert cfg.max_steps == 100
