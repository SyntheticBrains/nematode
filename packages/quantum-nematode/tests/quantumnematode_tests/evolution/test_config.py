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


def test_lstmppo_pilot_config_enables_cma_diagonal() -> None:
    """The shipped LSTMPPO+klinotaxis pilot SHALL set ``cma_diagonal: true``.

    Regression guard: at LSTMPPO weight scale (~47k dims) full-cov CMA-ES
    is intractable (each ``tell()`` takes minutes).  If this regresses,
    anyone running the pilot will hit the multi-minute hang.
    """
    pilot_path = PROJECT_ROOT / "configs/evolution/lstmppo_foraging_small_klinotaxis.yml"
    cfg = load_simulation_config(str(pilot_path))
    assert cfg.evolution is not None
    assert cfg.evolution.cma_diagonal is True


def test_cma_diagonal_yaml_propagates_to_optimizer_options(tmp_path: Path) -> None:
    """``cma_diagonal: true`` SHALL propagate from YAML to the cma library option.

    Locks in the full chain: YAML → ``EvolutionConfig`` → ``CMAESOptimizer``
    → ``cma.CMAEvolutionStrategy.opts['CMA_diagonal']``.  Without this, a
    future refactor could silently break the plumbing while leaving the
    config-level field intact.
    """
    from quantumnematode.optimizers.evolutionary import CMAESOptimizer

    yaml_content = {
        "max_steps": 100,
        "evolution": {
            "algorithm": "cmaes",
            "population_size": 4,
            "cma_diagonal": True,
        },
    }
    yaml_path = tmp_path / "diag.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))

    cfg = load_simulation_config(str(yaml_path))
    assert cfg.evolution is not None
    assert cfg.evolution.cma_diagonal is True

    # Same construction path as scripts/run_evolution.py:_build_optimizer
    opt = CMAESOptimizer(
        num_params=8,
        population_size=cfg.evolution.population_size,
        sigma0=cfg.evolution.sigma0,
        seed=42,
        diagonal=cfg.evolution.cma_diagonal,
    )
    # cma library translates True to True in the options dict.
    assert opt._es.opts.get("CMA_diagonal") is True
