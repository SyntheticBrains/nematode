"""Tests for the ``evolution:`` block on :class:`SimulationConfig`."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from quantumnematode.utils.config_loader import (
    EvolutionConfig,
    ParamSchemaEntry,
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


# =============================================================================
# ParamSchemaEntry + SimulationConfig.hyperparam_schema
# =============================================================================


def test_param_schema_entry_validates_type_metadata() -> None:
    """``ParamSchemaEntry`` SHALL reject mismatched type/metadata combinations."""
    # float without bounds
    with pytest.raises(ValidationError, match="requires 'bounds'"):
        ParamSchemaEntry(name="lr", type="float")

    # int without bounds
    with pytest.raises(ValidationError, match="requires 'bounds'"):
        ParamSchemaEntry(name="hidden", type="int")

    # categorical with 1 value
    with pytest.raises(ValidationError, match="at least 2"):
        ParamSchemaEntry(name="rnn", type="categorical", values=["lstm"])

    # categorical with no values
    with pytest.raises(ValidationError, match="at least 2"):
        ParamSchemaEntry(name="rnn", type="categorical")

    # bool with log_scale=True
    with pytest.raises(ValidationError, match="log_scale"):
        ParamSchemaEntry(name="gating", type="bool", log_scale=True)

    # bool with bounds
    with pytest.raises(ValidationError, match="bounds"):
        ParamSchemaEntry(name="gating", type="bool", bounds=(0.0, 1.0))

    # categorical with bounds
    with pytest.raises(ValidationError, match="bounds"):
        ParamSchemaEntry(
            name="rnn",
            type="categorical",
            values=["lstm", "gru"],
            bounds=(0.0, 1.0),
        )

    # int with log_scale=True
    with pytest.raises(ValidationError, match="log_scale"):
        ParamSchemaEntry(name="hidden", type="int", bounds=(32, 256), log_scale=True)

    # float with values
    with pytest.raises(ValidationError, match="values"):
        ParamSchemaEntry(name="lr", type="float", bounds=(1e-5, 1e-2), values=["a"])


def test_param_schema_entry_valid_examples() -> None:
    """Valid combinations SHALL parse without error."""
    # float with bounds
    e1 = ParamSchemaEntry(name="lr", type="float", bounds=(1e-5, 1e-2))
    assert e1.name == "lr"
    assert e1.log_scale is False

    # float with bounds + log_scale
    e2 = ParamSchemaEntry(name="lr", type="float", bounds=(1e-5, 1e-2), log_scale=True)
    assert e2.log_scale is True

    # int with bounds
    e3 = ParamSchemaEntry(name="hidden", type="int", bounds=(32, 256))
    assert e3.type == "int"

    # bool with no metadata
    e4 = ParamSchemaEntry(name="gating", type="bool")
    assert e4.type == "bool"

    # categorical with values
    e5 = ParamSchemaEntry(name="rnn", type="categorical", values=["lstm", "gru"])
    assert e5.values == ["lstm", "gru"]


def test_hyperparam_schema_yaml_parses(tmp_path: Path) -> None:
    """A YAML with mixed-type ``hyperparam_schema`` SHALL load into ``ParamSchemaEntry``."""
    yaml_content = {
        "brain": {"name": "mlpppo", "config": {}},
        "hyperparam_schema": [
            {"name": "actor_hidden_dim", "type": "int", "bounds": [32, 256]},
            {
                "name": "learning_rate",
                "type": "float",
                "bounds": [1.0e-5, 1.0e-2],
                "log_scale": True,
            },
            {"name": "feature_gating", "type": "bool"},
        ],
    }
    yaml_path = tmp_path / "schema.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))

    cfg = load_simulation_config(str(yaml_path))
    assert cfg.hyperparam_schema is not None
    assert len(cfg.hyperparam_schema) == 3
    assert all(isinstance(e, ParamSchemaEntry) for e in cfg.hyperparam_schema)
    assert cfg.hyperparam_schema[0].name == "actor_hidden_dim"
    assert cfg.hyperparam_schema[1].log_scale is True


def test_hyperparam_schema_rejects_typo(tmp_path: Path) -> None:
    """A schema entry with a name not on the brain config SHALL fail load.

    Must list valid alternatives so the user can correct the typo.
    """
    yaml_content = {
        "brain": {"name": "mlpppo", "config": {}},
        "hyperparam_schema": [
            {"name": "actor_hidden_dimm", "type": "int", "bounds": [32, 256]},
        ],
    }
    yaml_path = tmp_path / "typo.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))

    with pytest.raises(ValidationError) as exc_info:
        load_simulation_config(str(yaml_path))
    msg = str(exc_info.value)
    assert "actor_hidden_dimm" in msg
    # Should list valid alternatives — at minimum the real field
    # actor_hidden_dim should appear.
    assert "actor_hidden_dim" in msg


def test_hyperparam_schema_absence_preserves_m0_behaviour() -> None:
    """Existing scenario configs without ``hyperparam_schema`` SHALL load with ``None``."""
    cfg = load_simulation_config(str(MLPPPO_CONFIG))
    assert cfg.hyperparam_schema is None


def test_hyperparam_schema_requires_brain_block(tmp_path: Path) -> None:
    """A YAML with ``hyperparam_schema`` but no ``brain:`` SHALL fail clearly."""
    yaml_content = {
        "hyperparam_schema": [
            {"name": "anything", "type": "int", "bounds": [1, 10]},
        ],
    }
    yaml_path = tmp_path / "no_brain.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))

    with pytest.raises(ValidationError) as exc_info:
        load_simulation_config(str(yaml_path))
    msg = str(exc_info.value)
    assert "brain:" in msg


def test_hyperparam_schema_unknown_brain_name(tmp_path: Path) -> None:
    """A YAML with ``brain.name: bogus_brain`` SHALL fail with registered-brain hint."""
    yaml_content = {
        "brain": {"name": "bogus_brain", "config": {}},
        "hyperparam_schema": [
            {"name": "anything", "type": "int", "bounds": [1, 10]},
        ],
    }
    yaml_path = tmp_path / "bogus.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))

    with pytest.raises(ValidationError) as exc_info:
        load_simulation_config(str(yaml_path))
    msg = str(exc_info.value)
    assert "bogus_brain" in msg
    assert "mlpppo" in msg  # at least one registered brain in the message


# =============================================================================
# EvolutionConfig.learn_episodes_per_eval + eval_episodes_per_eval
# =============================================================================


def test_evolution_config_learn_eval_defaults() -> None:
    """The new fields SHALL default to ``0`` and ``None``."""
    cfg = EvolutionConfig()
    assert cfg.learn_episodes_per_eval == 0
    assert cfg.eval_episodes_per_eval is None


def test_evolution_config_learn_eval_bounds() -> None:
    """Pydantic SHALL enforce ge=0 on learn and ge=1 on eval (None allowed)."""
    with pytest.raises(ValidationError):
        EvolutionConfig(learn_episodes_per_eval=-1)
    with pytest.raises(ValidationError):
        EvolutionConfig(eval_episodes_per_eval=0)
    # Valid: positive values + None
    cfg = EvolutionConfig(learn_episodes_per_eval=30, eval_episodes_per_eval=5)
    assert cfg.learn_episodes_per_eval == 30
    assert cfg.eval_episodes_per_eval == 5


def test_param_schema_entry_rejects_inverted_bounds() -> None:
    """``bounds`` SHALL require strictly increasing (low < high)."""
    with pytest.raises(ValidationError, match="must be strictly increasing"):
        ParamSchemaEntry(name="lr", type="float", bounds=(1e-2, 1e-5))
    with pytest.raises(ValidationError, match="must be strictly increasing"):
        ParamSchemaEntry(name="hidden", type="int", bounds=(256, 32))
    # Equal-bounds also rejected
    with pytest.raises(ValidationError, match="must be strictly increasing"):
        ParamSchemaEntry(name="x", type="float", bounds=(1.0, 1.0))


def test_param_schema_entry_log_scale_requires_positive_bounds() -> None:
    """``log_scale=True`` SHALL require both bounds to be > 0."""
    with pytest.raises(ValidationError, match="log_scale=True requires"):
        ParamSchemaEntry(name="x", type="float", bounds=(0.0, 1.0), log_scale=True)
    with pytest.raises(ValidationError, match="log_scale=True requires"):
        ParamSchemaEntry(name="x", type="float", bounds=(-1.0, 1.0), log_scale=True)
    # Valid: positive bounds with log_scale
    e = ParamSchemaEntry(name="lr", type="float", bounds=(1e-5, 1e-2), log_scale=True)
    assert e.log_scale is True


def test_param_schema_entry_categorical_rejects_duplicates() -> None:
    """Categorical SHALL require ≥2 *distinct* values, not just len ≥ 2."""
    with pytest.raises(ValidationError, match="duplicates"):
        ParamSchemaEntry(name="rnn", type="categorical", values=["lstm", "lstm"])
    with pytest.raises(ValidationError, match="duplicates"):
        ParamSchemaEntry(name="rnn", type="categorical", values=["a", "a", "a"])
    # Valid: distinct values
    e = ParamSchemaEntry(name="rnn", type="categorical", values=["lstm", "gru"])
    assert e.values == ["lstm", "gru"]


def test_hyperparam_schema_rejects_empty_list(tmp_path: Path) -> None:
    """An empty ``hyperparam_schema:`` list SHALL fail YAML load.

    Falls into a useless state otherwise — select_encoder would return
    a HyperparameterEncoder with genome_dim=0.
    """
    yaml_content = {
        "brain": {"name": "mlpppo", "config": {"sensory_modules": ["food_chemotaxis"]}},
        "hyperparam_schema": [],
    }
    yaml_path = tmp_path / "empty.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))
    with pytest.raises(ValidationError, match="empty"):
        load_simulation_config(str(yaml_path))


def test_hyperparam_schema_rejects_duplicate_names(tmp_path: Path) -> None:
    """Duplicate ``entry.name`` SHALL fail YAML load.

    Two entries pointing at the same brain-config field would silently
    let the second one's value win at decode (model_copy(update=) is
    last-write-wins on duplicate keys), making the first slot's evolved
    genome value invisible.
    """
    yaml_content = {
        "brain": {"name": "mlpppo", "config": {"sensory_modules": ["food_chemotaxis"]}},
        "hyperparam_schema": [
            {"name": "learning_rate", "type": "float", "bounds": [1e-5, 1e-2]},
            {"name": "learning_rate", "type": "float", "bounds": [1e-4, 1e-1]},
        ],
    }
    yaml_path = tmp_path / "dup.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))
    with pytest.raises(ValidationError, match="duplicate"):
        load_simulation_config(str(yaml_path))


# ---------------------------------------------------------------------------
# warm_start_path x hyperparam_schema interaction
# ---------------------------------------------------------------------------


def test_warm_start_rejects_arch_changing_schema_entries(tmp_path: Path) -> None:
    """``warm_start_path`` set + arch-changing schema entry → YAML load fails.

    A warm-start checkpoint encodes a fixed tensor layout.  Letting the
    genome evolve ``actor_hidden_dim`` (or any architecture field) would
    crash on ``load_state_dict`` shape mismatch deep inside the first
    fitness evaluation — which on a 100-genome x 20-generation campaign
    means hours of wall time before discovery.  Catching at YAML load
    time means the user sees the error before generation 1 begins.
    """
    fake_checkpoint = tmp_path / "anywhere.pt"
    yaml_content = {
        "brain": {"name": "mlpppo", "config": {"sensory_modules": ["food_chemotaxis"]}},
        "evolution": {
            "algorithm": "cmaes",
            "population_size": 4,
            "generations": 1,
            "episodes_per_eval": 1,
            "learn_episodes_per_eval": 1,
            "warm_start_path": str(fake_checkpoint),
        },
        "hyperparam_schema": [
            {"name": "actor_hidden_dim", "type": "int", "bounds": [32, 256]},
            {"name": "learning_rate", "type": "float", "bounds": [1e-5, 1e-2]},
        ],
    }
    yaml_path = tmp_path / "arch_warm.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))
    with pytest.raises(ValidationError, match="architecture-changing"):
        load_simulation_config(str(yaml_path))


def test_warm_start_with_non_arch_schema_passes(tmp_path: Path) -> None:
    """``warm_start_path`` set + non-arch schema → YAML load succeeds.

    The validator MUST only reject architecture-changing entries; the
    common case (warm-start fine-tuning of learning rates, gamma, etc.)
    has to load cleanly.
    """
    fake_checkpoint = tmp_path / "ok.pt"
    yaml_content = {
        "brain": {"name": "mlpppo", "config": {"sensory_modules": ["food_chemotaxis"]}},
        "evolution": {
            "algorithm": "cmaes",
            "population_size": 4,
            "generations": 1,
            "episodes_per_eval": 1,
            "learn_episodes_per_eval": 1,
            "warm_start_path": str(fake_checkpoint),
        },
        "hyperparam_schema": [
            {"name": "learning_rate", "type": "float", "bounds": [1e-5, 1e-2]},
            {"name": "gamma", "type": "float", "bounds": [0.9, 0.999]},
        ],
    }
    yaml_path = tmp_path / "ok_warm.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))
    cfg = load_simulation_config(str(yaml_path))
    assert cfg.evolution is not None
    assert cfg.evolution.warm_start_path == fake_checkpoint


def test_warm_start_unset_allows_arch_changing_entries(tmp_path: Path) -> None:
    """``warm_start_path`` unset → arch-changing entries SHALL load fine.

    Architecture-evolution without warm-start must remain supported —
    the validator must not over-restrict.
    """
    yaml_content = {
        "brain": {"name": "mlpppo", "config": {"sensory_modules": ["food_chemotaxis"]}},
        "hyperparam_schema": [
            {"name": "actor_hidden_dim", "type": "int", "bounds": [32, 256]},
            {"name": "lstm_hidden_dim", "type": "int", "bounds": [32, 128]},
        ],
    }
    yaml_path = tmp_path / "no_warm_arch.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))
    # MLPPPO doesn't have lstm_hidden_dim, so this fails on the OTHER
    # validator (unknown field).  That's expected and correct — the test
    # is about the warm-start validator NOT firing.  We intercept the
    # specific error to confirm.
    with pytest.raises(ValidationError, match="lstm_hidden_dim"):
        load_simulation_config(str(yaml_path))


# ---------------------------------------------------------------------------
# Inheritance validators
# ---------------------------------------------------------------------------


def test_inheritance_lamarckian_with_learn_episodes_zero_raises() -> None:
    """``inheritance: lamarckian`` + ``learn_episodes_per_eval: 0`` SHALL raise.

    Spec scenario "Inheritance requires a training phase".
    """
    with pytest.raises(ValidationError, match="learn_episodes_per_eval"):
        EvolutionConfig(inheritance="lamarckian", learn_episodes_per_eval=0)


def test_inheritance_lamarckian_with_warm_start_path_raises(tmp_path: Path) -> None:
    """``inheritance: lamarckian`` + ``warm_start_path`` SHALL raise.

    Spec scenario "Inheritance is mutually exclusive with static warm-start".
    """
    fake_path = tmp_path / "fake.pt"
    with pytest.raises(ValidationError, match="mutually exclusive"):
        EvolutionConfig(
            inheritance="lamarckian",
            learn_episodes_per_eval=10,
            warm_start_path=fake_path,
        )


def test_inheritance_lamarckian_without_hyperparam_schema_raises(tmp_path: Path) -> None:
    """``inheritance: lamarckian`` requires ``hyperparam_schema`` SHALL raise.

    Spec scenario "Inheritance requires hyperparameter encoding".
    """
    yaml_content = {
        "max_steps": 100,
        "brain": {"name": "mlpppo", "config": {}},
        "evolution": {
            "inheritance": "lamarckian",
            "learn_episodes_per_eval": 10,
            "population_size": 4,
        },
    }
    yaml_path = tmp_path / "no_schema.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))
    with pytest.raises(ValidationError, match="no hyperparam_schema is set"):
        load_simulation_config(str(yaml_path))


def test_inheritance_lamarckian_with_architecture_field_in_schema_raises(
    tmp_path: Path,
) -> None:
    """``inheritance: lamarckian`` + arch-changing schema field SHALL raise.

    Spec scenario "Inheritance incompatible with architecture-changing schema entries".
    """
    yaml_content = {
        "max_steps": 100,
        "brain": {"name": "mlpppo", "config": {}},
        "evolution": {
            "inheritance": "lamarckian",
            "learn_episodes_per_eval": 10,
            "population_size": 4,
        },
        "hyperparam_schema": [
            {"name": "actor_hidden_dim", "type": "int", "bounds": [32, 128]},
        ],
    }
    yaml_path = tmp_path / "arch_schema.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))
    with pytest.raises(ValidationError, match="architecture-changing entries"):
        load_simulation_config(str(yaml_path))


def test_inheritance_elite_count_exceeds_population_size_raises() -> None:
    """``inheritance_elite_count > population_size`` SHALL raise.

    This rule is enforced even with ``inheritance: none`` so that
    future strategies can lift the ``!= 1`` restriction without
    re-exposing this trivially-impossible combination.  Use
    ``inheritance: none`` with elite_count=20 and population=12 so we
    exercise the population-size check rather than the != 1 check.
    """
    with pytest.raises(ValidationError, match=r"exceeds evolution\.population_size"):
        EvolutionConfig(
            inheritance="none",
            inheritance_elite_count=20,
            population_size=12,
        )


def test_inheritance_lamarckian_elite_count_not_one_raises() -> None:
    """``inheritance: lamarckian`` + ``inheritance_elite_count != 1`` SHALL raise.

    Single-elite-broadcast is currently the only supported variant;
    the field permits values >1 structurally so a future strategy can
    lift just this rule.
    """
    with pytest.raises(ValidationError, match="MUST be 1"):
        EvolutionConfig(
            inheritance="lamarckian",
            learn_episodes_per_eval=10,
            inheritance_elite_count=2,
            population_size=12,
        )


def test_inheritance_lamarckian_happy_path_loads(tmp_path: Path) -> None:
    """A well-formed lamarckian YAML SHALL load without error."""
    yaml_content = {
        "max_steps": 100,
        "brain": {"name": "mlpppo", "config": {}},
        "evolution": {
            "algorithm": "tpe",
            "inheritance": "lamarckian",
            "inheritance_elite_count": 1,
            "learn_episodes_per_eval": 10,
            "eval_episodes_per_eval": 5,
            "population_size": 4,
            "generations": 3,
        },
        "hyperparam_schema": [
            {"name": "learning_rate", "type": "float", "bounds": [1e-5, 1e-2], "log_scale": True},
        ],
    }
    yaml_path = tmp_path / "happy.yml"
    yaml_path.write_text(yaml.safe_dump(yaml_content))
    cfg = load_simulation_config(str(yaml_path))
    assert cfg.evolution is not None
    assert cfg.evolution.inheritance == "lamarckian"
    assert cfg.evolution.inheritance_elite_count == 1


def test_inheritance_defaults_preserve_pre_m3_behaviour() -> None:
    """An ``EvolutionConfig`` with default fields SHALL be ``inheritance: none``."""
    cfg = EvolutionConfig()
    assert cfg.inheritance == "none"
    assert cfg.inheritance_elite_count == 1
