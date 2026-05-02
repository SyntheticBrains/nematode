"""Tests for :class:`HyperparameterEncoder` and the dispatch helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain
from quantumnematode.evolution.encoders import (
    ENCODER_REGISTRY,
    HyperparameterEncoder,
    LSTMPPOEncoder,
    MLPPPOEncoder,
    build_birth_metadata,
    select_encoder,
)
from quantumnematode.evolution.genome import Genome
from quantumnematode.utils.config_loader import (
    ParamSchemaEntry,
    SimulationConfig,
    load_simulation_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mlpppo_sim_config(schema: list[ParamSchemaEntry]) -> SimulationConfig:
    """Build a minimal SimulationConfig with a hyperparam_schema for testing."""
    yaml_dict = {
        "brain": {
            "name": "mlpppo",
            "config": {"sensory_modules": ["food_chemotaxis"]},
        },
        "max_steps": 100,
    }
    cfg = SimulationConfig.model_validate(yaml_dict)
    return cfg.model_copy(update={"hyperparam_schema": schema})


# ---------------------------------------------------------------------------
# build_birth_metadata
# ---------------------------------------------------------------------------


def test_build_birth_metadata_with_schema_returns_dump() -> None:
    """``build_birth_metadata`` SHALL serialise schema entries to plain dicts."""
    schema = [
        ParamSchemaEntry(name="actor_hidden_dim", type="int", bounds=(32, 256)),
        ParamSchemaEntry(
            name="learning_rate",
            type="float",
            bounds=(1e-5, 1e-2),
            log_scale=True,
        ),
    ]
    sim_config = _make_mlpppo_sim_config(schema)
    md = build_birth_metadata(sim_config)
    assert "param_schema" in md
    assert isinstance(md["param_schema"], list)
    assert len(md["param_schema"]) == 2
    # Each entry SHALL be a plain dict, not a ParamSchemaEntry
    for entry in md["param_schema"]:
        assert isinstance(entry, dict)
        assert "name" in entry
        assert "type" in entry
    assert md["param_schema"][0]["name"] == "actor_hidden_dim"
    assert md["param_schema"][1]["log_scale"] is True


def test_build_birth_metadata_no_schema_returns_empty() -> None:
    """When ``hyperparam_schema is None``, the helper SHALL return an empty dict."""
    cfg = SimulationConfig.model_validate({"brain": {"name": "mlpppo", "config": {}}})
    assert cfg.hyperparam_schema is None
    md = build_birth_metadata(cfg)
    assert md == {}


# ---------------------------------------------------------------------------
# HyperparameterEncoder — registry / protocol membership
# ---------------------------------------------------------------------------


def test_hyperparam_encoder_not_in_brain_registry() -> None:
    """``HyperparameterEncoder`` is brain-agnostic; SHALL NOT pollute registry."""
    assert "hyperparam" not in ENCODER_REGISTRY
    # Registry SHALL only contain real brain-keyed encoder names
    assert set(ENCODER_REGISTRY.keys()) == {"mlpppo", "lstmppo"}
    # Encoder SHALL still be importable for programmatic callers
    enc = HyperparameterEncoder()
    assert enc.brain_name == ""  # empty string for brain-agnosticism


# ---------------------------------------------------------------------------
# initial_genome
# ---------------------------------------------------------------------------


def test_initial_genome_dim_matches_schema_length() -> None:
    """``initial_genome`` SHALL return params of length len(schema)."""
    schema = [
        ParamSchemaEntry(name="actor_hidden_dim", type="int", bounds=(32, 256)),
        ParamSchemaEntry(name="learning_rate", type="float", bounds=(1e-5, 1e-2)),
        ParamSchemaEntry(name="num_epochs", type="int", bounds=(1, 8)),
    ]
    sim_config = _make_mlpppo_sim_config(schema)
    enc = HyperparameterEncoder()
    rng = np.random.default_rng(42)
    genome = enc.initial_genome(sim_config, rng=rng)
    assert genome.params.shape == (3,)
    # birth_metadata SHALL carry the schema
    assert "param_schema" in genome.birth_metadata
    assert len(genome.birth_metadata["param_schema"]) == 3


def test_initial_genome_no_schema_raises() -> None:
    """No schema → clear error, not silent garbage."""
    cfg = SimulationConfig.model_validate({"brain": {"name": "mlpppo", "config": {}}})
    enc = HyperparameterEncoder()
    with pytest.raises(ValueError, match="hyperparam_schema"):
        enc.initial_genome(cfg, rng=np.random.default_rng(0))


def test_initial_genome_float_log_scale_samples_in_log_space() -> None:
    """Float slot with ``log_scale=True`` SHALL sample in log-space.

    The distribution is uniform-in-log, not uniform-in-linear.
    """
    schema = [
        ParamSchemaEntry(
            name="learning_rate",
            type="float",
            bounds=(1e-5, 1e-2),
            log_scale=True,
        ),
    ]
    sim_config = _make_mlpppo_sim_config(schema)
    enc = HyperparameterEncoder()
    rng = np.random.default_rng(42)
    samples = np.array(
        [enc.initial_genome(sim_config, rng=rng).params[0] for _ in range(50)],
    )
    # Genome value is in log-space, so it should be in [log(1e-5), log(1e-2)]
    assert samples.min() >= np.log(1e-5) - 1e-6
    assert samples.max() <= np.log(1e-2) + 1e-6


# ---------------------------------------------------------------------------
# decode — per-type transforms
# ---------------------------------------------------------------------------


def test_decode_round_trip_float() -> None:
    """Decode SHALL apply the genome value to BrainConfig within float tolerance."""
    schema = [ParamSchemaEntry(name="learning_rate", type="float", bounds=(1e-5, 1e-2))]
    sim_config = _make_mlpppo_sim_config(schema)
    enc = HyperparameterEncoder()

    genome = Genome(
        params=np.array([0.001], dtype=np.float32),
        genome_id="test",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )
    brain = cast("MLPPPOBrain", enc.decode(genome, sim_config))
    assert isinstance(brain, MLPPPOBrain)
    # MLPPPOBrain stores learning_rate on the config
    assert brain.config.learning_rate == pytest.approx(0.001, rel=1e-5)


def test_decode_round_trip_int() -> None:
    """Int slot SHALL clip to bounds and round."""
    schema = [ParamSchemaEntry(name="actor_hidden_dim", type="int", bounds=(32, 256))]
    sim_config = _make_mlpppo_sim_config(schema)
    enc = HyperparameterEncoder()

    genome = Genome(
        params=np.array([127.6], dtype=np.float32),
        genome_id="test",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )
    brain = cast("MLPPPOBrain", enc.decode(genome, sim_config))
    assert isinstance(brain, MLPPPOBrain)
    # 127.6 → round(127.6) = 128
    assert brain.config.actor_hidden_dim == 128


def test_decode_int_clips_to_bounds() -> None:
    """Out-of-bounds int values SHALL clip rather than crash."""
    schema = [ParamSchemaEntry(name="actor_hidden_dim", type="int", bounds=(32, 256))]
    sim_config = _make_mlpppo_sim_config(schema)
    enc = HyperparameterEncoder()

    # Below low bound
    genome_low = Genome(
        params=np.array([10.0], dtype=np.float32),
        genome_id="t1",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )
    brain_low = cast("MLPPPOBrain", enc.decode(genome_low, sim_config))
    assert brain_low.config.actor_hidden_dim == 32

    # Above high bound
    genome_high = Genome(
        params=np.array([1000.0], dtype=np.float32),
        genome_id="t2",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )
    brain_high = cast("MLPPPOBrain", enc.decode(genome_high, sim_config))
    assert brain_high.config.actor_hidden_dim == 256


def test_decode_log_scale_applies_exp() -> None:
    """``log_scale=True`` SHALL apply exp() to the genome value during decode."""
    schema = [
        ParamSchemaEntry(
            name="learning_rate",
            type="float",
            bounds=(1e-5, 1e-2),
            log_scale=True,
        ),
    ]
    sim_config = _make_mlpppo_sim_config(schema)
    enc = HyperparameterEncoder()

    # Genome value -6.9 is approximately log(1e-3); decode SHALL produce ~1e-3
    log_value = np.log(1e-3)
    genome = Genome(
        params=np.array([log_value], dtype=np.float32),
        genome_id="test",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )
    brain = cast("MLPPPOBrain", enc.decode(genome, sim_config))
    assert brain.config.learning_rate == pytest.approx(1e-3, rel=1e-3)


def test_decode_one_bool_threshold() -> None:
    """``_decode_one`` for bool SHALL apply ``value > 0.0``."""
    enc = HyperparameterEncoder()
    bool_entry = {"name": "x", "type": "bool"}
    assert enc._decode_one(bool_entry, 0.5) is True
    assert enc._decode_one(bool_entry, -0.1) is False
    assert enc._decode_one(bool_entry, 0.0) is False


def test_decode_one_categorical_indexing() -> None:
    """Categorical decode SHALL use ``int(round(value)) mod len(values)``."""
    enc = HyperparameterEncoder()
    cat_entry = {"name": "rnn", "type": "categorical", "values": ["lstm", "gru"]}
    # 0.0 → 0 → "lstm"
    assert enc._decode_one(cat_entry, 0.0) == "lstm"
    # 1.0 → 1 → "gru"
    assert enc._decode_one(cat_entry, 1.0) == "gru"
    # 1.5 → round(1.5) = 2 → 2 mod 2 = 0 → "lstm"
    # (Python uses banker's rounding; 1.5 rounds to 2 not 2 in a pinch)
    # Actually, Python's round(1.5) = 2 in Python 3.  2 mod 2 = 0.
    assert enc._decode_one(cat_entry, 1.5) == "lstm"
    # Out-of-bounds wraps via mod: 2.0 → 2 mod 2 = 0
    assert enc._decode_one(cat_entry, 2.0) == "lstm"


def test_decode_unspecified_brain_fields_unchanged() -> None:
    """Decode SHALL only patch fields named in the schema; others unchanged."""
    # Set actor_hidden_dim explicitly in YAML config; then evolve only learning_rate
    yaml_dict = {
        "brain": {
            "name": "mlpppo",
            "config": {
                "actor_hidden_dim": 96,
                "sensory_modules": ["food_chemotaxis"],
            },
        },
    }
    cfg = SimulationConfig.model_validate(yaml_dict)
    sim_config = cfg.model_copy(
        update={
            "hyperparam_schema": [
                ParamSchemaEntry(name="learning_rate", type="float", bounds=(1e-5, 1e-2)),
            ],
        },
    )
    enc = HyperparameterEncoder()
    genome = Genome(
        params=np.array([0.005], dtype=np.float32),
        genome_id="test",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )
    brain = cast("MLPPPOBrain", enc.decode(genome, sim_config))
    # learning_rate evolved
    assert brain.config.learning_rate == pytest.approx(0.005, rel=1e-5)
    # actor_hidden_dim NOT in schema → keeps the YAML-set value
    assert brain.config.actor_hidden_dim == 96


def test_decode_does_not_load_weights() -> None:
    """Decode SHALL NOT call WeightPersistence.load_weight_components.

    Hyperparameter genomes carry no weights — every evaluation builds a
    fresh brain from scratch.
    """
    schema = [ParamSchemaEntry(name="learning_rate", type="float", bounds=(1e-5, 1e-2))]
    sim_config = _make_mlpppo_sim_config(schema)
    enc = HyperparameterEncoder()
    genome = Genome(
        params=np.array([0.001], dtype=np.float32),
        genome_id="test",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )
    with patch(
        "quantumnematode.brain.weights.WeightPersistence.load_weight_components",
    ) as mock_load:
        enc.decode(genome, sim_config)
        mock_load.assert_not_called()


def test_decode_does_not_mutate_input_sim_config() -> None:
    """Decode SHALL NOT mutate the input sim_config in place."""
    yaml_dict = {
        "brain": {
            "name": "mlpppo",
            "config": {
                "learning_rate": 0.0003,
                "sensory_modules": ["food_chemotaxis"],
            },
        },
    }
    cfg = SimulationConfig.model_validate(yaml_dict)
    sim_config = cfg.model_copy(
        update={
            "hyperparam_schema": [
                ParamSchemaEntry(name="learning_rate", type="float", bounds=(1e-5, 1e-2)),
            ],
        },
    )
    assert sim_config.brain is not None
    original_lr = sim_config.brain.config.learning_rate  # type: ignore[union-attr]
    enc = HyperparameterEncoder()
    genome = Genome(
        params=np.array([0.005], dtype=np.float32),
        genome_id="test",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )
    enc.decode(genome, sim_config)
    # Original sim_config SHALL retain its original learning_rate
    assert sim_config.brain.config.learning_rate == original_lr  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# genome_dim
# ---------------------------------------------------------------------------


def test_genome_dim_matches_schema_length_no_brain_constructed() -> None:
    """``genome_dim`` SHALL return ``len(schema)`` without constructing a brain."""
    schema = [
        ParamSchemaEntry(name="actor_hidden_dim", type="int", bounds=(32, 256)),
        ParamSchemaEntry(name="learning_rate", type="float", bounds=(1e-5, 1e-2)),
    ]
    sim_config = _make_mlpppo_sim_config(schema)
    enc = HyperparameterEncoder()
    # Patch instantiate_brain_from_sim_config to detect any spurious construction
    with patch(
        "quantumnematode.evolution.encoders.instantiate_brain_from_sim_config",
    ) as mock_instantiate:
        n = enc.genome_dim(sim_config)
        mock_instantiate.assert_not_called()
    assert n == 2


# ---------------------------------------------------------------------------
# genome_stds — per-parameter standard deviations
# ---------------------------------------------------------------------------


def test_genome_stds_scales_to_bound_widths() -> None:
    """``genome_stds`` SHALL return std = bound-width / 6 for each slot.

    With ±3 stds at sigma=1.0 spanning the full bound range, the optimiser
    can explore each dimension proportionally without saturating tight
    bounds or under-exploring wide ones.
    """
    schema = [
        # Linear float, range 0.099
        ParamSchemaEntry(name="gamma", type="float", bounds=(0.9, 0.999)),
        # Log-scale float, log-range 6.908
        ParamSchemaEntry(
            name="learning_rate",
            type="float",
            bounds=(1e-5, 1e-2),
            log_scale=True,
        ),
        # Int, range 224
        ParamSchemaEntry(name="actor_hidden_dim", type="int", bounds=(32, 256)),
    ]
    sim_config = _make_mlpppo_sim_config(schema)
    enc = HyperparameterEncoder()
    stds = enc.genome_stds(sim_config)
    assert stds is not None
    assert len(stds) == 3
    # Linear float
    assert stds[0] == pytest.approx((0.999 - 0.9) / 6, rel=1e-6)
    # Log-scale float — std in log-space
    assert stds[1] == pytest.approx((np.log(1e-2) - np.log(1e-5)) / 6, rel=1e-6)
    # Int — treated as continuous
    assert stds[2] == pytest.approx((256 - 32) / 6, rel=1e-6)


def test_genome_stds_bool_is_unit() -> None:
    """``bool`` slots SHALL get std=1.0 (covers the ±1 range)."""
    schema = [ParamSchemaEntry(name="feature_gating", type="bool")]
    sim_config = _make_mlpppo_sim_config(schema)
    enc = HyperparameterEncoder()
    stds = enc.genome_stds(sim_config)
    assert stds == [1.0]


def test_genome_stds_categorical_scales_to_n_values() -> None:
    """Categorical slots SHALL get std = max(1, len(values) / 6) so ±3 stds spans bins."""
    schema_small = [
        ParamSchemaEntry(name="rnn_type", type="categorical", values=["lstm", "gru"]),
    ]
    sim_config_small = _make_mlpppo_sim_config(schema_small)
    enc = HyperparameterEncoder()
    stds_small = enc.genome_stds(sim_config_small)
    # 2 values, so max(1.0, 2/6) = 1.0
    assert stds_small == [1.0]

    # Hypothetical schema with 12 values: max(1.0, 12/6) = 2.0
    schema_big = [
        ParamSchemaEntry(
            name="some_categorical",
            type="categorical",
            values=[f"v{i}" for i in range(12)],
        ),
    ]
    sim_config_big = _make_mlpppo_sim_config(schema_big)
    stds_big = enc.genome_stds(sim_config_big)
    assert stds_big == [2.0]


def test_genome_stds_no_schema_raises() -> None:
    """No schema → clear error from genome_stds."""
    cfg = SimulationConfig.model_validate({"brain": {"name": "mlpppo", "config": {}}})
    enc = HyperparameterEncoder()
    with pytest.raises(ValueError, match="hyperparam_schema"):
        enc.genome_stds(cfg)


def test_weight_encoder_genome_stds_returns_none() -> None:
    """Weight encoders SHALL return None — no per-param scaling."""
    cfg = SimulationConfig.model_validate(
        {
            "brain": {
                "name": "mlpppo",
                "config": {"sensory_modules": ["food_chemotaxis"]},
            },
        },
    )
    enc = MLPPPOEncoder()
    assert enc.genome_stds(cfg) is None


# ---------------------------------------------------------------------------
# Pickling — schemas travel to workers
# ---------------------------------------------------------------------------


def test_genome_pickles_with_param_schema() -> None:
    """A Genome with ``birth_metadata['param_schema']`` SHALL pickle round-trip.

    Workers receive the schema this way: plain dicts in birth_metadata,
    not Pydantic instances, so worker decode has no Pydantic-import
    dependency.
    """
    schema = [
        ParamSchemaEntry(name="actor_hidden_dim", type="int", bounds=(32, 256)),
        ParamSchemaEntry(
            name="learning_rate",
            type="float",
            bounds=(1e-5, 1e-2),
            log_scale=True,
        ),
    ]
    sim_config = _make_mlpppo_sim_config(schema)

    genome = Genome(
        params=np.array([100.0, -3.0], dtype=np.float32),
        genome_id="g1",
        parent_ids=["p1"],
        generation=5,
        birth_metadata=build_birth_metadata(sim_config),
    )
    payload = pickle.dumps(genome)
    restored = pickle.loads(payload)  # noqa: S301
    assert restored.genome_id == "g1"
    assert "param_schema" in restored.birth_metadata
    assert len(restored.birth_metadata["param_schema"]) == 2
    assert restored.birth_metadata["param_schema"][0]["name"] == "actor_hidden_dim"
    np.testing.assert_array_equal(restored.params, genome.params)


# ---------------------------------------------------------------------------
# select_encoder
# ---------------------------------------------------------------------------


def test_select_encoder_with_hyperparam_schema_returns_hyperparam_encoder() -> None:
    """``select_encoder`` SHALL return ``HyperparameterEncoder`` when schema set."""
    schema = [ParamSchemaEntry(name="learning_rate", type="float", bounds=(1e-5, 1e-2))]
    sim_config = _make_mlpppo_sim_config(schema)
    enc = select_encoder(sim_config)
    assert isinstance(enc, HyperparameterEncoder)


def test_select_encoder_without_schema_falls_back_to_brain_encoder() -> None:
    """Without ``hyperparam_schema``, ``select_encoder`` SHALL return brain-keyed encoder."""
    cfg = SimulationConfig.model_validate({"brain": {"name": "mlpppo", "config": {}}})
    enc = select_encoder(cfg)
    assert isinstance(enc, MLPPPOEncoder)
    # And lstmppo
    cfg_lstm = SimulationConfig.model_validate({"brain": {"name": "lstmppo", "config": {}}})
    enc_lstm = select_encoder(cfg_lstm)
    assert isinstance(enc_lstm, LSTMPPOEncoder)


def test_select_encoder_brain_without_weight_encoder_works_under_hyperparam_schema() -> None:
    """A brain without a weight encoder SHALL still dispatch under hyperparam_schema.

    ``BRAIN_CONFIG_MAP`` includes brains like ``qvarcircuit`` that have no
    weight encoder in ``ENCODER_REGISTRY``.  Under ``hyperparam_schema``,
    select_encoder bypasses the registry — that's the point of
    brain-agnostic dispatch.
    """
    # qvarcircuit is in BRAIN_CONFIG_MAP but NOT in ENCODER_REGISTRY
    yaml_dict = {"brain": {"name": "qvarcircuit", "config": {}}}
    cfg = SimulationConfig.model_validate(yaml_dict)
    sim_config = cfg.model_copy(
        update={
            "hyperparam_schema": [
                # qvarcircuit has a `qubits` field
                ParamSchemaEntry(name="qubits", type="int", bounds=(2, 8)),
            ],
        },
    )
    enc = select_encoder(sim_config)
    assert isinstance(enc, HyperparameterEncoder)


def test_select_encoder_brain_without_weight_encoder_raises_under_weight_evolution() -> None:
    """Same brain WITHOUT hyperparam_schema SHALL raise (no weight encoder)."""
    cfg = SimulationConfig.model_validate(
        {"brain": {"name": "qvarcircuit", "config": {}}},
    )
    with pytest.raises(ValueError, match="qvarcircuit") as exc_info:
        select_encoder(cfg)
    # Error SHALL mention registered brain names
    msg = str(exc_info.value)
    assert "mlpppo" in msg


def test_select_encoder_pilot_config_loads_and_dispatches() -> None:
    """Loading the pilot YAML and dispatching SHALL produce a HyperparameterEncoder."""
    pilot_path = PROJECT_ROOT / "configs/evolution/hyperparam_mlpppo_pilot.yml"
    if not pilot_path.exists():
        pytest.skip("Pilot YAML not yet created in this PR slice")
    cfg = load_simulation_config(str(pilot_path))
    assert cfg.hyperparam_schema is not None
    enc = select_encoder(cfg)
    assert isinstance(enc, HyperparameterEncoder)
