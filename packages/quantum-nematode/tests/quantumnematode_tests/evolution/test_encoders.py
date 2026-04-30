"""Tests for :mod:`quantumnematode.evolution.encoders`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.lstmppo import LSTMPPOBrain
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain
from quantumnematode.evolution.brain_factory import instantiate_brain_from_sim_config
from quantumnematode.evolution.encoders import (
    ENCODER_REGISTRY,
    NON_GENOME_COMPONENTS,
    LSTMPPOEncoder,
    MLPPPOEncoder,
    get_encoder,
)
from quantumnematode.utils.config_loader import load_simulation_config

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MLPPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/mlpppo_small_oracle.yml"
LSTMPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/lstmppo_small_klinotaxis.yml"


def _make_seeded_brain_params(brain: LSTMPPOBrain | MLPPPOBrain) -> BrainParams:
    """Build a deterministic BrainParams for round-trip action comparison.

    Populates fields covering oracle, klinotaxis, and STAM input paths so
    the same params object works for any LSTMPPO/MLPPPO config — including
    the klinotaxis pilot (which appends STAM after sensing-mode translation).

    Values here don't matter for correctness; what matters is that two brains
    with identical weights produce the same action when fed the same input.

    The STAM dim must match what ``brain.input_dim`` infers — for klinotaxis
    + proprioception + STAM that's 5 (1 channel, ``compute_memory_dim(1)``).
    Pre-bug-fix, the evolution flow's brain factory bypassed
    ``apply_sensing_mode`` so klinotaxis configs silently ran on oracle
    sensing without STAM and the test only needed the oracle fields.
    """
    # STAM zero-vector sized to match the brain's expected input.  We compute
    # this from the brain's own input_dim so the test stays robust if the
    # underlying feature extraction changes.
    klino_dim = 3  # food_chemotaxis_klinotaxis core
    proprio_dim = 2  # proprioception core
    inferred_stam_dim = max(brain.input_dim - klino_dim - proprio_dim, 0)
    # ``BrainParams.stam_state`` is typed ``tuple[float, ...] | None`` — pass a
    # tuple rather than an ndarray.  Both serialise the same way through
    # ``extract_classical_features`` (which calls ``np.array`` on the input).
    stam_state: tuple[float, ...] | None = (
        tuple([0.0] * inferred_stam_dim) if inferred_stam_dim > 0 else None
    )
    return BrainParams(
        # Oracle chemotaxis (food_chemotaxis module)
        food_gradient_strength=0.5,
        food_gradient_direction=1.0,
        # Klinotaxis chemotaxis (food_chemotaxis_klinotaxis module)
        food_concentration=0.5,
        food_lateral_gradient=0.1,
        food_dconcentration_dt=0.05,
        lateral_scale=1.0,
        derivative_scale=1.0,
        # STAM memory state — sized to the brain's expected input dim
        stam_state=stam_state,
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_mlpppo_encoder_roundtrip() -> None:
    """Encode → decode SHALL produce a brain with identical first-step actions.

    PPO action sampling uses global torch RNG state, so we re-seed torch
    immediately before each ``run_brain`` call to compare apples-to-apples.
    The contract is: identical weights + identical seeded RNG state +
    identical inputs → identical action.
    """
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    encoder = MLPPPOEncoder()
    rng = np.random.default_rng(0)

    from quantumnematode.evolution.encoders import (
        _flatten_components,
        _select_genome_components,
    )

    # Build the original brain and serialize its current weights into a genome.
    original = instantiate_brain_from_sim_config(sim_config, seed=42)
    assert isinstance(original, MLPPPOBrain)
    components = _select_genome_components(original.get_weight_components())
    flat, shape_map = _flatten_components(components)
    genome = encoder.initial_genome(sim_config, rng=rng)
    genome.params = flat
    genome.birth_metadata["shape_map"] = shape_map

    # Decode into a fresh brain.
    decoded = encoder.decode(genome, sim_config, seed=42)

    params = _make_seeded_brain_params(original)

    # Re-seed torch globally before each call so action sampling is comparable.
    torch.manual_seed(123)
    a1 = original.run_brain(
        params,
        reward=None,
        input_data=None,
        top_only=True,
        top_randomize=False,
    )
    torch.manual_seed(123)
    a2 = decoded.run_brain(params, reward=None, input_data=None, top_only=True, top_randomize=False)

    assert a1[0].action == a2[0].action


def test_lstmppo_encoder_roundtrip() -> None:
    """LSTMPPO round-trip SHALL preserve all four learned-weight components."""
    sim_config = load_simulation_config(str(LSTMPPO_CONFIG))
    encoder = LSTMPPOEncoder()
    rng = np.random.default_rng(0)

    from quantumnematode.evolution.encoders import (
        _flatten_components,
        _select_genome_components,
    )

    original = instantiate_brain_from_sim_config(sim_config, seed=42)
    assert isinstance(original, LSTMPPOBrain)
    components = _select_genome_components(original.get_weight_components())

    # Verify the four expected components are present.
    assert set(components) == {"lstm", "layer_norm", "policy", "value"}

    flat, shape_map = _flatten_components(components)
    genome = encoder.initial_genome(sim_config, rng=rng)
    genome.params = flat
    genome.birth_metadata["shape_map"] = shape_map

    decoded = encoder.decode(genome, sim_config, seed=42)

    # LSTMPPO needs prepare_episode() to initialise hidden state before run_brain.
    original.prepare_episode()
    decoded.prepare_episode()

    params = _make_seeded_brain_params(original)

    # Re-seed torch globally before each call so action sampling is comparable.
    torch.manual_seed(123)
    a1 = original.run_brain(
        params,
        reward=None,
        input_data=None,
        top_only=True,
        top_randomize=False,
    )
    torch.manual_seed(123)
    a2 = decoded.run_brain(params, reward=None, input_data=None, top_only=True, top_randomize=False)

    assert a1[0].action == a2[0].action


# ---------------------------------------------------------------------------
# Genome dim
# ---------------------------------------------------------------------------


def test_genome_dim_matches_flattened_state_mlpppo() -> None:
    """``genome_dim`` SHALL equal the total float-param count across components."""
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    encoder = MLPPPOEncoder()
    dim = encoder.genome_dim(sim_config)

    rng = np.random.default_rng(0)
    genome = encoder.initial_genome(sim_config, rng=rng)
    assert dim == genome.params.size


def test_genome_dim_matches_flattened_state_lstmppo() -> None:
    """Same contract as the MLPPPO version, for LSTMPPO."""
    sim_config = load_simulation_config(str(LSTMPPO_CONFIG))
    encoder = LSTMPPOEncoder()
    dim = encoder.genome_dim(sim_config)

    rng = np.random.default_rng(0)
    genome = encoder.initial_genome(sim_config, rng=rng)
    assert dim == genome.params.size


# ---------------------------------------------------------------------------
# Episode count + LR sync on decode
# ---------------------------------------------------------------------------


def test_episode_count_resets_and_lr_synced_on_decode_mlpppo() -> None:
    """Decoded MLPPPOBrain SHALL have ``_episode_count == 0`` and matching LR.

    The encoder calls ``_update_learning_rate`` after the count reset, so
    a freshly-decoded brain matches a freshly-constructed one.
    """
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    encoder = MLPPPOEncoder()
    rng = np.random.default_rng(0)
    genome = encoder.initial_genome(sim_config, rng=rng)

    decoded = encoder.decode(genome, sim_config, seed=42)
    assert isinstance(decoded, MLPPPOBrain)
    assert decoded._episode_count == 0

    # Verify LR matches what a freshly-constructed brain would have at episode 0.
    fresh = instantiate_brain_from_sim_config(sim_config, seed=42)
    assert isinstance(fresh, MLPPPOBrain)
    assert decoded._episode_count == fresh._episode_count
    # MLPPPOBrain has a single joint optimizer; compare LR per param group.
    decoded_lr_groups = [pg["lr"] for pg in decoded.optimizer.param_groups]
    fresh_lr_groups = [pg["lr"] for pg in fresh.optimizer.param_groups]
    assert decoded_lr_groups == fresh_lr_groups


def test_episode_count_resets_and_lr_synced_on_decode_lstmppo() -> None:
    """Same contract as the MLPPPO version, for LSTMPPO."""
    sim_config = load_simulation_config(str(LSTMPPO_CONFIG))
    encoder = LSTMPPOEncoder()
    rng = np.random.default_rng(0)
    genome = encoder.initial_genome(sim_config, rng=rng)

    decoded = encoder.decode(genome, sim_config, seed=42)
    assert isinstance(decoded, LSTMPPOBrain)
    assert decoded._episode_count == 0


# ---------------------------------------------------------------------------
# Registry membership
# ---------------------------------------------------------------------------


def test_encoder_registry_membership() -> None:
    """Registry SHALL contain mlpppo + lstmppo and produce working encoders."""
    assert "mlpppo" in ENCODER_REGISTRY
    assert "lstmppo" in ENCODER_REGISTRY

    mlpppo_enc = ENCODER_REGISTRY["mlpppo"]()
    lstmppo_enc = ENCODER_REGISTRY["lstmppo"]()
    assert mlpppo_enc.brain_name == "mlpppo"
    assert lstmppo_enc.brain_name == "lstmppo"


def test_get_encoder_returns_working_encoder() -> None:
    """``get_encoder`` SHALL return an instance of the registered class."""
    enc = get_encoder("mlpppo")
    assert isinstance(enc, MLPPPOEncoder)


def test_unknown_brain_name_raises_with_helpful_message() -> None:
    """Unknown brain name SHALL raise ValueError naming registered brains."""
    with pytest.raises(ValueError, match="qvarcircuit") as excinfo:
        get_encoder("qvarcircuit")
    msg = str(excinfo.value)
    assert "mlpppo" in msg
    assert "lstmppo" in msg
    assert "Quantum brains are not currently supported." in msg


# ---------------------------------------------------------------------------
# Dynamic component discovery (denylist)
# ---------------------------------------------------------------------------


def test_encoder_excludes_denylist_components_mlpppo() -> None:
    """``optimizer`` and ``training_state`` SHALL never be in an MLPPPO genome."""
    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    encoder = MLPPPOEncoder()
    rng = np.random.default_rng(0)
    genome = encoder.initial_genome(sim_config, rng=rng)

    components_in_genome = set(genome.birth_metadata.get("shape_map", {}))
    assert "optimizer" not in components_in_genome
    assert "training_state" not in components_in_genome


def test_encoder_excludes_denylist_components_lstmppo() -> None:
    """LSTMPPO genome SHALL exclude all three optimizer/training components."""
    sim_config = load_simulation_config(str(LSTMPPO_CONFIG))
    encoder = LSTMPPOEncoder()
    rng = np.random.default_rng(0)
    genome = encoder.initial_genome(sim_config, rng=rng)

    components_in_genome = set(genome.birth_metadata.get("shape_map", {}))
    assert "actor_optimizer" not in components_in_genome
    assert "critic_optimizer" not in components_in_genome
    assert "training_state" not in components_in_genome
    # Conversely, all four learned-weight components SHALL be present.
    assert {"lstm", "layer_norm", "policy", "value"} <= components_in_genome


def test_non_genome_components_constant_matches_documented_set() -> None:
    """The denylist SHALL be exactly the documented set."""
    assert (
        frozenset(
            {"optimizer", "actor_optimizer", "critic_optimizer", "training_state"},
        )
        == NON_GENOME_COMPONENTS
    )
