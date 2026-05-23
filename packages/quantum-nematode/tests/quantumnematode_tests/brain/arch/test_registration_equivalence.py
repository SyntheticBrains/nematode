"""Equivalence tests for registry-based brain instantiation.

For each architecture migrated behind the brain plugin registry, this
module verifies that ``instantiate_brain(name, config, ...)`` produces a
brain with identical initial state and identical forward-pass outputs to
the brain constructed via the direct ``BrainCls(config=...)`` path. With
pinned seeds, both code paths must produce byte-identical parameter
tensors and outputs — this is the contract that lets out-of-tree callers
swap to the registry without behavioural regression.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import numpy as np
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch._registry import instantiate_brain
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.lstmppo import LSTMPPOBrain, LSTMPPOBrainConfig
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName

_SHARED_SEED = 2026
_FIXED_PARAMS_REWARD = 0.5


def _make_brain_params() -> BrainParams:
    """Construct a deterministic BrainParams sample for forward-pass comparisons."""
    return BrainParams(
        food_gradient_strength=0.42,
        food_gradient_direction=0.13,
        predator_gradient_strength=0.0,
        predator_gradient_direction=0.0,
    )


def _compare_actor_critic_weights(a: torch.nn.Module, b: torch.nn.Module) -> None:
    """Assert that two networks have byte-identical parameter tensors."""
    a_params = list(a.parameters())
    b_params = list(b.parameters())
    assert len(a_params) == len(b_params), (
        f"Parameter count mismatch: {len(a_params)} vs {len(b_params)}"
    )
    for i, (pa, pb) in enumerate(zip(a_params, b_params, strict=True)):
        assert torch.equal(pa, pb), (
            f"Parameter {i} differs between registry path and direct path "
            f"(max abs diff = {(pa - pb).abs().max().item():.2e})"
        )


def test_mlpppo_registry_equivalence() -> None:
    """Registry-instantiated MLPPPO matches direct-construction MLPPPO byte-for-byte.

    With both code paths fed the same seed, the seeded weight initialiser
    produces identical actor and critic tensors. Forward-pass outputs on
    a fixed BrainParams sample must match exactly. This is the equivalent
    of a byte-equivalence test for the registry refactor.
    """
    cfg_a = MLPPPOBrainConfig(
        seed=_SHARED_SEED,
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
    )
    cfg_b = MLPPPOBrainConfig(
        seed=_SHARED_SEED,
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
    )

    brain_direct = MLPPPOBrain(config=cfg_a, device=DeviceType.CPU)
    brain_registry = instantiate_brain("mlpppo", cfg_b, device=DeviceType.CPU)

    assert isinstance(brain_registry, MLPPPOBrain)
    _compare_actor_critic_weights(brain_direct.actor, brain_registry.actor)
    _compare_actor_critic_weights(brain_direct.critic, brain_registry.critic)

    # Forward-pass equivalence: feed both brains identical input, compare outputs.
    params = _make_brain_params()
    # Re-seed torch so any in-call RNG (e.g. action sampling) is matched.
    torch.manual_seed(_SHARED_SEED)
    np.random.seed(_SHARED_SEED)  # noqa: NPY002 — match brains' legacy global RNG
    actions_direct = brain_direct.run_brain(
        params,
        reward=_FIXED_PARAMS_REWARD,
        input_data=None,
        top_only=False,
        top_randomize=False,
    )
    torch.manual_seed(_SHARED_SEED)
    np.random.seed(_SHARED_SEED)  # noqa: NPY002 — match brains' legacy global RNG
    actions_registry = brain_registry.run_brain(
        params,
        reward=_FIXED_PARAMS_REWARD,
        input_data=None,
        top_only=False,
        top_randomize=False,
    )

    # The chosen-action list must match exactly with byte-identical probabilities.
    assert len(actions_direct) == len(actions_registry)
    for a_d, a_r in zip(actions_direct, actions_registry, strict=True):
        assert a_d.action is a_r.action, (
            f"Chosen action diverged: {a_d.action.value} vs {a_r.action.value}"
        )
        assert abs(a_d.probability - a_r.probability) < 1e-12, (
            f"Action probability diverged for {a_d.action.value}: "
            f"{a_d.probability!r} vs {a_r.probability!r}"
        )


def test_lstmppo_registry_equivalence() -> None:
    """Registry-instantiated LSTMPPO matches direct-construction LSTMPPO byte-for-byte.

    Same contract as MLPPPO with LSTM hidden state included implicitly via
    the actor/critic parameter comparison (the LSTM weights are part of
    those parameter lists).
    """
    cfg_a = LSTMPPOBrainConfig(
        seed=_SHARED_SEED,
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
    )
    cfg_b = LSTMPPOBrainConfig(
        seed=_SHARED_SEED,
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.NOCICEPTION],
    )

    brain_direct = LSTMPPOBrain(config=cfg_a, device=DeviceType.CPU)
    brain_registry = instantiate_brain("lstmppo", cfg_b, device=DeviceType.CPU)

    assert isinstance(brain_registry, LSTMPPOBrain)
    _compare_actor_critic_weights(brain_direct.actor, brain_registry.actor)
    _compare_actor_critic_weights(brain_direct.critic, brain_registry.critic)
    # LSTM/GRU weights ride on the recurrent module.
    _compare_actor_critic_weights(brain_direct.rnn, brain_registry.rnn)

    # Forward-pass equivalence.
    params = _make_brain_params()
    torch.manual_seed(_SHARED_SEED)
    np.random.seed(_SHARED_SEED)  # noqa: NPY002 — match brains' legacy global RNG
    actions_direct = brain_direct.run_brain(
        params,
        reward=_FIXED_PARAMS_REWARD,
        input_data=None,
        top_only=False,
        top_randomize=False,
    )
    torch.manual_seed(_SHARED_SEED)
    np.random.seed(_SHARED_SEED)  # noqa: NPY002 — match brains' legacy global RNG
    actions_registry = brain_registry.run_brain(
        params,
        reward=_FIXED_PARAMS_REWARD,
        input_data=None,
        top_only=False,
        top_randomize=False,
    )

    assert len(actions_direct) == len(actions_registry)
    for a_d, a_r in zip(actions_direct, actions_registry, strict=True):
        assert a_d.action is a_r.action, (
            f"Chosen action diverged: {a_d.action.value} vs {a_r.action.value}"
        )
        assert abs(a_d.probability - a_r.probability) < 1e-12, (
            f"Action probability diverged for {a_d.action.value}: "
            f"{a_d.probability!r} vs {a_r.probability!r}"
        )
