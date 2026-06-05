"""Byte-equivalence regression for the MLP-PPO migration onto ``_policy.py``.

The shared action-policy module (T5 §1) replaces the inline ``Categorical``
sampling / evaluation in MLP-PPO. The migration MUST be byte-equivalent for
MLP-PPO (``add-continuous-2d-and-action-heads`` design.md D6 — no declared
tolerance permitted for this brain).

The golden values below were captured from the pre-migration MLP-PPO sampling
path with a fixed ``config.seed`` (the brain calls ``set_global_seed`` at
construction, so a fixed seed pins torch + numpy and makes the run reproducible
across processes). The test must pass identically before and after the
migration; a failure means the migration was not byte-equivalent.
"""

from __future__ import annotations

import numpy as np
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName

_SEED = 1234
# Golden snapshot from the pre-migration sampling path (config.seed=1234).
_GOLDEN_ACTIONS = [2, 2, 3]
_GOLDEN_LOG_PROBS = [-1.32896841, -1.1639806, -1.34662962]
_GOLDEN_ENTROPIES = [1.38140881, 1.35867167, 1.37947333]
_STATES = [
    np.array([0.3, -0.7], dtype=np.float32),
    np.array([-0.1, 0.9], dtype=np.float32),
    np.array([0.5, 0.2], dtype=np.float32),
]


def _build_brain() -> MLPPPOBrain:
    config = MLPPPOBrainConfig(
        seed=_SEED,
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        num_hidden_layers=2,
        learning_rate=0.01,
        rollout_buffer_size=64,
        num_epochs=2,
        num_minibatches=2,
    )
    # Construction calls set_global_seed(_SEED), pinning torch + numpy.
    return MLPPPOBrain(config=config, num_actions=4, device=DeviceType.CPU)


def test_mlpppo_sampling_byte_equivalent_to_pre_migration() -> None:
    """MLP-PPO sampling matches the pre-migration golden snapshot exactly."""
    brain = _build_brain()
    actions, log_probs, entropies = [], [], []
    for state in _STATES:
        action, log_prob, entropy, _value = brain.get_action_and_value(state)
        actions.append(action)
        log_probs.append(round(float(log_prob.detach()), 8))
        entropies.append(round(float(entropy.detach()), 8))

    assert actions == _GOLDEN_ACTIONS
    assert log_probs == _GOLDEN_LOG_PROBS
    assert entropies == _GOLDEN_ENTROPIES
