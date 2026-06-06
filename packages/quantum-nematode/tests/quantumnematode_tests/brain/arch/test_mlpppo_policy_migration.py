"""Byte-equivalence regression for the MLP-PPO migration onto ``_policy.py``.

The shared action-policy module replaced MLP-PPO's inline
``softmax → Categorical → sample/log_prob/entropy``. The migration MUST be
byte-exact for MLP-PPO — no declared tolerance permitted for this brain.

This asserts the migrated `get_action_and_value` sampling path is **bitwise
identical to the inline reference ops it replaced**, computed in the SAME run on
the SAME logits and RNG seed. It deliberately avoids hard-coded absolute float
constants: those drift at ~1e-8 across BLAS / torch builds (so a golden snapshot
passes locally but fails CI) without indicating any change in computation. The
byte-equivalence of the extracted helpers themselves is covered, environment-
robustly, by ``test_policy.py``.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import numpy as np
import torch
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName

_SEED = 1234
_SAMPLE_SEED = 7
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


def test_mlpppo_sampling_matches_inline_reference() -> None:
    """Migrated sampling is bitwise-identical to the pre-migration inline ops.

    Same logits, same RNG seed, same run — so this holds on any machine.
    """
    brain = _build_brain()
    for state in _STATES:
        # The exact logits the brain will use (forward is deterministic, no RNG).
        x = torch.tensor(state, dtype=torch.float32, device=brain.device)
        logits = brain.actor(brain._apply_torch_gating(x))

        # Inline reference == the pre-migration ops, under a fixed RNG state.
        torch.manual_seed(_SAMPLE_SEED)
        probs_ref = torch.softmax(logits, dim=-1)
        dist_ref = torch.distributions.Categorical(probs_ref)
        action_ref = int(dist_ref.sample().item())
        log_prob_ref = dist_ref.log_prob(torch.tensor(action_ref, device=brain.device))
        entropy_ref = dist_ref.entropy()

        # Migrated brain path, under the SAME RNG state.
        torch.manual_seed(_SAMPLE_SEED)
        action, log_prob, entropy, _value = brain.get_action_and_value(state)

        assert action == action_ref
        assert torch.equal(log_prob, log_prob_ref)
        assert torch.equal(entropy, entropy_ref)
