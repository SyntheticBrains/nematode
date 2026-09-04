"""Frozen pre-extraction reference of the connectome brain's PPO update.

Copied VERBATIM from ``ConnectomePPOBrain._perform_ppo_update`` at the
``add-l4-trace-substrate`` merge-base (the M1 frozen-reference pattern —
see ``tests/.../env/_legacy_predator_reference.py``), rewritten only as a
free function over a brain object and reading hyperparameters from
``brain.config`` (the same values the pre-change brain cached on itself).
The byte-equivalence suite drives this against the extracted
``ConnectomePPORule`` from identical state and asserts bit-equal
parameters.

Do NOT modernise this file to track the live implementation — its entire
value is that it does not move.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from quantumnematode.brain.arch._policy import (
    categorical_evaluate_torch,
    continuous_evaluate_tanh_gaussian,
    ppo_clip_policy_loss,
)
from torch import nn

if TYPE_CHECKING:
    from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain


def legacy_perform_ppo_update(brain: ConnectomePPOBrain) -> None:
    """Run one PPO update over the rollout buffer (pre-extraction semantics).

    Skipped entirely when ``config.freeze_updates`` is True. Under the
    strict-mask mode, the chemical-synapse weights are projected onto the
    wild-type adjacency after every optimiser step.
    """
    config = brain.config
    if config.freeze_updates:
        return
    if len(brain.buffer) == 0:
        return

    last_value = (
        brain.last_value
        if brain.last_value is not None
        else torch.tensor(
            [0.0],
            device=brain.device,
        )
    )
    returns, advantages = brain.buffer.compute_returns_and_advantages(
        last_value,
        config.gamma,
        config.gae_lambda,
    )

    for _ in range(config.num_epochs):
        for batch in brain.buffer.get_minibatches(config.num_minibatches, returns, advantages):
            states = batch["states"]
            food_b, distal_b, mechano_b, zone_onehot_b, thermo_b = brain._unpack_state_batched(
                states,
            )
            new_head_out, hidden = brain.topology.forward_with_hidden_batched(
                food_b,
                predator_distal_features=distal_b,
                predator_mechano_features=mechano_b,
                contact_zone_onehot=zone_onehot_b,
                thermotaxis_features=thermo_b,
            )
            new_values = brain.critic(hidden).squeeze(-1)

            if brain.continuous:
                new_log_probs, entropy = continuous_evaluate_tanh_gaussian(
                    new_head_out,
                    brain.topology.log_std,
                    batch["actions"],
                    brain._action_low,
                    brain._action_high,
                )
            else:
                new_log_probs, entropy = categorical_evaluate_torch(
                    new_head_out,
                    batch["actions"],
                )
            policy_loss = ppo_clip_policy_loss(
                new_log_probs,
                batch["old_log_probs"],
                batch["advantages"],
                config.clip_epsilon,
            )
            value_loss = nn.functional.mse_loss(new_values, batch["returns"])
            loss = policy_loss + config.value_loss_coef * value_loss - config.entropy_coef * entropy

            brain.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                brain.topology.learnable_parameters + list(brain.critic.parameters()),
                config.max_grad_norm,
            )
            brain.optimizer.step()

            if config.chemical_mask_mode == "strict":
                with torch.no_grad():
                    brain.topology.w_chem.data.copy_(
                        brain.topology.apply_weight_mask(brain.topology.w_chem.data),
                    )
