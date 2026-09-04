"""PPO learning rule for the connectome brain.

The first genuine consumer of the ``LearningRule`` Protocol: the update
formerly inlined as ``ConnectomePPOBrain._perform_ppo_update`` moved here
verbatim (byte-equivalence guarded by
``tests/.../brain/arch/test_connectome_rule_extraction.py`` against the
frozen pre-extraction reference). The rule owns the optimiser, the critic,
and the PPO hyperparameters — including ``gamma``/``gae_lambda``, so it
computes returns and advantages itself; the brain retains experience
collection and feature unpacking, surfaced through ``ConnectomePPOBatch``.

Import discipline (change design Decision 3b): this module imports ONLY
leaf modules from ``quantumnematode.brain.arch`` — never the package —
so ``import quantumnematode.learning_rules.ppo`` works as a process's
first import even though ``brain/arch/__init__`` imports
``connectome_ppo`` at package load (whose import of this module is lazy,
inside ``ConnectomePPOBrain.__init__``).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn, optim

from quantumnematode.brain.arch._policy import (
    categorical_evaluate_torch,
    continuous_evaluate_tanh_gaussian,
    ppo_clip_policy_loss,
)
from quantumnematode.brain.arch._rule import RuleStepReport

if TYPE_CHECKING:
    from collections.abc import Callable

    from quantumnematode.brain.arch._ppo_buffer import RolloutBuffer
    from quantumnematode.brain.arch._topology import BrainTopology
    from quantumnematode.brain.arch.connectome_ppo import ConnectomeTopology


@dataclass
class ConnectomePPOBatch:
    """Experience surfaced by the brain to the rule for one update.

    Exactly what the pre-extraction inline update consumed: the rollout
    buffer handle (the rule calls ``get_minibatches`` once per epoch,
    preserving the per-epoch permutation draws), the brain's batched
    state-unpacker bound as a callable, and the bootstrap value of the
    step after the rollout.
    """

    buffer: RolloutBuffer
    unpack_batched: Callable[
        [torch.Tensor],
        tuple[
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
    ]
    last_value: torch.Tensor | None


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


class ConnectomePPORule:
    """Clipped-surrogate PPO over a ``ConnectomeTopology``.

    Satisfies the ``LearningRule`` Protocol. Owns the critic (constructed
    here, at the same point in the brain's ``__init__`` sequence as the
    pre-extraction code — immediately after topology construction — so the
    torch-RNG draw order is unchanged), the Adam optimiser over
    ``topology.learnable_parameters + critic.parameters()``, and every PPO
    hyperparameter.

    Two binding semantics, stated explicitly:

    - The optimiser is bound to the **construction-time** topology's
      parameters, so ``step`` refuses any other topology (a different one
      would silently train nothing — gradients on its parameters, optimiser
      stepping the originals). One rule per topology; construct a new rule
      for a new topology.
    - ``freeze_updates`` and ``chemical_mask_mode`` are **snapshots** taken
      at construction. The pre-extraction code read ``brain.config`` live
      on every update; no repo code mutates those fields mid-run, and doing
      so is unsupported — reconstruct the brain for a different mode.
    """

    def __init__(  # noqa: PLR0913 — mirrors the config fields it caches
        self,
        topology: ConnectomeTopology,
        *,
        learning_rate: float,
        gamma: float,
        gae_lambda: float,
        clip_epsilon: float,
        value_loss_coef: float,
        entropy_coef: float,
        num_epochs: int,
        num_minibatches: int,
        max_grad_norm: float,
        continuous: bool,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        strict_mask: bool,
        freeze_updates: bool,
        device: torch.device,
    ) -> None:
        # Critic: scalar value head over the same 302-dim activation vector.
        # Same construction + init calls as pre-extraction (byte-equivalence).
        self.critic = nn.Linear(topology.n_neurons, 1).to(device)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

        # Single Adam optimiser over the topology's learnable params + critic.
        learnable = topology.learnable_parameters + list(self.critic.parameters())
        self.optimizer = optim.Adam(learnable, lr=learning_rate)
        # Cached: the same Parameter objects in the same order for the
        # per-minibatch grad clip (the list can never change post-init).
        self._all_params = learnable
        self._topology = topology

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous
        self.action_low = action_low
        self.action_high = action_high
        self.strict_mask = strict_mask
        self.freeze_updates = freeze_updates
        self.device = device

    def step(
        self,
        topology: BrainTopology,
        batch: Any,  # noqa: ANN401 — LearningRule Protocol shape (rule-specific batch)
    ) -> RuleStepReport:
        """Run one PPO update over the rollout buffer.

        Short-circuits (still returning a ``RuleStepReport`` with ``None``
        loss fields) when ``freeze_updates`` is set — the paired-control
        branch — or when the buffer is empty. Under the strict-mask mode,
        the chemical-synapse weights are projected onto the wild-type
        adjacency after every optimiser step.
        """
        if topology is not self._topology:
            msg = (
                "ConnectomePPORule.step received a topology other than the one "
                "its optimiser was constructed over. The Adam state is bound to "
                "the construction-time parameters, so updating a different "
                "topology would silently train nothing — construct a new rule "
                "for a new topology."
            )
            raise ValueError(msg)
        topo = self._topology
        ppo_batch = cast("ConnectomePPOBatch", batch)
        if self.freeze_updates:
            return RuleStepReport()
        buffer = ppo_batch.buffer
        if len(buffer) == 0:
            return RuleStepReport()

        last_value = (
            ppo_batch.last_value
            if ppo_batch.last_value is not None
            else torch.tensor(
                [0.0],
                device=self.device,
            )
        )
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value,
            self.gamma,
            self.gae_lambda,
        )

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        total_losses: list[float] = []
        grad_norms: list[float] = []

        for _ in range(self.num_epochs):
            for minibatch in buffer.get_minibatches(self.num_minibatches, returns, advantages):
                # Batched forward pass through the topology + critic. The
                # minibatch's states are unpacked + run in ONE batched
                # connectome forward (and the post-K hidden states feed the
                # critic in one call).
                states = minibatch["states"]
                food_b, distal_b, mechano_b, zone_onehot_b, thermo_b = ppo_batch.unpack_batched(
                    states,
                )
                new_head_out, hidden = topo.forward_with_hidden_batched(
                    food_b,
                    predator_distal_features=distal_b,
                    predator_mechano_features=mechano_b,
                    contact_zone_onehot=zone_onehot_b,
                    thermotaxis_features=thermo_b,
                )
                new_values = self.critic(hidden).squeeze(-1)

                # Re-score actions under the current policy via the shared module:
                # discrete (Categorical) or continuous (tanh-Gaussian re-scoring the
                # stored pre-squash samples). Clipped surrogate is shared.
                if self.continuous:
                    new_log_probs, entropy = continuous_evaluate_tanh_gaussian(
                        new_head_out,
                        topo.log_std,
                        minibatch["actions"],
                        self.action_low,
                        self.action_high,
                    )
                else:
                    new_log_probs, entropy = categorical_evaluate_torch(
                        new_head_out,
                        minibatch["actions"],
                    )
                policy_loss = ppo_clip_policy_loss(
                    new_log_probs,
                    minibatch["old_log_probs"],
                    minibatch["advantages"],
                    self.clip_epsilon,
                )
                value_loss = nn.functional.mse_loss(new_values, minibatch["returns"])
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self._all_params,
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Strict-mask projection: under "strict" mode, zero out any
                # non-existent edges that the optimiser step would have
                # created. Under "soft_prior" mode, leave the weights as
                # the optimiser left them (the mask is then only an
                # initial-weight prior, and PPO is free to grow new edges).
                if self.strict_mask:
                    with torch.no_grad():
                        topo.w_chem.data.copy_(
                            topo.apply_weight_mask(topo.w_chem.data),
                        )

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                total_losses.append(loss.item())
                grad_norms.append(grad_norm.item())

        return RuleStepReport(
            policy_loss=_mean(policy_losses),
            value_loss=_mean(value_losses),
            entropy=_mean(entropies),
            total_loss=_mean(total_losses),
            grad_norm=_mean(grad_norms),
        )

    def reset_episode(self) -> None:
        """No per-episode rule state to clear.

        PPO's experience lives in the brain-owned rollout buffer (which
        deliberately spans episode boundaries), and GAE is computed fresh
        per update — so this is a documented no-op, kept for the
        ``LearningRule`` Protocol lifecycle.
        """
