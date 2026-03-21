"""Shared PPO rollout buffer for brain architectures.

Provides a single ``RolloutBuffer`` used by all PPO-based brains that store
standard (state, action, log_prob, value, reward, done) tuples and train with
random minibatches.  Brain architectures with additional per-step data
(e.g. LSTM hidden states, SNN spike caches) maintain their own specialised
buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator


class RolloutBuffer:
    """Buffer for storing rollout experience for PPO updates."""

    def __init__(
        self,
        buffer_size: int,
        device: torch.device,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.buffer_size = buffer_size
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        """Clear all stored experience."""
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.position = 0

    def add(  # noqa: PLR0913
        self,
        state: np.ndarray,
        action: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,  # noqa: FBT001
    ) -> None:
        """Add a single experience to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward)
        self.dones.append(done)
        self.position += 1

    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return self.position >= self.buffer_size

    def __len__(self) -> int:
        return self.position

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        advantages = torch.zeros(len(self), device=self.device)
        last_gae = 0.0

        values = torch.stack(self.values).reshape(-1)

        for t in reversed(range(len(self))):
            if t == len(self) - 1:
                next_value = last_value.item()
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = values[t + 1].item()
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - values[t].item()
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        return returns, advantages

    def get_minibatches(
        self,
        num_minibatches: int,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Generate minibatches for training."""
        batch_size = len(self)
        minibatch_size = max(1, batch_size // num_minibatches)

        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(self.log_probs)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices_np = self.rng.permutation(batch_size)
        indices = torch.tensor(indices_np, device=self.device)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            yield {
                "states": states[mb_indices],
                "actions": actions[mb_indices],
                "old_log_probs": old_log_probs[mb_indices],
                "returns": returns[mb_indices],
                "advantages": advantages[mb_indices],
            }
