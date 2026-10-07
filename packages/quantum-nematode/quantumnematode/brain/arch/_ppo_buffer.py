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
        *,
        continuous_actions: bool = False,
    ) -> None:
        self.buffer_size = buffer_size
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng()
        # Discrete (default): actions are int indices stored as ``torch.long``.
        # Continuous: actions are the per-step pre-squash sample vectors
        # (``pre_tanh``, shape ``(action_dim,)``) stored as ``torch.float32``.
        self.continuous_actions = continuous_actions
        self.reset()

    def reset(self) -> None:
        """Clear all stored experience."""
        self.states: list[np.ndarray] = []
        self.actions: list[int | np.ndarray] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.position = 0

    def add(  # noqa: PLR0913
        self,
        state: np.ndarray,
        action: int | np.ndarray,
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
        if self.continuous_actions:
            # Stored pre-squash sample vectors → (batch, action_dim) float tensor.
            actions = torch.tensor(
                np.array(self.actions),
                dtype=torch.float32,
                device=self.device,
            )
        else:
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


class ChunkedRolloutBuffer(RolloutBuffer):
    """Rollout buffer for a brain whose state persists across environment steps.

    Stores each step's starting state beside the usual experience and serves contiguous chunks
    instead of shuffled single steps, so a replay can start each chunk from the state the rollout
    actually had and carry it through time inside the chunk. The starting states are stored
    detached: gradients never reach back past a chunk's first step.
    """

    def reset(self) -> None:
        """Clear all stored experience, starting states included."""
        super().reset()
        self.start_states: list[torch.Tensor] = []

    def add(  # noqa: PLR0913
        self,
        state: np.ndarray,
        action: int | np.ndarray,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,  # noqa: FBT001
        start_state: torch.Tensor | None = None,
    ) -> None:
        """Add one step's experience and the state the step started from."""
        if start_state is None:
            msg = "ChunkedRolloutBuffer.add needs the step's start_state"
            raise ValueError(msg)
        super().add(state, action, log_prob, value, reward, done)
        self.start_states.append(start_state.detach())

    def chunk_count(self, chunk_length: int) -> int:
        """Return how many chunks of ``chunk_length`` the stored steps make, the last partial."""
        return -(-len(self) // chunk_length)

    def get_chunk_minibatches(
        self,
        num_minibatches: int,
        chunk_length: int,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield minibatches of whole chunks, each tensor shaped ``(chunks, chunk_length, ...)``.

        Chunks are cut from the buffer in order and shuffled as units with the buffer's generator.
        A partial final chunk is zero-padded; ``mask`` marks the real steps. ``restart`` marks the
        steps a replay starts from their stored state: each chunk's first step and every step
        that begins an episode, which is any step after one whose ``done`` is set.
        Advantages are normalised over the real steps, as the single-step buffer does.
        """
        n = len(self)
        n_chunks = self.chunk_count(chunk_length)
        padded = n_chunks * chunk_length

        def pad(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((padded, *x.shape[1:]), dtype=x.dtype, device=self.device)
            out[:n] = x
            return out.reshape(n_chunks, chunk_length, *x.shape[1:])

        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        if self.continuous_actions:
            actions = torch.tensor(np.array(self.actions), dtype=torch.float32, device=self.device)
        else:
            actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(self.log_probs).reshape(n)
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dones = torch.tensor(self.dones, dtype=torch.bool, device=self.device)
        restart = torch.zeros(padded, dtype=torch.bool, device=self.device)
        restart[1:n] = dones[: n - 1]
        restart = restart.reshape(n_chunks, chunk_length)
        restart[:, 0] = True
        mask = torch.zeros(padded, dtype=torch.bool, device=self.device)
        mask[:n] = True

        tensors = {
            "states": pad(states),
            "actions": pad(actions),
            "old_log_probs": pad(old_log_probs),
            "returns": pad(returns),
            "advantages": pad(advantages),
            "start_states": pad(torch.stack(self.start_states)),
            "restart": restart,
            "mask": mask.reshape(n_chunks, chunk_length),
        }
        order = torch.tensor(self.rng.permutation(n_chunks), device=self.device)
        per_minibatch = max(1, n_chunks // num_minibatches)
        for start in range(0, n_chunks, per_minibatch):
            picked = order[start : start + per_minibatch]
            yield {name: tensor[picked] for name, tensor in tensors.items()}
