"""
Shared infrastructure for hybrid brain architectures.

Extracts common code from HybridQuantumBrain and HybridClassicalBrain
(and future hybrids like HybridQuantumCortexBrain) into a reusable module.
Each brain imports only what it needs — no shared base class is imposed.

Shared components:
- _CortexRolloutBuffer: Rollout storage, GAE computation, minibatch iteration
- _ReinforceUpdateStats: Bundled statistics for REINFORCE logging
- _fuse(): Mode-gated fusion of reflex logits and cortex biases
- _cortex_forward() / _cortex_value(): Classical cortex MLP forward passes
- init_cortex_mlps(): Classical cortex actor + critic MLP initialization
- LR scheduling: get_cortex_lr() / update_cortex_learning_rates()
- Cortex weight persistence: save_cortex_weights() / load_cortex_weights()
- perform_ppo_update(): PPO update for cortex training
- Shared constants and defaults
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn

from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from collections.abc import Iterator


# ──────────────────────────────────────────────────────────────────────
# Shared constants and defaults
# ──────────────────────────────────────────────────────────────────────

# Cortex PPO defaults
DEFAULT_CORTEX_HIDDEN_DIM = 64
DEFAULT_CORTEX_NUM_LAYERS = 2
DEFAULT_NUM_MODES = 3
DEFAULT_CORTEX_ACTOR_LR = 0.001
DEFAULT_CORTEX_CRITIC_LR = 0.001
DEFAULT_PPO_CLIP_EPSILON = 0.2
DEFAULT_PPO_EPOCHS = 4
DEFAULT_PPO_MINIBATCHES = 4
DEFAULT_PPO_BUFFER_SIZE = 512
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_ENTROPY_COEFF = 0.01
DEFAULT_MAX_GRAD_NORM = 0.5

# Joint fine-tune
DEFAULT_JOINT_FINETUNE_LR_FACTOR = 0.1

# Entropy regulation
DEFAULT_ENTROPY_FLOOR = 0.5
DEFAULT_ENTROPY_BOOST_MAX = 20.0
DEFAULT_ENTROPY_CEILING_FRACTION = 0.95

# Reward normalization
DEFAULT_REWARD_NORM_ALPHA = 0.01

# REINFORCE
DEFAULT_GAMMA = 0.99
DEFAULT_ADVANTAGE_CLIP = 2.0
DEFAULT_CLIP_EPSILON = 0.2
DEFAULT_EXPLORATION_EPSILON = 0.1
DEFAULT_EXPLORATION_DECAY_EPISODES = 80
DEFAULT_LR_MIN_FACTOR = 0.1
MIN_REINFORCE_BATCH_SIZE = 2

# Training stages
STAGE_REFLEX_ONLY = 1
STAGE_CORTEX_ONLY = 2
STAGE_JOINT = 3


# ──────────────────────────────────────────────────────────────────────
# Protocol for brain config (structural subtyping for shared functions)
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class HybridBrainConfig(Protocol):
    """Structural protocol for config fields used by shared functions."""

    cortex_hidden_dim: int
    cortex_num_layers: int
    num_modes: int
    cortex_actor_lr: float
    cortex_critic_lr: float
    ppo_clip_epsilon: float
    ppo_epochs: int
    ppo_minibatches: int
    ppo_buffer_size: int
    gae_lambda: float
    entropy_coeff: float
    max_grad_norm: float
    gamma: float


# ──────────────────────────────────────────────────────────────────────
# Internal data structures
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _ReinforceUpdateStats:
    """Bundled statistics for REINFORCE optimizer step logging."""

    raw_mean: float
    returns_tensor: torch.Tensor
    epoch: int
    num_epochs: int


class _CortexRolloutBuffer:
    """Rollout buffer for cortex PPO training."""

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
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward)
        self.dones.append(done)
        self.position += 1

    def is_full(self) -> bool:
        return self.position >= self.buffer_size

    def __len__(self) -> int:
        return self.position

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        batch_size = len(self)
        minibatch_size = batch_size // num_minibatches

        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(self.log_probs)

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


# ──────────────────────────────────────────────────────────────────────
# Classical cortex MLP initialization
# ──────────────────────────────────────────────────────────────────────


def init_cortex_mlps(  # noqa: PLR0913
    input_dim: int,
    cortex_hidden_dim: int,
    cortex_num_layers: int,
    num_motor: int,
    num_modes: int,
    device: torch.device,
) -> tuple[nn.Sequential, nn.Sequential]:
    """Initialize cortex actor and critic MLPs with orthogonal init.

    Returns (cortex_actor, cortex_critic).
    """
    cortex_output_dim = num_motor + num_modes  # action biases + mode logits

    # Actor: sensory -> hidden -> (action_biases + mode_logits)
    actor_layers: list[nn.Module] = [
        nn.Linear(input_dim, cortex_hidden_dim),
        nn.ReLU(),
    ]
    for _ in range(cortex_num_layers - 1):
        actor_layers += [
            nn.Linear(cortex_hidden_dim, cortex_hidden_dim),
            nn.ReLU(),
        ]
    actor_layers.append(nn.Linear(cortex_hidden_dim, cortex_output_dim))
    cortex_actor = nn.Sequential(*actor_layers).to(device)

    # Critic: sensory -> hidden -> V(s)
    critic_layers: list[nn.Module] = [
        nn.Linear(input_dim, cortex_hidden_dim),
        nn.ReLU(),
    ]
    for _ in range(cortex_num_layers - 1):
        critic_layers += [
            nn.Linear(cortex_hidden_dim, cortex_hidden_dim),
            nn.ReLU(),
        ]
    critic_layers.append(nn.Linear(cortex_hidden_dim, 1))
    cortex_critic = nn.Sequential(*critic_layers).to(device)

    # Orthogonal initialization
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    cortex_actor.apply(_init_weights)
    cortex_critic.apply(_init_weights)

    return cortex_actor, cortex_critic


def init_critic_mlp(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
) -> nn.Sequential:
    """Initialize a standalone critic MLP with orthogonal init.

    Architecture: input_dim -> hidden -> ... -> hidden -> 1
    Used when the cortex actor is NOT a classical MLP (e.g., QSNN cortex).
    """
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
    ]
    for _ in range(num_layers - 1):
        layers += [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ]
    layers.append(nn.Linear(hidden_dim, 1))
    critic = nn.Sequential(*layers).to(device)

    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    critic.apply(_init_weights)
    return critic


# ──────────────────────────────────────────────────────────────────────
# Classical cortex forward passes
# ──────────────────────────────────────────────────────────────────────


def cortex_forward(
    cortex_actor: nn.Sequential,
    sensory_input: torch.Tensor,
    num_motor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run cortex actor forward pass, returning action biases and mode logits."""
    cortex_out = cortex_actor(sensory_input)
    action_biases = cortex_out[..., :num_motor]
    mode_logits = cortex_out[..., num_motor:]
    return action_biases, mode_logits


def cortex_value(
    cortex_critic: nn.Sequential,
    sensory_input: torch.Tensor,
) -> torch.Tensor:
    """Get critic value estimate from sensory input."""
    return cortex_critic(sensory_input).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# Mode-gated fusion
# ──────────────────────────────────────────────────────────────────────


def fuse(
    reflex_logits: torch.Tensor,
    action_biases: torch.Tensor,
    mode_logits: torch.Tensor,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Apply mode-gated fusion.

    Returns (final_logits, reflex_trust, mode_probs).
    """
    mode_probs = torch.softmax(mode_logits, dim=-1)
    reflex_trust = mode_probs[0]  # forage mode trusts reflex
    final_logits = reflex_logits * reflex_trust + action_biases
    return final_logits, float(reflex_trust.item()), mode_probs


# ──────────────────────────────────────────────────────────────────────
# Cortex LR scheduling
# ──────────────────────────────────────────────────────────────────────


def get_cortex_lr(  # noqa: PLR0913
    *,
    scheduling_enabled: bool,
    episode_count: int,
    base_lr: float,
    warmup_episodes: int,
    warmup_start: float,
    decay_episodes: int | None,
    decay_end: float,
) -> float:
    """Get the current cortex learning rate based on episode count.

    Supports warmup followed by optional decay.
    """
    if not scheduling_enabled:
        return base_lr

    # Warmup phase
    if episode_count < warmup_episodes:
        progress = episode_count / warmup_episodes
        return warmup_start + (base_lr - warmup_start) * progress

    # Decay phase (if enabled)
    if decay_episodes is not None:
        decay_start_episode = warmup_episodes
        decay_episode = episode_count - decay_start_episode
        if decay_episode < decay_episodes:
            progress = decay_episode / decay_episodes
            return base_lr + (decay_end - base_lr) * progress
        return decay_end

    return base_lr


def update_cortex_learning_rates(  # noqa: PLR0913
    *,
    scheduling_enabled: bool,
    episode_count: int,
    base_lr: float,
    warmup_episodes: int,
    warmup_start: float,
    decay_episodes: int | None,
    decay_end: float,
    cortex_actor_optimizer: torch.optim.Optimizer,
    cortex_critic_optimizer: torch.optim.Optimizer,
    critic_lr_independent: bool = False,
) -> None:
    """Update cortex optimizer learning rates based on current schedule.

    If critic_lr_independent is True, only the actor optimizer is updated;
    the critic keeps its own fixed learning rate.
    """
    if not scheduling_enabled:
        return

    new_lr = get_cortex_lr(
        scheduling_enabled=scheduling_enabled,
        episode_count=episode_count,
        base_lr=base_lr,
        warmup_episodes=warmup_episodes,
        warmup_start=warmup_start,
        decay_episodes=decay_episodes,
        decay_end=decay_end,
    )
    for param_group in cortex_actor_optimizer.param_groups:
        param_group["lr"] = new_lr
    if not critic_lr_independent:
        for param_group in cortex_critic_optimizer.param_groups:
            param_group["lr"] = new_lr


# ──────────────────────────────────────────────────────────────────────
# Cortex weight persistence (classical MLP cortex)
# ──────────────────────────────────────────────────────────────────────


def save_cortex_weights(
    cortex_actor: nn.Sequential,
    cortex_critic: nn.Sequential,
    session_id: str,
    *,
    brain_name: str = "Hybrid",
) -> None:
    """Save cortex actor and critic weights to disk."""
    export_dir = Path("exports") / session_id
    export_dir.mkdir(parents=True, exist_ok=True)
    save_path = export_dir / "cortex_weights.pt"

    weights_dict = {
        "cortex_actor": cortex_actor.state_dict(),
        "cortex_critic": cortex_critic.state_dict(),
    }
    torch.save(weights_dict, save_path)
    logger.info(f"{brain_name} cortex weights saved to {save_path}")


def load_cortex_weights(
    cortex_actor: nn.Sequential,
    cortex_critic: nn.Sequential,
    weights_path: str,
    *,
    brain_name: str = "Hybrid",
) -> None:
    """Load pre-trained cortex weights from disk."""
    path = Path(weights_path)
    if not path.exists():
        msg = f"Cortex weights file not found: {weights_path}"
        raise FileNotFoundError(msg)

    weights_dict = torch.load(path, weights_only=True)

    if "cortex_actor" not in weights_dict:
        msg = "Missing key 'cortex_actor' in cortex weights file"
        raise ValueError(msg)
    if "cortex_critic" not in weights_dict:
        msg = "Missing key 'cortex_critic' in cortex weights file"
        raise ValueError(msg)

    cortex_actor.load_state_dict(weights_dict["cortex_actor"])
    cortex_critic.load_state_dict(weights_dict["cortex_critic"])

    logger.info(f"{brain_name} cortex weights loaded from {weights_path}")


# ──────────────────────────────────────────────────────────────────────
# PPO update
# ──────────────────────────────────────────────────────────────────────


def perform_ppo_update(  # noqa: PLR0913
    ppo_buffer: _CortexRolloutBuffer,
    cortex_actor: nn.Sequential,
    cortex_critic: nn.Sequential,
    cortex_actor_optimizer: torch.optim.Optimizer,
    cortex_critic_optimizer: torch.optim.Optimizer,
    *,
    num_motor: int,
    gamma: float,
    gae_lambda: float,
    ppo_epochs: int,
    ppo_minibatches: int,
    ppo_clip_epsilon: float,
    entropy_coeff: float,
    max_grad_norm: float,
    device: torch.device,
    has_pending_cortex_data: bool,
    pending_cortex_state: np.ndarray | None,
    brain_name: str = "Hybrid",
) -> None:
    """Perform PPO update using collected cortex experience."""
    if len(ppo_buffer) == 0:
        return

    # Get last value for GAE
    if has_pending_cortex_data and pending_cortex_state is not None:
        with torch.no_grad():
            sensory_t = torch.tensor(
                pending_cortex_state,
                dtype=torch.float32,
                device=device,
            )
            last_value = cortex_value(cortex_critic, sensory_t)
    else:
        last_value = torch.tensor(0.0, device=device)

    returns, advantages = ppo_buffer.compute_returns_and_advantages(
        last_value,
        gamma,
        gae_lambda,
    )

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0
    total_approx_kl = 0.0
    num_updates = 0

    for _ in range(ppo_epochs):
        for batch in ppo_buffer.get_minibatches(
            ppo_minibatches,
            returns,
            advantages,
        ):
            # Cortex actor forward pass
            cortex_out = cortex_actor(batch["states"])
            logits = cortex_out[:, :num_motor]
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            new_log_probs = dist.log_prob(batch["actions"])
            entropy = dist.entropy().mean()

            # Values
            values = cortex_critic(batch["states"]).squeeze(-1)

            # PPO clipped surrogate
            log_ratio = new_log_probs - batch["old_log_probs"]
            ratio = torch.exp(log_ratio)
            with torch.no_grad():
                approx_kl_batch = ((ratio - 1) - log_ratio).mean().item()
            total_approx_kl += approx_kl_batch
            surr1 = ratio * batch["advantages"]
            surr2 = (
                torch.clamp(
                    ratio,
                    1 - ppo_clip_epsilon,
                    1 + ppo_clip_epsilon,
                )
                * batch["advantages"]
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (Huber for robustness)
            value_loss = nn.functional.smooth_l1_loss(values, batch["returns"])

            # Combined loss
            loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy

            # Optimize
            cortex_actor_optimizer.zero_grad()
            cortex_critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                cortex_actor.parameters(),
                max_grad_norm,
            )
            nn.utils.clip_grad_norm_(
                cortex_critic.parameters(),
                max_grad_norm,
            )
            cortex_actor_optimizer.step()
            cortex_critic_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy.item()
            num_updates += 1

    if num_updates > 0:
        # Compute explained variance
        with torch.no_grad():
            all_states = torch.tensor(
                np.array(ppo_buffer.states),
                dtype=torch.float32,
                device=device,
            )
            predicted = cortex_critic(all_states).squeeze(-1)
            target_var = returns.var()
            explained_var = (1.0 - (returns - predicted).var() / (target_var + 1e-8)).item()

        logger.info(
            f"{brain_name} cortex PPO: "
            f"policy_loss={total_policy_loss / num_updates:.4f}, "
            f"value_loss={total_value_loss / num_updates:.4f}, "
            f"entropy={total_entropy_loss / num_updates:.4f}, "
            f"explained_var={explained_var:.4f}, "
            f"approx_kl={total_approx_kl / num_updates:.4f}",
        )


# ──────────────────────────────────────────────────────────────────────
# Shared REINFORCE helpers
# ──────────────────────────────────────────────────────────────────────


def normalize_reward(
    reward: float,
    running_mean: float,
    running_var: float,
    alpha: float,
) -> tuple[float, float, float]:
    """Normalize reward using EMA running statistics.

    Returns (normalized_reward, new_running_mean, new_running_var).
    """
    new_mean = (1 - alpha) * running_mean + alpha * reward
    diff = reward - new_mean
    new_var = (1 - alpha) * running_var + alpha * diff * diff
    running_std = np.sqrt(new_var)
    normalized = (reward - new_mean) / (running_std + 1e-8)
    return normalized, new_mean, new_var


def adaptive_entropy_coef(  # noqa: PLR0913
    entropy_val: float,
    num_actions: int,
    *,
    entropy_coeff: float,
    entropy_floor: float,
    entropy_boost_max: float,
    entropy_ceiling_fraction: float,
) -> float:
    """Compute adaptive entropy coefficient based on current entropy level."""
    max_entropy = np.log(num_actions)
    entropy_ceiling = entropy_ceiling_fraction * max_entropy
    if entropy_val < entropy_floor:
        ratio = 1.0 - entropy_val / entropy_floor
        entropy_scale = 1.0 + ratio * (entropy_boost_max - 1.0)
    elif entropy_val > entropy_ceiling:
        ratio = (entropy_val - entropy_ceiling) / (max_entropy - entropy_ceiling)
        entropy_scale = max(0.0, 1.0 - ratio)
    else:
        entropy_scale = 1.0
    return entropy_coeff * entropy_scale


def exploration_schedule(
    episode_count: int,
    exploration_epsilon: float,
    exploration_decay_episodes: int,
) -> tuple[float, float]:
    """Compute exploration epsilon and temperature for current episode."""
    progress = min(1.0, episode_count / max(exploration_decay_episodes, 1))
    current_epsilon = exploration_epsilon * (1.0 - progress * 0.7)
    current_temperature = 1.5 - 0.5 * progress
    return current_epsilon, current_temperature


def preprocess_legacy(
    params: object,
) -> np.ndarray:
    """Extract legacy 2-feature input (gradient_strength, relative_angle).

    Accepts any object with gradient_strength, gradient_direction, and
    agent_direction attributes (BrainParams protocol).
    """
    from quantumnematode.env import Direction

    grad_strength = float(getattr(params, "gradient_strength", None) or 0.0)
    grad_direction = float(getattr(params, "gradient_direction", None) or 0.0)
    direction_map = {
        Direction.UP: np.pi / 2,
        Direction.DOWN: -np.pi / 2,
        Direction.LEFT: np.pi,
        Direction.RIGHT: 0.0,
    }
    agent_dir = getattr(params, "agent_direction", None) or Direction.UP
    agent_facing_angle = direction_map.get(agent_dir, np.pi / 2)
    relative_angle = (grad_direction - agent_facing_angle + np.pi) % (2 * np.pi) - np.pi
    rel_angle_norm = relative_angle / np.pi
    return np.array([grad_strength, rel_angle_norm], dtype=np.float32)
