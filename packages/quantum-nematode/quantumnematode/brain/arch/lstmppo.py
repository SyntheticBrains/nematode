"""LSTM PPO Brain Architecture.

A classical LSTM/GRU-augmented PPO brain that processes sensory features through
a recurrent layer before actor/critic MLP heads. Designed for temporal sensing
tasks where memoryless MLP processing is insufficient.

Key Features:
- **Shared LSTM with detached critic**: Single LSTM processes features; critic
  receives h_t.detach() to prevent value loss from distorting LSTM representations
- **Separate actor/critic optimizers**: Actor optimizer trains LSTM + LayerNorm +
  actor MLP; critic optimizer trains only critic MLP
- **Chunk-based truncated BPTT**: Memory-efficient recurrent training with
  sequential chunks (not shuffled minibatches)
- **Entropy decay**: Linear decay from entropy_coef to entropy_coef_end
- **GRU option**: rnn_type="gru" for ablation experiments

Architecture::

    Sensory Features → LayerNorm → LSTM/GRU → Actor MLP → Actions
                                            → Critic MLP (detached h) → Value

References
----------
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- Training loop pattern from _reservoir_lstm_base.py
- Rollout buffer pattern from qliflstm.py
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from pydantic import Field, model_validator
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Iterator

    from quantumnematode.brain.weights import WeightComponent
    from quantumnematode.initializers._initializer import ParameterInitializer

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.brain.modules import (
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────


class LSTMPPOBrainConfig(BrainConfig):
    """Configuration for the LSTMPPOBrain architecture.

    Uses modular feature extraction via sensory_modules (required).
    No legacy 2-feature mode — sensory_modules must be provided.
    """

    # RNN type
    rnn_type: Literal["lstm", "gru"] = "lstm"

    # RNN architecture
    lstm_hidden_dim: int = 64

    # BPTT chunk length for truncated backpropagation
    bptt_chunk_length: int = 16

    # Actor MLP
    actor_hidden_dim: int = 64
    actor_num_layers: int = 2

    # Critic MLP
    critic_hidden_dim: int = 128
    critic_num_layers: int = 2

    # Learning rates (separate for actor/critic)
    actor_lr: float = 0.0005
    critic_lr: float = 0.0005

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    num_epochs: int = 6
    rollout_buffer_size: int = 1024
    max_grad_norm: float = 0.5

    # Entropy decay
    entropy_coef: float = 0.02
    entropy_coef_end: float = 0.008
    entropy_decay_episodes: int = 500

    # Weight initialisation scale.  Multiplies the orthogonal-init
    # ``gain`` for the actor's hidden Linear layers and the critic's
    # Linear layers (the actor's output-layer ``gain=0.01`` is preserved
    # as a stable-initial-policy PPO trick; the LSTM/GRU module is
    # unaffected and uses PyTorch's default init).  Default 1.0 is
    # byte-equivalent to existing init.  Evolvable innate-bias knob
    # for Baldwin/hyperparameter evolution arms; bounds picked so
    # extreme values can be set explicitly while excluding pathological
    # cases that would prevent the brain from training at all.
    weight_init_scale: float = Field(default=1.0, ge=0.1, le=5.0)

    # LR scheduling
    lr_warmup_episodes: int = 0
    lr_warmup_start: float | None = None
    lr_decay_episodes: int | None = None
    lr_decay_end: float | None = None

    # Sensory modules (required — no legacy 2-feature mode)
    sensory_modules: list[ModuleName] | None = None

    # Device
    device_type: DeviceType = DeviceType.CPU

    @model_validator(mode="after")
    def _validate_config(self) -> LSTMPPOBrainConfig:
        if self.lstm_hidden_dim < 2:  # noqa: PLR2004
            msg = f"lstm_hidden_dim must be >= 2, got {self.lstm_hidden_dim}"
            raise ValueError(msg)
        if self.bptt_chunk_length < 4:  # noqa: PLR2004
            msg = f"bptt_chunk_length must be >= 4, got {self.bptt_chunk_length}"
            raise ValueError(msg)
        if self.rollout_buffer_size < self.bptt_chunk_length:
            msg = (
                f"rollout_buffer_size ({self.rollout_buffer_size}) must be "
                f">= bptt_chunk_length ({self.bptt_chunk_length})"
            )
            raise ValueError(msg)
        if self.sensory_modules is None:
            msg = "sensory_modules is required for LSTMPPOBrain (no legacy 2-feature mode)"
            raise ValueError(msg)
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ──────────────────────────────────────────────────────────────────────────────


class LSTMPPORolloutBuffer:
    """Rollout buffer that stores per-step LSTM hidden states.

    Stores per-step (features, action, log_prob, value, reward, done, h_state,
    c_state) for chunk-based truncated BPTT during PPO updates.
    """

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
        self.features: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.h_states: list[torch.Tensor] = []
        self.c_states: list[torch.Tensor | None] = []
        self.position = 0

    def add(  # noqa: PLR0913
        self,
        features: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,  # noqa: FBT001
        h_state: torch.Tensor,
        c_state: torch.Tensor | None,
    ) -> None:
        """Add a single experience to the buffer."""
        self.features.append(features)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.h_states.append(h_state.detach().clone())
        self.c_states.append(c_state.detach().clone() if c_state is not None else None)
        self.position += 1

    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return self.position >= self.buffer_size

    def __len__(self) -> int:
        """Return the current buffer size."""
        return self.position

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        advantages = torch.zeros(len(self), device=self.device)
        last_gae = 0.0

        for t in reversed(range(len(self))):
            next_value = last_value if t == len(self) - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        values_t = torch.tensor(
            self.values,
            dtype=torch.float32,
            device=self.device,
        )
        returns = advantages + values_t
        return returns, advantages

    def get_sequential_chunks(
        self,
        chunk_length: int,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Iterator[dict]:
        """Generate sequential chunks for truncated BPTT.

        Splits the buffer into chunks of chunk_length steps. Each chunk
        carries its initial LSTM hidden state. Chunks are yielded in
        shuffled order for PPO.
        """
        n = len(self)
        chunk_starts = list(range(0, n, chunk_length))

        # Shuffle chunk order for PPO
        chunk_order = self.rng.permutation(len(chunk_starts))

        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(
            self.log_probs,
            dtype=torch.float32,
            device=self.device,
        )

        # Normalize advantages
        adv_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for chunk_idx in chunk_order:
            start = chunk_starts[chunk_idx]
            end = min(start + chunk_length, n)

            yield {
                "start": start,
                "end": end,
                "h_init": self.h_states[start],
                "c_init": self.c_states[start],
                "features": self.features[start:end],
                "actions": actions[start:end],
                "old_log_probs": old_log_probs[start:end],
                "returns": returns[start:end],
                "advantages": adv_normalized[start:end],
                "dones": self.dones[start:end],
            }


# ──────────────────────────────────────────────────────────────────────────────
# Critic Network
# ──────────────────────────────────────────────────────────────────────────────


class _LSTMPPOCritic(nn.Module):
    """MLP critic network for LSTM PPO."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning scalar value."""
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Brain
# ──────────────────────────────────────────────────────────────────────────────


class LSTMPPOBrain(ClassicalBrain):
    """LSTM/GRU-augmented PPO brain with chunk-based truncated BPTT.

    Processes sensory features through LayerNorm → LSTM/GRU → Actor/Critic MLPs.
    The critic receives detached hidden state to prevent value loss from
    distorting LSTM representations.
    """

    def __init__(  # noqa: PLR0915
        self,
        config: LSTMPPOBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] = DEFAULT_ACTIONS,
        *,
        parameter_initializer: ParameterInitializer | None = None,  # noqa: ARG002
    ) -> None:
        super().__init__()

        self.config = config
        self.num_actions = num_actions
        self.device = torch.device(device.to_torch_device_str())
        self._action_set = action_set

        # Seeding
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)
        logger.info(f"LSTMPPOBrain using seed: {self.seed}")

        # Sensory modules (validated as non-None by config)
        self.sensory_modules = config.sensory_modules
        assert self.sensory_modules is not None  # noqa: S101
        self.input_dim = get_classical_feature_dimension(self.sensory_modules)
        logger.info(
            f"Using classical feature extraction with modules: "
            f"{[m.value for m in self.sensory_modules]} "
            f"(input_dim={self.input_dim})",
        )

        # Data tracking
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        # ── Networks ──

        # LayerNorm on input features
        self.feature_norm = nn.LayerNorm(self.input_dim).to(self.device)

        # LSTM or GRU
        rnn_cls = nn.GRU if config.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=self.input_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=1,
            batch_first=False,
        ).to(self.device)
        self._is_gru = config.rnn_type == "gru"

        # Actor MLP: h_t → action logits
        actor_layers: list[nn.Module] = [
            nn.Linear(config.lstm_hidden_dim, config.actor_hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(config.actor_num_layers - 1):
            actor_layers += [
                nn.Linear(config.actor_hidden_dim, config.actor_hidden_dim),
                nn.ReLU(),
            ]
        actor_layers.append(nn.Linear(config.actor_hidden_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # Critic MLP: h_t.detach() → value
        self.critic = _LSTMPPOCritic(
            input_dim=config.lstm_hidden_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
        ).to(self.device)

        # Initialize weights
        self._initialize_weights()

        # ── Optimizers ──

        # Actor optimizer: LSTM/GRU + LayerNorm + actor MLP
        self.actor_optimizer = torch.optim.Adam(
            list(self.rnn.parameters())
            + list(self.feature_norm.parameters())
            + list(self.actor.parameters()),
            lr=config.actor_lr,
        )
        # Critic optimizer: critic MLP only
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
        )

        # ── LR scheduling ──
        self._episode_count = 0
        self.base_actor_lr = config.actor_lr
        self.base_critic_lr = config.critic_lr
        self.lr_warmup_episodes = config.lr_warmup_episodes
        self.lr_warmup_start = (
            config.lr_warmup_start if config.lr_warmup_start is not None else 0.1 * config.actor_lr
        )
        self.lr_decay_episodes = config.lr_decay_episodes
        self.lr_decay_end = (
            config.lr_decay_end if config.lr_decay_end is not None else 0.1 * config.actor_lr
        )
        self.lr_scheduling_enabled = (
            config.lr_decay_episodes is not None or config.lr_warmup_episodes > 0
        )

        if self.lr_scheduling_enabled:
            parts = []
            if self.lr_warmup_episodes:
                parts.append(
                    f"warmup {self.lr_warmup_start:.6f} -> {self.base_actor_lr:.6f} "
                    f"over {self.lr_warmup_episodes} eps",
                )
            if self.lr_decay_episodes:
                parts.append(
                    f"decay {self.base_actor_lr:.6f} -> {self.lr_decay_end:.6f} "
                    f"over {self.lr_decay_episodes} eps",
                )
            logger.info(f"LR schedule: {', '.join(parts)}")
            self._update_learning_rate()

        # ── Rollout buffer ──
        self.buffer = LSTMPPORolloutBuffer(
            config.rollout_buffer_size,
            self.device,
            rng=self.rng,
        )

        # ── Hidden state ──
        self.h_t = torch.zeros(1, 1, config.lstm_hidden_dim, device=self.device)
        self.c_t: torch.Tensor | None = (
            None if self._is_gru else torch.zeros(1, 1, config.lstm_hidden_dim, device=self.device)
        )

        # ── State tracking ──
        self.training = True
        self.current_probabilities: np.ndarray | None = None
        self._step_count = 0

        # Pending step data (stored in run_brain, consumed in learn)
        self._pending_features: np.ndarray | None = None
        self._pending_action: int = 0
        self._pending_log_prob: float = 0.0
        self._pending_value: float = 0.0
        self._pending_h_state: torch.Tensor = self.h_t.squeeze(0).squeeze(0).clone()
        self._pending_c_state: torch.Tensor | None = (
            self.c_t.squeeze(0).squeeze(0).clone() if self.c_t is not None else None
        )

        # Parameter count logging
        actor_params = sum(
            p.numel()
            for p in (
                list(self.rnn.parameters())
                + list(self.actor.parameters())
                + list(self.feature_norm.parameters())
            )
        )
        critic_params = sum(p.numel() for p in self.critic.parameters())
        logger.info(
            f"LSTMPPOBrain initialized: rnn_type={config.rnn_type}, "
            f"input_dim={self.input_dim}, hidden_dim={config.lstm_hidden_dim}, "
            f"actor_params={actor_params:,}, critic_params={critic_params:,}, "
            f"total={actor_params + critic_params:,}",
        )

    def _initialize_weights(self) -> None:
        """Initialize network weights with orthogonal initialization.

        The actor's hidden Linears and the critic's Linears use
        ``orthogonal_`` with ``gain=sqrt(2) * weight_init_scale`` —
        the multiplicative ``weight_init_scale`` knob defaults to 1.0
        (byte-equivalent to standard PPO init) and is evolvable for
        Baldwin/hyperparameter-evolution arms.  The actor's output
        layer keeps a fixed ``gain=0.01`` (deliberate small init for
        stable initial policy — the standard PPO trick) regardless
        of ``weight_init_scale``.  The LSTM/GRU module (``self.rnn``)
        is not touched here; it uses PyTorch's default init.
        """
        scale = self.config.weight_init_scale
        hidden_gain = float(np.sqrt(2)) * scale

        def init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=hidden_gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.actor.apply(init_linear)
        self.critic.net.apply(init_linear)

        # Actor output layer: small init for stable initial policy.
        # NOTE: gain=0.01 is intentionally NOT scaled by weight_init_scale.
        last_layer = self.actor[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.orthogonal_(last_layer.weight, gain=0.01)
            if last_layer.bias is not None:
                nn.init.constant_(last_layer.bias, 0.0)

    # ──────────────────────────────────────────────────────────────────
    # Feature Extraction
    # ──────────────────────────────────────────────────────────────────

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Extract classical features from sensory modules."""
        assert self.sensory_modules is not None  # noqa: S101
        return extract_classical_features(params, self.sensory_modules)

    # ──────────────────────────────────────────────────────────────────
    # RNN Helpers
    # ──────────────────────────────────────────────────────────────────

    def _rnn_forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run one step through the RNN.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (input_dim,).
        h : torch.Tensor
            Hidden state, shape (1, 1, hidden_dim).
        c : torch.Tensor | None
            Cell state for LSTM, None for GRU.

        Returns
        -------
        tuple
            (output_h, new_h, new_c) where output_h has shape (hidden_dim,).
        """
        x_seq = x.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
        if self._is_gru:
            output, h_new = self.rnn(x_seq, h)
            return output.squeeze(0).squeeze(0), h_new, None
        output, (h_new, c_new) = self.rnn(x_seq, (h, c))
        return output.squeeze(0).squeeze(0), h_new, c_new

    def _zero_hidden(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return zero-initialized hidden state."""
        h = torch.zeros(1, 1, self.config.lstm_hidden_dim, device=self.device)
        c = (
            None
            if self._is_gru
            else torch.zeros(1, 1, self.config.lstm_hidden_dim, device=self.device)
        )
        return h, c

    # ──────────────────────────────────────────────────────────────────
    # Brain Protocol
    # ──────────────────────────────────────────────────────────────────

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run LSTM PPO to select an action."""
        features = self.preprocess(params)

        # Normalize features
        features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        normalized = self.feature_norm(features_t)

        # Store pre-step hidden state for buffer
        h_pre = self.h_t.squeeze(0).squeeze(0).detach().clone()
        c_pre = self.c_t.squeeze(0).squeeze(0).detach().clone() if self.c_t is not None else None

        # LSTM forward pass (no gradient during action selection)
        with torch.no_grad():
            h_out, h_new, c_new = self._rnn_forward(normalized, self.h_t, self.c_t)

        # Update hidden state
        self.h_t = h_new
        self.c_t = c_new

        # Actor: h_t → logits → action
        with torch.no_grad():
            logits = self.actor(h_out)
            action_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Sample action
        action_idx = self.rng.choice(self.num_actions, p=action_probs)
        action_name = self.action_set[action_idx]

        # Log probability
        log_prob = float(np.log(action_probs[action_idx] + 1e-8))

        # Critic value (detached h)
        with torch.no_grad():
            value = self.critic(h_out.detach()).item()

        # Store pending data for learn()
        self._pending_features = features
        self._pending_action = action_idx
        self._pending_log_prob = log_prob
        self._pending_value = value
        self._pending_h_state = h_pre
        self._pending_c_state = c_pre

        # Update tracking
        self.current_probabilities = action_probs
        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=action_probs[action_idx],
        )
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(action_probs[action_idx]))

        # Periodic logging
        self._step_count += 1
        if self._step_count % 50 == 0:
            logger.debug(
                f"LSTMPPOBrain step {self._step_count}: "
                f"probs={np.array2string(action_probs, precision=3)}, "
                f"value={value:.4f}, "
                f"h_norm={torch.norm(self.h_t).item():.3f}",
            )

        return [self.latest_data.action]

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Add experience to rollout buffer and trigger PPO update when full."""
        self.history_data.rewards.append(reward)

        # Add to buffer
        if self._pending_features is not None:
            self.buffer.add(
                features=self._pending_features,
                action=self._pending_action,
                log_prob=self._pending_log_prob,
                value=self._pending_value,
                reward=reward,
                done=episode_done,
                h_state=self._pending_h_state,
                c_state=self._pending_c_state,
            )

        # PPO update when buffer is full or episode ends with enough data
        if self.buffer.is_full() or (
            episode_done and len(self.buffer) >= self.config.bptt_chunk_length
        ):
            self._perform_ppo_update()
            self.buffer.reset()

    def _perform_ppo_update(self) -> None:  # noqa: PLR0915
        """Perform PPO update with chunk-based truncated BPTT."""
        if len(self.buffer) == 0:
            return

        ppo_start = time.monotonic()

        # Compute last value for GAE bootstrap
        if self._pending_features is not None:
            with torch.no_grad():
                features_t = torch.tensor(
                    self._pending_features,
                    dtype=torch.float32,
                    device=self.device,
                )
                normalized = self.feature_norm(features_t)
                h_out, _, _ = self._rnn_forward(normalized, self.h_t, self.c_t)
                last_value = self.critic(h_out.detach()).item()
        else:
            last_value = 0.0

        # GAE computation
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value,
            self.config.gamma,
            self.config.gae_lambda,
        )

        buffer_len = len(self.buffer)
        entropy_coef = self._get_entropy_coef()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        for _epoch in range(self.config.num_epochs):
            for chunk in self.buffer.get_sequential_chunks(
                self.config.bptt_chunk_length,
                returns,
                advantages,
            ):
                # Re-run LSTM forward pass within this chunk
                # Reshape initial hidden states: (hidden_dim,) → (1, 1, hidden_dim)
                h = chunk["h_init"].unsqueeze(0).unsqueeze(0).clone()
                c_init = chunk["c_init"]
                c = c_init.unsqueeze(0).unsqueeze(0).clone() if c_init is not None else None

                log_probs_list: list[torch.Tensor] = []
                entropies_list: list[torch.Tensor] = []
                values_list: list[torch.Tensor] = []

                for step_idx in range(chunk["end"] - chunk["start"]):
                    # Reset hidden state at episode boundaries
                    if step_idx > 0 and chunk["dones"][step_idx - 1]:
                        h, c = self._zero_hidden()

                    features = chunk["features"][step_idx]

                    # Normalize features (differentiable)
                    features_t = torch.tensor(
                        features,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    normalized = self.feature_norm(features_t)

                    # RNN forward (differentiable)
                    h_out, h, c = self._rnn_forward(normalized, h, c)

                    # Actor
                    logits = self.actor(h_out)
                    action_probs = torch.softmax(logits, dim=-1)

                    action_idx = chunk["actions"][step_idx].item()
                    log_prob = torch.log(action_probs[action_idx] + 1e-8)
                    log_probs_list.append(log_prob)

                    entropy = -torch.sum(
                        action_probs * torch.log(action_probs + 1e-10),
                    )
                    entropies_list.append(entropy)

                    # Critic (detached h)
                    value = self.critic(h_out.detach())
                    values_list.append(value)

                if not log_probs_list:
                    continue

                # Stack per-step results
                new_log_probs = torch.stack(log_probs_list)
                mean_entropy = torch.stack(entropies_list).mean()
                values = torch.stack(values_list)

                # PPO policy loss
                ratio = torch.exp(new_log_probs - chunk["old_log_probs"])
                surr1 = ratio * chunk["advantages"]
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
                    )
                    * chunk["advantages"]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Track clip fraction
                with torch.no_grad():
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.config.clip_epsilon).float().mean().item()
                    )
                    total_clip_fraction += clip_frac

                # Value loss (Huber)
                value_loss = torch.nn.functional.smooth_l1_loss(values, chunk["returns"])

                # Actor backward and step
                actor_loss = policy_loss - entropy_coef * mean_entropy
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.rnn.parameters())
                    + list(self.actor.parameters())
                    + list(self.feature_norm.parameters()),
                    self.config.max_grad_norm,
                )
                self.actor_optimizer.step()

                # Critic backward and step
                critic_loss = self.config.value_loss_coef * value_loss
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.config.max_grad_norm,
                )
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += mean_entropy.item()
                num_updates += 1

        # Logging
        if num_updates > 0:
            ppo_elapsed = time.monotonic() - ppo_start
            avg_policy = total_policy_loss / num_updates
            avg_value = total_value_loss / num_updates
            avg_entropy = total_entropy / num_updates
            self.latest_data.loss = avg_policy
            self.history_data.losses.append(avg_policy)

            current_lr = self._get_current_lr()
            logger.info(
                f"LSTMPPOBrain PPO update: policy_loss={avg_policy:.4f}, "
                f"value_loss={avg_value:.4f}, entropy={avg_entropy:.4f}, "
                f"entropy_coef={entropy_coef:.4f}, "
                f"clip_frac={total_clip_fraction / num_updates:.3f}, "
                f"buffer_size={buffer_len}, "
                f"lr={current_lr:.6f}, "
                f"ppo_time={ppo_elapsed:.1f}s, "
                f"episode={self._episode_count}",
            )

    # ──────────────────────────────────────────────────────────────────
    # Scheduling
    # ──────────────────────────────────────────────────────────────────

    def _get_entropy_coef(self) -> float:
        """Get current entropy coefficient with linear decay schedule."""
        if (
            self.config.entropy_decay_episodes <= 0
            or self._episode_count >= self.config.entropy_decay_episodes
        ):
            return self.config.entropy_coef_end
        progress = self._episode_count / self.config.entropy_decay_episodes
        return self.config.entropy_coef + progress * (
            self.config.entropy_coef_end - self.config.entropy_coef
        )

    def _get_current_lr(self) -> float:
        """Get current learning rate based on episode count with warmup + decay."""
        if not self.lr_scheduling_enabled:
            return self.base_actor_lr

        ep = self._episode_count

        # Phase 1: Warmup
        if self.lr_warmup_episodes and ep < self.lr_warmup_episodes:
            progress = ep / self.lr_warmup_episodes
            return self.lr_warmup_start + progress * (self.base_actor_lr - self.lr_warmup_start)

        # Phase 2: Decay (offset by warmup duration)
        if self.lr_decay_episodes is not None:
            warmup = self.lr_warmup_episodes or 0
            decay_ep = ep - warmup
            if decay_ep < self.lr_decay_episodes:
                progress = decay_ep / self.lr_decay_episodes
                return self.base_actor_lr + progress * (self.lr_decay_end - self.base_actor_lr)
            return self.lr_decay_end

        return self.base_actor_lr

    def _update_learning_rate(self) -> None:
        """Update optimizer learning rates based on current schedule."""
        if not self.lr_scheduling_enabled:
            return

        new_lr = self._get_current_lr()
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = new_lr
        # Scale critic LR proportionally
        critic_scale = self.base_critic_lr / self.base_actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = new_lr * critic_scale

    # ──────────────────────────────────────────────────────────────────
    # Episode Lifecycle
    # ──────────────────────────────────────────────────────────────────

    def prepare_episode(self) -> None:
        """Reset LSTM hidden state for a new episode."""
        self.h_t, self.c_t = self._zero_hidden()
        self._pending_features = None
        self._step_count = 0

    def post_process_episode(
        self,
        *,
        episode_success: bool | None = None,  # noqa: ARG002
    ) -> None:
        """Post-process after each episode."""
        self._episode_count += 1
        self._update_learning_rate()

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for LSTMPPOBrain."""

    # ──────────────────────────────────────────────────────────────────
    # Weight Persistence
    # ──────────────────────────────────────────────────────────────────

    def get_weight_components(
        self,
        *,
        components: set[str] | None = None,
    ) -> dict[str, WeightComponent]:
        """Return weight components for persistence.

        Components
        ----------
        ``"lstm"``
            LSTM/GRU state_dict.
        ``"layer_norm"``
            LayerNorm state_dict.
        ``"policy"``
            Actor MLP state_dict.
        ``"value"``
            Critic MLP state_dict.
        ``"actor_optimizer"``
            Actor optimizer state_dict.
        ``"critic_optimizer"``
            Critic optimizer state_dict.
        ``"training_state"``
            Episode count and other training metadata.
        """
        from quantumnematode.brain.weights import WeightComponent

        all_components: dict[str, WeightComponent] = {
            "lstm": WeightComponent(
                name="lstm",
                state=self.rnn.state_dict(),
            ),
            "layer_norm": WeightComponent(
                name="layer_norm",
                state=self.feature_norm.state_dict(),
            ),
            "policy": WeightComponent(
                name="policy",
                state=self.actor.state_dict(),
            ),
            "value": WeightComponent(
                name="value",
                state=self.critic.state_dict(),
            ),
            "actor_optimizer": WeightComponent(
                name="actor_optimizer",
                state=self.actor_optimizer.state_dict(),
            ),
            "critic_optimizer": WeightComponent(
                name="critic_optimizer",
                state=self.critic_optimizer.state_dict(),
            ),
            "training_state": WeightComponent(
                name="training_state",
                state={"episode_count": self._episode_count},
            ),
        }

        if components is None:
            return all_components

        unknown = components - set(all_components)
        if unknown:
            msg = f"Unknown weight components: {unknown}. Valid components: {set(all_components)}"
            raise ValueError(msg)
        return {k: v for k, v in all_components.items() if k in components}

    def load_weight_components(
        self,
        components: dict[str, WeightComponent],
    ) -> None:
        """Load weight components into this brain."""
        # Load networks first
        if "lstm" in components:
            self.rnn.load_state_dict(components["lstm"].state)
        if "layer_norm" in components:
            self.feature_norm.load_state_dict(components["layer_norm"].state)
        if "policy" in components:
            self.actor.load_state_dict(components["policy"].state)
        if "value" in components:
            self.critic.load_state_dict(components["value"].state)

        # Optimizers after networks
        if "actor_optimizer" in components:
            self.actor_optimizer.load_state_dict(components["actor_optimizer"].state)
        if "critic_optimizer" in components:
            self.critic_optimizer.load_state_dict(components["critic_optimizer"].state)

        # Training state
        if "training_state" in components:
            ts = components["training_state"].state
            if "episode_count" in ts:
                self._episode_count = int(ts["episode_count"])
                self._update_learning_rate()

        # Reset buffer to prevent stale experience
        self.buffer.reset()

        logger.info(
            "LSTMPPOBrain weights loaded (components: %s, episode_count=%d)",
            list(components.keys()),
            self._episode_count,
        )

    # ──────────────────────────────────────────────────────────────────
    # Unsupported Protocol Methods
    # ──────────────────────────────────────────────────────────────────

    def copy(self) -> LSTMPPOBrain:
        """LSTMPPOBrain does not support copying."""
        error_msg = "LSTMPPOBrain does not support copying. Use deepcopy if needed."
        raise NotImplementedError(error_msg)

    @property
    def action_set(self) -> list[Action]:
        """Get the list of actions."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        self._action_set = actions

    def build_brain(self) -> None:
        """Not applicable to LSTMPPOBrain."""
        error_msg = "LSTMPPOBrain does not have a quantum circuit."
        raise NotImplementedError(error_msg)

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,
        learning_rate: float = 0.01,
        **kwargs: object,
    ) -> None:
        """Not used — PPO uses its own optimizer."""
