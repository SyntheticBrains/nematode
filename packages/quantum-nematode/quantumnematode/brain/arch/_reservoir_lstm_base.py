"""
Reservoir + QLIF-LSTM Base Class.

Abstract base for composing a fixed reservoir (QRH quantum or CRH classical)
as a feature extractor with a QLIF-LSTM temporal readout trained via recurrent
PPO with chunk-based truncated BPTT.

Architecture::

    Sensory Input -> Reservoir (FIXED, no gradients)
        |
        v
    Reservoir Features (52-75 dims)
        |
        v
    LayerNorm
        |
        v
    QLIFLSTMCell (temporal readout, trainable)
    +-------------------------------------------+
    | z = [reservoir_features, h_{t-1}]         |
    |                                           |
    | f_t = QLIF(W_f . z)   <- quantum forget   |
    | i_t = QLIF(W_i . z)   <- quantum input    |
    | c_hat = tanh(W_c . z)  <- classical       |
    | o_t = sigmoid(W_o . z) <- classical       |
    |                                           |
    | c_t = f_t * c_{t-1} + i_t * c_hat         |
    | h_t = o_t * tanh(c_t)                     |
    +-------------------------------------------+
        |
        +---------------------------+
        v                           v
    Actor (Linear)            Critic (MLP)
    [features, h_t] -> logits  [features, h_t.detach()] -> V(s)

    Training: Recurrent PPO with chunk-based truncated BPTT
    (identical to QLIF-LSTM, operating on reservoir features)

Concrete subclasses (in separate files, mirroring qrh.py / crh.py pattern):

- ``qrhqlstm.py``: QRHQLSTMBrain — QRH quantum reservoir + QLIF-LSTM readout
- ``crhqlstm.py``: CRHQLSTMBrain — CRH classical reservoir + QLIF-LSTM readout

The reservoir is used as a feature extractor only — its ``preprocess()``
and ``_get_reservoir_features()`` methods are called, but its ``run_brain()``
and ``learn()`` are never invoked.

References
----------
- Brand & Petruccione (2024) "A quantum leaky integrate-and-fire spiking
  neuron and network." npj Quantum Information, 10(1), 16.
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from __future__ import annotations

import abc
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Self

import numpy as np
import torch
from pydantic import Field, field_validator, model_validator
from torch import nn

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._qlif_layers import get_qiskit_backend
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.brain.arch.qliflstm import QLIFLSTMCell, QLIFLSTMCritic, QLIFLSTMRolloutBuffer
from quantumnematode.brain.modules import ModuleName  # noqa: TC001
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

if TYPE_CHECKING:
    from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Defaults (override QLIF-LSTM defaults for reservoir composition)
# ──────────────────────────────────────────────────────────────────────

DEFAULT_LSTM_HIDDEN_DIM = 64
DEFAULT_BPTT_CHUNK_LENGTH = 32
DEFAULT_SHOTS = 1024
DEFAULT_MEMBRANE_TAU = 0.9
DEFAULT_REFRACTORY_PERIOD = 0

# PPO defaults (tuned for reservoir-based training)
DEFAULT_ACTOR_LR = 0.0005
DEFAULT_CRITIC_LR = 0.0005
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_EPSILON = 0.2
DEFAULT_ENTROPY_COEF = 0.02
DEFAULT_ENTROPY_COEF_END = 0.008
DEFAULT_ENTROPY_DECAY_EPISODES = 500
DEFAULT_VALUE_LOSS_COEF = 0.5
DEFAULT_NUM_EPOCHS = 6
DEFAULT_ROLLOUT_BUFFER_SIZE = 1024
DEFAULT_MAX_GRAD_NORM = 0.5

DEFAULT_CRITIC_HIDDEN_DIM = 128
DEFAULT_CRITIC_NUM_LAYERS = 2

# Validation constants
MIN_LSTM_HIDDEN_DIM = 2
MIN_SHOTS = 100
MIN_BPTT_CHUNK_LENGTH = 4


# ──────────────────────────────────────────────────────────────────────
# ReservoirLSTMBaseConfig — shared config for reservoir + LSTM brains
# ──────────────────────────────────────────────────────────────────────


class ReservoirLSTMBaseConfig(BrainConfig):
    """Shared configuration for reservoir + QLIF-LSTM brains (QRH-QLSTM, CRH-QLSTM).

    Contains all LSTM readout, QLIF gate, PPO, LR scheduling, and critic fields.
    Subclass configs add reservoir-specific fields.
    """

    # LSTM readout
    lstm_hidden_dim: int = Field(default=DEFAULT_LSTM_HIDDEN_DIM, description="LSTM hidden dim.")
    bptt_chunk_length: int = Field(
        default=DEFAULT_BPTT_CHUNK_LENGTH,
        description="Chunk length for truncated BPTT.",
    )

    # QLIF gate parameters
    shots: int = Field(default=DEFAULT_SHOTS, description="Quantum measurement shots per gate.")
    membrane_tau: float = Field(
        default=DEFAULT_MEMBRANE_TAU,
        description="QLIF leak time constant.",
    )
    refractory_period: int = Field(
        default=DEFAULT_REFRACTORY_PERIOD,
        description="Refractory period.",
    )
    use_quantum_gates: bool = Field(
        default=True,
        description="Use QLIF quantum gates (True) or classical sigmoid (False).",
    )

    # PPO hyperparameters
    actor_lr: float = Field(default=DEFAULT_ACTOR_LR, gt=0, description="Actor learning rate.")
    critic_lr: float = Field(default=DEFAULT_CRITIC_LR, gt=0, description="Critic learning rate.")
    gamma: float = Field(default=DEFAULT_GAMMA, description="Discount factor.")
    gae_lambda: float = Field(default=DEFAULT_GAE_LAMBDA, description="GAE lambda.")
    clip_epsilon: float = Field(default=DEFAULT_CLIP_EPSILON, description="PPO clipping epsilon.")
    entropy_coef: float = Field(default=DEFAULT_ENTROPY_COEF, description="Initial entropy coef.")
    entropy_coef_end: float = Field(
        default=DEFAULT_ENTROPY_COEF_END,
        description="Final entropy coef after decay.",
    )
    entropy_decay_episodes: int = Field(
        default=DEFAULT_ENTROPY_DECAY_EPISODES,
        description="Episodes over which entropy decays.",
    )
    value_loss_coef: float = Field(default=DEFAULT_VALUE_LOSS_COEF, description="Value loss coef.")
    num_epochs: int = Field(default=DEFAULT_NUM_EPOCHS, description="PPO epochs per rollout.")
    rollout_buffer_size: int = Field(
        default=DEFAULT_ROLLOUT_BUFFER_SIZE,
        description="Steps to collect before PPO update.",
    )
    max_grad_norm: float = Field(default=DEFAULT_MAX_GRAD_NORM, description="Max gradient norm.")

    # LR scheduling
    lr_warmup_episodes: int | None = Field(default=None, description="LR warmup episodes.")
    lr_warmup_start: float | None = Field(default=None, description="Starting LR for warmup.")
    lr_decay_episodes: int | None = Field(default=None, description="LR decay episodes.")
    lr_decay_end: float | None = Field(default=None, description="Final LR after decay.")

    # Critic architecture
    critic_hidden_dim: int = Field(
        default=DEFAULT_CRITIC_HIDDEN_DIM,
        description="Critic hidden dim.",
    )
    critic_num_layers: int = Field(default=DEFAULT_CRITIC_NUM_LAYERS, description="Critic layers.")

    # Sensory modules
    sensory_modules: list[ModuleName] | None = Field(
        default=None,
        description="Sensory modules for feature extraction (None = legacy mode).",
    )

    # Device
    device_type: DeviceType = Field(default=DeviceType.CPU, description="Device for computation.")

    @field_validator("lstm_hidden_dim")
    @classmethod
    def validate_lstm_hidden_dim(cls, v: int) -> int:
        """Validate lstm_hidden_dim >= 2."""
        if v < MIN_LSTM_HIDDEN_DIM:
            msg = f"lstm_hidden_dim must be >= {MIN_LSTM_HIDDEN_DIM}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("shots")
    @classmethod
    def validate_shots(cls, v: int) -> int:
        """Validate shots >= 100."""
        if v < MIN_SHOTS:
            msg = f"shots must be >= {MIN_SHOTS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("membrane_tau")
    @classmethod
    def validate_membrane_tau(cls, v: float) -> float:
        """Validate membrane_tau in (0, 1]."""
        if not (0 < v <= 1):
            msg = f"membrane_tau must be in (0, 1], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("num_epochs")
    @classmethod
    def validate_num_epochs(cls, v: int) -> int:
        """Validate num_epochs >= 1."""
        if v < 1:
            msg = f"num_epochs must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("bptt_chunk_length")
    @classmethod
    def validate_bptt_chunk_length(cls, v: int) -> int:
        """Validate bptt_chunk_length >= 4."""
        if v < MIN_BPTT_CHUNK_LENGTH:
            msg = f"bptt_chunk_length must be >= {MIN_BPTT_CHUNK_LENGTH}, got {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_buffer_vs_chunk(self) -> Self:
        """Validate rollout_buffer_size >= bptt_chunk_length."""
        if self.rollout_buffer_size < self.bptt_chunk_length:
            msg = (
                f"rollout_buffer_size ({self.rollout_buffer_size}) must be >= "
                f"bptt_chunk_length ({self.bptt_chunk_length})"
            )
            raise ValueError(msg)
        return self


# ──────────────────────────────────────────────────────────────────────
# ReservoirLSTMBase — abstract base for reservoir + LSTM composition
# ──────────────────────────────────────────────────────────────────────


class ReservoirLSTMBase(ClassicalBrain, abc.ABC):
    """Abstract base for reservoir + QLIF-LSTM temporal readout.

    Composes a fixed reservoir (as a feature extractor) with a trainable
    QLIF-LSTM cell, actor head, and critic MLP. Handles recurrent PPO
    training with truncated BPTT.

    Subclasses implement ``_create_reservoir()`` and
    ``_compute_reservoir_feature_dim()`` to specify which reservoir to use.
    """

    _brain_name: str = "ReservoirLSTM"

    @abc.abstractmethod
    def _create_reservoir(
        self,
        config: ReservoirLSTMBaseConfig,
    ) -> Any:  # noqa: ANN401
        """Create the reservoir brain instance used as a feature extractor."""

    @abc.abstractmethod
    def _compute_reservoir_feature_dim(
        self,
        config: ReservoirLSTMBaseConfig,
    ) -> int:
        """Compute the reservoir output feature dimension."""

    def __init__(
        self,
        config: ReservoirLSTMBaseConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.num_actions = num_actions
        self._device_type = device
        self.device = torch.device(device.value)
        self._action_set = action_set if action_set is not None else DEFAULT_ACTIONS[:num_actions]

        if self.num_actions != len(self._action_set):
            msg = (
                f"num_actions ({self.num_actions}) does not match "
                f"action_set length ({len(self._action_set)})"
            )
            raise ValueError(msg)

        # Seeding
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)
        logger.info(f"{self._brain_name}Brain using seed: {self.seed}")

        # Create reservoir (feature extractor only)
        self.reservoir = self._create_reservoir(config)
        self.feature_dim = self._compute_reservoir_feature_dim(config)

        # Data tracking
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        # Build networks, optimizers, buffer, and state
        self._build_networks_and_state(config, num_actions)

    def _build_networks_and_state(
        self,
        config: ReservoirLSTMBaseConfig,
        num_actions: int,
    ) -> None:
        """Initialize networks, optimizers, rollout buffer, and tracking state."""
        # Qiskit backend (lazy init for quantum gates)
        self._backend: Any = None

        # LayerNorm for reservoir features
        self.feature_norm = nn.LayerNorm(self.feature_dim).to(self.device)

        # QLIF-LSTM cell (input = reservoir features)
        self.lstm_cell = QLIFLSTMCell(
            input_dim=self.feature_dim,
            hidden_dim=config.lstm_hidden_dim,
            use_quantum_gates=config.use_quantum_gates,
            shots=config.shots,
            membrane_tau=config.membrane_tau,
            refractory_period=config.refractory_period,
            device=self.device,
        ).to(self.device)

        # Actor head: [reservoir_features, h_t] -> action logits
        actor_input_dim = self.feature_dim + config.lstm_hidden_dim
        self.actor_head = nn.Linear(actor_input_dim, num_actions).to(self.device)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        if self.actor_head.bias is not None:
            nn.init.constant_(self.actor_head.bias, 0.0)

        # Critic: [reservoir_features, h_t.detach()] -> value
        critic_input_dim = self.feature_dim + config.lstm_hidden_dim
        self.critic = QLIFLSTMCritic(
            input_dim=critic_input_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
        ).to(self.device)

        # Optimizers: actor includes LSTM cell + actor head + LayerNorm
        self.actor_optimizer = torch.optim.Adam(
            list(self.lstm_cell.parameters())
            + list(self.actor_head.parameters())
            + list(self.feature_norm.parameters()),
            lr=config.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
        )

        # Episode counter
        self._episode_count = 0

        # LR scheduling (warmup + decay)
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
            config.lr_decay_episodes is not None or config.lr_warmup_episodes is not None
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

        # Rollout buffer
        self.buffer = QLIFLSTMRolloutBuffer(
            config.rollout_buffer_size,
            self.device,
            rng=self.rng,
        )

        # LSTM hidden state
        self.h_t = torch.zeros(config.lstm_hidden_dim, device=self.device)
        self.c_t = torch.zeros(config.lstm_hidden_dim, device=self.device)

        # State tracking
        self.training = True
        self.current_probabilities: np.ndarray | None = None
        self._step_count = 0

        # Pending step data (stored in run_brain, consumed in learn)
        self._pending_features: np.ndarray | None = None
        self._pending_action: int = 0
        self._pending_log_prob: float = 0.0
        self._pending_value: float = 0.0
        self._pending_h_state: torch.Tensor = self.h_t.clone()
        self._pending_c_state: torch.Tensor = self.c_t.clone()

        # Count parameters
        actor_params = sum(
            p.numel()
            for p in (
                list(self.lstm_cell.parameters())
                + list(self.actor_head.parameters())
                + list(self.feature_norm.parameters())
            )
        )
        critic_params = sum(p.numel() for p in self.critic.parameters())
        logger.info(
            f"{self._brain_name}Brain initialized: "
            f"feature_dim={self.feature_dim}, hidden_dim={config.lstm_hidden_dim}, "
            f"quantum_gates={'ON' if config.use_quantum_gates else 'OFF'}, "
            f"actor_params={actor_params:,}, critic_params={critic_params:,}, "
            f"total={actor_params + critic_params:,}",
        )

    def _get_backend(self) -> Any:  # noqa: ANN401
        """Get or create the Qiskit Aer backend."""
        if self._backend is None:
            self._backend = get_qiskit_backend(self._device_type, seed=self.seed)
            self.lstm_cell.set_backend(self._backend)
        return self._backend

    # ──────────────────────────────────────────────────────────────────
    # Feature Extraction
    # ──────────────────────────────────────────────────────────────────

    def _extract_reservoir_features(self, params: BrainParams) -> np.ndarray:
        """Extract reservoir features from sensory input.

        Delegates to the reservoir's preprocess() and _get_reservoir_features().
        """
        sensory_features = self.reservoir.preprocess(params)
        return self.reservoir._get_reservoir_features(sensory_features)  # noqa: SLF001

    def _get_actor_input(
        self,
        features: torch.Tensor,
        h_state: torch.Tensor,
    ) -> torch.Tensor:
        """Build actor input from normalized features and LSTM hidden state."""
        return torch.cat([features, h_state])

    def _get_critic_input(
        self,
        features: torch.Tensor,
        h_state: torch.Tensor,
    ) -> torch.Tensor:
        """Build critic input from normalized features and detached LSTM state.

        Both features and h_state are detached so critic gradients don't flow
        into the actor's LayerNorm or LSTM cell (separate optimizer).
        """
        return torch.cat([features.detach(), h_state.detach()])

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
        """Run reservoir + QLIF-LSTM to select an action."""
        # Extract reservoir features
        reservoir_features = self._extract_reservoir_features(params)

        # Ensure backend for quantum gates
        if self.config.use_quantum_gates:
            self._get_backend()

        # Normalize reservoir features
        features_t = torch.tensor(
            reservoir_features,
            dtype=torch.float32,
            device=self.device,
        )
        normalized_features = self.feature_norm(features_t)

        # Store pre-step hidden state for buffer
        h_pre = self.h_t.detach().clone()
        c_pre = self.c_t.detach().clone()

        # LSTM cell forward pass (no gradient during action selection)
        with torch.no_grad():
            h_new, c_new = self.lstm_cell(normalized_features, self.h_t, self.c_t)

        # Update hidden state
        self.h_t = h_new
        self.c_t = c_new

        # Actor: [normalized_features, h_t] -> logits
        with torch.no_grad():
            actor_input = self._get_actor_input(normalized_features, self.h_t)
            logits = self.actor_head(actor_input)
            action_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Sample action
        action_idx = self.rng.choice(self.num_actions, p=action_probs)
        action_name = self.action_set[action_idx]

        # Compute log probability
        log_prob = float(np.log(action_probs[action_idx] + 1e-8))

        # Compute critic value
        with torch.no_grad():
            critic_input = self._get_critic_input(normalized_features, self.h_t)
            value = self.critic(critic_input).item()

        # Store pending data for learn()
        self._pending_features = reservoir_features
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
                f"{self._brain_name} step {self._step_count}: "
                f"probs={np.array2string(action_probs, precision=3)}, "
                f"value={value:.4f}, "
                f"h_norm={torch.norm(self.h_t).item():.3f}, "
                f"c_norm={torch.norm(self.c_t).item():.3f}",
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
                critic_input = self._get_critic_input(normalized, self.h_t)
                last_value = self.critic(critic_input).item()
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
                h = chunk["h_init"].clone()
                c = chunk["c_init"].clone()

                log_probs_list: list[torch.Tensor] = []
                entropies_list: list[torch.Tensor] = []
                values_list: list[torch.Tensor] = []

                for step_idx in range(chunk["end"] - chunk["start"]):
                    # Reset hidden state at episode boundaries
                    if step_idx > 0 and chunk["dones"][step_idx - 1]:
                        h = torch.zeros_like(h)
                        c = torch.zeros_like(c)

                    features = chunk["features"][step_idx]

                    # Normalize reservoir features (differentiable)
                    features_t = torch.tensor(
                        features,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    normalized = self.feature_norm(features_t)

                    # LSTM forward (differentiable)
                    h, c = self.lstm_cell(normalized, h, c)

                    # Actor
                    actor_input = self._get_actor_input(normalized, h)
                    logits = self.actor_head(actor_input)
                    action_probs = torch.softmax(logits, dim=-1)

                    action_idx = chunk["actions"][step_idx].item()
                    log_prob = torch.log(action_probs[action_idx] + 1e-8)
                    log_probs_list.append(log_prob)

                    entropy = -torch.sum(
                        action_probs * torch.log(action_probs + 1e-10),
                    )
                    entropies_list.append(entropy)

                    # Critic
                    critic_input = self._get_critic_input(normalized, h)
                    value = self.critic(critic_input)
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

                # Combined loss for actor
                actor_loss = policy_loss - entropy_coef * mean_entropy

                # Actor backward and step
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.lstm_cell.parameters())
                    + list(self.actor_head.parameters())
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
                f"{self._brain_name} PPO update: policy_loss={avg_policy:.4f}, "
                f"value_loss={avg_value:.4f}, entropy={avg_entropy:.4f}, "
                f"entropy_coef={entropy_coef:.4f}, "
                f"clip_frac={total_clip_fraction / num_updates:.3f}, "
                f"buffer_size={buffer_len}, "
                f"lr={current_lr:.6f}, "
                f"ppo_time={ppo_elapsed:.1f}s, "
                f"episode={self._episode_count}",
            )

    def _get_entropy_coef(self) -> float:
        """Get current entropy coefficient with linear decay schedule."""
        if self._episode_count >= self.config.entropy_decay_episodes:
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

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for reservoir-LSTM brains."""

    def prepare_episode(self) -> None:
        """Prepare for a new episode — reset LSTM hidden state."""
        self.h_t = torch.zeros(self.config.lstm_hidden_dim, device=self.device)
        self.c_t = torch.zeros(self.config.lstm_hidden_dim, device=self.device)
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

        # Per-episode summary
        ep_rewards = self.history_data.rewards
        if ep_rewards:
            total_reward = sum(ep_rewards[-self._step_count :]) if self._step_count > 0 else 0.0
            current_lr = self._get_current_lr()
            logger.info(
                f"{self._brain_name} episode {self._episode_count} summary: "
                f"steps={self._step_count}, "
                f"total_reward={total_reward:.2f}, "
                f"buffer_len={len(self.buffer)}, "
                f"lr={current_lr:.6f}",
            )

    def copy(self) -> Self:
        """Create a deep copy with fresh hidden states."""
        new_brain = deepcopy(self)
        new_brain.h_t = torch.zeros(self.config.lstm_hidden_dim, device=self.device)
        new_brain.c_t = torch.zeros(self.config.lstm_hidden_dim, device=self.device)
        new_brain._backend = None  # noqa: SLF001
        return new_brain

    @property
    def action_set(self) -> list[Action]:
        """Get the list of available actions."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        """Set the list of available actions."""
        if len(actions) != self.num_actions:
            msg = (
                f"Cannot set action_set of length {len(actions)}: "
                f"network expects {self.num_actions} actions"
            )
            raise ValueError(msg)
        self._action_set = actions

    def build_brain(self) -> None:
        """Not applicable to reservoir-LSTM brains."""
        error_msg = f"{self._brain_name}Brain does not have a standalone quantum circuit."
        raise NotImplementedError(error_msg)

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,
        learning_rate: float = 0.01,
        **kwargs: object,
    ) -> None:
        """Not used — PPO uses its own optimizers."""
