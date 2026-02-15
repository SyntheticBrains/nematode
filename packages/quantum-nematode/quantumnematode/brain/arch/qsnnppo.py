"""
Quantum Spiking Neural Network with PPO (QSNN-PPO) Brain Architecture.

A hybrid quantum-classical brain that pairs a QSNN actor (QLIF quantum circuits
with surrogate gradients) with a classical MLP critic, trained via Proximal
Policy Optimization (PPO). This directly addresses the three diagnosed root
causes of standalone QSNN's predator failure:

1. **No critic** -> Classical MLP critic with GAE advantages
2. **High REINFORCE variance** -> PPO clipped surrogate objective
3. **Insufficient gradient passes** -> Multi-epoch updates with quantum caching

Architecture::

    Sensory Input (e.g. food_chemotaxis + nociception = 8 features)
        |
        +---------------------------+
        v                           v
    QSNN Actor (QLIF)         Classical Critic (MLP)
    8 sensory -> 16 hidden     Input: 8 raw sensory +
    -> 4 motor QLIF             16 hidden spike rates
    ~212 quantum params         (detached) = 24-dim
                                Linear(24,64) -> ReLU ->
    Surrogate gradient          Linear(64,64) -> ReLU ->
    backward pass               Linear(64,1) = V(s)
                                ~5K classical params

    Training: PPO with quantum caching
    1. Collect rollout_buffer_size steps
    2. Compute GAE advantages (lambda=0.95, gamma=0.99)
    3. For each of num_epochs epochs:
       a. Epoch 0: quantum circuits -> cache spike probs
       b. Epochs 1+: reuse cached probs, recompute ry_angles
       c. PPO clipped surrogate loss on actor
       d. Huber loss on critic
    4. Clear buffer, repeat

References
----------
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- Brand & Petruccione (2024) "A quantum leaky integrate-and-fire spiking neuron
  and network." npj Quantum Information, 10(1), 16.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import Field, field_validator
from torch import nn

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._qlif_layers import (
    WEIGHT_INIT_SCALE,
    encode_sensory_spikes,
    execute_qlif_layer,
    execute_qlif_layer_differentiable,
    execute_qlif_layer_differentiable_cached,
    get_qiskit_backend,
)
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.brain.modules import (
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)
from quantumnematode.env import Direction
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

if TYPE_CHECKING:
    from collections.abc import Iterator

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_NUM_SENSORY_NEURONS = 8
DEFAULT_NUM_HIDDEN_NEURONS = 16
DEFAULT_NUM_MOTOR_NEURONS = 4
DEFAULT_MEMBRANE_TAU = 0.9
DEFAULT_THRESHOLD = 0.5
DEFAULT_REFRACTORY_PERIOD = 0
DEFAULT_SHOTS = 1024
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_EPSILON = 0.2
DEFAULT_ENTROPY_COEF = 0.05
DEFAULT_ENTROPY_COEF_END = 0.005
DEFAULT_ENTROPY_DECAY_EPISODES = 100
DEFAULT_VALUE_LOSS_COEF = 0.5
DEFAULT_NUM_EPOCHS = 2
DEFAULT_NUM_MINIBATCHES = 4
DEFAULT_ROLLOUT_BUFFER_SIZE = 256
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_ACTOR_LR = 0.003
DEFAULT_LOGIT_SCALE = 20.0
DEFAULT_CRITIC_LR = 0.001
DEFAULT_CRITIC_HIDDEN_DIM = 64
DEFAULT_CRITIC_NUM_LAYERS = 2
DEFAULT_NUM_INTEGRATION_STEPS = 10
DEFAULT_WEIGHT_CLIP = 3.0
DEFAULT_THETA_MOTOR_MAX_NORM = 2.0
DEFAULT_ACTOR_WEIGHT_DECAY = 0.0
DEFAULT_THETA_HIDDEN_MIN_NORM = 2.0

# Validation constants
MIN_SENSORY_NEURONS = 1
MIN_HIDDEN_NEURONS = 1
MIN_MOTOR_NEURONS = 2
MIN_SHOTS = 100

EPISODE_LOG_INTERVAL = 25


# ──────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ──────────────────────────────────────────────────────────────────────


class QSNNRolloutBuffer:
    """Rollout buffer for QSNN-PPO that stores quantum spike caches.

    In addition to standard PPO fields (states, actions, log_probs, values,
    rewards, dones), stores hidden spike rates for critic input and quantum
    spike probability caches for multi-epoch training.
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
        # Hidden spike rates for critic input (detached from actor graph)
        self.hidden_spike_rates: list[np.ndarray] = []
        # Quantum spike probability caches for multi-epoch training
        # Structure: [step][integration_step] -> {'hidden': [...], 'motor': [...]}
        self.spike_caches: list[list[dict[str, list[float]]]] = []
        self.position = 0

    def add(  # noqa: PLR0913
        self,
        features: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,  # noqa: FBT001
        hidden_spike_rates: np.ndarray,
    ) -> None:
        """Add a single experience to the buffer."""
        self.features.append(features)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.hidden_spike_rates.append(hidden_spike_rates)
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
        """Compute GAE advantages and returns.

        Parameters
        ----------
        last_value : float
            Value estimate for the state after the last step.
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE lambda parameter.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (returns, advantages) tensors.
        """
        advantages = torch.zeros(len(self), device=self.device)
        last_gae = 0.0

        for t in reversed(range(len(self))):
            if t == len(self) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
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

    def get_minibatches(
        self,
        num_minibatches: int,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Iterator[dict]:
        """Generate minibatches for training.

        Yields dictionaries with indices into the buffer. The caller is
        responsible for running the QSNN forward pass per-step (since
        quantum circuits are not batchable like MLP forward passes).

        Yields
        ------
        dict
            Dictionary with 'indices', 'actions', 'old_log_probs',
            'returns', 'advantages' tensors.
        """
        batch_size = len(self)
        minibatch_size = batch_size // num_minibatches

        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(
            self.log_probs,
            dtype=torch.float32,
            device=self.device,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Generate random indices using seeded RNG
        indices_np = self.rng.permutation(batch_size)
        indices = torch.tensor(indices_np, device=self.device)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            yield {
                "indices": mb_indices,
                "actions": actions[mb_indices],
                "old_log_probs": old_log_probs[mb_indices],
                "returns": returns[mb_indices],
                "advantages": advantages[mb_indices],
            }


# ──────────────────────────────────────────────────────────────────────
# Critic MLP
# ──────────────────────────────────────────────────────────────────────


class QSNNPPOCritic(nn.Module):
    """Classical MLP critic for QSNN-PPO.

    Input is the concatenation of raw sensory features and hidden
    spike rates (detached from the actor's autograd graph).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = DEFAULT_CRITIC_HIDDEN_DIM,
        num_layers: int = DEFAULT_CRITIC_NUM_LAYERS,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

        # Orthogonal initialization
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state value estimate."""
        return self.network(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


class QSNNPPOBrainConfig(BrainConfig):
    """Configuration for the QSNNPPOBrain architecture.

    Supports two modes for input feature extraction:

    1. **Legacy mode** (default): Uses 2 features (gradient_strength, relative_angle)
       - Set ``sensory_modules=None`` (default)

    2. **Unified sensory mode**: Uses modular feature extraction from brain/modules.py
       - Set ``sensory_modules`` to a list of ModuleName values
       - Each module contributes 2 features [strength, angle]
    """

    # QLIF network architecture
    num_sensory_neurons: int = Field(
        default=DEFAULT_NUM_SENSORY_NEURONS,
        description="Number of sensory layer neurons.",
    )
    num_hidden_neurons: int = Field(
        default=DEFAULT_NUM_HIDDEN_NEURONS,
        description="Number of hidden interneuron layer neurons.",
    )
    num_motor_neurons: int = Field(
        default=DEFAULT_NUM_MOTOR_NEURONS,
        description="Number of motor layer neurons.",
    )
    membrane_tau: float = Field(
        default=DEFAULT_MEMBRANE_TAU,
        description="Leak time constant for QLIF neurons in (0, 1].",
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        description="Firing threshold in (0, 1).",
    )
    refractory_period: int = Field(
        default=DEFAULT_REFRACTORY_PERIOD,
        description="Timesteps to suppress activity after firing.",
    )
    shots: int = Field(
        default=DEFAULT_SHOTS,
        description="Number of quantum measurement shots.",
    )
    num_integration_steps: int = Field(
        default=DEFAULT_NUM_INTEGRATION_STEPS,
        description="Number of QLIF timesteps per decision.",
    )
    logit_scale: float = Field(
        default=DEFAULT_LOGIT_SCALE,
        description="Scaling factor for converting spike probabilities to action logits.",
    )
    weight_clip: float = Field(
        default=DEFAULT_WEIGHT_CLIP,
        description="Maximum absolute weight value for stability.",
    )
    theta_motor_max_norm: float = Field(
        default=DEFAULT_THETA_MOTOR_MAX_NORM,
        description="Max L2 norm for theta_motor vector.",
    )
    theta_hidden_min_norm: float = Field(
        default=DEFAULT_THETA_HIDDEN_MIN_NORM,
        description="Min L2 norm for theta_hidden vector (prevents representational collapse).",
    )

    # PPO hyperparameters
    gamma: float = Field(
        default=DEFAULT_GAMMA,
        description="Discount factor.",
    )
    gae_lambda: float = Field(
        default=DEFAULT_GAE_LAMBDA,
        description="GAE lambda parameter.",
    )
    clip_epsilon: float = Field(
        default=DEFAULT_CLIP_EPSILON,
        description="PPO clipping epsilon.",
    )
    entropy_coef: float = Field(
        default=DEFAULT_ENTROPY_COEF,
        description="Initial entropy regularization coefficient (decays to entropy_coef_end).",
    )
    entropy_coef_end: float = Field(
        default=DEFAULT_ENTROPY_COEF_END,
        description="Final entropy coefficient after decay.",
    )
    entropy_decay_episodes: int = Field(
        default=DEFAULT_ENTROPY_DECAY_EPISODES,
        description="Episodes over which entropy_coef linearly decays from start to end.",
    )
    value_loss_coef: float = Field(
        default=DEFAULT_VALUE_LOSS_COEF,
        description="Value loss coefficient.",
    )
    num_epochs: int = Field(
        default=DEFAULT_NUM_EPOCHS,
        description="Number of PPO epochs per rollout. "
        "Epoch 0 runs quantum circuits; epochs 1+ reuse cached spike probs.",
    )
    num_minibatches: int = Field(
        default=DEFAULT_NUM_MINIBATCHES,
        description="Number of minibatches per epoch.",
    )
    max_grad_norm: float = Field(
        default=DEFAULT_MAX_GRAD_NORM,
        description="Maximum gradient norm for clipping.",
    )
    rollout_buffer_size: int = Field(
        default=DEFAULT_ROLLOUT_BUFFER_SIZE,
        description="Number of steps to collect before PPO update.",
    )

    # Learning rates
    actor_lr: float = Field(
        default=DEFAULT_ACTOR_LR,
        description="Learning rate for actor (QSNN) parameters.",
    )
    critic_lr: float = Field(
        default=DEFAULT_CRITIC_LR,
        description="Learning rate for critic MLP.",
    )
    actor_weight_decay: float = Field(
        default=DEFAULT_ACTOR_WEIGHT_DECAY,
        description="L2 weight decay for actor parameters.",
    )

    # Critic architecture
    critic_hidden_dim: int = Field(
        default=DEFAULT_CRITIC_HIDDEN_DIM,
        description="Hidden layer dimension for the critic MLP.",
    )
    critic_num_layers: int = Field(
        default=DEFAULT_CRITIC_NUM_LAYERS,
        description="Number of hidden layers in the critic MLP.",
    )

    # Sensory feature extraction
    sensory_modules: list[ModuleName] | None = Field(
        default=None,
        description="List of sensory modules for feature extraction (None = legacy mode).",
    )

    # Learning rate scheduling
    lr_decay_episodes: int | None = Field(
        default=None,
        description="Episodes over which actor LR decays (None = no decay).",
    )
    lr_min_factor: float = Field(
        default=0.1,
        description="Minimum LR as fraction of initial LR after decay.",
    )

    @field_validator("num_sensory_neurons")
    @classmethod
    def validate_num_sensory_neurons(cls, v: int) -> int:
        """Validate num_sensory_neurons >= 1."""
        if v < MIN_SENSORY_NEURONS:
            msg = f"num_sensory_neurons must be >= {MIN_SENSORY_NEURONS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("num_hidden_neurons")
    @classmethod
    def validate_num_hidden_neurons(cls, v: int) -> int:
        """Validate num_hidden_neurons >= 1."""
        if v < MIN_HIDDEN_NEURONS:
            msg = f"num_hidden_neurons must be >= {MIN_HIDDEN_NEURONS}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("num_motor_neurons")
    @classmethod
    def validate_num_motor_neurons(cls, v: int) -> int:
        """Validate num_motor_neurons >= 2."""
        if v < MIN_MOTOR_NEURONS:
            msg = f"num_motor_neurons must be >= {MIN_MOTOR_NEURONS}, got {v}"
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

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold in (0, 1)."""
        if not (0 < v < 1):
            msg = f"threshold must be in (0, 1), got {v}"
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

    @field_validator("num_epochs")
    @classmethod
    def validate_num_epochs(cls, v: int) -> int:
        """Validate num_epochs >= 1."""
        if v < 1:
            msg = f"num_epochs must be >= 1, got {v}"
            raise ValueError(msg)
        return v


# ──────────────────────────────────────────────────────────────────────
# Brain
# ──────────────────────────────────────────────────────────────────────


class QSNNPPOBrain(ClassicalBrain):
    """Quantum Spiking Neural Network with PPO training.

    Uses a QSNN actor (QLIF quantum circuits with surrogate gradients)
    paired with a classical MLP critic trained via PPO.
    """

    def __init__(  # noqa: PLR0915
        self,
        config: QSNNPPOBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.num_actions = num_actions
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
        logger.info(f"QSNNPPOBrain using seed: {self.seed}")

        # Sensory modules
        self.sensory_modules = config.sensory_modules

        if config.sensory_modules is not None:
            self.input_dim = get_classical_feature_dimension(config.sensory_modules)
            logger.info(
                f"Using unified sensory modules: "
                f"{[m.value for m in config.sensory_modules]} "
                f"(input_dim={self.input_dim})",
            )
        else:
            self.input_dim = 2
            logger.info(
                "Using legacy 2-feature preprocessing (gradient_strength, rel_angle)",
            )

        # Data tracking
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        # QLIF network config
        self.num_sensory = config.num_sensory_neurons
        self.num_hidden = config.num_hidden_neurons
        self.num_motor = config.num_motor_neurons
        self.membrane_tau = config.membrane_tau
        self.threshold = config.threshold
        self.refractory_period = config.refractory_period
        self.shots = config.shots
        self.num_integration_steps = config.num_integration_steps
        self.leak_angle = (1 - self.membrane_tau) * np.pi

        # Initialize QSNN actor weights
        self._init_actor_weights()

        # Critic input dimension: raw sensory features + hidden spike rates
        critic_input_dim = self.input_dim + self.num_hidden
        self.critic = QSNNPPOCritic(
            input_dim=critic_input_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
        ).to(self.device)

        # Separate optimizers for actor (raw tensors) and critic (nn.Module)
        self.actor_optimizer = torch.optim.Adam(
            [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor],
            lr=config.actor_lr,
            weight_decay=config.actor_weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
        )

        # LR scheduling for actor
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None
        if config.lr_decay_episodes is not None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer,
                T_max=config.lr_decay_episodes,
                eta_min=config.actor_lr * config.lr_min_factor,
            )

        # Rollout buffer
        self.buffer = QSNNRolloutBuffer(
            config.rollout_buffer_size,
            self.device,
            rng=self.rng,
        )

        # State tracking
        self.training = True
        self.current_probabilities: np.ndarray | None = None
        self._episode_count = 0
        self._step_count = 0

        # Pending step data (stored in run_brain, consumed in learn)
        self._pending_features: np.ndarray | None = None
        self._pending_action: int = 0
        self._pending_log_prob: float = 0.0
        self._pending_value: float = 0.0
        self._pending_hidden_spikes: np.ndarray = np.zeros(self.num_hidden)

        # Qiskit backend (lazy init)
        self._backend = None

        # Count parameters
        actor_params = (
            self.num_sensory * self.num_hidden
            + self.num_hidden * self.num_motor
            + self.num_hidden
            + self.num_motor
        )
        critic_params = sum(p.numel() for p in self.critic.parameters())
        logger.info(
            f"QSNNPPOBrain initialized: "
            f"{self.num_sensory}->{self.num_hidden}->{self.num_motor} QLIF, "
            f"actor_params={actor_params}, critic_params={critic_params:,}, "
            f"total={actor_params + critic_params:,}",
        )

    def _init_actor_weights(self) -> None:
        """Initialize QSNN actor weight matrices and theta parameters."""
        self.W_sh = (
            torch.randn(self.num_sensory, self.num_hidden, device=self.device) * WEIGHT_INIT_SCALE
        )
        self.W_hm = (
            torch.randn(self.num_hidden, self.num_motor, device=self.device) * WEIGHT_INIT_SCALE
        )
        self.theta_hidden = torch.full(
            (self.num_hidden,),
            np.pi / 4,
            device=self.device,
        )
        self.theta_motor = torch.linspace(-0.3, 0.3, self.num_motor, device=self.device)

        # Enable gradients for surrogate gradient training
        self.W_sh.requires_grad_(True)  # noqa: FBT003
        self.W_hm.requires_grad_(True)  # noqa: FBT003
        self.theta_hidden.requires_grad_(True)  # noqa: FBT003
        self.theta_motor.requires_grad_(True)  # noqa: FBT003

        # Refractory state
        self.refractory_hidden = np.zeros(self.num_hidden, dtype=np.int32)
        self.refractory_motor = np.zeros(self.num_motor, dtype=np.int32)

    def _get_backend(self):  # noqa: ANN202
        """Get or create the Qiskit Aer backend."""
        if self._backend is None:
            self._backend = get_qiskit_backend(
                DeviceType(self.device.type) if hasattr(self.device, "type") else DeviceType.CPU,
                seed=self.seed,
            )
        return self._backend

    # ──────────────────────────────────────────────────────────────────
    # QLIF Forward Pass
    # ──────────────────────────────────────────────────────────────────

    def _multi_timestep(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute multi-timestep QLIF forward pass (non-differentiable).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (averaged_motor_spikes, averaged_hidden_spikes)
        """
        motor_acc = np.zeros(self.num_motor)
        hidden_acc = np.zeros(self.num_hidden)

        for _ in range(self.num_integration_steps):
            sensory_spikes = encode_sensory_spikes(features, self.num_sensory)

            hidden_spikes, self.refractory_hidden = execute_qlif_layer(
                sensory_spikes,
                self.W_sh,
                self.theta_hidden,
                self.refractory_hidden,
                backend=self._get_backend(),
                shots=self.shots,
                threshold=self.threshold,
                refractory_period=self.refractory_period,
                leak_angle=self.leak_angle,
            )
            motor_spikes, self.refractory_motor = execute_qlif_layer(
                hidden_spikes,
                self.W_hm,
                self.theta_motor,
                self.refractory_motor,
                backend=self._get_backend(),
                shots=self.shots,
                threshold=self.threshold,
                refractory_period=self.refractory_period,
                leak_angle=self.leak_angle,
            )

            motor_acc += motor_spikes
            hidden_acc += hidden_spikes

        return (
            motor_acc / self.num_integration_steps,
            hidden_acc / self.num_integration_steps,
        )

    def _multi_timestep_differentiable_caching(
        self,
        features: np.ndarray,
        cache_out: list[dict[str, list[float]]],
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Multi-timestep forward with gradient tracking, caching spike probs.

        Returns
        -------
        tuple[torch.Tensor, np.ndarray]
            (averaged_motor_spikes_tensor, averaged_hidden_spikes_np)
        """
        motor_acc = torch.zeros(self.num_motor, device=self.device)
        hidden_acc = np.zeros(self.num_hidden)

        for _ in range(self.num_integration_steps):
            sensory_spikes = encode_sensory_spikes(features, self.num_sensory)
            sensory_tensor = torch.tensor(
                sensory_spikes,
                dtype=torch.float32,
                device=self.device,
            )

            hidden_spikes, self.refractory_hidden = execute_qlif_layer_differentiable(
                sensory_tensor,
                self.W_sh,
                self.theta_hidden,
                self.refractory_hidden,
                backend=self._get_backend(),
                shots=self.shots,
                threshold=self.threshold,
                refractory_period=self.refractory_period,
                leak_angle=self.leak_angle,
                device=self.device,
            )
            step_cache: dict[str, list[float]] = {
                "hidden": hidden_spikes.detach().cpu().tolist(),
            }
            hidden_acc += hidden_spikes.detach().cpu().numpy()

            motor_spikes, self.refractory_motor = execute_qlif_layer_differentiable(
                hidden_spikes,
                self.W_hm,
                self.theta_motor,
                self.refractory_motor,
                backend=self._get_backend(),
                shots=self.shots,
                threshold=self.threshold,
                refractory_period=self.refractory_period,
                leak_angle=self.leak_angle,
                device=self.device,
            )
            step_cache["motor"] = motor_spikes.detach().cpu().tolist()

            motor_acc = motor_acc + motor_spikes
            cache_out.append(step_cache)

        return (
            motor_acc / self.num_integration_steps,
            hidden_acc / self.num_integration_steps,
        )

    def _multi_timestep_differentiable_cached(
        self,
        features: np.ndarray,
        cached_timestep: list[dict[str, list[float]]],
    ) -> torch.Tensor:
        """Multi-timestep forward using cached spike probs.

        Returns
        -------
        torch.Tensor
            Averaged motor spike probabilities (with grad_fn).
        """
        motor_acc = torch.zeros(self.num_motor, device=self.device)

        for step_idx in range(self.num_integration_steps):
            sensory_spikes = encode_sensory_spikes(features, self.num_sensory)
            sensory_tensor = torch.tensor(
                sensory_spikes,
                dtype=torch.float32,
                device=self.device,
            )

            hidden_spikes, self.refractory_hidden = execute_qlif_layer_differentiable_cached(
                sensory_tensor,
                self.W_sh,
                self.theta_hidden,
                self.refractory_hidden,
                cached_spike_probs=cached_timestep[step_idx]["hidden"],
                threshold=self.threshold,
                refractory_period=self.refractory_period,
                device=self.device,
            )

            motor_spikes, self.refractory_motor = execute_qlif_layer_differentiable_cached(
                hidden_spikes,
                self.W_hm,
                self.theta_motor,
                self.refractory_motor,
                cached_spike_probs=cached_timestep[step_idx]["motor"],
                threshold=self.threshold,
                refractory_period=self.refractory_period,
                device=self.device,
            )

            motor_acc = motor_acc + motor_spikes

        return motor_acc / self.num_integration_steps

    # ──────────────────────────────────────────────────────────────────
    # Feature Extraction
    # ──────────────────────────────────────────────────────────────────

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Preprocess BrainParams to extract features.

        Two modes:
        1. **Unified sensory mode** (when sensory_modules is set)
        2. **Legacy mode** (default): gradient strength + relative angle
        """
        if self.sensory_modules is not None:
            return extract_classical_features(params, self.sensory_modules)

        grad_strength = float(params.gradient_strength or 0.0)
        grad_direction = float(params.gradient_direction or 0.0)
        direction_map = {
            Direction.UP: np.pi / 2,
            Direction.DOWN: -np.pi / 2,
            Direction.LEFT: np.pi,
            Direction.RIGHT: 0.0,
        }
        agent_facing_angle = direction_map.get(
            params.agent_direction or Direction.UP,
            np.pi / 2,
        )
        relative_angle = (grad_direction - agent_facing_angle + np.pi) % (2 * np.pi) - np.pi
        rel_angle_norm = relative_angle / np.pi

        return np.array([grad_strength, rel_angle_norm], dtype=np.float32)

    def _get_critic_input(
        self,
        features: np.ndarray,
        hidden_spikes: np.ndarray,
    ) -> torch.Tensor:
        """Build critic input from raw features and hidden spike rates."""
        critic_features = np.concatenate([features, hidden_spikes])
        return torch.tensor(
            critic_features,
            dtype=torch.float32,
            device=self.device,
        )

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
        """Run QSNN actor to select an action."""
        features = self.preprocess(params)

        # QLIF forward pass (non-differentiable for action selection)
        motor_probs, hidden_spikes = self._multi_timestep(features)

        # Convert spike probs to action logits
        motor_probs = np.clip(motor_probs, 1e-8, 1.0 - 1e-8)
        logits = (motor_probs - 0.5) * self.config.logit_scale

        # Softmax action probabilities
        exp_probs = np.exp(logits - np.max(logits))
        action_probs = exp_probs / np.sum(exp_probs)

        # Sample action
        action_idx = self.rng.choice(self.num_actions, p=action_probs)
        action_name = self.action_set[action_idx]

        # Compute log probability
        log_prob = float(np.log(action_probs[action_idx] + 1e-8))

        # Compute critic value
        with torch.no_grad():
            critic_input = self._get_critic_input(features, hidden_spikes)
            value = self.critic(critic_input).item()

        # Store pending data for learn()
        self._pending_features = features
        self._pending_action = action_idx
        self._pending_log_prob = log_prob
        self._pending_value = value
        self._pending_hidden_spikes = hidden_spikes

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
                f"QSNN-PPO step {self._step_count}: "
                f"motor={np.array2string(motor_probs, precision=3)}, "
                f"probs={np.array2string(action_probs, precision=3)}, "
                f"value={value:.4f}, "
                f"W_sh_norm={torch.norm(self.W_sh).item():.3f}, "
                f"W_hm_norm={torch.norm(self.W_hm).item():.3f}",
            )

        return [self.latest_data.action]

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Add experience to rollout buffer and trigger PPO update when full.

        The buffer accumulates across episodes until full, then triggers
        a PPO update and resets. This ensures PPO always has enough data
        for stable advantage estimates and minibatch updates.
        """
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
                hidden_spike_rates=self._pending_hidden_spikes,
            )

        # PPO update only when buffer is full
        if self.buffer.is_full():
            self._perform_ppo_update()
            self.buffer.reset()

    def _perform_ppo_update(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Perform PPO update with quantum caching across epochs."""
        if len(self.buffer) == 0:
            return

        # Compute last value for GAE bootstrap
        if self._pending_features is not None:
            with torch.no_grad():
                critic_input = self._get_critic_input(
                    self._pending_features,
                    self._pending_hidden_spikes,
                )
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

        # Multi-epoch PPO update with quantum caching
        # Epoch 0: run quantum circuits and cache spike probs
        # Epochs 1+: reuse cached spike probs, recompute ry_angles
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        total_grad_sh = 0.0
        total_grad_hm = 0.0
        total_grad_th = 0.0
        total_grad_tm = 0.0
        motor_spike_sum = np.zeros(self.num_motor)
        motor_spike_count = 0
        num_updates = 0

        entropy_coef = self._get_entropy_coef()

        for epoch in range(self.config.num_epochs):
            # On epoch 0, build spike caches for all buffer steps
            if epoch == 0 and self.config.num_epochs > 1:
                self.buffer.spike_caches = []
                for t in range(len(self.buffer)):
                    step_cache: list[dict[str, list[float]]] = []
                    self.refractory_hidden.fill(0)
                    self.refractory_motor.fill(0)
                    motor_spikes, _hidden = self._multi_timestep_differentiable_caching(
                        self.buffer.features[t],
                        step_cache,
                    )
                    self.buffer.spike_caches.append(step_cache)
                    # Don't need the result tensors here; they'll be
                    # recomputed per-minibatch below

            for batch in self.buffer.get_minibatches(
                self.config.num_minibatches,
                returns,
                advantages,
            ):
                # Per-step forward passes (quantum circuits are not batchable)
                log_probs_list: list[torch.Tensor] = []
                entropies_list: list[torch.Tensor] = []
                values_list: list[torch.Tensor] = []

                for idx_tensor in batch["indices"]:
                    idx = int(idx_tensor.item())
                    features = self.buffer.features[idx]
                    action_idx = self.buffer.actions[idx]

                    # Reset refractory state for each step's forward pass
                    self.refractory_hidden.fill(0)
                    self.refractory_motor.fill(0)

                    # QLIF forward pass (differentiable)
                    if epoch == 0 and self.config.num_epochs == 1:
                        # Single-epoch mode: no caching needed
                        step_cache_single: list[dict[str, list[float]]] = []
                        motor_spikes, _hidden_np = self._multi_timestep_differentiable_caching(
                            features,
                            step_cache_single,
                        )
                        # Store cache for potential future use
                        if len(self.buffer.spike_caches) <= idx:
                            self.buffer.spike_caches.extend(
                                [[] for _ in range(idx + 1 - len(self.buffer.spike_caches))],
                            )
                        self.buffer.spike_caches[idx] = step_cache_single
                    elif epoch == 0:
                        # Multi-epoch, epoch 0: use already-cached data
                        # (cached in the pre-pass above)
                        motor_spikes = self._multi_timestep_differentiable_cached(
                            features,
                            self.buffer.spike_caches[idx],
                        )
                    else:
                        # Epochs 1+: reuse cached spike probs
                        motor_spikes = self._multi_timestep_differentiable_cached(
                            features,
                            self.buffer.spike_caches[idx],
                        )

                    # Convert to action probabilities
                    motor_clipped = torch.clamp(motor_spikes, 1e-8, 1.0 - 1e-8)
                    logits = (motor_clipped - 0.5) * self.config.logit_scale
                    action_probs = torch.softmax(logits, dim=-1)

                    # Track motor spike probs for diagnostics
                    motor_spike_sum += motor_clipped.detach().cpu().numpy()
                    motor_spike_count += 1

                    log_prob = torch.log(action_probs[action_idx] + 1e-8)
                    log_probs_list.append(log_prob)

                    entropy = -torch.sum(
                        action_probs * torch.log(action_probs + 1e-10),
                    )
                    entropies_list.append(entropy)

                    # Critic forward pass
                    hidden_for_critic = self.buffer.hidden_spike_rates[idx]
                    critic_input = self._get_critic_input(
                        features,
                        hidden_for_critic,
                    )
                    value = self.critic(critic_input)
                    values_list.append(value)

                # Stack per-step results
                new_log_probs = torch.stack(log_probs_list)
                mean_entropy = torch.stack(entropies_list).mean()
                values = torch.stack(values_list)

                # PPO policy loss
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
                    )
                    * batch["advantages"]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Track clip fraction
                with torch.no_grad():
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.config.clip_epsilon).float().mean().item()
                    )
                    total_clip_fraction += clip_frac
                    approx_kl = (batch["old_log_probs"] - new_log_probs).mean().item()
                    total_approx_kl += approx_kl

                # Value loss (Huber for robustness to extreme penalties)
                value_loss = torch.nn.functional.smooth_l1_loss(
                    values,
                    batch["returns"],
                )

                # Combined loss for actor
                actor_loss = policy_loss - entropy_coef * mean_entropy

                # Actor backward and step
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_params = [
                    self.W_sh,
                    self.W_hm,
                    self.theta_hidden,
                    self.theta_motor,
                ]

                # Capture pre-clip gradient norms
                with torch.no_grad():
                    grad_sh = torch.norm(self.W_sh.grad).item() if self.W_sh.grad is not None else 0
                    grad_hm = torch.norm(self.W_hm.grad).item() if self.W_hm.grad is not None else 0
                    grad_th = (
                        torch.norm(self.theta_hidden.grad).item()
                        if self.theta_hidden.grad is not None
                        else 0
                    )
                    grad_tm = (
                        torch.norm(self.theta_motor.grad).item()
                        if self.theta_motor.grad is not None
                        else 0
                    )
                    total_grad_sh += grad_sh
                    total_grad_hm += grad_hm
                    total_grad_th += grad_th
                    total_grad_tm += grad_tm

                torch.nn.utils.clip_grad_norm_(
                    actor_params,
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

        # Post-update: clamp weights and theta norms
        self._clamp_weights()

        # Logging
        if num_updates > 0:
            avg_policy = total_policy_loss / num_updates
            avg_value = total_value_loss / num_updates
            avg_entropy = total_entropy / num_updates
            self.latest_data.loss = avg_policy

            actor_lr = self.actor_optimizer.param_groups[0]["lr"]
            adv_mean = advantages.mean().item()
            adv_std = advantages.std().item()
            motor_spike_mean = motor_spike_sum / max(motor_spike_count, 1)
            motor_spike_str = np.array2string(motor_spike_mean, precision=4, separator=",")
            logger.info(
                f"QSNN-PPO update: policy_loss={avg_policy:.4f}, "
                f"value_loss={avg_value:.4f}, entropy={avg_entropy:.4f}, "
                f"entropy_coef={entropy_coef:.4f}, "
                f"clip_frac={total_clip_fraction / num_updates:.3f}, "
                f"approx_kl={total_approx_kl / num_updates:.4f}, "
                f"grad_sh={total_grad_sh / num_updates:.4f}, "
                f"grad_hm={total_grad_hm / num_updates:.4f}, "
                f"grad_th={total_grad_th / num_updates:.4f}, "
                f"grad_tm={total_grad_tm / num_updates:.4f}, "
                f"W_sh_norm={torch.norm(self.W_sh).item():.4f}, "
                f"W_hm_norm={torch.norm(self.W_hm).item():.4f}, "
                f"theta_h_norm={torch.norm(self.theta_hidden).item():.4f}, "
                f"theta_m_norm={torch.norm(self.theta_motor).item():.4f}, "
                f"motor_spike_probs={motor_spike_str}, "
                f"adv_mean={adv_mean:.4f}, adv_std={adv_std:.4f}, "
                f"buffer_size={buffer_len}, "
                f"actor_lr={actor_lr:.6f}, "
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

    def _clamp_weights(self) -> None:
        """Clamp actor weights, theta motor max norm, and theta hidden min norm."""
        with torch.no_grad():
            for p in [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor]:
                p.clamp_(-self.config.weight_clip, self.config.weight_clip)

            tm_norm = torch.norm(self.theta_motor)
            if tm_norm > self.config.theta_motor_max_norm:
                self.theta_motor.mul_(self.config.theta_motor_max_norm / tm_norm)

            # Prevent theta_hidden collapse (preserves hidden layer capacity)
            th_norm = torch.norm(self.theta_hidden)
            if th_norm < self.config.theta_hidden_min_norm and th_norm > 0:
                self.theta_hidden.mul_(self.config.theta_hidden_min_norm / th_norm)

    # ──────────────────────────────────────────────────────────────────
    # Episode Lifecycle
    # ──────────────────────────────────────────────────────────────────

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for QSNNPPOBrain."""

    def prepare_episode(self) -> None:
        """Prepare for a new episode."""
        self.refractory_hidden.fill(0)
        self.refractory_motor.fill(0)
        self._step_count = 0

    def post_process_episode(
        self,
        *,
        episode_success: bool | None = None,  # noqa: ARG002
    ) -> None:
        """Post-process after each episode."""
        self._episode_count += 1

        # Step LR scheduler
        if self.scheduler is not None:
            self.scheduler.step()

    def copy(self) -> QSNNPPOBrain:
        """QSNNPPOBrain does not support copying."""
        error_msg = "QSNNPPOBrain does not support copying. Use deepcopy if needed."
        raise NotImplementedError(error_msg)

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
        """Not applicable to QSNNPPOBrain."""
        error_msg = "QSNNPPOBrain does not have a quantum circuit."
        raise NotImplementedError(error_msg)

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,
        learning_rate: float = 0.01,
        **kwargs: object,
    ) -> None:
        """Not used - PPO uses its own optimizers."""
