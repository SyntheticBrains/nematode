"""
QLIF-LSTM Brain Architecture.

A quantum-enhanced LSTM where forget and input gates use QLIF quantum neuron
measurements (via surrogate gradients) instead of classical sigmoid activations,
trained via recurrent PPO with chunk-based truncated BPTT. This is the first
temporal architecture in the codebase, introducing within-episode memory.

Architecture::

    Sensory Input (e.g. food_chemotaxis + nociception = 4 features)
        |
        v
    encode_sensory_spikes() -> spike probs
        |
        v
    QLIFLSTMCell (custom LSTM cell)
    +-------------------------------------------+
    | z = [x_t, h_{t-1}]  (concatenation)       |
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
    h_t -> logits -> action    [features, h_t.detach()] -> V(s)

    Training: Recurrent PPO with chunk-based truncated BPTT
    1. Collect rollout_buffer_size steps, storing (h_t, c_t) per step
    2. Split buffer into bptt_chunk_length chunks
    3. For each PPO epoch:
       a. Shuffle chunks
       b. Re-run LSTM forward from stored (h_0, c_0) per chunk
       c. PPO clipped surrogate loss on actor + Huber loss on critic
    4. Clear buffer, repeat

References
----------
- Brand & Petruccione (2024) "A quantum leaky integrate-and-fire spiking
  neuron and network." npj Quantum Information, 10(1), 16.
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- Chen et al. (2022) "Quantum Long Short-Term Memory" arXiv:2009.01783
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Self

import numpy as np
import torch
from pydantic import Field, field_validator, model_validator
from torch import nn

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._qlif_layers import (
    QLIFSurrogateSpike,
    build_qlif_circuit,
    encode_sensory_spikes,
)
from quantumnematode.brain.arch._quantum_utils import get_qiskit_backend
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.brain.modules import (
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_LSTM_HIDDEN_DIM = 32
DEFAULT_MEMBRANE_TAU = 0.9
DEFAULT_REFRACTORY_PERIOD = 0
DEFAULT_SHOTS = 1024
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_EPSILON = 0.2
DEFAULT_ENTROPY_COEF = 0.05
DEFAULT_ENTROPY_COEF_END = 0.005
DEFAULT_ENTROPY_DECAY_EPISODES = 200
DEFAULT_VALUE_LOSS_COEF = 0.5
DEFAULT_NUM_EPOCHS = 2
DEFAULT_ROLLOUT_BUFFER_SIZE = 256
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_ACTOR_LR = 0.003
DEFAULT_CRITIC_LR = 0.001
DEFAULT_CRITIC_HIDDEN_DIM = 64
DEFAULT_CRITIC_NUM_LAYERS = 2
DEFAULT_BPTT_CHUNK_LENGTH = 16

# Validation constants
MIN_LSTM_HIDDEN_DIM = 2
MIN_SHOTS = 100
MIN_BPTT_CHUNK_LENGTH = 4


# ──────────────────────────────────────────────────────────────────────
# Critic MLP
# ──────────────────────────────────────────────────────────────────────


class QLIFLSTMCritic(nn.Module):
    """Classical MLP critic for QLIF-LSTM.

    Input is the concatenation of raw sensory features and LSTM hidden
    state (detached from the actor's autograd graph).
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
# QLIF-LSTM Cell
# ──────────────────────────────────────────────────────────────────────


class QLIFLSTMCell(nn.Module):
    """Custom LSTM cell with QLIF quantum gates.

    Replaces forget and input gate sigmoid activations with QLIF quantum
    neuron measurements via QLIFSurrogateSpike. Cell candidate (tanh) and
    output gate (sigmoid) remain classical.

    When ``use_quantum_gates=False``, uses standard sigmoid for all gates
    (classical ablation mode).
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        use_quantum_gates: bool = True,
        shots: int = DEFAULT_SHOTS,
        membrane_tau: float = DEFAULT_MEMBRANE_TAU,
        refractory_period: int = DEFAULT_REFRACTORY_PERIOD,
        backend: Any = None,  # noqa: ANN401
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_quantum_gates = use_quantum_gates
        self.shots = shots
        self.membrane_tau = membrane_tau
        self.refractory_period = refractory_period
        self.leak_angle = (1 - membrane_tau) * np.pi
        self._backend = backend
        self._device = device or torch.device("cpu")

        combined_dim = input_dim + hidden_dim

        # Four separate linear projections for explicit gate access
        self.W_f = nn.Linear(combined_dim, hidden_dim)  # forget gate
        self.W_i = nn.Linear(combined_dim, hidden_dim)  # input gate
        self.W_c = nn.Linear(combined_dim, hidden_dim)  # cell candidate
        self.W_o = nn.Linear(combined_dim, hidden_dim)  # output gate

        # Trainable membrane potential parameters for QLIF gates
        # One theta per gate neuron (forget and input gates)
        self.theta_forget = nn.Parameter(
            torch.full((hidden_dim,), np.pi / 2, device=self._device),
        )
        self.theta_input = nn.Parameter(
            torch.full((hidden_dim,), np.pi / 2, device=self._device),
        )

        # Initialize linear layers
        for layer in [self.W_f, self.W_i, self.W_c, self.W_o]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

        # Bias forget gate toward remembering (standard LSTM practice)
        if self.W_f.bias is not None:
            nn.init.constant_(self.W_f.bias, 1.0)

    def set_backend(self, backend: Any) -> None:  # noqa: ANN401
        """Set the Qiskit backend for quantum circuit execution."""
        self._backend = backend

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the QLIF-LSTM cell.

        Parameters
        ----------
        x_t : torch.Tensor
            Input features at current timestep, shape (input_dim,).
        h_prev : torch.Tensor
            Previous hidden state, shape (hidden_dim,).
        c_prev : torch.Tensor
            Previous cell state, shape (hidden_dim,).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (h_t, c_t) — new hidden state and cell state.
        """
        # Combined input
        z = torch.cat([x_t, h_prev])

        # Linear projections
        proj_f = self.W_f(z)
        proj_i = self.W_i(z)
        proj_c = self.W_c(z)
        proj_o = self.W_o(z)

        # Gate activations
        if self.use_quantum_gates:
            f_t = self._qlif_gate(proj_f, self.theta_forget)
            i_t = self._qlif_gate(proj_i, self.theta_input)
        else:
            f_t = torch.sigmoid(proj_f)
            i_t = torch.sigmoid(proj_i)

        c_hat_t = torch.tanh(proj_c)
        o_t = torch.sigmoid(proj_o)

        # Cell and hidden state update
        c_t = f_t * c_prev + i_t * c_hat_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

    def _qlif_gate(
        self,
        linear_output: torch.Tensor,
        theta_membrane: nn.Parameter,
    ) -> torch.Tensor:
        """Compute gate activation using QLIF quantum neurons.

        Each gate neuron runs one QLIF circuit. The forward value is the
        quantum measurement P(|1⟩), and the backward uses the sigmoid
        surrogate gradient.

        Circuits are batched into a single ``backend.run()`` call for
        performance (hidden_dim circuits per gate instead of hidden_dim
        sequential calls).

        Parameters
        ----------
        linear_output : torch.Tensor
            Linear projection output, shape (hidden_dim,).
        theta_membrane : nn.Parameter
            Membrane potential parameters, shape (hidden_dim,).

        Returns
        -------
        torch.Tensor
            Gate activations in [0, 1], shape (hidden_dim,).
        """
        fan_in_scale = (self.W_f.in_features) ** 0.5

        # Vectorized differentiable RY angles (kept on autograd graph)
        scaled_inputs = linear_output / fan_in_scale
        ry_angles = theta_membrane + torch.tanh(scaled_inputs) * torch.pi

        # Build all circuits for this gate (detached values)
        scaled_np = scaled_inputs.detach().cpu().numpy()
        theta_np = theta_membrane.detach().cpu().numpy()
        circuits = [
            build_qlif_circuit(float(scaled_np[j]), float(theta_np[j]), self.leak_angle)
            for j in range(self.hidden_dim)
        ]

        # Batched execution: single backend.run() call for all neurons
        job = self._backend.run(circuits, shots=self.shots)
        result = job.result()

        # Extract spike probabilities from batched results
        gate_probs: list[torch.Tensor] = []
        for j in range(self.hidden_dim):
            counts = result.get_counts(j)
            quantum_spike_prob = counts.get("1", 0) / self.shots

            spike_prob: torch.Tensor = QLIFSurrogateSpike.apply(  # type: ignore[assignment]
                ry_angles[j],
                quantum_spike_prob,
            )
            gate_probs.append(spike_prob)

        return torch.stack(gate_probs)


# ──────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ──────────────────────────────────────────────────────────────────────


class QLIFLSTMRolloutBuffer:
    """Rollout buffer for QLIF-LSTM that stores LSTM hidden states.

    Stores per-step LSTM hidden states (h_t, c_t) for chunk-based
    truncated BPTT during PPO updates.
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
        # LSTM hidden states at each step (for chunk-based BPTT)
        self.h_states: list[torch.Tensor] = []
        self.c_states: list[torch.Tensor] = []
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
        c_state: torch.Tensor,
    ) -> None:
        """Add a single experience to the buffer."""
        self.features.append(features)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.h_states.append(h_state.detach().clone())
        self.c_states.append(c_state.detach().clone())
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

        Yields
        ------
        dict
            Dictionary with 'start', 'end', 'h_init', 'c_init',
            'features', 'actions', 'old_log_probs', 'returns',
            'advantages', 'dones'.
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


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


class QLIFLSTMBrainConfig(BrainConfig):
    """Configuration for the QLIFLSTMBrain architecture.

    Uses unified sensory module feature extraction from brain/modules.py.
    """

    # LSTM architecture
    lstm_hidden_dim: int = Field(
        default=DEFAULT_LSTM_HIDDEN_DIM,
        description="LSTM hidden state dimension.",
    )

    # QLIF gate parameters
    shots: int = Field(
        default=DEFAULT_SHOTS,
        description="Number of quantum measurement shots per gate neuron.",
    )
    membrane_tau: float = Field(
        default=DEFAULT_MEMBRANE_TAU,
        description="Leak time constant for QLIF neurons in (0, 1].",
    )
    refractory_period: int = Field(
        default=DEFAULT_REFRACTORY_PERIOD,
        description="Timesteps to suppress activity after firing.",
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
        description="Initial entropy regularization coefficient.",
    )
    entropy_coef_end: float = Field(
        default=DEFAULT_ENTROPY_COEF_END,
        description="Final entropy coefficient after decay.",
    )
    entropy_decay_episodes: int = Field(
        default=DEFAULT_ENTROPY_DECAY_EPISODES,
        description="Episodes over which entropy_coef decays.",
    )
    value_loss_coef: float = Field(
        default=DEFAULT_VALUE_LOSS_COEF,
        description="Value loss coefficient.",
    )
    num_epochs: int = Field(
        default=DEFAULT_NUM_EPOCHS,
        description="Number of PPO epochs per rollout.",
    )
    max_grad_norm: float = Field(
        default=DEFAULT_MAX_GRAD_NORM,
        description="Maximum gradient norm for clipping.",
    )
    rollout_buffer_size: int = Field(
        default=DEFAULT_ROLLOUT_BUFFER_SIZE,
        description="Number of steps to collect before PPO update.",
    )

    # Recurrent PPO
    bptt_chunk_length: int = Field(
        default=DEFAULT_BPTT_CHUNK_LENGTH,
        description="Chunk length for truncated BPTT.",
    )

    # Learning rates
    actor_lr: float = Field(
        default=DEFAULT_ACTOR_LR,
        gt=0,
        description="Learning rate for actor (LSTM cell + actor head) parameters.",
    )
    critic_lr: float = Field(
        default=DEFAULT_CRITIC_LR,
        gt=0,
        description="Learning rate for critic MLP.",
    )
    lr_warmup_episodes: int | None = Field(
        default=None,
        description="Episodes to warm up LR from lr_warmup_start to actor_lr (None = no warmup).",
    )
    lr_warmup_start: float | None = Field(
        default=None,
        description="Starting LR for warmup (None = 10% of actor_lr).",
    )
    lr_decay_episodes: int | None = Field(
        default=None,
        description="Episodes over which LR decays to lr_decay_end (None = no decay).",
    )
    lr_decay_end: float | None = Field(
        default=None,
        description="Final LR after decay (None = 10% of actor_lr).",
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

    # Ablation
    use_quantum_gates: bool = Field(
        default=True,
        description="Use QLIF quantum gates (True) or classical sigmoid (False).",
    )

    # Sensory feature extraction (required)
    sensory_modules: list[ModuleName] = Field(
        description="List of sensory modules for feature extraction.",
    )

    @field_validator("sensory_modules")
    @classmethod
    def validate_sensory_modules(cls, v: list[ModuleName]) -> list[ModuleName]:
        """Validate sensory_modules is non-empty."""
        if not v:
            msg = "sensory_modules must be non-empty"
            raise ValueError(msg)
        return v

    # Device
    device_type: DeviceType = Field(
        default=DeviceType.CPU,
        description="Device for tensor computation.",
    )

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
# Brain
# ──────────────────────────────────────────────────────────────────────


class QLIFLSTMBrain(ClassicalBrain):
    """Quantum QLIF-LSTM brain with recurrent PPO training.

    Uses a custom LSTM cell where forget and input gates are driven by
    QLIF quantum neuron measurements, with PPO training via chunk-based
    truncated BPTT.
    """

    def __init__(
        self,
        config: QLIFLSTMBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.num_actions = num_actions
        self._device_type = device
        self.device = torch.device(device.to_torch_device_str())
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
        logger.info(f"QLIFLSTMBrain using seed: {self.seed}")

        # Sensory modules
        self.sensory_modules = config.sensory_modules

        self.input_dim = get_classical_feature_dimension(config.sensory_modules)
        logger.info(
            f"Using sensory modules: "
            f"{[m.value for m in config.sensory_modules]} "
            f"(input_dim={self.input_dim})",
        )

        # Data tracking
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        # Qiskit backend (lazy init)
        self._backend = None

        # Build networks, optimizers, buffer, and state
        self._build_networks_and_state(config, num_actions)

    def _build_networks_and_state(
        self,
        config: QLIFLSTMBrainConfig,
        num_actions: int,
    ) -> None:
        """Initialize networks, optimizers, rollout buffer, and tracking state."""
        # QLIF-LSTM cell
        self.lstm_cell = QLIFLSTMCell(
            input_dim=self.input_dim,
            hidden_dim=config.lstm_hidden_dim,
            use_quantum_gates=config.use_quantum_gates,
            shots=config.shots,
            membrane_tau=config.membrane_tau,
            refractory_period=config.refractory_period,
            device=self.device,
        ).to(self.device)

        # Actor head: maps [features, h_t] to action logits
        # Raw features give direct access to current sensory signals (gradients, contact)
        # while h_t provides temporal context from LSTM memory
        actor_input_dim = self.input_dim + config.lstm_hidden_dim
        self.actor_head = nn.Linear(
            actor_input_dim,
            num_actions,
        ).to(self.device)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        if self.actor_head.bias is not None:
            nn.init.constant_(self.actor_head.bias, 0.0)

        # Critic: raw features + h_t (detached)
        critic_input_dim = self.input_dim + config.lstm_hidden_dim
        self.critic = QLIFLSTMCritic(
            input_dim=critic_input_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
        ).to(self.device)

        # Optimizers: actor includes LSTM cell + actor head
        self.actor_optimizer = torch.optim.Adam(
            list(self.lstm_cell.parameters()) + list(self.actor_head.parameters()),
            lr=config.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
        )

        # Episode counter (must be set before LR scheduling)
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
            for p in list(self.lstm_cell.parameters()) + list(self.actor_head.parameters())
        )
        critic_params = sum(p.numel() for p in self.critic.parameters())
        logger.info(
            f"QLIFLSTMBrain initialized: "
            f"input_dim={self.input_dim}, hidden_dim={config.lstm_hidden_dim}, "
            f"quantum_gates={'ON' if config.use_quantum_gates else 'OFF'}, "
            f"actor_params={actor_params:,}, critic_params={critic_params:,}, "
            f"total={actor_params + critic_params:,}",
        )

    def _get_backend(self) -> Any:  # noqa: ANN401
        """Get or create the Qiskit Aer backend."""
        if self._backend is None:
            self._backend = get_qiskit_backend(
                self._device_type,
                seed=self.seed,
            )
            self.lstm_cell.set_backend(self._backend)
        return self._backend

    # ──────────────────────────────────────────────────────────────────
    # Feature Extraction
    # ──────────────────────────────────────────────────────────────────

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Preprocess BrainParams to extract features via sensory modules."""
        return extract_classical_features(params, self.sensory_modules)

    def _get_actor_input(
        self,
        features: np.ndarray,
        h_state: torch.Tensor,
    ) -> torch.Tensor:
        """Build actor input from raw features and LSTM hidden state."""
        features_t = torch.tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        )
        return torch.cat([features_t, h_state])

    def _get_critic_input(
        self,
        features: np.ndarray,
        h_state: torch.Tensor,
    ) -> torch.Tensor:
        """Build critic input from raw features and LSTM hidden state."""
        features_t = torch.tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        )
        return torch.cat([features_t, h_state.detach()])

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
        """Run QLIF-LSTM to select an action."""
        features = self.preprocess(params)

        # Ensure backend is initialized for quantum gates
        if self.config.use_quantum_gates:
            self._get_backend()

        # Encode sensory input
        sensory_spikes = encode_sensory_spikes(features, len(features))
        x_t = torch.tensor(sensory_spikes, dtype=torch.float32, device=self.device)

        # Store pre-step hidden state for buffer
        h_pre = self.h_t.detach().clone()
        c_pre = self.c_t.detach().clone()

        # LSTM cell forward pass (no gradient during action selection)
        with torch.no_grad():
            h_new, c_new = self.lstm_cell(x_t, self.h_t, self.c_t)

        # Update hidden state
        self.h_t = h_new
        self.c_t = c_new

        # Actor: compute action logits from [features, h_t]
        with torch.no_grad():
            actor_input = self._get_actor_input(features, self.h_t)
            logits = self.actor_head(actor_input)
            action_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Sample action
        action_idx = self.rng.choice(self.num_actions, p=action_probs)
        action_name = self.action_set[action_idx]

        # Compute log probability
        log_prob = float(np.log(action_probs[action_idx] + 1e-8))

        # Compute critic value
        with torch.no_grad():
            critic_input = self._get_critic_input(features, self.h_t)
            value = self.critic(critic_input).item()

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
                f"QLIF-LSTM step {self._step_count}: "
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

        # Compute last value for GAE bootstrap
        if self._pending_features is not None:
            with torch.no_grad():
                critic_input = self._get_critic_input(
                    self._pending_features,
                    self.h_t,
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
                    sensory_spikes = encode_sensory_spikes(features, len(features))
                    x_t = torch.tensor(
                        sensory_spikes,
                        dtype=torch.float32,
                        device=self.device,
                    )

                    # LSTM forward (differentiable)
                    h, c = self.lstm_cell(x_t, h, c)

                    # Actor
                    actor_input = self._get_actor_input(features, h)
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
                    critic_input = self._get_critic_input(features, h)
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
                value_loss = torch.nn.functional.smooth_l1_loss(
                    values,
                    chunk["returns"],
                )

                # Combined loss for actor
                actor_loss = policy_loss - entropy_coef * mean_entropy

                # Actor backward and step
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.lstm_cell.parameters()) + list(self.actor_head.parameters()),
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
            avg_policy = total_policy_loss / num_updates
            avg_value = total_value_loss / num_updates
            avg_entropy = total_entropy / num_updates
            self.latest_data.loss = avg_policy

            logger.info(
                f"QLIF-LSTM PPO update: policy_loss={avg_policy:.4f}, "
                f"value_loss={avg_value:.4f}, entropy={avg_entropy:.4f}, "
                f"entropy_coef={entropy_coef:.4f}, "
                f"clip_frac={total_clip_fraction / num_updates:.3f}, "
                f"buffer_size={buffer_len}, "
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
        """No-op for QLIFLSTMBrain."""

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
        """Not applicable to QLIFLSTMBrain."""
        error_msg = "QLIFLSTMBrain does not have a standalone quantum circuit."
        raise NotImplementedError(error_msg)

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,
        learning_rate: float = 0.01,
        **kwargs: object,
    ) -> None:
        """Not used — PPO uses its own optimizers."""
