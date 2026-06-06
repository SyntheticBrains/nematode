r"""Spiking PPO Brain Architecture.

A classical recurrent PPO brain whose recurrent core is a **recurrent adaptive
leaky-integrate-and-fire** (LIF) spiking layer. The spiking core carries its
membrane / adaptation / spike state across env-steps (one LIF tick per step,
like a recurrent network), and a non-spiking leaky-integrator readout integrates
the hidden spikes into a smooth output membrane that supplies the action logits.
The deployed policy is the spiking actor; the readout is a linear integrator.

Key Features
------------
- **Recurrent adaptive-LIF core**: learnable per-neuron membrane decay, an
  adaptive (spike-frequency) threshold, and a learnable recurrent spike-feedback
  current. A single carried neuron state ``(v, a, s, m)`` (no LSTM cell-state
  duality).
- **Leaky-integrator readout**: a non-spiking output membrane integrates the
  hidden spikes; the membrane (not the binary spikes) is the action logits, for
  smoother policy gradients.
- **Plain-ANN critic on the detached membrane**: a small MLP over the hidden
  layer's detached membrane potential, keeping value-loss gradients out of the
  recurrent representation.
- **Surrogate-gradient training with a schedulable slope**: spikes are
  non-differentiable; a sigmoid-derivative surrogate (shallow slope, optionally
  sharpened over episodes) carries gradients through the spike during the
  truncated-BPTT replay.
- **Chunk-based truncated BPTT**: memory-efficient recurrent training over
  sequential chunks, detaching the chunk-start neuron state at the BPTT
  boundary.

Architecture::

    Sensory Features -> LayerNorm -> Linear (direct-current encoder)
        -> Recurrent adaptive-LIF (spikes)
            -> Leaky-integrator readout membrane -> action logits -> Actions
            -> Critic MLP (detached hidden membrane v)            -> Value

References
----------
- Neftci et al. (2019) "Surrogate Gradient Learning in Spiking Neural Networks"
- Bellec et al. (2020) "A solution to the learning dilemma for recurrent
  networks of spiking neurons" (adaptive LIF)
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
from pydantic import model_validator
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Iterator

    from quantumnematode.brain.weights import WeightComponent
    from quantumnematode.initializers._initializer import ParameterInitializer

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._policy import (
    categorical_logprob_entropy_torch,
    ppo_clip_policy_loss,
)
from quantumnematode.brain.arch._registry import register_brain
from quantumnematode.brain.arch._spiking_layers import (
    LeakyIntegratorReadout,
    RecurrentAdaptiveLIFCell,
    RecurrentAdaptiveLIFState,
)
from quantumnematode.brain.arch.dtypes import BrainConfig, BrainType, DeviceType
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


class SpikingPPOBrainConfig(BrainConfig):
    """Configuration for the SpikingPPOBrain architecture.

    Uses modular feature extraction via sensory_modules (required).
    """

    # Sensory modules (required)
    sensory_modules: list[ModuleName] | None = None

    # Recurrent adaptive-LIF core
    hidden_size: int = 64
    num_hidden_layers: int = 1
    timesteps_per_step: int = 1
    v_threshold: float = 1.0
    membrane_decay_init: float = 0.9
    adaptation_decay_init: float = 0.9
    adapt_scale_init: float = 0.1
    readout_decay_init: float = 0.9

    # Actor head: "spike" (leaky-integrator readout, default) | "mlp" (MLP on the hidden membrane)
    actor_head: str = "spike"
    actor_hidden_dim: int = 64
    actor_num_layers: int = 2

    # Surrogate gradient (slope schedulable as a pair)
    surrogate_slope: float = 2.0
    surrogate_slope_end: float | None = None  # anneal target (with anneal_episodes)
    surrogate_slope_anneal_episodes: int | None = None  # episodes to anneal the slope over

    # Critic MLP
    critic_hidden_dim: int = 64
    critic_num_layers: int = 2

    # Learning rates (separate for actor/critic, flat — no schedule)
    actor_lr: float = 0.0003
    critic_lr: float = 0.0003

    # PPO hyperparameters
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_coef_end: float | None = None  # anneal target (with entropy_decay_episodes)
    entropy_decay_episodes: int | None = None  # episodes to anneal entropy_coef over
    rollout_buffer_size: int = 512
    num_epochs: int = 4
    max_grad_norm: float = 0.5

    # BPTT chunk length for truncated backpropagation
    bptt_chunk_length: int = 64

    # Device
    device_type: DeviceType = DeviceType.CPU

    @model_validator(mode="after")
    def _validate_config(self) -> SpikingPPOBrainConfig:  # noqa: C901, PLR0912
        if self.sensory_modules is None or len(self.sensory_modules) == 0:
            msg = "sensory_modules is required and must be non-empty for SpikingPPOBrain"
            raise ValueError(msg)
        if self.hidden_size < 1:
            msg = f"hidden_size must be >= 1, got {self.hidden_size}"
            raise ValueError(msg)
        if self.num_hidden_layers < 1:
            msg = f"num_hidden_layers must be >= 1, got {self.num_hidden_layers}"
            raise ValueError(msg)
        if self.timesteps_per_step < 1:
            msg = f"timesteps_per_step must be >= 1, got {self.timesteps_per_step}"
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
        for field_name in ("membrane_decay_init", "adaptation_decay_init", "readout_decay_init"):
            value = getattr(self, field_name)
            if not 0.0 <= value < 1.0:
                msg = f"{field_name} must be in [0.0, 1.0), got {value}"
                raise ValueError(msg)
        if self.surrogate_slope <= 0.0:
            msg = f"surrogate_slope must be > 0.0, got {self.surrogate_slope}"
            raise ValueError(msg)
        if self.critic_hidden_dim < 1:
            msg = f"critic_hidden_dim must be >= 1, got {self.critic_hidden_dim}"
            raise ValueError(msg)
        if self.critic_num_layers < 1:
            msg = f"critic_num_layers must be >= 1, got {self.critic_num_layers}"
            raise ValueError(msg)
        if self.actor_lr <= 0.0:
            msg = f"actor_lr must be > 0.0, got {self.actor_lr}"
            raise ValueError(msg)
        if self.critic_lr <= 0.0:
            msg = f"critic_lr must be > 0.0, got {self.critic_lr}"
            raise ValueError(msg)
        if self.actor_head not in ("spike", "mlp"):
            msg = f"actor_head must be 'spike' or 'mlp', got {self.actor_head!r}"
            raise ValueError(msg)
        if self.actor_hidden_dim < 1:
            msg = f"actor_hidden_dim must be >= 1, got {self.actor_hidden_dim}"
            raise ValueError(msg)
        if self.actor_num_layers < 1:
            msg = f"actor_num_layers must be >= 1, got {self.actor_num_layers}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_entropy_schedule(self) -> SpikingPPOBrainConfig:
        """Require ``entropy_coef_end`` and ``entropy_decay_episodes`` to be set as a pair.

        Setting only one silently disables annealing (the flat fallback), which is a
        quiet misconfiguration. Reject the half-set case here so ``_get_entropy_coef``
        can trust the invariant: either both are ``None`` (flat) or both are set with
        a positive decay window (anneal).
        """
        end = self.entropy_coef_end
        decay = self.entropy_decay_episodes
        if (end is None) != (decay is None):
            msg = (
                "entropy_coef_end and entropy_decay_episodes must be set together: "
                "provide both to anneal, or neither for a flat schedule "
                f"(got entropy_coef_end={end}, entropy_decay_episodes={decay})"
            )
            raise ValueError(msg)
        if end is not None and decay is not None:
            if end < 0.0:
                msg = f"entropy_coef_end must be >= 0.0, got {end}"
                raise ValueError(msg)
            if decay < 1:
                msg = f"entropy_decay_episodes must be >= 1 when annealing, got {decay}"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_surrogate_schedule(self) -> SpikingPPOBrainConfig:
        """Require ``surrogate_slope_end`` and ``surrogate_slope_anneal_episodes`` as a pair.

        Same fail-fast paired-field contract as the entropy schedule: either both
        are ``None`` (flat slope) or both are set with a positive anneal window
        (slope sharpens from ``surrogate_slope`` to ``surrogate_slope_end``). A
        half-set config would silently run flat, so reject it here.
        """
        end = self.surrogate_slope_end
        anneal = self.surrogate_slope_anneal_episodes
        if (end is None) != (anneal is None):
            msg = (
                "surrogate_slope_end and surrogate_slope_anneal_episodes must be set "
                "together: provide both to anneal the slope, or neither for a flat "
                f"slope (got surrogate_slope_end={end}, "
                f"surrogate_slope_anneal_episodes={anneal})"
            )
            raise ValueError(msg)
        if end is not None and anneal is not None:
            if end <= 0.0:
                msg = f"surrogate_slope_end must be > 0.0, got {end}"
                raise ValueError(msg)
            if anneal < 1:
                msg = f"surrogate_slope_anneal_episodes must be >= 1 when annealing, got {anneal}"
                raise ValueError(msg)
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ──────────────────────────────────────────────────────────────────────────────


class ChunkBatch(TypedDict):
    """Typed payload yielded by ``SpikingPPORolloutBuffer.get_sequential_chunks``.

    ``features`` is a list of per-step feature vectors and ``dones`` a list of
    per-step terminal flags (both indexed by step within the chunk); the
    remaining tensor fields are slices over the chunk. ``state_init`` is the
    detached carried neuron state at the chunk's first step (the BPTT boundary).
    """

    start: int
    end: int
    state_init: NeuronState
    features: list[np.ndarray]
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    dones: list[bool]


class NeuronState(TypedDict):
    """The single per-step carried neuron state stored in the rollout buffer.

    Mirrors the per-layer hidden ``(v, a, s)`` of every recurrent adaptive-LIF
    layer plus the readout membrane ``m``. Stored detached.
    """

    v: list[torch.Tensor]  # per-layer membrane potentials
    a: list[torch.Tensor]  # per-layer adaptation variables
    s: list[torch.Tensor]  # per-layer last spikes
    m: torch.Tensor  # readout membrane


class SpikingPPORolloutBuffer:
    """Rollout buffer that stores the single per-step spiking neuron state.

    Stores per-step (features, action, log_prob, value, reward, done,
    neuron_state) for chunk-based truncated BPTT during PPO updates. The neuron
    state is the single carried state per step (no LSTM cell-state duality),
    exactly as the CfC buffer stores its single hidden state.
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
        self.states: list[NeuronState] = []
        self.position = 0

    def add(  # noqa: PLR0913
        self,
        features: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,  # noqa: FBT001
        state: NeuronState,
    ) -> None:
        """Add a single experience to the buffer (neuron state stored detached)."""
        self.features.append(features)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.states.append(_detach_neuron_state(state))
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
    ) -> Iterator[ChunkBatch]:
        """Generate sequential chunks for truncated BPTT.

        Splits the buffer into chunks of chunk_length steps. Each chunk carries
        its initial neuron state (detached — the BPTT boundary). Chunks are
        yielded in shuffled order for PPO.
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
                "state_init": self.states[start],
                "features": self.features[start:end],
                "actions": actions[start:end],
                "old_log_probs": old_log_probs[start:end],
                "returns": returns[start:end],
                "advantages": adv_normalized[start:end],
                "dones": self.dones[start:end],
            }


def _detach_neuron_state(state: NeuronState) -> NeuronState:
    """Return a detached deep copy of a neuron state (for buffer storage / BPTT boundary)."""
    return {
        "v": [t.detach().clone() for t in state["v"]],
        "a": [t.detach().clone() for t in state["a"]],
        "s": [t.detach().clone() for t in state["s"]],
        "m": state["m"].detach().clone(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Critic Network
# ──────────────────────────────────────────────────────────────────────────────


class _SpikingPPOCritic(nn.Module):
    """MLP critic network for Spiking PPO (reads the detached hidden membrane)."""

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


@register_brain(
    name="spikingppo",
    config_cls=SpikingPPOBrainConfig,
    brain_type=BrainType.SPIKING_PPO,
    families=("spiking",),
)
class SpikingPPOBrain(ClassicalBrain):
    """Recurrent adaptive-LIF spiking PPO brain.

    Processes sensory features through LayerNorm -> direct-current encoder ->
    recurrent adaptive-LIF core -> leaky-integrator readout (action logits) and a
    critic MLP on the detached hidden membrane.
    """

    def __init__(
        self,
        config: SpikingPPOBrainConfig,
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
        self._action_set = list(action_set)
        if len(self._action_set) != num_actions:
            msg = (
                f"SpikingPPOBrain action_set must have exactly {num_actions} "
                f"actions; got {len(self._action_set)}"
            )
            raise ValueError(msg)

        # Seeding
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)
        logger.info(f"SpikingPPOBrain using seed: {self.seed}")

        # Sensory modules (validated as non-empty by config)
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

        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.timesteps_per_step = config.timesteps_per_step

        # LayerNorm on input features.
        self.feature_norm = nn.LayerNorm(self.input_dim).to(self.device)

        # Direct-current encoder: maps features to the hidden input current.
        self.encoder = nn.Linear(self.input_dim, self.hidden_size).to(self.device)

        # Recurrent adaptive-LIF hidden layers (carried state, one tick/step).
        self.hidden_layers = nn.ModuleList(
            [
                RecurrentAdaptiveLIFCell(
                    input_dim=self.hidden_size,
                    num_neurons=self.hidden_size,
                    v_threshold=config.v_threshold,
                    membrane_decay_init=config.membrane_decay_init,
                    adaptation_decay=config.adaptation_decay_init,
                    adapt_scale_init=config.adapt_scale_init,
                )
                for _ in range(self.num_hidden_layers)
            ],
        ).to(self.device)

        # Non-spiking leaky-integrator readout -> action logits.
        self.readout = LeakyIntegratorReadout(
            input_dim=self.hidden_size,
            output_dim=num_actions,
            readout_decay_init=config.readout_decay_init,
        ).to(self.device)

        # Optional MLP actor head reading the hidden membrane v (mirrors CfC's mlp head).
        # When set, logits come from this MLP instead of the leaky-integrator readout.
        self.actor_mlp: nn.Sequential | None = None
        if config.actor_head == "mlp":
            # Same depth convention as the CfC / LSTM actor heads: `actor_num_layers`
            # hidden layers (=> num_layers + 1 Linears), honoured at 1.
            mlp_layers: list[nn.Module] = [
                nn.Linear(self.hidden_size, config.actor_hidden_dim),
                nn.ReLU(),
            ]
            for _ in range(config.actor_num_layers - 1):
                mlp_layers += [
                    nn.Linear(config.actor_hidden_dim, config.actor_hidden_dim),
                    nn.ReLU(),
                ]
            mlp_layers.append(nn.Linear(config.actor_hidden_dim, num_actions))
            self.actor_mlp = nn.Sequential(*mlp_layers).to(self.device)

        # Critic MLP on the detached hidden membrane.
        self.critic = _SpikingPPOCritic(
            input_dim=self.hidden_size,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
        ).to(self.device)

        # ── Optimizers ──

        # Actor optimizer: encoder + recurrent core + readout + input LayerNorm.
        self.actor_optimizer = torch.optim.Adam(self._actor_parameters(), lr=config.actor_lr)
        # Critic optimizer: critic MLP only.
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
        )

        # ── Rollout buffer ──
        self.buffer = SpikingPPORolloutBuffer(
            config.rollout_buffer_size,
            self.device,
            rng=self.rng,
        )

        # ── Neuron state ──
        self.neuron_state = self._zero_state()

        # ── State tracking ──
        self.training = True
        self.current_probabilities: np.ndarray | None = None
        self._step_count = 0
        self._episode_count = 0

        # Pending step data (stored in run_brain, consumed in learn)
        self._pending_features: np.ndarray | None = None
        self._pending_action: int = 0
        self._pending_log_prob: float = 0.0
        self._pending_value: float = 0.0
        self._pending_state: NeuronState = _detach_neuron_state(self.neuron_state)

        # Parameter count logging
        actor_param_count = sum(p.numel() for p in self._actor_parameters())
        critic_param_count = sum(p.numel() for p in self.critic.parameters())
        logger.info(
            f"SpikingPPOBrain initialized: input_dim={self.input_dim}, "
            f"hidden_size={self.hidden_size}, num_hidden_layers={self.num_hidden_layers}, "
            f"timesteps_per_step={self.timesteps_per_step}, "
            f"surrogate_slope={config.surrogate_slope}, "
            f"actor_params={actor_param_count:,}, critic_params={critic_param_count:,}, "
            f"total={actor_param_count + critic_param_count:,}",
        )

    # ──────────────────────────────────────────────────────────────────
    # Feature Extraction
    # ──────────────────────────────────────────────────────────────────

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Extract classical features from sensory modules."""
        assert self.sensory_modules is not None  # noqa: S101
        return extract_classical_features(params, self.sensory_modules)

    # ──────────────────────────────────────────────────────────────────
    # Spiking Core Helpers
    # ──────────────────────────────────────────────────────────────────

    def _zero_state(self) -> NeuronState:
        """Return a zero-initialized neuron state for one (batch=1) agent."""
        v: list[torch.Tensor] = []
        a: list[torch.Tensor] = []
        s: list[torch.Tensor] = []
        for layer in self.hidden_layers:
            assert isinstance(layer, RecurrentAdaptiveLIFCell)  # noqa: S101
            init = layer.init_state(1, self.device)
            v.append(init.v)
            a.append(init.a)
            s.append(init.s)
        m = self.readout.init_state(1, self.device)
        return {"v": v, "a": a, "s": s, "m": m}

    def _core_forward(
        self,
        features_t: torch.Tensor,
        state: NeuronState,
        slope: float,
    ) -> tuple[torch.Tensor, NeuronState]:
        """Run one env-step through the encoder -> recurrent LIF -> readout.

        Parameters
        ----------
        features_t : torch.Tensor
            Raw (un-normalized) feature vector, shape ``(input_dim,)``.
        state : NeuronState
            The carried neuron state from the previous step.
        slope : float
            Surrogate-gradient slope for this step's spikes.

        Returns
        -------
        tuple[torch.Tensor, NeuronState]
            ``(logits, new_state)`` where ``logits`` has shape ``(num_actions,)``.
        """
        normalized = self.feature_norm(features_t)
        # Direct-current input current, held constant across inner ticks.
        input_current = self.encoder(normalized).unsqueeze(0)  # (1, hidden_size)

        new_v = list(state["v"])
        new_a = list(state["a"])
        new_s = list(state["s"])
        membrane = state["m"]

        # ``timesteps_per_step`` inner LIF ticks with a constant input current
        # (>= 1 by config validation); the recurrent spike-feedback advances each
        # inner tick. The readout output IS the carried output membrane, so the
        # final membrane supplies the action logits.
        for _tick in range(self.timesteps_per_step):
            layer_input = input_current
            for layer_idx, layer in enumerate(self.hidden_layers):
                assert isinstance(layer, RecurrentAdaptiveLIFCell)  # noqa: S101
                layer_state = RecurrentAdaptiveLIFState(
                    v=new_v[layer_idx],
                    a=new_a[layer_idx],
                    s=new_s[layer_idx],
                )
                spikes, updated = layer(layer_input, layer_state, slope)
                new_v[layer_idx] = updated.v
                new_a[layer_idx] = updated.a
                new_s[layer_idx] = updated.s
                layer_input = spikes
            # Integrate the top layer's spikes into the readout membrane.
            _, membrane = self.readout(layer_input, membrane)

        if self.actor_mlp is not None:
            logits = self.actor_mlp(new_v[-1]).squeeze(0)
        else:
            logits = membrane.squeeze(0)
        new_state: NeuronState = {"v": new_v, "a": new_a, "s": new_s, "m": membrane}
        return logits, new_state

    def _hidden_membrane(self, state: NeuronState) -> torch.Tensor:
        """Return the top hidden layer's membrane potential, shape ``(hidden_size,)``."""
        return state["v"][-1].squeeze(0)

    def _actor_parameters(self) -> list[nn.Parameter]:
        """Return the actor optimizer's parameter list (for grad-norm clipping).

        Actor = input LayerNorm + direct-current encoder + recurrent core + readout.
        """
        params: list[nn.Parameter] = list(self.feature_norm.parameters())
        params += list(self.encoder.parameters())
        params += list(self.hidden_layers.parameters())
        params += list(self.readout.parameters())
        if self.actor_mlp is not None:
            params += list(self.actor_mlp.parameters())
        return params

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
        """Run the spiking PPO policy to select an action."""
        features = self.preprocess(params)

        features_t = torch.tensor(features, dtype=torch.float32, device=self.device)

        # Store the pre-step neuron state for the buffer.
        state_pre = _detach_neuron_state(self.neuron_state)

        slope = self._get_surrogate_slope()

        # Spiking core forward pass (no gradient during action selection).
        with torch.no_grad():
            logits, new_state = self._core_forward(features_t, self.neuron_state, slope)
            action_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Update the carried neuron state.
        self.neuron_state = new_state

        # Sample action (numpy RNG kept verbatim — trajectory byte-identical).
        action_idx = self.rng.choice(self.num_actions, p=action_probs)
        action_name = self.action_set[action_idx]

        # Log probability via the shared torch helper: the numpy sampler above is
        # unchanged; the log-prob moves onto torch's stabler log_softmax (~1e-7).
        log_prob_t, _entropy_t, _probs_t = categorical_logprob_entropy_torch(
            logits,
            int(action_idx),
        )
        log_prob = float(log_prob_t)

        # Critic value (detached hidden membrane).
        with torch.no_grad():
            value = self.critic(self._hidden_membrane(new_state).detach()).item()

        # Store pending data for learn().
        self._pending_features = features
        self._pending_action = int(action_idx)
        self._pending_log_prob = log_prob
        self._pending_value = value
        self._pending_state = state_pre

        # Update tracking.
        self.current_probabilities = action_probs
        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=float(action_probs[action_idx]),
        )
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(action_probs[action_idx]))

        # Periodic logging.
        self._step_count += 1
        if self._step_count % 50 == 0:
            membrane_norm = torch.norm(self._hidden_membrane(new_state)).item()
            logger.debug(
                f"SpikingPPOBrain step {self._step_count}: "
                f"probs={np.array2string(action_probs, precision=3)}, "
                f"value={value:.4f}, "
                f"v_norm={membrane_norm:.3f}, slope={slope:.2f}",
            )

        return [self.latest_data.action]

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Add experience to the rollout buffer and trigger a PPO update when full."""
        self.history_data.rewards.append(reward)

        if self._pending_features is not None:
            self.buffer.add(
                features=self._pending_features,
                action=self._pending_action,
                log_prob=self._pending_log_prob,
                value=self._pending_value,
                reward=reward,
                done=episode_done,
                state=self._pending_state,
            )

        # PPO update when the buffer is full or the episode ends with enough data.
        if self.buffer.is_full() or (
            episode_done and len(self.buffer) >= self.config.bptt_chunk_length
        ):
            self._perform_ppo_update()
            self.buffer.reset()

    def _get_entropy_coef(self) -> float:
        """Return the current entropy coefficient (linearly annealed if configured, else flat).

        With ``entropy_coef_end`` and ``entropy_decay_episodes`` set, the coefficient
        decays linearly from ``entropy_coef`` to ``entropy_coef_end`` over the first
        ``entropy_decay_episodes`` episodes, then holds at ``entropy_coef_end``. High
        early entropy breaks the dead-exploration basin; lower late entropy lets the
        converged policy peak.
        """
        end = self.config.entropy_coef_end
        decay = self.config.entropy_decay_episodes
        if end is None or decay is None:
            return self.config.entropy_coef  # flat schedule (validated: both None together)
        frac = min(1.0, self._episode_count / decay)
        return self.config.entropy_coef + frac * (end - self.config.entropy_coef)

    def _get_surrogate_slope(self) -> float:
        """Return the current surrogate-gradient slope (annealed if configured, else flat).

        With ``surrogate_slope_end`` and ``surrogate_slope_anneal_episodes`` set, the
        slope moves linearly from ``surrogate_slope`` to ``surrogate_slope_end`` over
        the first ``surrogate_slope_anneal_episodes`` episodes, then holds. A shallow
        early slope trains better while exploring; a sharper late slope tightens the
        spike approximation while fine-tuning.
        """
        end = self.config.surrogate_slope_end
        anneal = self.config.surrogate_slope_anneal_episodes
        if end is None or anneal is None:
            return self.config.surrogate_slope  # flat (validated: both None together)
        frac = min(1.0, self._episode_count / anneal)
        return self.config.surrogate_slope + frac * (end - self.config.surrogate_slope)

    def _perform_ppo_update(self) -> None:  # noqa: PLR0915
        """Perform a PPO update with chunk-based truncated BPTT."""
        if len(self.buffer) == 0:
            return

        ppo_start = time.monotonic()

        slope = self._get_surrogate_slope()

        # Compute the bootstrap value for GAE.
        if self._pending_features is not None:
            with torch.no_grad():
                features_t = torch.tensor(
                    self._pending_features,
                    dtype=torch.float32,
                    device=self.device,
                )
                _, boot_state = self._core_forward(features_t, self.neuron_state, slope)
                last_value = self.critic(self._hidden_membrane(boot_state).detach()).item()
        else:
            last_value = 0.0

        # GAE computation.
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
                # Re-run the spiking core within this chunk. The chunk-start
                # neuron state is detached (the truncated-BPTT boundary).
                state = _detach_neuron_state(chunk["state_init"])

                log_probs_list: list[torch.Tensor] = []
                entropies_list: list[torch.Tensor] = []
                values_list: list[torch.Tensor] = []

                for step_idx in range(chunk["end"] - chunk["start"]):
                    # Reset neuron state at episode boundaries.
                    if step_idx > 0 and chunk["dones"][step_idx - 1]:
                        state = self._zero_state()

                    features = chunk["features"][step_idx]
                    features_t = torch.tensor(
                        features,
                        dtype=torch.float32,
                        device=self.device,
                    )

                    # Spiking core forward (differentiable, surrogate slope).
                    logits, state = self._core_forward(features_t, state, slope)

                    # Shared torch log-prob/entropy for the stored action
                    # (differentiable, used inside the BPTT loop).
                    action_idx = int(chunk["actions"][step_idx].item())
                    log_prob, entropy, _probs = categorical_logprob_entropy_torch(
                        logits,
                        action_idx,
                    )
                    log_probs_list.append(log_prob)
                    entropies_list.append(entropy)

                    # Critic (detached hidden membrane).
                    value = self.critic(self._hidden_membrane(state).detach())
                    values_list.append(value)

                if not log_probs_list:
                    continue

                new_log_probs = torch.stack(log_probs_list)
                mean_entropy = torch.stack(entropies_list).mean()
                values = torch.stack(values_list)

                # PPO policy loss via the shared clipped surrogate (byte-identical
                # to the prior inline surr1/surr2/min). ``ratio`` kept for clip-frac.
                ratio = torch.exp(new_log_probs - chunk["old_log_probs"])
                policy_loss = ppo_clip_policy_loss(
                    new_log_probs,
                    chunk["old_log_probs"],
                    chunk["advantages"],
                    self.config.clip_epsilon,
                )

                with torch.no_grad():
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.config.clip_epsilon).float().mean().item()
                    )
                    total_clip_fraction += clip_frac

                # Value loss (Huber).
                value_loss = torch.nn.functional.smooth_l1_loss(values, chunk["returns"])

                # Actor backward and step.
                actor_loss = policy_loss - entropy_coef * mean_entropy
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._actor_parameters(),
                    self.config.max_grad_norm,
                )
                self.actor_optimizer.step()

                # Critic backward and step.
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

        # Logging.
        if num_updates > 0:
            ppo_elapsed = time.monotonic() - ppo_start
            avg_policy = total_policy_loss / num_updates
            avg_value = total_value_loss / num_updates
            avg_entropy = total_entropy / num_updates
            self.latest_data.loss = avg_policy
            self.history_data.losses.append(avg_policy)

            logger.info(
                f"SpikingPPOBrain PPO update: policy_loss={avg_policy:.4f}, "
                f"value_loss={avg_value:.4f}, entropy={avg_entropy:.4f}, "
                f"entropy_coef={entropy_coef:.4f}, surrogate_slope={slope:.2f}, "
                f"clip_frac={total_clip_fraction / num_updates:.3f}, "
                f"buffer_size={buffer_len}, "
                f"actor_lr={self.config.actor_lr:.6f}, "
                f"ppo_time={ppo_elapsed:.1f}s, "
                f"episode={self._episode_count}",
            )

    # ──────────────────────────────────────────────────────────────────
    # Episode Lifecycle
    # ──────────────────────────────────────────────────────────────────

    def prepare_episode(self) -> None:
        """Reset the spiking neuron state for a new episode."""
        self.neuron_state = self._zero_state()
        self._pending_features = None
        self._step_count = 0

    def post_process_episode(
        self,
        *,
        episode_success: bool | None = None,  # noqa: ARG002
    ) -> None:
        """Post-process after each episode (flat LRs — no schedule update)."""
        self._episode_count += 1

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for SpikingPPOBrain."""

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
        ``"encoder"``
            Direct-current input encoder state_dict.
        ``"feature_norm"``
            Input LayerNorm state_dict.
        ``"hidden_layers"``
            Recurrent adaptive-LIF core state_dict (recurrent weights +
            learnable decay/adaptation params).
        ``"readout"``
            Leaky-integrator readout state_dict.
        ``"actor_mlp"``
            MLP actor-head state_dict (present only when ``actor_head == "mlp"``).
        ``"critic"``
            Critic MLP state_dict.
        ``"actor_optimizer"``
            Actor optimizer state_dict.
        ``"critic_optimizer"``
            Critic optimizer state_dict.
        ``"training_state"``
            Episode count and other training metadata.

        The per-step neuron state is NOT persisted (reset at ``prepare_episode``).
        """
        from quantumnematode.brain.weights import WeightComponent

        all_components: dict[str, WeightComponent] = {
            "encoder": WeightComponent(
                name="encoder",
                state=self.encoder.state_dict(),
            ),
            "feature_norm": WeightComponent(
                name="feature_norm",
                state=self.feature_norm.state_dict(),
            ),
            "hidden_layers": WeightComponent(
                name="hidden_layers",
                state=self.hidden_layers.state_dict(),
            ),
            "readout": WeightComponent(
                name="readout",
                state=self.readout.state_dict(),
            ),
            "critic": WeightComponent(
                name="critic",
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
        # The MLP actor head exists only when actor_head == "mlp".
        if self.actor_mlp is not None:
            all_components["actor_mlp"] = WeightComponent(
                name="actor_mlp",
                state=self.actor_mlp.state_dict(),
            )

        if components is None:
            return all_components

        unknown = components - set(all_components)
        if unknown:
            msg = f"Unknown weight components: {unknown}. Valid components: {set(all_components)}"
            raise ValueError(msg)
        return {k: v for k, v in all_components.items() if k in components}

    def load_weight_components(  # noqa: C901
        self,
        components: dict[str, WeightComponent],
    ) -> None:
        """Load weight components into this brain."""
        # Load networks first.
        if "encoder" in components:
            self.encoder.load_state_dict(components["encoder"].state)
        if "feature_norm" in components:
            self.feature_norm.load_state_dict(components["feature_norm"].state)
        if "hidden_layers" in components:
            self.hidden_layers.load_state_dict(components["hidden_layers"].state)
        if "readout" in components:
            self.readout.load_state_dict(components["readout"].state)
        # Graceful: load the MLP head only when this brain also has one (same-mode
        # round-trip); an absent component or a cross-mode restore is a no-op.
        if "actor_mlp" in components and self.actor_mlp is not None:
            self.actor_mlp.load_state_dict(components["actor_mlp"].state)
        if "critic" in components:
            self.critic.load_state_dict(components["critic"].state)

        # Optimizers after networks.
        if "actor_optimizer" in components:
            self.actor_optimizer.load_state_dict(components["actor_optimizer"].state)
        if "critic_optimizer" in components:
            self.critic_optimizer.load_state_dict(components["critic_optimizer"].state)

        # Training state.
        if "training_state" in components:
            ts = components["training_state"].state
            if "episode_count" in ts:
                self._episode_count = int(ts["episode_count"])

        # Reset the buffer so a fresh rollout under the loaded weights starts
        # pristine; dropping the in-flight transition from the last run_brain
        # call under the OLD weights (the gating predicate in learn()).
        self.buffer.reset()
        self._pending_features = None

        logger.info(
            "SpikingPPOBrain weights loaded (components: %s, episode_count=%d)",
            list(components.keys()),
            self._episode_count,
        )

    # ──────────────────────────────────────────────────────────────────
    # Unsupported / trivial Protocol Methods
    # ──────────────────────────────────────────────────────────────────

    def copy(self) -> SpikingPPOBrain:
        """SpikingPPOBrain does not support copying."""
        error_msg = "SpikingPPOBrain does not support copying. Use deepcopy if needed."
        raise NotImplementedError(error_msg)

    @property
    def action_set(self) -> list[Action]:
        """Get the list of actions."""
        return list(self._action_set)

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        if len(actions) != self.num_actions:
            msg = (
                f"SpikingPPOBrain action_set must have exactly {self.num_actions} "
                f"actions; got {len(actions)}"
            )
            raise ValueError(msg)
        self._action_set = list(actions)

    def build_brain(self) -> None:
        """Not applicable to SpikingPPOBrain."""
        error_msg = "SpikingPPOBrain does not have a quantum circuit."
        raise NotImplementedError(error_msg)

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,
        learning_rate: float = 0.01,
        **kwargs: object,
    ) -> None:
        """Not used — PPO uses its own optimizer."""
