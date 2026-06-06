r"""CfC (Closed-form Continuous-time) Liquid Brain Architecture.

A classical continuous-time recurrent PPO brain whose recurrent core is a
Closed-form Continuous-time (CfC) network wired with a Neural Circuit Policy
``AutoNCP`` wiring — a sparse sensory -> interneuron -> command -> motor graph.
The structured wiring is the scientific point of this architecture: it mirrors
the layered organisation of a small biological nervous system rather than a
dense recurrent layer.

Key Features:
- **CfC core with AutoNCP wiring**: ``CfC(input_dim, AutoNCP(units, num_actions))``
  in both head modes. Single hidden state (no LSTM cell state).
- **Configurable actor head**:
    - ``"motor"`` (default): the ``num_actions`` AutoNCP motor neurons are the
      action logits directly, scaled by a learnable temperature so the bounded
      motor activations can still express a decisive policy. No actor MLP.
    - ``"mlp"``: a small actor MLP maps the recurrent hidden state to the action
      logits, ignoring the motor-neuron output.
- **Critic MLP on the detached hidden state**: prevents value-loss gradients
  from distorting the recurrent representation (both head modes).
- **Separate actor/critic optimizers**: actor optimizer trains the CfC core +
  input LayerNorm + (logit_scale | actor MLP); critic optimizer trains the
  critic MLP only.
- **Chunk-based truncated BPTT**: memory-efficient recurrent training with
  sequential chunks, detaching the chunk-start hidden state at the BPTT
  boundary.

Architecture::

    Sensory Features -> LayerNorm -> CfC(AutoNCP) ->  motor logits * logit_scale  -> Actions
                                                  \-> Actor MLP (hidden state)     -> Actions
                                                  \-> Critic MLP (detached hidden) -> Value

References
----------
- Hasani et al. (2022) "Closed-form Continuous-time Neural Networks"
- Lechner et al. (2020) "Neural Circuit Policies Enabling Auditable Autonomy"
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import torch
from ncps.torch import CfC
from ncps.wirings import AutoNCP
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
from quantumnematode.brain.arch.dtypes import BrainConfig, BrainType, DeviceType
from quantumnematode.brain.modules import (
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

# Minimum gap between ``units`` and ``num_actions`` required by AutoNCP: it
# allocates command + inter neurons above the motor neurons and raises
# ``ValueError`` when ``units <= num_actions + 2``.
_AUTONCP_MIN_UNITS_MARGIN = 2

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────


class CfCBrainConfig(BrainConfig):
    """Configuration for the CfCPPOBrain architecture.

    Uses modular feature extraction via sensory_modules (required).
    """

    # Sensory modules (required)
    sensory_modules: list[ModuleName] | None = None

    # CfC + AutoNCP recurrent core
    units: int = 32
    ncp_sparsity: float = 0.5
    cfc_mode: str = "default"

    # Actor head selection
    actor_head: Literal["motor", "mlp"] = "motor"

    # Motor-head learnable logit temperature (used only when actor_head == "motor")
    motor_logit_scale_init: float = 1.0

    # Actor MLP (used only when actor_head == "mlp")
    actor_hidden_dim: int = 64
    actor_num_layers: int = 2

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
    def _validate_config(self) -> CfCBrainConfig:  # noqa: C901
        if self.sensory_modules is None or len(self.sensory_modules) == 0:
            msg = "sensory_modules is required and must be non-empty for CfCPPOBrain"
            raise ValueError(msg)
        if self.units < 2:  # noqa: PLR2004
            msg = f"units must be >= 2, got {self.units}"
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
        if not 0.0 <= self.ncp_sparsity < 1.0:
            msg = f"ncp_sparsity must be in [0.0, 1.0), got {self.ncp_sparsity}"
            raise ValueError(msg)
        if self.actor_hidden_dim < 1:
            msg = f"actor_hidden_dim must be >= 1, got {self.actor_hidden_dim}"
            raise ValueError(msg)
        if self.critic_hidden_dim < 1:
            msg = f"critic_hidden_dim must be >= 1, got {self.critic_hidden_dim}"
            raise ValueError(msg)
        if self.actor_num_layers < 1:
            msg = f"actor_num_layers must be >= 1, got {self.actor_num_layers}"
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
        return self

    @model_validator(mode="after")
    def _validate_entropy_schedule(self) -> CfCBrainConfig:
        """Require ``entropy_coef_end`` and ``entropy_decay_episodes`` to be set as a pair.

        Setting only one silently disabled annealing (the flat fallback), which is a
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


# ──────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ──────────────────────────────────────────────────────────────────────────────


class ChunkBatch(TypedDict):
    """Typed payload yielded by ``CfCPPORolloutBuffer.get_sequential_chunks``.

    ``features`` is a list of per-step feature vectors and ``dones`` a list of
    per-step terminal flags (both indexed by step within the chunk); the
    remaining tensor fields are slices over the chunk.
    """

    start: int
    end: int
    h_init: torch.Tensor
    features: list[np.ndarray]
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    dones: list[bool]


class CfCPPORolloutBuffer:
    """Rollout buffer that stores per-step CfC hidden states.

    Stores per-step (features, action, log_prob, value, reward, done, h_state)
    for chunk-based truncated BPTT during PPO updates. Unlike the LSTM buffer
    there is a single hidden state per step (no cell state).
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
    ) -> None:
        """Add a single experience to the buffer."""
        self.features.append(features)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.h_states.append(h_state.detach().clone())
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
        its initial CfC hidden state (detached — the BPTT boundary). Chunks are
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
                "h_init": self.h_states[start],
                "features": self.features[start:end],
                "actions": actions[start:end],
                "old_log_probs": old_log_probs[start:end],
                "returns": returns[start:end],
                "advantages": adv_normalized[start:end],
                "dones": self.dones[start:end],
            }


# ──────────────────────────────────────────────────────────────────────────────
# Critic / Actor Networks
# ──────────────────────────────────────────────────────────────────────────────


class _CfCPPOCritic(nn.Module):
    """MLP critic network for CfC PPO."""

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


def _build_actor_mlp(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    output_dim: int,
) -> nn.Sequential:
    """Build the actor MLP used by the ``"mlp"`` head: input_dim -> ... -> output_dim."""
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# Brain
# ──────────────────────────────────────────────────────────────────────────────


@register_brain(
    name="cfcppo",
    config_cls=CfCBrainConfig,
    brain_type=BrainType.CFC_PPO,
    families=("classical",),
)
class CfCPPOBrain(ClassicalBrain):
    """CfC + AutoNCP continuous-time recurrent PPO brain.

    Processes sensory features through LayerNorm -> CfC(AutoNCP) -> actor/critic.
    The actor head is selectable (motor-direct or MLP); the critic always reads
    the detached recurrent hidden state.
    """

    def __init__(  # noqa: PLR0915
        self,
        config: CfCBrainConfig,
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
                f"CfCPPOBrain action_set must have exactly {num_actions} "
                f"actions; got {len(self._action_set)}"
            )
            raise ValueError(msg)

        # Validate the AutoNCP minimum-units requirement up front with a clear
        # message — AutoNCP itself raises a less specific ValueError.
        if config.units <= num_actions + _AUTONCP_MIN_UNITS_MARGIN:
            msg = (
                f"CfCPPOBrain requires units > num_actions + {_AUTONCP_MIN_UNITS_MARGIN} "
                f"(the AutoNCP minimum-units requirement); got units={config.units}, "
                f"num_actions={num_actions} "
                f"(need units > {num_actions + _AUTONCP_MIN_UNITS_MARGIN})."
            )
            raise ValueError(msg)

        # Seeding
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)
        logger.info(f"CfCPPOBrain using seed: {self.seed}")

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

        self.units = config.units
        self.actor_head = config.actor_head

        # LayerNorm on input features
        self.feature_norm = nn.LayerNorm(self.input_dim).to(self.device)

        # CfC recurrent core with the connectome-structured AutoNCP wiring.
        wiring = AutoNCP(config.units, num_actions, sparsity_level=config.ncp_sparsity)
        self.cfc = CfC(self.input_dim, wiring, mode=config.cfc_mode).to(self.device)

        # Actor head.
        self.logit_scale: nn.Parameter | None = None
        self.actor: nn.Sequential | None = None
        if self.actor_head == "motor":
            # Learnable scalar temperature on the bounded motor output. PPO
            # only ever pushes it in the reward-improving (larger-magnitude)
            # direction; a raw nn.Parameter keeps the gradient path simple and
            # the sign is preserved because growing it sharpens the policy.
            self.logit_scale = nn.Parameter(
                torch.tensor(float(config.motor_logit_scale_init), device=self.device),
            )
        else:
            self.actor = _build_actor_mlp(
                input_dim=config.units,
                hidden_dim=config.actor_hidden_dim,
                num_layers=config.actor_num_layers,
                output_dim=num_actions,
            ).to(self.device)

        # Critic MLP on the detached hidden state.
        self.critic = _CfCPPOCritic(
            input_dim=config.units,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
        ).to(self.device)

        # ── Optimizers ──

        # Actor optimizer: CfC core + input LayerNorm + (logit_scale | actor MLP).
        actor_params: list[nn.Parameter] = list(self.cfc.parameters()) + list(
            self.feature_norm.parameters(),
        )
        if self.logit_scale is not None:
            actor_params.append(self.logit_scale)
        if self.actor is not None:
            actor_params += list(self.actor.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=config.actor_lr)
        # Critic optimizer: critic MLP only.
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
        )

        # ── Rollout buffer ──
        self.buffer = CfCPPORolloutBuffer(
            config.rollout_buffer_size,
            self.device,
            rng=self.rng,
        )

        # ── Hidden state ──
        self.h_t = torch.zeros(1, self.units, device=self.device)

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
        self._pending_h_state: torch.Tensor = self.h_t.squeeze(0).clone()

        # Parameter count logging
        actor_param_count = sum(p.numel() for p in actor_params)
        critic_param_count = sum(p.numel() for p in self.critic.parameters())
        logger.info(
            f"CfCPPOBrain initialized: actor_head={self.actor_head}, "
            f"input_dim={self.input_dim}, units={self.units}, "
            f"ncp_sparsity={config.ncp_sparsity}, cfc_mode={config.cfc_mode}, "
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
    # CfC Helpers
    # ──────────────────────────────────────────────────────────────────

    def _cfc_forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one step through the CfC.

        Parameters
        ----------
        x : torch.Tensor
            Normalized input features, shape (input_dim,).
        h : torch.Tensor
            Hidden state, shape (1, units).

        Returns
        -------
        tuple
            (motor_out, new_h) where motor_out has shape (num_actions,) and
            new_h has shape (1, units).
        """
        x_seq = x.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
        out, h_new = self.cfc(x_seq, h)  # out: (1, 1, num_actions), h_new: (1, units)
        return out.squeeze(0).squeeze(0), h_new

    def _logits_from_hidden(
        self,
        motor_out: torch.Tensor,
        h_new: torch.Tensor,
    ) -> torch.Tensor:
        """Map a CfC step output to action logits per the configured head.

        ``"motor"``: scale the bounded motor output by the learnable temperature.
        ``"mlp"``: run the actor MLP on the (non-detached) hidden state.
        """
        if self.actor_head == "motor":
            assert self.logit_scale is not None  # noqa: S101
            return motor_out * self.logit_scale
        assert self.actor is not None  # noqa: S101
        return self.actor(h_new.squeeze(0))

    def _zero_hidden(self) -> torch.Tensor:
        """Return zero-initialized hidden state, shape (1, units)."""
        return torch.zeros(1, self.units, device=self.device)

    def _actor_parameters(self) -> list[nn.Parameter]:
        """Return the actor optimizer's parameter list (for grad-norm clipping)."""
        params: list[nn.Parameter] = list(self.cfc.parameters()) + list(
            self.feature_norm.parameters(),
        )
        if self.logit_scale is not None:
            params.append(self.logit_scale)
        if self.actor is not None:
            params += list(self.actor.parameters())
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
        """Run the CfC PPO policy to select an action."""
        features = self.preprocess(params)

        features_t = torch.tensor(features, dtype=torch.float32, device=self.device)

        # Store pre-step hidden state for the buffer.
        h_pre = self.h_t.squeeze(0).detach().clone()

        # CfC forward pass (no gradient during action selection).
        with torch.no_grad():
            normalized = self.feature_norm(features_t)
            motor_out, h_new = self._cfc_forward(normalized, self.h_t)
            logits = self._logits_from_hidden(motor_out, h_new)
            action_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Update hidden state.
        self.h_t = h_new

        # Sample action (numpy RNG kept verbatim — trajectory byte-identical).
        action_idx = self.rng.choice(self.num_actions, p=action_probs)
        action_name = self.action_set[action_idx]

        # Log probability via the shared torch helper: the numpy sampler above is
        # unchanged; the log-prob moves off manual log(softmax)+eps onto torch's
        # stabler log_softmax (~1e-7 deviation), consistent with the update path.
        log_prob_t, _entropy_t, _probs_t = categorical_logprob_entropy_torch(
            logits,
            int(action_idx),
        )
        log_prob = float(log_prob_t)

        # Critic value (detached hidden state).
        with torch.no_grad():
            value = self.critic(h_new.squeeze(0).detach()).item()

        # Store pending data for learn().
        self._pending_features = features
        self._pending_action = int(action_idx)
        self._pending_log_prob = log_prob
        self._pending_value = value
        self._pending_h_state = h_pre

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
            logger.debug(
                f"CfCPPOBrain step {self._step_count}: "
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
                h_state=self._pending_h_state,
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

    def _perform_ppo_update(self) -> None:  # noqa: PLR0915
        """Perform a PPO update with chunk-based truncated BPTT."""
        if len(self.buffer) == 0:
            return

        ppo_start = time.monotonic()

        # Compute the bootstrap value for GAE.
        if self._pending_features is not None:
            with torch.no_grad():
                features_t = torch.tensor(
                    self._pending_features,
                    dtype=torch.float32,
                    device=self.device,
                )
                normalized = self.feature_norm(features_t)
                _, h_new = self._cfc_forward(normalized, self.h_t)
                last_value = self.critic(h_new.squeeze(0).detach()).item()
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
                # Re-run the CfC within this chunk. The chunk-start hidden
                # state is detached (the truncated-BPTT boundary).
                h = chunk["h_init"].unsqueeze(0).clone()

                log_probs_list: list[torch.Tensor] = []
                entropies_list: list[torch.Tensor] = []
                values_list: list[torch.Tensor] = []

                for step_idx in range(chunk["end"] - chunk["start"]):
                    # Reset hidden state at episode boundaries.
                    if step_idx > 0 and chunk["dones"][step_idx - 1]:
                        h = self._zero_hidden()

                    features = chunk["features"][step_idx]
                    features_t = torch.tensor(
                        features,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    normalized = self.feature_norm(features_t)

                    # CfC forward (differentiable).
                    motor_out, h = self._cfc_forward(normalized, h)
                    logits = self._logits_from_hidden(motor_out, h)

                    # Shared torch log-prob/entropy for the stored action
                    # (differentiable, used inside the BPTT loop).
                    action_idx = int(chunk["actions"][step_idx].item())
                    log_prob, entropy, _probs = categorical_logprob_entropy_torch(
                        logits,
                        action_idx,
                    )
                    log_probs_list.append(log_prob)
                    entropies_list.append(entropy)

                    # Critic (detached hidden state).
                    value = self.critic(h.squeeze(0).detach())
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
                f"CfCPPOBrain PPO update: policy_loss={avg_policy:.4f}, "
                f"value_loss={avg_value:.4f}, entropy={avg_entropy:.4f}, "
                f"entropy_coef={entropy_coef:.4f}, "
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
        """Reset the CfC hidden state for a new episode."""
        self.h_t = self._zero_hidden()
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
        """No-op for CfCPPOBrain."""

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
        ``"cfc"``
            CfC recurrent-core state_dict.
        ``"feature_norm"``
            Input LayerNorm state_dict.
        ``"actor"``
            Actor head state_dict: the learnable ``logit_scale`` (motor head)
            or the actor MLP (mlp head).
        ``"critic"``
            Critic MLP state_dict.
        ``"actor_optimizer"``
            Actor optimizer state_dict.
        ``"critic_optimizer"``
            Critic optimizer state_dict.
        ``"training_state"``
            Episode count and other training metadata.
        """
        from quantumnematode.brain.weights import WeightComponent

        if self.actor_head == "motor":
            assert self.logit_scale is not None  # noqa: S101
            actor_state: dict[str, torch.Tensor] = {
                "logit_scale": self.logit_scale.detach().clone(),
            }
        else:
            assert self.actor is not None  # noqa: S101
            actor_state = dict(self.actor.state_dict())

        all_components: dict[str, WeightComponent] = {
            "cfc": WeightComponent(
                name="cfc",
                state=self.cfc.state_dict(),
            ),
            "feature_norm": WeightComponent(
                name="feature_norm",
                state=self.feature_norm.state_dict(),
            ),
            "actor": WeightComponent(
                name="actor",
                state=actor_state,
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
        # Load networks first.
        if "cfc" in components:
            self.cfc.load_state_dict(components["cfc"].state)
        if "feature_norm" in components:
            self.feature_norm.load_state_dict(components["feature_norm"].state)
        if "actor" in components:
            actor_state = components["actor"].state
            if self.actor_head == "motor":
                assert self.logit_scale is not None  # noqa: S101
                with torch.no_grad():
                    self.logit_scale.copy_(
                        actor_state["logit_scale"].to(self.logit_scale.device),
                    )
            else:
                assert self.actor is not None  # noqa: S101
                self.actor.load_state_dict(actor_state)
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
            "CfCPPOBrain weights loaded (components: %s, episode_count=%d)",
            list(components.keys()),
            self._episode_count,
        )

    # ──────────────────────────────────────────────────────────────────
    # Unsupported / trivial Protocol Methods
    # ──────────────────────────────────────────────────────────────────

    def copy(self) -> CfCPPOBrain:
        """CfCPPOBrain does not support copying."""
        error_msg = "CfCPPOBrain does not support copying. Use deepcopy if needed."
        raise NotImplementedError(error_msg)

    @property
    def action_set(self) -> list[Action]:
        """Get the list of actions."""
        return list(self._action_set)

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        if len(actions) != self.num_actions:
            msg = (
                f"CfCPPOBrain action_set must have exactly {self.num_actions} "
                f"actions; got {len(actions)}"
            )
            raise ValueError(msg)
        self._action_set = list(actions)

    def build_brain(self) -> None:
        """Not applicable to CfCPPOBrain."""
        error_msg = "CfCPPOBrain does not have a quantum circuit."
        raise NotImplementedError(error_msg)

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,
        learning_rate: float = 0.01,
        **kwargs: object,
    ) -> None:
        """Not used — PPO uses its own optimizer."""
