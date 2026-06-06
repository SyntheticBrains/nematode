"""Transformer (self-attention) PPO Brain Architecture.

A classical PPO brain whose policy/value trunk is a Transformer encoder over a
**temporal window** of recent sensory feature vectors. Where LSTM-PPO / CfC-PPO
carry temporal memory in a recurrent hidden state, this brain attends over the
last ``window_size`` per-step feature vectors directly — an attention-based
temporal-memory comparator in the architecture sweep.

Design notes
------------
- **Window-as-state.** Each step stores its own front-zero-padded
  ``(window_size, input_dim)`` window in the rollout buffer, so each buffer
  entry is self-contained: the PPO update re-forwards the stored window. This
  lets the brain reuse the shared shuffled-minibatch ``RolloutBuffer`` (no
  recurrent BPTT chain — the window *is* the temporal context).
- **Shared encoder, separate heads.** A learnable positional embedding is added
  to the projected window; the encoder's last-position output feeds an actor
  head and a critic head.
- **Both action modes** via the shared ``_policy`` helpers: discrete
  (Categorical) or continuous (tanh-squashed Gaussian over a normalized
  ``(speed, turn)``; the environment rescales to physical units).

References
----------
- Vaswani et al. (2017) "Attention Is All You Need"
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import model_validator
from torch import nn, optim

if TYPE_CHECKING:
    from quantumnematode.brain.weights import WeightComponent
    from quantumnematode.initializers._initializer import ParameterInitializer

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._policy import (
    CONTINUOUS_ACTION_DIM,
    categorical_evaluate_torch,
    categorical_sample_torch,
    continuous_evaluate_tanh_gaussian,
    continuous_sample_tanh_gaussian,
    ppo_clip_policy_loss,
)
from quantumnematode.brain.arch._ppo_buffer import RolloutBuffer
from quantumnematode.brain.arch._registry import register_brain
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


class TransformerPPOBrainConfig(BrainConfig):
    """Configuration for the TransformerPPOBrain architecture.

    Uses modular feature extraction via ``sensory_modules`` (required).
    """

    # Sensory modules (required)
    sensory_modules: list[ModuleName] | None = None

    # Temporal window + transformer encoder
    window_size: int = 16
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0  # deterministic policy by default (RL)

    # Heads
    critic_hidden_dim: int = 64

    # PPO hyperparameters
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    num_epochs: int = 4
    num_minibatches: int = 2
    rollout_buffer_size: int = 512
    max_grad_norm: float = 0.5

    # Device
    device_type: DeviceType = DeviceType.CPU

    @model_validator(mode="after")
    def _validate_dims(self) -> TransformerPPOBrainConfig:
        """Fail fast on invalid encoder geometry (clearer than torch's assertion)."""
        if self.d_model % self.nhead != 0:
            msg = (
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead}) "
                "for multi-head attention"
            )
            raise ValueError(msg)
        if self.window_size < 1:
            msg = f"window_size must be >= 1, got {self.window_size}"
            raise ValueError(msg)
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Brain
# ──────────────────────────────────────────────────────────────────────────────


@register_brain(
    name="transformerppo",
    config_cls=TransformerPPOBrainConfig,
    brain_type=BrainType.TRANSFORMER_PPO,
    families=("classical",),
)
class TransformerPPOBrain(ClassicalBrain):
    """PPO brain with a Transformer encoder over a temporal window of features."""

    def __init__(
        self,
        config: TransformerPPOBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] = DEFAULT_ACTIONS,
        *,
        parameter_initializer: ParameterInitializer | None = None,  # noqa: ARG002
    ) -> None:
        super().__init__()

        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)
        logger.info(f"TransformerPPOBrain using seed: {self.seed}")

        if config.sensory_modules is None or len(config.sensory_modules) == 0:
            msg = "sensory_modules is required and must be non-empty for TransformerPPOBrain"
            raise ValueError(msg)
        self.sensory_modules = config.sensory_modules
        self.input_dim = get_classical_feature_dimension(config.sensory_modules)

        self.config = config
        self.device = torch.device(device.to_torch_device_str())
        self.num_actions = num_actions
        # Defensive copy so the shared module-level DEFAULT_ACTIONS (the default
        # arg) can't be mutated across instances (matches CfC/connectome).
        self._action_set = list(action_set)
        self.window_size = config.window_size

        # PPO hyperparameters
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_epsilon = config.clip_epsilon
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.num_epochs = config.num_epochs
        self.num_minibatches = config.num_minibatches
        self.max_grad_norm = config.max_grad_norm

        # Action mode: discrete (categorical) or continuous (tanh-squashed Gaussian
        # over a normalized (speed, turn); the env rescales to physical units).
        self.continuous = config.action_mode == "continuous"
        self._action_low = torch.tensor([0.0, -1.0], device=self.device)
        self._action_high = torch.tensor([1.0, 1.0], device=self.device)
        actor_output_dim = CONTINUOUS_ACTION_DIM if self.continuous else num_actions

        # ── Networks ──
        self.input_proj = nn.Linear(self.input_dim, config.d_model).to(self.device)
        # Learnable positional embedding over the window positions.
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.window_size, config.d_model, device=self.device),
        )
        nn.init.normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_encoder_layers).to(
            self.device,
        )
        self.actor = nn.Linear(config.d_model, actor_output_dim).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(config.d_model, config.critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.critic_hidden_dim, 1),
        ).to(self.device)
        # Small actor-output gain for a stable initial policy (standard PPO trick).
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        if self.continuous:
            self.log_std = nn.Parameter(torch.zeros(CONTINUOUS_ACTION_DIM, device=self.device))

        params = list(self._trainable_parameters())
        self.optimizer = optim.Adam(params, lr=config.learning_rate)

        # Rollout buffer (states are the per-step windows; continuous stores
        # pre-squash sample vectors).
        self.buffer = RolloutBuffer(
            config.rollout_buffer_size,
            self.device,
            rng=self.rng,
            continuous_actions=self.continuous,
        )

        # Temporal window of recent feature vectors (reset per episode).
        self._window: deque[np.ndarray] = deque(maxlen=self.window_size)

        # State tracking
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()
        self.current_probabilities: np.ndarray | None = None
        self.last_value: torch.Tensor | None = None
        self.training = True
        self._episode_count = 0

        logger.info(
            f"TransformerPPOBrain initialized: input_dim={self.input_dim}, "
            f"window_size={self.window_size}, d_model={config.d_model}, "
            f"nhead={config.nhead}, layers={config.num_encoder_layers}, "
            f"action_mode={config.action_mode}",
        )

    def _trainable_parameters(self) -> list[nn.Parameter]:
        """All trainable parameters (encoder trunk + heads + optional log-std)."""
        params = [
            *self.input_proj.parameters(),
            self.pos_embedding,
            *self.encoder.parameters(),
            *self.actor.parameters(),
            *self.critic.parameters(),
        ]
        if self.continuous:
            params.append(self.log_std)
        return params

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Extract the per-step sensory feature vector."""
        return extract_classical_features(params, self.sensory_modules)

    def _windowed_input(self, feature: np.ndarray) -> np.ndarray:
        """Append ``feature`` to the rolling window; return the front-zero-padded window.

        Returns a ``(window_size, input_dim)`` array. Early-episode steps (fewer
        than ``window_size`` features) are zero-padded at the front so the most
        recent reading is always the last position.
        """
        self._window.append(np.asarray(feature, dtype=np.float32))
        window = np.zeros((self.window_size, self.input_dim), dtype=np.float32)
        recent = list(self._window)
        window[self.window_size - len(recent) :] = np.stack(recent)
        return window

    def _trunk(self, windows: torch.Tensor) -> torch.Tensor:
        """Encode windows and return the last-position representation.

        Args:
            windows: ``(batch, window_size, input_dim)`` float tensor.

        Returns
        -------
            ``(batch, d_model)`` — the encoder output at the most recent position.
        """
        projected = self.input_proj(windows) + self.pos_embedding
        encoded = self.encoder(projected)
        return encoded[:, -1, :]

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run the transformer policy over the temporal window and select an action."""
        feature = self.preprocess(params)
        window = self._windowed_input(feature)
        window_t = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            pooled = self._trunk(window_t)
            head_out = self.actor(pooled).squeeze(0)
            value = self.critic(pooled).squeeze(0)
        self.last_value = value

        if self.continuous:
            return self._run_brain_continuous(head_out, window, value)

        action_idx, log_prob, _entropy, probs = categorical_sample_torch(
            head_out,
            device=self.device,
        )
        action_name = self._action_set[action_idx]

        self._pending_state = window
        self._pending_action = action_idx
        self._pending_log_prob = log_prob
        self._pending_value = value

        probs_np = probs.detach().cpu().numpy()
        self.current_probabilities = probs_np
        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=float(probs_np[action_idx]),
        )
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(probs_np[action_idx]))
        return [self.latest_data.action]

    def _run_brain_continuous(
        self,
        mean: torch.Tensor,
        window: np.ndarray,
        value: torch.Tensor,
    ) -> list[ActionData]:
        """Continuous-mode action step: sample a normalized ``(speed, turn)`` action.

        Parameters
        ----------
        mean : torch.Tensor
            The 2-D Gaussian mean from the actor head.
        window : np.ndarray
            The current ``(window_size, input_dim)`` window (stored for the update).
        value : torch.Tensor
            The critic value estimate for the current step.

        Returns
        -------
        list[ActionData]
            A single-element list whose ``continuous`` carries the normalized
            ``(speed, turn)`` action (the environment rescales to physical units).
        """
        action_vec, log_prob, _entropy, pre_tanh = continuous_sample_tanh_gaussian(
            mean,
            self.log_std,
            self._action_low,
            self._action_high,
        )
        self._pending_state = window
        self._pending_action = pre_tanh.detach().cpu().numpy()
        self._pending_log_prob = log_prob
        self._pending_value = value
        self.current_probabilities = None

        action_data = ActionData(
            state="continuous",
            action=None,
            probability=torch.exp(log_prob.detach()).item(),
            continuous=(action_vec[0].item(), action_vec[1].item()),
        )
        self.latest_data.action = action_data
        self.history_data.actions.append(action_data)
        self.history_data.probabilities.append(action_data.probability)
        return [action_data]

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Add experience to the buffer; update when full; reset the window at episode end."""
        if hasattr(self, "_pending_state"):
            self.buffer.add(
                state=self._pending_state,
                action=self._pending_action,
                log_prob=self._pending_log_prob,
                value=self._pending_value,
                reward=reward,
                done=episode_done,
            )

        if self.buffer.is_full() or (episode_done and len(self.buffer) >= self.num_minibatches):
            self._perform_ppo_update()
            self.buffer.reset()

        self.history_data.rewards.append(reward)

        if episode_done:
            self._window.clear()

    def _perform_ppo_update(self) -> None:
        """PPO update with shuffled minibatches over the stored windows."""
        if len(self.buffer) == 0:
            return

        last_value = (
            self.last_value
            if self.last_value is not None
            else torch.tensor([0.0], device=self.device)
        )
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value,
            self.gamma,
            self.gae_lambda,
        )

        for _ in range(self.num_epochs):
            for batch in self.buffer.get_minibatches(self.num_minibatches, returns, advantages):
                pooled = self._trunk(batch["states"])
                head_out = self.actor(pooled)
                values = self.critic(pooled).squeeze(-1)

                if self.continuous:
                    new_log_probs, entropy = continuous_evaluate_tanh_gaussian(
                        head_out,
                        self.log_std,
                        batch["actions"],
                        self._action_low,
                        self._action_high,
                    )
                else:
                    new_log_probs, entropy = categorical_evaluate_torch(
                        head_out,
                        batch["actions"],
                    )

                policy_loss = ppo_clip_policy_loss(
                    new_log_probs,
                    batch["old_log_probs"],
                    batch["advantages"],
                    self.clip_epsilon,
                )
                value_loss = nn.functional.mse_loss(values, batch["returns"])
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._trainable_parameters(), self.max_grad_norm)
                self.optimizer.step()

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for TransformerPPOBrain."""

    def prepare_episode(self) -> None:
        """Reset the temporal window at the start of a new episode."""
        self._window.clear()

    def post_process_episode(self, *, episode_success: bool | None = None) -> None:  # noqa: ARG002
        """Post-process after each episode."""
        self._episode_count += 1

    # ------------------------------------------------------------------
    # Weight persistence (WeightPersistence protocol)
    # ------------------------------------------------------------------

    def get_weight_components(
        self,
        *,
        components: set[str] | None = None,
    ) -> dict[str, WeightComponent]:
        """Return weight components for persistence."""
        from quantumnematode.brain.weights import WeightComponent

        all_components: dict[str, WeightComponent] = {
            "policy": WeightComponent(
                name="policy",
                state={
                    "input_proj": self.input_proj.state_dict(),
                    "pos_embedding": self.pos_embedding.detach().clone(),
                    "encoder": self.encoder.state_dict(),
                    "actor": self.actor.state_dict(),
                },
            ),
            "value": WeightComponent(name="value", state=self.critic.state_dict()),
            "optimizer": WeightComponent(name="optimizer", state=self.optimizer.state_dict()),
            "training_state": WeightComponent(
                name="training_state",
                state={"episode_count": self._episode_count},
            ),
        }
        if self.continuous:
            all_components["log_std"] = WeightComponent(
                name="log_std",
                state={"log_std": self.log_std.data.clone()},
            )

        if components is None:
            return all_components
        unknown = components - set(all_components)
        if unknown:
            msg = f"Unknown weight components: {unknown}. Valid components: {set(all_components)}"
            raise ValueError(msg)
        return {k: v for k, v in all_components.items() if k in components}

    def load_weight_components(self, components: dict[str, WeightComponent]) -> None:
        """Load weight components into this brain."""
        if "policy" in components:
            state = components["policy"].state
            self.input_proj.load_state_dict(state["input_proj"])
            with torch.no_grad():
                self.pos_embedding.copy_(state["pos_embedding"].to(self.pos_embedding.device))
            self.encoder.load_state_dict(state["encoder"])
            self.actor.load_state_dict(state["actor"])
        if "value" in components:
            self.critic.load_state_dict(components["value"].state)
        if "log_std" in components and self.continuous:
            self.log_std.data.copy_(components["log_std"].state["log_std"])
        if "optimizer" in components:
            self.optimizer.load_state_dict(components["optimizer"].state)
        if "training_state" in components:
            ts = components["training_state"].state
            if "episode_count" in ts:
                self._episode_count = int(ts["episode_count"])
        self.buffer.reset()
        self._window.clear()

    def copy(self) -> TransformerPPOBrain:
        """TransformerPPOBrain does not support copying."""
        error_msg = "TransformerPPOBrain does not support copying. Use deepcopy if needed."
        raise NotImplementedError(error_msg)

    @property
    def action_set(self) -> list[Action]:
        """Get the list of actions."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        self._action_set = list(actions)

    def build_brain(self) -> None:
        """Not applicable to TransformerPPOBrain."""
        error_msg = "TransformerPPOBrain does not have a quantum circuit."
        raise NotImplementedError(error_msg)

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,
        learning_rate: float = 0.01,
        **kwargs: object,
    ) -> None:
        """Not used - PPO uses its own optimizer."""
