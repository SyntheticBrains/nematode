"""Feed-forward brain whose weights are evolved by the existing GA optimiser.

Matched-capacity feed-forward network (hidden width 64, 2 hidden layers, no
critic) registered through the L1 plugin registry as ``feedforwardga``.
Consumed by the evolution-framework pipeline (``GeneticAlgorithmOptimizer``
via the ``FeedforwardGAEncoder`` registered in ``ENCODER_REGISTRY``).

The brain is passive in the fitness pipeline: ``learn`` / ``update_memory`` /
``post_process_episode`` are no-op-safe because ``FrozenEvalRunner`` monkey-
patches them to no-ops during evaluation; fitness is computed externally
by ``EpisodicSuccessRate``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pydantic import field_validator
from torch import nn

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._registry import register_brain
from quantumnematode.brain.arch.dtypes import BrainConfig, BrainType, DeviceType
from quantumnematode.brain.modules import (
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

if TYPE_CHECKING:
    import numpy as np

    from quantumnematode.brain.weights import WeightComponent

DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_HIDDEN_LAYERS = 2
_N_ACTIONS = 4


class FeedforwardGABrainConfig(BrainConfig):
    """Configuration for the FeedforwardGABrain architecture.

    Exposes only topology + sensory-module fields. GA optimiser
    hyperparameters live in the YAML ``evolution:`` block per the
    ``configs/evolution/mlpppo_foraging_small.yml`` precedent and are
    consumed by the evolution loop, not by the brain.

    Example config:
        >>> config = FeedforwardGABrainConfig(
        ...     sensory_modules=[ModuleName.FOOD_CHEMOTAXIS],
        ... )
        >>> # input_dim auto-computed from sensory_modules at brain construction time.
    """

    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_hidden_layers: int = DEFAULT_NUM_HIDDEN_LAYERS
    sensory_modules: list[ModuleName]

    @field_validator("sensory_modules")
    @classmethod
    def validate_sensory_modules(cls, v: list[ModuleName]) -> list[ModuleName]:
        """Validate sensory_modules is non-empty."""
        if not v:
            msg = "sensory_modules must be non-empty"
            raise ValueError(msg)
        return v


@register_brain(
    name="feedforwardga",
    config_cls=FeedforwardGABrainConfig,
    brain_type=BrainType.FEEDFORWARDGA,
    families=("classical",),
)
class FeedforwardGABrain(ClassicalBrain):
    """Feed-forward brain whose weights are evolved by the GA optimiser.

    Matched capacity to ``MLPPPOBrain`` small (hidden_dim=64, 2 hidden
    layers, no critic — GA fitness is the episode return directly, so a
    critic auxiliary network would confound the cross-architecture
    comparison).
    """

    def __init__(
        self,
        config: FeedforwardGABrainConfig,
        num_actions: int = _N_ACTIONS,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] = DEFAULT_ACTIONS,
    ) -> None:
        super().__init__()

        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)
        logger.info(f"FeedforwardGABrain using seed: {self.seed}")

        self.config = config
        self.sensory_modules = config.sensory_modules
        self.input_dim = get_classical_feature_dimension(config.sensory_modules)
        self.num_actions = num_actions
        self.device = torch.device(device.to_torch_device_str())
        self._action_set = action_set
        if len(self._action_set) != num_actions:
            msg = (
                f"FeedforwardGABrain action_set must have exactly {num_actions} "
                f"actions; got {len(self._action_set)}"
            )
            raise ValueError(msg)

        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        self.policy = self._build_network(
            config.hidden_dim,
            config.num_hidden_layers,
            output_dim=num_actions,
        ).to(self.device)

        logger.info(
            f"FeedforwardGABrain initialized with sensory_modules="
            f"{[m.value for m in config.sensory_modules]} "
            f"(input_dim={self.input_dim}, hidden_dim={config.hidden_dim}, "
            f"num_hidden_layers={config.num_hidden_layers}, num_actions={num_actions})",
        )

        # Encoder contract shims: _ClassicalPPOEncoder.decode() assigns
        # _episode_count = 0 and calls _update_learning_rate() after
        # restoring weights from a Genome. Both are PPO-specific; the GA
        # brain provides no-op shims so the encoder base class works
        # without modification.
        self._episode_count: int = 0

    def _build_network(
        self,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
    ) -> nn.Sequential:
        """Build a feed-forward network: input_dim -> hidden_dim -> ... -> output_dim."""
        layers: list[nn.Module] = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def _update_learning_rate(self) -> None:
        """No-op shim for the _ClassicalPPOEncoder.decode() contract.

        The encoder base class calls this after assigning _episode_count
        on a freshly-decoded brain to bring a PPO LR scheduler back into
        sync. GA has no LR scheduler, so this is a no-op.
        """

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Extract classical features for the network."""
        return extract_classical_features(params, self.sensory_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing action logits."""
        return self.policy(x)

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run forward pass and sample an action via softmax + categorical."""
        x = self.preprocess(params)
        state = torch.tensor(x, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.forward(state)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_idx = int(dist.sample().item())

        probs_np = probs.detach().cpu().numpy()
        action_name = self._action_set[action_idx]

        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=float(probs_np[action_idx]),
        )
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(probs_np[action_idx]))

        return [
            ActionData(
                state=action_name,
                action=action_name,
                probability=float(probs_np[action_idx]),
            ),
        ]

    def learn(
        self,
        params: BrainParams,
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """No-op: weights are evolved by the GA optimiser across generations.

        FrozenEvalRunner monkey-patches this to a no-op during GA
        evaluation; the brain's own implementation here is the belt-and-
        braces guard that ensures no accidental gradient flow if the
        brain is constructed outside the evolution framework.
        """

    def update_memory(self, reward: float | None = None) -> None:
        """No-op for GA brain (see learn for rationale)."""

    def prepare_episode(self) -> None:
        """No-op: GA brain has no per-episode state to reset."""

    def post_process_episode(
        self,
        *,
        episode_success: bool | None = None,
    ) -> None:
        """No-op for GA brain (no LR scheduler, no episode-level mutation)."""

    def copy(self) -> FeedforwardGABrain:
        """FeedforwardGABrain does not support copying."""
        msg = "FeedforwardGABrain does not support copying. Use deepcopy if needed."
        raise NotImplementedError(msg)

    @property
    def action_set(self) -> list[Action]:
        """Get the list of actions."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        """Set the list of actions."""
        self._action_set = actions

    # ------------------------------------------------------------------
    # Weight persistence (WeightPersistence protocol)
    # ------------------------------------------------------------------

    def get_weight_components(
        self,
        *,
        components: set[str] | None = None,
    ) -> dict[str, WeightComponent]:
        """Return the policy network's weight components for evolution.

        Components
        ----------
        ``"policy"``
            Feed-forward network ``state_dict``. This is the only weight
            component the GA optimiser evolves.
        """
        from quantumnematode.brain.weights import WeightComponent

        all_components: dict[str, WeightComponent] = {
            "policy": WeightComponent(
                name="policy",
                state=self.policy.state_dict(),
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
        """Load weight components into the brain's policy network."""
        if "policy" in components:
            self.policy.load_state_dict(components["policy"].state)
        logger.debug(
            "FeedforwardGABrain weights loaded (components: %s)",
            list(components.keys()),
        )
