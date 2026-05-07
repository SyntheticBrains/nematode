"""Learnable MLP-PPO predator brain.

Implements the `PredatorBrain` Protocol from M1 with a small actor + value
network operating on a fixed-dimensional observation derived from
`PredatorBrainParams`. Designed for M5 co-evolution: the network's weights
are evolved by `CMAESOptimizer(diagonal=True)` via the
`MLPPPOPredatorEncoder`'s `WeightPersistence` round-trip; the brain itself
runs frozen-weight at evaluation time (no inner-loop PPO training inside
`run_brain` — fitness is `PredatorEpisodicKillRate` per design.md D13).

The architecture mirrors the agent-side MLPPPO defaults
(`DEFAULT_ACTOR_HIDDEN_DIM`, `DEFAULT_CRITIC_HIDDEN_DIM`,
`DEFAULT_NUM_HIDDEN_LAYERS` from `quantumnematode.brain.arch.mlpppo`) plus a
value head, but **does not** import `MLPPPOBrain` directly — the agent brain
is coupled to `BrainParams`/`BrainData`/sensory modules and would force a
heavyweight inheritance chain on the predator. This module composes raw
`torch.nn` instead.

Input encoding (per design.md D8 / spec "MLPPPO Predator I/O Encoding
Contract"): an 11-float vector built from `PredatorBrainParams`:

- `predator_position[0] / grid_size`, `predator_position[1] / grid_size`
- For each of `agent_positions[:k_nearest=2]`:
  `(x / grid_size, y / grid_size, present_flag)` — present_flag is 1 when
  the slot is filled, 0 when the slot is padding (fewer alive prey than
  k_nearest).
- `detection_radius / grid_size`, `damage_radius / grid_size`.
- `step_index / max_steps`.

Output: 5-way categorical over `PredatorAction` in fixed index order
`0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT`. The brain returns the argmax
action by default (deterministic given fixed weights + fixed input);
sampling can be enabled via the `sample` constructor flag for cases where
exploration noise is desired during pretraining.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from quantumnematode.env.predator_brain import PredatorAction

if TYPE_CHECKING:
    from quantumnematode.brain.weights import WeightComponent
    from quantumnematode.env.predator_brain import PredatorBrainParams

# Architectural defaults are mirrored from `quantumnematode.brain.arch.mlpppo`
# (`DEFAULT_ACTOR_HIDDEN_DIM=64`, `DEFAULT_CRITIC_HIDDEN_DIM=64`,
# `DEFAULT_NUM_HIDDEN_LAYERS=2` at the time this module was written). Pinning
# them as local module-level constants here breaks an `env -> brain.arch ->
# env` circular import that arises because `brain.arch._brain` imports
# `Direction` from `env`. Keep these in sync with
# `brain.arch.mlpppo.DEFAULT_*` if those defaults are ever revised — the
# values are equal by construction, not by reference.
DEFAULT_ACTOR_HIDDEN_DIM = 64
DEFAULT_CRITIC_HIDDEN_DIM = 64
DEFAULT_NUM_HIDDEN_LAYERS = 2


# Fixed input dimension per spec "Input Encoding Components".
INPUT_DIM = 11

# k_nearest agent positions encoded into the input vector. Per design.md D8:
# k_nearest=2 covers multi-prey observation cleanly without exploding input
# dim.
K_NEAREST = 2

# Action index mapping per spec "Output Action Mapping".
_ACTION_BY_INDEX: tuple[PredatorAction, ...] = (
    PredatorAction.STAY,
    PredatorAction.UP,
    PredatorAction.DOWN,
    PredatorAction.LEFT,
    PredatorAction.RIGHT,
)
NUM_ACTIONS = len(_ACTION_BY_INDEX)


class MLPPPOPredatorBrain:
    """Actor + value MLP predator policy implementing `PredatorBrain`.

    Composition (NOT inheritance from `MLPPPOBrain` — agent-side brain is
    coupled to `BrainParams` etc. and would force unwanted machinery onto
    the predator). Two `nn.Sequential` networks share the input dim:

    - `actor`: 11 → hidden → ... → 5 (action logits).
    - `critic`: 11 → hidden → ... → 1 (value estimate).

    Hidden dims + layer count default to the agent-MLPPPO constants
    (`DEFAULT_ACTOR_HIDDEN_DIM=64`, `DEFAULT_CRITIC_HIDDEN_DIM=64`,
    `DEFAULT_NUM_HIDDEN_LAYERS=2`) but can be overridden at construction
    for ablation. Total parameter count at defaults: ~10k (per design.md D2
    estimate).

    Parameters
    ----------
    actor_hidden_dim
        Width of the actor's hidden layers.
    critic_hidden_dim
        Width of the critic's hidden layers.
    num_hidden_layers
        Number of hidden layers per network (>=1).
    seed
        Optional seed for parameter initialisation reproducibility. When
        provided, all RNG sources are seeded before network construction so
        two brains constructed with the same seed have identical weights.
    sample
        When True, `run_brain` samples from the action distribution (used
        during pretraining to inject exploration noise). When False
        (default), `run_brain` returns the argmax action.
    """

    def __init__(
        self,
        *,
        actor_hidden_dim: int = DEFAULT_ACTOR_HIDDEN_DIM,
        critic_hidden_dim: int = DEFAULT_CRITIC_HIDDEN_DIM,
        num_hidden_layers: int = DEFAULT_NUM_HIDDEN_LAYERS,
        seed: int | None = None,
        sample: bool = False,
    ) -> None:
        if num_hidden_layers < 1:
            msg = f"num_hidden_layers must be >= 1, got {num_hidden_layers}"
            raise ValueError(msg)

        self._actor_hidden_dim = actor_hidden_dim
        self._critic_hidden_dim = critic_hidden_dim
        self._num_hidden_layers = num_hidden_layers
        self._sample = sample

        if seed is not None:
            # Seed torch's global generator so `nn.init.orthogonal_` (called
            # in `_initialize_parameters` below) produces reproducible weights.
            # Modern `np.random.Generator` instances created at call sites
            # (e.g. inside `pretrain_against_heuristic`) are seeded
            # independently via `np.random.default_rng(...)`.
            torch.manual_seed(seed)

        self.actor = self._build_network(actor_hidden_dim, num_hidden_layers, NUM_ACTIONS)
        self.critic = self._build_network(critic_hidden_dim, num_hidden_layers, output_dim=1)
        self._initialize_parameters()

    @staticmethod
    def _build_network(hidden_dim: int, num_hidden_layers: int, output_dim: int) -> nn.Sequential:
        """Build an MLP `INPUT_DIM → hidden → ... → output_dim`.

        Mirrors `MLPPPOBrain._build_network` shape: `Linear(in, hidden) +
        ReLU` followed by `(num_hidden_layers - 1) * (Linear(hidden, hidden)
        + ReLU)`, then `Linear(hidden, output_dim)`.
        """
        layers: list[nn.Module] = [nn.Linear(INPUT_DIM, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def _initialize_parameters(self) -> None:
        """Orthogonal-init linear layers (matches MLPPPOBrain pattern)."""

        def init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=float(np.sqrt(2)))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

    # ------------------------------------------------------------------
    # Input encoding
    # ------------------------------------------------------------------

    def encode_observation(self, params: PredatorBrainParams) -> np.ndarray:
        """Convert `PredatorBrainParams` to the 11-float input vector.

        Pure function of `params`; no torch tensors involved (returns
        numpy float32 for clean handoff to `_forward`). Padding rule for
        fewer than `K_NEAREST` agents alive: zero-fill the missing slots
        with `present_flag=0`.

        max_steps is currently NOT carried on `PredatorBrainParams`; the
        last component normalises `step_index` by a fixed conventional
        upper bound (1000 — matches the M3 lamarckian config's
        `max_steps`). If a future scenario uses a different `max_steps`,
        the normalisation factor can be passed through `extra` config; for
        M5 pilot/full all scenarios use 1000.
        """
        grid_size = float(params.grid_size)
        out = np.zeros(INPUT_DIM, dtype=np.float32)

        out[0] = params.predator_position[0] / grid_size
        out[1] = params.predator_position[1] / grid_size

        # Slots beyond `len(agent_positions)` are zero-filled (present_flag=0)
        # because `out` was initialised with zeros above.
        for slot in range(K_NEAREST):
            if slot >= len(params.agent_positions):
                break
            base = 2 + slot * 3
            ax, ay = params.agent_positions[slot]
            out[base] = ax / grid_size
            out[base + 1] = ay / grid_size
            out[base + 2] = 1.0

        out[8] = params.detection_radius / grid_size
        out[9] = params.damage_radius / grid_size
        # max_steps default (1000) — see docstring note. Future predator
        # configs may pass a different value via `extra` config.
        out[10] = params.step_index / 1000.0

        return out

    # ------------------------------------------------------------------
    # PredatorBrain Protocol
    # ------------------------------------------------------------------

    def run_brain(self, params: PredatorBrainParams) -> PredatorAction:
        """Decide one cardinal action given the predator's current params.

        Encodes `params` into the 11-float input, forwards through the
        actor, then returns either the argmax action (default) or a
        sampled action (when `sample=True`). The critic is forwarded too
        but its output is unused at inference time — kept attached for
        weight-persistence symmetry with M3 LearnedPerformanceFitness's
        actor + critic round-trip.
        """
        obs = self.encode_observation(params)
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # (1, INPUT_DIM)
            logits = self.actor(obs_tensor).squeeze(0)  # (NUM_ACTIONS,)
            if self._sample:
                # Sample using the env's RNG for determinism. We draw a
                # uniform via numpy (env-RNG-shared invariant from M1) and
                # invert the categorical via cumulative softmax.
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                cumulative = np.cumsum(probs)
                u = float(params.rng.random())
                idx = int(np.searchsorted(cumulative, u))
                idx = min(idx, NUM_ACTIONS - 1)  # guard against floating-point edge
            else:
                idx = int(torch.argmax(logits).item())
        return _ACTION_BY_INDEX[idx]

    def prepare_episode(self) -> None:
        """No-op (frozen-weight evaluation has no per-episode state)."""
        return

    def post_process_episode(
        self,
        *,
        episode_success: bool | None = None,
    ) -> None:
        """No-op (frozen-weight evaluation has no per-episode state)."""
        del episode_success

    def copy(self) -> MLPPPOPredatorBrain:
        """Return an independent copy with the same weights.

        Construction order: build a fresh brain (no `seed` passed — the
        clone's orthogonal-init values are about to be overwritten anyway),
        then `state_dict` copy from `self` for both networks.

        Note on torch global RNG: this method does NOT seed the clone's
        torch generator. If a downstream caller invokes `run_brain` with
        `sample=True`, the clone consumes the env-supplied `params.rng`
        for the action draw (per `run_brain` implementation), so torch's
        global state is irrelevant to action determinism. If a future
        method ever consumes torch's global RNG directly, this docstring
        will need revision.
        """
        clone = MLPPPOPredatorBrain(
            actor_hidden_dim=self._actor_hidden_dim,
            critic_hidden_dim=self._critic_hidden_dim,
            num_hidden_layers=self._num_hidden_layers,
            sample=self._sample,
        )
        clone.actor.load_state_dict(self.actor.state_dict())
        clone.critic.load_state_dict(self.critic.state_dict())
        return clone

    # ------------------------------------------------------------------
    # WeightPersistence Protocol
    # ------------------------------------------------------------------

    def get_weight_components(
        self,
        *,
        components: set[str] | None = None,
    ) -> dict[str, WeightComponent]:
        """Return weight components for genome encoder round-trip.

        Components
        ----------
        ``"policy"``
            Actor network state_dict.
        ``"value"``
            Critic network state_dict.

        Note: no `"optimizer"` or `"training_state"` components — the
        predator brain is frozen-weight at evaluation time and has no
        optimizer (CMA-ES at the outer-loop owns the weight gradient).
        """
        # Deferred import: `quantumnematode.brain.weights` imports lightly
        # but the top-level `quantumnematode.brain` package eagerly loads
        # arch modules that import from `env`, which would create a cycle
        # at module-import time of this file. Importing inside the method
        # defers the resolution to call-time when both packages are fully
        # initialised.
        from quantumnematode.brain.weights import WeightComponent

        all_components: dict[str, WeightComponent] = {
            "policy": WeightComponent(name="policy", state=self.actor.state_dict()),
            "value": WeightComponent(name="value", state=self.critic.state_dict()),
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
        """Load weight components into this brain.

        Subset allowed. Unknown component keys are silently ignored
        (matches MLPPPOBrain.load_weight_components convention — the
        brain may have evolved a wider component set in a later schema
        version).
        """
        if "policy" in components:
            self.actor.load_state_dict(components["policy"].state)
        if "value" in components:
            self.critic.load_state_dict(components["value"].state)
