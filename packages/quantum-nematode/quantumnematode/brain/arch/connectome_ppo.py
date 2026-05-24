"""Connectome-constrained PPO brain architecture.

Uses the Cook 2019 *C. elegans* hermaphrodite connectome as a fixed
topology, with chemical synapses strict-masked (PPO-learnable scalar
weights along wild-type edges only) and gap junctions non-learnable
(fixed Cook 2019 synapse counts, fan-in normalised).

Sensor projection: food-chemotaxis input → ASEL/ASER/AWCL/AWCR/AWAL/AWAR
sensory neurons via additive injection scaled by per-input learnable
gains. Motor readout: VB/DB/VA/DA motor-class activations mean-pooled
into a 4-vector, then projected to the discrete 4-action set via a
learnable 4x4 readout matrix.

Forward pass (single step): repeat K times before motor readout::

    h = tanh((W_chem * M_chem).T @ h + G_gap.T @ h)

K is configurable; default 4 (matches the canonical *C. elegans*
klinotaxis pathway depth: sensory → primary-interneuron →
command-interneuron → motor). K=1 produces a degenerate output because
the food signal cannot propagate from sensory neurons to motor neurons
in a single chemical-synapse hop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn, optim

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._ppo_buffer import RolloutBuffer
from quantumnematode.brain.arch._registry import register_brain
from quantumnematode.brain.arch.dtypes import BrainConfig, BrainType, DeviceType
from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

if TYPE_CHECKING:
    from quantumnematode.connectome.model import Connectome


# ──────────────────────────────────────────────────────────────────────────────
# Defaults (mirror MLPPPOBrainConfig where applicable)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_LEARNING_RATE = 3.0e-4
_DEFAULT_GAMMA = 0.99
_DEFAULT_GAE_LAMBDA = 0.95
_DEFAULT_CLIP_EPSILON = 0.2
_DEFAULT_VALUE_LOSS_COEF = 0.5
_DEFAULT_ENTROPY_COEF = 0.01
_DEFAULT_NUM_EPOCHS = 4
_DEFAULT_NUM_MINIBATCHES = 4
_DEFAULT_ROLLOUT_BUFFER_SIZE = 2048
_DEFAULT_MAX_GRAD_NORM = 0.5

# Sensory neurons that receive food-chemotaxis injection (canonical
# Bargmann-lab klinotaxis pathway: ASE for salt + AWC + AWA for odorant).
_SENSOR_NEURONS_FOOD: tuple[str, ...] = (
    "ASEL",
    "ASER",
    "AWCL",
    "AWCR",
    "AWAL",
    "AWAR",
)

# Motor-neuron class prefixes for the readout (matches the connectome
# neurons table: VB/DB/VA/DA all carry numeric suffixes).
_MOTOR_CLASSES: tuple[str, ...] = ("VB", "DB", "VA", "DA")

# Number of food-chemotaxis input features per sensing mode.
# - oracle: [strength, angle]                      → 2 features
# - klinotaxis: [concentration, lateral, dC/dt]    → 3 features
_N_FOOD_FEATURES_BY_MODE: dict[str, int] = {"oracle": 2, "klinotaxis": 3}

# Number of discrete actions.
_N_ACTIONS = 4


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


class ConnectomePPOBrainConfig(BrainConfig):
    """Configuration for the connectome-constrained PPO brain."""

    connectome_source: Literal["cook_2019_hermaphrodite"] = "cook_2019_hermaphrodite"
    enable_gap_junctions: bool = True
    chemical_mask_mode: Literal["strict", "soft_prior"] = "strict"
    # Sensing mode controls what env-side fields the sensor projection
    # consumes. ``oracle`` reads ``[food_gradient_strength, food_gradient_direction]``
    # directly (2 features, default env mode). ``klinotaxis`` reads
    # ``[food_concentration, food_lateral_gradient, food_dconcentration_dt]``
    # (3 features) — the biologically-faithful head-sweep mode emitted
    # when the env YAML carries ``environment.sensing.chemotaxis_mode: klinotaxis``.
    # The brain's ``food_gains`` matrix is sized to match.
    sensing_mode: Literal["oracle", "klinotaxis"] = "oracle"
    # Forward-pass depth: number of chemical+gap-junction recurrence
    # iterations between sensor injection and motor pooling. The
    # canonical *C. elegans* klinotaxis pathway is roughly 4 hops
    # (sensory → AIY/AIZ → AVA/AVB → motor), so the default is 4. K=1
    # produces a degenerate output because the food signal cannot reach
    # motor neurons in one chemical-synapse hop.
    forward_pass_depth: int = 4
    freeze_updates: bool = False

    # PPO hyperparameters (mirror MLPPPOBrainConfig)
    learning_rate: float = _DEFAULT_LEARNING_RATE
    gamma: float = _DEFAULT_GAMMA
    gae_lambda: float = _DEFAULT_GAE_LAMBDA
    clip_epsilon: float = _DEFAULT_CLIP_EPSILON
    value_loss_coef: float = _DEFAULT_VALUE_LOSS_COEF
    entropy_coef: float = _DEFAULT_ENTROPY_COEF
    num_epochs: int = _DEFAULT_NUM_EPOCHS
    num_minibatches: int = _DEFAULT_NUM_MINIBATCHES
    rollout_buffer_size: int = _DEFAULT_ROLLOUT_BUFFER_SIZE
    max_grad_norm: float = _DEFAULT_MAX_GRAD_NORM


# ──────────────────────────────────────────────────────────────────────────────
# Topology
# ──────────────────────────────────────────────────────────────────────────────


class ConnectomeTopology(nn.Module):
    """Forward-pass network whose connectivity is the Cook 2019 connectome.

    Holds the chemical-synapse weight tensor ``W_chem`` (PPO-learnable),
    the chemical-synapse strict-mask ``M_chem`` (boolean), and the
    gap-junction weight tensor ``G_gap`` (non-learnable, fan-in
    normalised). Sensor gains and motor readout are also held here so a
    single ``forward(food_strength, food_angle)`` call covers the whole
    env-observation → action-logits path.

    ``forward`` returns the 4 action logits aligned with
    ``DEFAULT_ACTIONS = [FORWARD, LEFT, RIGHT, STAY]``.
    """

    # Class-level attribute annotations narrow pyright's view of the
    # registered buffers from ``Tensor | Module`` to ``torch.Tensor``.
    m_chem: torch.Tensor
    g_gap: torch.Tensor
    _food_neuron_indices: torch.Tensor
    _motor_flat_indices: torch.Tensor
    _motor_class_boundaries: torch.Tensor
    w_chem: nn.Parameter
    food_gains: nn.Parameter
    readout: nn.Parameter

    def __init__(  # noqa: C901, PLR0912, PLR0913, PLR0915 — one-time topology construction
        self,
        connectome: Connectome,
        *,
        enable_gap_junctions: bool,
        forward_pass_depth: int,
        n_food_features: int,
        enforce_strict_mask: bool,
        device: torch.device,
        rng: np.random.Generator,
    ) -> None:
        super().__init__()
        if forward_pass_depth < 1:
            msg = f"forward_pass_depth must be >= 1, got {forward_pass_depth}"
            raise ValueError(msg)
        if n_food_features < 1:
            msg = f"n_food_features must be >= 1, got {n_food_features}"
            raise ValueError(msg)
        self.forward_pass_depth = forward_pass_depth
        self.enable_gap_junctions = enable_gap_junctions
        self.n_food_features = n_food_features
        # When True, the forward pass multiplies ``w_chem`` by ``m_chem`` so
        # gradients on non-wild-type edges are pinned to zero (and ``w_chem``
        # data on those positions never moves from its zero init). When
        # False, the forward pass uses raw ``w_chem`` so gradients flow
        # through every entry — letting the optimiser grow new edges and
        # treating the wild-type adjacency as an initial-weight prior only.
        self.enforce_strict_mask = enforce_strict_mask

        # ── Index layout ────────────────────────────────────────────────
        self.neuron_names: list[str] = sorted(connectome.neurons)
        self.n_neurons = len(self.neuron_names)
        self._idx = {name: i for i, name in enumerate(self.neuron_names)}

        # ── Chemical synapses: strict-mask + learnable weights ──────────
        # Initialised at N(0, 1/sqrt(fan_in)) along existing edges; zero
        # elsewhere. The boolean mask is registered as a buffer (moves
        # with .to(device) but is not a learnable parameter).
        chem_in_degree = np.zeros(self.n_neurons, dtype=np.int64)
        for syn in connectome.chemical_synapses:
            chem_in_degree[self._idx[syn.post]] += 1

        safe_in = np.where(chem_in_degree > 0, chem_in_degree, 1)
        chem_scale = np.where(
            chem_in_degree > 0,
            1.0 / np.sqrt(safe_in.astype(np.float64)),
            0.0,
        )

        m_chem_np = np.zeros((self.n_neurons, self.n_neurons), dtype=bool)
        w_chem_np = np.zeros((self.n_neurons, self.n_neurons), dtype=np.float32)
        for syn in connectome.chemical_synapses:
            pre_i = self._idx[syn.pre]
            post_j = self._idx[syn.post]
            m_chem_np[pre_i, post_j] = True
            w_chem_np[pre_i, post_j] = np.float32(
                rng.normal(loc=0.0, scale=float(chem_scale[post_j])),
            )

        self.register_buffer(
            "m_chem",
            torch.from_numpy(m_chem_np).to(device=device),
        )
        self.w_chem = nn.Parameter(torch.from_numpy(w_chem_np).to(device=device))

        # ── Gap junctions: non-learnable, symmetric, fan-in normalised ──
        # Each entry ``G[i, j]`` is divided by ``sqrt(fan_in[i] * fan_in[j])``
        # so per-neuron total gap-junction input stays bounded while
        # preserving the symmetry of the underlying physics (gap junctions
        # are bidirectional). A pure row-or-column scaling would break
        # ``G[a, b] == G[b, a]``; the symmetric scaling is the only
        # normalisation consistent with both spec scenarios.
        if enable_gap_junctions:
            g_gap_np = np.zeros((self.n_neurons, self.n_neurons), dtype=np.float32)
            for gj in connectome.gap_junctions:
                a_i = self._idx[gj.neuron_a]
                b_j = self._idx[gj.neuron_b]
                w = float(gj.weight)
                g_gap_np[a_i, b_j] = w
                g_gap_np[b_j, a_i] = w
            gap_in_degree = (g_gap_np != 0).sum(axis=0)
            safe_gap_in = np.where(gap_in_degree > 0, gap_in_degree, 1)
            gap_scale = np.where(
                gap_in_degree > 0,
                1.0 / np.sqrt(safe_gap_in.astype(np.float64)),
                0.0,
            ).astype(np.float32)
            # Symmetric scaling: divide G[i,j] by sqrt(d_i * d_j).
            g_gap_np = g_gap_np * gap_scale[:, np.newaxis] * gap_scale[np.newaxis, :]
            self.register_buffer(
                "g_gap",
                torch.from_numpy(g_gap_np).to(device=device),
            )
        else:
            self.register_buffer(
                "g_gap",
                torch.zeros(self.n_neurons, self.n_neurons, device=device),
            )

        # ── Sensor projection: food-chemotaxis → sensory neurons ────────
        # Learnable 2x6 gain matrix: 2 food features by 6 sensory neurons.
        # ``self._food_neuron_indices`` records which 6 neuron indices the
        # gains write to.
        food_indices: list[int] = []
        for name in _SENSOR_NEURONS_FOOD:
            if name not in self._idx:
                msg = (
                    f"Sensor-projection neuron {name!r} not present in connectome "
                    f"(have {self.n_neurons} neurons; this is a connectome-data bug)."
                )
                raise ValueError(msg)
            food_indices.append(self._idx[name])
        self.register_buffer(
            "_food_neuron_indices",
            torch.tensor(food_indices, dtype=torch.long, device=device),
        )
        self.food_gains = nn.Parameter(
            torch.zeros(n_food_features, len(_SENSOR_NEURONS_FOOD), device=device),
        )
        # Initialise gains so the food signal arrives at the sensory
        # neurons with comparable magnitude to other neural input. With
        # 6 target neurons and per-feature values in approximately [-1, 1],
        # unit-variance gains keep the injection in a range where tanh
        # hasn't saturated.
        nn.init.normal_(self.food_gains, mean=0.0, std=1.0)

        # ── Motor readout: VB/DB/VA/DA mean-pool → 4x4 learnable matrix ─
        motor_class_indices: dict[str, list[int]] = {cls: [] for cls in _MOTOR_CLASSES}
        for name in self.neuron_names:
            cls = connectome.neurons[name].cell_class
            if cls != "motor":
                continue
            for prefix in _MOTOR_CLASSES:
                if name.startswith(prefix) and name[len(prefix) :].isdigit():
                    motor_class_indices[prefix].append(self._idx[name])
                    break
        for cls in _MOTOR_CLASSES:
            if not motor_class_indices[cls]:
                msg = (
                    f"Motor class {cls!r} has zero matching neurons in the "
                    "connectome (this is a connectome-data bug)."
                )
                raise ValueError(msg)
        # Store as a flat tensor + class-boundary offsets so the forward
        # pass can mean-pool with scatter_reduce without Python loops.
        flat: list[int] = []
        boundaries: list[int] = [0]
        for cls in _MOTOR_CLASSES:
            flat.extend(motor_class_indices[cls])
            boundaries.append(len(flat))
        self.register_buffer(
            "_motor_flat_indices",
            torch.tensor(flat, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "_motor_class_boundaries",
            torch.tensor(boundaries, dtype=torch.long, device=device),
        )
        self.readout = nn.Parameter(torch.zeros(_N_ACTIONS, _N_ACTIONS, device=device))
        # Stronger orthogonal init so the initial policy isn't a constant
        # uniform across actions. ``gain=1.0`` is the standard PPO-actor
        # initial-gain choice; the small ``0.01`` PPO stable-policy trick
        # produces a degenerate-uniform initial policy here because the
        # whole upstream pipeline is small (one 302-vec → 4-vec readout).
        nn.init.orthogonal_(self.readout, gain=1.0)

        # Apply strict-mask to initial weights too (initialisation already
        # respects the mask above, but defence-in-depth).
        with torch.no_grad():
            self.w_chem.data.copy_(self.apply_weight_mask(self.w_chem.data))

    def apply_weight_mask(self, weights: torch.Tensor) -> torch.Tensor:
        """Project a candidate weight tensor onto the strict-mask manifold.

        Pure function: returns ``weights * M_chem`` so values along
        non-existent chemical-synapse edges are zeroed. The caller owns
        the policy of when to apply this (e.g. only under
        ``chemical_mask_mode="strict"``) and is responsible for writing
        the result back into the topology's parameter storage.

        Conforms to the ``BrainTopology`` Protocol's pure-projector
        contract: stateless, no in-place mutation of ``self``.
        """
        return weights * self.m_chem.to(weights.dtype)

    def _pool_motor(self, h: torch.Tensor) -> torch.Tensor:
        """Mean-pool ``h`` over the four motor classes → 4-vector.

        Vectorised: index ``h`` at the flat motor indices, then take
        per-class mean using boundaries.
        """
        flat_acts = h.index_select(0, self._motor_flat_indices)
        boundaries = self._motor_class_boundaries
        pooled = torch.empty(_N_ACTIONS, device=h.device, dtype=h.dtype)
        for k in range(_N_ACTIONS):
            start = int(boundaries[k].item())
            end = int(boundaries[k + 1].item())
            pooled[k] = flat_acts[start:end].mean()
        return pooled

    def forward_with_hidden(
        self,
        food_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action logits AND the post-K hidden state in one pass.

        Returns a tuple ``(logits, hidden)`` where:
          - ``logits`` is shape ``(4,)`` — the 4 action logits after motor
            pooling + readout.
          - ``hidden`` is shape ``(302,)`` — the 302-dim activation vector
            after K recurrent updates, BEFORE motor pooling. Callers that
            need a value-head input over the full activation vector
            consume this directly without re-running the topology.

        ``food_features`` is shape ``(self.n_food_features,)`` carrying
        ``[strength, angle]`` in oracle mode or
        ``[concentration, lateral, dC/dt]`` in klinotaxis mode. Sensor
        injection produces the initial 302-vec hidden state; K recurrent
        updates iterate the chemical + gap-junction sum through ``tanh``;
        motor pooling + readout produce the action logits.
        """
        if food_features.shape != (self.n_food_features,):
            msg = (
                f"food_features must have shape ({self.n_food_features},); "
                f"got {tuple(food_features.shape)}"
            )
            raise ValueError(msg)

        # Sensor injection: 2-vec @ (2, 6) → 6-vec onto food sensory neurons.
        h = torch.zeros(self.n_neurons, device=food_features.device, dtype=food_features.dtype)
        food_injection = food_features @ self.food_gains  # shape (6,)
        h = h.index_add(0, self._food_neuron_indices, food_injection)

        # K recurrent updates through chemical + gap-junction connectivity.
        # Each neuron's pre-activation = sum_pre(W[pre, post] * h[pre]) = (W.T @ h)[post].
        # Under ``enforce_strict_mask`` the forward uses ``w_chem * m_chem`` so
        # gradients on ~m_chem are zero; under soft-prior it uses raw
        # ``w_chem`` so the optimiser can grow new edges from zero init.
        chem_mat = self.w_chem * self.m_chem if self.enforce_strict_mask else self.w_chem
        gap_mat = self.g_gap if self.enable_gap_junctions else torch.zeros_like(self.g_gap)
        for _ in range(self.forward_pass_depth):
            preact = chem_mat.T @ h + gap_mat.T @ h
            h = torch.tanh(preact)

        # Motor pooling + readout.
        motor_acts = self._pool_motor(h)
        logits = self.readout @ motor_acts  # shape (4,)
        return logits, h

    def forward(self, food_features: torch.Tensor) -> torch.Tensor:
        """Compute 4 action logits from a 2-vector of food-chemotaxis features.

        Thin shim over :meth:`forward_with_hidden` for callers that only
        need the logits. Callers that also need the post-K hidden state
        (e.g. for a value head) should call ``forward_with_hidden`` directly
        to avoid recomputing the connectome forward.
        """
        logits, _ = self.forward_with_hidden(food_features)
        return logits

    @property
    def learnable_parameters(self) -> list[nn.Parameter]:
        """Return the PPO-learnable parameter tensors only."""
        return [self.w_chem, self.food_gains, self.readout]


# ──────────────────────────────────────────────────────────────────────────────
# Brain
# ──────────────────────────────────────────────────────────────────────────────


@register_brain(
    name="connectomeppo",
    config_cls=ConnectomePPOBrainConfig,
    brain_type=BrainType.CONNECTOMEPPO,
    families=("classical",),
)
class ConnectomePPOBrain(ClassicalBrain):
    """PPO brain trained over the Cook 2019 *C. elegans* connectome topology."""

    def __init__(
        self,
        config: ConnectomePPOBrainConfig,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        super().__init__()

        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)
        logger.info(f"ConnectomePPOBrain using seed: {self.seed}")

        self.config = config
        self.device = torch.device(device.to_torch_device_str())
        self._action_set = action_set if action_set is not None else list(DEFAULT_ACTIONS)
        if len(self._action_set) != _N_ACTIONS:
            msg = (
                f"ConnectomePPOBrain action_set must have exactly {_N_ACTIONS} "
                f"actions; got {len(self._action_set)}"
            )
            raise ValueError(msg)

        # Load the connectome (currently only Cook 2019 is supported).
        if config.connectome_source != "cook_2019_hermaphrodite":
            msg = f"Unsupported connectome_source: {config.connectome_source!r}"
            raise ValueError(msg)
        connectome = load_cook_2019_hermaphrodite()

        # Derive the per-mode food-feature count.
        self._n_food_features = _N_FOOD_FEATURES_BY_MODE[config.sensing_mode]

        # Topology owns weight tensors + forward pass. ``enforce_strict_mask``
        # is the structural choice that pins gradient flow to wild-type edges
        # under "strict" mode; under "soft_prior" the forward uses raw
        # ``w_chem`` so the optimiser can grow new edges.
        self.topology = ConnectomeTopology(
            connectome,
            enable_gap_junctions=config.enable_gap_junctions,
            forward_pass_depth=config.forward_pass_depth,
            n_food_features=self._n_food_features,
            enforce_strict_mask=(config.chemical_mask_mode == "strict"),
            device=self.device,
            rng=self.rng,
        ).to(self.device)

        # Critic: scalar value head over the same 302-dim activation vector.
        self.critic = nn.Linear(self.topology.n_neurons, 1).to(self.device)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

        # Single Adam optimiser over the topology's learnable params + critic.
        learnable = self.topology.learnable_parameters + list(self.critic.parameters())
        self.optimizer = optim.Adam(learnable, lr=config.learning_rate)

        # PPO hyperparameters (cached for the learn loop).
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_epsilon = config.clip_epsilon
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.num_epochs = config.num_epochs
        self.num_minibatches = config.num_minibatches
        self.max_grad_norm = config.max_grad_norm

        # Rollout buffer.
        self.buffer = RolloutBuffer(config.rollout_buffer_size, self.device, rng=self.rng)

        # Brain Protocol state.
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()
        self.current_probabilities: np.ndarray | None = None
        self.last_value: torch.Tensor | None = None

        # Per-step pending data (added to buffer when next reward arrives).
        self._pending_state: np.ndarray | None = None
        self._pending_action: int | None = None
        self._pending_log_prob: torch.Tensor | None = None
        self._pending_value: torch.Tensor | None = None

    # ── Brain Protocol surface ──────────────────────────────────────────

    @property
    def action_set(self) -> list[Action]:
        """Return the 4-action discrete action set."""
        return self._action_set

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Extract the food-chemotaxis feature vector for the topology.

        Shape depends on ``config.sensing_mode``:

        - ``oracle`` → ``[strength, angle]`` (2 features). ``strength`` is
          the env's ``food_gradient_strength`` field (∈ [0, 1]); ``angle``
          is ``food_gradient_direction`` (radians ∈ [-π, π]) normalised to
          [-1, 1] by dividing by π.
        - ``klinotaxis`` → ``[concentration, lateral, dC/dt]`` (3 features).
          ``concentration`` is ``food_concentration`` (env-side scalar);
          ``lateral`` is ``tanh(food_lateral_gradient * lateral_scale)``;
          ``dC/dt`` is ``tanh(food_dconcentration_dt * derivative_scale)``.
          These mirror the canonical klinotaxis sensory-module emission
          shape so the env-side klinotaxis pipeline drives the brain
          without re-implementing the head-sweep computation here.
        """
        if self.config.sensing_mode == "oracle":
            strength = float(params.food_gradient_strength or 0.0)
            angle = float(params.food_gradient_direction or 0.0) / np.pi
            return np.array([strength, angle], dtype=np.float32)

        # klinotaxis mode
        strength = float(params.food_concentration or 0.0)
        raw_lateral = float(params.food_lateral_gradient or 0.0)
        lateral = float(np.tanh(raw_lateral * params.lateral_scale))
        raw_deriv = float(params.food_dconcentration_dt or 0.0)
        dcdt = float(np.tanh(raw_deriv * params.derivative_scale))
        return np.array([strength, lateral, dcdt], dtype=np.float32)

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002 — reward delivered via ``learn``
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Sample an action via the connectome forward pass."""
        state = self.preprocess(params)
        state_t = torch.from_numpy(state).to(self.device)

        # Single forward pass produces both the action logits and the
        # post-K 302-dim hidden state used by the critic.
        logits, hidden = self.topology.forward_with_hidden(state_t)
        value = self.critic(hidden)

        # Action distribution.
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_idx = int(dist.sample().item())
        log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device))

        action_name = self._action_set[action_idx]

        # Store pending data for the buffer (added on the next reward).
        self._pending_state = state
        self._pending_action = action_idx
        self._pending_log_prob = log_prob
        self._pending_value = value
        self.last_value = value

        probs_np = probs.detach().cpu().numpy()
        self.current_probabilities = probs_np

        # Update history.
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
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Add experience to the buffer; trigger a PPO update when full or at episode end."""
        if self._pending_state is not None:
            # When ``_pending_state`` is set, ``run_brain`` populates
            # ``_pending_action``, ``_pending_log_prob``, and
            # ``_pending_value`` in the same call. The runtime guard
            # surfaces any out-of-order use rather than silently
            # substituting zeros.
            if (
                self._pending_action is None
                or self._pending_log_prob is None
                or self._pending_value is None
            ):
                msg = (
                    "ConnectomePPOBrain.learn called with inconsistent pending "
                    "state: _pending_state set but action/log_prob/value missing. "
                    "Did the env call learn() before run_brain()?"
                )
                raise RuntimeError(msg)
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

    def update_memory(self, reward: float | None = None) -> None:
        """No-op (Brain Protocol surface; PPO updates land in ``learn``)."""

    def prepare_episode(self) -> None:
        """No-op."""

    def post_process_episode(self, *, episode_success: bool | None = None) -> None:
        """No-op."""

    def copy(self) -> ConnectomePPOBrain:
        """ConnectomePPOBrain does not support copying."""
        msg = "ConnectomePPOBrain does not support copying. Use deepcopy if needed."
        raise NotImplementedError(msg)

    # ── PPO update ──────────────────────────────────────────────────────

    def _perform_ppo_update(self) -> None:
        """Run one PPO update over the rollout buffer.

        Skipped entirely when ``config.freeze_updates`` is True — that is
        the paired-control branch of the training-signal evaluation. Under
        the strict-mask mode, the chemical-synapse weights are projected
        onto the wild-type adjacency after every optimiser step.
        """
        if self.config.freeze_updates:
            return
        if len(self.buffer) == 0:
            return

        last_value = (
            self.last_value
            if self.last_value is not None
            else torch.tensor(
                [0.0],
                device=self.device,
            )
        )
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value,
            self.gamma,
            self.gae_lambda,
        )

        for _ in range(self.num_epochs):
            for batch in self.buffer.get_minibatches(self.num_minibatches, returns, advantages):
                # Per-step forward pass through the topology + critic.
                # ``forward_with_hidden`` returns both the action logits
                # and the post-K hidden state in one pass (no redundant
                # connectome recompute for the critic).
                states = batch["states"]
                batch_size = states.shape[0]
                new_logits = torch.empty(batch_size, _N_ACTIONS, device=self.device)
                new_values = torch.empty(batch_size, device=self.device)
                for k in range(batch_size):
                    logits_k, hidden_k = self.topology.forward_with_hidden(states[k])
                    new_logits[k] = logits_k
                    new_values[k] = self.critic(hidden_k).squeeze(-1)

                new_probs = torch.softmax(new_logits, dim=-1)
                dist = torch.distributions.Categorical(new_probs)
                new_log_probs = dist.log_prob(batch["actions"])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch["advantages"]
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(new_values, batch["returns"])
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.topology.learnable_parameters + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Strict-mask projection: under "strict" mode, zero out any
                # non-existent edges that the optimiser step would have
                # created. Under "soft_prior" mode, leave the weights as
                # the optimiser left them (the mask is then only a
                # initial-weight prior, and PPO is free to grow new edges).
                if self.config.chemical_mask_mode == "strict":
                    with torch.no_grad():
                        self.topology.w_chem.data.copy_(
                            self.topology.apply_weight_mask(self.topology.w_chem.data),
                        )
