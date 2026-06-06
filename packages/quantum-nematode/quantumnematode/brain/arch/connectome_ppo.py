"""Connectome-constrained PPO brain architecture.

Uses the Cook 2019 *C. elegans* hermaphrodite connectome as a fixed
topology, with chemical synapses strict-masked (PPO-learnable scalar
weights along wild-type edges only) and gap junctions non-learnable
(fixed Cook 2019 synapse counts, fan-in normalised).

Sensor projection: food-chemotaxis input → ASEL/ASER/AWCL/AWCR/AWAL/AWAR
sensory neurons via additive injection scaled by per-input learnable
gains. Motor readout: VB/DB/VA/DA motor-class activations mean-pooled
into a 4-vector, then projected via a learnable readout matrix to either the
discrete 4-action logits (4x4) or the 2-D continuous Gaussian mean (2x4).

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
from quantumnematode.connectome.loader import load_cook_2019_hermaphrodite
from quantumnematode.env.env import ContactZone
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

# Sensory neurons that receive predator-evasion injection. Bargmann/Liu/Pirri
# canonical wiring: distal sulfolipid signal lights up ASH (polymodal
# nociceptors) + ASI (Liu et al. 2018 distal sensors); contact mechanosensation
# routes by approach direction — ALM/AVM (anterior touch, reversal-driving),
# PLM (posterior touch, acceleration-driving); lateral approach has no
# canonical mechanosensor, so ALM+PLM together carry it at half-weight.
_SENSOR_NEURONS_PREDATOR_DISTAL: tuple[str, ...] = ("ASHL", "ASHR", "ASIL", "ASIR")
_SENSOR_NEURONS_PREDATOR_ANTERIOR: tuple[str, ...] = ("ALML", "ALMR", "AVM")
_SENSOR_NEURONS_PREDATOR_POSTERIOR: tuple[str, ...] = ("PLML", "PLMR")

# Lateral-zone routing: reuses anterior + posterior gains scaled by this
# fixed (non-learnable) factor. ALML/ALMR receive the first 2 columns of
# the anterior gain matrix (AVM is excluded — unilateral, not part of the
# bilateral lateral pathway); PLML/PLMR receive the full posterior matrix.
_LATERAL_HALF_WEIGHT: float = 0.5

# Predator feature counts: distal mirrors the env-side
# predator_distal_concentration + its dC/dt derivative; mechano mirrors
# predator_contact_intensity + its intensity-derivative. Both are 2-vectors.
_N_PREDATOR_DISTAL_FEATURES: int = 2
_N_PREDATOR_MECHANO_FEATURES: int = 2

# One-hot encoding of ContactZone for the buffered state vector:
# index 0 = NONE, 1 = ANTERIOR, 2 = POSTERIOR, 3 = LATERAL.
_N_CONTACT_ZONES: int = 4

# Sensory neurons that receive thermotaxis injection. AFD (AFDL/AFDR) is
# the canonical dominant *C. elegans* thermosensor (Mori & Ohshima 1995;
# ~0.01°C sensitivity, ablation abolishes thermotaxis). Targeting AFD
# alone follows the same primary-role-only convention the food projection
# (ASE/AWC/AWA) and predator projection (ASH/ASI/ALM/PLM) already use —
# secondary thermosensory contributors (AWC per Kuhara et al. 2008; AWB,
# ASI) are deliberately not modelled here. AWC's dual odor+temperature
# role is a polymodal-integration refinement deferred to a dedicated
# future change (it would also revisit ASH/AWA dual roles consistently
# across all projections, rather than entangle food + thermo through AWC
# ad-hoc here).
_SENSOR_NEURONS_THERMOTAXIS: tuple[str, ...] = ("AFDL", "AFDR")

# Thermotaxis feature count (klinotaxis head-sweep, mirrors the env-side
# thermotaxis_klinotaxis sensory module): [temp_deviation, lateral, dT/dt].
_N_THERMOTAXIS_FEATURES: int = 3

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
    # Opt-in predator-sensor projection: routes the corrected two-channel
    # predator-sensing biology (distal-chemo onto ASH+ASI; contact-mechano
    # onto ALM/AVM/PLM by ContactZone) into the connectome via three
    # additional learnable gain matrices. Default off so foraging-only
    # configs construct byte-identical parameter sets to pre-projection
    # builds; opting in adds an ``nn.Parameter`` allocation pass that
    # perturbs the RNG-stream consumption order but does not introduce
    # activation drift on the food path.
    enable_predator_projection: bool = False
    # Opt-in thermotaxis-sensor projection: routes the klinotaxis
    # temperature signal (deviation + lateral head-sweep gradient + dT/dt)
    # onto the AFD thermosensory pair (AFDL/AFDR) via one learnable gain
    # matrix. Default off for the same byte-identity reason as the predator
    # projection; constructed after the predator block so flipping it on
    # perturbs neither the food nor the predator RNG-stream order.
    enable_thermotaxis_projection: bool = False
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
    # Predator projection (allocated only when ``enable_predator_projection``).
    _predator_distal_neuron_indices: torch.Tensor
    _predator_anterior_neuron_indices: torch.Tensor
    _predator_posterior_neuron_indices: torch.Tensor
    _predator_lateral_alm_indices: torch.Tensor
    # Thermotaxis projection (allocated only when ``enable_thermotaxis_projection``).
    _thermotaxis_neuron_indices: torch.Tensor

    def __init__(  # noqa: C901, PLR0912, PLR0913, PLR0915 — one-time topology construction
        self,
        connectome: Connectome,
        *,
        enable_gap_junctions: bool,
        forward_pass_depth: int,
        n_food_features: int,
        enforce_strict_mask: bool,
        enable_predator_projection: bool,
        enable_thermotaxis_projection: bool,
        device: torch.device,
        rng: np.random.Generator,
        continuous: bool = False,
    ) -> None:
        super().__init__()
        # Continuous mode: the motor readout maps the 4 motor classes to the 2-D
        # Gaussian mean (instead of 4 discrete logits) + a learnable log-std. The
        # chemical strict-mask / gap junctions are upstream of the readout and
        # untouched by the output mode.
        self.continuous = continuous
        if forward_pass_depth < 1:
            msg = f"forward_pass_depth must be >= 1, got {forward_pass_depth}"
            raise ValueError(msg)
        if n_food_features < 1:
            msg = f"n_food_features must be >= 1, got {n_food_features}"
            raise ValueError(msg)
        self.forward_pass_depth = forward_pass_depth
        self.enable_gap_junctions = enable_gap_junctions
        self.n_food_features = n_food_features
        self.enable_predator_projection = enable_predator_projection
        self.enable_thermotaxis_projection = enable_thermotaxis_projection
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

        # ── Motor readout: VB/DB/VA/DA mean-pool → learnable matrix ─
        # (4x4 → discrete logits, or 2x4 → continuous Gaussian mean)
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
        # Python-int slice bounds for the motor classes, cached at construction
        # so ``_pool_motor`` does not call ``.item()`` per forward (each such
        # call forces a CPU sync). Numerically identical to slicing by the
        # buffer's values — just without the per-step synchronisation.
        self._motor_class_slices: list[tuple[int, int]] = [
            (boundaries[k], boundaries[k + 1]) for k in range(_N_ACTIONS)
        ]
        # Readout maps the 4 motor classes → action outputs: 4 discrete logits,
        # or the 2-D continuous Gaussian mean.
        readout_out_dim = CONTINUOUS_ACTION_DIM if continuous else _N_ACTIONS
        self.readout = nn.Parameter(torch.zeros(readout_out_dim, _N_ACTIONS, device=device))
        # Stronger orthogonal init so the initial policy isn't a constant
        # uniform across actions. ``gain=1.0`` is the standard PPO-actor
        # initial-gain choice; the small ``0.01`` PPO stable-policy trick
        # produces a degenerate-uniform initial policy here because the
        # whole upstream pipeline is small (one 302-vec → 4-vec readout).
        nn.init.orthogonal_(self.readout, gain=1.0)
        # Continuous mode: state-independent learnable log-std (one per action dim).
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(CONTINUOUS_ACTION_DIM, device=device))

        # ── Predator-sensor projection (opt-in) ─────────────────────────
        # Three learnable gain matrices route the corrected two-channel
        # predator-sensing biology into the connectome:
        #   distal-chemo (2 features) -> ASHL+ASHR+ASIL+ASIR  (2x4 gain)
        #   anterior contact (2 feat) -> ALML+ALMR+AVM        (2x3 gain)
        #   posterior contact (2 feat) -> PLML+PLMR           (2x2 gain)
        # Lateral-zone contact reuses anterior + posterior gains scaled by
        # ``_LATERAL_HALF_WEIGHT`` (a fixed constant, not learnable) per
        # the spec — degenerate routing because the connectome has no
        # canonical lateral-only mechanosensor neuron.
        #
        # Bilateral broadcast: independent learnable column per L/R member
        # (e.g. ASHL and ASHR each have their own column in the distal
        # gain matrix) initialised to identical values. Mirrors the
        # food_gains layout and lets PPO updates discover L/R asymmetries
        # that real *C. elegans* sensory pairs develop through experience
        # (Hobert-lab AWC/ASE asymmetry literature).
        #
        # Constructed AFTER the food + readout init so flipping
        # ``enable_predator_projection`` from False → True does not perturb
        # the food-path RNG-stream order: a foraging-only configuration
        # (predator off) consumes the same RNG draws regardless of whether
        # this branch is compiled in.
        if enable_predator_projection:
            # Validate every target neuron exists in the connectome registry
            # BEFORE indexing — a missing name would otherwise raise a bare
            # KeyError from the list-comp below, masking the helpful
            # "connectome-data bug" diagnostic.
            for name in (
                *_SENSOR_NEURONS_PREDATOR_DISTAL,
                *_SENSOR_NEURONS_PREDATOR_ANTERIOR,
                *_SENSOR_NEURONS_PREDATOR_POSTERIOR,
                "ALML",
                "ALMR",
            ):
                if name not in self._idx:
                    msg = (
                        f"Predator-projection neuron {name!r} not present in "
                        f"connectome (have {self.n_neurons} neurons; this is a "
                        "connectome-data bug)."
                    )
                    raise ValueError(msg)
            distal_indices = [self._idx[name] for name in _SENSOR_NEURONS_PREDATOR_DISTAL]
            anterior_indices = [self._idx[name] for name in _SENSOR_NEURONS_PREDATOR_ANTERIOR]
            posterior_indices = [self._idx[name] for name in _SENSOR_NEURONS_PREDATOR_POSTERIOR]
            # Lateral routing: ALML+ALMR receive the first 2 columns of the
            # anterior gain matrix scaled by ``_LATERAL_HALF_WEIGHT``; AVM is
            # excluded because the lateral pathway is bilateral. PLML+PLMR
            # receive the full posterior matrix scaled by ``_LATERAL_HALF_WEIGHT``.
            lateral_alm_indices = [self._idx[name] for name in ("ALML", "ALMR")]
            self.register_buffer(
                "_predator_distal_neuron_indices",
                torch.tensor(distal_indices, dtype=torch.long, device=device),
            )
            self.register_buffer(
                "_predator_anterior_neuron_indices",
                torch.tensor(anterior_indices, dtype=torch.long, device=device),
            )
            self.register_buffer(
                "_predator_posterior_neuron_indices",
                torch.tensor(posterior_indices, dtype=torch.long, device=device),
            )
            self.register_buffer(
                "_predator_lateral_alm_indices",
                torch.tensor(lateral_alm_indices, dtype=torch.long, device=device),
            )
            # Gain matrices: small-magnitude init (std=0.1, not 1.0 like
            # food_gains) so the projection starts close to inert and PPO
            # ramps it up if predator signal is informative. Stronger init
            # would dominate the food signal at step 1 on configs where
            # both projections are active simultaneously.
            self.predator_distal_gains = nn.Parameter(
                torch.zeros(
                    _N_PREDATOR_DISTAL_FEATURES,
                    len(_SENSOR_NEURONS_PREDATOR_DISTAL),
                    device=device,
                ),
            )
            self.predator_anterior_gains = nn.Parameter(
                torch.zeros(
                    _N_PREDATOR_MECHANO_FEATURES,
                    len(_SENSOR_NEURONS_PREDATOR_ANTERIOR),
                    device=device,
                ),
            )
            self.predator_posterior_gains = nn.Parameter(
                torch.zeros(
                    _N_PREDATOR_MECHANO_FEATURES,
                    len(_SENSOR_NEURONS_PREDATOR_POSTERIOR),
                    device=device,
                ),
            )
            nn.init.normal_(self.predator_distal_gains, mean=0.0, std=0.1)
            nn.init.normal_(self.predator_anterior_gains, mean=0.0, std=0.1)
            nn.init.normal_(self.predator_posterior_gains, mean=0.0, std=0.1)

        # ── Thermotaxis-sensor projection (opt-in) ──────────────────────
        # One learnable gain matrix routes the klinotaxis temperature signal
        # (deviation + lateral head-sweep gradient + dT/dt, 3 features) onto
        # the AFD thermosensory pair (AFDL/AFDR) — shape (3, 2). Same
        # bilateral-broadcast convention (one independent column per L/R
        # member, identical small-magnitude init) and the same post-predator
        # construction order as the predator projection, so flipping this
        # flag on perturbs neither the food nor the predator RNG-stream order.
        if enable_thermotaxis_projection:
            # Validate before indexing (mirrors the predator block).
            for name in _SENSOR_NEURONS_THERMOTAXIS:
                if name not in self._idx:
                    msg = (
                        f"Thermotaxis-projection neuron {name!r} not present in "
                        f"connectome (have {self.n_neurons} neurons; this is a "
                        "connectome-data bug)."
                    )
                    raise ValueError(msg)
            thermotaxis_indices = [self._idx[name] for name in _SENSOR_NEURONS_THERMOTAXIS]
            self.register_buffer(
                "_thermotaxis_neuron_indices",
                torch.tensor(thermotaxis_indices, dtype=torch.long, device=device),
            )
            self.thermotaxis_gains = nn.Parameter(
                torch.zeros(
                    _N_THERMOTAXIS_FEATURES,
                    len(_SENSOR_NEURONS_THERMOTAXIS),
                    device=device,
                ),
            )
            nn.init.normal_(self.thermotaxis_gains, mean=0.0, std=0.1)

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
        """Mean-pool ``h`` over the four motor classes → ``(4,)`` or ``(B, 4)``.

        Pools over the LAST dim, so it handles both a single ``(302,)`` hidden
        state and a batched ``(B, 302)`` one. Uses the Python-int class slices
        cached at construction, so there is no per-call ``.item()`` CPU sync.
        Numerically identical to the prior per-class mean.
        """
        last_dim = h.dim() - 1
        flat_acts = h.index_select(last_dim, self._motor_flat_indices)
        means = [
            flat_acts[..., start:end].mean(dim=last_dim) for start, end in self._motor_class_slices
        ]
        return torch.stack(means, dim=last_dim)

    def _inject_predator(
        self,
        h: torch.Tensor,
        predator_distal_features: torch.Tensor | None,
        predator_mechano_features: torch.Tensor | None,
        predator_contact_zone: ContactZone | None,
    ) -> torch.Tensor:
        """Apply the predator-sensor projection onto ``h``.

        Distal-chemo injection fires whenever ``predator_distal_features``
        is provided. Mechano injection fires only when both
        ``predator_mechano_features`` and a non-NONE ``predator_contact_zone``
        are provided — ``ContactZone.NONE`` means "out of damage radius /
        no contact" and produces zero mechano injection per the spec.

        Lateral routing reuses the anterior + posterior gain matrices
        scaled by ``_LATERAL_HALF_WEIGHT`` rather than introducing a
        separate lateral-only gain matrix. ALML/ALMR receive the first
        two columns of the anterior matrix (AVM is excluded — unilateral,
        not part of the bilateral lateral pathway); PLML/PLMR receive
        the full posterior matrix.
        """
        if predator_distal_features is not None:
            distal_inj = predator_distal_features @ self.predator_distal_gains
            h = h.index_add(0, self._predator_distal_neuron_indices, distal_inj)

        if predator_mechano_features is not None and predator_contact_zone not in (
            None,
            ContactZone.NONE,
        ):
            if predator_contact_zone == ContactZone.ANTERIOR:
                ant_inj = predator_mechano_features @ self.predator_anterior_gains
                h = h.index_add(0, self._predator_anterior_neuron_indices, ant_inj)
            elif predator_contact_zone == ContactZone.POSTERIOR:
                post_inj = predator_mechano_features @ self.predator_posterior_gains
                h = h.index_add(0, self._predator_posterior_neuron_indices, post_inj)
            elif predator_contact_zone == ContactZone.LATERAL:
                # ALML/ALMR receive first 2 columns of anterior @ half-weight.
                ant_inj = (
                    predator_mechano_features @ self.predator_anterior_gains
                ) * _LATERAL_HALF_WEIGHT
                h = h.index_add(0, self._predator_lateral_alm_indices, ant_inj[:2])
                # PLML/PLMR receive posterior @ half-weight.
                post_inj = (
                    predator_mechano_features @ self.predator_posterior_gains
                ) * _LATERAL_HALF_WEIGHT
                h = h.index_add(0, self._predator_posterior_neuron_indices, post_inj)
        return h

    def _inject_thermotaxis(
        self,
        h: torch.Tensor,
        thermotaxis_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the thermotaxis-sensor projection onto ``h``.

        Injects the 3-feature klinotaxis temperature vector
        (``[deviation, lateral, dT/dt]``) onto the AFD pair (AFDL/AFDR)
        via the ``(3, 2)`` gain matrix. Fires whenever
        ``thermotaxis_features`` is provided; a zero-filled vector
        (no thermal stimulus) produces zero injection.
        """
        if thermotaxis_features is not None:
            thermo_inj = thermotaxis_features @ self.thermotaxis_gains
            h = h.index_add(0, self._thermotaxis_neuron_indices, thermo_inj)
        return h

    def forward_with_hidden(
        self,
        food_features: torch.Tensor,
        predator_distal_features: torch.Tensor | None = None,
        predator_mechano_features: torch.Tensor | None = None,
        predator_contact_zone: ContactZone | None = None,
        thermotaxis_features: torch.Tensor | None = None,
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

        Predator parameters are only consumed when the topology was
        constructed with ``enable_predator_projection=True`` AND the
        corresponding tensor is non-None. Distal-chemo injection always
        fires when ``predator_distal_features`` is provided; mechano
        injection fires only when ``predator_mechano_features`` AND a
        non-NONE ``predator_contact_zone`` are provided.

        ``thermotaxis_features`` is only consumed when the topology was
        constructed with ``enable_thermotaxis_projection=True`` AND the
        tensor is non-None; it injects the 3-feature temperature vector
        onto the AFD pair.
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

        # Predator-sensor injection (no-op when projection disabled).
        if self.enable_predator_projection:
            h = self._inject_predator(
                h,
                predator_distal_features,
                predator_mechano_features,
                predator_contact_zone,
            )

        # Thermotaxis-sensor injection (no-op when projection disabled).
        if self.enable_thermotaxis_projection:
            h = self._inject_thermotaxis(h, thermotaxis_features)

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
        # (num_actions,) discrete logits, or (2,) continuous Gaussian mean.
        logits = self.readout @ motor_acts
        return logits, h

    def _inject_predator_batched(
        self,
        h: torch.Tensor,
        predator_distal_features: torch.Tensor | None,
        predator_mechano_features: torch.Tensor | None,
        contact_zone_onehot: torch.Tensor | None,
    ) -> torch.Tensor:
        """Batched predator injection — mathematically equal to per-sample.

        ``h`` is ``(B, 302)``; the feature tensors are ``(B, ·)``;
        ``contact_zone_onehot`` is ``(B, 4)`` (columns NONE/ANTERIOR/POSTERIOR/
        LATERAL). The per-sample :meth:`_inject_predator` branches on the
        active zone; this version computes every zone's injection for the
        whole batch and zeroes the inactive ones with the one-hot mask, so
        each sample receives exactly its zone's contribution (adding ``0.0``
        is exact in float, so the masked terms do not perturb).
        """
        if predator_distal_features is not None:
            distal_inj = predator_distal_features @ self.predator_distal_gains  # (B, 4)
            h = h.index_add(1, self._predator_distal_neuron_indices, distal_inj)

        if predator_mechano_features is not None and contact_zone_onehot is not None:
            ant_mask = contact_zone_onehot[:, 1:2]  # ANTERIOR  (B, 1)
            post_mask = contact_zone_onehot[:, 2:3]  # POSTERIOR (B, 1)
            lat_mask = contact_zone_onehot[:, 3:4]  # LATERAL   (B, 1)
            ant_full = predator_mechano_features @ self.predator_anterior_gains  # (B, 3)
            post_full = predator_mechano_features @ self.predator_posterior_gains  # (B, 2)
            # ANTERIOR zone → full anterior gains onto ALML/ALMR/AVM.
            h = h.index_add(1, self._predator_anterior_neuron_indices, ant_full * ant_mask)
            # POSTERIOR zone → full posterior gains onto PLML/PLMR.
            h = h.index_add(1, self._predator_posterior_neuron_indices, post_full * post_mask)
            # LATERAL zone → anterior[:, :2] and posterior at half-weight onto
            # ALML/ALMR and PLML/PLMR (AVM excluded — unilateral).
            h = h.index_add(
                1,
                self._predator_lateral_alm_indices,
                (ant_full[:, :2] * _LATERAL_HALF_WEIGHT) * lat_mask,
            )
            h = h.index_add(
                1,
                self._predator_posterior_neuron_indices,
                (post_full * _LATERAL_HALF_WEIGHT) * lat_mask,
            )
        return h

    def _inject_thermotaxis_batched(
        self,
        h: torch.Tensor,
        thermotaxis_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Batched thermotaxis injection onto AFDL/AFDR (``h`` is ``(B, 302)``)."""
        if thermotaxis_features is not None:
            thermo_inj = thermotaxis_features @ self.thermotaxis_gains  # (B, 2)
            h = h.index_add(1, self._thermotaxis_neuron_indices, thermo_inj)
        return h

    def forward_with_hidden_batched(
        self,
        food_features: torch.Tensor,
        predator_distal_features: torch.Tensor | None = None,
        predator_mechano_features: torch.Tensor | None = None,
        contact_zone_onehot: torch.Tensor | None = None,
        thermotaxis_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched analogue of :meth:`forward_with_hidden`.

        Processes ``B`` samples in one pass: ``food_features`` is
        ``(B, n_food_features)`` and the optional predator/thermotaxis tensors
        are ``(B, ·)`` (``contact_zone_onehot`` is the ``(B, 4)`` zone one-hot,
        consumed directly by the masked predator injection). Returns
        ``(logits, hidden)`` of shapes ``(B, 4)`` and ``(B, 302)``.

        Mathematically equivalent to calling :meth:`forward_with_hidden` once
        per sample, but the recurrence becomes a single ``(B, 302) @ (302, 302)``
        matmul per depth-step instead of B separate matvecs — the load-bearing
        speedup for the PPO update. Float results differ from the per-sample
        path by accumulation-order ulps (batched matmul vs matvec); validated
        equivalent within float32 tolerance by the equivalence test suite.
        """
        batched_ndim = 2  # (B, n_food_features)
        if food_features.ndim != batched_ndim or food_features.shape[1] != self.n_food_features:
            msg = (
                f"food_features must have shape (B, {self.n_food_features}); "
                f"got {tuple(food_features.shape)}"
            )
            raise ValueError(msg)

        batch = food_features.shape[0]
        h = torch.zeros(
            batch,
            self.n_neurons,
            device=food_features.device,
            dtype=food_features.dtype,
        )
        food_injection = food_features @ self.food_gains  # (B, 6)
        h = h.index_add(1, self._food_neuron_indices, food_injection)

        if self.enable_predator_projection:
            h = self._inject_predator_batched(
                h,
                predator_distal_features,
                predator_mechano_features,
                contact_zone_onehot,
            )
        if self.enable_thermotaxis_projection:
            h = self._inject_thermotaxis_batched(h, thermotaxis_features)

        # K recurrent updates. Single-sample uses ``chem_mat.T @ h`` for a
        # 302-vec ``h``; for a ``(B, 302)`` batch the equivalent is
        # ``h @ chem_mat`` (preact[b, post] = sum_pre chem[pre, post] * h[b, pre]).
        # The mask multiply happens once per minibatch here (vs once per sample
        # in the old loop). NOTE: ``h @ gap_mat`` (no transpose) equals the
        # single-sample ``gap_mat.T @ h`` only because ``g_gap`` is constructed
        # symmetric (gap junctions are bidirectional); the chem term needs no
        # such assumption (``h @ chem_mat == chem_mat.T @ h`` for any matrix).
        # If gap junctions ever become directional, transpose ``gap_mat`` here.
        chem_mat = self.w_chem * self.m_chem if self.enforce_strict_mask else self.w_chem
        gap_mat = self.g_gap if self.enable_gap_junctions else torch.zeros_like(self.g_gap)
        for _ in range(self.forward_pass_depth):
            preact = h @ chem_mat + h @ gap_mat
            h = torch.tanh(preact)

        motor_acts = self._pool_motor(h)  # (B, 4)
        # (B, num_actions) discrete logits, or (B, 2) continuous Gaussian mean.
        logits = motor_acts @ self.readout.T
        return logits, h

    def forward(
        self,
        food_features: torch.Tensor,
        predator_distal_features: torch.Tensor | None = None,
        predator_mechano_features: torch.Tensor | None = None,
        predator_contact_zone: ContactZone | None = None,
        thermotaxis_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute 4 action logits from a 2-vector of food-chemotaxis features.

        Thin shim over :meth:`forward_with_hidden` for callers that only
        need the logits. Callers that also need the post-K hidden state
        (e.g. for a value head) should call ``forward_with_hidden`` directly
        to avoid recomputing the connectome forward.
        """
        logits, _ = self.forward_with_hidden(
            food_features,
            predator_distal_features=predator_distal_features,
            predator_mechano_features=predator_mechano_features,
            predator_contact_zone=predator_contact_zone,
            thermotaxis_features=thermotaxis_features,
        )
        return logits

    @property
    def learnable_parameters(self) -> list[nn.Parameter]:
        """Return the PPO-learnable parameter tensors only.

        Includes the predator gain matrices only when the predator
        projection was constructed (``enable_predator_projection=True``)
        and the thermotaxis gain matrix only when the thermotaxis
        projection was constructed (``enable_thermotaxis_projection=True``).
        Foraging-only configs allocate neither, so the optimiser sees
        byte-identical parameter sets to pre-projection builds.
        """
        params = [self.w_chem, self.food_gains, self.readout]
        if self.enable_predator_projection:
            params.extend(
                [
                    self.predator_distal_gains,
                    self.predator_anterior_gains,
                    self.predator_posterior_gains,
                ],
            )
        if self.enable_thermotaxis_projection:
            params.append(self.thermotaxis_gains)
        if self.continuous:
            params.append(self.log_std)
        return params


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
        # Action mode: discrete (categorical) or continuous (tanh-squashed Gaussian
        # over a normalized (speed, turn) vector; the env rescales to physical units).
        self.continuous = config.action_mode == "continuous"
        self._action_low = torch.tensor([0.0, -1.0], device=self.device)
        self._action_high = torch.tensor([1.0, 1.0], device=self.device)
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
            enable_predator_projection=config.enable_predator_projection,
            enable_thermotaxis_projection=config.enable_thermotaxis_projection,
            device=self.device,
            rng=self.rng,
            continuous=self.continuous,
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
        self.buffer = RolloutBuffer(
            config.rollout_buffer_size,
            self.device,
            rng=self.rng,
            continuous_actions=self.continuous,
        )

        # Brain Protocol state.
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()
        self.current_probabilities: np.ndarray | None = None
        self.last_value: torch.Tensor | None = None

        # Per-step pending data (added to buffer when next reward arrives).
        self._pending_state: np.ndarray | None = None
        self._pending_action: int | np.ndarray | None = None
        self._pending_log_prob: torch.Tensor | None = None
        self._pending_value: torch.Tensor | None = None

    # ── Brain Protocol surface ──────────────────────────────────────────

    @property
    def action_set(self) -> list[Action]:
        """Return the 4-action discrete action set."""
        return self._action_set

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Extract the per-step feature vector for the topology.

        Food-only layout (``enable_predator_projection=False``):

        - ``oracle`` → ``[strength, angle]`` (2 features). ``strength`` is
          the env's ``food_gradient_strength`` field (∈ [0, 1]); ``angle``
          is ``food_gradient_direction`` (radians ∈ [-π, π]) normalised to
          [-1, 1] by dividing by π.
        - ``klinotaxis`` → ``[concentration, lateral, dC/dt]`` (3 features).

        With predator projection enabled, the vector is extended with:
        ``[predator_distal_concentration, predator_distal_dconcentration_dt,
        predator_contact_intensity, predator_mechano_dintensity_dt,
        zone_NONE, zone_ANTERIOR, zone_POSTERIOR, zone_LATERAL]`` (8 extra
        features; the zone slots are a one-hot of ``ContactZone``). Slots
        default to 0 (and zone defaults to NONE) when the corresponding
        ``BrainParams`` fields are unset — so a predator-enabled brain
        running on a config without active predator sensors produces zero
        injection at the predator path.

        With thermotaxis projection enabled, the vector is further extended
        with ``[temp_deviation, temperature_lateral_gradient, temperature_ddt]``
        (3 features), appended AFTER any predator slots. Same zero-default
        semantics: an isothermal / no-thermal-stimulus step produces a
        zero-filled thermo block.
        """
        parts = [self._extract_food_features(params)]
        if self.config.enable_predator_projection:
            parts.append(self._extract_predator_features(params))
        if self.config.enable_thermotaxis_projection:
            parts.append(self._extract_thermotaxis_features(params))
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts).astype(np.float32)

    def _extract_food_features(self, params: BrainParams) -> np.ndarray:
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

    def _extract_predator_features(self, params: BrainParams) -> np.ndarray:
        """Pack predator features + one-hot zone into an 8-vector."""
        distal_c = float(params.predator_distal_concentration or 0.0)
        distal_dc = float(params.predator_distal_dconcentration_dt or 0.0)
        contact_i = float(params.predator_contact_intensity or 0.0)
        mechano_di = float(params.predator_mechano_dintensity_dt or 0.0)
        zone = params.predator_contact_zone or ContactZone.NONE
        zone_onehot = np.zeros(_N_CONTACT_ZONES, dtype=np.float32)
        zone_idx = {
            ContactZone.NONE: 0,
            ContactZone.ANTERIOR: 1,
            ContactZone.POSTERIOR: 2,
            ContactZone.LATERAL: 3,
        }[zone]
        zone_onehot[zone_idx] = 1.0
        return np.concatenate(
            [
                np.array([distal_c, distal_dc, contact_i, mechano_di], dtype=np.float32),
                zone_onehot,
            ],
        )

    def _extract_thermotaxis_features(self, params: BrainParams) -> np.ndarray:
        """Pack klinotaxis thermotaxis features into a 3-vector.

        Mirrors the env-side ``thermotaxis_klinotaxis`` sensory module:
        ``[temp_deviation, lateral, dT/dt]`` where ``temp_deviation`` is
        ``clip((temperature - cultivation_temp) / 15, -1, 1)``. A step with
        no thermal stimulus (``temperature is None``) yields all zeros.
        """
        if params.temperature is None:
            return np.zeros(_N_THERMOTAXIS_FEATURES, dtype=np.float32)
        cultivation = (
            20.0 if params.cultivation_temperature is None else params.cultivation_temperature
        )
        deviation = float(np.clip((params.temperature - cultivation) / 15.0, -1.0, 1.0))
        raw_lateral = float(params.temperature_lateral_gradient or 0.0)
        lateral = float(np.tanh(raw_lateral * params.lateral_scale))
        raw_deriv = float(params.temperature_ddt or 0.0)
        dtdt = float(np.tanh(raw_deriv * params.derivative_scale))
        return np.array([deviation, lateral, dtdt], dtype=np.float32)

    def _unpack_state(
        self,
        state_t: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        ContactZone | None,
        torch.Tensor | None,
    ]:
        """Split a buffered state tensor back into the forward-pass inputs.

        Returns ``(food, predator_distal, predator_mechano, contact_zone,
        thermotaxis)``. The state vector is laid out by :meth:`preprocess`
        as ``[food] + [predator(8) if enabled] + [thermo(3) if enabled]``;
        this helper reverses that layout. Disabled blocks return ``None``
        (and ``contact_zone`` returns ``None`` when the predator block is
        absent) so the topology runs only the active projections.
        """
        n_food = self._n_food_features
        food = state_t[:n_food]
        offset = n_food

        distal: torch.Tensor | None = None
        mechano: torch.Tensor | None = None
        zone: ContactZone | None = None
        if self.config.enable_predator_projection:
            distal = state_t[offset : offset + _N_PREDATOR_DISTAL_FEATURES]
            mech_start = offset + _N_PREDATOR_DISTAL_FEATURES
            mechano = state_t[mech_start : mech_start + _N_PREDATOR_MECHANO_FEATURES]
            zone_start = mech_start + _N_PREDATOR_MECHANO_FEATURES
            zone_onehot = state_t[zone_start : zone_start + _N_CONTACT_ZONES]
            zone_idx = int(torch.argmax(zone_onehot).item())
            zone = (
                ContactZone.NONE,
                ContactZone.ANTERIOR,
                ContactZone.POSTERIOR,
                ContactZone.LATERAL,
            )[zone_idx]
            offset = zone_start + _N_CONTACT_ZONES

        thermo: torch.Tensor | None = None
        if self.config.enable_thermotaxis_projection:
            thermo = state_t[offset : offset + _N_THERMOTAXIS_FEATURES]

        return food, distal, mechano, zone, thermo

    def _unpack_state_batched(
        self,
        states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Batched :meth:`_unpack_state` for the PPO update.

        ``states`` is ``(B, state_dim)``; returns ``(food, predator_distal,
        predator_mechano, contact_zone_onehot, thermotaxis)`` sliced along dim
        1. Unlike the single-sample version, the contact zone is returned as
        the raw ``(B, 4)`` one-hot (not decoded to a ``ContactZone`` enum) —
        :meth:`forward_with_hidden_batched` consumes the one-hot directly for
        its masked injection.
        """
        n_food = self._n_food_features
        food = states[:, :n_food]
        offset = n_food

        distal: torch.Tensor | None = None
        mechano: torch.Tensor | None = None
        zone_onehot: torch.Tensor | None = None
        if self.config.enable_predator_projection:
            distal = states[:, offset : offset + _N_PREDATOR_DISTAL_FEATURES]
            mech_start = offset + _N_PREDATOR_DISTAL_FEATURES
            mechano = states[:, mech_start : mech_start + _N_PREDATOR_MECHANO_FEATURES]
            zone_start = mech_start + _N_PREDATOR_MECHANO_FEATURES
            zone_onehot = states[:, zone_start : zone_start + _N_CONTACT_ZONES]
            offset = zone_start + _N_CONTACT_ZONES

        thermo: torch.Tensor | None = None
        if self.config.enable_thermotaxis_projection:
            thermo = states[:, offset : offset + _N_THERMOTAXIS_FEATURES]

        return food, distal, mechano, zone_onehot, thermo

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
        # post-K 302-dim hidden state used by the critic. ``_unpack_state``
        # splits the buffered state vector back into the active projections'
        # forward-pass inputs; disabled projections stay None and the
        # topology runs only the active paths.
        food, distal, mechano, zone, thermo = self._unpack_state(state_t)
        head_out, hidden = self.topology.forward_with_hidden(
            food,
            predator_distal_features=distal,
            predator_mechano_features=mechano,
            predator_contact_zone=zone,
            thermotaxis_features=thermo,
        )
        value = self.critic(hidden)
        self.last_value = value

        if self.continuous:
            return self._run_brain_continuous(head_out, state, value)

        # Action distribution via the shared discrete policy helper (byte-equivalent
        # to the prior inline softmax → Categorical → sample/log_prob).
        action_idx, log_prob, _entropy, probs = categorical_sample_torch(
            head_out,
            device=self.device,
        )

        action_name = self._action_set[action_idx]

        # Store pending data for the buffer (added on the next reward).
        self._pending_state = state
        self._pending_action = action_idx
        self._pending_log_prob = log_prob
        self._pending_value = value

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

    def _run_brain_continuous(
        self,
        mean: torch.Tensor,
        state: np.ndarray,
        value: torch.Tensor,
    ) -> list[ActionData]:
        """Continuous-mode action step: sample a normalized ``(speed, turn)`` action.

        Parameters
        ----------
        mean : torch.Tensor
            The 2-D Gaussian mean from the motor readout.
        state : np.ndarray
            Preprocessed state (stored for the PPO update's batched re-forward).
        value : torch.Tensor
            The critic value estimate for the current step.

        Returns
        -------
        list[ActionData]
            A single-element list whose ``continuous`` carries the normalized
            ``(speed, turn)`` action.

        Notes
        -----
        Samples from the tanh-squashed Gaussian head and stores the pre-squash
        sample for the PPO update; the environment rescales the normalized action
        to physical units. The chemical strict-mask / gap junctions are upstream of
        the readout and unaffected.
        """
        action_vec, log_prob, _entropy, pre_tanh = continuous_sample_tanh_gaussian(
            mean,
            self.topology.log_std,
            self._action_low,
            self._action_high,
        )
        continuous_action = (action_vec[0].item(), action_vec[1].item())

        self._pending_state = state
        self._pending_action = pre_tanh.detach().cpu().numpy()
        self._pending_log_prob = log_prob
        self._pending_value = value
        self.current_probabilities = None

        action_data = ActionData(
            state="continuous",
            action=None,
            probability=torch.exp(log_prob.detach()).item(),
            continuous=continuous_action,
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
                # Batched forward pass through the topology + critic. The
                # minibatch's states are unpacked + run in ONE batched
                # connectome forward (and the post-K hidden states feed the
                # critic in one call) — replacing the prior per-sample Python
                # loop, where ~90% of connectome forward passes occurred.
                states = batch["states"]
                food_b, distal_b, mechano_b, zone_onehot_b, thermo_b = self._unpack_state_batched(
                    states,
                )
                new_head_out, hidden = self.topology.forward_with_hidden_batched(
                    food_b,
                    predator_distal_features=distal_b,
                    predator_mechano_features=mechano_b,
                    contact_zone_onehot=zone_onehot_b,
                    thermotaxis_features=thermo_b,
                )
                new_values = self.critic(hidden).squeeze(-1)

                # Re-score actions under the current policy via the shared module:
                # discrete (Categorical) or continuous (tanh-Gaussian re-scoring the
                # stored pre-squash samples). Clipped surrogate is shared.
                if self.continuous:
                    new_log_probs, entropy = continuous_evaluate_tanh_gaussian(
                        new_head_out,
                        self.topology.log_std,
                        batch["actions"],
                        self._action_low,
                        self._action_high,
                    )
                else:
                    new_log_probs, entropy = categorical_evaluate_torch(
                        new_head_out,
                        batch["actions"],
                    )
                policy_loss = ppo_clip_policy_loss(
                    new_log_probs,
                    batch["old_log_probs"],
                    batch["advantages"],
                    self.clip_epsilon,
                )
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
