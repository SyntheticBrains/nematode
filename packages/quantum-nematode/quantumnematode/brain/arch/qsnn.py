"""
Quantum Spiking Neural Network (QSNN) Brain Architecture.

A hybrid quantum-classical spiking neural network that executes quantum circuits
(QLIF neurons) in the forward pass and uses classical surrogate gradients in the
backward pass for REINFORCE policy gradient learning. This sidesteps both the
parameter-shift rule's computational cost and the barren plateau problem inherent
in deep variational quantum circuits.

Key Features:
- **QLIF Neurons**: 2-gate circuit per Brand & Petruccione (2024):
  |0> -> RY(theta + tanh(w·x)·pi) -> RX(theta_leak) -> Measure
- **Surrogate Gradient Learning**: QLIFSurrogateSpike autograd function — forward
  pass returns quantum-measured spike probability, backward pass uses sigmoid
  surrogate centered at pi/2 (the RY gate's quantum transition point)
- **REINFORCE Policy Gradient**: Advantage-normalized returns with clipping,
  entropy regularization, and exploration decay schedule
- **Layered Feedforward SNN**: sensory -> hidden -> motor (e.g. 6->8->4, ~60 params)
- **Dual Learning Modes**: Surrogate gradient (default) or 3-factor Hebbian (legacy)

Architecture::

    Sensors -> Sigmoid -> QLIF(sensory->hidden) -> QLIF(hidden->motor) -> Softmax -> Action
                          [quantum circuits]       [quantum circuits]

The hybrid training approach works as follows:
1. Forward: Encode features as spike probabilities, propagate through QLIF quantum
   circuits layer by layer, measure spike probabilities from quantum state
2. Action: Convert motor spike probabilities to action logits, apply temperature
   scaling and epsilon-greedy exploration, sample action
3. Backward: Recompute forward pass with QLIFSurrogateSpike wrapping each quantum
   measurement — sigmoid surrogate gradient flows through RY angles to both
   theta (membrane bias) and weight parameters via autograd
4. Update: REINFORCE loss with advantage clipping and entropy bonus, Adam optimizer
   with gradient clipping, weight clamping for stability

References
----------
- Brand & Petruccione (2024). "A quantum leaky integrate-and-fire spiking neuron
  and network." npj Quantum Information, 10(1), 16.
- Neftci et al. (2019). "Surrogate gradient learning in spiking neural networks."
  IEEE Signal Processing Magazine, 36(6), 51-63.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pydantic import Field, field_validator
from qiskit import QuantumCircuit

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch.dtypes import BrainConfig, DeviceType
from quantumnematode.brain.modules import (
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)
from quantumnematode.env import Direction
from quantumnematode.errors import ERROR_MISSING_IMPORT_QISKIT_AER
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

# Defaults
DEFAULT_NUM_SENSORY_NEURONS = 6
DEFAULT_NUM_HIDDEN_NEURONS = 8
DEFAULT_NUM_MOTOR_NEURONS = 4
DEFAULT_MEMBRANE_TAU = 0.9
DEFAULT_THRESHOLD = 0.5
DEFAULT_REFRACTORY_PERIOD = 0
DEFAULT_USE_LOCAL_LEARNING = True
DEFAULT_SHOTS = 1024
DEFAULT_GAMMA = 0.99
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_WEIGHT_CLIP = 5.0
DEFAULT_UPDATE_INTERVAL = 20

# Validation constants
MIN_SENSORY_NEURONS = 1
MIN_HIDDEN_NEURONS = 1
MIN_MOTOR_NEURONS = 2
MIN_SHOTS = 100

# L2 weight decay — gentle regularization now that eligibility normalization
# bounds update magnitude. Keeps weights from drifting unboundedly over many
# episodes without fighting against the learning signal.
WEIGHT_DECAY_FACTOR = 0.01

# Weight initialization scale (small to keep RY angles in meaningful range)
WEIGHT_INIT_SCALE = 0.1

# Eligibility trace normalization — caps the Frobenius norm of eligibility traces
# before multiplying by lr * advantage. This bounds the maximum weight delta per
# update to approximately lr * MAX_ELIGIBILITY_NORM * |advantage|.
# With lr=0.1, max_norm=1.0, and typical advantage ~1-5, max delta per update
# is ~0.1-0.5, keeping weights in tanh's useful range [-2, 2].
MAX_ELIGIBILITY_NORM = 1.0

# Exploration floor — mixes softmax action probs with uniform distribution,
# guaranteeing every action is sampled with at least epsilon/num_actions
# probability and preventing policy collapse where one action reaches >99%.
# With epsilon=0.1 and 4 actions, minimum per-action probability = 2.5%.
EXPLORATION_EPSILON = 0.1

# Surrogate gradient sharpness parameter (alpha).
# Controls how closely the sigmoid surrogate approximates the Heaviside step.
# Higher = sharper but noisier gradients. 1.0 matches SpikingReinforceBrain's
# proven setting; smoother gradients reduce noise from quantum shot variance.
DEFAULT_SURROGATE_ALPHA = 1.0

# Gradient clipping max norm for surrogate gradient mode.
# Prevents large gradient spikes from destabilizing training.
SURROGATE_GRAD_CLIP = 1.0

# Advantage clipping for surrogate gradient REINFORCE.
# Prevents outlier returns from producing catastrophically large policy updates.
# SpikingReinforceBrain uses 2.0; this caps normalized advantages to [-2, +2].
ADVANTAGE_CLIP = 2.0

# Logit scaling factor for converting spike probabilities to action logits.
# Maps spike probs in [0,1] to logits via (prob - 0.5) * scale.
# Higher values create sharper action differentiation from small spike differences.
# With 4 actions and single-timestep spike probs, 20.0 gives meaningful separation.
LOGIT_SCALE = 20.0

# Exploration decay: number of episodes over which exploration decreases.
# Epsilon and temperature decay linearly from initial to final values over
# this many episodes, allowing more exploitation as the policy matures.
EXPLORATION_DECAY_EPISODES = 30

# Learning rate decay: number of episodes over which LR decays via cosine
# annealing. After this many episodes, LR reaches its minimum value.
# 200 matches the typical training length for foraging experiments.
LR_DECAY_EPISODES = 200

# Minimum LR as a fraction of the initial LR. With lr=0.01 and factor=0.1,
# the LR decays from 0.01 to 0.001 over LR_DECAY_EPISODES episodes.
# This prevents late-episode weight perturbation that causes catastrophic
# forgetting in converged policies (observed in R12f/R12g sessions).
LR_MIN_FACTOR = 0.1

# Multi-timestep integration: number of QLIF timesteps per decision.
# SpikingReinforceBrain uses 100 classical timesteps; QSNN uses quantum circuits
# (much more expensive per timestep), so we use a smaller default. Averaging
# spike probabilities across multiple timesteps reduces quantum shot noise
# variance, giving REINFORCE cleaner gradient signal. With shots=1024 and
# 10 timesteps, effective samples per decision = 10240.
DEFAULT_NUM_INTEGRATION_STEPS = 10

# Adaptive entropy bonus: when mean episode entropy drops below this threshold,
# entropy_coef is scaled up to push the policy back toward exploration.
# 0.5 nats ≈ 36% of max entropy for 4 actions (ln(4)≈1.386). Sessions that
# collapse below this in R12h never recover; adaptive scaling rescues them.
ENTROPY_FLOOR = 0.5

# Maximum multiplier for entropy_coef when entropy is critically low.
# When entropy → 0, effective entropy_coef = base * ENTROPY_BOOST_MAX.
# With base entropy_coef=0.02, max effective = 0.02 * 5.0 = 0.10.
ENTROPY_BOOST_MAX = 5.0


class QLIFSurrogateSpike(torch.autograd.Function):
    """Custom autograd function for QLIF neuron surrogate gradients.

    Forward pass uses the actual quantum-measured spike probability (from QLIF
    circuit execution). Backward pass uses a sigmoid surrogate gradient, which
    approximates how the spike probability changes with the RY angle.

    The QLIF circuit computes: |0> -> RY(ry_angle) -> RX(leak) -> Measure
    where ry_angle = theta + tanh(w·x) * pi. The surrogate approximates
    d(spike_prob)/d(ry_angle), and autograd chains through to both theta
    and weights via d(ry_angle)/d(theta) = 1 and
    d(ry_angle)/d(weighted_input) = pi * sech²(w·x).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        ry_angle: torch.Tensor,
        quantum_spike_prob: float,
        alpha: float = DEFAULT_SURROGATE_ALPHA,
    ) -> torch.Tensor:
        """Forward pass: return quantum spike probability as a differentiable tensor.

        Parameters
        ----------
        ctx : torch.autograd.Function context
            Autograd context for saving tensors.
        ry_angle : torch.Tensor
            The RY rotation angle (theta + tanh(w·x)*pi), scalar tensor with grad.
            This is the primary differentiable input connecting both theta and weights.
        quantum_spike_prob : float
            The spike probability measured from the quantum circuit.
        alpha : float
            Surrogate gradient sharpness parameter.
        """
        ctx.save_for_backward(
            ry_angle,
            torch.tensor(alpha, device=ry_angle.device),
        )
        return torch.tensor(
            quantum_spike_prob,
            dtype=torch.float32,
            device=ry_angle.device,
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        """Backward pass: sigmoid surrogate gradient.

        Approximates d(spike_prob)/d(ry_angle) using a sigmoid derivative.
        The sigmoid is centered at pi/2 (where spike probability transitions
        from low to high for the RY gate).
        """
        ry_angle, alpha = ctx.saved_tensors
        # Sigmoid surrogate centered at pi/2 (the RY transition point)
        # For RY(angle): P(1) ~ sin²(angle/2), which transitions at angle=pi/2
        shifted = alpha * (ry_angle - torch.pi / 2)
        sigma = torch.sigmoid(shifted)
        grad_surrogate = alpha * sigma * (1 - sigma)
        return grad_output * grad_surrogate, None, None


class QSNNBrainConfig(BrainConfig):
    """Configuration for the QSNNBrain architecture.

    Supports two modes for input feature extraction:

    1. **Legacy mode** (default): Uses 2 features (gradient_strength, relative_angle)
       - Set `sensory_modules=None` (default)

    2. **Unified sensory mode**: Uses modular feature extraction from brain/modules.py
       - Set `sensory_modules` to a list of ModuleName values
       - Uses extract_classical_features() which outputs semantic-preserving ranges
       - Each module contributes 2 features [strength, angle] in [0,1] and [-1,1]

    Attributes
    ----------
    num_sensory_neurons : int
        Number of sensory layer neurons (default 6).
    num_hidden_neurons : int
        Number of hidden interneuron layer neurons (default 4).
    num_motor_neurons : int
        Number of motor layer neurons matching action space (default 4).
    membrane_tau : float
        Leak time constant for QLIF neurons in (0, 1] (default 0.9).
    threshold : float
        Firing threshold in (0, 1) (default 0.5).
    refractory_period : int
        Timesteps to suppress activity after firing (default 2).
    use_local_learning : bool
        Use 3-factor local learning vs REINFORCE fallback (default True).
    shots : int
        Number of quantum measurement shots (default 1024).
    gamma : float
        Discount factor for returns (default 0.99).
    learning_rate : float
        Learning rate for weight updates (default 0.01).
    entropy_coef : float
        Entropy regularization coefficient (default 0.01).
    weight_clip : float
        Maximum absolute weight value for stability (default 5.0).
    sensory_modules : list[ModuleName] | None
        List of sensory modules for feature extraction (None = legacy mode).
    """

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
    use_local_learning: bool = Field(
        default=DEFAULT_USE_LOCAL_LEARNING,
        description="Use 3-factor local learning vs REINFORCE fallback.",
    )
    shots: int = Field(
        default=DEFAULT_SHOTS,
        description="Number of quantum measurement shots.",
    )
    gamma: float = Field(
        default=DEFAULT_GAMMA,
        description="Discount factor for returns.",
    )
    learning_rate: float = Field(
        default=DEFAULT_LEARNING_RATE,
        description="Learning rate for weight updates.",
    )
    entropy_coef: float = Field(
        default=DEFAULT_ENTROPY_COEF,
        description="Entropy regularization coefficient.",
    )
    weight_clip: float = Field(
        default=DEFAULT_WEIGHT_CLIP,
        description="Maximum absolute weight value for stability.",
    )
    update_interval: int = Field(
        default=DEFAULT_UPDATE_INTERVAL,
        description="Apply weight updates every N steps within an episode (0 = episode-end only).",
    )
    num_integration_steps: int = Field(
        default=DEFAULT_NUM_INTEGRATION_STEPS,
        description="Number of QLIF timesteps per decision (averages out quantum shot noise).",
    )

    # Unified sensory feature extraction (optional)
    sensory_modules: list[ModuleName] | None = Field(
        default=None,
        description="List of sensory modules for feature extraction (None = legacy mode).",
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


class QSNNBrain(ClassicalBrain):
    """
    Quantum Spiking Neural Network brain architecture.

    Uses QLIF neurons with trainable synaptic weights and local learning rules
    to avoid both QRC's representation problem and QVarCircuit's barren plateaus.
    """

    def __init__(
        self,
        config: QSNNBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] | None = None,
    ) -> None:
        """Initialize the QSNNBrain.

        Parameters
        ----------
        config : QSNNBrainConfig
            Configuration for the QSNN brain architecture.
        num_actions : int
            Number of available actions (default 4: forward, left, right, backward).
        device : DeviceType
            Device for PyTorch operations (default CPU).
        action_set : list[Action] | None
            Custom action set (default is DEFAULT_ACTIONS).
        """
        super().__init__()

        self.config = config
        self.num_actions = num_actions
        self.device = torch.device(device.value)
        self._action_set = action_set if action_set is not None else DEFAULT_ACTIONS[:num_actions]

        # Validate action_set length matches num_actions
        if self.num_actions != len(self._action_set):
            msg = (
                f"num_actions ({self.num_actions}) does not match "
                f"action_set length ({len(self._action_set)})"
            )
            raise ValueError(msg)

        # Initialize seeding for reproducibility
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)
        logger.info(f"QSNNBrain using seed: {self.seed}")

        # Store sensory modules for feature extraction
        self.sensory_modules = config.sensory_modules

        # Determine input dimension based on sensory modules
        if config.sensory_modules is not None:
            self.input_dim = get_classical_feature_dimension(config.sensory_modules)
            logger.info(
                f"Using unified sensory modules: "
                f"{[m.value for m in config.sensory_modules]} "
                f"(input_dim={self.input_dim})",
            )
        else:
            self.input_dim = 2
            logger.info("Using legacy 2-feature preprocessing (gradient_strength, rel_angle)")

        # Initialize data tracking
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()

        # Network configuration from config
        self.num_sensory = config.num_sensory_neurons
        self.num_hidden = config.num_hidden_neurons
        self.num_motor = config.num_motor_neurons
        self.membrane_tau = config.membrane_tau
        self.threshold = config.threshold
        self.refractory_period = config.refractory_period
        self.shots = config.shots
        self.use_local_learning = config.use_local_learning
        self.gamma = config.gamma
        self.learning_rate = config.learning_rate
        self.entropy_coef = config.entropy_coef
        self.weight_clip = config.weight_clip
        self.update_interval = config.update_interval
        self.num_integration_steps = config.num_integration_steps

        # Step counter for intra-episode updates
        self._step_count = 0

        # Episode counter for exploration decay
        self._episode_count = 0

        # Compute leak angle: theta_leak = (1 - membrane_tau) * pi
        self.leak_angle = (1 - self.membrane_tau) * np.pi

        # Initialize network components
        self._init_network_weights()
        self._init_episode_state()

        logger.info(
            f"QSNNBrain initialized: {self.num_sensory}->{self.num_hidden}->{self.num_motor} "
            f"neurons, membrane_tau={self.membrane_tau}, threshold={self.threshold}, "
            f"use_local_learning={self.use_local_learning}, "
            f"num_integration_steps={self.num_integration_steps}",
        )

    def _init_network_weights(self) -> None:
        """Initialize trainable weight matrices and neuron parameters."""
        # Random Gaussian init with small scale to keep weighted inputs
        # in tanh's linear region. Random magnitudes provide natural symmetry
        # breaking — each column has different norm, so hidden neurons produce
        # varied spike probabilities from step 1. This lets REINFORCE shape
        # the policy immediately. (R12i/R12j showed orthogonal init creates
        # a symmetry trap: uniform column norms → identical spike probs →
        # 25-episode dead zone where entropy is locked at maximum.)
        self.W_sh = torch.randn(
            self.num_sensory,
            self.num_hidden,
            device=self.device,
        ) * WEIGHT_INIT_SCALE

        self.W_hm = torch.randn(
            self.num_hidden,
            self.num_motor,
            device=self.device,
        ) * WEIGHT_INIT_SCALE

        # Trainable membrane potential parameters per neuron.
        # Theta=0 provides a natural "cold start": hidden neurons begin near
        # P(spike)≈0, so early gradients are small and ramp up as weights grow.
        # This prevents premature policy commitment and entropy collapse.
        # (R12i showed theta=π/2 causes catastrophic entropy collapse →0.13
        # within 30-50 episodes due to maximum gradient sensitivity from step 1.)
        # Motor thetas stay at zero to avoid biasing initial action preferences.
        self.theta_hidden = torch.zeros(self.num_hidden, device=self.device)
        self.theta_motor = torch.zeros(self.num_motor, device=self.device)

        # Surrogate gradient mode: enable autograd and create optimizer
        if not self.use_local_learning:
            self.W_sh.requires_grad_(True)  # noqa: FBT003
            self.W_hm.requires_grad_(True)  # noqa: FBT003
            self.theta_hidden.requires_grad_(True)  # noqa: FBT003
            self.theta_motor.requires_grad_(True)  # noqa: FBT003
            self.optimizer = torch.optim.Adam(
                [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor],
                lr=self.learning_rate,
            )
            # Cosine annealing LR decay: lr decays from initial to initial*LR_MIN_FACTOR
            # over LR_DECAY_EPISODES episodes. Prevents late-episode weight perturbation
            # that causes catastrophic forgetting in converged policies.
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=LR_DECAY_EPISODES,
                eta_min=self.learning_rate * LR_MIN_FACTOR,
            )
        else:
            self.optimizer = None
            self.scheduler = None

        # Eligibility traces for local learning (accumulated per episode)
        self.eligibility_sh = torch.zeros_like(self.W_sh)
        self.eligibility_hm = torch.zeros_like(self.W_hm)

        # Eligibility traces for theta (membrane) parameters
        self.eligibility_theta_hidden = torch.zeros_like(self.theta_hidden)
        self.eligibility_theta_motor = torch.zeros_like(self.theta_motor)

        # Refractory state tracking
        self.refractory_hidden = np.zeros(self.num_hidden, dtype=np.int32)
        self.refractory_motor = np.zeros(self.num_motor, dtype=np.int32)

        # Last timestep spike data for deferred eligibility accumulation
        self._last_sensory_spikes = np.zeros(self.num_sensory)
        self._last_hidden_spikes = np.zeros(self.num_hidden)
        self._last_motor_spikes = np.zeros(self.num_motor)

        # Qiskit backend for circuit execution
        self._backend = None

    def _init_episode_state(self) -> None:
        """Initialize episode tracking state."""
        self.episode_rewards: list[float] = []
        self.episode_actions: list[int] = []
        self.episode_features: list[np.ndarray] = []

        # REINFORCE baseline
        self.baseline = 0.0
        self.baseline_alpha = 0.05

        # Current action probabilities
        self.current_probabilities: np.ndarray | None = None
        self.training = True

    def _get_backend(self):  # noqa: ANN202
        """Get or create the Qiskit Aer backend for circuit execution."""
        if self._backend is None:
            try:
                from qiskit_aer import AerSimulator
            except ImportError as err:
                error_message = ERROR_MISSING_IMPORT_QISKIT_AER
                logger.error(error_message)
                raise ImportError(error_message) from err

            self._backend = AerSimulator(
                device="CPU",
                seed_simulator=self.seed,
            )
        return self._backend

    def _build_qlif_circuit(
        self,
        weighted_input: float,
        theta_membrane: float,
    ) -> QuantumCircuit:
        """Build a QLIF neuron circuit.

        The minimal 2-gate QLIF circuit from Brand & Petruccione (2024):
        |0⟩ → RY(θ_membrane + weighted_input) → RX(θ_leak) → Measure

        Parameters
        ----------
        weighted_input : float
            Sum of w_ij * spike_j for all presynaptic neurons.
        theta_membrane : float
            Trainable membrane potential parameter.

        Returns
        -------
        QuantumCircuit
            The QLIF neuron circuit.
        """
        qc = QuantumCircuit(1, 1)

        # RY encodes membrane potential + weighted input
        # tanh bounds the weighted input to [-1, 1] before scaling by pi,
        # preventing angle wrapping through multiple cycles (hash function behavior)
        normalized_input = float(np.tanh(weighted_input))
        ry_angle = float(theta_membrane + normalized_input * np.pi)
        qc.ry(ry_angle, 0)

        # RX implements leak
        qc.rx(self.leak_angle, 0)

        # Measure
        qc.measure(0, 0)

        return qc

    def _execute_qlif_layer(
        self,
        pre_spikes: np.ndarray,
        weights: torch.Tensor,
        theta_membrane: torch.Tensor,
        refractory_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute a layer of QLIF neurons.

        Parameters
        ----------
        pre_spikes : np.ndarray
            Spike probabilities from presynaptic layer, shape (num_pre,).
        weights : torch.Tensor
            Weight matrix, shape (num_pre, num_post).
        theta_membrane : torch.Tensor
            Membrane potential parameters, shape (num_post,).
        refractory_state : np.ndarray
            Refractory countdown per neuron, shape (num_post,).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (spike_probs, updated_refractory_state) for the layer.
        """
        num_post = weights.shape[1]
        spike_probs = np.zeros(num_post)
        backend = self._get_backend()

        # Convert weights to numpy for computation
        weights_np = weights.detach().cpu().numpy()
        theta_np = theta_membrane.detach().cpu().numpy()

        for j in range(num_post):
            # Check refractory period
            if refractory_state[j] > 0:
                refractory_state[j] -= 1
                spike_probs[j] = 0.0
                continue

            # Compute weighted input: sum(w_ij * spike_i)
            weighted_input = np.dot(pre_spikes, weights_np[:, j])

            # Build and execute QLIF circuit
            qc = self._build_qlif_circuit(weighted_input, theta_np[j])
            job = backend.run(qc, shots=self.shots)
            result = job.result()
            counts = result.get_counts()

            # Firing probability = P(measure |1⟩)
            spike_prob = counts.get("1", 0) / self.shots
            spike_probs[j] = spike_prob

            # Update refractory state if fired (using threshold for discrete spike decision)
            if spike_prob > self.threshold:
                refractory_state[j] = self.refractory_period

        return spike_probs, refractory_state

    def _execute_qlif_layer_differentiable(
        self,
        pre_spikes: torch.Tensor,
        weights: torch.Tensor,
        theta_membrane: torch.Tensor,
        refractory_state: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Execute a QLIF layer with surrogate gradient support.

        Same quantum circuit execution as _execute_qlif_layer(), but wraps
        the result in QLIFSurrogateSpike so that gradients flow back through
        the weight matrix via the sigmoid surrogate.

        Parameters
        ----------
        pre_spikes : torch.Tensor
            Spike probabilities from presynaptic layer, shape (num_pre,).
        weights : torch.Tensor
            Weight matrix with requires_grad=True, shape (num_pre, num_post).
        theta_membrane : torch.Tensor
            Membrane potential parameters with requires_grad=True, shape (num_post,).
        refractory_state : np.ndarray
            Refractory countdown per neuron, shape (num_post,).

        Returns
        -------
        tuple[torch.Tensor, np.ndarray]
            (spike_probs_tensor, updated_refractory_state) — spike probs
            are torch tensors with grad_fn for backprop.
        """
        num_post = weights.shape[1]
        spike_probs_list: list[torch.Tensor] = []
        backend = self._get_backend()

        for j in range(num_post):
            if refractory_state[j] > 0:
                refractory_state[j] -= 1
                # Zero spike with gradient connection through weights
                spike_probs_list.append(
                    torch.zeros(1, device=self.device, dtype=torch.float32).squeeze(),
                )
                continue

            # Differentiable weighted input (in autograd graph)
            weighted_input = torch.dot(pre_spikes, weights[:, j])

            # Differentiable RY angle: theta + tanh(w·x) * pi
            # Both theta_membrane[j] and weighted_input are in the autograd graph,
            # so gradients flow to both weights AND theta parameters.
            ry_angle = theta_membrane[j] + torch.tanh(weighted_input) * torch.pi

            # Execute quantum circuit for forward spike probability (detached)
            wi_np = float(weighted_input.detach().cpu())
            qc = self._build_qlif_circuit(wi_np, float(theta_membrane[j].detach().cpu()))
            job = backend.run(qc, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            quantum_spike_prob = counts.get("1", 0) / self.shots

            # Wrap in surrogate gradient function — gradient flows through ry_angle
            # to both theta and weights
            spike_prob: torch.Tensor = QLIFSurrogateSpike.apply(  # type: ignore[assignment]
                ry_angle,
                quantum_spike_prob,
                DEFAULT_SURROGATE_ALPHA,
            )
            spike_probs_list.append(spike_prob)

            # Update refractory state
            if quantum_spike_prob > self.threshold:
                refractory_state[j] = self.refractory_period

        spike_probs = torch.stack(spike_probs_list)
        return spike_probs, refractory_state

    def _encode_sensory_spikes(self, features: np.ndarray) -> np.ndarray:
        """Encode sensory features as spike probabilities.

        Uses sigmoid function with scaling to convert continuous features
        to spike probabilities in [0, 1].

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.

        Returns
        -------
        np.ndarray
            Spike probabilities for sensory neurons.
        """
        # Map features to sensory neurons
        # If we have more neurons than features, replicate/interpolate
        num_features = len(features)
        sensory_spikes = np.zeros(self.num_sensory)

        for i in range(self.num_sensory):
            # Cycle through features if we have more neurons than features
            feature_idx = i % num_features
            feature_val = features[feature_idx]

            # Sigmoid with scaling: sigmoid(feature * 5.0)
            # This maps [0,1] inputs to roughly [0.5, 0.99] probabilities
            sensory_spikes[i] = 1.0 / (1.0 + np.exp(-feature_val * 5.0))

        return sensory_spikes

    def _timestep(self, features: np.ndarray) -> np.ndarray:
        """Execute one timestep of QSNN dynamics.

        Propagates spikes through sensory→hidden→motor layers.
        Stores spike data for deferred eligibility accumulation
        (called after action selection in run_brain).

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.

        Returns
        -------
        np.ndarray
            Motor neuron firing probabilities.
        """
        # Encode sensory inputs
        sensory_spikes = self._encode_sensory_spikes(features)

        # Sensory → Hidden layer
        hidden_spikes, self.refractory_hidden = self._execute_qlif_layer(
            sensory_spikes,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
        )

        # Hidden → Motor layer
        motor_spikes, self.refractory_motor = self._execute_qlif_layer(
            hidden_spikes,
            self.W_hm,
            self.theta_motor,
            self.refractory_motor,
        )

        # Store spike data for deferred eligibility accumulation.
        # Eligibility is accumulated after action selection so we can
        # apply action-specific credit assignment to W_hm.
        self._last_sensory_spikes = sensory_spikes
        self._last_hidden_spikes = hidden_spikes
        self._last_motor_spikes = motor_spikes

        return motor_spikes

    def _timestep_differentiable(self, features: np.ndarray) -> torch.Tensor:
        """Execute one timestep with gradient tracking for surrogate gradient mode.

        Same QLIF dynamics as _timestep(), but uses _execute_qlif_layer_differentiable()
        so that motor spike probabilities are torch tensors with grad_fn, enabling
        backpropagation through the weight matrices via surrogate gradients.

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.

        Returns
        -------
        torch.Tensor
            Motor neuron firing probabilities (with grad_fn for backprop).
        """
        sensory_spikes = self._encode_sensory_spikes(features)

        # Convert to tensor for differentiable computation
        sensory_tensor = torch.tensor(
            sensory_spikes,
            dtype=torch.float32,
            device=self.device,
        )

        # Sensory → Hidden (differentiable)
        hidden_spikes, self.refractory_hidden = self._execute_qlif_layer_differentiable(
            sensory_tensor,
            self.W_sh,
            self.theta_hidden,
            self.refractory_hidden,
        )

        # Hidden → Motor (differentiable)
        motor_spikes, self.refractory_motor = self._execute_qlif_layer_differentiable(
            hidden_spikes,
            self.W_hm,
            self.theta_motor,
            self.refractory_motor,
        )

        return motor_spikes

    def _multi_timestep(self, features: np.ndarray) -> np.ndarray:
        """Execute multiple QLIF timesteps and average motor spike probabilities.

        Runs num_integration_steps timesteps with the same input, averaging
        motor spike probabilities across steps. This reduces quantum shot noise
        variance by a factor of num_integration_steps, giving REINFORCE cleaner
        gradient signal.

        Refractory state accumulates across timesteps within a single decision,
        providing natural temporal dynamics.

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.

        Returns
        -------
        np.ndarray
            Averaged motor neuron firing probabilities.
        """
        motor_accumulator = np.zeros(self.num_motor)
        for _ in range(self.num_integration_steps):
            motor_spikes = self._timestep(features)
            motor_accumulator += motor_spikes
        return motor_accumulator / self.num_integration_steps

    def _multi_timestep_differentiable(self, features: np.ndarray) -> torch.Tensor:
        """Execute multiple QLIF timesteps with gradient tracking and average.

        Same as _multi_timestep() but uses _timestep_differentiable() so that
        gradients flow through the averaged motor spike probabilities back to
        weights and theta parameters.

        Parameters
        ----------
        features : np.ndarray
            Input features from preprocessing.

        Returns
        -------
        torch.Tensor
            Averaged motor neuron firing probabilities (with grad_fn).
        """
        motor_accumulator = torch.zeros(self.num_motor, device=self.device)
        for _ in range(self.num_integration_steps):
            motor_spikes = self._timestep_differentiable(features)
            motor_accumulator = motor_accumulator + motor_spikes
        return motor_accumulator / self.num_integration_steps

    def _accumulate_eligibility(
        self,
        sensory_spikes: np.ndarray,
        hidden_spikes: np.ndarray,
        motor_spikes: np.ndarray,
        chosen_action: int,
    ) -> None:
        """Accumulate eligibility traces for local learning.

        Each step, existing traces decay by gamma before new correlations
        are added. Spike probabilities are centered at the firing threshold
        before computing outer products, producing mixed-sign eligibility
        traces that enable directional weight updates (neurons firing above
        threshold get positive eligibility, below get negative).

        **Action-specific credit assignment**: Only the chosen action's
        column in eligibility_hm is updated each step. This prevents all
        columns from receiving identical correlated updates (which caused
        column direction collapse in R2-R10). Each motor neuron column
        accumulates eligibility only when its action is selected, giving
        each column independent training signal over time.

        Parameters
        ----------
        sensory_spikes : np.ndarray
            Sensory layer spike probabilities.
        hidden_spikes : np.ndarray
            Hidden layer spike probabilities.
        motor_spikes : np.ndarray
            Motor layer spike probabilities.
        chosen_action : int
            Index of the action selected this step.
        """
        # Decay existing traces so recent steps matter more
        self.eligibility_sh *= self.gamma
        self.eligibility_hm *= self.gamma
        self.eligibility_theta_hidden *= self.gamma
        self.eligibility_theta_motor *= self.gamma

        # Center spike probabilities at firing threshold before computing
        # outer products. Raw spikes are in [0, 1] and always positive,
        # which produces all-positive eligibility traces — meaning every
        # weight gets pushed in the same direction. Centering at threshold
        # gives values in [-0.5, +0.5], so neurons firing above threshold
        # contribute positive eligibility and below contribute negative.
        # This enables the learning rule to develop directional structure
        # in W_sh (e.g., strengthen connections from active sensory neurons
        # to responsive hidden neurons while weakening others).
        pre_sh = torch.tensor(
            sensory_spikes - self.threshold,
            dtype=torch.float32,
            device=self.device,
        )
        post_sh = torch.tensor(
            hidden_spikes - self.threshold,
            dtype=torch.float32,
            device=self.device,
        )
        self.eligibility_sh += torch.outer(pre_sh, post_sh)

        # Action-specific eligibility for W_hm: only update the chosen
        # action's column. This is the key difference from standard Hebbian
        # learning — in REINFORCE, only the chosen action's parameters are
        # updated. By restricting eligibility to the chosen column, each
        # motor neuron develops independent weight structure based on the
        # outcomes when IT was selected, preventing column convergence.
        pre_hm = torch.tensor(
            hidden_spikes - self.threshold,
            dtype=torch.float32,
            device=self.device,
        )
        motor_centered = float(motor_spikes[chosen_action] - self.threshold)
        self.eligibility_hm[:, chosen_action] += pre_hm * motor_centered

        # Theta eligibility: W_sh uses all hidden neurons, W_hm only chosen action
        self.eligibility_theta_hidden += post_sh
        self.eligibility_theta_motor[chosen_action] += motor_centered

    @staticmethod
    def _normalize_trace(
        trace: torch.Tensor,
        max_norm: float = MAX_ELIGIBILITY_NORM,
    ) -> torch.Tensor:
        """Normalize an eligibility trace to cap its Frobenius/L2 norm.

        Preserves direction while bounding magnitude so that weight deltas
        stay proportional to lr * max_norm * advantage regardless of how
        large the raw trace accumulated during the update interval.

        Parameters
        ----------
        trace : torch.Tensor
            Raw eligibility trace (matrix or vector).
        max_norm : float
            Maximum allowed norm.

        Returns
        -------
        torch.Tensor
            Normalized trace (same shape).
        """
        norm = torch.norm(trace)
        if norm > max_norm:
            return trace * (max_norm / norm)
        return trace

    def _local_learning_update(self, total_reward: float) -> None:
        """Apply local learning weight update with baseline advantage.

        dw = lr * normalize(eligibility) * advantage - weight_decay * w

        Eligibility traces are normalized before computing deltas to bound
        the maximum weight update magnitude. This prevents weights from
        blowing past tanh's useful range regardless of trace accumulation.

        Parameters
        ----------
        total_reward : float
            Total discounted reward for the episode.
        """
        # Compute advantage relative to baseline
        advantage = total_reward - self.baseline

        # Update baseline (exponential moving average)
        self.baseline = (
            1 - self.baseline_alpha
        ) * self.baseline + self.baseline_alpha * total_reward

        # Log raw eligibility norms before normalization
        raw_sh_norm = torch.norm(self.eligibility_sh).item()
        raw_hm_norm = torch.norm(self.eligibility_hm).item()

        # Normalize eligibility traces — caps update magnitude while preserving direction
        norm_elig_sh = self._normalize_trace(self.eligibility_sh)
        norm_elig_hm = self._normalize_trace(self.eligibility_hm)
        norm_elig_theta_h = self._normalize_trace(self.eligibility_theta_hidden)
        norm_elig_theta_m = self._normalize_trace(self.eligibility_theta_motor)

        # Compute weight updates using normalized traces
        delta_sh = self.learning_rate * norm_elig_sh * advantage
        delta_hm = self.learning_rate * norm_elig_hm * advantage

        # Compute theta updates
        delta_theta_hidden = self.learning_rate * norm_elig_theta_h * advantage
        delta_theta_motor = self.learning_rate * norm_elig_theta_m * advantage

        # Apply updates with L2 weight decay for soft stability
        self.W_sh += delta_sh - WEIGHT_DECAY_FACTOR * self.W_sh
        self.W_hm += delta_hm - WEIGHT_DECAY_FACTOR * self.W_hm

        # Update theta parameters (membrane potential biases)
        self.theta_hidden += delta_theta_hidden - WEIGHT_DECAY_FACTOR * self.theta_hidden
        self.theta_motor += delta_theta_motor - WEIGHT_DECAY_FACTOR * self.theta_motor

        # Log update statistics
        col_diffs = self._compute_column_diversity()
        logger.debug(
            f"QSNN local learning: reward={total_reward:.4f}, "
            f"advantage={advantage:.4f}, baseline={self.baseline:.4f}, "
            f"raw_elig_sh_norm={raw_sh_norm:.4f}, "
            f"raw_elig_hm_norm={raw_hm_norm:.4f}, "
            f"norm_elig_sh_norm={torch.norm(norm_elig_sh).item():.4f}, "
            f"norm_elig_hm_norm={torch.norm(norm_elig_hm).item():.4f}, "
            f"delta_sh_norm={torch.norm(delta_sh).item():.4f}, "
            f"delta_hm_norm={torch.norm(delta_hm).item():.4f}, "
            f"theta_hidden_norm={torch.norm(self.theta_hidden).item():.4f}, "
            f"theta_motor_norm={torch.norm(self.theta_motor).item():.4f}, "
            f"W_hm_col_diversity={col_diffs:.4f}",
        )

    def _last_entropy(self) -> float:
        """Compute entropy of the most recent action probability distribution.

        Returns ln(num_actions) if no probabilities are stored (maximum entropy).
        """
        if self.current_probabilities is None:
            return float(np.log(self.num_actions))
        probs = np.clip(self.current_probabilities, 1e-10, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def _compute_column_diversity(self) -> float:
        """Compute mean pairwise L2 distance between W_hm columns.

        Returns 0.0 when all columns are identical (collapsed state).
        """
        cols = self.W_hm.T  # shape: (num_motor, num_hidden)
        n = cols.shape[0]
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += torch.norm(cols[i] - cols[j]).item()
                count += 1
        return total_dist / max(count, 1)

    def _exploration_schedule(self) -> tuple[float, float]:
        """Compute current exploration parameters based on episode count.

        Linearly decays epsilon and temperature over EXPLORATION_DECAY_EPISODES
        episodes, allowing more exploitation as the policy matures.

        Returns
        -------
        tuple[float, float]
            (current_epsilon, current_temperature)
        """
        progress = min(1.0, self._episode_count / max(EXPLORATION_DECAY_EPISODES, 1))
        # Epsilon decays from 0.1 to 0.03 (still ensures min 0.75% per action)
        current_epsilon = EXPLORATION_EPSILON * (1.0 - progress * 0.7)
        # Temperature decays from 1.5 to 1.0 (sharper softmax over time)
        current_temperature = 1.5 - 0.5 * progress
        return current_epsilon, current_temperature

    def _current_lr(self) -> float:
        """Get the current learning rate from the optimizer (or config default)."""
        if self.optimizer is not None:
            return float(self.optimizer.param_groups[0]["lr"])
        return self.learning_rate

    def _reinforce_update(self) -> None:
        """REINFORCE policy gradient update with surrogate gradients.

        Recomputes forward passes with gradient tracking enabled, computes
        policy loss from log-probabilities and advantages, and backpropagates
        through the QLIFSurrogateSpike function to update weights.

        This requires episode_features to be stored during run_brain().
        """
        if len(self.episode_rewards) == 0 or self.optimizer is None:
            return

        num_steps = len(self.episode_rewards)

        # Compute discounted returns (reward-to-go)
        returns: list[float] = []
        discounted_return = 0.0
        for reward in reversed(self.episode_rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)

        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        raw_mean = returns_tensor.mean().item()

        # Update baseline (exponential moving average)
        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * raw_mean

        # Normalize advantages for variance reduction
        if len(returns_tensor) > 1:
            advantages = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        else:
            advantages = returns_tensor - self.baseline

        # Clip advantages to prevent outlier returns from corrupting policy
        advantages = torch.clamp(advantages, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

        # Recompute forward passes WITH gradient tracking to build computation graph.
        # Reset refractory state for clean recomputation.
        self.refractory_hidden.fill(0)
        self.refractory_motor.fill(0)

        log_probs_list: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        for t in range(num_steps):
            features = self.episode_features[t]
            action_idx = self.episode_actions[t]

            # Forward pass with gradient tracking (surrogate gradient enabled).
            # Uses multi-timestep integration to average out quantum shot noise.
            motor_spikes = self._multi_timestep_differentiable(features)

            # Convert spike probabilities to action logits
            motor_clipped = torch.clamp(motor_spikes, 1e-8, 1.0 - 1e-8)
            logits = (motor_clipped - 0.5) * LOGIT_SCALE

            # Softmax with temperature + epsilon floor (same as run_brain)
            epsilon, temperature = self._exploration_schedule()
            softmax_probs = torch.softmax(logits / temperature, dim=-1)
            uniform = torch.ones_like(softmax_probs) / self.num_actions
            action_probs = (1 - epsilon) * softmax_probs + epsilon * uniform

            # Log probability of chosen action
            log_prob = torch.log(action_probs[action_idx] + 1e-8)
            log_probs_list.append(log_prob)

            # Entropy for regularization
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
            entropies.append(entropy)

        # Compute policy loss: REINFORCE with adaptive entropy bonus.
        # When entropy drops below ENTROPY_FLOOR, scale entropy_coef up
        # proportionally to push the policy back toward exploration. This
        # rescues sessions from the entropy collapse death spiral (R12h
        # analysis: 50% of sessions collapse and never recover).
        log_probs = torch.stack(log_probs_list)
        mean_entropy = torch.stack(entropies).mean()
        entropy_val = mean_entropy.item()
        if entropy_val < ENTROPY_FLOOR:
            # Linear ramp: boost from 1.0x at floor to ENTROPY_BOOST_MAX x at 0
            ratio = 1.0 - entropy_val / ENTROPY_FLOOR
            entropy_scale = 1.0 + ratio * (ENTROPY_BOOST_MAX - 1.0)
        else:
            entropy_scale = 1.0
        effective_entropy_coef = self.entropy_coef * entropy_scale
        policy_loss = -(log_probs * advantages).mean() - effective_entropy_coef * mean_entropy

        if entropy_scale > 1.0:
            logger.debug(
                f"QSNN adaptive entropy: entropy={entropy_val:.4f} < floor={ENTROPY_FLOOR}, "
                f"scale={entropy_scale:.2f}x, effective_coef={effective_entropy_coef:.4f}",
            )

        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()

        # Clip gradients, step optimizer, clamp weights, and log
        self._apply_gradients_and_log(
            policy_loss,
            mean_entropy,
            advantages,
            raw_mean,
            returns_tensor,
        )

    def _apply_gradients_and_log(
        self,
        policy_loss: torch.Tensor,
        mean_entropy: torch.Tensor,
        advantages: torch.Tensor,
        raw_mean: float,
        returns_tensor: torch.Tensor,
    ) -> None:
        """Clip gradients, step optimizer, clamp weights, and log diagnostics."""
        if self.optimizer is None:
            return
        params_list = [self.W_sh, self.W_hm, self.theta_hidden, self.theta_motor]
        grad_norm = torch.nn.utils.clip_grad_norm_(params_list, max_norm=SURROGATE_GRAD_CLIP)

        # Per-parameter gradient norms (after clipping)
        grad_norms = [p.grad.norm().item() if p.grad is not None else 0.0 for p in params_list]

        w_sh_pre = torch.norm(self.W_sh).item()
        w_hm_pre = torch.norm(self.W_hm).item()

        self.optimizer.step()

        with torch.no_grad():
            for p in params_list:
                p.clamp_(-self.weight_clip, self.weight_clip)

        w_sh_post = torch.norm(self.W_sh).item()
        w_hm_post = torch.norm(self.W_hm).item()
        col_div = self._compute_column_diversity()
        epsilon, temperature = self._exploration_schedule()

        logger.debug(
            f"QSNN surrogate gradient update: loss={policy_loss.item():.4f}, "
            f"mean_entropy={mean_entropy.item():.4f}, "
            f"W_sh_norm={w_sh_post:.4f}, W_hm_norm={w_hm_post:.4f}, "
            f"W_sh_delta={w_sh_post - w_sh_pre:.6f}, "
            f"W_hm_delta={w_hm_post - w_hm_pre:.6f}, "
            f"W_hm_col_div={col_div:.4f}, "
            f"theta_h_norm={torch.norm(self.theta_hidden).item():.4f}, "
            f"theta_m_norm={torch.norm(self.theta_motor).item():.4f}, "
            f"grad_total={grad_norm.item():.4f}, "
            f"grad_sh={grad_norms[0]:.4f}, grad_hm={grad_norms[1]:.4f}, "
            f"grad_th={grad_norms[2]:.4f}, grad_tm={grad_norms[3]:.4f}, "
            f"returns_mean={raw_mean:.4f}, returns_std={returns_tensor.std().item():.4f}, "
            f"adv_mean={advantages.mean().item():.4f}, "
            f"adv_std={advantages.std().item():.4f}, "
            f"epsilon={epsilon:.4f}, temperature={temperature:.4f}, "
            f"lr={self.optimizer.param_groups[0]['lr']:.6f}, "
            f"episode={self._episode_count}",
        )

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Preprocess BrainParams to extract features.

        Two modes:
        1. **Unified sensory mode** (when sensory_modules is set)
        2. **Legacy mode** (default): gradient strength + relative angle

        Parameters
        ----------
        params : BrainParams
            Brain parameters containing sensory information.

        Returns
        -------
        np.ndarray
            Preprocessed features.
        """
        # Unified sensory mode
        if self.sensory_modules is not None:
            return extract_classical_features(params, self.sensory_modules)

        # Legacy mode: 2-feature preprocessing
        grad_strength = float(params.gradient_strength or 0.0)

        grad_direction = float(params.gradient_direction or 0.0)
        direction_map = {
            Direction.UP: np.pi / 2,
            Direction.DOWN: -np.pi / 2,
            Direction.LEFT: np.pi,
            Direction.RIGHT: 0.0,
        }
        agent_facing_angle = direction_map.get(params.agent_direction or Direction.UP, np.pi / 2)
        relative_angle = (grad_direction - agent_facing_angle + np.pi) % (2 * np.pi) - np.pi
        rel_angle_norm = relative_angle / np.pi

        return np.array([grad_strength, rel_angle_norm], dtype=np.float32)

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run the QSNN brain and select an action.

        Parameters
        ----------
        params : BrainParams
            Brain parameters containing sensory information.
        reward : float | None
            Current reward (unused).
        input_data : list[float] | None
            Optional input data (unused).
        top_only : bool
            Whether to return only top action (unused).
        top_randomize : bool
            Whether to randomize selection (unused).

        Returns
        -------
        list[ActionData]
            List containing the selected action.
        """
        # Preprocess inputs
        features = self.preprocess(params)

        # Execute QSNN timestep(s) to get motor firing probabilities.
        # Multi-timestep integration averages out quantum shot noise.
        motor_probs = self._multi_timestep(features)

        # Convert spike probabilities [0,1] to action logits
        # Shift from [0,1] to centered range, then scale to create spread
        motor_probs = np.clip(motor_probs, 1e-8, 1.0 - 1e-8)
        logits = (motor_probs - 0.5) * LOGIT_SCALE

        # Exploration schedule: epsilon and temperature decay over episodes
        epsilon, temperature = self._exploration_schedule()

        # Softmax with temperature scaling
        scaled_logits = logits / temperature
        exp_probs = np.exp(scaled_logits - np.max(scaled_logits))
        softmax_probs = exp_probs / np.sum(exp_probs)

        # Exploration floor: mix with uniform to prevent policy collapse.
        # Guarantees every action has at least epsilon/num_actions probability,
        # breaking the positive feedback loop where one dominant action
        # accumulates all eligibility and locks the policy.
        uniform = np.ones(self.num_actions) / self.num_actions
        action_probs = (1 - epsilon) * softmax_probs + epsilon * uniform

        # Compute action entropy for collapse detection
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

        # Log motor dynamics periodically for debugging
        if self._step_count % 50 == 0:
            col_div = self._compute_column_diversity()
            # Compute weighted_input stats to verify tanh stays in useful range
            sensory = self._encode_sensory_spikes(features)
            w_sh_np = self.W_sh.detach().cpu().numpy()
            w_hm_np = self.W_hm.detach().cpu().numpy()
            wi_sh = sensory @ w_sh_np  # weighted inputs to hidden layer
            # Estimate motor weighted inputs using max column sum (worst case)
            wi_hm_max = float(np.abs(w_hm_np).sum(axis=0).max())
            logger.debug(
                f"QSNN step {self._step_count}: "
                f"features={features}, "
                f"motor_spikes={np.array2string(motor_probs, precision=3)}, "
                f"logits={np.array2string(logits, precision=2)}, "
                f"action_probs={np.array2string(action_probs, precision=3)}, "
                f"entropy={action_entropy:.3f}, "
                f"W_sh_norm={torch.norm(self.W_sh).item():.3f}, "
                f"W_hm_norm={torch.norm(self.W_hm).item():.3f}, "
                f"W_hm_col_div={col_div:.3f}, "
                f"wi_sh_range=[{wi_sh.min():.2f},{wi_sh.max():.2f}], "
                f"wi_hm_max_possible={wi_hm_max:.2f}, "
                f"theta_h_norm={torch.norm(self.theta_hidden).item():.3f}, "
                f"theta_m_norm={torch.norm(self.theta_motor).item():.3f}",
            )

        # Sample action from categorical distribution
        action_idx = self.rng.choice(self.num_actions, p=action_probs)
        action_name = self.action_set[action_idx]

        # Update tracking data
        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=action_probs[action_idx],
        )

        # Accumulate eligibility traces with action-specific credit assignment.
        # Must happen after action selection so we know which W_hm column to update.
        if self.use_local_learning and self.training:
            self._accumulate_eligibility(
                self._last_sensory_spikes,
                self._last_hidden_spikes,
                self._last_motor_spikes,
                action_idx,
            )

        # Store features for surrogate gradient recomputation
        if not self.use_local_learning:
            self.episode_features.append(features)

        # Store for learning
        self.episode_actions.append(action_idx)

        self.current_probabilities = action_probs
        self.latest_data.probability = float(action_probs[action_idx])
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(action_probs[action_idx]))

        return [self.latest_data.action]

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Accumulate rewards and trigger learning at episode end.

        Parameters
        ----------
        params : BrainParams
            Brain parameters (unused).
        reward : float
            Reward for the current step.
        episode_done : bool
            Whether the episode has ended.
        """
        self.episode_rewards.append(reward)
        self.history_data.rewards.append(reward)
        self._step_count += 1

        # Intra-episode updates: apply learning every update_interval steps
        if (
            self.use_local_learning
            and self.update_interval > 0
            and self._step_count % self.update_interval == 0
            and not episode_done
        ):
            recent_reward = sum(self.episode_rewards[-self.update_interval :])
            self._local_learning_update(recent_reward)
            # Reset eligibility after applying update (fresh accumulation)
            self.eligibility_sh.zero_()
            self.eligibility_hm.zero_()
            self.eligibility_theta_hidden.zero_()
            self.eligibility_theta_motor.zero_()

        if episode_done and len(self.episode_rewards) > 0:
            # Compute total discounted reward for final update
            total_reward = 0.0
            discount = 1.0
            for r in self.episode_rewards:
                total_reward += discount * r
                discount *= self.gamma

            epsilon, temperature = self._exploration_schedule()
            col_div = self._compute_column_diversity()
            logger.info(
                f"QSNN episode complete: episode={self._episode_count}, "
                f"steps={len(self.episode_rewards)}, "
                f"total_reward={total_reward:.4f}, "
                f"W_sh_norm={torch.norm(self.W_sh).item():.4f}, "
                f"W_hm_norm={torch.norm(self.W_hm).item():.4f}, "
                f"W_hm_col_div={col_div:.4f}, "
                f"theta_h_norm={torch.norm(self.theta_hidden).item():.4f}, "
                f"theta_m_norm={torch.norm(self.theta_motor).item():.4f}, "
                f"entropy={self._last_entropy():.4f}, "
                f"epsilon={epsilon:.4f}, temperature={temperature:.4f}, "
                f"lr={self._current_lr():.6f}",
            )

            # Apply final learning update
            if self.use_local_learning:
                self._local_learning_update(total_reward)
            else:
                self._reinforce_update()

            # Increment episode counter (for exploration decay)
            self._episode_count += 1

            # Step LR scheduler (cosine annealing decay)
            if self.scheduler is not None:
                self.scheduler.step()

            # Reset episode state
            self._reset_episode()

    def update_memory(self, reward: float | None = None) -> None:
        """Update internal memory (no-op for QSNNBrain)."""

    def prepare_episode(self) -> None:
        """Prepare for a new episode."""
        self._reset_episode()

    def post_process_episode(self, *, episode_success: bool | None = None) -> None:
        """Post-process the episode (no-op, learning happens in learn())."""

    def _reset_episode(self) -> None:
        """Reset episode state."""
        self.episode_rewards.clear()
        self.episode_actions.clear()
        self.episode_features.clear()

        # Reset eligibility traces
        self.eligibility_sh.zero_()
        self.eligibility_hm.zero_()
        self.eligibility_theta_hidden.zero_()
        self.eligibility_theta_motor.zero_()

        # Reset refractory states
        self.refractory_hidden.fill(0)
        self.refractory_motor.fill(0)

        # Reset step counter
        self._step_count = 0

    def copy(self) -> QSNNBrain:
        """Create an independent copy of the QSNNBrain.

        Returns
        -------
        QSNNBrain
            Independent copy of this brain.
        """
        config_copy = QSNNBrainConfig(
            **{**self.config.model_dump(), "seed": self.seed},
        )

        new_brain = QSNNBrain(
            config=config_copy,
            num_actions=self.num_actions,
            device=DeviceType(self.device.type),
            action_set=self._action_set,
        )

        # Copy weights (independent copy, preserving requires_grad)
        new_brain.W_sh = self.W_sh.clone().detach().requires_grad_(self.W_sh.requires_grad)
        new_brain.W_hm = self.W_hm.clone().detach().requires_grad_(self.W_hm.requires_grad)
        new_brain.theta_hidden = (
            self.theta_hidden.clone().detach().requires_grad_(self.theta_hidden.requires_grad)
        )
        new_brain.theta_motor = (
            self.theta_motor.clone().detach().requires_grad_(self.theta_motor.requires_grad)
        )

        # Recreate optimizer and scheduler for surrogate gradient mode with new parameter refs
        if not self.use_local_learning:
            new_brain.optimizer = torch.optim.Adam(
                [new_brain.W_sh, new_brain.W_hm, new_brain.theta_hidden, new_brain.theta_motor],
                lr=self.learning_rate,
            )
            new_brain.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                new_brain.optimizer,
                T_max=LR_DECAY_EPISODES,
                eta_min=self.learning_rate * LR_MIN_FACTOR,
            )
            # Advance scheduler to match current episode count
            for _ in range(self._episode_count):
                new_brain.scheduler.step()

        # Copy learning state
        new_brain.baseline = self.baseline
        new_brain._episode_count = self._episode_count

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
