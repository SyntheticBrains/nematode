"""
Quantum Spiking Neural Network (QSNN) Brain Architecture.

This architecture implements Quantum Leaky Integrate-and-Fire (QLIF) neurons with
trainable synaptic weights and local learning rules. QSNN addresses two key problems:

1. QRC's representation issue: Uses trainable quantum parameters (vs fixed reservoir)
2. QVarCircuitBrain's barren plateaus: Uses local learning rules (vs global backprop)

Key Features:
- **QLIF Neurons**: 2-gate circuit: |0> -> RY(theta + input) -> RX(theta_leak) -> Measure
- **Local Learning**: 3-factor Hebbian rule: dw = lr * pre_spike * post_spike * reward
- **Layered Architecture**: sensory -> hidden -> motor layers with trainable connections
- **Eligibility Traces**: Accumulate spike correlations for delayed reward

Architecture:
- Input: Sensory features encoded as spike probabilities via sigmoid
- Hidden: QLIF interneuron layer for feature processing
- Output: Motor neurons corresponding to actions

The QSNN brain learns by:
1. Encoding sensory inputs as spike probabilities
2. Propagating spikes through QLIF circuits layer by layer
3. Computing eligibility traces from pre/post spike correlations
4. Applying reward-modulated weight updates at episode end

References
----------
- Brand & Petruccione (2024). "A quantum leaky integrate-and-fire spiking neuron and network"
"""

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
DEFAULT_NUM_HIDDEN_NEURONS = 4
DEFAULT_NUM_MOTOR_NEURONS = 4
DEFAULT_MEMBRANE_TAU = 0.9
DEFAULT_THRESHOLD = 0.5
DEFAULT_REFRACTORY_PERIOD = 2
DEFAULT_USE_LOCAL_LEARNING = True
DEFAULT_SHOTS = 1024
DEFAULT_GAMMA = 0.99
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_WEIGHT_CLIP = 5.0

# Validation constants
MIN_SENSORY_NEURONS = 1
MIN_HIDDEN_NEURONS = 1
MIN_MOTOR_NEURONS = 2
MIN_SHOTS = 100


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

        # Compute leak angle: theta_leak = (1 - membrane_tau) * pi
        self.leak_angle = (1 - self.membrane_tau) * np.pi

        # Initialize network components
        self._init_network_weights()
        self._init_episode_state()

        logger.info(
            f"QSNNBrain initialized: {self.num_sensory}->{self.num_hidden}->{self.num_motor} "
            f"neurons, membrane_tau={self.membrane_tau}, threshold={self.threshold}, "
            f"use_local_learning={self.use_local_learning}",
        )

    def _init_network_weights(self) -> None:
        """Initialize trainable weight matrices and neuron parameters."""
        # Xavier/Glorot initialization for better gradient flow
        self.W_sh = torch.nn.init.xavier_uniform_(
            torch.empty(self.num_sensory, self.num_hidden, device=self.device),
        )
        self.W_hm = torch.nn.init.xavier_uniform_(
            torch.empty(self.num_hidden, self.num_motor, device=self.device),
        )

        # Trainable membrane potential parameters per neuron
        self.theta_hidden = torch.zeros(self.num_hidden, device=self.device)
        self.theta_motor = torch.zeros(self.num_motor, device=self.device)

        # Eligibility traces for local learning (accumulated per episode)
        self.eligibility_sh = torch.zeros_like(self.W_sh)
        self.eligibility_hm = torch.zeros_like(self.W_hm)

        # Refractory state tracking
        self.refractory_hidden = np.zeros(self.num_hidden, dtype=np.int32)
        self.refractory_motor = np.zeros(self.num_motor, dtype=np.int32)

        # Qiskit backend for circuit execution
        self._backend = None

    def _init_episode_state(self) -> None:
        """Initialize episode tracking state."""
        self.episode_rewards: list[float] = []
        self.episode_actions: list[int] = []
        self.episode_log_probs: list[torch.Tensor] = []
        self.episode_probs: list[torch.Tensor] = []

        # REINFORCE baseline (for fallback mode)
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
        ry_angle = float(theta_membrane + weighted_input * np.pi)
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

        # Accumulate eligibility traces (local learning)
        if self.use_local_learning and self.training:
            self._accumulate_eligibility(sensory_spikes, hidden_spikes, motor_spikes)

        return motor_spikes

    def _accumulate_eligibility(
        self,
        sensory_spikes: np.ndarray,
        hidden_spikes: np.ndarray,
        motor_spikes: np.ndarray,
    ) -> None:
        """Accumulate eligibility traces for local learning.

        Eligibility = pre_spike * post_spike (correlation)

        Parameters
        ----------
        sensory_spikes : np.ndarray
            Sensory layer spike probabilities.
        hidden_spikes : np.ndarray
            Hidden layer spike probabilities.
        motor_spikes : np.ndarray
            Motor layer spike probabilities.
        """
        # Sensory → Hidden eligibility
        for i in range(self.num_sensory):
            for j in range(self.num_hidden):
                self.eligibility_sh[i, j] += sensory_spikes[i] * hidden_spikes[j]

        # Hidden → Motor eligibility
        for i in range(self.num_hidden):
            for j in range(self.num_motor):
                self.eligibility_hm[i, j] += hidden_spikes[i] * motor_spikes[j]

    def _local_learning_update(self, total_reward: float) -> None:
        """Apply local learning weight update.

        dw = lr * eligibility * reward

        Parameters
        ----------
        total_reward : float
            Total discounted reward for the episode.
        """
        # Compute weight updates
        delta_sh = self.learning_rate * self.eligibility_sh * total_reward
        delta_hm = self.learning_rate * self.eligibility_hm * total_reward

        # Apply updates
        self.W_sh += delta_sh
        self.W_hm += delta_hm

        # Weight clipping for stability
        if self.weight_clip > 0:
            self.W_sh = torch.clamp(self.W_sh, -self.weight_clip, self.weight_clip)
            self.W_hm = torch.clamp(self.W_hm, -self.weight_clip, self.weight_clip)

        # Log update statistics
        logger.debug(
            f"QSNN local learning: reward={total_reward:.4f}, "
            f"eligibility_sh_norm={torch.norm(self.eligibility_sh).item():.4f}, "
            f"eligibility_hm_norm={torch.norm(self.eligibility_hm).item():.4f}, "
            f"delta_sh_norm={torch.norm(delta_sh).item():.4f}, "
            f"delta_hm_norm={torch.norm(delta_hm).item():.4f}",
        )

    def _reinforce_update(self) -> None:
        """Fallback REINFORCE update for comparison (use_local_learning=False)."""
        if len(self.episode_rewards) == 0:
            return

        # Compute discounted returns
        returns = []
        discounted_return = 0.0
        for reward in reversed(self.episode_rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)

        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        raw_mean = returns_tensor.mean().item()

        # Update baseline
        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * raw_mean

        # Normalize returns
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        # Compute policy loss
        policy_loss = torch.tensor(0.0, device=self.device)
        for t in range(len(self.episode_log_probs)):
            policy_loss = policy_loss - self.episode_log_probs[t] * returns_tensor[t]

        # Would need gradient tracking on weights for this to work properly
        # For now, just use local learning as the primary method
        logger.warning("REINFORCE fallback not fully implemented - using local learning instead")

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

        # Execute QSNN timestep to get motor firing probabilities
        motor_probs = self._timestep(features)

        # Apply softmax to get action probabilities
        # Add small epsilon for numerical stability
        motor_probs = np.clip(motor_probs, 1e-8, 1.0 - 1e-8)
        exp_probs = np.exp(motor_probs * 5.0)  # Scale for sharper distribution
        action_probs = exp_probs / np.sum(exp_probs)

        # Convert to tensor for tracking
        probs_tensor = torch.tensor(action_probs, dtype=torch.float32, device=self.device)

        # Sample action from categorical distribution
        action_idx = self.rng.choice(self.num_actions, p=action_probs)
        action_name = self.action_set[action_idx]

        # Store log probability for REINFORCE fallback
        log_prob = torch.log(probs_tensor[action_idx] + 1e-8)

        # Update tracking data
        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=action_probs[action_idx],
        )

        # Store for learning
        self.episode_actions.append(action_idx)
        self.episode_log_probs.append(log_prob)
        self.episode_probs.append(probs_tensor)

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

        if episode_done and len(self.episode_rewards) > 0:
            # Compute total discounted reward
            total_reward = 0.0
            discount = 1.0
            for r in self.episode_rewards:
                total_reward += discount * r
                discount *= self.gamma

            logger.info(
                f"QSNN episode complete: {len(self.episode_rewards)} steps, "
                f"total_reward={total_reward:.4f}",
            )

            # Apply learning update
            if self.use_local_learning:
                self._local_learning_update(total_reward)
            else:
                self._reinforce_update()

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
        self.episode_log_probs.clear()
        self.episode_probs.clear()

        # Reset eligibility traces
        self.eligibility_sh.zero_()
        self.eligibility_hm.zero_()

        # Reset refractory states
        self.refractory_hidden.fill(0)
        self.refractory_motor.fill(0)

    def copy(self) -> "QSNNBrain":
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

        # Copy weights (independent copy)
        new_brain.W_sh = self.W_sh.clone()
        new_brain.W_hm = self.W_hm.clone()
        new_brain.theta_hidden = self.theta_hidden.clone()
        new_brain.theta_motor = self.theta_motor.clone()

        # Copy learning state
        new_brain.baseline = self.baseline

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
