"""
Modular Quantum Brain Architecture for Multi-Modal Sensing.

This quantum brain architecture implements a modular approach where different sensory and
cognitive functions are mapped to specific qubits, allowing for specialized processing
of different types of environmental information while maintaining quantum coherence
and entanglement between modules.

Key Features:
- **Modular Quantum Architecture**: Different modules (vision, chemotaxis, mechanosensation,
  memory, etc.) are mapped to specific qubits for specialized processing
- **Parameter-Shift Gradient Descent**: Uses quantum parameter-shift rule for gradient
  computation and parameter optimization
- **Trajectory Learning (Optional)**: Episode-level REINFORCE with discounted returns for
  temporal credit assignment (enabled via `use_trajectory_learning: true`)
- **Multi-Layer Quantum Circuits**: Configurable circuit depth with feature encoding
  and entanglement layers
- **Quantum Feature Encoding**: Environmental features encoded as RX/RY/RZ rotations
  on module-specific qubits
- **Cross-Module Entanglement**: CZ gates create entanglement between all qubit pairs
  for information sharing
- **Momentum-Based Optimization**: Uses momentum updates with L2 regularization for
  stable parameter learning
- **Adaptive Learning Rate**: Dynamic learning rate scheduling with optional boost
  mechanism for poor performance periods
- **Hardware-Agnostic Execution**: Supports both classical simulation (AerSimulator)
  and real quantum hardware (IBM QPU) with Q-CTRL performance management
- **Overfitting Detection**: Built-in monitoring for learning stability and
  generalization performance

Architecture:
- Input: Multi-modal environmental features (gradients, distances, orientations, etc.)
- Modules: Specialized qubit groups for chemotaxis, mechanosensation, memory, etc.
- Layers: Multiple encoding-entanglement layers with parameterized rotation gates
- Output: Action probabilities derived from quantum measurement statistics
- Learning: Parameter-shift rule gradients with momentum-based optimization

The modular brain learns by:
1. Encoding environmental features into module-specific qubit rotations
2. Applying parameterized quantum gates in multiple layers with entanglement
3. Measuring quantum states to obtain action probability distributions
4. Using parameter-shift rule to compute gradients of action log-probabilities
5. Updating quantum parameters using momentum-based gradient descent with regularization
6. Adapting learning rate based on performance and gradient characteristics

Trajectory Learning:
When `use_trajectory_learning: true`, the brain uses episode-level REINFORCE:
1. Buffers (params, actions, rewards) for each timestep during an episode
2. At episode end, computes discounted returns: G_t = r_t + gamma * G_{t+1}
3. Computes trajectory-level gradients: grad_i = sum_t 0.5 * (P_+ - P_-) * G_t
4. Updates parameters once per episode using accumulated gradients

Example Configuration:
```yaml
brain:
  name: modular
  config:
    use_trajectory_learning: true  # Enable trajectory learning (default: false)
    gamma: 0.99  # Discount factor for returns (default: 0.99)
    num_layers: 2
    modules:
      chemotaxis: [0, 1]
```

This architecture provides quantum advantages through superposition and entanglement
while maintaining interpretable modular structure for different sensory modalities.
Supports both noisy intermediate-scale quantum (NISQ) devices and classical simulation
for scalable quantum machine learning applications.
"""

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, QuantumBrain
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch.dtypes import DEFAULT_SHOTS, BrainConfig, DeviceType
from quantumnematode.brain.modules import (
    DEFAULT_MODULES,
    ModuleName,
    RotationAxis,
    count_total_qubits,
    extract_features_for_module,
)
from quantumnematode.errors import (
    ERROR_MISSING_IMPORT_QISKIT_AER,
    ERROR_MISSING_IMPORT_QISKIT_IBM_RUNTIME,
)
from quantumnematode.executors.ibm_job import monitor_job
from quantumnematode.initializers.manual_initializer import ManualParameterInitializer
from quantumnematode.initializers.random_initializer import (
    RandomPiUniformInitializer,
    RandomSmallUniformInitializer,
)
from quantumnematode.initializers.zero_initializer import ZeroInitializer
from quantumnematode.logging_config import logger
from quantumnematode.monitoring.overfitting_detector import create_overfitting_detector_for_brain
from quantumnematode.optimizers.gradient_methods import (
    DEFAULT_MAX_CLIP_GRADIENT,
    DEFAULT_MAX_GRADIENT_NORM,
    GradientCalculationMethod,
    compute_gradients,
)
from quantumnematode.optimizers.learning_rate import ConstantLearningRate, DynamicLearningRate

if TYPE_CHECKING:
    from qiskit.primitives import PrimitiveResult
    from qiskit.providers import BackendV2
    from qiskit_aer import AerSimulator

# Defaults
DEFAULT_L2_REG = 0.005
DEFAULT_LARGE_GRADIENT_THRESHOLD = 0.1
DEFAULT_MIN_GRADIENT_MAGNITUDE = 1e-4
DEFAULT_NOISE_STD = 0.005
DEFAULT_NUM_LAYERS = 2
DEFAULT_PARAM_CLIP = True
DEFAULT_PARAM_MODULO = True
DEFAULT_SIGNIFICANT_REWARD_THRESHOLD = 0.1
DEFAULT_SMALL_GRADIENT_THRESHOLD = 1e-4

# Trajectory learning defaults
DEFAULT_USE_TRAJECTORY_LEARNING = False
DEFAULT_GAMMA = 0.99

# Overfitting detector defaults
OVERFIT_DETECTOR_EPISODE_LOG_INTERVAL = 25

# Learning rate boost defaults
DEFAULT_LR_BOOST = True
DEFAULT_LR_BOOST_FACTOR = 5.0
DEFAULT_LR_BOOST_DURATION = 10
DEFAULT_LOW_REWARD_THRESHOLD = -0.25
DEFAULT_LOW_REWARD_WINDOW = 15

if TYPE_CHECKING:
    from qiskit_serverless.core.function import RunnableQiskitFunction


@dataclass
class EpisodeBuffer:
    """
    Buffer for storing episode trajectory data for trajectory learning.

    Stores parameters, actions, and rewards for each timestep in an episode,
    enabling discounted return computation and trajectory-level gradient updates.
    """

    params: list[BrainParams] = field(default_factory=list)
    actions: list[ActionData] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    parameter_values: list[dict[str, float]] = field(default_factory=list)

    def append(
        self,
        params: BrainParams,
        action: ActionData,
        reward: float,
        parameter_values: dict[str, float],
    ) -> None:
        """Append a timestep's data to the buffer."""
        self.params.append(params)
        self.actions.append(action)
        self.rewards.append(reward)
        self.parameter_values.append(parameter_values.copy())

    def clear(self) -> None:
        """Clear all buffered data."""
        self.params.clear()
        self.actions.clear()
        self.rewards.clear()
        self.parameter_values.clear()

    def __len__(self) -> int:
        """Return the number of timesteps in the buffer."""
        return len(self.rewards)


class ModularBrainConfig(BrainConfig):
    """Configuration for the ModularBrain architecture."""

    l2_reg: float = DEFAULT_L2_REG  # L2 regularization strength
    large_gradient_threshold: float = (
        DEFAULT_LARGE_GRADIENT_THRESHOLD  # Threshold for large gradients
    )
    min_gradient_magnitude: float = (
        DEFAULT_MIN_GRADIENT_MAGNITUDE  # Minimum gradient magnitude for updates
    )
    modules: dict[ModuleName, list[int]] = (
        DEFAULT_MODULES  # Mapping of module names to qubit indices
    )
    noise_std: float = DEFAULT_NOISE_STD  # Standard deviation for parameter noise
    num_layers: int = DEFAULT_NUM_LAYERS  # Number of layers in the quantum circuit
    param_clip: bool = DEFAULT_PARAM_CLIP  # Toggle parameter clipping
    param_modulo: bool = DEFAULT_PARAM_MODULO  # Toggle parameter modulo
    significant_reward_threshold: float = (
        DEFAULT_SIGNIFICANT_REWARD_THRESHOLD  # Threshold for significant rewards
    )

    # Momentum configuration
    momentum_decay: float = 0.99  # Decay factor for momentum updates
    momentum_coefficient: float = 0.9  # Coefficient for momentum updates

    # Overfitting detector configuration
    overfit_detector_episode_log_interval: int = (
        OVERFIT_DETECTOR_EPISODE_LOG_INTERVAL  # Interval for episode logging
    )

    # Learning rate boost configuration
    lr_boost: bool = DEFAULT_LR_BOOST  # Toggle learning rate boost
    lr_boost_factor: float = DEFAULT_LR_BOOST_FACTOR  # Factor by which to boost learning rate
    lr_boost_duration: int = DEFAULT_LR_BOOST_DURATION  # Duration of learning rate boost
    low_reward_threshold: float = DEFAULT_LOW_REWARD_THRESHOLD  # Threshold for low rewards
    low_reward_window: int = DEFAULT_LOW_REWARD_WINDOW  # Window for low reward tracking

    # Trajectory learning configuration
    use_trajectory_learning: bool = DEFAULT_USE_TRAJECTORY_LEARNING  # Toggle trajectory learning
    gamma: float = DEFAULT_GAMMA  # Discount factor for trajectory learning


class ModularBrain(QuantumBrain):
    """
    Modular quantum brain architecture.

    Each sensory/cognitive module is mapped to one or more qubits.
    Features for each module are encoded as RX/RY/RZ rotations on their assigned qubits.
    Entanglement can be added within and between modules.
    """

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        config: ModularBrainConfig,
        shots: int = DEFAULT_SHOTS,
        device: DeviceType = DeviceType.CPU,
        learning_rate: ConstantLearningRate | DynamicLearningRate | None = None,
        parameter_initializer: ZeroInitializer
        | RandomPiUniformInitializer
        | RandomSmallUniformInitializer
        | ManualParameterInitializer
        | None = None,
        gradient_method: GradientCalculationMethod | None = None,
        gradient_max_norm: float | None = None,
        action_set: list[Action] = DEFAULT_ACTIONS,
        perf_mgmt: "RunnableQiskitFunction | None" = None,
    ) -> None:
        """
        Initialize the ModularBrain.

        Args:
            config: Configuration for the ModularBrain architecture.
            shots: Number of shots for simulation.
            device: Device string for AerSimulator or real QPU backend.
            learning_rate: Learning rate strategy (default is dynamic).
            parameter_initializer : The initializer to use for parameter initialization.
            gradient_method: Optional gradient processing method (raw/normalize/clip/norm_clip).
            gradient_max_norm: Maximum gradient norm for norm_clip method.
            action_set: List of available actions (default is DEFAULT_ACTIONS).
            perf_mgmt: Q-CTRL performance management function instance.
        """
        self.config = config
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()
        num_qubits = count_total_qubits(config.modules)
        logger.info(
            f"Using configuration: {config}",
        )

        self.num_qubits: int = num_qubits
        self.modules: dict[ModuleName, list[int]] = config.modules or deepcopy(DEFAULT_MODULES)
        self.shots: int = shots
        self.device: DeviceType = device
        self.learning_rate = learning_rate or DynamicLearningRate()
        logger.info(
            f"Using learning rate: {str(self.learning_rate).replace('θ', 'theta_')}",
        )

        self.parameter_initializer = parameter_initializer or RandomSmallUniformInitializer()
        logger.info(
            "Using parameter initializer: "
            f"{str(self.parameter_initializer).replace('θ', 'theta_')}",
        )

        self.gradient_method = gradient_method
        self.gradient_max_norm = gradient_max_norm or DEFAULT_MAX_GRADIENT_NORM
        logger.info(
            f"Using gradient calculation method: {self.gradient_method}, "
            f"max_norm: {self.gradient_max_norm}",
        )

        self.action_set = action_set
        self.perf_mgmt = perf_mgmt

        if perf_mgmt is not None:
            if device != DeviceType.QPU:
                error_message = (
                    "Q-CTRL Fire Opal Performance Management can only be used with QPU devices."
                )
                logger.error(error_message)
                raise ValueError(error_message)
            logger.info("Q-CTRL Fire Opal Performance Management enabled.")

        self.num_layers = config.num_layers
        # Dynamically create parameters for each layer
        self.parameters = {}
        for layer in range(self.num_layers):
            self.parameters[f"rx_{layer + 1}"] = [
                Parameter(f"θ_rx{layer + 1}_{i}") for i in range(self.num_qubits)
            ]
            self.parameters[f"ry_{layer + 1}"] = [
                Parameter(f"θ_ry{layer + 1}_{i}") for i in range(self.num_qubits)
            ]
            self.parameters[f"rz_{layer + 1}"] = [
                Parameter(f"θ_rz{layer + 1}_{i}") for i in range(self.num_qubits)
            ]

        # Initialize parameter values using the configured initializer
        param_names = []
        for layer in range(self.num_layers):
            for axis in ["rx", "ry", "rz"]:
                param_names.extend([f"θ_{axis}{layer + 1}_{i}" for i in range(self.num_qubits)])

        self.parameter_values = self.parameter_initializer.initialize(self.num_qubits, param_names)

        self._circuit_cache: QuantumCircuit | None = None
        self._transpiled_cache: Any = None
        self._backend: AerSimulator | BackendV2 | None = None

        self._momentum = {}

        # Learning rate boost logic
        self._lr_boost_active = False
        self._lr_boost_steps_remaining = 0
        self._lr_boost_factor = config.lr_boost_factor
        self._lr_boost_duration = config.lr_boost_duration
        self._low_reward_threshold = config.low_reward_threshold
        self._low_reward_window = config.low_reward_window

        # Overfitting detection
        self.overfitting_detector = create_overfitting_detector_for_brain("modular")
        self.overfit_detector_episode_count = 0
        self.overfit_detector_episode_actions = []
        self.overfit_detector_current_episode_positions = []
        self.overfit_detector_current_episode_rewards = []
        self.overfit_detector_episode_log_interval = config.overfit_detector_episode_log_interval

        # Trajectory learning
        self.use_trajectory_learning = config.use_trajectory_learning
        self.gamma = config.gamma
        self.episode_buffer: EpisodeBuffer | None = (
            EpisodeBuffer() if config.use_trajectory_learning else None
        )

    def build_brain(
        self,
        input_params: dict[str, dict[str, float]] | None,
    ) -> QuantumCircuit:
        """
        Build the quantum circuit for the modular brain.

        Args:
            input_params: Feature dict for each module.

        Returns
        -------
            QuantumCircuit with parameterized gates and entanglement.
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Add Hadamard layer to create initial superposition
        for q in range(self.num_qubits):
            qc.h(q)

        # Dynamically add layers
        for layer in range(self.num_layers):
            # Feature encoding for this layer
            for module, qubit_indices in self.modules.items():
                features = input_params.get(module, {}) if input_params else {}
                for _idx, q in enumerate(qubit_indices):
                    rx = features.get(RotationAxis.RX.value, 0.0)
                    ry = features.get(RotationAxis.RY.value, 0.0)
                    rz = features.get(RotationAxis.RZ.value, 0.0)
                    qc.rx(rx + self.parameters[f"{RotationAxis.RX.value}_{layer + 1}"][q], q)
                    qc.ry(ry + self.parameters[f"{RotationAxis.RY.value}_{layer + 1}"][q], q)
                    qc.rz(rz + self.parameters[f"{RotationAxis.RZ.value}_{layer + 1}"][q], q)
            # Entanglement for this layer
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    qc.cz(i, j)

        qc.measure(range(self.num_qubits), range(self.num_qubits))

        return qc

    def _get_backend(self) -> "AerSimulator | BackendV2":
        """Return the backend: AerSimulator or IBM QPU backend."""
        if self._backend is None:
            if self.device == DeviceType.QPU:
                try:
                    from qiskit_ibm_runtime import QiskitRuntimeService

                    service = QiskitRuntimeService()
                    if backend_name := os.environ.get("IBM_QUANTUM_BACKEND"):
                        self._backend = service.backend(backend_name)
                    else:
                        self._backend = service.least_busy(
                            operational=True,
                            simulator=False,
                            min_num_qubits=self.num_qubits,
                        )
                except ImportError as err:
                    error_message = ERROR_MISSING_IMPORT_QISKIT_IBM_RUNTIME
                    logger.error(error_message)
                    raise ImportError(error_message) from err
                except Exception as err:
                    error_message = f"Failed to load IBM Quantum backend: {err}"
                    logger.error(error_message)
                    raise RuntimeError(error_message) from err
            else:
                try:
                    from qiskit_aer import AerSimulator
                except ImportError as err:
                    error_message = ERROR_MISSING_IMPORT_QISKIT_AER
                    logger.error(error_message)
                    raise ImportError(error_message) from err

                self._backend = AerSimulator(device=self.device.value.upper())
        return self._backend

    def _get_cached_circuit(self) -> QuantumCircuit:
        """Build and cache the parameterized circuit structure (unbound parameters)."""
        if self._circuit_cache is None:
            # Use zeros for features, just to build the structure
            input_params = {module.value: {} for module in self.modules}
            self._circuit_cache = self.build_brain(input_params)

        return self._circuit_cache

    def _get_transpiled(self) -> QuantumCircuit:
        """Transpile and cache the parameterized circuit structure."""
        if self._transpiled_cache is None:
            qc = self._get_cached_circuit()
            backend = self._get_backend()
            self._transpiled_cache = transpile(qc, backend)

        return self._transpiled_cache

    def _get_backend_name(self) -> str:
        """Return the backend name as a string for Q-CTRL Fire Opal."""
        if self.device == DeviceType.QPU:
            # For QPU, get the backend name from environment or use least busy
            if backend_name := os.environ.get("IBM_QUANTUM_BACKEND"):
                return backend_name
            # Get the backend object to extract its name
            backend = self._get_backend()
            return getattr(backend, "name", str(backend))
        # For simulators, use the device type
        return f"aer_simulator_{self.device.value}"

    def run_brain(  # noqa: C901, PLR0912, PLR0915
        self,
        params: BrainParams,
        reward: float | None = None,
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool = True,
        top_randomize: bool = True,
    ) -> list[ActionData]:
        """
        Run the quantum brain simulation for the given parameters.

        Run the quantum brain simulation for the given parameters,
        return action probabilities based on measurement counts,
        and update parameters if reward is provided.

        Args:
            params: BrainParams for the agent/environment state.
            reward: Optional reward (unused).
            input_data: Optional input data (unused).
            top_only: If True, return only the most probable action.
            top_randomize: If True, randomly select among top actions.

        Returns
        -------
            list[ActionData]: List of ActionData with action and probability.
        """
        gradient_strength = params.gradient_strength
        if gradient_strength:
            self.history_data.gradient_strengths.append(
                gradient_strength,
            )  # Used for reporting only

        gradient_direction = params.gradient_direction
        if gradient_direction:
            self.history_data.gradient_directions.append(
                gradient_direction,
            )  # Used for reporting only

        input_params = {
            module.value: extract_features_for_module(module, params) for module in self.modules
        }

        flat_input_params = {
            f"{module}_{k}": v
            for module, features in input_params.items()
            for k, v in features.items()
        }
        self.latest_data.input_parameters = flat_input_params
        self.history_data.input_parameters.append(flat_input_params)

        # Build the circuit with current input_params (features)
        qc = self.build_brain(input_params)
        param_values = self.parameter_values.copy()

        if self.device == DeviceType.QPU:
            if self.perf_mgmt is not None:
                # Use Q-CTRL's Fire Opal Performance Management Sampler

                # Q-CTRL recommends not transpiling circuits before submission
                # Bind parameters to the untranspiled circuit
                bound_qc = qc.assign_parameters(param_values, inplace=False)

                # Create sampler PUB (circuit, parameter_values, shots)
                sampler_pubs = [(bound_qc, None, self.shots)]

                # Execute Q-CTRL sampler job
                result = self._execute_qctrl_sampler_job(sampler_pubs)
                pub_result = result[0]
                bitstrings = pub_result.data.c
                counts = bitstrings.get_counts()
            else:
                # Use standard Qiskit Runtime Sampler
                try:
                    from qiskit_ibm_runtime import Sampler
                except ImportError as err:
                    error_message = ERROR_MISSING_IMPORT_QISKIT_IBM_RUNTIME
                    logger.error(error_message)
                    raise ImportError(error_message) from err

                # Get backend
                backend = self._get_backend()

                # Transpile and bind parameters
                bound_qc = transpile(qc, backend).assign_parameters(param_values, inplace=False)

                # Create sampler
                sampler = Sampler(mode=backend)

                # Submit job
                job = sampler.run([bound_qc], shots=self.shots)
                logger.info(f"Qiskit Runtime Job ID: {job.job_id}")

                result = job.result()
                pub_result = result[0]
                bitstrings = pub_result.data.c
                counts = bitstrings.get_counts()
        else:
            # Use AerSimulator
            backend = self._get_backend()
            bound_qc = transpile(qc, backend).assign_parameters(param_values, inplace=False)
            job = backend.run(bound_qc, shots=self.shots)
            if job is None:
                error_message = "Backend run did not return a valid job object."
                logger.error(error_message)
                raise RuntimeError(error_message)

            result = job.result()
            counts = result.get_counts()

        self.latest_data.counts = counts
        self.history_data.counts.append(counts)

        actions = self._interpret_counts(
            counts,
            top_only=top_only,
            top_randomize=top_randomize,
        )

        # --- Reward-based learning: compute gradients and update parameters ---
        if reward is not None and self.latest_data.action is not None:
            if self.use_trajectory_learning and self.episode_buffer is not None:
                # Trajectory learning mode: buffer data for episode-level update
                self.episode_buffer.append(
                    params=params,
                    action=self.latest_data.action,
                    reward=reward,
                    parameter_values=self.parameter_values,
                )
            else:
                # Single-step learning mode: immediate gradient update
                # Learning rate boost logic
                if self.config.lr_boost:
                    lr = self._handle_learning_rate_boost()
                else:
                    lr = self.learning_rate.get_learning_rate()
                gradients = self.parameter_shift_gradients(params, self.latest_data.action, reward)
                self.update_parameters(gradients, reward=reward, learning_rate=lr)

        # Track for overfitting detection
        if actions:
            self._track_episode_metrics(params, actions, reward)

        self.history_data.rewards.append(reward or 0.0)

        return actions

    def _handle_learning_rate_boost(self) -> float:
        """Handle learning rate boost based on recent rewards."""
        recent_rewards = (
            self.history_data.rewards[-self._low_reward_window :]
            if len(self.history_data.rewards) >= self._low_reward_window
            else self.history_data.rewards
        )
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        if (
            not self._lr_boost_active
            and avg_reward < self._low_reward_threshold
            and len(recent_rewards) == self._low_reward_window
        ):
            self._lr_boost_active = True
            self._lr_boost_steps_remaining = self._lr_boost_duration
            logger.info(f"Learning rate boost activated: avg_reward={avg_reward:.4f}")
        if self._lr_boost_active:
            lr = self.learning_rate.get_learning_rate() * self._lr_boost_factor
            self._lr_boost_steps_remaining -= 1
            if self._lr_boost_steps_remaining <= 0:
                self._lr_boost_active = False
                logger.info("Learning rate boost deactivated.")
        else:
            lr = self.learning_rate.get_learning_rate()

        return lr

    def _track_episode_metrics(
        self,
        params: BrainParams,
        actions: list[ActionData],
        reward: float | None,
    ) -> None:
        """Track metrics for the current episode."""
        selected_action = actions[0].action
        self.overfit_detector_episode_actions.append(selected_action)

        if params.agent_position is not None:
            self.overfit_detector_current_episode_positions.append(params.agent_position)

        self.overfit_detector_current_episode_rewards.append(reward or 0.0)

        # Track policy outputs (action probabilities) for consistency analysis
        action_probs = np.array([action.probability for action in actions])

        # Pad to full action space if needed
        if len(action_probs) < len(self.action_set):
            full_probs = np.zeros(len(self.action_set))
            for _, action in enumerate(actions):
                if action.action in self.action_set:
                    action_idx = self.action_set.index(action.action)
                    full_probs[action_idx] = action.probability
            action_probs = full_probs

        self.overfitting_detector.update_learning_metrics(None, action_probs)

    def _execute_qctrl_sampler_job(self, sampler_pubs: list[tuple]) -> "PrimitiveResult":
        """
        Execute a Q-CTRL Qiskit Function sampler job using the provided sampler publications.

        Args:
            sampler_pubs (list[tuple]): A list of tuples representing sampler
                publications to be executed.

        Returns
        -------
            PrimitiveResult: The results obtained from executing the Q-CTRL sampler job.

        Raises
        ------
            RuntimeError: If performance management is not initialized.

        Logs:
            - Logs an error if performance management is not initialized.
            - Logs detailed job status updates during execution.
            - Logs job completion status and timing information.
        """
        if not self.perf_mgmt:
            error_message = "Performance management is not initialized."
            logger.error(error_message)
            raise RuntimeError(error_message)

        # Submit the Q-CTRL job
        logger.info(f"Submitting Q-CTRL sampler job with {len(sampler_pubs)} publication(s)")
        qctrl_sampler_job = self.perf_mgmt.run(
            primitive="sampler",
            pubs=sampler_pubs,
            backend_name=self._get_backend_name(),
        )

        # Monitor job status with detailed logging
        monitor_job(qctrl_sampler_job, "Q-CTRL Sampler")

        # Retrieve results
        logger.info("Retrieving Q-CTRL sampler job results...")
        result = qctrl_sampler_job.result()
        logger.info("Q-CTRL sampler job results retrieved successfully")

        return result  # type: ignore[reportReturnType]

    def _interpret_counts(
        self,
        counts: dict,
        *,
        top_only: bool = True,
        top_randomize: bool = True,
    ) -> list[ActionData]:
        """
        Interpret measurement counts and return the most probable action(s).

        Args:
            counts: Measurement counts from the quantum circuit.
            top_only: If True, return only the most probable action.
            top_randomize: If True, randomly select among top actions.

        Returns
        -------
            ActionData or list of ActionData with action and probability.
        """
        # Map binary states to actions
        num_states = 2**self.num_qubits
        binary_states = [f"{{:0{self.num_qubits}b}}".format(i) for i in range(num_states)]

        action_pool = DEFAULT_ACTIONS
        action_map = {
            state: action_pool[i % len(action_pool)] for i, state in enumerate(binary_states)
        }

        # Filter counts
        counts = {k: v for k, v in counts.items() if k in action_map}
        if not counts:
            error_message = "No valid actions found in counts."
            logger.error(error_message)
            raise ValueError(error_message)

        total_counts = sum(counts.values())
        probabilities = {k: v / total_counts for k, v in counts.items()}
        sorted_actions = sorted(
            [
                ActionData(state=k, action=action_map[k], probability=probabilities[k])
                for k in probabilities
            ],
            key=lambda x: x.probability,
            reverse=True,
        )

        if top_only:
            if top_randomize:
                actions = [a.state for a in sorted_actions]
                probs = [a.probability for a in sorted_actions]
                rng = np.random.default_rng()
                chosen_state = rng.choice(actions, p=probs)
                chosen_action = next(a for a in sorted_actions if a.state == chosen_state)
                self.latest_data.action = chosen_action
                self.history_data.actions.append(chosen_action)

                return [chosen_action]

            self.latest_data.action = sorted_actions[0]
            self.history_data.actions.append(sorted_actions[0])

            return [sorted_actions[0]]

        self.latest_data.action = sorted_actions[0]
        self.history_data.actions.append(sorted_actions[0])

        return sorted_actions

    def compute_discounted_returns(self, rewards: list[float], gamma: float) -> list[float]:
        """
        Compute discounted returns backward through time.

        Implements the equation: G_t = r_t + gamma * G_{t+1}
        where G_T = r_T for the terminal state.

        Args:
            rewards: List of immediate rewards for each timestep [r_0, r_1, ..., r_T].
            gamma: Discount factor in range [0, 1].

        Returns
        -------
            List of discounted returns [G_0, G_1, ..., G_T].

        Raises
        ------
            ValueError: If gamma is not in range [0, 1].
        """
        if not 0 <= gamma <= 1:
            error_message = f"Gamma must be in range [0, 1], got {gamma}"
            logger.error(error_message)
            raise ValueError(error_message)

        if not rewards:
            return []

        returns = []
        discounted_return = 0.0

        # Iterate backward from terminal state
        for reward in reversed(rewards):
            discounted_return = reward + gamma * discounted_return
            returns.insert(0, discounted_return)

        return returns

    def _normalize_returns(self, returns: list[float]) -> list[float]:
        """
        Normalize returns to zero mean and unit variance.

        This prevents gradient explosion when returns have large magnitudes
        or vary significantly across episodes.

        Args:
            returns: List of discounted returns to normalize.

        Returns
        -------
            List of normalized returns with mean ≈ 0 and std ≈ 1.
        """
        if not returns:
            return []

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Avoid division by zero for constant returns
        if std_return < 1e-8:  # noqa: PLR2004
            logger.debug(
                f"Returns have near-zero variance (std={std_return:.2e}), returning zero gradients",
            )
            return [0.0] * len(returns)

        normalized = [float((r - mean_return) / std_return) for r in returns]
        logger.debug(
            f"Normalized returns: mean={mean_return:.3f}, std={std_return:.3f} -> "
            f"normalized mean={np.mean(normalized):.3e}, std={np.std(normalized):.3f}",
        )
        return normalized

    def update_memory(self, reward: float | None) -> None:
        """
        Update internal memory based on reward.

        Args:
            reward: Reward signal (positive or negative).
        """
        # Reserved for future brain-internal memory mechanisms

    def post_process_episode(self) -> None:
        """Post-process the episode data."""
        # Trajectory learning: compute returns and update parameters
        if (
            self.use_trajectory_learning
            and self.episode_buffer is not None
            and len(self.episode_buffer) > 0
        ):
            logger.info(
                f"Trajectory learning: processing {len(self.episode_buffer)} timesteps",
            )

            # 1. Compute discounted returns from buffered rewards
            returns = self.compute_discounted_returns(
                self.episode_buffer.rewards,
                self.gamma,
            )
            logger.debug(f"Computed returns: min={min(returns):.4f}, max={max(returns):.4f}")

            # 1b. Normalize returns to prevent gradient explosion
            returns = self._normalize_returns(returns)

            # 2. Compute trajectory-level gradients
            gradients = self.trajectory_parameter_shift_gradients(
                self.episode_buffer,
                returns,
            )
            logger.debug(
                f"Computed gradients: "
                f"min={min(gradients):.6f}, max={max(gradients):.6f}, "
                f"mean={np.mean(gradients):.6f}",
            )

            # 3. Update parameters using accumulated gradients
            # Get learning rate (with boost logic if enabled)
            if self.config.lr_boost:
                lr = self._handle_learning_rate_boost()
            else:
                lr = self.learning_rate.get_learning_rate()

            # Use total episode reward for update
            total_reward = sum(self.episode_buffer.rewards)
            self.update_parameters(gradients, reward=total_reward, learning_rate=lr)

            logger.info(
                f"Trajectory learning update complete: "
                f"lr={lr:.6f}, total_reward={total_reward:.4f}",
            )

            # Clear buffer for next episode
            self.episode_buffer.clear()

        final_reward = self.history_data.rewards[-1] if self.history_data.rewards else 0.0
        self._complete_episode_tracking(final_reward=final_reward)

    def _complete_episode_tracking(self, final_reward: float) -> None:
        """Complete episode tracking for overfitting detection."""
        if not self.overfit_detector_episode_actions:
            return

        # Calculate episode metrics
        total_steps = len(self.overfit_detector_episode_actions)
        total_reward = (
            sum(self.overfit_detector_current_episode_rewards[-total_steps:])
            if self.overfit_detector_current_episode_rewards
            else final_reward
        )

        # Update overfitting detector
        self.overfitting_detector.update_performance_metrics(total_steps, total_reward)

        if (
            self.overfit_detector_episode_actions
            and self.overfit_detector_current_episode_positions
        ):
            start_pos = (
                self.overfit_detector_current_episode_positions[0]
                if self.overfit_detector_current_episode_positions
                else (0, 0)
            )
            self.overfitting_detector.update_behavioral_metrics(
                self.overfit_detector_episode_actions.copy(),
                self.overfit_detector_current_episode_positions.copy(),
                start_pos,
            )

        self.overfit_detector_episode_count += 1

        # Log overfitting analysis every n episodes
        if self.overfit_detector_episode_count % self.overfit_detector_episode_log_interval == 0:
            self.overfitting_detector.log_overfitting_analysis()

        # Reset overfitting tracking for new episode
        self.overfit_detector_episode_actions.clear()
        self.overfit_detector_current_episode_positions.clear()
        self.overfit_detector_current_episode_rewards.clear()

    def inspect_circuit(self) -> QuantumCircuit:
        """
        Return a text drawing of the current quantum circuit.

        Returns
        -------
            QuantumCircuit: The current quantum circuit.
        """
        qc = self._get_cached_circuit()
        qc.draw("text")

        return qc

    def copy(self) -> "ModularBrain":
        """
        Create a deep copy of the ModularBrain instance.

        Returns
        -------
            ModularBrain: A new instance with the same state.
        """
        new_brain = ModularBrain(
            config=self.config,
            shots=self.shots,
            device=self.device,
            learning_rate=deepcopy(self.learning_rate),
            parameter_initializer=deepcopy(self.parameter_initializer),
            gradient_method=self.gradient_method,
            gradient_max_norm=self.gradient_max_norm,
            action_set=self.action_set,
            perf_mgmt=self.perf_mgmt,
        )
        new_brain.parameter_values = deepcopy(self.parameter_values)
        new_brain._circuit_cache = deepcopy(self._circuit_cache)
        new_brain._transpiled_cache = deepcopy(self._transpiled_cache)
        new_brain._backend = self._backend

        return new_brain

    def trajectory_parameter_shift_gradients(  # noqa: C901, PLR0912, PLR0915
        self,
        episode_buffer: EpisodeBuffer,
        returns: list[float],
        shift: float = np.pi / 4,
    ) -> list[float]:
        """
        Compute trajectory-level parameter-shift gradients using discounted returns.

        Instead of using immediate rewards, this method accumulates gradients across
        all timesteps in an episode, weighted by discounted returns G_t.

        Implements: grad_i = sum_t 0.5 * (P_+(a_t) - P_-(a_t)) * G_t

        Args:
            episode_buffer: Buffer containing params, actions,
                and parameter values for each timestep.
            returns: List of discounted returns [G_0, G_1, ..., G_T] for each timestep.
            shift: The parameter shift value.

        Returns
        -------
            List of accumulated gradients, one per parameter.
        """
        if len(episode_buffer) == 0:
            error_message = "Cannot compute gradients from empty episode buffer"
            logger.error(error_message)
            raise ValueError(error_message)

        if len(returns) != len(episode_buffer):
            error_message = (
                f"Returns length ({len(returns)}) must match episode buffer length "
                f"({len(episode_buffer)})"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # Initialize accumulated gradients
        param_keys = list(episode_buffer.parameter_values[0].keys())
        n_params = len(param_keys)
        accumulated_gradients = [0.0] * n_params

        # For each timestep, compute parameter-shift gradients weighted by G_t
        import time

        total_timesteps = len(episode_buffer)
        logger.info(
            f"Computing trajectory gradients for {total_timesteps} timesteps "
            f"({n_params} parameters each, {total_timesteps * n_params * 2} circuit evaluations)",
        )
        start_time = time.time()

        for t in range(len(episode_buffer)):
            params_t = episode_buffer.params[t]
            action_t = episode_buffer.actions[t]
            param_values_t = episode_buffer.parameter_values[t]
            return_t = returns[t]

            # Log progress every 10% of timesteps
            if t % max(1, total_timesteps // 10) == 0:
                elapsed = time.time() - start_time
                progress_pct = (t / total_timesteps) * 100
                logger.debug(
                    f"Trajectory gradient progress: {progress_pct:.0f}% "
                    f"({t}/{total_timesteps} timesteps, {elapsed:.1f}s elapsed)",
                )

            # Prepare shifted parameter sets for this timestep
            param_sets = []
            for k in param_keys:
                plus = param_values_t.copy()
                minus = param_values_t.copy()
                plus[k] += shift
                minus[k] -= shift
                param_sets.append((plus, minus))

            # Build input features for this timestep
            input_params = {
                module.value: extract_features_for_module(module, params_t)
                for module in self.modules
            }

            # Batch circuits for this timestep
            circuits = []
            if self.device == DeviceType.QPU and self.perf_mgmt is not None:
                cached_circuit = self._get_cached_circuit()
                for plus, minus in param_sets:
                    circuits.append(cached_circuit.assign_parameters(plus, inplace=False))
                    circuits.append(cached_circuit.assign_parameters(minus, inplace=False))
            else:
                # Build circuit with input features for this timestep
                qc = self.build_brain(input_params)
                backend = self._get_backend()
                transpiled_template = transpile(qc, backend)
                for plus, minus in param_sets:
                    circuits.append(transpiled_template.assign_parameters(plus, inplace=False))
                    circuits.append(transpiled_template.assign_parameters(minus, inplace=False))

            # Run all circuits for this timestep
            if self.device == DeviceType.QPU:
                if self.perf_mgmt is not None:
                    sampler_pubs = [(circuit, None, self.shots) for circuit in circuits]
                    qctrl_result = self._execute_qctrl_sampler_job(sampler_pubs)

                    def get_counts_qctrl(
                        idx: int,
                        res: "PrimitiveResult" = qctrl_result,
                    ) -> dict[str, int]:
                        return res[idx].data.c.get_counts()

                    get_counts = get_counts_qctrl
                else:
                    try:
                        from qiskit_ibm_runtime import Sampler
                    except ImportError as err:
                        error_message = ERROR_MISSING_IMPORT_QISKIT_IBM_RUNTIME
                        logger.error(error_message)
                        raise ImportError(error_message) from err

                    backend = self._get_backend()
                    sampler = Sampler(mode=backend)
                    job = sampler.run(circuits, shots=self.shots)
                    qpu_result = job.result()

                    def get_counts_qpu(
                        idx: int,
                        res: "PrimitiveResult" = qpu_result,
                    ) -> dict[str, int]:
                        return res[idx].data.c.get_counts()

                    get_counts = get_counts_qpu
            else:
                backend = self._get_backend()
                job = backend.run(circuits, shots=self.shots)
                if job is None:
                    error_message = "Backend run did not return a valid job object."
                    logger.error(error_message)
                    raise RuntimeError(error_message)
                sim_results = job.result()

                def get_counts_sim(idx: int, res: object = sim_results) -> dict[str, int]:
                    return res.get_counts(idx)  # type: ignore[attr-defined]

                get_counts = get_counts_sim

            # Accumulate gradients for each parameter
            for i in range(n_params):
                counts_plus = get_counts(i * 2)
                counts_minus = get_counts(i * 2 + 1)
                prob_plus = self._get_action_probability(counts_plus, action_t.state)
                prob_minus = self._get_action_probability(counts_minus, action_t.state)

                prob_diff = prob_plus - prob_minus

                # Gradient contribution for this timestep weighted by return
                grad_t = 0.5 * prob_diff * return_t
                accumulated_gradients[i] += grad_t

        elapsed_total = time.time() - start_time
        logger.info(
            f"Trajectory gradient computation complete: {total_timesteps} timesteps "
            f"in {elapsed_total:.1f}s ({elapsed_total / total_timesteps:.2f}s/timestep)",
        )

        # Average gradients by episode length to prevent explosion with long episodes
        averaged_gradients = [g / total_timesteps for g in accumulated_gradients]
        logger.debug(
            f"Averaged gradients by {total_timesteps} timesteps: "
            f"sum_magnitude={np.linalg.norm(accumulated_gradients):.6f} -> "
            f"avg_magnitude={np.linalg.norm(averaged_gradients):.6f}",
        )

        return averaged_gradients

    def parameter_shift_gradients(  # noqa: C901, PLR0912, PLR0915
        self,
        params: BrainParams,  # noqa: ARG002
        action: ActionData,
        reward: float,
        shift: float = np.pi / 4,
    ) -> list[float]:
        """
        Compute parameter-wise gradients using the parameter-shift rule, batching all runs.

        Args:
            params: BrainParams for the agent/environment state.
            action: The action taken (for log-prob gradient).
            reward: Reward signal to guide gradient computation.
            shift: The parameter shift value.

        Returns
        -------
            List of gradients, one per parameter.
        """
        param_keys = list(self.parameter_values.keys())
        base_param_values = self.parameter_values.copy()
        n = len(param_keys)

        # Prepare all shifted parameter sets
        param_sets = []
        for _i, k in enumerate(param_keys):
            plus = base_param_values.copy()
            minus = base_param_values.copy()
            plus[k] += shift
            minus[k] -= shift
            param_sets.append((plus, minus))

        # Batch circuits
        circuits = []

        if self.device == DeviceType.QPU and self.perf_mgmt is not None:
            # For Q-CTRL, use untranspiled circuit
            cached_circuit = self._get_cached_circuit()
            for plus, minus in param_sets:
                circuits.append(cached_circuit.assign_parameters(plus, inplace=False))
                circuits.append(cached_circuit.assign_parameters(minus, inplace=False))
        else:
            # For standard execution, use transpiled circuit
            transpiled = self._get_transpiled()
            for plus, minus in param_sets:
                circuits.append(transpiled.assign_parameters(plus, inplace=False))
                circuits.append(transpiled.assign_parameters(minus, inplace=False))

        gradients = []

        # Run all circuits in a batch
        if self.device == DeviceType.QPU:
            if self.perf_mgmt is not None:
                # Use Q-CTRL's Fire Opal Performance Management Sampler for gradients

                # Q-CTRL recommends not transpiling circuits before submission
                # Create sampler PUBs for all circuits
                sampler_pubs = [(circuit, None, self.shots) for circuit in circuits]

                # Execute Q-CTRL sampler job
                result = self._execute_qctrl_sampler_job(sampler_pubs)

                def get_counts_qctrl(idx: int) -> dict[str, int]:
                    return result[idx].data.c.get_counts()

                get_counts = get_counts_qctrl
            else:
                # Use standard Qiskit Runtime Sampler for real QPU
                try:
                    from qiskit_ibm_runtime import Sampler
                except ImportError as err:
                    error_message = ERROR_MISSING_IMPORT_QISKIT_IBM_RUNTIME
                    logger.error(error_message)
                    raise ImportError(error_message) from err

                # Get backend
                backend = self._get_backend()

                # Create sampler
                sampler = Sampler(mode=backend)

                # Submit job
                job = sampler.run(circuits, shots=self.shots)
                logger.info(f"Qiskit Runtime Job ID: {job.job_id}")

                # Get results
                result = job.result()

                def get_counts_qpu(idx: int) -> dict[str, int]:
                    return result[idx].data.c.get_counts()

                get_counts = get_counts_qpu
        else:
            # Use AerSimulator
            backend = self._get_backend()
            job = backend.run(circuits, shots=self.shots)
            if job is None:
                error_message = "Backend run did not return a valid job object."
                logger.error(error_message)
                raise RuntimeError(error_message)
            results = job.result()

            def get_counts_sim(idx: int) -> dict[str, int]:
                return results.get_counts(idx)

            get_counts = get_counts_sim

        for i in range(n):
            counts_plus = get_counts(i * 2)
            counts_minus = get_counts(i * 2 + 1)
            prob_plus = self._get_action_probability(counts_plus, action.state)
            prob_minus = self._get_action_probability(counts_minus, action.state)

            # Enhanced gradient computation with minimum threshold
            prob_diff = prob_plus - prob_minus

            # Add small regularization to prevent vanishing gradients
            min_gradient_magnitude = self.config.min_gradient_magnitude
            if abs(prob_diff) < min_gradient_magnitude:
                # Use sign preservation with minimum magnitude
                prob_diff = (
                    min_gradient_magnitude * np.sign(prob_diff)
                    if prob_diff != 0
                    else min_gradient_magnitude
                )
                logger.warning(f"Gradient below minimum threshold: {prob_diff}")

            grad = 0.5 * prob_diff * reward

            # Scale gradient based on reward magnitude to improve learning signal
            if abs(reward) > self.config.significant_reward_threshold:  # For significant rewards
                grad *= min(2.0, abs(reward))  # Amplify but cap at 2x

            gradients.append(grad)

        # Store gradients for tracking
        self.latest_data.computed_gradients = gradients
        self.history_data.computed_gradients.append(gradients)

        return gradients

    def compute_gradients(
        self,
        counts: dict[str, int],  # noqa: ARG002
        reward: float,  # noqa: ARG002
        action: ActionData,  # noqa: ARG002
    ) -> list[float]:
        """
        Compute gradients based on measurement counts, reward, and action.

        This method is not implemented for ModularBrain. Use parameter_shift_gradients instead.

        Args:
            counts: Measurement counts from the quantum circuit.
            reward: Reward signal to guide gradient computation.
            action: The action taken (for log-prob gradient).

        Raises
        ------
            NotImplementedError: This method is not implemented for ModularBrain.
        """
        error_message = (
            "compute_gradients is not implemented for ModularBrain. "
            "Use parameter_shift_gradients instead."
        )
        logger.error(error_message)
        raise NotImplementedError(error_message)

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,  # noqa: ARG002
        learning_rate: float = 0.01,
    ) -> None:
        """Update quantum circuit parameter values based on gradients and learning rate."""
        param_keys = list(self.parameter_values.keys())

        # Initialize momentum if not exists
        if len(self._momentum) == 0:
            self._momentum = dict.fromkeys(param_keys, 0.0)

        # Momentum coefficient and base decay
        momentum_coefficient = self.config.momentum_coefficient
        base_momentum_decay = self.config.momentum_decay  # Prevents unbounded momentum accumulation

        # Adaptive momentum decay: reduce momentum retention when learning rate is low
        # This prevents momentum-driven drift when gradient signal is weak
        init_lr = self.learning_rate.initial_learning_rate
        if init_lr > 0:
            lr_ratio = learning_rate / init_lr
            # When lr is high (ratio ~1.0) → decay ~0.99 (high momentum retention)
            # When lr is low (ratio ~0.1) → decay ~0.82 (reduced momentum retention)
            # This prevents residual momentum from dominating when LR decays
            adaptive_momentum_decay = min(0.99, 0.80 + 0.19 * lr_ratio)
        else:
            adaptive_momentum_decay = base_momentum_decay

        # Apply gradient processing (clip/normalize/raw/norm_clip) based on config
        if self.gradient_method is not None:
            gradients = compute_gradients(
                gradients,
                self.gradient_method,
                DEFAULT_MAX_CLIP_GRADIENT,
                self.gradient_max_norm,
            )

        rng = np.random.default_rng()
        for i, k in enumerate(param_keys):
            # L2 regularization
            reg = self.config.l2_reg * self.parameter_values[k]

            # Add exploration noise (scaled with learning rate for stability after convergence)
            # Noise decays proportionally with LR
            effective_noise_std = 0.0
            if init_lr > 0:
                effective_noise_std = self.config.noise_std * (learning_rate / init_lr)
            noise = rng.normal(0, effective_noise_std)

            # Momentum update with adaptive learning rate and decay
            # Note: Using adaptive momentum decay instead of fixed momentum_decay
            # to prevent momentum drift when learning rate is low
            self._momentum[k] = (
                adaptive_momentum_decay * momentum_coefficient * self._momentum[k]
                + learning_rate * (gradients[i] - reg)  # L2 reg pushes parameters toward zero
            )

            # Update parameter
            self.parameter_values[k] -= self._momentum[k] + noise

        if self.config.param_clip:
            for k in param_keys:
                self.parameter_values[k] = np.clip(self.parameter_values[k], -np.pi, np.pi)

        if self.config.param_modulo:
            for k in param_keys:
                self.parameter_values[k] = (
                    (self.parameter_values[k] + np.pi) % (2 * np.pi)
                ) - np.pi

        self.latest_data.learning_rate = learning_rate
        self.latest_data.updated_parameters = deepcopy(self.parameter_values)

        self.history_data.learning_rates.append(learning_rate)
        self.history_data.updated_parameters.append(deepcopy(self.parameter_values))

    def _get_action_probability(self, counts: dict[str, int], state: str) -> float:
        """Return the probability of a given state (bitstring) from measurement counts."""
        total = sum(counts.values())
        if total == 0:
            return 0.0

        return counts.get(state, 0) / total
