"""Modular Quantum Brain Architecture for Multi-Modal Sensing."""

import os
from copy import deepcopy
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
from quantumnematode.initializers.random_initializer import (
    RandomPiUniformInitializer,
    RandomSmallUniformInitializer,
)
from quantumnematode.initializers.zero_initializer import ZeroInitializer
from quantumnematode.logging_config import logger
from quantumnematode.optimizers.learning_rate import DynamicLearningRate

if TYPE_CHECKING:
    from qiskit.providers import BackendV2
    from qiskit_aer import AerSimulator

# Defaults
DEFAULT_GRADIENT_NORM_THRESHOLD = 1e-1
DEFAULT_L2_REG = 0.01
DEFAULT_NOISE_STD = 0.01
DEFAULT_NUM_LAYERS = 2
DEFAULT_PARAM_CLIP = True
DEFAULT_PARAM_MODULO = True
DEFAULT_RESET_ON_STALL = True
DEFAULT_RESET_PATIENCE = 20
DEFAULT_RESET_STD = 0.1


class ModularBrainConfig(BrainConfig):
    """Configuration for the ModularBrain architecture."""

    gradient_norm_threshold: float = DEFAULT_GRADIENT_NORM_THRESHOLD  # Threshold for gradient reset
    l2_reg: float = DEFAULT_L2_REG  # L2 regularization strength
    modules: dict[ModuleName, list[int]] = (
        DEFAULT_MODULES  # Mapping of module names to qubit indices
    )
    noise_std: float = DEFAULT_NOISE_STD  # Standard deviation for parameter noise
    num_layers: int = DEFAULT_NUM_LAYERS  # Number of layers in the quantum circuit
    param_clip: bool = DEFAULT_PARAM_CLIP  # Toggle parameter clipping
    param_modulo: bool = DEFAULT_PARAM_MODULO  # Toggle parameter modulo
    reset_on_stall: bool = DEFAULT_RESET_ON_STALL  # Toggle reset
    reset_patience: int = DEFAULT_RESET_PATIENCE  # Number of steps with low gradient before reset
    reset_std: float = DEFAULT_RESET_STD  # Standard deviation for parameter reset


class ModularBrain(QuantumBrain):
    """
    Modular quantum brain architecture.

    Each sensory/cognitive module is mapped to one or more qubits.
    Features for each module are encoded as RX/RY/RZ rotations on their assigned qubits.
    Entanglement can be added within and between modules.
    """

    def __init__(  # noqa: PLR0913
        self,
        config: ModularBrainConfig,
        shots: int = DEFAULT_SHOTS,
        device: DeviceType = DeviceType.CPU,
        learning_rate: DynamicLearningRate | None = None,
        parameter_initializer: ZeroInitializer
        | RandomPiUniformInitializer
        | RandomSmallUniformInitializer
        | None = None,
        action_set: list[Action] = DEFAULT_ACTIONS,
    ) -> None:
        """
        Initialize the ModularBrain.

        Args:
            modules: Mapping of module names to qubit indices.
            shots: Number of shots for simulation.
            device: Device string for AerSimulator or real QPU backend.
            learning_rate: Learning rate strategy (default is dynamic).
            parameter_initializer : The initializer to use for parameter initialization.
            action_set: List of available actions (default is DEFAULT_ACTIONS).
            num_layers: Number of layers in the quantum circuit.
        """
        self.config = config
        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()
        num_qubits = count_total_qubits(config.modules)

        self.num_qubits: int = num_qubits
        self.modules: dict[ModuleName, list[int]] = config.modules or deepcopy(DEFAULT_MODULES)
        self.shots: int = shots
        self.device: DeviceType = device
        self.satiety: float = 1.0
        self.learning_rate = learning_rate or DynamicLearningRate()
        logger.info(
            f"Using learning rate: {str(self.learning_rate).replace('θ', 'theta_')}",
        )

        self.parameter_initializer = parameter_initializer or RandomSmallUniformInitializer()
        logger.info(
            "Using parameter initializer: "
            f"{str(self.parameter_initializer).replace('θ', 'theta_')}",
        )

        self.action_set = action_set

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
        self.parameter_values = {}
        for layer in range(self.num_layers):
            for axis in ["rx", "ry", "rz"]:
                param_names = [f"θ_{axis}{layer + 1}_{i}" for i in range(self.num_qubits)]
                self.parameter_values.update(
                    self.parameter_initializer.initialize(self.num_qubits, param_names),
                )

        self._circuit_cache: QuantumCircuit | None = None
        self._transpiled_cache: Any = None
        self._backend: AerSimulator | BackendV2 | None = None

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

    def run_brain(
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
            module.value: extract_features_for_module(module, params, satiety=self.satiety)
            for module in self.modules
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
        backend = self._get_backend()
        param_values = self.parameter_values.copy()
        bound_qc = transpile(qc, backend).assign_parameters(param_values, inplace=False)

        if self.device == DeviceType.QPU:
            try:
                from qiskit_ibm_runtime import Sampler
            except ImportError as err:
                error_message = ERROR_MISSING_IMPORT_QISKIT_IBM_RUNTIME
                logger.error(error_message)
                raise ImportError(error_message) from err

            # Use Qiskit Runtime Sampler
            sampler = Sampler(mode=backend)
            job = sampler.run([bound_qc], shots=self.shots)
            result = job.result()
            pub_result = result[0]
            bitstrings = pub_result.data.c
            counts = bitstrings.get_counts()
        else:
            # Use AerSimulator
            job = backend.run(bound_qc, shots=self.shots)
            if job is None:
                error_message = "Backend run did not return a valid job object."
                logger.error(error_message)
                raise RuntimeError(error_message)

            result = job.result()
            counts = result.get_counts()

        self.latest_data.counts = counts
        self.history_data.counts.append(counts)

        # --- Reward-based learning: compute gradients and update parameters ---
        if reward is not None and self.latest_data.action is not None:
            gradients = self.parameter_shift_gradients(params, self.latest_data.action, reward)
            lr = self.learning_rate.get_learning_rate()
            self.update_parameters(gradients, reward=reward, learning_rate=lr)

        self.history_data.rewards.append(reward or 0.0)

        return self._interpret_counts(
            counts,
            top_only=top_only,
            top_randomize=top_randomize,
        )

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

    def update_memory(self, reward: float | None) -> None:
        """
        Update internal memory (e.g., satiety) based on reward.

        Args:
            reward: Reward signal (positive or negative).
        """
        # Example: satiety increases with positive reward, decreases otherwise
        if reward is not None:
            self.satiety = min(1.0, max(0.0, self.satiety + reward))

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
        )
        new_brain.parameter_values = deepcopy(self.parameter_values)
        new_brain.satiety = self.satiety
        new_brain._circuit_cache = deepcopy(self._circuit_cache)
        new_brain._transpiled_cache = deepcopy(self._transpiled_cache)
        new_brain._backend = self._backend

        return new_brain

    def parameter_shift_gradients(
        self,
        params: BrainParams,  # noqa: ARG002
        action: ActionData,
        reward: float,
        shift: float = np.pi / 2,
    ) -> list[float]:
        """
        Compute parameter-wise gradients using the parameter-shift rule, batching all runs.

        Args:
            params: BrainParams for the agent/environment state.
            action: The action taken (for log-prob gradient).
            reward: Reward signal to guide gradient computation.
            shift: The parameter shift value (default π/2).

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
        transpiled = self._get_transpiled()
        backend = self._get_backend()
        circuits = []
        for plus, minus in param_sets:
            circuits.append(transpiled.assign_parameters(plus, inplace=False))
            circuits.append(transpiled.assign_parameters(minus, inplace=False))

        gradients = []

        # Run all circuits in a batch
        if self.device == DeviceType.QPU:
            try:
                from qiskit_ibm_runtime import Sampler
            except ImportError as err:
                error_message = ERROR_MISSING_IMPORT_QISKIT_IBM_RUNTIME
                logger.error(error_message)
                raise ImportError(error_message) from err

            # Use Qiskit Runtime Sampler for real QPU
            sampler = Sampler(mode=backend)
            job = sampler.run(circuits, shots=self.shots)
            result = job.result()

            def get_counts_qpu(idx: int) -> dict[str, int]:
                return result[idx].data.c.get_counts()

            get_counts = get_counts_qpu
        else:
            # Use AerSimulator
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
            grad = 0.5 * (prob_plus - prob_minus) * reward
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

    def update_parameters(  # noqa: C901
        self,
        gradients: list[float],
        reward: float | None = None,  # noqa: ARG002
        learning_rate: float = 0.01,
    ) -> None:
        """
        Update quantum circuit parameter values based on gradients and learning rate.

        If the mean gradient norm is below a threshold for several steps,
        parameters are re-initialized (reset).
        """
        param_keys = list(self.parameter_values.keys())
        for i, k in enumerate(param_keys):
            reg = self.config.l2_reg * self.parameter_values[k]
            rng = np.random.default_rng()
            noise = rng.normal(0, self.config.noise_std)
            self.parameter_values[k] -= learning_rate * (gradients[i] + reg) + noise

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

        # --- Gradient norm monitoring and parameter reset logic ---
        if self.config.reset_on_stall:
            grad_norm = np.mean(np.abs(gradients))
            if not hasattr(self, "_low_grad_steps"):
                self._low_grad_steps = 0
            if grad_norm < self.config.gradient_norm_threshold:
                self._low_grad_steps += 1
            else:
                self._low_grad_steps = 0
            if self._low_grad_steps >= self.config.reset_patience:
                logger.warning(
                    f"Parameter reset: gradient norm {grad_norm:.2e} "
                    "below threshold for {reset_patience} steps. Re-initializing parameters.",
                )
                rng = np.random.default_rng()
                for k in param_keys:
                    self.parameter_values[k] = rng.uniform(
                        -self.config.reset_std,
                        self.config.reset_std,
                    )
                self._low_grad_steps = 0

    def _get_action_probability(self, counts: dict[str, int], state: str) -> float:
        """Return the probability of a given state (bitstring) from measurement counts."""
        total = sum(counts.values())
        if total == 0:
            return 0.0

        return counts.get(state, 0) / total
