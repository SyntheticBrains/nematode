"""Modular Quantum Brain Architecture for Multi-Modal Sensing."""

from copy import deepcopy
from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]
from qiskit import QuantumCircuit, transpile  # pyright: ignore[reportMissingImports]
from qiskit.circuit import Parameter  # pyright: ignore[reportMissingImports]
from qiskit_aer import AerSimulator  # pyright: ignore[reportMissingImports]

from quantumnematode.brain.arch import Brain, BrainParams
from quantumnematode.brain.modules import extract_features_for_module
from quantumnematode.logging_config import logger
from quantumnematode.models import ActionData
from quantumnematode.optimizer.learning_rate import DynamicLearningRate

# Example: Define the available modules and their qubit assignments
DEFAULT_MODULES: dict[str, list[int]] = {
    "proprioception": [0],
    "chemotaxis": [1],
}

ENTROPY_BETA = 0.07


class ModularBrain(Brain):
    """
    Modular quantum brain architecture.

    Each sensory/cognitive module is mapped to one or more qubits.
    Features for each module are encoded as RX/RY/RZ rotations on their assigned qubits.
    Entanglement can be added within and between modules.
    """

    def __init__(
        self,
        num_qubits: int | None = None,
        modules: dict[str, list[int]] | None = None,
        shots: int = 100,
        device: str = "CPU",
        learning_rate: DynamicLearningRate | None = None,
    ) -> None:
        """
        Initialize the ModularBrain.

        Args:
            num_qubits: Number of qubits (if None, inferred from modules).
            modules: Mapping of module names to qubit indices.
            shots: Number of shots for simulation.
            device: Device string for AerSimulator.
        """
        num_qubits = 2  # TODO: Get num qubits from module definitions

        self.num_qubits: int = num_qubits
        self.modules: dict[str, list[int]] = modules or deepcopy(DEFAULT_MODULES)
        self.shots: int = shots
        self.device: str = device.upper()
        self.satiety: float = 1.0
        self.learning_rate = learning_rate or DynamicLearningRate()

        self.parameters: dict[str, list[Parameter]] = {
            "rx": [Parameter(f"θ_rx_{i}") for i in range(self.num_qubits)],
            "ry": [Parameter(f"θ_ry_{i}") for i in range(self.num_qubits)],
            "rz": [Parameter(f"θ_rz_{i}") for i in range(self.num_qubits)],
        }
        self.parameter_values: dict[str, float] = {f"θ_rx_{i}": 0.0 for i in range(self.num_qubits)}
        self.parameter_values.update({f"θ_ry_{i}": 0.0 for i in range(self.num_qubits)})
        self.parameter_values.update({f"θ_rz_{i}": 0.0 for i in range(self.num_qubits)})

        self.latest_input_parameters: dict[str, float] | None = None
        self.latest_updated_parameters: dict[str, float] | None = None
        self.latest_counts: dict[str, int] | None = None
        self.latest_action: ActionData | None = None
        self.latest_gradients: list[float] | None = None
        self.latest_learning_rate: float | None = None
        self.latest_temperature: float | None = None  # Not used

        self.history_input_parameters: list[dict[str, float]] = []
        self.history_updated_parameters: list[dict[str, float]] = []
        self.history_gradients: list[list[float]] = []
        self.history_gradient_strengths: list[float] = []
        self.history_gradient_directions: list[float] = []
        self.history_rewards: list[float] = []
        self.history_learning_rates: list[float | dict[str, float]] = []
        self.history_counts: list[dict[str, int]] = []
        self.history_actions: list[ActionData] = []
        self.history_temperature: list[float] = []  # Not used

        self._circuit_cache: QuantumCircuit | None = None
        self._transpiled_cache: Any = None
        self._simulator: AerSimulator | None = None

    def build_brain(
        self,
        input_params: dict[str, dict[str, float]] | None = None,
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

        # Encode features for each module into their assigned qubits
        for module, qubit_indices in self.modules.items():
            features = input_params.get(module, {}) if input_params else {}
            for _idx, q in enumerate(qubit_indices):
                rx = features.get("rx", 0.0)
                ry = features.get("ry", 0.0)
                rz = features.get("rz", 0.0)
                qc.rx(rx + self.parameters["rx"][q], q)
                qc.ry(ry + self.parameters["ry"][q], q)
                qc.rz(rz + self.parameters["rz"][q], q)

        # Entangle qubits
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                qc.cz(i, j)

        qc.measure(range(self.num_qubits), range(self.num_qubits))

        return qc

    def _get_simulator(self) -> AerSimulator:
        if self._simulator is None:
            self._simulator = AerSimulator(device=self.device)
        return self._simulator

    def _get_cached_circuit(self) -> QuantumCircuit:
        """Build and cache the parameterized circuit structure (unbound parameters)."""
        if self._circuit_cache is None:
            # Use zeros for features, just to build the structure
            input_params = {module: {} for module in self.modules}
            self._circuit_cache = self.build_brain(input_params)

        return self._circuit_cache

    def _get_transpiled(self) -> QuantumCircuit:
        """Transpile and cache the parameterized circuit structure."""
        if self._transpiled_cache is None:
            qc = self._get_cached_circuit()
            simulator = self._get_simulator()
            self._transpiled_cache = transpile(qc, simulator)

        return self._transpiled_cache

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,
        input_data: list[float] | None = None,  # noqa: ARG002
    ) -> dict:
        """
        Run the quantum brain simulation for the given parameters.

        Run the quantum brain simulation for the given parameters,
        and update parameters if reward is provided.

        Args:
            params: BrainParams for the agent/environment state.
            reward: Optional reward (unused).
            input_data: Optional input data (unused).

        Returns
        -------
            Measurement counts from the quantum circuit.
        """
        gradient_strength = params.gradient_strength
        if gradient_strength:
            self.history_gradient_strengths.append(gradient_strength)  # Used for reporting only

        gradient_direction = params.gradient_direction
        if gradient_direction:
            self.history_gradient_directions.append(gradient_direction)  # Used for reporting only

        input_params = {
            module: extract_features_for_module(module, params, satiety=self.satiety)
            for module in self.modules
        }

        flat_input_params = {
            f"{module}_{k}": v
            for module, features in input_params.items()
            for k, v in features.items()
        }
        self.latest_input_parameters = flat_input_params
        self.history_input_parameters.append(flat_input_params)

        # Efficient: use cached transpiled circuit and bind parameters
        param_values = self.parameter_values.copy()
        transpiled = self._get_transpiled()
        bound_qc = transpiled.assign_parameters(param_values, inplace=False)
        simulator = self._get_simulator()
        result = simulator.run(bound_qc, shots=self.shots).result()
        counts = result.get_counts()

        self.latest_counts = counts
        self.history_counts.append(counts)

        # --- Reward-based learning: compute gradients and update parameters ---
        if reward is not None and self.latest_action is not None:
            gradients = self.parameter_shift_gradients(params, self.latest_action, reward)
            lr = self.learning_rate.get_learning_rate()
            self.update_parameters(gradients, reward=reward, learning_rate=lr)

        self.history_rewards.append(reward or 0.0)

        return counts

    def interpret_counts(
        self,
        counts: dict,
        *,
        top_only: bool = True,
        top_randomize: bool = True,
    ) -> ActionData | list[ActionData]:
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

        action_pool = ["forward", "left", "right", "stay"]
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
                self.latest_action = chosen_action
                self.history_actions.append(chosen_action)

                return chosen_action

            self.latest_action = sorted_actions[0]
            self.history_actions.append(sorted_actions[0])

            return sorted_actions[0]

        self.latest_action = sorted_actions[0]
        self.history_actions.append(sorted_actions[0])

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
            num_qubits=self.num_qubits,
            modules=deepcopy(self.modules),
            shots=self.shots,
            device=self.device,
        )
        new_brain.parameter_values = deepcopy(self.parameter_values)
        new_brain.satiety = self.satiety
        new_brain._circuit_cache = deepcopy(self._circuit_cache)
        new_brain._transpiled_cache = deepcopy(self._transpiled_cache)
        new_brain._simulator = self._simulator

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
        simulator = self._get_simulator()
        circuits = []
        for plus, minus in param_sets:
            circuits.append(transpiled.assign_parameters(plus, inplace=False))
            circuits.append(transpiled.assign_parameters(minus, inplace=False))

        # Run all circuits in one batch
        results = simulator.run(circuits, shots=self.shots).result()
        gradients = []
        for i in range(n):
            counts_plus = results.get_counts(i * 2)
            counts_minus = results.get_counts(i * 2 + 1)
            prob_plus = self._get_action_probability(counts_plus, action.state)
            prob_minus = self._get_action_probability(counts_minus, action.state)
            grad = 0.5 * (prob_plus - prob_minus) * reward
            gradients.append(grad)

        # Store gradients for tracking
        self.latest_gradients = gradients
        self.history_gradients.append(gradients)

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
        *,
        param_clip: bool = True,
        param_modulo: bool = True,
    ) -> None:
        """
        Update quantum circuit parameter values based on gradients and learning rate.

        Args:
            gradients: Gradients for each parameter.
            reward: Reward signal (optional, for performance-based LR).
            learning_rate: Learning rate for parameter updates.
            param_clip: Whether to clip parameters to [-pi, pi].
            param_modulo: Whether to wrap parameters to [-pi, pi].
        """
        param_keys = list(self.parameter_values.keys())
        for i, k in enumerate(param_keys):
            self.parameter_values[k] -= learning_rate * gradients[i]

        if param_clip:
            for k in param_keys:
                self.parameter_values[k] = np.clip(self.parameter_values[k], -np.pi, np.pi)

        if param_modulo:
            for k in param_keys:
                self.parameter_values[k] = (
                    (self.parameter_values[k] + np.pi) % (2 * np.pi)
                ) - np.pi

        self.latest_learning_rate = learning_rate
        self.latest_updated_parameters = deepcopy(self.parameter_values)

        self.history_learning_rates.append(learning_rate)
        self.history_updated_parameters.append(deepcopy(self.parameter_values))

    def _get_action_probability(self, counts: dict[str, int], state: str) -> float:
        """Return the probability of a given state (bitstring) from measurement counts."""
        total = sum(counts.values())
        if total == 0:
            return 0.0

        return counts.get(state, 0) / total
