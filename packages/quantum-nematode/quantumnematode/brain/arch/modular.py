"""Modular Quantum Brain Architecture for Multi-Modal Sensing."""

from copy import deepcopy
from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]
from qiskit import QuantumCircuit, transpile  # pyright: ignore[reportMissingImports]
from qiskit.circuit import Parameter  # pyright: ignore[reportMissingImports]
from qiskit_aer import AerSimulator  # pyright: ignore[reportMissingImports]

from quantumnematode.brain import Brain, BrainParams
from quantumnematode.brain.modules import extract_features_for_module
from quantumnematode.logging_config import logger
from quantumnematode.models import ActionData

# Example: Define the available modules and their qubit assignments
DEFAULT_MODULES: dict[str, list[int]] = {
    "proprioception": [0],
    "chemotaxis": [1],
}


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

        self.parameters: dict[str, list[Parameter]] = {
            "rx": [Parameter(f"θ_rx_{i}") for i in range(self.num_qubits)],
            "ry": [Parameter(f"θ_ry_{i}") for i in range(self.num_qubits)],
            "rz": [Parameter(f"θ_rz_{i}") for i in range(self.num_qubits)],
        }
        self.parameter_values: dict[str, float] = {f"θ_rx_{i}": 0.0 for i in range(self.num_qubits)}
        self.parameter_values.update({f"θ_ry_{i}": 0.0 for i in range(self.num_qubits)})
        self.parameter_values.update({f"θ_rz_{i}": 0.0 for i in range(self.num_qubits)})

        self.latest_input_parameters: dict[str, dict[str, float]] | None = None
        self.latest_counts: dict[str, int] | None = None
        self.latest_action: ActionData | None = None
        self.latest_gradients: Any = None
        self.latest_learning_rate: Any = None
        self.latest_updated_parameters: Any = None
        self.latest_temperature: Any = None

        self.history_params: list[Any] = []
        self.history_input_parameters: list[Any] = []
        self.history_counts: list[Any] = []
        self.history_actions: list[Any] = []

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

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
    ) -> dict:
        """
        Run the quantum brain simulation for the given parameters.

        Args:
            params: BrainParams for the agent/environment state.
            reward: Optional reward (unused).
            input_data: Optional input data (unused).

        Returns
        -------
            Measurement counts from the quantum circuit.
        """
        # Extract features for each module
        input_params = {
            module: extract_features_for_module(module, params, satiety=self.satiety)
            for module in self.modules
        }
        self.latest_input_parameters = input_params
        self.history_input_parameters.append(input_params)

        qc = self.build_brain(input_params)

        # Bind parameter values to the circuit
        bound_qc = qc.assign_parameters(self.parameter_values, inplace=False)
        simulator = AerSimulator(device=self.device)
        transpiled = transpile(bound_qc, simulator)

        result = simulator.run(transpiled, shots=self.shots).result()
        counts = result.get_counts()

        self.latest_counts = counts
        self.history_counts.append(counts)

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
        qc = self.build_brain(self.latest_input_parameters)
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

        return new_brain
