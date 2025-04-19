import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit import Parameter
from quantumnematode.logging_config import logger


parameter_values = {
    "θx": 0.0,
    "θy": 0.0,
    "θz": 0.0,
    "θentangle": 0.0,
}


def update_parameters(parameters, gradients, learning_rate=0.1):
    """Update quantum circuit parameter values based on gradients."""
    for param, grad in zip(parameters, gradients):
        param_name = param.name
        if param_name in parameter_values:
            parameter_values[param_name] -= learning_rate * grad

    logger.debug(
        f"Updated parameters: {parameter_values}"
    )



def compute_gradients(counts, reward):
    """Compute gradients based on counts and reward."""
    # Normalize counts to probabilities
    total_shots = sum(counts.values())
    probabilities = {key: value / total_shots for key, value in counts.items()}

    # Define a simple gradient computation based on reward and probabilities
    gradients = []
    for key in ["00", "01", "10", "11"]:
        probability = probabilities.get(key, 0)
        gradient = reward * (1 - probability)  # Encourage actions with higher rewards
        gradients.append(gradient)

    return gradients


def build_brain():
    theta_x = Parameter("θx")
    theta_y = Parameter("θy")
    theta_z = Parameter("θz")
    theta_entangle = Parameter("θentangle")  # Parameter for entanglement
    qc = QuantumCircuit(2, 2)
    qc.rx(theta_x, 0)  # Rotate qubit 0 around the X-axis
    qc.ry(theta_y, 1)  # Rotate qubit 1 around the Y-axis
    qc.rz(theta_z, 0)  # Rotate qubit 0 around the Z-axis
    qc.cx(0, 1)  # Add entanglement
    qc.ry(theta_entangle, 1)  # Add parameterized rotation for entanglement
    qc.measure([0, 1], [0, 1])  # Measure both qubits
    return qc, theta_x, theta_y, theta_z, theta_entangle


def run_brain(dx, dy, grid_size, reward=None):
    qc, theta_x, theta_y, theta_z, theta_entangle = build_brain()

    # Use stored parameter values
    input_x = (
        parameter_values["θx"]
        + dx / (grid_size - 1) * np.pi
        + np.random.uniform(-0.1, 0.1)
    )
    input_y = (
        parameter_values["θy"]
        + dy / (grid_size - 1) * np.pi
        + np.random.uniform(-0.1, 0.1)
    )
    input_z = parameter_values["θz"] + np.random.uniform(0, 2 * np.pi)
    input_entangle = parameter_values["θentangle"] + np.random.uniform(
        0, 2 * np.pi
    )

    logger.debug(
        f"dx={dx}, dy={dy}, input_x={input_x}, input_y={input_y}, input_z={input_z}, input_entangle={input_entangle}"
    )

    bound_qc = qc.assign_parameters(
        {
            theta_x: input_x,
            theta_y: input_y,
            theta_z: input_z,
            theta_entangle: input_entangle,
        },
        inplace=False,
    )

    simulator = Aer.get_backend("aer_simulator")
    transpiled = transpile(bound_qc, simulator)
    result = simulator.run(transpiled, shots=1024).result()
    counts = result.get_counts()

    logger.debug(f"Counts: {counts}")

    # If reward is provided, update parameters
    if reward is not None:
        gradients = compute_gradients(
            counts, reward
        )
        update_parameters([theta_x, theta_y, theta_z, theta_entangle], gradients)

    return counts


def interpret_counts(counts, agent_pos, grid_size):
    # Sort counts by frequency
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    logger.debug(f"Sorted counts: {sorted_counts}")

    # Map the quantum output to valid actions dynamically
    valid_action_map = {}
    if agent_pos[1] < grid_size - 1:  # Can move up
        valid_action_map["00"] = "up"
    if agent_pos[1] > 0:  # Can move down
        valid_action_map["01"] = "down"
    if agent_pos[0] < grid_size - 1:  # Can move right
        valid_action_map["11"] = "right"
    if agent_pos[0] > 0:  # Can move left
        valid_action_map["10"] = "left"

    # Select the most common result or randomly choose among ties
    top_results = [
        result for result, count in sorted_counts if count == sorted_counts[0][1]
    ]
    most_common = np.random.choice(top_results)

    # Map the result to an action
    action = valid_action_map.get(most_common, None)

    return action
