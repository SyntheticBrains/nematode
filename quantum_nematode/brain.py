import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit import Parameter


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


def run_brain(dx, dy, grid_size=5):
    qc, θx, θy, θz, θentangle = build_brain()
    input_x = dx / (grid_size - 1) * np.pi + np.random.uniform(-0.1, 0.1)
    input_y = dy / (grid_size - 1) * np.pi + np.random.uniform(-0.1, 0.1)
    input_z = np.random.uniform(0, 2 * np.pi)  # Random value for theta_z
    input_entangle = np.random.uniform(0, 2 * np.pi)  # Random value for entanglement

    # Debug: Print parameter values
    print(
        f"dx={dx}, dy={dy}, input_x={input_x}, input_y={input_y}, input_z={input_z}, input_entangle={input_entangle}"
    )

    bound_qc = qc.assign_parameters(
        {θx: input_x, θy: input_y, θz: input_z, θentangle: input_entangle},
        inplace=False,
    )

    simulator = Aer.get_backend("aer_simulator")
    transpiled = transpile(bound_qc, simulator)
    result = simulator.run(transpiled, shots=1024).result()
    counts = result.get_counts()

    # Debug: Print the counts
    print(f"Counts: {counts}")

    return counts


def interpret_counts(counts, agent_pos, grid_size):
    action_map = {"00": "up", "01": "down", "10": "left", "11": "right"}

    # Sort counts by frequency
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Debug: Print sorted counts
    print(f"Sorted counts: {sorted_counts}")

    # Select the most common result or randomly choose among ties
    # TODO: Avoid randomly choosing among ties
    top_results = [
        result for result, count in sorted_counts if count == sorted_counts[0][1]
    ]
    most_common = np.random.choice(top_results)

    # Map the result to an action
    action = action_map.get(most_common, "up")

    # Validate the action based on the agent's position
    valid_actions = []
    if agent_pos[1] < grid_size - 1:  # Can move up
        valid_actions.append("up")
    if agent_pos[1] > 0:  # Can move down
        valid_actions.append("down")
    if agent_pos[0] < grid_size - 1:  # Can move right
        valid_actions.append("right")
    if agent_pos[0] > 0:  # Can move left
        valid_actions.append("left")

    # If the selected action is invalid, choose a random valid action
    if action not in valid_actions:
        print(f"Invalid action: {action}, selecting a random valid action.")
        action = np.random.choice(valid_actions)

    return action
