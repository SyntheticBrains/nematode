# 🧠 Quantum Nematode Simulation

This project simulates a simplified nematode (C. elegans) navigating a 2D grid maze to find food, using a **quantum variational circuit** as its decision-making brain. It leverages [Qiskit](https://qiskit.org) to simulate quantum behavior and integrates classical logic for the environment.

---

## 📦 Project Structure

```
quantum_nematode/
├── quantum_nematode/
│   ├── agent.py           # Core simulation logic combining brain + environment
│   ├── brain.py           # Quantum circuit logic for nematode decision-making
│   └── env.py             # Simple grid-based environment for navigation
├── notebooks/
│   └── simulate.ipynb     # Lightweight notebook that runs a simulation
├── pyproject.toml         # Project config for uv / PEP 621-based tooling
├── README.md              # You're here!
└── .gitignore             # Standard ignore rules
```

---

## 🧪 Features

- ✅ Quantum circuit (2 qubits) that takes in state and outputs movement
- ✅ Classical grid-world environment
- ✅ Agent that navigates based on circuit output
- ✅ Supports both CLI scripts and Jupyter notebook workflows
- 🚧 Expandable for training, hybrid QML, or hardware backends

### Expanded Details

#### Quantum Circuit for Decision-Making

- The nematode's brain is implemented as a parameterized quantum circuit with 2 qubits.
- The circuit uses RX, RY, and RZ gates to encode the agent's state and entanglement to model complex decision-making.
- Measurements on the qubits are mapped to one of four possible actions: up, down, left, or right.

#### Classical Grid-World Environment

- A simple 2D grid maze serves as the nematode's environment.
- The agent starts at a random position and must navigate to a food source while avoiding collisions with its own body.
- The environment dynamically updates based on the agent's actions and provides feedback for learning.

#### Quantum Reinforcement Learning (QRL)

- A reward-based learning mechanism has been integrated to improve the agent's navigation efficiency.
- Gradients are computed based on the quantum circuit's output probabilities and the reward signal.
- The quantum circuit's parameters are updated iteratively to optimize the agent's performance.

#### Hybrid Workflows

- The project supports both CLI-based simulations and interactive Jupyter notebook workflows.
- Users can visualize the agent's path and learning process in real-time.

#### Expandability

- The framework is designed to be modular and extensible.
- Future enhancements could include integration with real quantum hardware, advanced quantum learning techniques, and visualization tools.

---

## 🧠 Quantum Brain Architectures

This project now supports multiple quantum brain architectures:

- **SimpleBrain**: A lightweight quantum brain using parameterized circuits for basic decision-making.
- **ComplexBrain**: A detailed quantum brain simulating 302 neurons, designed for real QPU testing.
- **ReducedBrain**: A scaled-down version of the complex brain using 30 qubits, optimized for simulators.

### How to Choose a Brain Architecture

You can select the brain architecture when running the simulation using the `--brain` argument:

```bash
python scripts/run_simulation.py --brain simple
python scripts/run_simulation.py --brain complex
python scripts/run_simulation.py --brain reduced
```

The default architecture is `simple`. Use `complex` for real QPU testing and `reduced` for simulator-friendly experiments.

---

## 🚀 Getting Started

### 1. Install [uv](https://github.com/astral-sh/uv)

```bash
brew install uv
```

### 2. Install dependencies

```bash
uv sync
```

---

## 📓 Running the Simulation

### Jupyter Notebook (Recommended for Exploration)

```bash
jupyter notebook notebooks/simulate.ipynb
```

### CLI (Python script version)

```bash
uv run ./scripts/run_simulation.py
```

> This script runs the simulation headlessly and logs the output.

---

## 🧠 How It Works

- The agent receives its state: distance (`dx`, `dy`) to food.
- A quantum circuit is created with parameterized RX/RY gates.
- Measurements on the 2-qubit output map to one of 4 actions: up, down, left, right.
- The environment updates the agent's position.
- The process repeats until the agent reaches the food or max steps are hit.

---

## 🧪 Example Output

```
Step 1: Action=right, Position=[1, 0]
Step 2: Action=right, Position=[2, 0]
...
Step 11: Action=up, Position=[4, 4]

Final path:
[(0, 0), (1, 0), ..., (4, 4)]
```

---

## 🧰 Tools Used

- [Qiskit](https://qiskit.org/)
- [Python 3.10+]
- [uv](https://github.com/astral-sh/uv) for modern dependency management
- Jupyter for notebook visualization

---

## 🗺️ Roadmap

### Planned Features

- **Advanced Quantum Learning**: Explore more sophisticated quantum learning techniques, such as Quantum Boltzmann Machines (QBM) and Quantum Memory Encoding.
- **Real Quantum Hardware Integration**: Transition from simulation to real quantum hardware for testing and validation.
- **Visualization Tools**: Develop tools to visualize the agent's learning process and decision-making.

### Other Possible Learning Approaches

#### Quantum Memory Encoding

- **Description**: Use quantum states to encode the nematode's memory of past actions or visited positions.
- **Implementation**:
  - Use quantum registers to store information about visited grid positions or actions.
  - Apply quantum superposition to explore multiple paths simultaneously.
  - Use quantum interference to reinforce paths that lead to food and suppress inefficient paths.

#### Quantum Grover Search for Pathfinding

- **Description**: Use Grover's algorithm to search for the shortest path to the food.
- **Implementation**:
  - Encode the maze as a quantum oracle.
  - Use Grover's search to find the optimal sequence of actions leading to the goal.

#### Quantum Boltzmann Machines (QBM)

- **Description**: Use QBMs to model the nematode's environment and learn optimal navigation strategies.
- **Implementation**:
  - Train a QBM to represent the probability distribution of successful paths.
  - Use the trained QBM to sample actions during navigation.

#### Quantum Amplitude Amplification for Action Selection

- **Description**: Use amplitude amplification to bias the nematode's action selection towards more promising actions.
- **Implementation**:
  - Encode action probabilities in quantum amplitudes.
  - Amplify actions that are more likely to lead to the goal based on past experience.

---

## 🤝 Contributing

PRs welcome! To extend:

- Add more sophisticated brain logic
- Integrate Qiskit runtime or real IBM Quantum backends
- Create visualization tools for maze traversal

### Pre-commit

We use [pre-commit](https://pre-commit.com/) to automate linting and code validation checks. Run the following to install the pre-commmit:

```sh
uv run pre-commit install
```

Use the following command to manually run the pre-commit checks:

```sh
poetry run pre-commit run -a
```

---

## 🧬 License

Unlicensed
