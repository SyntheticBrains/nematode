# 🧠 Quantum Nematode Simulation

This project simulates a simplified nematode (C. elegans) navigating a 2D grid maze to find food, using either a **quantum variational circuit** or a **classical neural network** as its decision-making brain. It leverages [Qiskit](https://qiskit.org) to simulate quantum behavior and integrates classical logic for the environment.

---

## 🧪 Features

- ✅ Modular quantum circuit (2+ qubits) that takes in state and outputs movement
- ✅ Classical MLP (multi-layer perceptron) brain option
- ✅ Classical grid-world environment
- ✅ Agent that navigates based on brain output
- ✅ Supports both CLI scripts and Jupyter notebook workflows
- 🚧 Expandable for training, hybrid QML, or hardware backends

### Expanded Details

#### Modular Quantum Brain

- The nematode's brain can be implemented as a parameterized quantum circuit with 2 or more qubits.
- The circuit uses RX, RY, and RZ gates to encode the agent's state and entanglement to model complex decision-making.
- Measurements on the qubits are mapped to one of four possible actions: forward, left, right, or stay.

#### Classical MLP Brain

- Alternatively, the agent can use a classical multi-layer perceptron (MLP) policy network for decision-making.
- The MLP processes the agent's state and selects actions based on learned policies.
- Supports GPU acceleration via PyTorch.

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

## 🧠 Brain Architectures

This project now supports two brain architectures:

- **ModularBrain**: A modular quantum brain using parameterized circuits for decision-making.
- **MLPBrain**: A classical multi-layer perceptron (MLP) brain for policy-based action selection.

### How to Choose a Brain Architecture

You can select the brain architecture when running the simulation using the `--brain` argument:

```bash
python scripts/run_simulation.py --brain modular
python scripts/run_simulation.py --brain mlp
```

The default architecture is `modular` (quantum). Use `mlp` for a classical neural network agent.

---

## 🚀 Getting Started

### 1. Install [uv](https://github.com/astral-sh/uv)

```bash
brew install uv
```

### 2. Install dependencies

You can install the project with either CPU or GPU support.

Install with QPU support:

```bash
uv sync --extra qpu
```

Install with CPU support:

```bash
uv sync --extra cpu
```

Install with GPU support:

```bash
uv sync --extra gpu
```

To run classical ML brain architectures, you will also need to install `torch` as an extra:

```bash
uv sync [OTHER_EXTRAS] --extra torch
```

> ℹ️ Only the `cpu` and `qpu` extras conflict and cannot be installed together.

### 3. Set up environment variables

Copy the provided `.env.template` file to `.env` and edit it to include your required environment variables (such as your IBM Quantum API key):

```bash
cp .env.template .env
```

Open `.env` in your favorite editor and set the values as needed, for example:

```
IBM_QUANTUM_API_KEY=your-ibm-quantum-api-key-here
```

This file will be loaded automatically when running the simulation.

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
- The selected brain (quantum or classical) processes the state and outputs an action.
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
- [PyTorch](https://pytorch.org/) (for MLPBrain)
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
uv run pre-commit run -a
```

---

## 🧬 License

Unlicensed
