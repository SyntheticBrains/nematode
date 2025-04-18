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

## 🤝 Contributing

PRs welcome! To extend:

- Add more sophisticated brain logic
- Integrate Qiskit runtime or real IBM Quantum backends
- Create visualization tools for maze traversal

---

## 🧬 License

Unlicensed
