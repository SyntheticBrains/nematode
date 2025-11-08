# 🪱 Quantum Nematode

[![Pre-commit](https://github.com/SyntheticBrains/nematode/workflows/Pre-commit/badge.svg)](https://github.com/SyntheticBrains/nematode/actions/workflows/pre-commit.yml)
[![Tests](https://github.com/SyntheticBrains/nematode/workflows/Tests/badge.svg)](https://github.com/SyntheticBrains/nematode/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/SyntheticBrains/nematode/branch/main/graph/badge.svg)](https://codecov.io/gh/SyntheticBrains/nematode)

<p align="center">
  <img src="./docs/assets/images/demo-dynamic.gif" alt="nematode simulation demo" />
</p>

This project simulates a simplified nematode (C. elegans) navigating dynamic foraging environments to find food while managing satiety, using either a **quantum variational circuit** or a **classical neural network** as its decision-making brain. It leverages [Qiskit](https://qiskit.org) to simulate quantum behavior and integrates classical logic for realistic foraging dynamics.

## 🧪 Features

- ✅ **Dynamic Foraging Environment**: Realistic multi-food foraging with satiety management and distance efficiency tracking
- ✅ **Modular Quantum Brain**: Parameterized quantum circuits with 2+ qubits for decision-making
- ✅ **Classical ML Alternatives**: MLP, Q-learning, and spiking neural network brain architectures
- ✅ **Static Maze Environment**: Traditional 2D grid navigation with single-goal seeking
- ✅ **Quantum Learning**: Parameter-shift rule for gradient-based optimization
- ✅ **Hardware Support**: Classical simulation (AerSimulator) and real quantum hardware (IBM QPU)
- ✅ **Comprehensive Tracking**: Per-run and session-level metrics, plots, and CSV exports
- ✅ **Interactive Workflows**: CLI scripts with flexible configuration
- 🚧 **Expandable Framework**: Modular design for research and experimentation

## 🧠 Brain Architectures

Choose from multiple brain architectures for your nematode:

- **ModularBrain**: Quantum variational circuit with modular sensory processing
- **QModularBrain**: Hybrid quantum-classical Q-learning with experience replay
- **MLPBrain**: Classical multi-layer perceptron with policy gradients (REINFORCE)
- **QMLPBrain**: Classical MLP with Deep Q-Network (DQN) learning
- **SpikingBrain**: Biologically realistic spiking neural network with LIF neurons and STDP learning

Select the brain architecture when running simulations:

```bash
python scripts/run_simulation.py --brain modular    # Quantum (default)
python scripts/run_simulation.py --brain qmodular  # Hybrid quantum-classical
python scripts/run_simulation.py --brain mlp       # Classical policy gradient
python scripts/run_simulation.py --brain qmlp      # Classical Q-learning
python scripts/run_simulation.py --brain spiking   # Biologically realistic
```

## 🚀 Quick Start

### 1. Install Dependencies

Install [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
brew install uv
```

Install the project (choose one based on your needs):

```bash
# For CPU simulation (recommended for beginners)
uv sync --extra cpu --extra torch

# For quantum hardware access (requires IBM Quantum account)
uv sync --extra qpu

# For GPU acceleration (local installation)
uv sync --extra gpu --extra torch

# For GPU acceleration (Docker with NVIDIA GPU support)
docker compose up --build
```

> **Docker GPU Requirements**: For the Docker setup, you need Docker with NVIDIA Container Toolkit installed for GPU acceleration.

### 2. Configure Environment (Optional)

If using quantum hardware, set up your IBM Quantum API key:

```bash
cp .env.template .env
# Edit .env to add your IBM_QUANTUM_API_KEY
```

### 3. Run a Simulation

**Command Line Examples:**

```bash
# Dynamic foraging with quantum modular brain (recommended)
uv run ./scripts/run_simulation.py --log-level DEBUG --show-last-frame-only --track-per-run --runs 20 --config ./configs/examples/modular_dynamic_medium.yml --theme emoji

# Dynamic foraging with classical MLP brain
uv run ./scripts/run_simulation.py --log-level DEBUG --show-last-frame-only --track-per-run --runs 20 --config ./configs/examples/mlp_dynamic_medium.yml --theme emoji

# Static maze with quantum modular brain
uv run ./scripts/run_simulation.py --log-level DEBUG --show-last-frame-only --track-per-run --runs 20 --config ./configs/examples/modular_simple_medium.yml --theme emoji

# Spiking neural network brain
uv run ./scripts/run_simulation.py --log-level DEBUG --show-last-frame-only --track-per-run --runs 20 --config ./configs/examples/spiking_simple_medium.yml --theme emoji

# Quantum hardware (IBM QPU) with dynamic foraging
uv run ./scripts/run_simulation.py --log-level DEBUG --show-last-frame-only --track-per-run --runs 1 --config ./configs/examples/modular_dynamic_small.yml --theme emoji --device qpu

# Many-worlds quantum simulation
uv run ./scripts/run_simulation.py --log-level WARNING --show-last-frame-only --track-per-run --runs 1 --config ./configs/examples/modular_simple_medium.yml --theme emoji --manyworlds
```

**Docker GPU Examples:**

```bash
# Run dynamic foraging with MLP brain and GPU acceleration
docker-compose exec quantum-nematode uv run ./scripts/run_simulation.py --log-level DEBUG --show-last-frame-only --track-per-run --runs 20 --config ./configs/examples/mlp_dynamic_medium.yml --theme emoji

# Interactive Docker shell for development
docker-compose exec quantum-nematode bash
```

## � How It Works

### Dynamic Foraging Environment (Primary)

1. **State Perception**: The nematode perceives its environment through a viewport (distance to nearest food, gradient information, satiety level)
2. **Brain Processing**: The selected brain architecture processes the state
3. **Action Selection**: Brain outputs action probabilities (forward, left, right, stay)
4. **Environment Update**: Agent moves, satiety decays, and receives reward signal
5. **Food Collection**: When reaching food, satiety is restored and new food spawns
6. **Learning**: Brain parameters are updated based on reward feedback
7. **Repeat**: Process continues until all foods are collected, satiety reaches zero (starvation), or maximum steps reached

### Static Maze Environment (Legacy)

The traditional single-goal navigation follows a simpler loop without satiety management, terminating when the goal is reached or maximum steps are exceeded.

### Quantum Learning Process

For quantum brains, the learning process uses:
- **Quantum Feature Encoding**: Environmental data encoded as qubit rotations
- **Parameterized Quantum Circuits**: Trainable quantum gates for decision-making
- **Parameter-Shift Rule**: Quantum gradient computation for optimization
- **Entanglement**: Quantum correlations between different sensory modules

### Spiking Neural Network

The spiking brain architecture provides biologically realistic neural computation:

- **Leaky Integrate-and-Fire (LIF) Neurons**: Membrane potential dynamics with spike generation
- **Spike-Timing-Dependent Plasticity (STDP)**: Temporal Hebbian learning rule
- **Rate Coding**: Environmental input encoding as spike frequency patterns
- **Reward Modulation**: STDP learning strength modulated by environmental rewards

**Key Features:**
- Biologically realistic temporal dynamics
- Event-driven sparse computation
- Plasticity rules based on spike timing
- Configurable neuron and synapse parameters
```

## 📊 Example Output

The simulation provides real-time visualization of the nematode's navigation:

```
⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️
⬜️ 🦠 ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️
⬜️ 🔼 ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️
⬜️ 🔵 ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️
⬜️ 🔵 ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️
⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️
⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️
⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️
⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️
⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️ ⬜️

Run:          10/10
Steps(Avg):   35.89/10
Step:         20/300
Wins:         10

Session ID: 20250101_000000
All runs completed:
Run 1: 28 steps    Run 6: 23 steps
Run 2: 44 steps    Run 7: 40 steps
Run 3: 21 steps    Run 8: 30 steps
Run 4: 66 steps    Run 9: 29 steps
Run 5: 33 steps    Run 10: 23 steps

Average steps per run: 33.70
Average efficiency score: -19.70
Improvement metric (steps): 17.86%
Success rate: 100.00%
```

Where:
- 🦠 = Nematode agent
- 🔼 = Food target
- 🔵 = Agent's trail/body
- ⬜️ = Empty space

## 🧰 Built With

- **[Qiskit](https://qiskit.org/)**: Quantum computing framework
- **[PyTorch](https://pytorch.org/)**: Classical neural networks
- **[uv](https://github.com/astral-sh/uv)**: Modern Python dependency management
- **[Pydantic](https://pydantic.dev/)**: Data validation and settings
- **[Rich](https://rich.readthedocs.io/)**: Beautiful terminal output

## 🔬 Research Applications

This project serves as a platform for exploring:

- **Quantum Machine Learning**: Investigating quantum advantages in learning tasks
- **Biological Modeling**: Simplified models of neural decision-making
- **Hybrid Algorithms**: Combining quantum and classical computation
- **NISQ Applications**: Near-term quantum computing applications

## 🗺️ Roadmap

### Upcoming Features

- **Enhanced Brain Architectures**: More sophisticated quantum learning algorithms
- **Extended Foraging Dynamics**: Temperature gradients, food quality variations, and social foraging
- **Multi-Agent Scenarios**: Cooperative and competitive foraging behaviors
- **Better Visualization**: Real-time learning analysis and 3D environment rendering
- **Hardware Optimization**: Circuit compilation and quantum error mitigation
- **Research Tools**: Advanced analysis and experimentation capabilities

### Research Applications

This platform enables research in:
- Quantum advantages in reinforcement learning and foraging tasks
- Bio-inspired quantum algorithms for decision-making
- Hybrid quantum-classical computation in realistic environments
- Near-term quantum device applications (NISQ algorithms)
- Comparative analysis of quantum vs classical learning in complex environments

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for complete development setup instructions, code style guidelines, testing procedures, and pull request process.

### Areas We Need Help With

- **Quantum Algorithm Development**: New quantum learning techniques for foraging
- **Foraging Environment Extensions**: Social behaviors, food quality, temperature gradients
- **Multi-Agent Scenarios**: Cooperative and competitive foraging dynamics
- **Visualization Tools**: Real-time learning analysis and environment rendering
- **Documentation**: Tutorials and examples for dynamic environments
- **Testing**: Performance benchmarks and foraging strategy analysis

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **[Q-CTRL](https://q-ctrl.com/)**: For providing quantum hardware access with Fire Opal performance management tools to suppress quantum hardware errors and optimize quantum circuits
- **[OpenSpec](https://github.com/Fission-AI/OpenSpec)**: For providing the OpenSpec framework for structured, spec-driven AI development
- **C. elegans Research Community**: For inspiring this computational model
- **Qiskit Team**: For providing excellent quantum computing tools
- **Quantum ML Community**: For advancing the field of quantum machine learning
