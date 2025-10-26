# Project Context

## Purpose
Quantum Nematode is a research platform that simulates a simplified nematode (C. elegans) navigating a 2D grid maze to find food using either quantum variational circuits or classical neural networks as its decision-making brain. The project serves as a platform for exploring quantum machine learning, biological modeling, hybrid algorithms, and NISQ (Near-term Intermediate-Scale Quantum) applications.

## Tech Stack
- **Python 3.12+** - Core language with strict version constraint
- **Qiskit** - Quantum computing framework for quantum circuits and simulation
- **PyTorch** - Classical neural networks and machine learning
- **uv** - Modern Python dependency management and workspace tooling
- **Pydantic** - Data validation and configuration management
- **Rich** - Beautiful terminal output and visualization
- **Matplotlib** - Data visualization and plotting
- **NumPy** - Numerical computing foundation
- **Docker** - Containerization with GPU support via NVIDIA Container Toolkit

## Project Conventions

### Code Style
- **Linting**: Ruff with comprehensive rule selection (ALL rules except TD002, TD003)
- **Type Checking**: Pyright with strict configuration and error reporting
- **Line Length**: 100 characters maximum
- **Docstring Style**: NumPy convention for documentation
- **Python Version**: Strict 3.12+ requirement for modern language features
- **Import Organization**: Explicit __all__ exports in modules

### Architecture Patterns
- **Modular Brain Architecture**: Pluggable brain implementations (ModularBrain, QModularBrain, MLPBrain, QMLPBrain)
- **Workspace Structure**: UV workspace with packages/quantum-nematode as the core package
- **Configuration-Driven**: YAML-based simulation configurations
- **Quantum-Classical Hybrid**: Support for both quantum circuits and classical ML approaches
- **Environment Abstraction**: Separate environment simulation from brain logic
- **Executor Pattern**: Different execution backends (CPU, GPU, QPU)

### Testing Strategy
- **pytest** as the testing framework
- Test exclusions for annotations, arguments, docstrings in test files
- Separate test directories maintaining package structure
- Support for different execution environments (CPU, GPU, QPU testing)

### Git Workflow
- **Apache 2.0 License** - Open source with proper attribution requirements
- **GitHub Repository**: SyntheticBrains/nematode organization
- **Conventional Structure**: Standard Python package layout with clear separation

## Domain Context
- **C. elegans Biology**: Nematode navigation based on chemotaxis, thermotaxis, tactile feedback, and learning
- **Quantum Computing**: Variational quantum circuits, parameter-shift rule optimization, quantum feature encoding
- **Reinforcement Learning**: Policy gradients (REINFORCE), Deep Q-Networks (DQN), experience replay
- **Grid World Environments**: 2D maze navigation with food-seeking behavior and reward systems
- **Quantum-Classical Hybrid Learning**: Combining quantum circuits with classical optimization techniques

## Important Constraints
- **Python Version**: Strict requirement for Python 3.12+ (no 3.13+ support yet)
- **Quantum Hardware**: IBM Quantum API key required for QPU execution
- **GPU Support**: NVIDIA Container Toolkit required for Docker GPU acceleration
- **Memory Considerations**: Quantum simulation can be memory-intensive for larger circuits
- **NISQ Limitations**: Near-term quantum devices have limited qubit counts and high error rates
- **Research Focus**: Private package (Do Not Upload classifier) - research/experimental codebase

## External Dependencies
- **IBM Quantum Platform**: Real quantum hardware access via qiskit-ibm-runtime
- **Q-CTRL Fire Opal**: Quantum error suppression and circuit optimization tools
- **Qiskit Aer**: High-performance quantum circuit simulation
- **Docker Hub**: Container registry for GPU-enabled execution environments
- **Pre-commit Hooks**: Code quality automation and validation
