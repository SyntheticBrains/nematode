# Contributing to Quantum Nematode Simulation

Thank you for your interest in contributing to the Quantum Nematode Simulation project! This guide will help you get started with development and contributing to the project.

## 🚀 Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management

### 1. Install uv

```bash
brew install uv
```

### 2. Clone and Setup

```bash
git clone https://github.com/SyntheticBrains/nematode.git
cd nematode
```

### 3. Install Dependencies

Choose the appropriate installation based on your development needs:

#### For Quantum Hardware Development (QPU)
```bash
uv sync --extra qpu
```

#### For CPU-only Development
```bash
uv sync --extra cpu
```

#### For GPU Development
```bash
uv sync --extra gpu
```

#### For Classical ML Brain Development
```bash
uv sync [OTHER_EXTRAS] --extra torch
```

> ⚠️ **Note**: Only the `cpu` and `qpu` extras conflict and cannot be installed together.

### 4. Environment Configuration

Copy the environment template and configure your settings:

```bash
cp .env.template .env
```

Edit `.env` to include your API keys and configuration:

```env
IBM_QUANTUM_API_KEY=your-ibm-quantum-api-key-here
IBM_QUANTUM_BACKEND=ibm_brisbane  # Optional: specify backend
```

## 🛠️ Development Tools

### Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to maintain code quality. Install the hooks:

```bash
uv run pre-commit install
```

Run checks manually:

```bash
uv run pre-commit run -a
```

### Code Quality Tools

The project uses several tools for code quality:

- **Ruff**: Fast Python linter and formatter
- **Pyright**: Static type checker
- **Pytest**: Testing framework

Configuration is in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ['ALL']
ignore = ['TD002', 'TD003']

[tool.ruff.lint.pydocstyle]
convention = "numpy"
```

## 🧠 Architecture Overview

### Brain Architectures

The project supports multiple brain architectures:

1. **ModularBrain**: Quantum variational circuit with modular design
2. **QModularBrain**: Hybrid quantum-classical Q-learning
3. **MLPBrain**: Classical MLP with policy gradients
4. **QMLPBrain**: Classical MLP with Q-learning

Each brain architecture follows a common interface defined in `quantumnematode.brain.arch`.

## 🔧 Development Workflows

### Running Tests

```bash
uv run pytest
```

### Running Simulations

#### Development Testing

##### Testing with Dynamic Foraging Environment

```bash
# Quick test with small dynamic environment
uv run ./scripts/run_simulation.py --runs 5 --config ./configs/examples/modular_dynamic_small.yml --theme emoji

# Full test with medium environment
uv run ./scripts/run_simulation.py --runs 20 --config ./configs/examples/modular_dynamic_medium.yml --theme emoji
```

##### Testing with Static Environment

```bash
uv run ./scripts/run_simulation.py --runs 10 --config ./configs/examples/modular_simple_medium.yml --theme emoji
```

### Adding New Features

#### Adding a New Brain Architecture

1. Create a new file in `packages/quantum-nematode/quantumnematode/brain/arch/`
2. Inherit from appropriate base class (`QuantumBrain` or `ClassicalBrain`)
3. Implement required methods:
   - `run_brain()`: Execute brain and return actions
   - `learn()`: Update parameters based on rewards
   - `update_parameters()`: Low-level parameter updates

Example structure:
```python
from quantumnematode.brain.arch import QuantumBrain

class MyNewBrain(QuantumBrain):
    def run_brain(self, params, reward=None, **kwargs):
        # Implement brain execution logic
        pass
    
    def learn(self, params, reward, **kwargs):
        # Implement learning logic
        pass
```

4. Add configuration class:
```python
from quantumnematode.brain.arch.dtypes import BrainConfig

class MyNewBrainConfig(BrainConfig):
    # Define configuration parameters
    pass
```

5. Update `__init__.py` files to export new classes
6. Add tests in the appropriate test directory

#### Adding New Quantum Modules

1. Define module in `quantumnematode.brain.modules`
2. Add feature extraction logic
3. Update `DEFAULT_MODULES` mapping
4. Test with existing brain architectures

#### Adding New Environment Features

1. Extend `quantumnematode.env` classes (base classes: `StaticEnvironment`, `DynamicForagingEnvironment`)
2. Ensure compatibility with `BrainParams` interface
3. Add visualization support for new features
4. Update environment state encoding for brain input
5. Add tracking for new metrics in `EpisodeTracker`
6. Create corresponding plots and CSV exports

Example for adding a new foraging feature:
```python
# In quantumnematode/env/dynamic_foraging.py
class ExtendedForagingEnvironment(DynamicForagingEnvironment):
    def __init__(self, temperature_variation: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.temperature_variation = temperature_variation

    def get_state_vector(self) -> list[float]:
        state = super().get_state_vector()
        if self.temperature_variation:
            state.append(self.get_temperature_at_position())
        return state
```

### Code Style Guidelines

1. **Type Hints**: Use comprehensive type hints
   ```python
   def my_function(param: int, optional: str | None = None) -> list[float]:
       return [1.0, 2.0]
   ```

2. **Docstrings**: Use NumPy-style docstrings
   ```python
   def compute_gradient(params: dict[str, float]) -> list[float]:
       """
       Compute parameter gradients using parameter-shift rule.
       
       Parameters
       ----------
       params : dict[str, float]
           Parameter values for quantum circuit.
           
       Returns
       -------
       list[float]
           Computed gradients for each parameter.
       """
   ```

3. **Error Handling**: Use descriptive error messages
   ```python
   if not isinstance(params, dict):
       error_message = f"Expected dict for params, got {type(params)}"
       logger.error(error_message)
       raise TypeError(error_message)
   ```

4. **Logging**: Use structured logging
   ```python
   from quantumnematode.logging_config import logger
   
   logger.info(f"Training episode {episode} completed with reward {reward}")
   ```

## 🧪 Testing Guidelines

### Unit Tests

- Test individual functions and methods
- Use meaningful test names: `test_parameter_shift_gradients_with_valid_params`
- Mock external dependencies (Qiskit backends, etc.)

### Integration Tests

- Test complete workflows
- Use small configurations for faster execution
- Test both classical and quantum backends

### Performance Tests

- Benchmark critical paths
- Monitor memory usage for large circuits
- Test scalability with different qubit counts

## 🚀 Contribution Areas

### High Priority

1. **Dynamic Foraging Enhancements**
   - Temperature gradient integration
   - Food quality variations and preferences
   - Social foraging behaviors (multi-agent)
   - Realistic chemotaxis modeling

2. **Quantum Hardware Integration**
   - Add support for new quantum backends
   - Implement noise-aware training for foraging tasks
   - Add hardware-specific optimizations

3. **Advanced Learning Algorithms**
   - Quantum natural gradients for foraging
   - Meta-learning across environment variations
   - Transfer learning from simple to complex foraging

4. **Visualization and Analysis**
   - Real-time foraging trajectory visualization
   - Satiety and efficiency heatmaps
   - Comparative analysis tools (quantum vs classical)

### Medium Priority

1. **Environment Extensions**
   - Multi-agent cooperative foraging
   - Competitive foraging scenarios
   - Predator-prey dynamics
   - Continuous action spaces

2. **Monitoring and Logging**
   - Advanced foraging strategy detection
   - Performance metrics tracking per food type
   - Experiment management and reproducibility

### Documentation

1. **API Documentation**
   - Complete docstring coverage
   - Usage examples
   - Best practices guides

2. **Tutorials**
   - Getting started tutorials
   - Advanced usage patterns
   - Research applications

## 📝 Pull Request Process

1. **Fork and Branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Develop and Test**
   - Write tests for new functionality
   - Ensure all tests pass
   - Run pre-commit checks

3. **Documentation**
   - Update docstrings
   - Add usage examples
   - Update README if needed

4. **Submit PR**
   - Clear description of changes
   - Link to related issues
   - Include testing instructions

### PR Checklist

- [ ] All tests pass
- [ ] Pre-commit checks pass
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Backward compatibility maintained
- [ ] Performance impact considered

## 🔬 Research Directions

### Quantum Machine Learning in Foraging

- **Quantum Advantage in Foraging**: Identify scenarios where quantum algorithms excel at multi-objective foraging tasks
- **Noise Resilience**: Develop training methods robust to quantum hardware noise in dynamic environments
- **Hybrid Algorithms**: Explore quantum-classical hybrid approaches for foraging strategy optimization
- **Entanglement in Decision-Making**: Study role of quantum entanglement in balancing exploration vs exploitation

### Biological Modeling

- **Neural Modeling**: More accurate modeling of C. elegans chemosensory neurons and interneurons
- **Behavioral Patterns**: Implementation of realistic nematode foraging behaviors (area-restricted search, klinokinesis)
- **Multi-scale Modeling**: From molecular signaling to behavioral strategies
- **Satiety and Homeostasis**: Realistic modeling of internal state management

### Algorithm Development

- **Novel Quantum Algorithms**: Development of new quantum learning algorithms for sequential decision-making
- **Optimization Techniques**: Advanced parameter optimization methods for foraging
- **Scalability**: Methods for larger quantum systems and complex environments
- **Transfer Learning**: Strategies for adapting from simple to complex foraging scenarios

## 🤝 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: Direct contact with maintainers

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

Thank you for contributing to the Quantum Nematode Simulation project! Your contributions help advance the field of quantum machine learning and computational biology.
