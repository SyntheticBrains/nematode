# Tasks: Add Evolutionary Parameter Optimization

## 1. Core Infrastructure

- [x] 1.1 Add `cma` dependency to pyproject.toml
- [x] 1.2 Create `quantumnematode/optimizers/evolutionary.py` module
- [x] 1.3 Implement `EvolutionaryOptimizer` base class with fitness evaluation interface
- [x] 1.4 Implement `CMAESOptimizer` wrapper for cma library
- [x] 1.5 Implement `GeneticAlgorithmOptimizer` for simple GA alternative

## 2. Brain Integration

- [x] 2.1 Ensure `brain.parameter_values` property returns flat parameter dict
- [x] 2.2 Add `brain.set_parameters(param_array)` method for setting from flat array
- [x] 2.3 Add `brain.copy()` method for creating independent brain instances (verify existing)
- [x] 2.4 Verify brain state reset between fitness evaluations

## 3. Fitness Evaluation

- [x] 3.1 Create `FitnessFunction` class with configurable episode count
- [x] 3.2 Implement episode runner that returns success rate
- [x] 3.3 Add support for parallel fitness evaluation (multiprocessing)
- [x] 3.4 Implement fitness aggregation (mean success rate, optional variance penalty)

## 4. Evolution Script

- [x] 4.1 Create `scripts/run_evolution.py` main script
- [x] 4.2 Implement CLI arguments (config, population size, generations, etc.)
- [x] 4.3 Add logging for generation progress (best fitness, mean, std)
- [x] 4.4 Save best parameters to file after evolution
- [x] 4.5 Add checkpoint/resume capability for long runs

## 5. Configuration

- [x] 5.1 Extend YAML schema for evolution parameters
- [x] 5.2 Create example config `configs/examples/evolution_modular_foraging_small.yml`
- [x] 5.3 Create example config `configs/examples/evolution_modular_predators_small.yml`
- [x] 5.4 Document configuration options in CONTRIBUTING.md

## 6. Classical Baseline

- [ ] 6.1 Implement `LinearClassicalBrain` with 12 parameters (matching quantum)
- [ ] 6.2 Add same parameter interface (get/set/copy)
- [ ] 6.3 Create evolution config for classical baseline comparison
- [ ] 6.4 Document comparison methodology

## 7. Testing

- [x] 7.1 Unit tests for EvolutionaryOptimizer base class
- [x] 7.2 Unit tests for CMAESOptimizer
- [x] 7.3 Unit tests for GeneticAlgorithmOptimizer
- [x] 7.4 Unit tests for FitnessFunction
- [ ] 7.5 Integration test with small evolution run

## 8. Documentation

- [x] 8.1 Add docstrings to all new classes and functions
- [x] 8.2 Update documentation explaining how evolution workflow works and how to use (CONTRIBUTING.md)

## 9. Validation

- [x] 9.1 Run evolution on 2-qubit quantum brain (50+ generations) - CMA-ES 75 gen, GA 150 gen
- [x] 9.2 Compare evolved parameters to best known manual parameters - 80-88% vs 22.5% (3.5-4x improvement)
- [ ] 9.3 Run evolution on classical baseline
- [ ] 9.4 Document quantum vs classical comparison results
