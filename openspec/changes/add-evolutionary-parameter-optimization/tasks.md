# Tasks: Add Evolutionary Parameter Optimization

## 1. Core Infrastructure

- [ ] 1.1 Add `cma` dependency to pyproject.toml
- [ ] 1.2 Create `quantumnematode/optimizers/evolutionary.py` module
- [ ] 1.3 Implement `EvolutionaryOptimizer` base class with fitness evaluation interface
- [ ] 1.4 Implement `CMAESOptimizer` wrapper for cma library
- [ ] 1.5 Implement `GeneticAlgorithmOptimizer` for simple GA alternative

## 2. Brain Integration

- [ ] 2.1 Ensure `brain.parameter_values` property returns flat parameter dict
- [ ] 2.2 Add `brain.set_parameters(param_array)` method for setting from flat array
- [ ] 2.3 Add `brain.copy()` method for creating independent brain instances (verify existing)
- [ ] 2.4 Verify brain state reset between fitness evaluations

## 3. Fitness Evaluation

- [ ] 3.1 Create `FitnessFunction` class with configurable episode count
- [ ] 3.2 Implement episode runner that returns success rate
- [ ] 3.3 Add support for parallel fitness evaluation (multiprocessing)
- [ ] 3.4 Implement fitness aggregation (mean success rate, optional variance penalty)

## 4. Evolution Script

- [ ] 4.1 Create `scripts/run_evolution.py` main script
- [ ] 4.2 Implement CLI arguments (config, population size, generations, etc.)
- [ ] 4.3 Add logging for generation progress (best fitness, mean, std)
- [ ] 4.4 Save best parameters to file after evolution
- [ ] 4.5 Add checkpoint/resume capability for long runs

## 5. Configuration

- [ ] 5.1 Extend YAML schema for evolution parameters
- [ ] 5.2 Create example config `configs/examples/evolution_cmaes_2qubit.yml`
- [ ] 5.3 Create example config `configs/examples/evolution_ga_2qubit.yml`
- [ ] 5.4 Document configuration options

## 6. Classical Baseline

- [ ] 6.1 Implement `LinearClassicalBrain` with 12 parameters (matching quantum)
- [ ] 6.2 Add same parameter interface (get/set/copy)
- [ ] 6.3 Create evolution config for classical baseline comparison
- [ ] 6.4 Document comparison methodology

## 7. Testing

- [ ] 7.1 Unit tests for EvolutionaryOptimizer base class
- [ ] 7.2 Unit tests for CMAESOptimizer
- [ ] 7.3 Unit tests for GeneticAlgorithmOptimizer
- [ ] 7.4 Unit tests for FitnessFunction
- [ ] 7.5 Integration test with small evolution run

## 8. Documentation

- [ ] 8.1 Add docstrings to all new classes and functions
- [ ] 8.2 Update documentation explaining how evolution workflow works and how to use

## 9. Validation

- [ ] 9.1 Run evolution on 2-qubit quantum brain (50+ generations)
- [ ] 9.2 Compare evolved parameters to best known manual parameters
- [ ] 9.3 Run evolution on classical baseline
- [ ] 9.4 Document quantum vs classical comparison results
