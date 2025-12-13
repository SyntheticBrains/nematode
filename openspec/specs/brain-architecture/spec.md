# brain-architecture Specification

## Purpose
TBD - created by archiving change add-spiking-neural-network-brain. Update Purpose after archive.
## Requirements
### Requirement: Spiking Neural Network Architecture Support
The system SHALL support a spiking neural network brain architecture as a third computational paradigm alongside quantum and classical approaches.

#### Scenario: CLI Brain Selection
**Given** a user wants to run a simulation with a spiking neural network brain  
**When** they execute `python scripts/run_simulation.py --brain spiking --config config.yml`  
**Then** the system SHALL initialize a SpikingBrain instance  
**And** the simulation SHALL proceed using spiking neural dynamics for decision-making  

#### Scenario: Configuration Loading
**Given** a configuration file specifies brain type as "spiking"  
**When** the configuration is loaded  
**Then** the system SHALL validate spiking-specific parameters  
**And** SHALL initialize the spiking brain with the specified neuron model and plasticity rules

### Requirement: Biological Neural Dynamics
The SpikingBrain SHALL implement biologically plausible neural dynamics using spiking neuron models.

#### Scenario: Leaky Integrate-and-Fire Dynamics
**Given** a LIF neuron receives input current  
**When** the membrane potential exceeds the threshold  
**Then** the neuron SHALL generate a spike  
**And** SHALL reset the membrane potential to the reset value  
**And** SHALL propagate the spike to connected neurons  

#### Scenario: Temporal Information Processing
**Given** environmental state information  
**When** encoded as spike trains  
**Then** the network SHALL process temporal patterns  
**And** SHALL maintain spike timing information for learning

### Requirement: Spike-Timing Dependent Plasticity Learning
The SpikingBrain SHALL implement STDP-based learning for synaptic weight adaptation.

#### Scenario: Pre-Post Spike Timing
**Given** a presynaptic spike occurs before a postsynaptic spike  
**When** the time difference is within the STDP window  
**Then** the synaptic weight SHALL be potentiated  
**And** the change SHALL be modulated by the reward signal  

#### Scenario: Post-Pre Spike Timing
**Given** a postsynaptic spike occurs before a presynaptic spike  
**When** the time difference is within the STDP window  
**Then** the synaptic weight SHALL be depressed  
**And** the change SHALL be modulated by the reward signal

### Requirement: Input/Output Encoding
The SpikingBrain SHALL convert between continuous environmental states and discrete spike patterns.

#### Scenario: Rate Coding Input
**Given** continuous state variables (gradient strength, direction)  
**When** encoding to spike trains  
**Then** higher values SHALL generate higher spike rates  
**And** SHALL maintain consistent encoding across time steps  

#### Scenario: Action Probability Decoding
**Given** output neuron spike counts over a decision period  
**When** decoding to action probabilities  
**Then** SHALL convert spike rates to softmax probabilities  
**And** SHALL ensure valid action selection

### Requirement: Protocol Compatibility
The SpikingBrain SHALL implement the ClassicalBrain protocol for seamless integration.

#### Scenario: Brain Interface Compliance
**Given** the existing brain architecture framework  
**When** SpikingBrain is instantiated  
**Then** it SHALL implement all required ClassicalBrain methods  
**And** SHALL return compatible data structures  
**And** SHALL integrate with existing simulation infrastructure  

#### Scenario: Configuration Schema Extension
**Given** the existing YAML configuration system  
**When** spiking brain parameters are specified  
**Then** the system SHALL validate neuron model parameters  
**And** SHALL validate plasticity rule parameters  
**And** SHALL provide meaningful error messages for invalid configurations

### Requirement: Brain Factory Extension
The brain factory method SHALL support spiking neural network instantiation.

#### Scenario: Brain Type Resolution
**Given** a configuration specifies brain type as "spiking"
**When** the brain factory creates a brain instance
**Then** it SHALL return a SpikingBrain object
**And** SHALL pass through all spiking-specific configuration parameters

### Requirement: CLI Argument Extension
The command-line interface SHALL accept "spiking" as a valid brain type option.

#### Scenario: Argument Validation
**Given** a user specifies `--brain spiking`
**When** command-line arguments are parsed
**Then** the system SHALL recognize "spiking" as a valid brain type
**And** SHALL pass the selection to the brain factory

### Requirement: Evolutionary Parameter Optimization

The system SHALL support evolutionary optimization of brain parameters as an alternative to gradient-based learning.

#### Scenario: CMA-ES Optimization

- **GIVEN** a quantum brain with N trainable parameters
- **WHEN** the user runs evolution with CMA-ES algorithm
- **THEN** the system SHALL create a population of candidate parameter sets
- **AND** SHALL evaluate fitness by running multiple episodes per candidate
- **AND** SHALL update the search distribution based on fitness rankings
- **AND** SHALL return the best-performing parameters after convergence

#### Scenario: Genetic Algorithm Optimization

- **GIVEN** a quantum brain with N trainable parameters
- **WHEN** the user runs evolution with genetic algorithm
- **THEN** the system SHALL maintain a population of parameter sets
- **AND** SHALL select elite performers for reproduction
- **AND** SHALL apply crossover and mutation to generate offspring
- **AND** SHALL return the best-performing parameters after generations complete

#### Scenario: Parallel Fitness Evaluation

- **GIVEN** a population of candidate parameter sets
- **WHEN** fitness evaluation is requested with parallel workers > 1
- **THEN** the system SHALL evaluate candidates concurrently using multiprocessing
- **AND** SHALL aggregate episode results into per-candidate fitness scores

### Requirement: Fitness Function Interface

The system SHALL provide a configurable fitness function for evolutionary optimization.

#### Scenario: Success Rate Fitness

- **GIVEN** a candidate parameter set
- **WHEN** fitness is evaluated with episodes_per_evaluation = N
- **THEN** the system SHALL run N episodes with those parameters
- **AND** SHALL compute fitness as negative success rate (for minimization)
- **AND** SHALL reset the environment between episodes

#### Scenario: Fitness Aggregation

- **GIVEN** multiple episode results for a candidate
- **WHEN** aggregating to a single fitness value
- **THEN** the system SHALL compute mean success rate across episodes
- **AND** MAY optionally penalize high variance

### Requirement: Brain Parameter Interface

Brain implementations SHALL expose a uniform interface for parameter manipulation required by evolutionary optimization.

#### Scenario: Parameter Export

- **GIVEN** a brain instance with trainable parameters
- **WHEN** `brain.get_parameter_array()` is called
- **THEN** the system SHALL return a flat numpy array of all trainable parameters
- **AND** SHALL maintain consistent ordering across calls

#### Scenario: Parameter Import

- **GIVEN** a brain instance and a parameter array
- **WHEN** `brain.set_parameter_array(params)` is called
- **THEN** the system SHALL update all trainable parameters from the array
- **AND** SHALL validate array length matches expected parameter count

#### Scenario: Brain Copying

- **GIVEN** a brain instance
- **WHEN** `brain.copy()` is called
- **THEN** the system SHALL return an independent copy of the brain
- **AND** modifications to the copy SHALL NOT affect the original

### Requirement: Evolution Configuration

The configuration system SHALL support evolutionary optimization parameters.

#### Scenario: CMA-ES Configuration

- **GIVEN** a YAML configuration with evolution settings
- **WHEN** algorithm is set to "cmaes"
- **THEN** the system SHALL accept population_size, generations, sigma0 parameters
- **AND** SHALL validate parameter ranges

#### Scenario: GA Configuration

- **GIVEN** a YAML configuration with evolution settings
- **WHEN** algorithm is set to "ga"
- **THEN** the system SHALL accept elite_fraction, mutation_rate, crossover_rate parameters
- **AND** SHALL validate parameter ranges

#### Scenario: Parallel Workers Configuration

- **GIVEN** a YAML configuration with evolution settings
- **WHEN** parallel_workers is specified
- **THEN** the system SHALL use that many processes for fitness evaluation
- **AND** SHALL default to 1 (sequential) if not specified

### Requirement: Evolution Script Interface

The system SHALL provide a command-line script for running evolutionary optimization.

#### Scenario: Basic Evolution Run

- **GIVEN** a user wants to optimize brain parameters
- **WHEN** they execute `python scripts/run_evolution.py --config evolution.yml`
- **THEN** the system SHALL load the configuration
- **AND** SHALL run evolutionary optimization
- **AND** SHALL log generation progress (best fitness, mean, std)
- **AND** SHALL save best parameters on completion

#### Scenario: Evolution Checkpoint Resume

- **GIVEN** an interrupted evolution run with checkpoint file
- **WHEN** the user runs with `--resume checkpoint.pkl`
- **THEN** the system SHALL load the optimizer state from checkpoint
- **AND** SHALL continue evolution from the saved generation

### Requirement: Classical Baseline Brain

The system SHALL provide a classical brain with matched parameter count for quantum advantage comparison.

#### Scenario: Linear Classical Brain Creation

- **GIVEN** a configuration specifies brain type as "linear_classical"
- **WHEN** the brain is instantiated
- **THEN** the system SHALL create a linear model with configurable parameter count
- **AND** SHALL implement the same parameter interface as quantum brains

#### Scenario: Matched Parameter Comparison

- **GIVEN** a quantum brain with N parameters
- **WHEN** comparing to classical baseline
- **THEN** the classical brain SHALL have exactly N parameters
- **AND** SHALL use the same evolution process for fair comparison
