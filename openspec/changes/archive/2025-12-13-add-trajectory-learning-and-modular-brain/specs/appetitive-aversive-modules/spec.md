# appetitive-aversive-modules Capability Specification

## Purpose

Introduce biologically-inspired modular architecture with separate appetitive (food-seeking) and aversive (predator-avoidance) behavioral circuits, enabling specialized learning with independent gradient signals.

## ADDED Requirements

### Requirement: Appetitive Module

The system SHALL provide an appetitive module for food-seeking behavior using attractive chemical gradients.

#### Scenario: Module Identification

**Given** a config specifies `modules: { appetitive: [0, 1] }`
**When** the ModularBrain is initialized
**Then** it SHALL recognize "appetitive" as a valid module name
**And** SHALL assign qubits [0, 1] to the appetitive module
**And** SHALL use appetitive feature extraction for these qubits

#### Scenario: Food Gradient Feature Extraction

**Given** environment state with food gradient information
**When** extracting appetitive features
**Then** it SHALL encode food_gradient_strength on RX rotation
**And** SHALL encode relative angle to food on RY rotation
**And** SHALL use RZ = 0.0 (reserved for future use)
**And** SHALL scale values to quantum-appropriate ranges [-π/2, π/2]

#### Scenario: Attractive Gradient Behavior

**Given** the appetitive module is active
**When** food is present in the environment
**Then** features SHALL encode positive attractive gradients
**And** SHALL guide the agent toward food sources
**And** SHALL increase gradient strength with proximity

### Requirement: Aversive Module

The system SHALL provide an aversive module for predator-avoidance behavior using repulsive threat gradients.

#### Scenario: Module Identification

**Given** a config specifies `modules: { aversive: [2, 3] }`
**When** the ModularBrain is initialized
**Then** it SHALL recognize "aversive" as a valid module name
**And** SHALL assign qubits [2, 3] to the aversive module
**And** SHALL use aversive feature extraction for these qubits

#### Scenario: Predator Gradient Feature Extraction

**Given** environment state with predator gradient information
**When** extracting aversive features
**Then** it SHALL encode abs(predator_gradient_strength) on RX rotation
**And** SHALL encode relative angle to predator on RY rotation
**And** SHALL use RZ = 0.0 (reserved for future use)
**And** SHALL scale RX and RY values to [-π/2, π/2]
**And** SHALL use the same encoding pattern as appetitive features (biological symmetry)

#### Scenario: Repulsive Gradient Behavior

**Given** the aversive module is active
**When** predators are present in the environment
**Then** features SHALL encode repulsive chemical gradients (as positive magnitude)
**And** SHALL encode direction to threat source (for avoidance)
**And** SHALL increase gradient strength with proximity (exponential decay)
**And** SHALL use ONLY sensory information available to the nematode (no external state)

### Requirement: Gradient Mode Configuration

The environment SHALL support configurable gradient modes for unified vs split gradient computation.

#### Scenario: Unified Gradient Mode (Default)

**Given** a config specifies `gradient_mode: unified` or omits the setting
**When** the environment computes gradients
**Then** it SHALL superpose food and predator gradients into single values
**And** SHALL return gradient_strength = combined magnitude
**And** SHALL return gradient_direction = combined direction
**And** SHALL preserve backward compatibility with single-module configs

#### Scenario: Split Gradient Mode

**Given** a config specifies `gradient_mode: split`
**When** the environment computes gradients
**Then** it SHALL compute food gradients separately from predator gradients
**And** SHALL return food_gradient_strength and food_gradient_direction
**And** SHALL return predator_gradient_strength and predator_gradient_direction
**And** SHALL provide both to the brain for module-specific feature extraction

#### Scenario: Gradient Mode Validation

**Given** a config specifies a gradient_mode value
**When** the configuration is loaded
**Then** it SHALL validate that gradient_mode is either "unified" or "split"
**And** SHALL raise a validation error for invalid values
**And** SHALL default to "unified" if not specified

### Requirement: Module Configuration Schema

The system SHALL extend the configuration schema to support appetitive/aversive module specification.

#### Scenario: Module Declaration

**Given** a YAML config file
**When** specifying brain.config.modules
**Then** it SHALL accept "appetitive" as a module name
**And** SHALL accept "aversive" as a module name
**And** SHALL map each module name to a list of qubit indices
**And** SHALL validate that qubit indices don't overlap between modules

#### Scenario: Multi-Module Qubit Allocation

**Given** a config specifies both appetitive and aversive modules
**When** validating the configuration
**Then** it SHALL verify total qubits matches sum of module allocations
**And** SHALL verify no qubit is assigned to multiple modules
**And** SHALL require qubits parameter to match allocated qubits
**And** SHALL provide clear error messages for misconfigurations

#### Scenario: Single-Module Backward Compatibility

**Given** a config specifies only `modules: { appetitive: [0, 1] }`
**When** the brain is initialized
**Then** it SHALL function correctly with only the appetitive module
**And** SHALL NOT require aversive module to be present
**And** SHALL allow gradual migration from single to multi-module configs

### Requirement: Feature Extraction Registration

The system SHALL provide a feature extraction registry mapping module names to feature functions.

#### Scenario: Appetitive Feature Registration

**Given** the module system is initialized
**When** looking up features for "appetitive" module
**Then** it SHALL return the appetitive_features() function
**And** SHALL use this function to extract features for appetitive qubits

#### Scenario: Aversive Feature Registration

**Given** the module system is initialized
**When** looking up features for "aversive" module
**Then** it SHALL return the aversive_features() function
**And** SHALL use this function to extract features for aversive qubits

#### Scenario: Unknown Module Handling

**Given** a config specifies an unknown module name
**When** the brain attempts to extract features
**Then** it SHALL raise a descriptive error
**And** SHALL list valid module names in the error message

## MODIFIED Requirements

### Requirement: Module Name Enumeration

The ModuleName enum SHALL include appetitive and aversive modules as standard behavioral circuits.

#### Scenario: Enum Extension

**Given** the ModuleName enum definition
**When** modules are referenced in code
**Then** it SHALL include ModuleName.APPETITIVE = "appetitive"
**And** SHALL include ModuleName.AVERSIVE = "aversive"
**And** SHALL maintain existing modules (PROPRIOCEPTION, etc.) for compatibility

### Requirement: Environment Gradient Computation

The environment's get_state() method SHALL compute gradients based on configured gradient mode.

#### Scenario: Separate Gradient Tracking

**Given** gradient_mode is "split"
**When** computing environment state
**Then** it SHALL track food gradient vector separately: (food_x, food_y)
**And** SHALL track predator gradient vector separately: (pred_x, pred_y)
**And** SHALL compute magnitudes and directions independently
**And** SHALL pass both gradient sets to BrainParams

#### Scenario: Unified Gradient Tracking

**Given** gradient_mode is "unified"
**When** computing environment state
**Then** it SHALL superpose food and predator vectors: (total_x, total_y)
**And** SHALL compute single gradient magnitude and direction
**And** SHALL pass combined gradient to BrainParams (current behavior)

## REMOVED Requirements

None. The existing `chemotaxis` module was retained for backward compatibility.

## Implementation Notes

### Chemotaxis vs Appetitive/Aversive

The original proposal planned to rename `chemotaxis` to `appetitive`. During implementation, we decided to keep all three as separate modules with distinct purposes:

- **chemotaxis**: Uses combined/superposed gradient (food - predator). Suitable for simpler 2-qubit configurations.
- **appetitive**: Uses separated food-only gradient. Requires `use_separated_gradients: true`.
- **aversive**: Uses separated predator-only gradient. Requires `use_separated_gradients: true`.

This approach provides full backward compatibility while enabling new experiments with specialized modules.
