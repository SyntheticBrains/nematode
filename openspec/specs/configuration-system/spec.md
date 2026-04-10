# configuration-system Specification

## Purpose

This specification defines the YAML-based configuration system for the Quantum Nematode simulation platform. It governs how brain architectures (qvarcircuit, qqlearning, mlpreinforce, mlpppo, mlpdqn, spikingreinforce; legacy aliases: modular, qmodular, mlp, qmlp, ppo, spiking), environment parameters, and learning hyperparameters are specified, validated, and loaded. The configuration system ensures all parameters fall within valid ranges, applies sensible defaults for optional settings, and provides example configurations for common use cases. This spec is intended for developers extending the platform with new brain types or environment features.

## Requirements

### Requirement: Spiking Brain Configuration Schema

The configuration system SHALL support a complete schema for spiking neural network parameters.

#### Scenario: YAML Configuration Parsing

- **GIVEN** a YAML configuration file with spiking brain section
- **WHEN** the configuration is loaded
- **THEN** the system SHALL parse neuron model parameters
- **AND** SHALL parse plasticity rule parameters
- **AND** SHALL parse network topology parameters
- **AND** SHALL validate all parameter ranges and constraints

#### Scenario: Default Parameter Application

- **GIVEN** a spiking brain configuration with missing optional parameters
- **WHEN** the configuration is processed
- **THEN** the system SHALL apply sensible defaults
- **AND** SHALL ensure all required parameters are present
- **AND** SHALL log applied defaults for user awareness

### Requirement: Parameter Validation

The configuration system SHALL validate spiking neural network parameters for biological and computational feasibility.

#### Scenario: Neuron Parameter Validation

- **GIVEN** LIF neuron parameters in configuration
- **WHEN** validation is performed
- **THEN** tau_m SHALL be positive (> 0)
- **AND** v_threshold SHALL be greater than v_reset
- **AND** simulation time_step SHALL be appropriate for tau_m

#### Scenario: STDP Parameter Validation

- **GIVEN** STDP plasticity parameters in configuration
- **WHEN** validation is performed
- **THEN** tau_plus and tau_minus SHALL be positive
- **AND** learning_rate SHALL be in reasonable range (0.0001 - 0.1)
- **AND** A_plus and A_minus SHALL be positive

### Requirement: Configuration Examples

The system SHALL provide example configurations for common spiking brain use cases.

#### Scenario: Small Network Configuration

- **GIVEN** a need for basic spiking brain testing
- **WHEN** loading spiking_small.yml configuration
- **THEN** the system SHALL configure a minimal viable spiking network
- **AND** SHALL use parameters suitable for fast convergence

#### Scenario: Medium Network Configuration

- **GIVEN** a need for standard experimental setup
- **WHEN** loading spikingreinforce_foraging_medium.yml configuration
- **THEN** the system SHALL configure a balanced network
- **AND** SHALL use parameters suitable for robust learning

### Requirement: Brain Type Enumeration Extension

The brain type validation SHALL include "spiking" as a valid option.

#### Scenario: Brain Type Validation

- **GIVEN** configuration specifies brain type
- **WHEN** validation occurs
- **THEN** "spikingreinforce" SHALL be accepted as valid (legacy alias: "spiking")
- **AND** existing "qvarcircuit", "qqlearning", "mlpreinforce", "mlpppo", "mlpdqn" types are also valid (legacy aliases: "modular", "qmodular", "mlp", "qmlp")

### Requirement: Dynamic Environment Configuration Schema

The configuration system SHALL support a complete schema for dynamic foraging environment parameters.

#### Scenario: Dynamic Environment Parameters

- **GIVEN** a configuration
- **WHEN** the configuration is parsed
- **THEN** the system SHALL parse `grid_size` as tuple (width, height)
- **AND** SHALL parse `foods_on_grid` (integer, default: grid_area / 50)
- **AND** SHALL parse `target_foods_to_collect` (integer, default: foods_on_grid × 1.5)
- **AND** SHALL parse `min_food_distance` (integer, default: max(5, min(width, height) / 10))
- **AND** SHALL parse `food_spawn_interval` (integer, default: 0 for immediate)
- **AND** SHALL parse `viewport_size` as tuple (width, height, default: (11, 11))
- **AND** SHALL parse `agent_exclusion_radius` (integer, default: 10)

#### Scenario: Satiety Configuration Schema

- **GIVEN** a configuration with dynamic environment
- **WHEN** satiety parameters are specified
- **THEN** the system SHALL parse `initial_satiety` (float, default: 200.0)
- **AND** SHALL parse `satiety_decay_rate` (float, default: 1.0)
- **AND** SHALL parse `satiety_gain_per_food` (float as fraction, default: 0.2)
- **AND** SHALL validate satiety_decay_rate > 0
- **AND** SHALL validate satiety_gain_per_food between 0.0 and 1.0

#### Scenario: Gradient Configuration

- **GIVEN** a configuration with environment gradient settings
- **WHEN** the configuration is parsed
- **THEN** the system SHALL parse `gradient_decay_constant` (float, default: 10.0)
- **AND** SHALL parse `gradient_strength` (float, default: 1.0)
- **AND** SHALL parse `gradient_scaling` (enum: "exponential" or "tanh", default: "exponential")

#### Scenario: Exploration Bonus Configuration

- **GIVEN** a configuration with reward settings
- **WHEN** exploration parameters are specified
- **THEN** the system SHALL parse `exploration_bonus` (float, default: 0.05)
- **AND** SHALL allow 0.0 to disable exploration rewards
- **AND** SHALL validate exploration_bonus >= 0.0

#### Scenario: Configuration File Examples

- **GIVEN** example configuration files in `configs/scenarios/`
- **WHEN** users need preset foraging environments
- **THEN** `configs/scenarios/foraging/<brain>_small_oracle.yml` SHALL provide small foraging configuration
- **AND** `<brain>_medium_oracle.yml` SHALL provide medium foraging configuration
- **AND** `<brain>_large_oracle.yml` SHALL provide large foraging configuration
- **AND** each SHALL include commented parameter explanations

### Requirement: Configuration Validation for Dynamic Environments

The configuration system SHALL validate dynamic environment parameters for logical consistency and computational feasibility.

#### Scenario: Food Count Validation

- **GIVEN** a dynamic environment configuration
- **WHEN** validation is performed
- **THEN** `foods_on_grid` SHALL be positive and > 0
- **AND** `target_foods_to_collect` SHALL be >= `foods_on_grid`
- **AND** if `foods_on_grid` is too large for grid with `min_food_distance`, SHALL emit warning
- **AND** if `agent_exclusion_radius` exceeds `min_food_distance`, SHALL warn that exclusion zones may prevent food placement

#### Scenario: Grid Size Validation

- **GIVEN** a dynamic environment configuration
- **WHEN** grid size is validated
- **THEN** both width and height SHALL be >= 10
- **AND** both SHALL be \<= 200 (performance limit)
- **AND** if grid size > 100×100, SHALL log performance warning

#### Scenario: Satiety Balance Validation

- **GIVEN** a dynamic environment configuration
- **WHEN** satiety parameters are validated
- **THEN** the system SHALL check that `initial_satiety / satiety_decay_rate` provides reasonable episode length
- **AND** SHALL warn if satiety allows fewer than 50 steps
- **AND** SHALL warn if food consumption cannot sustain foraging (gain < expected consumption rate)

#### Scenario: Viewport Size Validation

- **GIVEN** a dynamic environment configuration with viewport
- **WHEN** validation occurs
- **THEN** viewport width and height SHALL be odd numbers (for centered agent)
- **AND** SHALL be at least 3×3
- **AND** SHALL not exceed grid size
- **AND** if even number provided, SHALL auto-adjust to next odd number and warn

### Requirement: Predator Configuration Schema

The system SHALL support comprehensive configuration of predator behavior, appearance, and mechanics within dynamic environments.

#### Scenario: Basic Predator Configuration

- **GIVEN** a YAML configuration file with predator settings
- **WHEN** the configuration is loaded
- **THEN** the system SHALL accept the following predator parameters under `environment.dynamic.predators`:
  - `enabled` (boolean, default false)
  - `count` (integer, default 2)
  - `speed` (float, default 1.0)
  - `movement_pattern` (string: "stationary" or "pursuit", default "pursuit")
  - `detection_radius` (integer, default 8)
  - `damage_radius` (integer, default 0)
- **AND** all parameters SHALL have sensible defaults allowing minimal configuration

#### Scenario: Predator Gradient Configuration

- **GIVEN** a configuration specifying predator gradient parameters
- **WHEN** the configuration is loaded
- **THEN** the system SHALL accept under `environment.dynamic.predators`:
  - `gradient_decay_constant` (float, default 12.0)
  - `gradient_strength` (float, default 1.0)
- **AND** these MAY differ from food gradient parameters in `environment.dynamic.foraging`
- **AND** SHALL be used to compute predator repulsion gradients

#### Scenario: Predator Penalty Configuration

- **GIVEN** a configuration with predator reward penalties
- **WHEN** the configuration is loaded
- **THEN** the system SHALL accept under `reward`:
  - `penalty_predator_proximity` (float, default 0.1) - reward penalty per step within detection radius
- **AND** penalty value interpretation: a `penalty_predator_proximity: 0.1` configuration means -0.1 reward per step (positive value subtracted to create penalty)
- **AND** penalty value of 0.0 SHALL disable proximity penalties
- **AND** the penalty SHALL use positive values that are subtracted from reward (consistent with other penalty values)

#### Scenario: Minimal Predator Enablement

- **GIVEN** a configuration with only `predators.enabled: true`
- **WHEN** the configuration is loaded
- **THEN** all other predator parameters SHALL use default values
- **AND** the simulation SHALL run with 2 predators at speed 1.0, detection radius 8, kill radius 0

### Requirement: Restructured Dynamic Environment Configuration

The system SHALL organize dynamic environment settings into logical subsections for foraging and predators to improve clarity and maintainability.

#### Scenario: Foraging Subsection Configuration

- **GIVEN** a configuration using the new structure
- **WHEN** foraging parameters are specified
- **THEN** the system SHALL accept under `environment.dynamic.foraging`:
  - `foods_on_grid` (integer) (previously named `num_initial_foods`)
  - `target_foods_to_collect` (integer) (previously named `max_active_foods`)
  - `min_food_distance` (integer)
  - `agent_exclusion_radius` (integer)
  - `gradient_decay_constant` (float)
  - `gradient_strength` (float)
- **AND** these SHALL be nested under `foraging` subsection, not at `dynamic` root level

#### Scenario: Grid and Viewport at Dynamic Root

- **GIVEN** a configuration using the new structure
- **WHEN** environment structure is specified
- **THEN** the following SHALL remain at `environment.dynamic` root level:
  - `grid_size` (integer or tuple)
  - `viewport_size` (tuple)
- **AND** these SHALL not be nested under `foraging` or `predators`
- **AND** they SHALL apply to the entire environment regardless of feature enablement

#### Scenario: Complete Restructured Configuration Example

- **GIVEN** a full dynamic environment configuration
- **WHEN** all sections are specified
- **THEN** the structure SHALL be:

```yaml
environment:
  type: dynamic
  dynamic:
    grid_size: 100
    viewport_size: [11, 11]

    foraging:
      foods_on_grid: 50
      target_foods_to_collect: 50
      min_food_distance: 10
      agent_exclusion_radius: 15
      gradient_decay_constant: 12.0
      gradient_strength: 1.0

    predators:
      enabled: true
      count: 3
      speed: 1.0
      movement_pattern: "pursuit"
      detection_radius: 8
      damage_radius: 0
      gradient_decay_constant: 12.0
      gradient_strength: 1.0

reward:
  penalty_predator_proximity: 0.1
```

### Requirement: Backward Compatibility with Legacy Configuration

The system SHALL automatically migrate legacy flat configuration structure to new nested structure with deprecation warnings.

#### Scenario: Legacy Flat Configuration Migration

- **GIVEN** an existing configuration with flat structure:

```yaml
environment:
  type: dynamic
  dynamic:
    grid_size: 50
    foods_on_grid: 20
    target_foods_to_collect: 30
    gradient_decay_constant: 12.0
```

- **WHEN** the configuration is loaded
- **THEN** the system SHALL automatically migrate to nested structure
- **AND** food-related parameters SHALL be moved under `foraging` subsection
- **AND** a deprecation warning SHALL be logged
- **AND** the simulation SHALL run correctly with migrated configuration

#### Scenario: Migration Warning Message

- **GIVEN** a legacy flat configuration is loaded
- **WHEN** migration is performed
- **THEN** a warning message SHALL be logged stating:
  - "Flat dynamic environment configuration is deprecated"
  - "Please restructure configuration with 'foraging' subsection"
  - Specific parameters that were migrated
- **AND** the warning SHALL include example of new structure

#### Scenario: New Configuration No Migration

- **GIVEN** a configuration already using nested `foraging` subsection
- **WHEN** the configuration is loaded
- **THEN** no migration SHALL be performed
- **AND** no deprecation warning SHALL be logged
- **AND** configuration SHALL be used as-is

#### Scenario: Mixed Configuration Handling

- **GIVEN** a configuration with some parameters in `foraging` subsection and some at root level
- **WHEN** the configuration is loaded
- **THEN** the system SHALL prioritize `foraging` subsection values
- **AND** root-level values SHALL be used only if not present in `foraging`
- **AND** a warning SHALL be logged about the inconsistent structure

### Requirement: Predator Movement Pattern Validation

The system SHALL validate predator movement pattern configuration and provide clear errors for invalid values.

#### Scenario: Valid Movement Pattern — Pursuit

- **GIVEN** a configuration with `movement_pattern: "pursuit"`
- **WHEN** the configuration is validated
- **THEN** validation SHALL pass
- **AND** the predator SHALL actively pursue the agent when within detection radius

#### Scenario: Valid Movement Pattern — Stationary

- **GIVEN** a configuration with `movement_pattern: "stationary"`
- **WHEN** the configuration is validated
- **THEN** validation SHALL pass
- **AND** the predator SHALL remain at its spawn position (toxic zone)

#### Scenario: Invalid Movement Pattern

- **GIVEN** a configuration with `movement_pattern: "invalid_pattern"`
- **WHEN** the configuration is validated
- **THEN** validation SHALL fail with clear error message
- **AND** error SHALL list valid options: "stationary", "pursuit"

### Requirement: Configuration Examples and Templates

The system SHALL provide example configuration files demonstrating predator-enabled setups for different difficulty levels.

#### Scenario: Pursuit Predator Small Environment Example

- **GIVEN** an example configuration file `configs/scenarios/pursuit/mlpppo_small_oracle.yml`
- **WHEN** the file is read
- **THEN** it SHALL demonstrate:
  - 20×20 grid with pursuit predators enabled
  - 2 predators for introductory difficulty
  - All predator parameters explicitly shown with comments
  - Foraging parameters in nested subsection
- **AND** the configuration SHALL be immediately runnable
  - Both foraging and predator subsections fully configured

#### Scenario: Configuration Documentation

- **GIVEN** the example configuration files
- **WHEN** a user reads the files
- **THEN** each predator parameter SHALL have inline comment explaining:
  - Parameter purpose and effect
  - Valid value range
  - Recommended values for different difficulty levels
  - How parameter affects learning difficulty

### Requirement: Weight Path Configuration Field

The base `BrainConfig` class SHALL support an optional `weights_path` field for config-based weight loading.

#### Scenario: Weights Path Field on BrainConfig

- **WHEN** any brain configuration is parsed
- **THEN** `BrainConfig` SHALL accept an optional `weights_path` field (str | None, default None)
- **AND** the field SHALL be inherited by all brain-specific config classes

#### Scenario: Weights Path in YAML

- **WHEN** a YAML config includes `brain.config.weights_path: "artifacts/models/stage1.pt"`
- **THEN** the configuration system SHALL parse it as a string path
- **AND** SHALL pass it through to the brain config instance

#### Scenario: Weights Path Default

- **WHEN** a YAML config does not include `weights_path` under brain config
- **THEN** the value SHALL default to `None`
- **AND** no unified-path weight loading SHALL occur via the `WeightPersistence` system
- **AND** legacy brain-specific fields (`qsnn_weights_path`, `cortex_weights_path`, `reflex_weights_path`, `critic_weights_path`) SHALL remain supported and MAY still trigger weight loading independently

#### Scenario: Backward Compatibility

- **WHEN** existing YAML configs that do not include `weights_path` are loaded
- **THEN** parsing SHALL succeed without errors
- **AND** all existing brain config fields SHALL continue to work unchanged

<!-- Synced from change: add-temporal-sensing-and-stam -->

<!-- Synced from change: add-aerotaxis-system -->

### Requirement: Aerotaxis Configuration Schema

The configuration system SHALL support an aerotaxis configuration section for defining oxygen field parameters, zone thresholds, and reward/penalty values.

#### Scenario: Aerotaxis Configuration Section

- **WHEN** a YAML configuration includes an `aerotaxis` section under `environment`
- **THEN** the system SHALL accept the following fields:
  - `enabled: bool` (default: false)
  - `base_oxygen: float` (default: 10.0, O2 percentage at grid center)
  - `gradient_direction: float` (default: 0.0, radians)
  - `gradient_strength: float` (default: 0.1, O2 % per cell)
  - `high_oxygen_spots: list[list[float]]` (optional, [x, y, intensity] tuples)
  - `low_oxygen_spots: list[list[float]]` (optional, [x, y, intensity] tuples)
  - `spot_decay_constant: float` (default: 5.0)
  - `comfort_reward: float` (default: 0.0)
  - `danger_penalty: float` (default: -0.5)
  - `danger_hp_damage: float` (default: 0.5)
  - `lethal_hp_damage: float` (default: 6.0)
  - `reward_discomfort_food: float` (default: 0.0, bonus for collecting food in danger zones)
  - `lethal_hypoxia_upper: float` (default: 2.0)
  - `danger_hypoxia_upper: float` (default: 5.0)
  - `comfort_lower: float` (default: 5.0)
  - `comfort_upper: float` (default: 12.0)
  - `danger_hyperoxia_upper: float` (default: 17.0)

#### Scenario: Aerotaxis Not Configured

- **WHEN** no `aerotaxis` section is present in the environment configuration
- **THEN** aerotaxis SHALL be disabled
- **AND** the system SHALL behave identically to pre-aerotaxis versions

#### Scenario: Aerotaxis With Thermotaxis

- **WHEN** both `thermotaxis` and `aerotaxis` sections are present and enabled
- **THEN** both SHALL be parsed and initialized independently
- **AND** both OxygenField and TemperatureField SHALL coexist in the environment

## ADDED Requirements

### Requirement: Sensing Configuration Schema

The configuration system SHALL support a sensing configuration section for selecting sensing modes and STAM parameters.

#### Scenario: Sensing Mode Configuration

- **WHEN** a YAML configuration includes a `sensing` section under `environment`
- **THEN** the system SHALL accept `chemotaxis_mode` (string, one of: "oracle", "temporal", "derivative", default: "oracle")
- **AND** SHALL accept `thermotaxis_mode` (string, one of: "oracle", "temporal", "derivative", default: "oracle")
- **AND** SHALL accept `nociception_mode` (string, one of: "oracle", "temporal", "derivative", default: "oracle")
- **AND** SHALL accept `aerotaxis_mode` (string, one of: "oracle", "temporal", "derivative", default: "oracle")
- **AND** each mode SHALL be independently configurable

#### Scenario: STAM Configuration

- **WHEN** a YAML configuration includes STAM parameters under `environment.sensing`
- **THEN** the system SHALL accept `stam_enabled` (boolean, default: false)
- **AND** SHALL accept `stam_buffer_size` (integer, default: 30, must be > 0)
- **AND** SHALL accept `stam_decay_rate` (float, default: 0.1, must be > 0)

#### Scenario: Sensing Configuration Absent

- **WHEN** no `sensing` section is provided in the environment configuration
- **THEN** all sensing modes SHALL default to "oracle"
- **AND** STAM SHALL be disabled
- **AND** the system SHALL behave identically to pre-temporal-sensing versions

#### Scenario: Invalid Sensing Mode

- **WHEN** a sensing mode is set to an unrecognised value (e.g., `chemotaxis_mode: "invalid"`)
- **THEN** configuration validation SHALL fail with a clear error message
- **AND** the error SHALL list the valid options: "oracle", "temporal", "derivative"

### Requirement: Sensing Mode Validation

The configuration system SHALL validate sensing mode and STAM parameter combinations.

#### Scenario: Temporal Mode Without STAM Warning

- **WHEN** any modality including aerotaxis is set to `temporal` mode without `stam_enabled: true`
- **THEN** the system SHALL log a warning that Mode A (temporal) without STAM may result in very limited sensory information
- **AND** the configuration SHALL still be accepted (STAM is recommended but not required)

#### Scenario: Derivative Mode Auto-Enables STAM

- **WHEN** any modality including aerotaxis is set to `derivative` mode and `stam_enabled` is not explicitly set to `true`
- **THEN** the system SHALL auto-enable STAM with default parameters (`buffer_size: 30`, `decay_rate: 0.1`)
- **AND** SHALL log an info message indicating that STAM was auto-enabled because derivative mode requires temporal history
- **AND** explicitly-set STAM parameters SHALL be preserved if provided

#### Scenario: STAM Buffer Size Validation

- **WHEN** `stam_buffer_size` is set to 0 or a negative value
- **THEN** configuration validation SHALL fail
- **AND** the error SHALL indicate that buffer size must be a positive integer

#### Scenario: STAM Decay Rate Validation

- **WHEN** `stam_decay_rate` is set to 0 or a negative value
- **THEN** configuration validation SHALL fail
- **AND** the error SHALL indicate that decay rate must be a positive float

### Requirement: Temporal Sensing Example Configurations

The system SHALL provide example configurations demonstrating temporal sensing modes.

#### Scenario: Temporal Foraging Example

- **WHEN** example config `mlpppo_foraging_small_temporal.yml` is loaded
- **THEN** it SHALL configure Mode A (temporal) chemotaxis with STAM enabled
- **AND** SHALL use MLPPPO brain architecture
- **AND** SHALL use existing small environment parameters (20×20 grid, 5 foods, 500 steps)
- **AND** SHALL include inline comments explaining the sensing mode and STAM parameters

#### Scenario: Temporal Thermotaxis With Foraging Example

- **WHEN** example config `mlpppo_thermotaxis_foraging_small_temporal.yml` is loaded
- **THEN** it SHALL configure Mode A (temporal) thermotaxis with STAM enabled
- **AND** chemotaxis SHALL also be configured in temporal mode
- **AND** SHALL use existing small thermotaxis foraging environment parameters

#### Scenario: Temporal Pursuit Predators Example

- **WHEN** example config `mlpppo_pursuit_predators_small_temporal.yml` is loaded
- **THEN** it SHALL configure Mode A (temporal) nociception and chemotaxis with STAM enabled
- **AND** SHALL use pursuit predators (`movement_pattern: pursuit`)
- **AND** SHALL use existing small pursuit predator environment parameters

#### Scenario: Temporal Thermotaxis With Pursuit Predators Example

- **WHEN** example config `mlpppo_thermotaxis_pursuit_predators_small_temporal.yml` is loaded
- **THEN** it SHALL configure temporal sensing for all three modalities (chemotaxis, thermotaxis, nociception) with STAM enabled
- **AND** SHALL use pursuit predators (`movement_pattern: pursuit`)
- **AND** SHALL use existing small thermotaxis pursuit predator environment parameters

### Requirement: Complete Temporal Sensing Configuration Example

The configuration system SHALL support the following YAML structure for temporal sensing.

#### Scenario: Full Configuration Structure

- **WHEN** a complete temporal sensing configuration is provided
- **THEN** the system SHALL accept the following structure:

```yaml
environment:
  grid_size: 20
  viewport_size: [11, 11]
  sensing:
    chemotaxis_mode: temporal
    thermotaxis_mode: derivative
    nociception_mode: oracle
    stam_enabled: true
    stam_buffer_size: 30
    stam_decay_rate: 0.1
  foraging:
    foods_on_grid: 5
    target_foods_to_collect: 10
```

- **AND** the sensing section SHALL be parsed before brain construction
- **AND** sensory module translation SHALL be applied based on the sensing modes

<!-- Synced from change: add-lstm-ppo-brain -->

### Requirement: LSTM PPO Brain Configuration Schema

The configuration system SHALL support the `lstmppo` brain type with LSTM/GRU-specific parameters.

#### Scenario: Brain Type Registration

- **WHEN** a YAML configuration specifies `brain.name: lstmppo`
- **THEN** the system SHALL accept the configuration and create an LSTMPPOBrain instance
- **AND** `lstmppo` SHALL be registered in BRAIN_CONFIG_MAP and BrainType enum
- **AND** it SHALL be classified as a CLASSICAL_BRAIN_TYPE

#### Scenario: LSTM PPO Configuration Parameters

- **WHEN** a YAML configuration includes `lstmppo` brain config
- **THEN** the system SHALL accept:
  - `rnn_type` (string, "lstm" or "gru", default "lstm")
  - `lstm_hidden_dim` (integer, default 64, must be >= 2)
  - `bptt_chunk_length` (integer, default 16, must be >= 4)
  - `actor_hidden_dim` (integer, default 64)
  - `critic_hidden_dim` (integer, default 128)
  - `actor_num_layers` (integer, default 2)
  - `critic_num_layers` (integer, default 2)
  - `actor_lr` (float, default 0.0005)
  - `critic_lr` (float, default 0.0005)
  - `gamma`, `gae_lambda`, `clip_epsilon`, `value_loss_coef` (standard PPO params)
  - `num_epochs` (integer, default 6)
  - `rollout_buffer_size` (integer, default 1024, must be >= bptt_chunk_length)
  - `max_grad_norm` (float, default 0.5)
  - `entropy_coef`, `entropy_coef_end`, `entropy_decay_episodes` (entropy decay)
  - `lr_warmup_episodes`, `lr_warmup_start`, `lr_decay_episodes`, `lr_decay_end` (LR scheduling)
  - `sensory_modules` (list of ModuleName, **required** — no legacy 2-feature mode; validation SHALL reject None)

#### Scenario: Example Configurations

- **WHEN** example configs are provided for `lstmppo`
- **THEN** small foraging configs SHALL be provided for both derivative and temporal modes
- **AND** large thermotaxis + pursuit predator configs SHALL be provided for both derivative and temporal modes
- **AND** large thermotaxis + stationary predator configs SHALL be provided for both derivative and temporal modes
- **AND** all SHALL use sensory modules compatible with temporal sensing
- **AND** all SHALL use `rnn_type: gru` as the default (GRU outperforms LSTM across all evaluated environments)

### Requirement: Multi-Agent Configuration Schema

The configuration system SHALL support an optional `multi_agent` section for multi-agent scenarios.

#### Scenario: Multi-Agent Disabled (Default)

- **GIVEN** a config without `multi_agent` section
- **THEN** simulation SHALL run in single-agent mode

#### Scenario: Homogeneous Population

- **GIVEN** `multi_agent.enabled: true` with `count: N`
- **THEN** N agents SHALL be created using the top-level `brain` config

#### Scenario: Heterogeneous Population

- **GIVEN** `multi_agent.enabled: true` with `agents` list
- **THEN** agents SHALL be created with per-agent brain configs

#### Scenario: Validation

- Count and agents are mutually exclusive
- Enabled requires either count or agents

### Requirement: Multi-Agent Config Parameters

#### Scenario: Default Parameters

- `food_competition` defaults to "first_arrival" (valid: "first_arrival", "random")
- `social_detection_radius` defaults to 5 (positive integer)
- `termination_policy` defaults to "freeze" (valid: "freeze", "remove", "end_all")
- `min_agent_distance` defaults to 5 (positive integer)

### Requirement: Per-Agent Configuration

Each agent supports: `id`, `brain`, `weights_path`, `social_phenotype`.

#### Scenario: Shared vs Per-Agent Config

- All agents share: environment, reward, satiety, sensing, max_steps
- Per-agent: brain config, pre-trained weights, social phenotype

### Requirement: Multi-Agent Model Persistence

- Multi-agent saves: `final_agent_0.pt`, `final_agent_1.pt`, etc.
- Single-agent saves: `final.pt` (unchanged)

### Requirement: Multi-Agent CSV Export

#### Scenario: Per-Agent Results

- `simulation_results.csv` includes `agent_id` column (one row per agent per episode)
- Single-agent: agent_id is "default"

#### Scenario: Aggregate Summary

- `multi_agent_summary.csv` with: run, total_food, competition_events, proximity_events, alive_at_end, mean_success, gini, social_feeding_events, aggregation_index, alarm_evasion_events, food_sharing_events

### Requirement: Pheromone Configuration Schema

The environment config SHALL support pheromone parameters.

#### Scenario: Pheromone Config

- `environment.pheromones.enabled`, `food_marking`, `alarm`, `aggregation` (optional) sub-blocks
- Each type: `emission_strength`, `spatial_decay_constant`, `temporal_half_life`, `max_sources`

#### Scenario: Defaults

- Pheromones disabled by default; aggregation is None unless explicitly configured

### Requirement: SensingConfig Pheromone and Aggregation Modes

- `pheromone_food_mode`, `pheromone_alarm_mode`, `pheromone_aggregation_mode` all default to ORACLE
- `apply_sensing_mode()` translates oracle modules to temporal variants when non-oracle
- `validate_sensing_config()` auto-enables STAM for derivative/temporal modes

### Requirement: Social Feeding Configuration

- `environment.social_feeding` block with `enabled`, `decay_reduction`, `solitary_decay`
- Defaults: disabled, decay_reduction=0.7, solitary_decay=1.0
- Detection radius shared with `multi_agent.social_detection_radius`

### Requirement: Per-Agent Social Phenotype

- `AgentConfig.social_phenotype`: "social" (default) or "solitary"
- Homogeneous configs default all agents to "social"
