## ADDED Requirements

### Requirement: Evaluation Scenario Configs

The configuration system SHALL provide scenario configs for Phase 4 evaluation.

#### Scenario: Single-Agent Baselines

- **GIVEN** evaluation needs single-agent vs multi-agent comparison
- **THEN** single-agent foraging configs SHALL exist for both oracle and temporal modes
- **AND** they SHALL use identical environment parameters (20x20 grid, 3 food) to multi-agent configs

#### Scenario: Agent Scaling Series

- **GIVEN** evaluation needs to measure coordination difficulty scaling
- **THEN** configs SHALL exist for 1, 2, 5, and 10 agents on 20x20 grid
- **AND** all SHALL use identical environment, reward, and satiety parameters

#### Scenario: Pursuit Predator Configs

- **GIVEN** evaluation needs to measure collective predator response
- **THEN** configs SHALL exist for 5-agent pursuit with alarm pheromones enabled
- **AND** a matched control with alarm pheromones disabled
- **AND** both SHALL use identical predator, environment, and agent parameters

#### Scenario: Temporal Mode Configs

- **GIVEN** evaluation needs to assess pheromone value in temporal sensing
- **THEN** configs SHALL exist for LSTM PPO GRU with temporal chemotaxis and STAM
- **AND** variants with and without pheromone modules SHALL be provided
