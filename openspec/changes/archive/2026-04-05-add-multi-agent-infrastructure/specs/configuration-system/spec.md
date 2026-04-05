## ADDED Requirements

### Requirement: Multi-Agent Configuration Schema

The configuration system SHALL support an optional `multi_agent` section in SimulationConfig for multi-agent scenarios.

#### Scenario: Multi-Agent Disabled (Default)

- **GIVEN** a YAML configuration without a `multi_agent` section
- **WHEN** the configuration is loaded
- **THEN** `multi_agent` SHALL be None or have `enabled=False`
- **AND** the simulation SHALL run in single-agent mode identically to existing behavior

#### Scenario: Homogeneous Population (Shorthand)

- **GIVEN** a YAML configuration with:

```yaml
brain:
  name: mlpppo
  config: { ... }
multi_agent:
  enabled: true
  count: 5
```

- **WHEN** the configuration is loaded
- **THEN** 5 agents SHALL be created using the top-level `brain` config
- **AND** agents SHALL be assigned ids: "agent_0", "agent_1", ..., "agent_4"

#### Scenario: Heterogeneous Population (Explicit)

- **GIVEN** a YAML configuration with:

```yaml
multi_agent:
  enabled: true
  agents:
    - id: agent_0
      brain: { name: mlpppo, config: { ... } }
    - id: agent_1
      brain: { name: lstmppo, config: { ... } }
```

- **WHEN** the configuration is loaded
- **THEN** 2 agents SHALL be created with their respective brain configs
- **AND** agent ids SHALL match the specified ids

#### Scenario: Validation - Count and Agents Mutual Exclusion

- **GIVEN** a multi_agent config with both `count` and `agents` specified
- **WHEN** validation is performed
- **THEN** a ValidationError SHALL be raised
- **AND** the error message SHALL explain that only one of `count` or `agents` may be set

#### Scenario: Validation - Enabled Without Population

- **GIVEN** `multi_agent.enabled=true` with neither `count` nor `agents`
- **WHEN** validation is performed
- **THEN** a ValidationError SHALL be raised

### Requirement: Multi-Agent Config Parameters

The MultiAgentConfig model SHALL support the following parameters with defaults.

#### Scenario: Food Competition Policy

- **GIVEN** a multi_agent config
- **THEN** `food_competition` SHALL default to "first_arrival"
- **AND** valid values SHALL be "first_arrival" and "random"

#### Scenario: Social Detection Radius

- **GIVEN** a multi_agent config
- **THEN** `social_detection_radius` SHALL default to 5
- **AND** SHALL be a positive integer

#### Scenario: Termination Policy

- **GIVEN** a multi_agent config
- **THEN** `termination_policy` SHALL default to "freeze"
- **AND** valid values SHALL be "freeze", "remove", and "end_all"

#### Scenario: Minimum Agent Distance

- **GIVEN** a multi_agent config
- **THEN** `min_agent_distance` SHALL default to 5
- **AND** SHALL be a positive integer

### Requirement: Per-Agent Configuration

Each agent in a multi-agent scenario SHALL support individual configuration.

#### Scenario: AgentConfig Fields

- **GIVEN** an AgentConfig
- **THEN** it SHALL contain:
  - `id: str` -- unique agent identifier
  - `brain: BrainContainerConfig` -- brain architecture and parameters
  - `weights_path: str | None` -- optional path to pre-trained model weights (default None)

#### Scenario: Shared Configuration

- **GIVEN** a multi-agent config
- **THEN** all agents SHALL share the following from the top-level config:
  - `environment` (grid, food, predators, thermotaxis, aerotaxis)
  - `reward` (reward function parameters)
  - `satiety` (hunger system parameters)
  - `sensing` (sensing mode per modality)
  - `max_steps` (episode length)

#### Scenario: Pre-Trained Weight Loading

- **GIVEN** an AgentConfig with `weights_path: "results/pretrained/final.pt"`
- **WHEN** the agent is created
- **THEN** the brain SHALL load weights from the specified path after construction
- **AND** training SHALL continue from the loaded weights

#### Scenario: Weight Path Not Found

- **GIVEN** an AgentConfig with `weights_path` pointing to a non-existent file
- **WHEN** the agent is created
- **THEN** a clear FileNotFoundError SHALL be raised with the path in the message

### Requirement: Multi-Agent Model Persistence

The system SHALL save trained model weights with agent-specific filenames.

#### Scenario: Multi-Agent Weight Saving

- **GIVEN** a completed multi-agent training session with agents "agent_0", "agent_1", "agent_2"
- **WHEN** model weights are saved
- **THEN** files SHALL be named `final_agent_0.pt`, `final_agent_1.pt`, `final_agent_2.pt`

#### Scenario: Single-Agent Backward Compatibility

- **GIVEN** a single-agent training session (agent_id="default")
- **WHEN** model weights are saved
- **THEN** the file SHALL be named `final.pt` (unchanged from current behavior)

### Requirement: Multi-Agent CSV Export

The system SHALL export per-agent and aggregate results for multi-agent sessions.

#### Scenario: Per-Agent Results CSV

- **GIVEN** a completed multi-agent session
- **WHEN** results are exported to CSV
- **THEN** `simulation_results.csv` SHALL include an `agent_id` column
- **AND** each row SHALL represent one agent's result for one episode

#### Scenario: Aggregate Summary CSV

- **GIVEN** a completed multi-agent session
- **WHEN** results are exported
- **THEN** a `multi_agent_summary.csv` SHALL be generated
- **AND** each row SHALL contain per-episode aggregates: total_food, competition_events, proximity_events, agents_alive_at_end, mean_success, food_gini_coefficient
