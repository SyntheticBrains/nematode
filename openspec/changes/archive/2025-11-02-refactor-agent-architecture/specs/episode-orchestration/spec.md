# Spec Delta: Episode Orchestration

## ADDED Requirements

### Requirement: Episode runner strategy pattern for execution modes
The system SHALL provide an `EpisodeRunner` protocol and concrete implementations (`StandardEpisodeRunner`, `ManyworldsEpisodeRunner`) to encapsulate different episode execution strategies.

#### Scenario: Standard episode execution
**GIVEN** a StandardEpisodeRunner with a step processor, metrics tracker, and renderer
**WHEN** run() is called with max_steps=100 and reward_config
**THEN** the runner SHALL:
- Execute steps sequentially until goal reached or max_steps
- Track metrics for each step
- Render frames according to rendering configuration
- Return EpisodeResult with path, success status, and metrics

#### Scenario: Many-worlds episode execution
**GIVEN** a ManyworldsEpisodeRunner with branching_factor=3
**WHEN** run() is called with max_steps=100 and reward_config
**THEN** the runner SHALL:
- Execute initial step to get action probabilities
- Create branching_factor parallel trajectories weighted by probabilities
- Execute each branch until termination
- Select the highest-reward trajectory
- Return EpisodeResult for the best trajectory

#### Scenario: Episode termination on goal reached
**GIVEN** any EpisodeRunner executing an episode
**WHEN** the agent reaches a goal location
**THEN** the episode SHALL terminate immediately
**AND** EpisodeResult.success SHALL be True
**AND** EpisodeResult.steps_taken SHALL reflect actual steps

#### Scenario: Episode termination on max steps
**GIVEN** any EpisodeRunner executing an episode
**WHEN** max_steps is reached without reaching a goal
**THEN** the episode SHALL terminate
**AND** EpisodeResult.success SHALL be False
**AND** EpisodeResult.steps_taken SHALL equal max_steps

#### Scenario: Episode termination on starvation
**GIVEN** any EpisodeRunner executing an episode with satiety enabled
**WHEN** the agent's satiety reaches the starvation threshold
**THEN** the episode SHALL terminate
**AND** EpisodeResult.success SHALL be False
**AND** EpisodeResult.info SHALL contain "starvation" as the termination reason

### Requirement: Episode runners delegate to components for step execution
Episode runners SHALL use the provided StepProcessor for executing individual steps rather than implementing step logic directly.

#### Scenario: Runner delegates step execution
**GIVEN** a StandardEpisodeRunner with a StepProcessor
**WHEN** executing a step in the episode loop
**THEN** the runner SHALL call step_processor.process_step()
**AND** SHALL NOT directly interact with the brain or environment
**AND** SHALL use the returned StepResult to make control flow decisions

### Requirement: Episode runners maintain backward compatibility
The QuantumNematodeAgent class SHALL maintain its existing public API by delegating run_episode() and run_manyworlds_mode() to the appropriate episode runners.

#### Scenario: Backward compatible run_episode
**GIVEN** a QuantumNematodeAgent instance
**WHEN** run_episode(reward_config, max_steps) is called
**THEN** it SHALL internally create a StandardEpisodeRunner
**AND** delegate execution to the runner
**AND** return list[tuple] path format (same as before refactoring)
**AND** all existing tests SHALL pass without modification

#### Scenario: Backward compatible run_manyworlds_mode
**GIVEN** a QuantumNematodeAgent instance
**WHEN** run_manyworlds_mode(reward_config, manyworlds_config, max_steps) is called
**THEN** it SHALL internally create a ManyworldsEpisodeRunner
**AND** delegate execution to the runner
**AND** return list[tuple] path format (same as before refactoring)
**AND** all existing tests SHALL pass without modification

## MODIFIED Requirements

### Requirement: Agent class focuses on component coordination
The QuantumNematodeAgent class SHALL transition from implementing episode logic directly to coordinating injected components.

**Before**: Agent implements run_episode with 268 lines of complex logic
**After**: Agent creates components in **init** and delegates run_episode to StandardEpisodeRunner in ~20 lines

#### Scenario: Agent initialization creates components
**GIVEN** a QuantumNematodeAgent constructor called with brain, env, and config
**WHEN** initialization completes
**THEN** the agent SHALL instantiate:
- SatietyManager with satiety config
- MetricsTracker with default initialization
- FoodConsumptionHandler with env and satiety manager
- StepProcessor with brain, env, food handler, satiety manager
- EpisodeRenderer with rendering config
**AND** SHALL NOT execute any episode logic during initialization

#### Scenario: Agent exposes components for advanced usage
**GIVEN** a QuantumNematodeAgent instance
**WHEN** external code needs direct access to components
**THEN** the agent SHALL provide read-only properties:
- agent.step_processor
- agent.metrics_tracker
- agent.satiety_manager
**AND** SHALL maintain backward-compatible attributes (brain, env, etc.)
