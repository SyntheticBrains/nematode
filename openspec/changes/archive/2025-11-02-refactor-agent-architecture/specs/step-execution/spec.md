# Spec Delta: Step Execution

## ADDED Requirements

### Requirement: Step processor encapsulates single-step logic

The system SHALL provide a `StepProcessor` class that encapsulates all logic for executing a single simulation step, including brain parameter preparation, action execution, and result processing.

#### Scenario: Execute single step with gradient information

**GIVEN** a StepProcessor with a brain and environment
**WHEN** process_step is called with gradient strength 0.5, gradient direction π/4, previous action FORWARD, and previous reward 1.0
**THEN** the StepProcessor SHALL:

- Construct BrainParams with the provided gradient and action data
- Call brain.forward() with the constructed parameters and current satiety
- Execute the returned action on the environment
- Return a StepResult containing the action, reward, done flag, and info dict

#### Scenario: Handle step with no previous action

**GIVEN** a StepProcessor at the start of an episode
**WHEN** process_step is called with previous_action=None and previous_reward=0.0
**THEN** the BrainParams SHALL be constructed with ActionData(action=None, reward=0.0)
**AND** the brain SHALL still produce a valid action

#### Scenario: Detect episode termination

**GIVEN** a StepProcessor during an episode
**WHEN** process_step is called and the agent reaches a goal or max steps
**THEN** the StepResult.done flag SHALL be True
**AND** the StepResult.info SHALL contain the termination reason

### Requirement: Step processor supports dependency injection

The StepProcessor class SHALL accept brain, environment, food handler, and satiety manager as constructor dependencies to enable testing with mocks.

#### Scenario: Instantiate with mocked dependencies

**GIVEN** mock objects for brain, environment, food_handler, and satiety_manager
**WHEN** a StepProcessor is instantiated with these mocks
**THEN** the StepProcessor SHALL use the provided dependencies
**AND** SHALL NOT create or access global state

### Requirement: Step processor is stateless

The StepProcessor SHALL NOT maintain episode state (path, history, counters) and SHALL only process individual steps based on provided inputs.

#### Scenario: Process multiple steps independently

**GIVEN** a StepProcessor instance
**WHEN** process_step is called multiple times with different inputs
**THEN** each call SHALL produce results independent of previous calls
**AND** no state SHALL be retained between calls
