# trajectory-learning Capability Specification

## Purpose

Enable quantum ModularBrain to learn from complete episode trajectories using discounted returns for temporal credit assignment, matching the REINFORCE algorithm used by classical MLP.

## ADDED Requirements

### Requirement: Episode Trajectory Buffering

The ModularBrain SHALL buffer complete episode trajectories when trajectory learning is enabled.

#### Scenario: Episode Data Accumulation

**Given** trajectory learning is enabled via config (`use_trajectory_learning: true`)
**When** the brain processes each time step during an episode
**Then** it SHALL store the BrainParams, ActionData, and immediate reward for that step
**And** SHALL NOT perform parameter updates until the episode completes
**And** SHALL accumulate all data in episode-specific buffers

#### Scenario: Episode Boundary Handling

**Given** an episode completes (via `post_process_episode()` call)
**When** trajectory learning is enabled
**Then** the brain SHALL compute discounted returns from the buffered rewards
**And** SHALL compute trajectory-level gradients using the returns
**And** SHALL update parameters once using accumulated gradients
**And** SHALL clear the episode buffers for the next episode

#### Scenario: Backward Compatibility

**Given** trajectory learning is disabled via config (`use_trajectory_learning: false` or omitted)
**When** the brain processes a time step
**Then** it SHALL use immediate single-step learning (current behavior)
**And** SHALL NOT buffer episode data
**And** SHALL update parameters immediately as before

### Requirement: Discounted Return Computation

The ModularBrain SHALL compute discounted returns backward through episode trajectories using a configurable discount factor gamma.

#### Scenario: Backward Accumulation

**Given** an episode with rewards [r_0, r_1, r_2, ..., r_T]
**When** computing discounted returns
**Then** it SHALL iterate backward from r_T to r_0
**And** SHALL compute G_t = r_t + gamma * G_{t+1} for each timestep
**And** SHALL return the list [G_0, G_1, ..., G_T]

#### Scenario: Discount Factor Configuration

**Given** a config specifies `gamma: 0.99`
**When** the brain is initialized
**Then** it SHALL use gamma=0.99 for return computation
**And** SHALL validate that 0 <= gamma <= 1
**And** SHALL use gamma=0.99 as default if not specified

#### Scenario: Terminal State Handling

**Given** an episode terminates at step T
**When** computing G_T
**Then** G_T SHALL equal r_T (no future returns)
**And** SHALL bootstrap correctly for non-terminal timeouts

### Requirement: Trajectory Parameter-Shift Gradients

The ModularBrain SHALL compute parameter-shift gradients using discounted returns instead of immediate rewards.

#### Scenario: Gradient Linearity

**Given** an episode with discounted returns [G_0, G_1, ..., G_T]
**When** computing gradients for parameter theta_i
**Then** it SHALL evaluate shifted circuits at theta_i + shift and theta_i - shift
**And** SHALL compute grad_i = 0.5 * sum_t (P_+(action_t) - P_-(action_t)) * G_t
**And** SHALL accumulate contributions across all timesteps

#### Scenario: Action Probability Recomputation

**Given** stored episode actions [a_0, a_1, ..., a_T]
**When** computing trajectory gradients
**Then** it SHALL re-evaluate quantum circuits for each timestep's parameters
**And** SHALL compute action probabilities for both shifted parameter values
**And** SHALL use the stored action indices for gradient computation

#### Scenario: Circuit Batching Optimization

**Given** trajectory learning enabled
**When** evaluating circuits for gradient computation
**Then** it MAY batch circuit evaluations across timesteps for efficiency
**And** SHALL produce mathematically equivalent gradients to single-step evaluation

### Requirement: Configuration Schema Extension

The system SHALL extend the ModularBrain configuration schema to support trajectory learning parameters.

#### Scenario: Trajectory Learning Toggle

**Given** a YAML config file
**When** specifying brain configuration
**Then** it SHALL accept `use_trajectory_learning: bool` (default: false)
**And** SHALL validate the boolean value
**And** SHALL pass this flag to ModularBrain initialization

#### Scenario: Discount Factor Configuration

**Given** trajectory learning is enabled
**When** specifying `gamma: float` in config
**Then** it SHALL validate that 0.0 <= gamma <= 1.0
**And** SHALL use gamma=0.99 as default if not specified
**And** SHALL apply this gamma value in return computation

#### Scenario: Optional Variance Reduction

**Given** trajectory learning configuration
**When** specifying optional parameters
**Then** it MAY accept `baseline_alpha: float` for baseline subtraction
**And** MAY accept `normalize_returns: bool` for return normalization
**And** SHALL provide sensible defaults if omitted

### Requirement: Learning Trigger Integration

The agent runner SHALL trigger trajectory updates at episode boundaries when trajectory learning is enabled.

#### Scenario: Episode Completion Detection

**Given** trajectory learning is enabled
**When** an episode terminates (goal reached, starved, max steps, or predator death)
**Then** the runner SHALL call `brain.post_process_episode()`
**And** the brain SHALL perform trajectory-level parameter updates
**And** SHALL return control to the runner for the next episode

#### Scenario: Mid-Episode Behavior

**Given** trajectory learning is enabled
**When** processing a non-terminal timestep
**Then** the runner SHALL NOT trigger parameter updates
**And** SHALL continue episode execution with current parameters
**And** SHALL only update at episode boundary

## MODIFIED Requirements

### Requirement: ModularBrain Initialization

The ModularBrain initialization SHALL support both single-step and trajectory learning modes.

#### Scenario: Trajectory Learning Initialization

**Given** a config with `use_trajectory_learning: true`
**When** ModularBrain is instantiated
**Then** it SHALL initialize episode buffers (params, actions, rewards)
**And** SHALL set the discount factor gamma from config
**And** SHALL initialize baseline tracking if baseline_alpha is specified
**And** SHALL configure trajectory-mode learning triggers

#### Scenario: Single-Step Learning Initialization

**Given** a config with `use_trajectory_learning: false` or omitted
**When** ModularBrain is instantiated
**Then** it SHALL NOT allocate episode buffers
**And** SHALL use immediate reward-based learning (existing behavior)
**And** SHALL maintain full backward compatibility
