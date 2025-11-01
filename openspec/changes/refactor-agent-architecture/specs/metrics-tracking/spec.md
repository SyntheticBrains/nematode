# Spec Delta: Metrics Tracking

## ADDED Requirements

### Requirement: Metrics tracker separates tracking from business logic
The system SHALL provide a `MetricsTracker` class that accumulates episode metrics independently from episode execution logic.

#### Scenario: Initialize metrics tracker
**GIVEN** a new episode is starting
**WHEN** a MetricsTracker is created
**THEN** it SHALL initialize with zero counters:
- success_count = 0
- total_steps = 0
- total_rewards = 0.0
- foods_collected = 0
- distance_efficiencies = []

#### Scenario: Track successful episode
**GIVEN** a MetricsTracker during an episode
**WHEN** track_episode_completion(success=True, steps=42, total_reward=15.5) is called
**THEN** success_count SHALL increment by 1
**AND** total_steps SHALL increment by 42
**AND** total_rewards SHALL increment by 15.5

#### Scenario: Track food collection in dynamic environment
**GIVEN** a MetricsTracker for a dynamic foraging environment
**WHEN** track_food_collection(distance_efficiency=0.85) is called
**THEN** foods_collected SHALL increment by 1
**AND** distance_efficiency 0.85 SHALL be appended to distance_efficiencies list

#### Scenario: Calculate final metrics
**GIVEN** a MetricsTracker with success_count=7, total_runs=10, total_steps=420, total_rewards=175.5, foods_collected=35, distance_efficiencies=[0.8, 0.9, ...]
**WHEN** calculate_metrics(total_runs=10) is called
**THEN** it SHALL return PerformanceMetrics with:
- success_rate = 0.7
- average_steps = 42.0
- average_reward = 17.55
- foraging_efficiency = 3.5 (foods/run)

### Requirement: Metrics tracker supports incremental updates
The MetricsTracker SHALL allow tracking individual steps and events as they occur, not just final episode results.

#### Scenario: Track individual steps
**GIVEN** a MetricsTracker during an episode
**WHEN** track_step(reward=0.1) is called multiple times
**THEN** each call SHALL accumulate the reward
**AND** step count SHALL be available for partial metrics calculation

### Requirement: Metrics tracker handles environment-agnostic and environment-specific metrics
The MetricsTracker SHALL track both universal metrics (success, steps, rewards) and environment-specific metrics (distance_efficiency for dynamic environments).

#### Scenario: Track metrics without distance efficiency
**GIVEN** a MetricsTracker for a static maze environment
**WHEN** episodes are tracked without calling track_food_collection()
**THEN** calculate_metrics() SHALL compute success_rate, average_steps, and average_reward
**AND** foraging_efficiency SHALL be 0.0 or None if no foods were tracked

#### Scenario: Track metrics with distance efficiency
**GIVEN** a MetricsTracker for a dynamic environment
**WHEN** track_food_collection() is called with efficiency values
**THEN** calculate_metrics() SHALL include average distance_efficiency
**AND** foraging_efficiency SHALL reflect foods_collected per run
