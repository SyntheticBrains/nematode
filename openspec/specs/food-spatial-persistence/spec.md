# food-spatial-persistence Specification

## Purpose

Defines requirements for food spatial persistence via configurable hotspot regions, satiety-gated food collection, and static food mode, enabling meaningful food-marking pheromone evaluation.

## Requirements

### Requirement: Food Spawning with Hotspot Bias

Food spawning SHALL use the hotspot bias probability to decide between hotspot-biased and uniform random candidate generation when hotspots and a positive bias are configured, and SHALL behave identically to uniform random placement otherwise.

#### Scenario: Hotspot bias configured

- **WHEN** `ForagingParams` is configured with `food_hotspots` and `food_hotspot_bias > 0`

- **THEN** food spawning (both initial placement and respawn) SHALL use the hotspot bias probability to decide between hotspot-biased and uniform random candidate generation

- **AND** hotspot-biased candidates SHALL be sampled from an exponential distribution centered on a randomly selected hotspot, with decay controlled by `food_hotspot_decay`

- **AND** hotspot selection SHALL be weighted by the hotspot `weight` values (higher weight = more spawns near that hotspot)

- **AND** candidates SHALL be clamped to grid bounds

- **AND** all existing constraints (`min_food_distance`, `agent_exclusion_radius`) SHALL still be enforced on hotspot-biased candidates

#### Scenario: No hotspot bias configured

- **WHEN** `ForagingParams` has `food_hotspots=None` or `food_hotspot_bias=0.0`

- **THEN** food spawning SHALL behave identically to current uniform random placement

### Requirement: Satiety-Gated Food Collection

When a satiety food threshold is set, food consumption SHALL be refused for agents whose satiety exceeds the threshold, and food collection SHALL behave identically to current behavior when no threshold is set.

#### Scenario: Agent attempts to consume food while gated

- **WHEN** `ForagingParams.satiety_food_threshold` is set (not None)

- **AND** an agent attempts to consume food

- **THEN** the food consumption logic SHALL check the agent's current satiety against `satiety_food_threshold * max_satiety`

- **AND** if the agent's satiety exceeds this value, food collection SHALL be refused (food remains on the grid, agent receives no satiety restoration)

- **AND** the agent SHALL still occupy the food cell without consuming it

#### Scenario: Food competition with sated agents

- **WHEN** food competition is resolved in multi-agent mode

- **THEN** sated agents (satiety > threshold) SHALL be excluded from the contested map before competition resolution

- **AND** food at a position with only sated agents SHALL remain on the grid unconsumed

#### Scenario: Reward when agent cannot eat due to satiety gate

- **WHEN** an agent is on food but cannot eat due to satiety gate

- **THEN** the reward calculator SHALL NOT award the goal bonus for that step (callers pass `can_eat=False` to `calculate_reward()`; the parameter defaults to True for backward compatibility)

#### Scenario: No satiety threshold configured

- **WHEN** `ForagingParams.satiety_food_threshold` is None (default)

- **THEN** food collection SHALL behave identically to current behavior (no satiety gate)

### Requirement: Food Respawn after Consumption

When food is consumed, a replacement SHALL be spawned using the same hotspot bias logic as initial placement unless no-respawn mode is enabled, in which case no replacement SHALL be spawned.

#### Scenario: Respawn enabled

- **WHEN** food is consumed and `ForagingParams.no_respawn` is False (default)

- **THEN** `spawn_food()` SHALL spawn a replacement food item using the same hotspot bias logic as initial placement

#### Scenario: Respawn disabled

- **WHEN** food is consumed and `ForagingParams.no_respawn` is True

- **THEN** no replacement food SHALL be spawned

- **AND** the food count on the grid SHALL decrease permanently

### Requirement: Food Hotspot Configuration

A YAML config that specifies `food_hotspots` SHALL define each hotspot as a `[x, y, weight]` list and SHALL validate hotspot coordinates and bias/decay parameters.

#### Scenario: YAML config specifies food hotspots

- **WHEN** a YAML config specifies `food_hotspots`

- **THEN** each hotspot SHALL be a list of `[x, y, weight]` where x and y are grid coordinates and weight is a positive float controlling relative spawn density

- **AND** the config loader SHALL validate that hotspot coordinates are within grid bounds

- **AND** `food_hotspot_bias` SHALL be validated as a float in range [0.0, 1.0]

- **AND** `food_hotspot_decay` SHALL be validated as a positive float

### Requirement: FoodHotspot Type

The `FoodHotspot` type SHALL be defined as `tuple[int, int, float]` matching the existing `TemperatureSpot` and `OxygenSpot` patterns.

#### Scenario: FoodHotspot type used

- **WHEN** `FoodHotspot` type is used

- **THEN** it SHALL be defined as `tuple[int, int, float]` matching the existing `TemperatureSpot` and `OxygenSpot` patterns

### Requirement: Hotspot Bias Composition

When both `food_hotspot_bias` and `safe_zone_food_bias` are configured, the biases SHALL compose independently and neither SHALL override or disable the other.

#### Scenario: Both biases configured

- **WHEN** both `food_hotspot_bias` and `safe_zone_food_bias` are configured

- **THEN** the biases SHALL compose independently: hotspot bias controls the sampling distribution, safe zone bias acts as a rejection filter afterward

- **AND** neither bias SHALL override or disable the other
