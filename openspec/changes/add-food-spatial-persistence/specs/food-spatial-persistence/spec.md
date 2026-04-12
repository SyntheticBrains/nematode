## Purpose

Defines requirements for food spatial persistence via configurable hotspot regions, satiety-gated food collection, and static food mode, enabling meaningful food-marking pheromone evaluation.

## MODIFIED Requirement: Food Spawning with Hotspot Bias

- WHEN `ForagingParams` is configured with `food_hotspots` and `food_hotspot_bias > 0`

- THEN food spawning (both initial placement and respawn) SHALL use the hotspot bias probability to decide between hotspot-biased and uniform random candidate generation

- AND hotspot-biased candidates SHALL be sampled from an exponential distribution centered on a randomly selected hotspot, with decay controlled by `food_hotspot_decay`

- AND hotspot selection SHALL be weighted by the hotspot `weight` values (higher weight = more spawns near that hotspot)

- AND candidates SHALL be clamped to grid bounds

- AND all existing constraints (`min_food_distance`, `agent_exclusion_radius`) SHALL still be enforced on hotspot-biased candidates

- WHEN `ForagingParams` has `food_hotspots=None` or `food_hotspot_bias=0.0`

- THEN food spawning SHALL behave identically to current uniform random placement

## ADDED Requirement: Satiety-Gated Food Collection

- WHEN `ForagingParams.satiety_food_threshold` is set (not None)

- AND an agent attempts to consume food

- THEN the environment SHALL check the agent's current satiety against `satiety_food_threshold * max_satiety`

- AND if the agent's satiety exceeds this value, food collection SHALL be refused (food remains on the grid, agent receives no satiety restoration)

- AND the agent SHALL still occupy the food cell without consuming it

- WHEN `ForagingParams.satiety_food_threshold` is None (default)

- THEN food collection SHALL behave identically to current behavior (no satiety gate)

## MODIFIED Requirement: Food Respawn after Consumption

- WHEN food is consumed and `ForagingParams.no_respawn` is False (default)

- THEN `spawn_food()` SHALL spawn a replacement food item using the same hotspot bias logic as initial placement

- WHEN food is consumed and `ForagingParams.no_respawn` is True

- THEN no replacement food SHALL be spawned

- AND the food count on the grid SHALL decrease permanently

## ADDED Requirement: Food Hotspot Configuration

- WHEN a YAML config specifies `food_hotspots`

- THEN each hotspot SHALL be a list of `[x, y, weight]` where x and y are grid coordinates and weight is a positive float controlling relative spawn density

- AND the config loader SHALL validate that hotspot coordinates are within grid bounds

- AND `food_hotspot_bias` SHALL be validated as a float in range [0.0, 1.0]

- AND `food_hotspot_decay` SHALL be validated as a positive float

## ADDED Requirement: FoodHotspot Type

- WHEN `FoodHotspot` type is used

- THEN it SHALL be defined as `tuple[int, int, float]` matching the existing `TemperatureSpot` and `OxygenSpot` patterns

## ADDED Requirement: Hotspot Bias Composition

- WHEN both `food_hotspot_bias` and `safe_zone_food_bias` are configured

- THEN the biases SHALL compose independently: hotspot bias controls the sampling distribution, safe zone bias acts as a rejection filter afterward

- AND neither bias SHALL override or disable the other
