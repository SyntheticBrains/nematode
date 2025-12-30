# Spec Delta: Food Consumption

## ADDED Requirements

### Requirement: Food consumption handler encapsulates food interaction logic

The system SHALL provide a `FoodConsumptionHandler` class that handles food detection, consumption, satiety restoration, and distance efficiency tracking.

#### Scenario: Consume food in static environment

**GIVEN** a FoodConsumptionHandler with a static maze environment
**WHEN** check_and_consume_food is called with agent position at a food location
**THEN** the handler SHALL:

- Return FoodConsumptionResult with food_consumed=True
- Restore satiety to 1.0 via the SatietyManager
- Remove the food from the environment
- Set distance_efficiency to None (not applicable for static environments)

#### Scenario: Consume food in dynamic foraging environment

**GIVEN** a FoodConsumptionHandler with a DynamicForagingEnvironment
**AND** initial distance to nearest food is 10 units
**WHEN** check_and_consume_food is called after the agent travels 12 steps to reach food
**THEN** the handler SHALL:

- Return FoodConsumptionResult with food_consumed=True
- Calculate distance_efficiency as 10/12 ≈ 0.833
- Restore satiety via the SatietyManager
- Update the environment to spawn new food

#### Scenario: Agent not at food location

**GIVEN** a FoodConsumptionHandler
**WHEN** check_and_consume_food is called with agent position not at any food
**THEN** the handler SHALL return FoodConsumptionResult with food_consumed=False
**AND** satiety SHALL NOT be restored
**AND** distance_efficiency SHALL be None

### Requirement: Food handler abstracts environment-specific behavior

The FoodConsumptionHandler SHALL detect the environment type and apply appropriate food consumption logic without the caller needing to check environment types.

#### Scenario: Transparent handling of different environment types

**GIVEN** a FoodConsumptionHandler initialized with either static or dynamic environment
**WHEN** check_and_consume_food is called
**THEN** the handler SHALL internally determine the correct behavior
**AND** the caller SHALL NOT need to use isinstance checks
**AND** the returned FoodConsumptionResult SHALL indicate environment type via presence/absence of distance_efficiency

### Requirement: Food handler integrates with satiety system

The FoodConsumptionHandler SHALL use the provided SatietyManager to restore satiety when food is consumed.

#### Scenario: Restore satiety on food consumption

**GIVEN** a FoodConsumptionHandler with a SatietyManager at satiety level 0.3
**WHEN** food is consumed
**THEN** the handler SHALL call satiety_manager.restore()
**AND** satiety SHALL be set to 1.0
**AND** the FoodConsumptionResult SHALL include the satiety_restored value
