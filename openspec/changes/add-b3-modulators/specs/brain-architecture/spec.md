## ADDED Requirements

### Requirement: Food ingestion in BrainParams

`BrainParams` SHALL carry `food_ingested`, the intake value of the worm's last move under the lawn food
model, and `max_food_ingested`, the largest intake value one move can yield, both None otherwise.

#### Scenario: A lawn step reports what was eaten

- **WHEN** the worm eats on a lawn
- **THEN** the next step's `BrainParams.food_ingested` SHALL equal that step's intake value

#### Scenario: The largest bite is reported beside it

- **WHEN** the lawn food model is on
- **THEN** `BrainParams.max_food_ingested` SHALL equal the intake fraction times the best lawn's
  quality
