## ADDED Requirements

### Requirement: Food ingestion in BrainParams

`BrainParams` SHALL carry `food_ingested`, the intake value of the worm's last move under the lawn food
model, and None otherwise.

#### Scenario: A lawn step reports what was eaten

- **WHEN** the worm eats on a lawn
- **THEN** the next step's `BrainParams.food_ingested` SHALL equal that step's intake value
