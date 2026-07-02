## MODIFIED Requirements

### Requirement: Capture-radius food consumption

The continuous-2D environment SHALL consume a food source when the agent is within a configurable capture radius of it (Euclidean distance), replacing exact grid-cell-equality consumption. When the config-gated source-depletion dynamic is enabled, a consume event SHALL instead **decrement** the matched source's remaining amount once per event (the source persists at reduced amplitude), removing and respawning it only when it crosses the exhaustion threshold; with depletion disabled (the default) consumption removes the source outright as before.

#### Scenario: Food consumed within radius

- **WHEN** the agent's position is within the capture radius of a food source
- **THEN** that food is consumed (removed) and the foraging reward/satiety update fires exactly once for it

#### Scenario: Food not consumed outside radius

- **WHEN** the agent is farther than the capture radius from every food source
- **THEN** no food is consumed that step

#### Scenario: Depletion decrements instead of removing

- **WHEN** source-depletion is enabled and the agent consumes a source within the capture radius
- **THEN** the matched source's remaining amount SHALL be decremented (the source persists at its position with reduced amplitude), and the source SHALL be removed and respawned only when its remaining amount crosses the exhaustion threshold
