## ADDED Requirements

### Requirement: The lawn food model

The continuous environment SHALL offer `foraging.food_model: lawns`, which places disc lawns, each with a
density grid of cells about a body length wide and a quality, in place of point food sources. The default,
`points`, SHALL leave every existing behaviour byte-identical.

#### Scenario: The default is unchanged

- **WHEN** a configuration omits `food_model` or sets it to `points`
- **THEN** food placement, fields, consumption and rewards SHALL be identical to before

#### Scenario: Lawns are placed as separate patches

- **WHEN** an environment is built with `food_model: lawns`
- **THEN** each lawn SHALL lie inside the arena, its edge at least `min_separation_mm` from every other
  lawn's, and every cell inside its disc SHALL start at density 1

### Requirement: The lawn odour field

The food concentration at a point SHALL be the sum, over every lawn cell, of the cell's remaining density
times its share of its lawn's area times the configured per-source kernel at the cell's distance,
normalised as point sources are, so a full lawn smells like one point source of strength 1. The food
gradient vector SHALL sum over cells with the same weights. A lawn's quality SHALL NOT change its odour.

#### Scenario: A grazed region smells weaker

- **WHEN** the cells near one side of a lawn are depleted
- **THEN** the concentration there SHALL fall and the gradient inside the lawn SHALL point toward the
  remaining cells

#### Scenario: A full lawn smells like one source

- **WHEN** a full lawn and a point source of strength 1 at its centre are read at a point more than ten
  lawn radii away
- **THEN** the two concentrations SHALL agree within 5%

#### Scenario: Quality is not smelled

- **WHEN** two lawns differ only in quality
- **THEN** their odour fields SHALL be identical

### Requirement: Continuous intake

Each step, a worm inside a lawn SHALL eat `intake_fraction` of the density of the cell it is on, which
the cell SHALL lose. The worm SHALL gain reward and satiety in proportion to what it ate times the lawn's
quality. Intake SHALL NOT depend on the worm's speed.

#### Scenario: Eating depletes the cell under the worm

- **WHEN** a worm stays on one cell for several steps
- **THEN** that cell's density SHALL fall geometrically, and the worm's intake with it

#### Scenario: Speed does not change intake

- **WHEN** two worms occupy cells of equal density, one moving and one still
- **THEN** their intake SHALL be equal

#### Scenario: Off a lawn there is no intake

- **WHEN** a worm is outside every lawn
- **THEN** its intake SHALL be zero

### Requirement: No shaping that favours a state

A configuration with `food_model: lawns` SHALL be refused if it sets a non-zero `penalty_stuck_position`,
`penalty_anti_dithering`, `reward_exploration` or `reward_distance_scale`. Under lawns there SHALL be no capture event, so the goal
bonus never applies and intake is the only food reward.

#### Scenario: A dwelling penalty is refused

- **WHEN** a lawn configuration sets `penalty_stuck_position` above zero
- **THEN** loading it SHALL fail with a message naming the key

#### Scenario: Exploration is not paid

- **WHEN** a lawn configuration sets `reward_exploration` above zero
- **THEN** loading it SHALL fail with a message naming the key

### Requirement: Lawn episode outcomes

Under `food_model: lawns`, an episode SHALL end survived at `max_steps` or starved, and each episode's
intake and outcome SHALL be written to the run's summary and its log line.

#### Scenario: Intake is recorded per episode

- **WHEN** a lawn episode ends
- **THEN** its total intake and its outcome SHALL appear in the run's summary CSV

### Requirement: The lawn model's scope

A configuration with `food_model: lawns` SHALL be refused if it runs more than one agent or off the
continuous substrate, and a lawn environment SHALL refuse a pixel renderer, which would draw no lawns. Point-food keys (`foods_on_grid`, `target_foods_to_collect`) SHALL be ignored under
lawns rather than validated.

#### Scenario: Multi-agent lawns are refused

- **WHEN** a lawn configuration declares two agents
- **THEN** loading it SHALL fail with a message naming the lawn model
