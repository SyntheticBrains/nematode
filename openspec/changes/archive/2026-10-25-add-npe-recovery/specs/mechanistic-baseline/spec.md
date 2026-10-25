## ADDED Requirements

### Requirement: C.1's mechanistic controller

A batched controller `BatchedMechanistic` SHALL drive the primitives from 20 parameters:

- a roaming/dwelling state whose switching hazards depend log-linearly on intake and satiety;
- per-state run speed and curvature noise, with weathervane bias from the head-sweep difference;
- per-state reversal and omega hazards, scaled by klinokinesis on dC/dt;
- a reversal edge response to falling food contact;
- a search multiplier that decays with time off food.

It SHALL draw every random number from each animal's own generator, and refuse `noise=None`.

#### Scenario: Klinokinesis lowers hazards up-gradient

- **WHEN** dC/dt is positive and β_r > 0
- **THEN** the reversal hazard SHALL be lower than at dC/dt = 0

#### Scenario: Leaving food raises reversals

- **WHEN** food contact falls between two seconds and e > 0
- **THEN** that second's reversal hazard SHALL rise by exp(e × the fall)

#### Scenario: Search decays off food

- **WHEN** an animal has been off food for t seconds with A > 0
- **THEN** its hazards SHALL be multiplied by 1 + A·exp(−t/τ)

#### Scenario: An animal is the same alone and in a batch

- **WHEN** one animal runs alone and inside a batch
- **THEN** its actions SHALL be the same
