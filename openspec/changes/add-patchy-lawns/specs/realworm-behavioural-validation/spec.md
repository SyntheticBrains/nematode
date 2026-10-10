## ADDED Requirements

### Requirement: Roaming and dwelling from speed and angular speed

The validation package SHALL classify behaviour into roaming and dwelling from each 10-second window's
speed and angular speed, measured from positions sampled once per 5-second step. A window SHALL be a
roaming observation when its speed times a slope exceeds its angular speed, and a two-state hidden
Markov model SHALL decode the observations within each run of on-food windows. The slope SHALL be
calibrated on vendored real-worm tracks, measured the same way, against their authors' labels, and the
slope and model SHALL be applied unchanged to simulated tracks.

#### Scenario: Real and simulated worms are read alike

- **WHEN** a real track resampled to 5-second steps and a simulated track are classified
- **THEN** both SHALL be measured by the same functions and decoded with the same slope and model,
  with no refitting on simulated data

#### Scenario: The calibration is checked on held-out animals

- **WHEN** the slope is calibrated on half of the real animals
- **THEN** its agreement with the authors' labels on the other half SHALL be reported as Cohen's
  kappa, and the instrument SHALL NOT be used if kappa is below 0.6

#### Scenario: Off-food windows are not classified

- **WHEN** a simulated worm is off every lawn
- **THEN** its windows SHALL be reported as off-food and SHALL NOT count toward either state

### Requirement: Behaviour capture on lawns

When behaviour is captured under `food_model: lawns`, each step SHALL also record the worm's satiety, its
intake, and whether it is on a lawn.

#### Scenario: A lawn step is recorded

- **WHEN** a worm eats on a lawn with behaviour capture on
- **THEN** that step's record SHALL carry its satiety, a positive intake, and that it is on a lawn
