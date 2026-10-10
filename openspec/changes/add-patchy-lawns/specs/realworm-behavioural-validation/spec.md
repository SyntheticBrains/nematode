## ADDED Requirements

### Requirement: Roaming and dwelling from speed

The validation package SHALL classify behaviour into roaming and dwelling as Ji et al. 2021 classified
their tracked animals: each time point's median and variance of speed over a sliding 20-second window,
with a two-state hidden Markov model with Gaussian emissions fitted on the vendored real-worm track and
applied unchanged to simulated tracks. States SHALL be read only while a worm is on food.

#### Scenario: Real and simulated worms are read alike

- **WHEN** a real track and a simulated track are resampled to 5-second steps
- **THEN** both SHALL be classified by the same fitted model, with no refitting on simulated data

#### Scenario: The fit is checked against the authors' labels

- **WHEN** the model is fitted on the vendored real track
- **THEN** its agreement with the authors' own roaming/dwelling labels on that track SHALL be reported,
  and the instrument SHALL NOT be used if the fit does not reproduce them

#### Scenario: Off-food windows are not classified

- **WHEN** a simulated worm is off every lawn
- **THEN** its windows SHALL be reported as off-food and SHALL NOT count toward either state

### Requirement: Behaviour capture on lawns

When behaviour is captured under `food_model: lawns`, each step SHALL also record the worm's satiety, its
intake, and whether it is on a lawn.

#### Scenario: A lawn step is recorded

- **WHEN** a worm eats on a lawn with behaviour capture on
- **THEN** that step's record SHALL carry its satiety, a positive intake, and that it is on a lawn
