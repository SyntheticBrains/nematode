# mechanistic-baseline Specification

## Purpose

C.1's mechanistic baseline: a closed-loop roaming/dwelling worm on the primitives, with intake- and
satiety-dependent switching, klinokinetic reversal and omega hazards, weathervane steering, an edge
reversal on falling food contact, and a search term that decays off food. It has 20 parameters, and
every random draw comes from each animal's own generator. Optional off-food run kinematics, a per-state
speed spread and an off-food weathervane gain extend it to C.1 (v3)'s 25 parameters, and an optional
on-food dwelling reversal rate to 26; without them it is unchanged.

## Requirements

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

### Requirement: Optional off-food run kinematics and a per-state speed spread

`BatchedMechanistic` SHALL accept two optional mechanisms, each used only when its parameters are given:

- `v_off_food` and `sigma_off_food`: the run speed and curvature noise wherever food contact is 0, whatever the state;
- `speed_log_sd_dwell` and `speed_log_sd_roam`, given together: each state's run speed varies across 10-s windows as exp(sd·z) about its median.

The spread's z SHALL be the sine partner of that second's curvature normal, so that no random draw is added. Without these parameters, every action SHALL be byte-identical to the 20-parameter model's.

#### Scenario: Off-food kinematics change only food-free seconds

- **WHEN** an animal is on food
- **THEN** its run speed and curvature SHALL be its state's, with or without the off-food parameters

#### Scenario: The spread holds within a window

- **WHEN** the spread is given, and an animal's state and food contact stay unchanged within a window
- **THEN** its speed SHALL hold within that 10-s window, and SHALL vary across windows where its state's spread is positive (a spread of 0 leaves the speed unchanged); its curvature and hazards SHALL equal those without the spread

#### Scenario: One spread without the other is refused

- **WHEN** only `speed_log_sd_dwell` is given
- **THEN** the controller SHALL raise an error

### Requirement: An optional on-food dwelling reversal rate

`BatchedMechanistic` SHALL accept `r_dwell_on_food`, used only when given: a dwelling animal's reversal rate wherever food contact is above 0, in place of `r_dwell`, before the klinokinesis, edge and search terms. Off food, and while roaming, the state's rate SHALL apply. Without it, every action SHALL be byte-identical to the model's without the option.

#### Scenario: Off food nothing changes

- **WHEN** an animal is off food
- **THEN** its actions SHALL be those of the model without `r_dwell_on_food`

#### Scenario: On food only dwelling reversals change

- **WHEN** an animal is on food
- **THEN** its run speed, curvature, omega and delta hazards SHALL be unchanged, and its reversal hazard SHALL scale from `r_dwell` to `r_dwell_on_food` while dwelling and stay unchanged while roaming
