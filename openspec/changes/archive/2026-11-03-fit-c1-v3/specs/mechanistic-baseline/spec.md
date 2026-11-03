## ADDED Requirements

### Requirement: Optional off-food run kinematics and a per-state speed spread

`BatchedMechanistic` SHALL accept two optional mechanisms, each used only when its parameters are given:

- `v_off_food` and `sigma_off_food`: the run speed and curvature noise wherever food contact is 0, whatever the state;
- `speed_log_sd_dwell` and `speed_log_sd_roam`, given together: each state's run speed varies across 10-s windows as exp(sd·z) about its median.

The spread's z SHALL be the sine partner of that second's curvature normal, so that no random draw is added. Without these parameters, every action SHALL be byte-identical to the 20-parameter model's.

#### Scenario: Off-food kinematics change only food-free seconds

- **WHEN** an animal is on food
- **THEN** its run speed and curvature SHALL be its state's, with or without the off-food parameters

#### Scenario: The spread holds within a window

- **WHEN** the spread is given
- **THEN** an animal's speed SHALL hold within each 10-s window and vary across windows, and its curvature and hazards SHALL equal those without the spread

#### Scenario: One spread without the other is refused

- **WHEN** only `speed_log_sd_dwell` is given
- **THEN** the controller SHALL raise an error
