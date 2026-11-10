## ADDED Requirements

### Requirement: On-food reversal speed that follows the run speed

Under `continuous.on_food_reversal_speed: follows_run_speed`, a reversal bout that starts on a lawn
SHALL back at the `on_food_reversal_speed_coupling` entry's line applied to the worm's recent run
speed, clipped to the entry's range: clip(intercept + slope × v̄, low, high).

- **v̄:** the mean of the commanded run speeds of the worm's last N run steps, oldest first, with
  N = round(window ÷ step).
- **Fallback:** until the worm has taken N run steps, the reversal SHALL take the pooled on-food speed.

The window SHALL take each run step's speed. Bout steps SHALL neither enter it nor clear it, and it
SHALL persist across bouts and assay phases.

The reversal's duration SHALL still be drawn from the on-food gamma. Reversals that start off every
lawn, and omega and delta bouts, SHALL be unaffected.

`on_food_reversal_speed` SHALL be `pooled` when unset, and runs SHALL then be byte-identical to before
the option. Under either setting, the per-step runner and the batched evaluator SHALL make the same RNG
calls in the same order, and SHALL give each reversal the same speed. Setting `on_food_reversal_speed`
without `locomotion: primitives` SHALL be refused, and so SHALL `follows_run_speed` with
`reversal_kinematics: off_food`.

#### Scenario: A reversal after steady runs follows their speed

- **WHEN** the option is on, and a worm on a lawn has run at 0.02 mm/s for its last 3 s of run steps
  before a reversal starts
- **THEN** the reversal SHALL back at intercept + slope × 0.02 mm/s, from the vendored entry
- **AND** its duration SHALL be drawn from gamma(2.047 s, SD 0.899 s)

#### Scenario: The speed is clipped to the measured range

- **WHEN** the option is on and a worm's recent run speed puts the line above the entry's high or
  below its low
- **THEN** the reversal SHALL back at the high or the low
- **AND** a worm that has been still for its last 3 s of run steps SHALL back at the low, 0.048 mm/s

#### Scenario: Before the window fills, the pooled speed

- **WHEN** the option is on and a reversal starts on a lawn before the worm has taken N run steps
- **THEN** it SHALL back at the pooled on-food speed, 0.069 mm/s

#### Scenario: Bouts do not enter the window

- **WHEN** a worm runs for 3 s, makes an omega turn, then starts a reversal on a lawn
- **THEN** the reversal's speed SHALL follow the 3 s of runs before the omega

#### Scenario: A shorter step keeps a 3-s window

- **WHEN** `step_seconds` is 0.5 and the option is on
- **THEN** the window SHALL hold the last 6 run steps

#### Scenario: Off the lawn nothing changes

- **WHEN** the option is on and a reversal starts off every lawn, or in an assay phase without food
- **THEN** it SHALL draw the off-food entries, 2.69 ± 1.75 s at 0.14 mm/s

#### Scenario: Unset, nothing changes

- **WHEN** a primitives config leaves `on_food_reversal_speed` unset, or sets it to `pooled`
- **THEN** its runs SHALL be byte-identical to before the option, RNG state included

#### Scenario: Both runners give each reversal the same speed

- **WHEN** the batched evaluator and the per-step runner run the same seed on a lawn assay with the
  option on
- **THEN** their bouts' kinds and start steps SHALL match exactly over 300 steps
- **AND** each reversal's duration and speed SHALL match

#### Scenario: The option is refused where it cannot act

- **WHEN** a config sets `on_food_reversal_speed` with `locomotion: free`, or `follows_run_speed`
  with `reversal_kinematics: off_food`
- **THEN** loading SHALL raise, naming both settings

#### Scenario: The coupling entry is measured and checked

- **WHEN** the kinematics are loaded
- **THEN** `on_food_reversal_speed_coupling` SHALL have status `measured` with an existing script and
  record
- **AND** loading SHALL raise if its intercept is not positive, its slope is negative, its low is not
  below its high, or its window is not positive
