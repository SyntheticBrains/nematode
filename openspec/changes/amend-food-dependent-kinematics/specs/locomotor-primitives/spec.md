## MODIFIED Requirements

### Requirement: Kinematics from data

Every primitive's fixed kinematics SHALL be read from `data/locomotion/primitives.json`. Each entry
SHALL carry its value, its source with figure or table, its condition (on or off food), and its status:

- `verified`, read in the text;
- `figure`, read from a figure;
- `measured`, computed from deposited tracks by a committed script, its source naming the script and
  its record.

A test SHALL pin each value to its entry.

#### Scenario: Executed bouts match their source

- **WHEN** each primitive is forced 2,000 times and every bout is measured from the trajectory alone
- **THEN** its whole-step durations SHALL match source draws rounded to whole steps
- **AND** its reversal distances and turn angles SHALL match the vendored distributions directly (each
  a two-sample KS, p > 0.01)
- **AND** reversals SHALL be forced both on a lawn and off every lawn, each matched to its condition's
  entries

#### Scenario: A measured value without its record is refused

- **WHEN** an entry has status `measured` and its source names no script and record
- **THEN** loading the kinematics SHALL raise, naming the entry

## ADDED Requirements

### Requirement: Food-dependent reversal kinematics

Under `continuous.reversal_kinematics: food_dependent`, the default under primitives, a reversal bout
that starts while the worm is on a lawn SHALL draw its duration and speed from the on-food entries. One
that starts off every lawn SHALL draw them from the off-food entries. The worm is on a lawn when its
position at the start of the step lies in a lawn's cell footprint, in a phase that has lawns, whatever
the cell's density. A bout's kinematics SHALL be fixed when it starts.

Under `reversal_kinematics: off_food` every reversal SHALL draw the off-food entries, and runs SHALL be
byte-identical to before the option. Either setting SHALL draw the same number of values from the RNG.
Setting `reversal_kinematics` with `locomotion: free` SHALL be refused.

Omega and delta bouts SHALL be unaffected.

#### Scenario: A reversal on a lawn is short and slow

- **WHEN** a reversal starts on a lawn under the default
- **THEN** its duration SHALL be drawn from gamma(2.047 s, SD 0.899 s), and it SHALL back at
  0.069 mm/s

#### Scenario: A reversal off the lawn keeps the published values

- **WHEN** a reversal starts off every lawn, or in an assay phase without food
- **THEN** its duration SHALL be drawn from gamma(2.69 s, SD 1.75 s), and it SHALL back at 0.14 mm/s

#### Scenario: A depleted cell is still a lawn

- **WHEN** a reversal starts on a lawn cell whose density is zero
- **THEN** it SHALL draw the on-food entries

#### Scenario: A bout that crosses the edge keeps its kinematics

- **WHEN** a reversal that started on a lawn carries the worm off it
- **THEN** its speed and remaining duration SHALL not change

#### Scenario: The off-food setting reproduces earlier records

- **WHEN** a primitives config sets `reversal_kinematics: off_food`
- **THEN** its runs SHALL be byte-identical to before the option, RNG draws included

#### Scenario: Both runners draw the same

- **WHEN** the batched evaluator and the per-step runner run the same seed on a lawn assay under the
  default
- **THEN** their bouts' kinds and start steps SHALL match exactly over 300 steps, and each reversal's
  duration SHALL match

#### Scenario: The option without primitives is refused

- **WHEN** a config sets `reversal_kinematics` with `locomotion: free`
- **THEN** loading SHALL raise, naming both settings
