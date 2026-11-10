# locomotor-primitives Specification

## Purpose

The 1-second point worm can move only by locomotor primitives: runs, reversal bouts, and omega and
delta turns. Their kinematics are fixed from published data. Every controller drives them the same
way, through five numbers per second: run speed, run curvature, and three start hazards. With the
primitives come a food-contact sense and a locomotor-state sense, and two non-learning controllers
that serve as the layer's positive control and the benchmark's Poisson negative.

## Requirements

### Requirement: The primitive layer

The continuous environment SHALL offer `continuous.locomotion: primitives` beside the default `free`.
Under it, the point worm SHALL move only by:

- **runs**: one step at a controller-set speed in [0, v_max] and curvature in [−ω_max, ω_max];
- **reversal bouts**: backing at a fixed speed along an unchanged heading for a drawn duration;
- **omega and delta bouts**: a drawn heading change, toward the episode's ventral side for an omega and
  its dorsal side for a delta, over a drawn duration, without translation.

A pause SHALL be a run at zero speed. The ventral side SHALL be drawn once per episode. During a bout
the controller's outputs SHALL move nothing. With `locomotion: free` the environment SHALL draw nothing
new from its RNG, and its runs SHALL be byte-identical to before the option.

#### Scenario: A reversal backs without turning

- **WHEN** a reversal bout of drawn duration D runs at a 1-second step
- **THEN** the worm SHALL move `v_rev · D` backward along its heading, over `⌈D⌉` steps, with its
  heading unchanged

#### Scenario: Omega and delta turn to opposite sides

- **WHEN** an episode's worm makes an omega and a delta
- **THEN** their heading changes SHALL have opposite signs

#### Scenario: Free locomotion is unchanged

- **WHEN** a config leaves `locomotion` at `free`
- **THEN** its runs SHALL be byte-identical to before the option, RNG draws included

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

### Requirement: The hazard interface

Under `action_space: primitives` a controller SHALL emit five numbers in `[-1, 1]` each step:

- run speed, mapped linearly to [0, v_max];
- run curvature, mapped linearly to [−ω_max, ω_max];
- a start hazard each for reversal, omega and delta, mapped log-linearly to [10⁻⁴, 1] per second.

While no bout runs, a bout SHALL start with probability `1 − exp(−Λ Δt)`, `Λ` the hazards' sum, and
its kind SHALL be drawn in proportion to its hazard, both from the environment's RNG.

#### Scenario: Constant hazards are a Poisson process

- **WHEN** a controller holds every hazard constant
- **THEN** the intervals between bout starts, less the bouts themselves, SHALL be geometric at the
  per-step rate `1 − exp(−Λ Δt)`

### Requirement: Bout boundaries

A start SHALL be drawn only at the beginning of a step with no bout running. A bout's partial last
step SHALL end it, and no bout SHALL start within that step.

#### Scenario: No start inside a bout's last step

- **WHEN** a reversal of 2.4 s runs at a 1-second step
- **THEN** it SHALL occupy three steps, the third moving 0.4 s's distance
- **AND** the earliest next start SHALL be drawn at the fourth step

### Requirement: A food-contact sense

A sensory module `food_contact` SHALL report the density of the lawn cell under the worm, and 0 off
every lawn, independent of the lawn's quality. It SHALL be refused unless `food_model` is `lawns`.

#### Scenario: Crossing the edge

- **WHEN** a worm moves from a fresh lawn cell to bare agar
- **THEN** its food-contact feature SHALL fall from 1 to 0

### Requirement: A locomotor-state sense

A sensory module `locomotor_state` SHALL report a one-hot of the running primitive (run, reversal,
omega, delta). It SHALL be refused unless `locomotion` is `primitives`.

#### Scenario: A reversal is felt

- **WHEN** a reversal bout runs
- **THEN** the module SHALL report reversal on each of its steps, and run on the step after it ends

### Requirement: Primitives are scoped to the 1-second point worm

`locomotion: primitives` SHALL require the point body, `allow_reversal: false`, an environment step
of at most one second, and a single worm. A multi-agent run SHALL be refused when the environment or
any agent's brain uses the primitives.

#### Scenario: A multi-agent run is refused

- **WHEN** a multi-agent config gives one agent a primitives brain
- **THEN** loading SHALL fail, naming the multi-agent setting

#### Scenario: The 5-second worm is refused

- **WHEN** a config sets `locomotion: primitives` without `step_seconds`, at its 5-second reference step
- **THEN** loading SHALL fail, naming the step

### Requirement: Non-learning hazard controllers

The package SHALL provide `ConstantHazards` (fixed speed, zero curvature, fixed hazards) and
`PirouetteHazards` (start hazards `λ_0 · exp(−β · d)`, with `d` the food derivative feature as brains
see it, and straight runs). Each SHALL implement the Brain protocol, learn nothing, and run through the
same runner as a brain.

#### Scenario: Klinokinesis is expressible and timing alone is not

- **WHEN** worms start 6 mm off a central lawn at uniform headings on seeds 5994–5995, 32 episodes each, and run for 1,200
  one-second steps
- **THEN** `PirouetteHazards` SHALL reach the lawn more often than `ConstantHazards` at its mean hazard
  and speed, bootstrap-significant at 95%
- **AND** the pirouette rule with its sign inverted SHALL reach it less often than `ConstantHazards` at
  its own mean hazard

### Requirement: Food-dependent reversal kinematics

Under `continuous.reversal_kinematics: food_dependent`, the default under primitives, a reversal bout
that starts while the worm is on a lawn SHALL draw its duration and speed from the on-food entries. One
that starts off every lawn SHALL draw them from the off-food entries. The worm is on a lawn when its
position at the start of the step lies in a lawn's cell footprint, in a phase that has lawns, whatever
the cell's density. A bout's kinematics SHALL be fixed when it starts.

Under `reversal_kinematics: off_food` every reversal SHALL draw the off-food entries, and runs SHALL be
byte-identical to before the option. Under either setting, the per-step runner and the batched evaluator
SHALL make the same RNG calls in the same order. Setting `reversal_kinematics` with `locomotion: free`
SHALL be refused.

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

### Requirement: On-food reversal speed that follows the run speed

Under `continuous.on_food_reversal_speed: follows_run_speed`, a reversal bout that starts on a lawn
SHALL back at a speed interpolated linearly between the `on_food_reversal_speed_coupling` entry's
knots at the worm's recent run speed v̄, and flat beyond the outermost knot. Each knot pairs a mean
speed before with real worms' mean reversal speed at it.

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
- **THEN** the reversal SHALL back at the entry's knots interpolated at 0.02 mm/s
- **AND** its duration SHALL be drawn from gamma(2.047 s, SD 0.899 s)

#### Scenario: Between the knots the speed is interpolated, beyond them flat

- **WHEN** the option is on and a worm's recent run speed lies between two knots
- **THEN** the reversal SHALL back at the speed on the line joining them
- **AND** a worm slower than the first knot or faster than the last SHALL back at that knot's speed:
  a still worm at 0.048 mm/s, and one running at 0.15 mm/s at 0.096 mm/s

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

- **WHEN** the batched evaluator and the per-step runner run the same seeds on a lawn assay with the
  option on, under a controller whose run speed changes every step inside the knots
- **THEN** their bouts' kinds and start steps SHALL match exactly over 300 steps
- **AND** every bout's kind, food, window mean, speed and duration SHALL match exactly, animal by
  animal
- **AND** a batched evaluator whose window is kept wrongly SHALL fail that comparison

#### Scenario: The option is refused where it cannot act

- **WHEN** a config sets `on_food_reversal_speed` with `locomotion: free`, or `follows_run_speed`
  with `reversal_kinematics: off_food`
- **THEN** loading SHALL raise, naming both settings
- **AND** a primitives config without lawns, where no reversal could start on one, SHALL not load

#### Scenario: The coupling entry is measured and checked

- **WHEN** the kinematics are loaded
- **THEN** `on_food_reversal_speed_coupling` SHALL have status `measured` with an existing script and
  record
- **AND** loading SHALL raise if the entry is missing, its knots are fewer than two, unequal in
  length or not finite, its speeds before are negative or not strictly increasing, a reversal speed
  is not positive, or its window is not positive
