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
SHALL carry its value, its source with figure or table, its condition (on or off food), and whether it
was verified in the text or read from a figure. A test SHALL pin each value to its entry.

#### Scenario: Executed bouts match their source

- **WHEN** each primitive is forced 2,000 times and every bout is measured from the trajectory alone
- **THEN** its whole-step durations SHALL match source draws rounded to whole steps
- **AND** its reversal distances and turn angles SHALL match the vendored distributions directly (each
  a two-sample KS, p > 0.01)

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

`locomotion: primitives` SHALL require the point body, `allow_reversal: false`, and an environment step
of at most one second.

#### Scenario: The 5-second worm is refused

- **WHEN** a config sets `locomotion: primitives` without `step_seconds`, at its 5-second reference step
- **THEN** loading SHALL fail, naming the step

### Requirement: Non-learning hazard controllers

The package SHALL provide `ConstantHazards` (fixed speed, zero curvature, fixed hazards) and
`PirouetteHazards` (start hazards `λ_0 · exp(−β · d)`, with `d` the food derivative feature as brains
see it, and straight runs). Each SHALL implement the Brain protocol, learn nothing, and run through the
same runner as a brain.

#### Scenario: Klinokinesis is expressible and timing alone is not

- **WHEN** worms start 6 mm off a central lawn at uniform headings on seeds 5994–5999 and run for 1,200
  one-second steps
- **THEN** `PirouetteHazards` SHALL reach the lawn more often than `ConstantHazards` at its mean hazard
  and speed, bootstrap-significant at 95%
- **AND** the pirouette rule with its sign inverted SHALL reach it less often than `ConstantHazards` at
  its own mean hazard
