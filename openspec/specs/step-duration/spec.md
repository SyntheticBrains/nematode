# step-duration Specification

## Purpose

One configured value sets how many worm-seconds a simulation step represents. A config's per-step
constants stay written for its reference step, and the loader converts each to the configured step by
the rule for its kind, so every physical rate keeps its value per worm-second. The conversion covers
the continuous point worm on lawns with the connectome or MLP-PPO brain, and a non-reference step is
refused everywhere else.

## Requirements

### Requirement: One configured step duration

A simulation config SHALL accept a top-level `step_seconds`, a positive float, unset by default. The
**reference step** SHALL be the duration the config's per-step values are written for: its
`max_step_mm` divided by the crawl speed, as written. When `step_seconds` is set, the reference step
divided by it SHALL be a whole number from 1 to 100. The config SHALL be refused at load otherwise, and
refused when `max_step_mm` is zero.

#### Scenario: Unset is byte-identical

- **WHEN** a config leaves `step_seconds` unset
- **THEN** no value SHALL be derived, and its runs SHALL be byte-identical to before the key existed

#### Scenario: The reference step is the config's own

- **WHEN** a config with `max_step_mm: 1.0` sets `step_seconds: 1.0`
- **THEN** its reference step SHALL be 5 seconds and its ratio 5

#### Scenario: A ratio that is not whole is refused

- **WHEN** a config with a 5-second reference step sets `step_seconds: 2.0`, or `step_seconds: 10.0`
- **THEN** loading SHALL fail with a message naming the ratio and the reference step

### Requirement: Step-coupled constants derive from the step duration

When `step_seconds` differs from the reference step, the loader SHALL derive every step-coupled
constant from the reference step to the configured one, with `k` the configured step over the reference
step and `r = 1/k`:

- the full-speed step becomes the configured step times the crawl speed;
- amounts per step scale by `k`: `max_turn_rad`, `satiety_decay_rate`,
  `regrowth_per_step`, `penalty_step`, `penalty_boundary_collision`, and the connectome's
  `modulator_step_seconds`;
- fractions removed per step become `1 − (1 − p)^k`: `intake_fraction`, `adaptive_chemosensor_alpha`;
- per-step discounts become `d^k`: `gamma`, `gae_lambda`;
- the STAM decay rate scales by `k`;
- lengths in steps scale by `r`, exactly: `max_steps`, `stam_buffer_size`, `stuck_position_threshold`;
- the connectome's `membrane_tau_steps` scales by `r`, under leaky dynamics only;
- `derivative_scale` scales by `r`.

Event rewards, per-unit-eaten and per-millimetre terms, spatial settings, `forward_pass_depth` and the
training-budget settings SHALL NOT be derived. An unset `satiety` or `reward` block SHALL be
derived from its defaults. The loader SHALL log each derived constant with its written and derived
values, and `run_simulation` SHALL print the step, the reference step, the ratio and each derived
constant to stdout. The derivation SHALL be idempotent.

#### Scenario: A 1-s lawn worm's derived values

- **WHEN** D.1's lawn config is loaded with `step_seconds: 1.0`
- **THEN** `max_step_mm` SHALL be 0.2, `max_turn_rad` π/5, `max_steps` 3600, `stam_buffer_size` 150,
  `gamma` 0.99^0.2, and `intake_fraction` 1 − 0.98^0.2

#### Scenario: A campaign run records its step

- **WHEN** a 1-second config runs under `--no-file-log`
- **THEN** its stdout SHALL carry a `Step:` line naming 1.0 s, the 5.0-s reference and the ratio 5,
  and each derived constant

#### Scenario: Settling dynamics keep their time constant

- **WHEN** a connectome config under the default settling dynamics sets `step_seconds: 1.0`
- **THEN** `membrane_tau_steps` SHALL stay 1.0, and the config SHALL load

#### Scenario: Deriving twice changes nothing

- **WHEN** the derivation is applied to an already derived config
- **THEN** the result SHALL equal its input

#### Scenario: Physical quantities are preserved

- **WHEN** a scripted worm runs for 30 worm-seconds at full speed on a straight heading, well clear of
  every lawn, at a 5-second and a 1-second step
- **THEN** its position, path length and satiety SHALL match to floating-point tolerance
- **AND** a cell grazed for 60 worm-seconds, the STAM weights by age in seconds, and the adaptive
  baseline's step response SHALL match at both steps

#### Scenario: The positive control can fail

- **WHEN** the same scripted worm runs at a 1-second step with its per-step values unconverted
- **THEN** its path length SHALL differ from the 5-second worm's, so the preservation test can fail

#### Scenario: Every field is classified

- **WHEN** a field is added to a config model the lawn point worm reads
- **THEN** a test SHALL fail until the field is assigned a derivation rule, or is recorded as not
  step-coupled with a reason

### Requirement: A non-default step is refused outside its derived surface

A config whose `step_seconds` differs from its reference step SHALL be refused at load, with a message
naming the key and the reason, if any of these holds:

- the substrate is not `continuous_2d`;
- the food model is not `lawns`;
- thermotaxis, aerotaxis, predators, pheromones or social feeding are enabled;
- it is a multi-agent run or a memory task;
- the body model is `kinematic`;
- the brain is not `connectomeppo` or `mlpppo`;
- within-lifetime plasticity is on;
- `hyperparam_schema` is set;
- `max_steps` is unset;
- `penalty_stuck_position`, `penalty_anti_dithering`, `reward_exploration` or `reward_distance_scale` is
  non-zero.

#### Scenario: A thermal config at one second is refused

- **WHEN** a thermotaxis config sets `step_seconds: 1.0`
- **THEN** loading SHALL fail, naming thermotaxis and the scope that excludes it

#### Scenario: A frozen scenario at its default still loads

- **WHEN** any committed scenario config is loaded unchanged
- **THEN** it SHALL load as before

### Requirement: A configured 1-second lawn worm

The repository SHALL carry generated 1-second variants of D.1's lawn cell, for the connectome with
modulators and the MLP reading internal state. Each SHALL be its base config plus `step_seconds: 1.0`.

#### Scenario: The 1-second configs run

- **WHEN** each 1-second config runs three episodes on smoke seeds with behaviour capture on
- **THEN** it SHALL complete, and its capture SHALL classify into roaming, dwelling and off-food windows
- **AND** a smoke-tier test SHALL run each for a few steps
