## ADDED Requirements

### Requirement: An assay is a list of phases

An environment config SHALL accept an optional `assay` block: a name, a list of phases, the name of the
assay phase, and an analysis start in seconds within it. Each phase SHALL give a name, a duration in
worm-seconds, its food (`lawns` or `none`), and a placement (`keep`, `lawn_centre`, `off_lawn` or
`arena_centre`). At each phase's start the environment SHALL turn its lawns on, freshly placed, or off,
and place the worm. Every placement but `keep` SHALL draw a uniform heading. The worm's satiety, its
controller's state and its primitive state SHALL carry across phases unchanged.

#### Scenario: Food is removed and the worm transferred

- **WHEN** a `fed` phase on lawns ends and an `assay` phase with `food: none` and `place: arena_centre`
  begins
- **THEN** the lawn field SHALL be gone, with zero odour and intake everywhere
- **AND** the worm SHALL be at the arena centre with a newly drawn heading
- **AND** its satiety and its controller's state SHALL be those it had at the end of the `fed` phase

#### Scenario: An off-lawn start

- **WHEN** a phase places the worm `off_lawn`
- **THEN** it SHALL start on bare agar, at least `start_clearance_mm` from every lawn's edge, inside the
  arena

#### Scenario: The phases fix the episode's length

- **WHEN** a config's `max_steps` times its step differs from the sum of its phases' durations
- **THEN** loading SHALL fail, naming both

#### Scenario: No assay is unchanged

- **WHEN** a config has no `assay` block
- **THEN** its runs SHALL be byte-identical to before the option

### Requirement: The assay's scope

An `assay` SHALL require `food_model: lawns`, the continuous substrate, one worm, `start_on_lawn: false`, and `satiety.starvation_ends_episode: false`.

#### Scenario: Starvation would end an assay

- **WHEN** an assay config leaves `starvation_ends_episode` at true
- **THEN** loading SHALL fail, naming the setting

### Requirement: A circular arena

`continuous.arena_shape: circle` SHALL bound the worm by a circle of diameter `world_size_mm` centred
in the square. A move ending outside it SHALL be projected radially onto the wall, and sensing positions
SHALL be clamped the same way. `square`, the default, SHALL be unchanged.

#### Scenario: The worm never leaves the well

- **WHEN** a worm in a circular arena makes 10,000 random moves
- **THEN** no position SHALL lie outside the circle

### Requirement: A centred lawn

`lawns.placement: centre` SHALL put a single lawn at the arena centre, and SHALL be refused with
`count` above 1. `random`, the default, SHALL be unchanged, and SHALL be refused on a circular arena.

#### Scenario: Scheer's lawn

- **WHEN** a 10 mm circular arena has one centred lawn of radius 1.5 mm
- **THEN** the lawn's centre SHALL be the arena's centre

### Requirement: Satiety as a slow reserve

`satiety.starvation_ends_episode: false` SHALL keep satiety clamped at zero and run the episode to its
end. The generated assay configs SHALL set satiety to empty over 12 h of basal decay off food, and
SHALL set the movement cost to zero.

#### Scenario: Twelve hours off food

- **WHEN** a worm in an assay config spends 43,200 worm-seconds on a bare plate
- **THEN** its satiety SHALL reach zero at 43,200 s, within one step, and the episode SHALL continue

#### Scenario: A dwelling worm stays fed

- **WHEN** a worm dwells on a fresh lawn for 90 minutes
- **THEN** its satiety SHALL stay above 90% of its maximum

### Requirement: Generated assay configs

The repository SHALL carry generated assay configs on the 1-second primitives worm, for the MLP and the
connectome:

- `scheer_lawn`: a 10 mm well, a 3 mm centred lawn, 60 min from an off-lawn start, analysed from 20 min;
- `food_removal_lopezcruz`: a 100 mm circle, 45 min on a 25 mm-radius lawn, then 45 min bare from the
  centre;
- `food_removal_calhoun`: a 25.4 mm square, 45 min fed, then 30 min bare, analysed from 2 min;
- `uniform_lawn_flavell`: a 35 mm circle, a 16 mm-radius lawn, 90 min.

#### Scenario: The assay configs run

- **WHEN** each assay config runs one episode on a smoke seed with capture on
- **THEN** it SHALL complete with its phases in order
- **AND** its capture SHALL classify where the protocol has food
