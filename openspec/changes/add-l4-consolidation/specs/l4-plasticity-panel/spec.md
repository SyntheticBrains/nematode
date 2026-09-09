## ADDED Requirements

### Requirement: A consolidation variant is screened by the clone assay before any panel

Every candidate consolidation mechanism SHALL be screened by the clone assay before the 2×2 panel
is re-run under it. The assay is the protocol registered with the clone-destruction diagnostic and
is not restated with new values here: the wild-type plastic clone arm with the variant's rule keys
and nothing else changed, started from the warm-start panel's plastic-set wild-type clone for that
seed, seeds 1–8 paired, a 2000-episode budget with no extension, the committed plateau-tail
full-clear success metric, and the same seeds' published frozen-clone values as the comparator.
A variant **holds** when its mean is within 5 points of the frozen clone's mean and at least 6 of
8 seeds are no more than 10 points below their own frozen clone; it **improves** when its mean is
above the frozen clone's and at least 6 of 8 seeds are above their own; it **passes** when it
holds or improves.

The assay SHALL be reported as a **screen and not a confirmatory test**. It reuses seeds already
reported, so it SHALL NOT declare a multiple-comparisons family or a verdict map, and the record
SHALL state that a pass licenses running the registered panel and nothing more.

Each variant's screen SHALL report the eight per-seed values, the mean delta against the frozen
clone, the count of seeds at or above their own frozen clone, and the cosine of the endpoint
weights to the clone the run started from, since a variant can pass on behaviour while having
rewritten the policy.

#### Scenario: A variant is screened before the panel

- **GIVEN** a consolidation variant proposed for the panel
- **WHEN** the panel is scheduled
- **THEN** the variant SHALL have a recorded clone-assay result
- **AND** the panel SHALL NOT be run under a variant whose screen did not pass

#### Scenario: The screen is reported as a screen

- **WHEN** a screen result is written up
- **THEN** the record SHALL state that it reuses previously reported seeds and licenses the panel
  only
- **AND** it SHALL NOT assign a panel verdict

#### Scenario: Holding by consolidation is distinguished from holding by not moving

- **WHEN** a variant passes the screen
- **THEN** the record SHALL report the endpoint cosine to the clone and the effective rate
  multiplier alongside the metric

### Requirement: Consolidation hyperparameters are pinned by a pre-declared pilot

Where a mechanism has hyperparameters with no value to inherit, they SHALL be pinned by a pilot
declared before it runs: seeds 1–2 of the same clone arm at the same budget over a grid written
into the launch record, pinning the combination with the highest mean plateau tail across the two
seeds and breaking ties toward the weaker constraint. The pilot's grid, its criterion and its
results SHALL be recorded before the screen is run, and the pilot seeds SHALL be reported with the
screen so that a pin which only worked on its own seeds is visible.

A mechanism whose values are fixed by the comparator rather than chosen SHALL record that instead
of running a pilot.

#### Scenario: The pins are declared before they are used

- **WHEN** a screen is launched
- **THEN** the launch record SHALL already contain the pilot grid, the pinning criterion and the
  pinned values

#### Scenario: The oracle arm is declared as a bound, not a candidate

- **WHEN** the oracle variant is screened
- **THEN** the record SHALL state that it consumes the environment's episode-success flag, that it
  is not a mechanism the animal could host, and that it exists to bound what a quality-gated
  consolidation could achieve
