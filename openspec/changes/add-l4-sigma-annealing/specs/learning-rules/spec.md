## ADDED Requirements

### Requirement: The perturbation scale may follow a schedule

The perturbation scale SHALL be allowed to decay over episodes rather than being fixed at
construction. Two configuration values on the plasticity configuration mixin SHALL express the
schedule: a final scale and an anneal length in episodes. With neither set, the scale SHALL be
constant and every arm SHALL be bit-identical to the arm without this requirement.

With both set, the scale in an episode indexed `e` (counting episodes begun, from zero) SHALL be
`σ_final` where `e ≥ E`, and otherwise `σ_0 · (σ_final / σ_0) ** (e / E)`, where `σ_0` is the
configured perturbation scale and `E` the anneal length. The schedule SHALL therefore be monotone,
SHALL reach `σ_final` exactly at `E`, and SHALL be constant thereafter.

A final scale of zero SHALL be rejected at load: the substrates decide at forward time whether
they perturb by testing the scale against zero, and a scale that reached zero would return the
eligibility to pre-synaptic times post-synaptic activity rather than silencing it. The existing
refusal of a zero initial scale SHALL stand.

Setting a final scale without an anneal length, or an anneal length without a final scale, SHALL be
rejected at load, since neither alone defines a schedule; a final scale exceeding the initial scale
SHALL be rejected, since the mechanism is a decay. Every refusal SHALL be repeated at brain
construction, since a copied configuration skips validators.

The schedule's counter SHALL live on the plastic topology and SHALL advance only through an
explicit seam method that a caller invokes where an episode begins — the brain at episode start,
and a harness driving the topology without a brain at each of its trials, a trial being that
harness's schedule step. The counter SHALL NOT advance when the traces are reset, since the traces
are also reset when a policy is loaded, and a load SHALL set the counter to zero. The counter SHALL
NOT be persisted, so that a warm-started arm begins its schedule at `σ_0` and a checkpoint written
before this requirement still loads.

#### Scenario: The default path is unchanged

- **WHEN** neither a final scale nor an anneal length is configured
- **THEN** the perturbation scale SHALL equal the configured scale in every episode, and the weight
  trajectory SHALL be bit-identical to the rule without this requirement

#### Scenario: The scale decays to its floor and stays there

- **GIVEN** an initial scale, a smaller final scale and an anneal length `E`
- **WHEN** episodes are begun in sequence
- **THEN** the scale SHALL fall monotonically, SHALL equal the final scale at episode `E`, and SHALL
  remain equal to it in every later episode

#### Scenario: A schedule that is only half specified is refused

- **GIVEN** a configuration setting a final scale but no anneal length, or an anneal length but no
  final scale, or a final scale above the initial scale, or a final scale of zero
- **WHEN** the configuration is loaded, or a brain is built from a copy of it
- **THEN** it SHALL fail with a message naming the missing, inverted or zero bound

#### Scenario: Resetting the traces does not advance the schedule

- **GIVEN** an annealed arm partway through its schedule
- **WHEN** the traces are reset without an episode beginning
- **THEN** the scheduled scale SHALL be unchanged

#### Scenario: A harness without a brain advances the schedule itself

- **GIVEN** a plastic topology driven directly, with no brain, by a harness that invokes the seam
  method once per trial
- **WHEN** `E` trials have begun
- **THEN** the scheduled scale SHALL equal the final scale

#### Scenario: A warm start begins the schedule again

- **GIVEN** an annealed arm loading a policy saved by another run
- **WHEN** the first episode after the load is begun
- **THEN** the scale SHALL equal the initial scale, and the loaded weights SHALL be unchanged by the
  load

#### Scenario: The schedule reaches the eligibility and the forward pass alike

- **GIVEN** an annealed arm partway through its schedule
- **WHEN** a trace-accumulating forward pass and a step are applied
- **THEN** the perturbation injected into the pre-activations and the perturbation the eligibility
  carries SHALL both be drawn at the episode's scheduled scale, and SHALL be the same draw

### Requirement: An annealed arm SHALL declare its rate regime

Because the eligibility carries the perturbation itself, the update's magnitude scales with the
perturbation scale, so a decaying scale also decays the effective step unless the trace scaling is
enabled to divide it out. An arm using the schedule SHALL therefore record, with its result, whether
trace normalisation was enabled, so that a result obtained while exploration and rate decayed
together is not compared with one obtained at a fixed step.

#### Scenario: The regime is recorded

- **GIVEN** a run using the schedule
- **WHEN** its result is written
- **THEN** the record SHALL state the schedule's bounds and length and whether trace normalisation
  was enabled
