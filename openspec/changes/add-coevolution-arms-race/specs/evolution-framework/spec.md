## ADDED Requirements

### Requirement: Predator Genome Encoder Registry

The system SHALL provide a separate `PREDATOR_ENCODER_REGISTRY` that maps predator-brain kind strings to `GenomeEncoder` implementations, parallel to the existing `ENCODER_REGISTRY` for agent brains. This separation keeps predator-side dispatch isolated from the agent-side registry so neither side accidentally selects the other's encoder.

#### Scenario: Registry Surface

- **GIVEN** the `quantumnematode.evolution.predator_encoders` module
- **THEN** it SHALL expose `PREDATOR_ENCODER_REGISTRY: dict[str, type[GenomeEncoder]]` as a top-level mapping
- **AND** it SHALL include the entry `"mlpppo_predator" -> MLPPPOPredatorEncoder`
- **AND** the registry SHALL be lookup-only (no fallback to `ENCODER_REGISTRY`)

#### Scenario: MLPPPOPredatorEncoder Round-Trip

- **GIVEN** a `MLPPPOPredatorEncoder` instance and a `MLPPPOPredatorBrain` with arbitrary weights
- **WHEN** the encoder produces a genome via the encoder's flatten round-trip and decodes it back into a fresh brain
- **THEN** the decoded brain SHALL produce the same action as the original on a fixed test set of `PredatorBrainParams`
- **AND** the genome dimension SHALL match the brain's total `WeightPersistence` parameter count

#### Scenario: Initial Genome Reproducibility

- **GIVEN** a `MLPPPOPredatorEncoder` instance and a fixed seed
- **WHEN** `initial_genome(seed)` is called twice with the same seed
- **THEN** both calls SHALL return identical genome arrays

### Requirement: Predator Fitness Functions

The system SHALL provide predator-specific fitness functions that conform to the existing `FitnessFunction` Protocol surface (`evaluate(genome, sim_config, encoder, *, episodes, seed) -> float`) from `quantumnematode.evolution.fitness`. Internally, predator fitness drives the multi-agent runner against frozen prey opponents (configured via `sim_config` patching at the call site) and aggregates per-predator metrics from the resulting `MultiAgentEpisodeResult` instances (`per_predator_kills` for the primary signal, `per_predator_prey_proximity_steps` for the secondary fallback).

#### Scenario: FitnessFunction Protocol Conformance

- **GIVEN** any predator fitness implementation (`PredatorEpisodicKillRate`, `PredatorLearnedPerformanceFitness`)
- **THEN** it SHALL satisfy the existing `FitnessFunction` Protocol from `quantumnematode.evolution.fitness`
- **AND** it SHALL be `isinstance(impl, FitnessFunction)` via the Protocol's `@runtime_checkable` decorator
- **AND** its `evaluate` method SHALL accept the canonical signature `evaluate(genome, sim_config, encoder, *, episodes, seed) -> float` (no diverging signature)
- **AND** it SHALL be picklable so it can flow through the existing `EvolutionLoop._evaluate_in_worker` 11-tuple worker

#### Scenario: PredatorEpisodicKillRate Mean Kills Calculation

- **GIVEN** a `PredatorEpisodicKillRate` fitness evaluating a predator genome over `episodes` multi-agent runs against frozen prey opponents (configured in the patched `sim_config`)
- **WHEN** `evaluate(genome, sim_config, encoder, episodes=N, seed=S)` runs
- **THEN** the predator brain SHALL be decoded from the genome via `encoder.decode(genome, ..., seed=S)`
- **AND** N multi-agent episodes SHALL run with that brain occupying the predator slots
- **AND** the returned fitness SHALL be the mean of `MultiAgentEpisodeResult.per_predator_kills[predator_id]` across the N episodes for the predator slot under evaluation
- **AND** when N is 0 the fitness SHALL be `0.0`

#### Scenario: Secondary Proximity Signal When Kill Count Is Zero

- **GIVEN** a `PredatorEpisodicKillRate` fitness with `secondary_signal=True`
- **AND** N episodes complete where the predator under evaluation has zero kills across ALL episodes
- **WHEN** `evaluate` aggregates the results
- **THEN** the returned fitness SHALL fall back to the mean `per_predator_prey_proximity_steps` divided by `max_steps` for that predator slot, scaled to a sub-unity range
- **AND** the secondary fallback SHALL be strictly less than the smallest non-zero kill-rate fitness (so a predator with one kill always beats a predator with zero kills, regardless of proximity); concretely the fallback range SHALL be `[0, 1/N)` while any non-zero kill-rate is `>= 1/N`

### Requirement: Hall-of-Fame Buffer

The system SHALL provide a `HallOfFame` primitive — a bounded buffer of past champion genomes with a configurable replacement policy — usable both as a co-evolution opposition pool and as a single-population novelty-search primitive.

#### Scenario: Quality-Based Eviction (Default)

- **GIVEN** a `HallOfFame(capacity=N, replacement="quality")` filled to capacity with N genomes at known fitnesses
- **WHEN** `push(genome, fitness)` is called with a new fitness value strictly greater than the lowest existing HoF entry
- **THEN** the lowest-fitness existing entry SHALL be evicted
- **AND** the new entry SHALL be inserted
- **AND** when the new fitness is less than or equal to all existing entries, the buffer SHALL be unchanged

#### Scenario: FIFO Eviction Ablation

- **GIVEN** a `HallOfFame(capacity=N, replacement="fifo")` filled to capacity
- **WHEN** `push(genome, fitness)` is called
- **THEN** the oldest-pushed entry SHALL be evicted regardless of fitness
- **AND** the new entry SHALL be inserted

#### Scenario: Mix-With-Pop Sampling

- **GIVEN** a `HallOfFame` with at least one entry and a population list `pop` of length P
- **WHEN** `mix_with_pop(rng, pop, frac_hof=0.3)` is called with sample-size P
- **THEN** approximately `0.3 * P` entries SHALL come from the HoF (sampled with replacement)
- **AND** the remaining `~0.7 * P` entries SHALL come from `pop` (sampled with replacement)
- **AND** when the HoF is empty, all P entries SHALL come from `pop`

#### Scenario: Reproducible Sampling Under Seeded RNG

- **GIVEN** two `HallOfFame` instances with identical contents
- **WHEN** `mix_with_pop` is called on each with separate but identically-seeded RNGs
- **THEN** both calls SHALL return identical sample sequences

#### Scenario: Checkpoint Round-Trip

- **GIVEN** a `HallOfFame` instance with arbitrary contents
- **WHEN** `to_dict()` is called and `from_dict(d)` reconstructs a new instance
- **THEN** the reconstructed instance SHALL have the same capacity, replacement policy, and entry order
- **AND** subsequent `mix_with_pop` calls under the same RNG seed SHALL produce identical results
