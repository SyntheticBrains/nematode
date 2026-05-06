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

The system SHALL provide predator-specific fitness functions that consume `MultiAgentEpisodeResult.per_predator_kills` and `per_predator_prey_proximity_steps` (from the M1 multi-agent capability) to evaluate predator genomes during co-evolution.

#### Scenario: PredatorEpisodicKillRate Evaluation

- **GIVEN** a `PredatorEpisodicKillRate` fitness instance and a list of `MultiAgentEpisodeResult` instances containing per-predator kill counts for a specific `predator_id`
- **WHEN** `evaluate(results, predator_id)` runs
- **THEN** the returned fitness SHALL be the mean kills per episode for that `predator_id` across the result list
- **AND** when the result list is empty the fitness SHALL be `0.0`

#### Scenario: Secondary Proximity Signal When Kill Count Is Zero

- **GIVEN** a `PredatorEpisodicKillRate` fitness instance with `secondary_signal=True`
- **AND** a result list where the predator has zero kills across all episodes
- **WHEN** `evaluate` runs
- **THEN** the returned fitness SHALL fall back to the mean `per_predator_prey_proximity_steps` divided by `max_steps` (a normalised proximity ratio in [0, 1])
- **AND** the secondary signal SHALL be strictly less than the smallest kill-rate fitness (so a predator with one kill always beats a predator with zero kills, regardless of proximity)

#### Scenario: FitnessFunction Protocol Conformance

- **GIVEN** any predator fitness implementation
- **THEN** it SHALL satisfy the existing `FitnessFunction` Protocol from `quantumnematode.evolution.fitness`
- **AND** it SHALL be `isinstance`-compatible via the Protocol's `@runtime_checkable` decorator

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
