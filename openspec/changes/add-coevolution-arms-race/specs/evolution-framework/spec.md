## ADDED Requirements

### Requirement: Predator Brain Factory

The system SHALL provide a predator-side equivalent of `instantiate_brain_from_sim_config` (the existing agent-side helper at [`evolution/brain_factory.py:38`](packages/quantum-nematode/quantumnematode/evolution/brain_factory.py#L38)) so that genome encoders + fitness functions on the predator side can construct brains without depending on the agent-side `setup_brain_model` dispatch (which only knows the 19 registered agent brains).

#### Scenario: Predator Brain Factory Surface

- **GIVEN** the `quantumnematode.evolution._predator_brain_factory` module
- **THEN** it SHALL expose `instantiate_predator_brain_from_sim_config(sim_config: SimulationConfig, *, seed: int | None = None) -> MLPPPOPredatorBrain`
- **AND** it SHALL read predator config from `sim_config.environment.predator.brain_config` (NOT `sim_config.brain` which is agent-side)
- **AND** it SHALL seed the brain's RNG sources at construction (matching `set_global_seed(seed)` semantics from the agent factory)
- **AND** when no predator config is present, it SHALL raise `ValueError` with a clear diagnostic

### Requirement: Predator Genome Encoder Registry

The system SHALL provide a separate `PREDATOR_ENCODER_REGISTRY` that maps predator-brain kind strings to `GenomeEncoder` implementations, parallel to the existing `ENCODER_REGISTRY` for agent brains. This separation keeps predator-side dispatch isolated from the agent-side registry so neither side accidentally selects the other's encoder. **`MLPPPOPredatorEncoder` overrides `initial_genome`, `decode`, and `genome_dim` from the `_ClassicalPPOEncoder` parent** to call `instantiate_predator_brain_from_sim_config` rather than the agent-side `instantiate_brain_from_sim_config`; plain subclassing wouldn't suffice (the parent's three methods all dispatch through agent-side `setup_brain_model`).

#### Scenario: Registry Surface

- **GIVEN** the `quantumnematode.evolution.predator_encoders` module
- **THEN** it SHALL expose `PREDATOR_ENCODER_REGISTRY: dict[str, type[GenomeEncoder]]` as a top-level mapping
- **AND** it SHALL include the entry `"mlpppo_predator" -> MLPPPOPredatorEncoder`
- **AND** the registry SHALL be lookup-only (no fallback to `ENCODER_REGISTRY`)

#### Scenario: MLPPPOPredatorEncoder Brain Factory Override

- **GIVEN** a `MLPPPOPredatorEncoder` instance
- **THEN** its `initial_genome`, `decode`, and `genome_dim` methods SHALL call `instantiate_predator_brain_from_sim_config` (NOT the agent-side `instantiate_brain_from_sim_config`)
- **AND** the brain returned by `decode` SHALL be an `MLPPPOPredatorBrain` instance, NOT an agent MLPPPO brain
- **AND** the encoder SHALL pin `brain_name = "mlpppo_predator"` (used for genome birth_metadata + registry lookup)

#### Scenario: MLPPPOPredatorEncoder Round-Trip

- **GIVEN** a `MLPPPOPredatorEncoder` instance and a `MLPPPOPredatorBrain` with arbitrary weights
- **WHEN** the encoder produces a genome via the encoder's flatten round-trip and decodes it back into a fresh brain
- **THEN** the decoded brain SHALL produce the same action as the original on a fixed test set of `PredatorBrainParams`
- **AND** the genome dimension SHALL match the brain's total `WeightPersistence` parameter count
- **AND** the round-trip SHALL succeed under `CMAESOptimizer(diagonal=True)` (verifies that unbounded weight encoding + sep-CMA-ES sampling produces a valid genome)

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
- **AND** N multi-agent episodes SHALL run with **all predator slots in the env using the same decoded brain** (the genome under evaluation is the strategy being measured, not just slot 0)
- **AND** the returned fitness SHALL be the per-episode mean of `sum(MultiAgentEpisodeResult.per_predator_kills.values())` across the N episodes (summing across all predator slots within each episode, then averaging across episodes)
- **AND** when N is 0 the fitness SHALL be `0.0`
- **Rationale:** all predator slots run the same decoded brain (the genome's *strategy*); fitness measures the *strategy*'s effectiveness, not the slot. Summing kills across slots within each episode is correct because each prey kill is one event regardless of which slot delivered the damage.

#### Scenario: Secondary Proximity Signal When Kill Count Is Zero

- **GIVEN** a `PredatorEpisodicKillRate` fitness with `secondary_signal=True`
- **AND** N episodes complete where `sum(per_predator_kills.values())` is zero across ALL episodes (no slot in any episode killed any prey)
- **WHEN** `evaluate` aggregates the results
- **THEN** the returned fitness SHALL fall back to the per-episode mean of `sum(per_predator_prey_proximity_steps.values()) / (max_steps * num_predator_slots)` (a normalised cross-slot proximity ratio in [0, 1])
- **AND** the secondary fallback SHALL be strictly less than the smallest non-zero kill-rate fitness (so a predator strategy with even one kill in any episode always beats a strategy with zero kills, regardless of proximity); concretely the fallback range SHALL be `[0, 1/N)` while the lowest non-zero kill-rate fitness is `>= 1/N`

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
