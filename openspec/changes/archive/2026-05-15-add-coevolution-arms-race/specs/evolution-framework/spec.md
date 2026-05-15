## ADDED Requirements

### Requirement: Predator Brain Factory

The system SHALL provide a predator-side equivalent of `instantiate_brain_from_sim_config` (the existing agent-side helper at [`evolution/brain_factory.py:38`](packages/quantum-nematode/quantumnematode/evolution/brain_factory.py#L38)) so that genome encoders + fitness functions on the predator side can construct brains without depending on the agent-side `setup_brain_model` dispatch (which only knows the 19 registered agent brains).

#### Scenario: Predator Brain Factory Surface

- **GIVEN** the `quantumnematode.evolution._predator_brain_factory` module
- **THEN** it SHALL expose `instantiate_predator_brain_from_sim_config(sim_config: SimulationConfig, *, seed: int | None = None) -> MLPPPOPredatorBrain`
- **AND** it SHALL read predator config from `sim_config.environment.predators.brain_config` (NOT `sim_config.brain` which is agent-side; the `predators` field on `EnvironmentConfig` is plural per [`config_loader.py:872`](packages/quantum-nematode/quantumnematode/utils/config_loader.py#L872))
- **AND** it SHALL seed the brain's RNG sources at construction by calling `set_global_seed(seed)` BEFORE constructing the brain (covers numpy + torch globals; the brain's `__init__` separately calls `torch.manual_seed(seed)` for orthogonal-init reproducibility — belt-and-braces). This matches the agent-side factory's seeding semantics; distinct from the env-side dispatcher (`_build_predator_brain`) which uses `torch.manual_seed` only because it has no `BrainConfig`-shaped seed plumbing
- **AND** when no predator config is present, it SHALL raise `ValueError` with a clear diagnostic
- **AND** when `kind` is anything other than `"mlpppo_predator"` (e.g. `"heuristic"`, which has no encoder counterpart), it SHALL raise `ValueError` with a clear diagnostic

### Requirement: Predator Genome Encoder Registry

The system SHALL provide a separate `PREDATOR_ENCODER_REGISTRY` that maps predator-brain kind strings to `GenomeEncoder` implementations, parallel to the existing `ENCODER_REGISTRY` for agent brains. This separation keeps predator-side dispatch isolated from the agent-side registry so neither side accidentally selects the other's encoder. **`MLPPPOPredatorEncoder` overrides `initial_genome`, `decode`, and `genome_dim` from the `_ClassicalPPOEncoder` parent** to call `instantiate_predator_brain_from_sim_config` rather than the agent-side `instantiate_brain_from_sim_config`; plain subclassing wouldn't suffice (the parent's three methods all dispatch through agent-side `setup_brain_model`).

#### Scenario: Registry Surface

- **GIVEN** the `quantumnematode.evolution.predator_encoders` module
- **THEN** it SHALL expose `PREDATOR_ENCODER_REGISTRY: dict[str, type[GenomeEncoder]]` as a top-level mapping
- **AND** it SHALL include the entry `"mlpppo_predator" -> MLPPPOPredatorEncoder`
- **AND** the registry SHALL be lookup-only (no fallback to `ENCODER_REGISTRY`)

#### Scenario: Registry Lookup Helper

- **GIVEN** the `quantumnematode.evolution.predator_encoders` module
- **THEN** it SHALL expose `get_predator_encoder(brain_name: str) -> GenomeEncoder` parallel to the agent-side `get_encoder` helper
- **AND** when `brain_name` is in `PREDATOR_ENCODER_REGISTRY`, it SHALL return a fresh encoder instance
- **AND** when `brain_name` is unknown (including agent-side names like `"mlpppo"`), it SHALL raise `ValueError` listing the registered predator-side keys; the helper SHALL NOT consult `ENCODER_REGISTRY` as a fallback

#### Scenario: MLPPPOPredatorEncoder Brain Factory Override

- **GIVEN** a `MLPPPOPredatorEncoder` instance
- **THEN** its `initial_genome`, `decode`, and `genome_dim` methods SHALL call `instantiate_predator_brain_from_sim_config` (NOT the agent-side `instantiate_brain_from_sim_config`)
- **AND** the brain returned by `decode` at runtime SHALL be an `MLPPPOPredatorBrain` instance, NOT an agent MLPPPO brain
- **AND** `decode`'s static return type SHALL be annotated `Any` (rather than `Brain` from the parent `_ClassicalPPOEncoder` signature) because `MLPPPOPredatorBrain` satisfies the *separate* `PredatorBrain` Protocol — not the agent-side `Brain` Protocol — and forcing the annotation back to `Brain` would lie about the runtime type. Callers that need to assign the return value to a `PredatorBrain`-typed slot (e.g. `Predator.brain`) SHOULD `cast("PredatorBrain", encoder.decode(...))` at the assignment boundary; the predator fitness module does this internally
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

`PredatorEpisodicKillRate` is the production fitness used by `CoevolutionLoop` per design.md D13. It accepts optional `warm_start_path_override` and `weight_capture_path` kwargs on `evaluate(...)` mirroring `LearnedPerformanceFitness.evaluate`'s surface — when populated by the loop (under `predator_evolution.inheritance: "lamarckian"`), the predator brain is built with `enable_learning=True` so the multi-agent runner's per-step `predator.brain.learn(reward, episode_done)` hook fires during evaluation and trained weights are captured for `LamarckianInheritance` to carry across generations. When both kwargs are `None` (the design.md D13 default under `inheritance: "none"`), evaluation is frozen-weight only and matches the original D13 contract. `PredatorLearnedPerformanceFitness` is shipped as a stub raising `NotImplementedError` — a notional separate fitness class with an explicit K_train + L_eval split (like `LearnedPerformanceFitness` on the prey side) remains out of scope; the in-place inner-loop training path on `PredatorEpisodicKillRate` covers the substrate ablation more cleanly. The stub class still satisfies the `FitnessFunction` Protocol via `runtime_checkable` so type-driven dispatch keeps working; calling `evaluate` on it raises clearly.

`PredatorEpisodicKillRate.__init__(*, secondary_signal: bool = True)` exposes one configuration knob: when `True` (default), the proximity fallback below applies on all-zero-kills evaluations; when `False`, all-zero-kills evaluations score exactly `0.0` (matching the literal kill-rate definition; useful for ablations that disable the proximity assist).

#### Scenario: FitnessFunction Protocol Conformance

- **GIVEN** any predator fitness implementation (`PredatorEpisodicKillRate`, `PredatorLearnedPerformanceFitness`)
- **THEN** it SHALL satisfy the existing `FitnessFunction` Protocol from `quantumnematode.evolution.fitness`
- **AND** it SHALL be `isinstance(impl, FitnessFunction)` via the Protocol's `@runtime_checkable` decorator
- **AND** its `evaluate` method SHALL accept the canonical signature `evaluate(genome, sim_config, encoder, *, episodes, seed) -> float` (no diverging signature)
- **AND** it SHALL be picklable so it can flow through the existing `EvolutionLoop._evaluate_in_worker` 11-tuple worker

#### Scenario: PredatorLearnedPerformanceFitness Is A Deferred Stub

- **GIVEN** a `PredatorLearnedPerformanceFitness` instance
- **THEN** it SHALL satisfy the `FitnessFunction` Protocol via `runtime_checkable` (so type-driven dispatch and Protocol-conformance checks pass)
- **AND** `evaluate(...)` SHALL raise `NotImplementedError` with a clear diagnostic pointing at `PredatorEpisodicKillRate` as the production alternative and design.md D13 as the rationale
- **AND** the stub SHALL NOT be wired into `CoevolutionLoop.__init__` — `CoevolutionLoop` hardcodes `PredatorEpisodicKillRate` per D13 + B20; the predator-side inner-loop PPO ablation rides on `PredatorEpisodicKillRate`'s opt-in kwargs (see "Predator Lamarckian Opt-In") rather than a separate fitness class

#### Scenario: PredatorEpisodicKillRate Lamarckian Opt-In

- **GIVEN** a `PredatorEpisodicKillRate` evaluating a predator genome with `predator_evolution.inheritance: "lamarckian"` configured at the loop level
- **WHEN** `CoevolutionLoop` invokes `evaluate(genome, sim_config, encoder, *, episodes, seed, warm_start_path_override=<parent_pt>, weight_capture_path=<child_pt>)`
- **THEN** the predator brain SHALL be decoded with `enable_learning=True` so the optimizer + rollout buffer are constructed
- **AND** when `warm_start_path_override` is a non-None `Path`, the post-decode brain SHALL `load_weights` from that path BEFORE the N evaluation episodes (Lamarckian warm-start from the parent's K-block-end weights)
- **AND** the N multi-agent episodes SHALL exercise the per-step `predator.brain.learn(reward, episode_done)` hook (PPO inner-loop training fires during evaluation)
- **AND** when `weight_capture_path` is a non-None `Path`, the brain's trained weights SHALL be `save_weights`'d to that path AFTER the N episodes complete (so `LamarckianInheritance` can broadcast them to children in the next generation)
- **AND** with both kwargs `None` (the design.md D13 default under `inheritance: "none"`), behaviour SHALL match the original frozen-weight contract (no warm-start, no `.learn()`-eligible brain, no weight capture)

#### Scenario: PredatorEpisodicKillRate Mean Kills Calculation

- **GIVEN** a `PredatorEpisodicKillRate` fitness evaluating a predator genome over `episodes` multi-agent runs against frozen prey opponents (configured in the patched `sim_config`)
- **WHEN** `evaluate(genome, sim_config, encoder, episodes=N, seed=S)` runs
- **THEN** the predator brain SHALL be decoded from the genome via `encoder.decode(genome, ..., seed=S)`
- **AND** N multi-agent episodes SHALL run with **all predator slots in the env using the same decoded brain** (the genome under evaluation is the strategy being measured, not just slot 0)
- **AND** the returned fitness SHALL be the per-episode mean of `sum(MultiAgentEpisodeResult.per_predator_kills.values())` across the N episodes (summing across all predator slots within each episode, then averaging across episodes)
- **AND** when N is 0 the fitness SHALL be `0.0`
- **Rationale:** all predator slots run the same decoded brain (the genome's *strategy*); fitness measures the *strategy*'s effectiveness, not the slot. Summing kills across slots within each episode is correct because each prey kill is one event regardless of which slot delivered the damage.

#### Scenario: Secondary Proximity Signal When Kill Count Is Zero

- **GIVEN** a `PredatorEpisodicKillRate` fitness with `secondary_signal=True` (the constructor default)
- **AND** N episodes complete where `sum(per_predator_kills.values())` is zero across ALL episodes (no slot in any episode killed any prey)
- **WHEN** `evaluate` aggregates the results
- **THEN** the returned fitness SHALL fall back to the per-episode mean of `sum(per_predator_prey_proximity_steps.values()) / (max_steps * num_predator_slots)` (a normalised cross-slot proximity ratio in [0, 1])
- **AND** the secondary fallback SHALL be strictly less than the smallest non-zero kill-rate fitness (so a predator strategy with even one kill in any episode always beats a strategy with zero kills, regardless of proximity); concretely the fallback range SHALL be `[0, 1/N)` while the lowest non-zero kill-rate fitness is `>= 1/N`
- **Implementation note (non-normative):** the strict-less-than bound is enforced by multiplying the proximity ratio by a headroom factor `< 1.0` before dividing by `N`. The shipped value is `_PROXIMITY_FALLBACK_HEADROOM = 0.99` in `predator_fitness.py`, producing fallback values in `[0, 0.99/N)` — comfortably below the `1/N` floor without distorting the proximity ranking among zero-kill strategies. The exact constant is a free parameter; only the strict-less-than invariant is normative.

#### Scenario: Secondary Signal Disabled Returns Zero On All-Zero Kills

- **GIVEN** a `PredatorEpisodicKillRate` fitness constructed with `secondary_signal=False`
- **AND** N episodes complete where `sum(per_predator_kills.values())` is zero across ALL episodes
- **WHEN** `evaluate` aggregates the results
- **THEN** the returned fitness SHALL be exactly `0.0` (matching the literal kill-rate definition; the proximity fallback is not applied)
- **AND** this knob exists for ablations that disable the proximity assist; it is NOT used in the M5 production run

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
