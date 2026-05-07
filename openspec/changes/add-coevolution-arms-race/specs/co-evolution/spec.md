## ADDED Requirements

### Requirement: CoevolutionLoop Orchestrator

The system SHALL provide a `CoevolutionLoop` orchestrator that drives two populations (prey and predator) through an alternating-schedule co-evolution run. The loop SHALL compose two side-state objects (each with its own encoder, fitness, optimiser, inheritance strategy, hall-of-fame, and population) and SHALL reuse the existing `EvolutionLoop._evaluate_in_worker` multiprocessing worker pattern per side, with opponent brain weights injected via `sim_config` patching following the M2 idiom.

#### Scenario: Side State Surface

- **GIVEN** a `CoevolutionLoop` instance
- **THEN** each side state SHALL carry an `encoder: GenomeEncoder`, `fitness: FitnessFunction`, `optimizer: CMAESOptimizer` constructed with `diagonal=True` (sep-CMA-ES, imported from `quantumnematode.optimizers.evolutionary`), `inheritance: InheritanceStrategy`, `hof: HallOfFame`, `population: list[Genome]`, and `champion_history: list[Genome]`
- **AND** the prey side state SHALL be configured with `LearnedPerformanceFitness` (Lamarckian-style: K_train=50 inner-loop training + L_eval=25 frozen evaluation episodes per genome) plus `LamarckianInheritance`, mirroring the M3 Lamarckian-LSTMPPO winner stack
- **AND** the predator side state SHALL be configured with `PredatorEpisodicKillRate` (frozen-weight evaluation only: N_eval=25 multi-agent episodes per genome, NO inner-loop PPO training, NO weight capture) plus `NoInheritance`
- **AND** the asymmetric fitness shapes are intentional per design.md D13 — prey trains per evaluation (large policy space, M3 substrate); predator evaluates frozen (small policy space, direct CMA-ES weight gradient suffices)
- **AND** `hof` SHALL be a bounded buffer with eviction policy (the runtime opposition-sampling pool)
- **AND** `champion_history` SHALL be unbounded — exactly one entry per completed K-block (the K-block's top-fitness genome at K-block end), never evicted; the time-ordered audit log walked by aggregator-time analysis (cycling, escalation). Distinct from per-generation lineage rows AND distinct from `hof`. A K-block's elite genome lands in BOTH `hof` (subject to eviction) and `champion_history` (unbounded)

#### Scenario: Composition Over Inheritance

- **GIVEN** the `CoevolutionLoop` class
- **THEN** it SHALL NOT subclass `EvolutionLoop`
- **AND** it SHALL invoke worker evaluation via `EvolutionLoop._evaluate_in_worker` so the existing 11-tuple worker pattern (`(params, sim_config, encoder, fitness, episodes, seed, generation, index, parent_ids, warm_start_path_override, weight_capture_path)`) is preserved verbatim
- **AND** the opposing-side opponent brain weights SHALL be injected at evaluation time via `sim_config` patching at the call site (the worker tuple ABI is unchanged)
- **AND** each side's `fitness.evaluate` SHALL conform to the existing `FitnessFunction` Protocol surface `evaluate(genome, sim_config, encoder, *, episodes, seed) -> float`

### Requirement: Alternating Training Schedule

The system SHALL drive the co-evolution loop under an alternating schedule with a configurable block size K (default K=10 generations per side). Within a K-block, only one side trains; the opposing side is frozen.

#### Scenario: K-Block Boundary

- **GIVEN** a `CoevolutionLoop` configured with `K_per_block=10` and `generation_pairs=3`
- **WHEN** `run` executes
- **THEN** the loop SHALL execute 3 prey K-blocks of 10 generations each interleaved with 3 predator K-blocks of 10 generations each (total 60 per-side generations, 6 K-blocks)
- **AND** the order SHALL be: prey K-block 1 → predator K-block 1 → prey K-block 2 → predator K-block 2 → ... (configurable starting side via a `start_side` parameter; default = prey)

#### Scenario: Opposing Side Frozen During Off-Block

- **GIVEN** an active K-block training side X
- **WHEN** the loop evaluates a candidate genome from side X
- **THEN** the opposing side Y's population SHALL NOT change for the duration of the K-block
- **AND** Y's optimizer SHALL NOT receive new trial results during X's block
- **AND** Y's HoF SHALL NOT receive any new pushes during X's block

#### Scenario: Fresh CMA-ES Optimizer At K-Block Start

- **GIVEN** a side that is about to begin a new K-block (the opposing side just finished its block)
- **WHEN** the loop transitions to this side
- **THEN** this side's `CMAESOptimizer(diagonal=True)` SHALL be re-constructed as a fresh instance with a new seed (the underlying optimizer has no public reset method; re-construction is the equivalent operation, clearing the covariance state from the prior K-block's opposition)
- **AND** the re-construction SHALL happen exactly once per K-block transition
- **AND** the seed for the new instance SHALL be deterministic given the run's master seed and the K-block index, so checkpoint resume reproduces the same optimizer state
- **AND** `diagonal=True` (sep-CMA-ES) SHALL be set unconditionally for both sides since predator/prey weight counts (~5k for MLPPPO predator, ~30k+ for LSTMPPO prey) are above the n>~100 tractability threshold for full-covariance CMA-ES

#### Scenario: Block Elite Pushed To HoF

- **GIVEN** a side that has just completed its K-block (K generations evaluated)
- **WHEN** the K-block ends
- **THEN** the side's top-fitness genome from the block SHALL be pushed to its HoF
- **AND** the HoF push SHALL respect the configured eviction policy (default quality-based)

#### Scenario: Prey Gen-0 Warm-Start From M3 Lamarckian Elite

- **GIVEN** a `CoevolutionLoop` instance and a YAML config specifying `prey_gen0_seed_path: configs/evolution/coevolution_warmstart_prey/seed_<run_seed>.json`
- **WHEN** the loop initialises
- **THEN** the prey-side `CMAESOptimizer` SHALL be constructed with `x0` set to the loaded elite genome's weight vector (NOT zeros)
- **AND** the elite genome SHALL be sourced from prior M3-Lamarckian-LSTMPPO full-run elites (one elite per full-run seed, deterministic mapping)
- **AND** the warmstart bundle SHALL be committed in-repo at `configs/evolution/coevolution_warmstart_prey/` so a fresh checkout can run the campaign reproducibly (matching the held-out bundle convention from "Held-Out Set Construction")

#### Scenario: Predator Gen-0 Bootstrap Per D7 Pilot Ablation Arms

- **GIVEN** a `CoevolutionLoop` instance and a YAML config specifying `predator_gen0_bootstrap: "heuristic_imitation_pretrain"` (D7 arm A) or `"cold_start"` (D7 arm B)
- **WHEN** the loop initialises
- **THEN** for arm A, the predator-side `CMAESOptimizer` SHALL be constructed with `x0` set to the result of `instantiate_predator_brain_from_sim_config` followed by 50-episode `pretrain_against_heuristic` (per task 1.4)
- **AND** for arm B, the predator-side `CMAESOptimizer` SHALL be constructed with `x0=zeros` (random-init MLPPPO weights via the brain's own constructor)
- **AND** seed 42 of the pilot SHALL use arm A; seed 43 SHALL use arm B (per "Pilot Configuration" scenario)

### Requirement: Hall-of-Fame Opposition Sampling

When evaluating a candidate's fitness, the system SHALL draw the opposing side's evaluation pool from a 70% / 30% mixture of the current opposing population and the opposing-side hall-of-fame.

#### Scenario: 70/30 Mixture During Evaluation

- **GIVEN** an active K-block on side X with opposing side Y having a non-empty HoF
- **WHEN** a candidate genome on side X is evaluated against Y
- **THEN** approximately 70% of the evaluation episodes SHALL use opponents drawn from Y's current population
- **AND** approximately 30% SHALL use opponents drawn from Y's HoF
- **AND** sampling SHALL be deterministic given a seeded RNG

#### Scenario: Empty HoF Fallback

- **GIVEN** the opposing side Y has an empty HoF (e.g. at the start of the run before any K-block has completed)
- **WHEN** a candidate on side X is evaluated against Y
- **THEN** all evaluation episodes SHALL use opponents drawn from Y's current population

### Requirement: Generality Probe

The system SHALL evaluate the elite genome of each side against a held-out frozen opponent set every N generations (default N=10) to detect self-play overfitting versus real generalising progress.

#### Scenario: Held-Out Set Construction

- **GIVEN** a `CoevolutionLoop` configured with `held_out_size=8`
- **WHEN** the loop initialises
- **THEN** a held-out opponent set of size `held_out_size` SHALL be constructed for each side
- **AND** the prey-side held-out set SHALL be loaded from a committed in-repo bundle at `configs/evolution/coevolution_held_out_prey/*.json` (one genome per file, ~tens of KB each); the bundle SHALL contain at least `held_out_size` genomes drawn from prior M3-Lamarckian-style runs
- **AND** the predator-side held-out set SHALL be drawn from a heuristic-predator Cartesian grid `detection_radius × damage_radius`, with a deterministic widen-or-sub-sample strategy (`held_out_rng.choice` with a fixed seed) when the natural grid size differs from `held_out_size`
- **AND** held-out opponents SHALL NEVER be used in training evaluations
- **AND** held-out genome bundles SHALL be committed to the repo (NOT stored only in `artifacts/`) so a fresh checkout can run the campaign reproducibly

#### Scenario: Probe Cadence

- **GIVEN** `generality_probe_every=10`
- **WHEN** generation index G is reached such that `G % 10 == 0`
- **THEN** the loop SHALL evaluate each side's elite against the full held-out opponent set
- **AND** results SHALL be written to a `generality_probe.csv` file with columns `(generation, side, opponent_index, fitness)`

#### Scenario: Probe Does Not Mutate Population State

- **GIVEN** a generality probe in progress
- **WHEN** probe evaluations execute
- **THEN** they SHALL NOT alter either side's `population`, `optimizer` state, or `hof`
- **AND** they SHALL NOT advance the generation counter

### Requirement: Co-Evolution Decision Gate

The system SHALL evaluate full-run results against a softened-disjunctive decision gate: the M5 verdict GO requires either phenotypic cycling OR trait escalation to fire in at least 2 of 4 full-run seeds.

#### Scenario: Cycling Criterion

- **GIVEN** a per-generation Red Queen metric series for a single seed
- **WHEN** the verdict aggregator evaluates the cycling criterion
- **THEN** the criterion SHALL fire if either:
  - The Lomb-Scargle / autocorrelation peak for the series occurs at lag ∈ [3, 15] generations with significance p < 0.05, OR
  - The dominant FFT bin (excluding DC) has power greater than 2× the median bin power

#### Scenario: Escalation Criterion

- **GIVEN** a per-generation trait-mean series for a single seed
- **WHEN** the verdict aggregator evaluates the escalation criterion
- **THEN** the criterion SHALL fire if the linear regression of the series over generations 5..30 (skipping bootstrap noise) yields `|slope| / SE > 2.0` (significant non-zero slope, p < 0.05)
- **AND** the slope sign SHALL match the directional expectation declared for the trait in the aggregator's trait-spec table

#### Scenario: Verdict Aggregation Across Seeds

- **GIVEN** four full-run seeds with computed cycling/escalation results
- **WHEN** the aggregator emits the verdict
- **THEN** the verdict SHALL be `GO` iff (cycling fires OR escalation fires) in at least 2 of 4 seeds
- **AND** the verdict SHALL be `STOP` iff neither criterion fires in any seed
- **AND** the verdict SHALL be `PIVOT` iff exactly 1 of 4 seeds has a firing criterion

#### Scenario: Generality Probe Is Reported But Not A Verdict Input

- **GIVEN** a full-run aggregation
- **WHEN** the aggregator emits `summary.md`
- **THEN** the generality-probe trajectory SHALL be reported alongside the verdict
- **AND** the probe results SHALL NOT alter the GO/STOP/PIVOT decision

### Requirement: Pilot-First Sequencing

The system SHALL gate the full M5 campaign on a pilot run; pilot thresholds SHALL be locked into the OpenSpec change before the full run starts.

#### Scenario: Pilot Configuration

- **GIVEN** the pilot scenario (`coevolution_pilot.yml`)
- **THEN** it SHALL configure 30 generations × 2 seeds × prey-pop 24 × predator-pop 16
- **AND** seed 42 SHALL run the heuristic-imitation pretrain bootstrap arm
- **AND** seed 43 SHALL run the cold-start bootstrap arm

#### Scenario: Pilot Decision Gate

- **GIVEN** completed pilot results
- **WHEN** the pilot aggregator evaluates the pilot signal
- **THEN** if cycling OR escalation fires in at least 1 of the 2 pilot seeds, the pilot SHALL pass and the full run SHALL proceed
- **AND** if the signal is ambiguous (zero seeds firing), one additional seed SHALL run before committing to the full run
- **AND** if no signal is detected after the additional seed, the M5 verdict SHALL be STOP without running the full campaign

**Note on asymmetric pilot vs full thresholds:** the pilot uses a more permissive ≥1 of 2 bar (50%) because pilot is *calibration* (lock thresholds + choose pretrain on/off), not *verdict*. The stricter ≥2 of 4 bar (also 50% but stable across more seeds) applies only to the full run. A single-seed signal is sufficient to greenlight the full campaign for further investigation, but not sufficient to declare M5 GO.
