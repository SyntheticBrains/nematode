## ADDED Requirements

### Requirement: CoevolutionLoop Orchestrator

The system SHALL provide a `CoevolutionLoop` orchestrator that drives two populations (prey and predator) through an alternating-schedule co-evolution run. The loop SHALL compose two side-state objects (each with its own encoder, fitness, optimiser, inheritance strategy, hall-of-fame, and population). The per-evaluation interface SHALL align with the existing `EvolutionLoop._evaluate_in_worker` 11-tuple ABI so the loop can swap in `multiprocessing.Pool` + `pool.map(_evaluate_in_worker, eval_args)` for parallelism without changing the loop body. Opponent brain weights are injected at evaluation time via `sim_config` patching at the call site (the worker tuple ABI is unchanged).

#### Scenario: Side State Surface

- **GIVEN** a `CoevolutionLoop` instance
- **THEN** each side state SHALL carry an `encoder: GenomeEncoder`, `fitness: FitnessFunction`, `optimizer: CMAESOptimizer` constructed with `diagonal=True` (sep-CMA-ES, imported from `quantumnematode.optimizers.evolutionary`), `inheritance: InheritanceStrategy`, `hof: HallOfFame`, `population: list[Genome]`, and `champion_history: list[dict[str, Any]]`
- **AND** the prey side state SHALL be configured with `LearnedPerformanceFitness` (Lamarckian-style: K_train=50 inner-loop training + L_eval=25 frozen evaluation episodes per genome) plus `LamarckianInheritance`, mirroring the M3 Lamarckian-LSTMPPO winner stack
- **AND** the predator side state SHALL be configured with `PredatorEpisodicKillRate` (frozen-weight evaluation only: N_eval=25 multi-agent episodes per genome, NO inner-loop PPO training, NO weight capture) plus `NoInheritance`
- **AND** the asymmetric fitness shapes are intentional per design.md D13 — prey trains per evaluation (large policy space, M3 substrate); predator evaluates frozen (small policy space, direct CMA-ES weight gradient suffices)
- **AND** `hof` SHALL be a bounded buffer with eviction policy (the runtime opposition-sampling pool)
- **AND** `champion_history` SHALL be unbounded — exactly one entry per completed K-block (the K-block's top-fitness genome at K-block end), never evicted; the time-ordered audit log walked by aggregator-time analysis (cycling, escalation). Distinct from per-generation lineage rows AND distinct from `hof`. A K-block's elite genome lands in BOTH `hof` (subject to eviction) and `champion_history` (unbounded). Each entry is a dict `{genome_id: str, generation: int, k_block_index: int, fitness: float, params: list[float]}` — JSON-serialisable so the aggregator can read it without a numpy dependency

#### Scenario: Composition Over Inheritance

- **GIVEN** the `CoevolutionLoop` class
- **THEN** it SHALL NOT subclass `EvolutionLoop`
- **AND** the per-evaluation call shape (`fitness.evaluate(genome, sim_config, encoder, *, episodes, seed)`) SHALL be compatible with the existing 11-tuple worker pattern (`(params, sim_config, encoder, fitness, episodes, seed, generation, index, parent_ids, warm_start_path_override, weight_capture_path)`) so that `pool.map(_evaluate_in_worker, eval_args)` can be wired in with no body change
- **AND** the loop dispatch is sequential at first ship; multi-process parallelism is a swap-in runtime detail behind the 11-tuple worker shape
- **AND** opposing-side opponent brain weights SHALL be injected at evaluation time by patching `sim_config` at the call site. The integration pattern: decode each opposing genome via `opposing_side.encoder.decode(...)`, materialise the brain's weights to a tempfile-managed `.pt`, then patch:
  - prey-training: `sim_config.environment.predators.brain_config.extra["weights_path"]` — the env's `_build_predator_brain` honours this key and calls `load_weights` on the freshly-constructed predator brain. **Diversity simplification:** all N predator slots load the SAME opposition genome (the head of the HoF-mixed list), matching the focal genome's "same brain on every slot" semantic. The 70/30 HoF mix still varies across evaluations of the same K-block, so opposition diversity is exercised at the call level rather than the slot level. A future refinement could rotate genomes per slot if pilot evidence shows insufficient diversity.
  - predator-training: `sim_config.multi_agent.agents[i].weights_path` (one per opposition genome) — `_build_prey_agents` honours this key and calls `load_weights` post-construction. The opposition list is **capped at 10 entries** (the schema cap on `MultiAgentConfig.agents`); larger pop sizes truncate to the first 10 opposition genomes after the HoF mix. With `frac_hof=0.3`, the 10 entries cover ~3 HoF + ~7 current-pop in the typical pop=24 case.
- **AND** `sim_config.evolution` SHALL be patched to the training side's per-side `EvolutionConfig` so per-side fitness fields (`learn_episodes_per_eval`, `eval_episodes_per_eval`) resolve correctly for `LearnedPerformanceFitness`. The base sim_config carries `evolution=None` for co-evolution runs (the YAML uses the `coevolution:` block instead)
- **AND** empty opposition (first K-block before the opposing side has trained) SHALL bootstrap with random-init opposing brains: prey side gets random-init predator weights via the env's native `mlpppo_predator` dispatcher; predator side spawns one random-init prey opponent so the multi-agent runner has at least one slot
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
- **AND** `diagonal=True` (sep-CMA-ES) SHALL be set unconditionally for both sides since predator/prey weight counts (~10k for MLPPPO predator actor + value head, ~30k+ for LSTMPPO prey) are above the n>~100 tractability threshold for full-covariance CMA-ES

#### Scenario: Block Elite Pushed To HoF

- **GIVEN** a side that has just completed its K-block (K generations evaluated)
- **WHEN** the K-block ends
- **THEN** the side's top-fitness genome from the block SHALL be pushed to its HoF
- **AND** the HoF push SHALL respect the configured eviction policy (default quality-based)

#### Scenario: Prey Gen-0 Warm-Start From M3 Lamarckian Elite

- **GIVEN** a `CoevolutionLoop` instance and a YAML config specifying `prey_gen0_seed_path: configs/evolution/coevolution_warmstart_prey/seed_<run_seed>.json` (where `<run_seed>` is a template variable substituted by the campaign driver — `run_coevolution.py` — based on the `--seed` CLI argument before the path is passed to `CoevolutionLoop.__init__`)
- **WHEN** the loop initialises
- **THEN** the prey-side `CMAESOptimizer` SHALL be constructed with `x0` set to the loaded elite genome's weight vector (NOT zeros)
- **AND** the elite genome SHALL be sourced from prior M3-Lamarckian-LSTMPPO full-run elites (one elite per full-run seed, deterministic mapping)
- **AND** the warmstart bundle SHALL be committed in-repo at `configs/evolution/coevolution_warmstart_prey/` so a fresh checkout can run the campaign reproducibly (matching the held-out bundle convention from "Held-Out Set Construction")

#### Scenario: Predator Gen-0 Bootstrap Per D7 Pilot Ablation Arms

- **GIVEN** a `CoevolutionLoop` instance and a YAML config specifying `predator_gen0_bootstrap: "heuristic_imitation_pretrain"` (D7 arm A) or `"cold_start"` (D7 arm B)
- **WHEN** the loop initialises
- **THEN** for arm A, the predator-side `CMAESOptimizer` SHALL be constructed with `x0` set to the flattened weights of a fresh `MLPPPOPredatorBrain` (default constructor) trained for 50 batches via `pretrain_against_heuristic` (the helper at `quantumnematode/env/_predator_brain_pretrain.py`). Note: PR 3 calls `MLPPPOPredatorBrain()` directly rather than going through `instantiate_predator_brain_from_sim_config` because gen-0 init is a one-shot operation that doesn't need the full sim_config + factory plumbing — orthogonal-init weights are about to be overwritten by 50 batches of pretrain anyway. The encoder factory IS used at decode time (per genome) inside `MLPPPOPredatorEncoder.decode`, which is the load-bearing path the factory was designed for.
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

- **GIVEN** a `CoevolutionLoop` configured with `held_out_size=N` (the schema default is 8 for forward-compat with future expanded bundles; production YAMLs ship with `held_out_size=4` to match the curated bundle exactly — see footnote)
- **WHEN** the loop initialises
- **THEN** a held-out opponent set of size `held_out_size` SHALL be constructed for each side
- **AND** the prey-side held-out set SHALL be loaded from a committed in-repo bundle at `configs/evolution/coevolution_warmstart_prey/*.json` (one genome per file, ~1.2 MB each at LSTMPPO + klinotaxis brain shape; routed through Git LFS via `.gitattributes`). The same bundle directory serves as both the gen-0 warm-start anchor source (per-seed `prey_gen0_seed_path`) and the held-out probe opponents — one bundle, two roles, avoiding ~5 MB of byte-identical duplication. When the bundle ships fewer than `held_out_size` distinct genomes, the loader samples WITH replacement (so the configured set size is honoured at the cost of sample repetition); when at least `held_out_size` are available, the loader samples WITHOUT replacement
- **AND** the predator-side held-out set SHALL be drawn from a heuristic-predator Cartesian grid `detection_radius × damage_radius` (default `{4, 6, 8, 10} × {0, 1}` = 8 combos at default `held_out_size=8`); when `held_out_size > grid_size` the rng samples WITH replacement; when `held_out_size < grid_size` the rng samples WITHOUT replacement; both via `held_out_rng.choice` with a fixed seed
- **AND** held-out opponents SHALL NEVER be used in training evaluations
- **AND** held-out genome bundles SHALL be committed to the repo (NOT stored only in `artifacts/`) so a fresh checkout can run the campaign reproducibly
- **Footnote on bundle size:** the production prey bundle ships 4 distinct genomes (one per source-campaign seed). The original "8 genomes, 2 per seed" plan reduced because the source single-population campaign retained only the final-generation elite checkpoint per seed. All shipped co-evolution YAMLs (`coevolution_pilot_arm_a.yml`, `coevolution_pilot_arm_b.yml`, `coevolution_full.yml`) set `held_out_size: 4` to avoid implicit with-replacement sampling. The schema default stays at 8 so future expanded bundles can drop in without a config-schema migration

#### Scenario: Probe Cadence and Output Layout

- **GIVEN** `generality_probe_every=10` and a `CoevolutionLoop` configured with `output_dir`
- **WHEN** generation index G is reached such that `G % 10 == 0`
- **THEN** the loop SHALL evaluate each side's elite against the full OPPOSING-side held-out opponent set (per "Probe Held-Out Wiring (Option B)" scenario): the prey elite is evaluated against `_predator_held_out_specs` (heuristic predator radius variants), and the predator elite is evaluated against `_prey_held_out` (frozen prey genomes from the M3 lamarckian bundle)
- **AND** results SHALL be written to `{output_dir}/generality_probe.csv` (top-level, single file across both sides) with columns `(generation, side, opponent_index, fitness)`. The `fitness` column SHALL be a real-valued float produced by routing the elite + per-opponent patched `sim_config` through `_evaluate_in_worker` with `warm_start_path_override=None, weight_capture_path=None` (eval-only — no germline weight capture, no inheritance side-effects). The probe runs sequentially in the master process — pool dispatch overhead is not worth the complication at probe volumes (~8 evals every 10 gens)
- **AND** the probe SHALL be a no-op for sides whose `champion_history` is empty (the first K-block on each side hasn't completed yet, so there is no elite to probe)
- **AND** per-side lineage CSVs SHALL live at `{output_dir}/prey/lineage.csv` and `{output_dir}/predator/lineage.csv` (per-side subdirs match the existing `EvolutionLoop` output shape — M3 single-population analysis tooling reuses unchanged)
- **AND** champion_history SHALL live at `{output_dir}/champion_history.json` (top-level, single file with `prey` + `predator` dict keys)

#### Scenario: Probe Held-Out Wiring (Option B)

- **GIVEN** a generality probe firing for a given side
- **WHEN** the probe iterates per-opponent evaluations
- **THEN** the prey-side branch SHALL iterate over `_predator_held_out_specs` (heuristic-radius `(detection_radius, damage_radius)` tuples). For each opponent index, the loop SHALL patch `sim_config.environment.predators` to a heuristic predator at the held-out spec — flipping `brain_config.kind` from `mlpppo_predator` to `heuristic` and clearing any pre-existing `extra["weights_path"]` — so the held-out yardstick is independent of the co-evolution lineage
- **AND** the predator-side branch SHALL iterate over `_prey_held_out` (frozen prey `Genome` entries). For each opponent index, the loop SHALL patch `sim_config.multi_agent.agents` to install one held-out prey's weights via `weights_path`, padding to the schema's `MultiAgentConfig._validate_population` minimum (2 agents) with random-init bootstrap entries when needed
- **AND** both branches SHALL set `sim_config.evolution = side.evolution_config` so the side's `episodes_per_eval` (and, where applicable, `learn_episodes_per_eval` / `eval_episodes_per_eval`) drive the per-opponent evaluation budget — keeping probe-fitness magnitudes directly comparable to training-time `lineage.csv` fitness rows on the same axis
- **Rationale:** matches the published Sakana 2025 generality-probe protocol — focal elite vs. frozen *opposing-side* yardstick, never peer-comparison vs. same-side held-out — and corresponds to the cross-species fitness measurement used in published microbial co-evolution chemostats (Gallet 2018; Hall 2011). A rising probe-fitness curve = real escalation; flat or falling = self-play overfitting

#### Scenario: Held-Out Bundle Missing Falls Back To No-Op

- **GIVEN** a `CoevolutionLoop` instance whose prey reference bundle directory (`configs/evolution/coevolution_warmstart_prey/`) is missing or empty
- **WHEN** the loop initialises and constructs the prey held-out set
- **THEN** the loader SHALL log a one-time warning and return an empty list rather than raising
- **AND** under Option B wiring (per "Probe Held-Out Wiring (Option B)") the predator-side probe SHALL be a no-op for that run because `_prey_held_out` is its opponent set; the prey-side probe still fires unaffected because it iterates `_predator_held_out_specs` (heuristic-radius grid, constructed at `__init__` without a bundle)
- **Rationale:** allows fresh-checkout unit tests + smoke runs to exercise the loop end-to-end on machines where the production bundle is unavailable; the production bundle ships in-repo so the warning is a no-op for normal runs.

#### Scenario: Checkpoint File Layout

- **GIVEN** a `CoevolutionLoop` instance configured with `output_dir`
- **WHEN** the loop reaches a K-block boundary and `_save_checkpoint` fires
- **THEN** five files SHALL be written atomically (tmp file + rename per file):
  - `{output_dir}/prey/checkpoint.pkl` — per-side pickle: optimizer + population_params + population_genome_ids + prev_generation_ids + generation + champion_history + checkpoint_version + k_block_index. Reuses the single-population `EvolutionLoop._save_checkpoint` shape so single-population resume tooling can introspect.
  - `{output_dir}/predator/checkpoint.pkl` — same shape as prey.
  - `{output_dir}/coevolution_state.json` — top-level human-readable JSON: k_block_index + current_side + prey_hof + predator_hof (via `HallOfFame.to_dict`) + predator_held_out_specs + prey_held_out_ids + k_block_mean_fitness (rebalance heuristic state).
  - `{output_dir}/champion_history.json` — top-level JSON, `{prey: list[dict], predator: list[dict]}` plus a `k_block_index` field for cross-file consistency. Time-ordered audit log of K-block elites; consumed by aggregator-time analysis (cycling, escalation) without a numpy dependency on the per-side pickles.
  - `{output_dir}/coevolution_rng.pkl` — master RNG state + held-out RNG state + immutable `run_seed` (used by `_derive_optimizer_seed`) + k_block_index. Pickled because numpy bit_generator state has nested arrays that don't JSON natively.
- **AND** the RNG pickle SHALL be written LAST and SHALL hold the canonical `k_block_index`; `_load_checkpoint` SHALL fail if the RNG pickle is absent (signals the prior save was interrupted mid-write — refuse to resume rather than load partial state).
- **AND** every file SHALL embed `k_block_index`; `_load_checkpoint` SHALL cross-check all four non-RNG files' `k_block_index` against the canonical RNG-pickle value and raise `ValueError` on any mismatch (torn-save detection, naming the divergent file in the diagnostic).
- **AND** `_load_checkpoint` SHALL verify `checkpoint_version` on each per-side pickle; mismatch raises `ValueError`. The prey held-out bundle SHALL be cross-checked against `prey_held_out_ids` recorded in the JSON; bundle drift between save and resume raises `ValueError` to prevent silent state corruption.

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
- **THEN** the criterion SHALL fire if `phenotypic_cycling(series, lag_range=(3, 15))` returns `cycling_detected=True` (i.e. the OLS-detrended autocorrelation peak in the lag range has permutation-null `p_value < 0.05`); see "Phenotypic Cycling Metric" in `red-queen-analysis/spec.md` for the full algorithm
- **Implementation note:** earlier drafts of this scenario disjoined an FFT-bin-power threshold ("dominant FFT bin > 2× median bin power") with the autocorrelation peak. The shipped `phenotypic_cycling` uses autocorrelation only — it suffices for the verdict-gate inference task and avoids a scipy dependency. The verdict aggregator (PR 5) consumes the autocorrelation result directly.

#### Scenario: Escalation Criterion

- **GIVEN** a per-generation trait-mean series for a single seed
- **WHEN** the verdict aggregator evaluates the escalation criterion
- **THEN** the criterion SHALL fire if the linear regression of the series over generations 5..30 (skipping bootstrap noise) yields `|slope| / SE > 2.0` (significant non-zero slope, p < 0.05)
- **AND** the slope sign SHALL match the directional expectation declared for the trait in the aggregator's trait-spec table

#### Scenario: Verdict Aggregation Across Seeds

- **GIVEN** four full-run seeds with computed cycling/escalation results
- **WHEN** the aggregator emits the verdict
- **THEN** the verdict SHALL be `GO` iff (cycling fires OR escalation fires) in at least 2 of 4 seeds
- **AND** the verdict SHALL be `STOP` iff neither criterion fires in any seed (with at least one seed resolved)
- **AND** the verdict SHALL be `PIVOT` iff exactly 1 of 4 seeds has a firing criterion
- **AND** the verdict SHALL be `INCONCLUSIVE` iff zero seeds were resolvable (e.g. all per-seed dirs missing or all sessions corrupted), distinct from `STOP` (which is a substantive null result over ≥1 resolved seeds)

#### Scenario: Generality Probe Is Reported But Not A Verdict Input

- **GIVEN** a full-run aggregation
- **WHEN** the aggregator emits `summary.md`
- **THEN** the generality-probe trajectory SHALL be reported alongside the verdict
- **AND** the probe results SHALL NOT alter the GO/STOP/PIVOT decision

### Requirement: Pilot-First Sequencing

The system SHALL gate the full M5 campaign on a pilot run; pilot thresholds SHALL be locked into the OpenSpec change before the full run starts.

#### Scenario: Pilot Configuration

- **GIVEN** the two pilot arm YAML files (`coevolution_pilot_arm_a.yml` and `coevolution_pilot_arm_b.yml`)
- **THEN** each SHALL configure **30 per-side generations** (3 K-pairs × K_per_block=10 = 30 gens per side; total wall-clock loop generations = 60) × prey-pop 24 × predator-pop 16 × K=10 × HoF=8 × probe every 10 gens. Specifically, `generation_pairs=3`, `K_per_block=10`, `prey_evolution.population_size=24`, `predator_evolution.population_size=16`, `held_out_size=4` (matching the curated 4-genome held-out bundle; the loader samples WITH replacement when `held_out_size > len(bundle)` so an oversized config still works at the cost of sample repetition), `generality_probe_every=10`.
- **AND** arm A SHALL run with seed=42 and `predator_gen0_bootstrap: "heuristic_imitation_pretrain"` (D7 arm A)
- **AND** arm B SHALL run with seed=43 and `predator_gen0_bootstrap: "cold_start"` (D7 arm B)
- **AND** the bash wrapper `phase5_m5_coevolution_pilot.sh` SHALL run both arms sequentially with distinct output directories

#### Scenario: Pilot Decision Gate

- **GIVEN** completed pilot results
- **WHEN** the pilot aggregator evaluates the pilot signal
- **THEN** if cycling OR escalation fires in at least 1 of the 2 pilot seeds, the pilot SHALL pass and the full run SHALL proceed
- **AND** if the signal is ambiguous (zero seeds firing), one additional seed SHALL run before committing to the full run
- **AND** if no signal is detected after the additional seed, the M5 verdict SHALL be STOP without running the full campaign

**Note on asymmetric pilot vs full thresholds:** the pilot uses a more permissive ≥1 of 2 bar (50%) because pilot is *calibration* (lock thresholds + choose pretrain on/off), not *verdict*. The stricter ≥2 of 4 bar (also 50% but stable across more seeds) applies only to the full run. A single-seed signal is sufficient to greenlight the full campaign for further investigation, but not sufficient to declare M5 GO.
