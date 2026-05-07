## Context

M5 is the headline scientific milestone for Phase 5 — a Red Queen arms race between learnable prey (LSTMPPO+klinotaxis, the M3 Lamarckian winner) and learnable predators (MLPPPO with predator-specific I/O). M1 (just merged) shipped the substrate: `PredatorBrain` Protocol + `HeuristicPredatorBrain` adapter + per-predator metrics in `MultiAgentEpisodeResult`. The substrate is byte-equivalent against the legacy `_update_pursuit` / `_update_random` heuristic (23 byte-equivalence tests + 80 metric-cells with 0.0 delta in the M1 regression baseline). What's missing is everything *above* the substrate: a learnable predator brain, predator-side encoder + fitness, a two-population co-evolution loop, hall-of-fame opposition, generality probe, and Red Queen analysis primitives.

The codebase already provides a single-population evolution stack (M2/M3): `EvolutionLoop` with multiprocessing 11-tuple worker, `GenomeEncoder` Protocol + `ENCODER_REGISTRY`, `FitnessFunction` Protocol, `InheritanceStrategy` Protocol with `LamarckianInheritance` (M3 GO winner), `CMAESOptimizer` (M0/M3 weight-evolution canonical) and `OptunaTPEOptimizer` (M2.12 RQ1 winner for hyperparameter evolution). The design challenge is to **layer a co-evolution orchestrator on top** rather than fork the loop — reuse `_evaluate_in_worker` per side, with each side carrying its own encoder + fitness + (optional) inheritance strategy. M5 evolves predator/prey **weights** (not hyperparameters), so `CMAESOptimizer(diagonal=True)` is the optimiser of choice; TPE is structurally incompatible with weight encoders (see D2).

A second design constraint comes from M4's STOP closure: single-task K=50 PPO has no Baldwin axis. M5 doesn't gate on Baldwin (the verdict is purely Red Queen dynamics), but co-evolution intrinsically introduces task variation — the prey's "task" is to survive the *current* predator pop, which shifts every K predator generations. M5.7 ships secondary Baldwin instrumentation that observes this naturally varying task distribution; if the readout fires it triggers M4.7 follow-up rather than re-opening M4.

**Stakeholders:** chris (Phase 5 lead, decision authority on verdict gates and compute budget). The OpenSpec change `add-coevolution-arms-race` (this change) is the spec contract; `openspec/changes/2026-04-26-phase5-tracking/tasks.md` (M5.0–M5.9) is the milestone tracker that this change ticks.

## Goals / Non-Goals

**Goals:**

- Ship a learnable `MLPPPOPredatorBrain` that satisfies the M1 `PredatorBrain` Protocol byte-for-byte (no substrate breakage; `HeuristicPredatorBrain` remains the default for any scenario YAML that doesn't opt in to `kind: mlpppo_predator`).
- Build a `CoevolutionLoop` that orchestrates two populations under an alternating training schedule (K=10 gens per side, opposing pop frozen during off-block, per-K-block fresh `CMAESOptimizer(diagonal=True)` instance to clear stale opposition-conditioned covariance).
- Provide hall-of-fame opposition (70%/30% current-pop / HoF mixture, HoF size 8, quality-based eviction) to prevent cycling-without-progress.
- Provide a generality probe (every 10 gens against 8 frozen held-out opponents) to flag self-play overfitting.
- Provide Red Queen analysis primitives (cycling, escalation, fitness lag, coupled rate, generality) and a softened-disjunctive verdict gate (cycling OR escalation in ≥2 of 4 full-run seeds → GO).
- Ship M5.7 secondary Baldwin instrumentation (reduced under CMA-ES weight evolution: signal-delta only, hyperparam-spread criterion dropped per D11) that reuses M4.5's F1 evaluator wholesale; observation only (does NOT alter M5 verdict).
- Pilot-first sequencing: 30 gens × 2 seeds (~8 wall-hours) calibrates Red Queen thresholds + chooses pretrain on/off; full 50–70 gens × 4 seeds (~30–40 wall-hours) delivers the verdict.

**Non-Goals:**

- GA optimiser ablation. Literature recommends GA for primary co-evolution because of objective drift, but the alternating schedule flattens drift within K-blocks and CMA-ES (sep-CMA-ES with `diagonal=True`) is M0/M3's canonical optimiser for unbounded weight evolution. Switching introduces re-validation cost mid-milestone. Reserved as follow-up if CMA-ES diversity collapses.
- Predator-side pheromones / signalling. Predators stay solitary; coordination would muddle the trait-escalation signal.
- Multi-predator cooperation (intra-pop predator-vs-predator dynamics).
- Transfer to single-agent runners. M5 verdict is purely an evolution-loop result; transfer evaluation lives in M5.6 as post-pilot ablation.
- Quantum predator brains. Out of scope per M5 brain-target spec (MLPPPO predator).
- Architecture-changing schema knobs for the predator. Only weights co-evolve (matches M2.10's Lamarckian/Baldwin invariants).
- Baldwin VERDICT under M5. M5.7 is observation only; M4 STOP unchanged unless readout fires → triggers M4.7 follow-up.
- Per-predator visualisation badges in the renderer. Out of scope.

## Decisions

### D1. Training schedule: alternating, K=10 gens per side

**Decision:** For 10 generations, freeze predator pop at the prior K-block elite (HoF-mix sample) and evolve prey; then swap. K=10, 3 swap pairs across a 30-gen pilot, 5–7 across a 50–70-gen full run.

**Rationale:** Entropy 2021 review favours alternating over simultaneous for stability. Simultaneous co-evolution makes both objectives non-stationary every generation, which TPE handles poorly (TPE's KDE assumes a stationary objective). Alternating gives each side ~K stationary gens before the opponent flips. K=10 is short enough that within-block PPO has time to train (each gen runs K=50 train + L=25 eval episodes per evaluated genome), and long enough to amortise per-block TPE-restart overhead.

**Asymmetric start-side note:** with `start_side="prey"` (default), prey trains first against gen-0 predators. When predators are bootstrapped from heuristic-imitation pretrain (D7 arm A), gen-0 prey trains against bootstrapped predators rather than random-init ones — meaningful asymmetry. With cold-start predators (D7 arm B) gen-0 prey trains against random-policy predators, which is gentle bootstrap. Pilot ablation reveals which setup yields the cleaner Red Queen signal; full run's `start_side` choice flips if pilot evidence suggests predator-first works better.

**Alternatives:**

- Simultaneous (rejected: TPE-stationarity violation; literature warns against it).
- Alternating K=20 (rejected: coarser co-evolution signal — only 1–2 swaps in a 30-gen pilot, harder to detect cycling).
- Alternating K=5 (rejected: too short for within-block PPO convergence; TPE-restart overhead dominates).

**Confirmed by user (Phase 3 question).**

### D2. Optimiser: CMA-ES (sep-CMA-ES, `diagonal=True`) for both sides, with per-K-block fresh-instance construction

**Decision:** `CMAESOptimizer` (canonical import path `quantumnematode.optimizers.evolutionary.CMAESOptimizer`, [evolutionary.py:105](packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py#L105)) per side with `diagonal=True` (sep-CMA-ES — drops `tell()` cost from O(n²) to O(n), tractability requirement for neuroevolution at n > ~100; full MLPPPO predator weight count is ~5k params at the input dim of 11 → 64 → 64 → 5 + value head). At the start of each K-block, the just-flipped side's optimizer is **re-constructed as a fresh instance** with a new seed so the covariance matrix doesn't carry stale opposition-conditioned adaptation. The frozen side's optimizer is unaffected.

**Rationale (revised from earlier TPE decision — see "Why this changed" below):**

- **TPE is incompatible with weight encoders.** `OptunaTPEOptimizer.__init__` requires `bounds: list[tuple[float, float]]` ([evolutionary.py:519-577](packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py#L519-L577)) and rejects construction without them. `_ClassicalPPOEncoder.genome_bounds` returns `None` with the explicit comment *"TPE-based optimisers can't be used with weight encoders for this reason; CMA-ES handles unbounded search natively."* ([encoders.py:330-342](packages/quantum-nematode/quantumnematode/evolution/encoders.py#L330-L342)). The earlier "TPE for both sides" decision was based on the M2.12 RQ1 closure, but that closure was for **hyperparameter** evolution (bounded schema), not weight evolution (unbounded). For M5 the predator/prey are evolving weights, so TPE is structurally incompatible — construction would fail at run start.
- **CMA-ES handles unbounded weights natively** and is M0/M3's canonical optimiser for weight evolution. M3's Lamarckian-LSTMPPO closed GO using CMA-ES; the pattern is proven at the scale we need.
- **`diagonal=True` (sep-CMA-ES) is the neuroevolution-scale knob.** Full-covariance CMA-ES becomes minutes per `tell()` at n > ~1000 and won't fit the K=10 / K-block budget. Sep-CMA-ES gives up off-diagonal covariance adaptation (~2-10x more generations to converge per Ros & Hansen 2008) but each generation is O(n) instead of O(n²). Net wall-clock is dramatically faster at large n.
- **Per-block fresh construction** still applies — covariance state should reset when the opponent flips. Costs ~6 gens of cold-restart exploration per seed (3 K-block swaps × 2 sides), negligible against the ~30-gen pilot budget.
- **Re-construction rather than reset-in-place** keeps the optimizer base-class surface unchanged. Resume-from-checkpoint code only needs to serialise the *seed sequence* used per K-block, not the in-flight CMA-ES state.

**Why this changed (vs the earlier TPE decision):** the earlier Phase-3-confirmed answer of "TPE for both sides with per-block reset" was based on an incorrect premise: that TPE could evolve weights. Code review revealed `_ClassicalPPOEncoder.genome_bounds = None` rejects bounded sampling, and `OptunaTPEOptimizer` requires bounds at construction. The B8 finding in the second-pass review forced a re-decision. CMA-ES is the correct fit for weight evolution; the fix doesn't compromise the M5 verdict gate (Red Queen dynamics are agnostic to which optimiser produces the weights). One downstream effect: M5.7's "hyperparam spread tightens by ≥30%" condition is dropped because there is no hyperparam spread under weight evolution (see D11 + tracker M5.7 update).

**Alternatives:**

- TPE for both sides (REJECTED — structurally incompatible with weight encoders, see rationale above).
- GA both sides (rejected: CMA-ES outperforms GA at neural-network weight evolution per M0/M3 calibration; reserved as ablation if CMA-ES diversity collapses).
- Hyperparameter evolution + TPE for prey, weight evolution + CMA-ES for predator (rejected: asymmetric substrate confounds Red Queen dynamics interpretation; pilot wall blows past budget because hyperparameter evaluation requires K=50 inner-loop training per genome; reserved as M4.7 follow-up under a dedicated hyperparameter-evolution milestone).
- Add a `restart(seed)` method to `EvolutionaryOptimizer` and reset in-place (rejected: enlarges base-class surface mid-milestone for one consumer).

**User confirmation:** revised in second-pass review after B8 finding — user accepted CMA-ES switch with awareness that M5.7 reduces to a single (signal-delta) Baldwin readout.

### D3. Hall-of-fame opposition: 70% current pop / 30% HoF, HoF size 8, quality-based eviction

**Decision:** When evaluating a candidate's fitness against the opposite side, sample opponents from a 70%/30% mixture: 70% from the current opposing population, 30% from a per-side hall-of-fame `deque[Genome]` of size 8. At the end of each K-block, the side that just trained pushes its block elite to its HoF; if HoF is full, the lowest-fitness HoF entry is evicted (NOT FIFO — preserves quality, not recency).

**Rationale:** Sakana DRQ 2025 found HoF prevents cycling-without-progress (two populations chase each other in trait space without ever beating their old selves). 70/30 mix preserves the live signal as the primary driver while ensuring no champion is lost to forgetting. Quality-based eviction means HoF reflects the strongest opposition seen rather than just the most recent.

**Alternatives:**

- FIFO eviction (rejected: a strong early champion can be evicted by a weak recent one).
- Pure quality + no recency (the `deque` size of 8 acts as the recency cap; older champions of equal quality stay until evicted by a strictly better one).
- HoF size 16 (rejected: each HoF opponent adds an episode to per-eval cost; 8 fits within the K=20 episodes-per-eval budget).
- 50/50 or 90/10 mix ratios (rejected: 70/30 gives the live signal majority while preventing forgetting; tuneable post-pilot if cycling shows persistence).

**Confirmed by user (Phase 3 — recommended core, not stretch).**

### D4. Population sizes: prey 24, predator 16 (asymmetric)

**Decision:** 24 prey genomes per generation, 16 predator genomes.

**Rationale:** Prey have richer phenotype surface (LSTMPPO+klinotaxis carries Phase-4's bilateral-sensing + recurrent-state machinery — substantially more degrees of freedom than the predator's MLPPPO with limited perception). Asymmetric pops are a standard co-evolution heuristic — the side with more phenotypic variety needs more samples to explore.

Compute envelope: ~40 evaluations per generation pair × ~24 episodes/eval × ~K=50 train + L=25 eval = ~96k episode-equivalents per K-block ≈ ~40 min/K-block × 6 K-blocks = ~4 wall-hours/seed at parallel_workers=4.

**Alternatives:**

- Symmetric 24/24 (rejected: ~+50% per-gen cost for the predator side; predator phenotype space doesn't justify it).
- Symmetric 16/16 (rejected: too small for prey trait dim).
- Smaller 16/12 (deferred: stretch fallback if pilot compute headroom is tight; 24/16 is the default).
- Larger 32/24 (rejected: ~6 hours/seed × 4 full-run seeds blows compute budget).

**Confirmed by user (Phase 3 — 24/16 default).**

### D5. Generality probe: every 10 gens against 8 frozen held-out opponents

**Decision:** Every 10 generations, evaluate the current population's elite (top-1) on each side against a *held-out* opponent set — `held_out_size` opponents (default 8) pre-built at run start that have NEVER been used in training. Held-out set sources:

- **Prey side:** drawn from a small bundle of pre-trained M3-Lamarckian-style prey elite genomes committed to the repo at `configs/evolution/coevolution_held_out_prey/*.json` (8 genomes × ~tens of KB each). Committing the bundle keeps the probe reproducible on a fresh checkout — gitignored `artifacts/` paths cannot be relied on.
- **Predator side:** drawn from heuristic-radius variants spanning a configurable Cartesian grid `detection_radius × damage_radius`. Default grid is `detection_radius ∈ {4, 6, 8, 10} × damage_radius ∈ {0, 1}` (8 combos = `held_out_size=8`); when `held_out_size` differs, the loop SHALL widen or sub-sample the grid deterministically (e.g. via `held_out_rng.choice` with a fixed seed) so the held-out set count always matches `held_out_size`.

**Rationale:** Sakana 2025 — generality probe is the discriminator between "escalation" (real progress generalising to held-out opposition) and "self-play overfitting" (training fitness climbs while held-out flat or drops). The probe is reported alongside the verdict gate but is not itself a verdict input — softened-disjunctive (cycling OR escalation in ≥2 of 4 seeds) is the gate, generality is the *interpretation guide*.

**Alternatives:**

- Probe every 5 gens (rejected: more cost without proportional value; signal smoothed at gen-10 cadence).
- Probe every 20 gens (rejected: only 1–2 datapoints in a 30-gen pilot; under-sampled).
- Held-out set built from random-policy opponents (rejected: too easy; probe saturates at 1.0).
- Held-out set built from co-evolution lineage (rejected: not held-out — opponents that the population may already have seen indirectly).

### D6. Decision gate: softened-disjunctive (cycling OR escalation), ≥2 of 4 seeds

**Decision:** Full-run GO if at least one of:

- **(a) Phenotypic cycling**: Lomb-Scargle / autocorrelation peak at lag ∈ [3, 15] gens with p < 0.05, OR dominant FFT bin (excluding DC) has power > 2× median bin power. Applied to per-gen mean of the four Red Queen metrics.
- **(b) Trait escalation**: linear regression of mean trait value over gens 5–30 (skip TPE bootstrap noise) has |slope|/SE > 2.0 (p < 0.05) AND sign aligned with directional expectation per trait.

…fires in **≥2 of 4 full-run seeds**. STOP if neither fires in any seed. PIVOT if exactly 1/4. The thresholds above are the spec's default falsification gate; pilot results may motivate revising them, in which case **the OpenSpec change SHALL be amended (and re-validated `--strict`) BEFORE the full run launches**. The spec is the contract; we don't run the full campaign with thresholds different from what's written here. A pilot-revision PR (small, limited to spec + design rewording) is acceptable mid-flight if the change `add-coevolution-arms-race` has not yet been archived.

**Rationale:** Sci Reports 2026 study cited in tracker — stable populations and persistent trait cycling can coexist; demanding monotone escalation alone rules out a known-realistic regime. Disjunctive lowers false-STOP risk on a substrate that has only been validated for trait-evolution viability via M1's predator-brain refactor. ≥2 of 4 seeds is the same stability threshold M3 / M4 used.

**Alternatives:**

- Conjunctive (cycling AND escalation; rejected: too strict, rules out cycling-only regime).
- ≥3 of 4 seeds (rejected: too strict; M3 / M4 calibration shows ~25% per-seed variance is normal).
- ≥1 of 4 seeds (rejected: too lax; one outlier seed could tip the verdict).

### D7. Predator brain bootstrapping: pilot ablation (heuristic-imitation pretrain on one seed, cold-start on the other)

**Decision:** Pilot pop runs both bootstrap arms — pilot seed 42 uses 50-episode behavioural-cloning pretrain against `HeuristicPredatorBrain`, pilot seed 43 starts cold (random-init MLPPPO weights). Pilot result chooses pretrain on/off for the full run.

**Rationale:** Cold-start gen-0 risks zero fitness gradient — untrained MLPPPO predators will lose every episode against trained-prey baselines. But heuristic-imitation pretraining biases initial behaviour toward heuristic dynamics, which could constrain the policy space the predator can later explore. Costs nothing extra to run both arms in pilot (2 seeds either way) and gives the empirical answer for the full run.

**Alternatives:**

- Pretrain only (rejected: foreclose the cold-start question without evidence).
- Cold-start only (rejected: zero-gradient risk).
- Random-prey-trained MLPPPO bootstrap (rejected: more code; reserved as future work).

**Confirmed by user (Phase 3 — both arms).**

### D8. Predator I/O encoding contract

**Decision:** `MLPPPOPredatorBrain` reads `PredatorBrainParams` and emits a flat float vector with these components, in this order:

- `predator_position[0] / grid_size`, `predator_position[1] / grid_size` (2 floats)
- For each of `agent_positions[:k_nearest=2]`: `(x / grid_size, y / grid_size, present_flag)` where `present_flag ∈ {0, 1}` (3 floats × 2 = 6 floats; padded with zeros if fewer than k_nearest agents alive).
- `detection_radius / grid_size`, `damage_radius / grid_size` (2 floats).
- `step_index / max_steps` (1 float).
- Total input dim: **11 floats**.

Output: 5-way categorical over `PredatorAction.{STAY, UP, DOWN, LEFT, RIGHT}` (matches the Protocol-defined action space from M1).

**MLP architecture:** the network mirrors the agent-side MLPPPO via the existing `DEFAULT_ACTOR_HIDDEN_DIM`, `DEFAULT_CRITIC_HIDDEN_DIM`, and `DEFAULT_NUM_HIDDEN_LAYERS` constants in `quantumnematode.brain.arch.mlpppo` (currently 64 / 64 / 2; spec stays correct even if defaults change). Pilot YAML may override these via the `extra` block on `PredatorBrainConfig` if a smaller predator network is desired post-pilot.

**Rationale:** Fixed input dim is required for MLP. k_nearest=2 covers multi-prey observation cleanly without exploding input dim; predators in pilot scenarios face 5-prey populations, so the nearest 2 are sufficient observation surface for greedy chase. Normalised positions / radii / step keep the network input range bounded for stable training.

**Alternatives:**

- k_nearest=1 (rejected: too narrow when prey cluster).
- k_nearest=5 (rejected: input dim 17, larger network, harder to train at small genome budget).
- Raw agent_positions list (rejected: variable-length; not MLP-compatible).
- Egocentric (delta-from-predator) coords (deferred: cleaner but breaks fixed-frame normalisation; explore in follow-up).

### D9. Hall-of-fame Protocol surface

**Decision:** `HallOfFame` lives in `evolution/hall_of_fame.py` (NOT in `evolution/coevolution.py`) because the eviction policy + sampling semantics are general-purpose primitives that future single-population novelty-search experiments could reuse. Public surface:

```python
class HallOfFame:
    def __init__(self, capacity: int, *, replacement: Literal["quality", "fifo"] = "quality") -> None
    def push(self, genome: Genome, fitness: float) -> None
    def sample(self, rng: np.random.Generator, k: int) -> list[Genome]
    def mix_with_pop(self, rng, pop: list[Genome], frac_hof: float) -> list[Genome]
    def __len__(self) -> int
    def to_dict(self) -> dict  # for checkpointing
    @classmethod
    def from_dict(cls, d: dict) -> HallOfFame  # for resume
```

`replacement="quality"` is the default. `replacement="fifo"` is exposed for ablation testing.

**Rationale:** Two-population co-evolution uses one HoF per side. Single-population novelty-search would use one HoF total. Same primitive; different harness.

**Alternatives:**

- HoF tied to `CoevolutionLoop` (rejected: foreclosed reuse).
- HoF as a Protocol (rejected: only one implementation foreseen; concrete class with a `replacement` enum is simpler).

### D10. CoevolutionLoop architecture: composition over inheritance

**Decision:** `CoevolutionLoop` is a new orchestrator class that *composes* two side-state objects (one per population). It does NOT subclass `EvolutionLoop`. Each side carries:

- `encoder: GenomeEncoder` (MLPPPO for prey, MLPPPOPredatorEncoder for predator)
- `fitness: FitnessFunction` (LearnedPerformanceFitness for prey, PredatorEpisodicKillRate for predator); both implement the existing `evaluate(genome, sim_config, encoder, *, episodes, seed) -> float` Protocol surface
- `optimizer: CMAESOptimizer(diagonal=True)` from `quantumnematode.optimizers.evolutionary` (re-constructed fresh at each K-block start per D2)
- `inheritance: InheritanceStrategy` (Lamarckian for prey, NoInheritance for predator gen-0; configurable)
- `hof: HallOfFame` — bounded buffer with eviction policy; the runtime opposition-sampling pool. Loses entries when capacity is exceeded.
- `population: list[Genome]` (current generation)
- `champion_history: list[Genome]` — **unbounded** time-ordered audit log; one entry per completed K-block (the K-block's top-fitness genome at K-block end). Never evicted. Distinct from `hof` (which is the bounded sample-from set) and from per-generation lineage rows (which include all genomes per gen, not just K-block elites). Aggregator-time analysis (cycling, escalation) walks `champion_history`.

A K-block's elite genome is appended to **both** `hof` (subject to eviction) and `champion_history` (unbounded log) — they share content but serve different purposes: HoF is for runtime sampling, champion_history is for post-hoc analysis.

`CoevolutionLoop.run(*, generation_pairs, K_per_block, generality_probe_every)` drives:

1. Smoke gen 0: evaluate both sides against random-init opponents to populate gen-0 lineage.
2. For each K-block in alternating order (prey first, then predator):
   - Re-construct training side's `CMAESOptimizer(diagonal=True)` with a fresh seed (per D2).
   - For K generations: ask training-side optimizer for trial params → evaluate against frozen-side population (HoF-mixed per D3) → tell.
   - Append training-side block elite to `champion_history` AND push it to its HoF (the K-block's elite genome lands in both — HoF is bounded with eviction; champion_history is unbounded log).
3. Every `generality_probe_every` generations, evaluate both elites against the held-out set and write a probe row.
4. Checkpoint to JSON every K-block (resume support).

**Inheritance bookkeeping inside `CoevolutionLoop`:** because we use `_evaluate_in_worker` directly (not `EvolutionLoop.run`), `CoevolutionLoop` must replicate per-side inheritance bookkeeping (parent_ids, warm_start_path_override, weight_capture_path) as it walks K-blocks, mirroring `EvolutionLoop.run`'s logic at [loop.py:498-700+](packages/quantum-nematode/quantumnematode/evolution/loop.py#L498). This is implementation overhead (~100 LoC) but avoids the alternative: forcing single-population checkpoint shape via `EvolutionLoop` subclassing.

**Worker reuse:** `EvolutionLoop._evaluate_in_worker` ([loop.py:104-162](packages/quantum-nematode/quantumnematode/evolution/loop.py#L104-L162)) takes an 11-tuple `(params, sim_config, encoder, fitness, episodes, seed, generation, index, parent_ids, warm_start_path_override, weight_capture_path)`. The tuple ABI does NOT change for co-evolution — instead, the **opponent brain weights are injected via `sim_config` patching** before the worker is invoked, following the M2 idiom for sim-config patching at evaluation time. The fitness function then drives the multi-agent runner internally with both brains decoded and reads `MultiAgentEpisodeResult.per_predator_kills` (or per-agent metrics for prey).

**Rationale:** Inheritance from `EvolutionLoop` would force its single-population checkpoint shape and its single-side-evaluator design; composition keeps the existing `_evaluate_in_worker` 11-tuple worker reusable while adding the alternating-schedule controller as a thin layer. Each side's lineage CSV stays in the same shape as M3's lineage.csv, so existing analysis scripts work.

**Alternatives:**

- Subclass `EvolutionLoop` (rejected: forces single-population shape; would require deep changes to `EvolutionLoop`).
- Two parallel `EvolutionLoop` instances + a thin coordinator (rejected: each `EvolutionLoop` owns its own multiprocessing pool; we want a single pool with the alternating-schedule controller orchestrating workers).

### D11. M5.7 Baldwin instrumentation (reduced): reuse M4.5 F1 evaluator, signal-delta only

**Decision:** `scripts/campaigns/baldwin_m5_secondary_eval.py` invokes `baldwin_f1_postpilot_eval.py` (319 LoC, currently used for M4.5 F1 readout) with M5-specific inputs: per-gen elite snapshots from prey lineage CSV, current-gen predator pop as the "task" axis, K′ ∈ {10, 25} paired-train comparison. Observation only; **single readout: signal-delta > +0.05 at K′=10 across ≥2 of 4 seeds**. If it fires, M5.7 *suggests* a Baldwin signal but cannot be definitive without a hyperparameter axis; it arms M4.7's deferred-retry trigger but does NOT close the Baldwin question.

**Why reduced (vs original two-readout tracker entry):** the original tracker M5.7 (per [openspec/changes/2026-04-26-phase5-tracking/tasks.md:215](openspec/changes/2026-04-26-phase5-tracking/tasks.md#L215)) specified a conjunctive readout: signal-delta AND hyperparam spread tightening. The hyperparam-spread criterion required prey to evolve hyperparameters (e.g. `actor_lr` / `entropy_coef`); under D2's CMA-ES weight evolution, prey have a single fixed hyperparameter set across all genomes, so per-generation hyperparam spread is undefined. The criterion is structurally inapplicable. Dropping it is honest about what M5's substrate can measure.

**What remains observable under weight evolution:** condition 1 (signal-delta) measures whether the elite genome's brain has learned more than a fresh schema-prior brain at K′=10 episodes. Under weight evolution + Lamarckian inheritance, this *is* a meaningful Baldwin proxy: if elite's weight inheritance gives it a head start on K′-train compared to schema-prior, that's evidence the evolved weight init encodes faster learning. Less direct than hyperparameter-axis evidence but not zero.

**Rationale (unchanged):** reuses M4.5 substrate (`baldwin_f1_postpilot_eval.py`). Marginal cost ~1 day of script glue + aggregator integration. Defer-to-M4.7 alternative would require re-running M5 to capture per-gen elite snapshots, a much larger cost. Runs regardless of M5 verdict (per task 11.4) — informative even on STOP.

**M4.7 trigger condition (revised under reduced M5.7):** if M5.7's signal-delta fires across ≥2 of 4 seeds, M4.7 stays armed and motivated by *suggestive* M5 evidence; if it stays silent, M4.7 stays armed but without M5 motivation. M4.7 itself would run with proper hyperparameter-evolution substrate (M2.12 / M3 stack) where both signal-delta AND hyperparam-spread are observable, and would be the definitive Baldwin closure.

**Confirmed by user (Phase 3 — ship in M5; second-pass review — accept reduced readout under CMA-ES weight evolution).**

## Risks / Trade-offs

\[**Cycling without progress**\] → Mitigation: HoF (D3) + generality probe (D5). Disjunctive gate accepts cycling alone but probe surfaces if cycling co-occurs with held-out skill loss (overfitting) — flagged in verdict.

\[**One side dominates → other's gradient saturates**\] → Mitigation: alternating schedule rebalances; rebalancing knob — if a side's mean fitness drops below pilot-calibrated threshold for ≥3 consecutive K-blocks, automatically freeze the dominant side for an extra K-block. Knob lives in `CoevolutionLoop`.

\[**Compute blowout**\] → Mitigation: pilot-first sequencing; pilot wall is ~8h, full is ~30–40h. Stretch fallback: drop pop sizes to 16/12 if pilot compute headroom is tight.

\[**Bootstrap mismatch — predator pretraining biases initial behaviour**\] → Mitigation: pilot ablation (D7); empirical answer chooses full-run config.

\[**M5.7 readout noise**\] → Mitigation: observation only (D11); does NOT alter M5's GO/STOP. Aggregator emits readout in `summary.md` with the explicit note "primary observation, not a verdict input."

\[**CMA-ES covariance pollution across K-blocks**\] → Mitigation: per-block fresh-instance construction (D2). Risk if cold-restart exploration noise dominates — fall back to "warm" CMA-ES state across K-blocks (carry covariance forward) and document.

\[**Predator within-K-block PPO instability**\] → Pilot sanity check; if predator fails to train within K=10, raise K to 15 in full run. The K=10 default is calibrated for prey (M3 evidence) — predator may need different K.

\[**Generality-probe held-out set saturated**\] → Pilot inspects the held-out fitness range; if all probes saturated at 0.0 or 1.0, re-sample held-out from a wider pool (mix in random-policy or heuristic-radius variants).

\[**Predator I/O encoding choice (D8) limits learned policy**\] → k_nearest=2 may not generalise to higher-prey-density scenarios. Reserved for follow-up: egocentric coords + larger k_nearest as ablation knob if predator policy plateaus early.

## Migration Plan

No data migration. Behaviour change is opt-in via `PredatorBrainConfig.kind: "mlpppo_predator"`. Existing scenario YAMLs default to `"heuristic"` and continue to instantiate `HeuristicPredatorBrain`.

**Deployment steps:**

1. Land predator brain + dispatcher (PR 1).
2. Land predator encoder + fitness (PR 2).
3. Land HoF + Red Queen metrics + CoevolutionLoop (PR 3).
4. Land configs + driver + aggregator (PR 4).
5. Run pilot, land logbook section (PR 5).
6. Run full, land verdict logbook + tracker flips (PR 6).
7. Land M5.7 Baldwin readout + observation logbook (PR 7).
8. (After M5 verdict GO) Sync OpenSpec deltas to main specs and archive change.

**Rollback:** Each PR is independently revertable. The predator-brain dispatcher's `mlpppo_predator` branch can be removed without affecting heuristic-brain behaviour. The `CoevolutionLoop` is gated behind its own driver script — no impact on the existing single-population `EvolutionLoop`.

## Open Questions

1. **Should the rebalancing knob (D2 risk row) ship in M5 or be deferred?** It's optional code — `CoevolutionLoop` works without it as long as no side collapses. Recommend ship-as-disabled-default with a feature flag, enable in pilot if domination is observed. *(Resolved as Phase-3 follow-up — design decision punted to pilot.)*

2. **Held-out predator opponent pool composition for generality probe (D5).** Mix of detection_radius ∈ {4, 6, 8, 10} × damage_radius ∈ {0, 1} gives 8 combinations; or sample from a wider distribution? Recommend the {4,6,8,10} × {0,1} grid for reproducibility; pilot inspects probe range.

3. **Pilot threshold calibration cadence.** Lock thresholds *before* full-run starts, or refine after seeing more data? Recommend lock-before — prevents post-hoc selection bias; thresholds documented in pilot logbook section.

4. **Predator inheritance strategy (D10).** Default to `NoInheritance` for predator gen-0 (cleaner — predator side starts fresh each generation). Lamarckian for predator is an ablation that could surface differential learning advantages between sides. Reserved for follow-up.
