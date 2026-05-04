# 015: Baldwin Effect — iterative evaluation (M4.5 + follow-ups)

**Status**: `CLOSED — STOP after iteration step 2 (M4.6)`. The Baldwin Effect question is closed for Phase 5 after three iterations (M4 → M4.5 → M4.6). M4.6 ran a pre-flight smoke that ruled out the candidate selection-feedback abstractions (B3, B6, truncated-K) and identified the **substrate constraint** as the real blocker: a single fixed task with K=50 PPO training cannot demonstrate the Baldwin Effect because the substrate has no Baldwin axis (the optimal strategy on a single task is "innate good behaviour for THIS task," which is the opposite of what Baldwin selects for). The published mechanism (Fernando 2018, Chiu 2024) requires task distribution — a ~7-day infrastructure investment (~5 days impl + ~9-14h pilot wall + ~1 day overhead) Phase 5 deferred in favour of M5 (co-evolution), which intrinsically introduces task variation via co-evolving predators and may demonstrate Baldwin as a side-effect.

**Branch**: `feat/m4-6-baldwin-selection-feedback` (M4.6 STOP PR; M4.5 already merged)

**Date Started**: 2026-05-03

**Date Last Updated**: 2026-05-04 — M4.6 STOP verdict committed; logbook closed; Baldwin question deferred to potential M4.7 post-M5 if M5's co-evolution doesn't surface a Baldwin signal serendipitously.

This logbook covers the **iterative refinement of the Baldwin Effect evaluation** in this framework. M4 (logbook 014) shipped INCONCLUSIVE because three audit blockers prevented a clean test. M4.5 fixed all five audit findings and ran cleanly — and that clean run revealed a different problem: the framework's `BaldwinInheritance` abstraction records lineage but doesn't feed it back into selection. M4.6 explored selection-feedback abstractions, ran a pre-flight smoke, and discovered the deeper issue is substrate-level, not abstraction-level. The Baldwin question is closed in Phase 5 with a STOP verdict that's defensible by reference to published methodology.

## Objective

Iteratively refine the Baldwin Effect evaluation in this framework until we have a valid implementation that can produce a definitive GO/PIVOT/STOP verdict. Each iteration:

- Identifies any remaining design flaws (M4 → M4.5 → M4.6 → ...)
- Implements the smallest change needed to address them
- Re-runs the 4-arm pilot under the same comparison harness
- Reports findings honestly (audit closures + new findings + structural observations)

The iteration terminates when a clean evaluation produces a verdict that's defensible against all known design flaws. Each iteration's findings get recorded as a new section in this logbook (one logbook can span multiple related iterations of the same scientific question, per the established pattern from logbooks 002-008 etc.).

## Background

- **M4 (logbook 014)**: Baldwin pilot under TPE with 6-field schema for Baldwin and 4-field schema for Control. Shipped INCONCLUSIVE after a post-pilot audit found three blocking design flaws (A1 schema-shift, A2 F1 biologically incoherent, A3 F1 apples-to-oranges) plus two significant issues (A4 n=4 underpowered, A5 chosen knobs may not be optimal for K=50).
- **M4.5 (this logbook, step 1)**: Fix all five audit findings. Equalised 8-field schemas, F1 redesigned as paired K'-train learning-acceleration test, n=8 seeds, gate thresholds recalibrated. Implementation under OpenSpec change `add-baldwin-retry` (archived in this PR).
- **M4.6 (next PR, step 2)**: Implement a Baldwin abstraction where selection feedback uses lineage. Reuses M4.5's F1 evaluator + 4-way aggregator + 8-field configs.

## Iteration step 1 — M4.5 (LSTMPPO + klinotaxis + pursuit predators, TPE)

### Hypothesis

Under a properly designed evaluation (audit findings A1-A5 all addressed), Baldwin inheritance will accelerate convergence over the from-scratch Control by ≥2 generations on the predator arm — the speed-gate threshold from add-baldwin-retry design Decision 5. If the speed gate passes plus the F1 learning-acceleration gate (mean elite − mean baseline > 0.05 at K' = 10) and the comparative gate (Baldwin within 4 gens of Lamarckian), the Baldwin Effect is demonstrated on this testbed.

### Method

#### Architecture

Same as M4: LSTMPPO brain (GRU + 64-dim hidden + 2-layer actor + 2-layer critic) on the predator arm (klinotaxis sensing, 2 pursuit predators, max_steps=1000, body_length=2). Reward + satiety + environment configs unchanged from M4.

#### Evolutionary configuration

- **Optimizer**: TPE (Optuna's `TPESampler`)
- **Population size**: 12
- **Generations**: 20 (with `early_stop_on_saturation: 5` enabled)
- **K-train budget**: `learn_episodes_per_eval: 50`
- **L-eval budget**: `eval_episodes_per_eval: 25`
- **Parallel workers**: 4
- **Seeds**: 42, 43, 44, 45, 46, 47, 48, 49 (n=8)

#### Evolved hyperparameter schema (8 fields, identical across Baldwin and Control)

| # | Field | Type | Bounds | Source |
|---|---|---|---|---|
| 1 | `actor_lr` | float (log) | [1e-5, 1e-3] | M4 control |
| 2 | `critic_lr` | float (log) | [1e-5, 1e-3] | M4 control |
| 3 | `gamma` | float | [0.9, 0.999] | M4 control |
| 4 | `entropy_coef` | float (log) | [1e-4, 1e-1] | M4 control |
| 5 | `weight_init_scale` | float | [0.5, 2.0] | M4 Baldwin |
| 6 | `entropy_decay_episodes` | int | [200, 2000] | M4 Baldwin |
| 7 | `actor_hidden_dim` | int | [64, 256] | NEW (audit A5 hypothesis) |
| 8 | `actor_num_layers` | int | [1, 3] | NEW (audit A5 hypothesis) |

The 8-field "head-to-head" schema lets a single pilot answer audit A5 ("were M4's knobs the right choice?") in either direction: TPE's posterior across n=8 seeds reveals which fields it preferred to explore vs pin at defaults.

#### Arms

| Arm | Schema | `inheritance` | Purpose |
|---|---|---|---|
| Baldwin | 8-field | `baldwin` | Treatment arm |
| Control | 8-field (identical to Baldwin) | `none` | Audit A1 closure — same TPE prior at gen-0 across arms |
| Lamarckian rerun | 4-field (M3's, no arch knobs) | `lamarckian` | Comparative-gate baseline at n=8 + reproducibility check on n=4 subset |
| Hand-tuned baseline | n/a (re-uses M2.11's artefacts) | n/a (no evolution) | Convergence-plot context (n=4 only — annotated on plot) |

#### F1 evaluator (redesigned, addresses audit A2 + A3)

Per pilot seed:

1. Reconstruct the elite genome from `best_params.json` via `HyperparameterEncoder.decode`
2. Construct a schema-prior baseline genome via `HyperparameterEncoder.initial_genome(sim_config, rng=np.random.default_rng(seed))` (deterministic per-seed sample from the schema's prior distribution)
3. Build a `sim_config` copy with `learn_episodes_per_eval = K'` and `eval_episodes_per_eval = L`
4. Run BOTH genomes through `LearnedPerformanceFitness.evaluate(genome, sim_config_copy, encoder, episodes=L, seed=seed)` — same per-seed seed → identical env trajectory; only the genome differs (apples-to-apples)
5. Append a row to `f1_learning_acceleration.csv` with columns `seed, k_prime, episodes, elite_success_rate, baseline_success_rate, signal_delta`

Both K' = 10 (default per design Decision 3) and K' = 25 (Risk 4 fallback) measured for sensitivity analysis.

#### Decision gates (recalibrated per design Decision 5)

- **Schema-equalisation pre-flight (audit A1 closure)**: `|Baldwin gen-0 mean − Control gen-0 mean| ≤ 0.05` — if `>0.05`, force verdict to INCONCLUSIVE without consulting the gates.
- **Speed gate** (Baldwin vs Control): `mean_gen_baldwin_to_092 + 2 ≤ mean_gen_control_to_092`
- **F1 learning-acceleration gate**: `mean_elite_success_rate − mean_baseline_success_rate > 0.05` (at K' = 10)
- **Comparative gate** (Baldwin vs Lamarckian): `mean_gen_baldwin_to_092 ≤ mean_gen_lamarckian_to_092 + 4`

GO if all three; PIVOT if speed only; STOP otherwise.

### Results

#### Schema-equalisation pre-flight (audit A1)

| Arm | First-gen mean best_fitness |
|---|---|
| Baldwin | 0.7150 |
| Control | 0.7150 |
| **Abs delta** | **0.0000** (tolerance: 0.05) |
| **Status** | **PASS** |

**Audit A1 closed perfectly.** Identical 8-field schema + identical TPE seed → byte-identical gen-0 starting populations across arms. M4's measured `|Δ| = 0.14` (with mismatched schemas) is now `|Δ| = 0.0000` (with matched schemas).

#### Aggregator literal verdict (default K' = 10 per design)

```text
Speed gate (Baldwin vs Control + 2):           FAIL  margin +0.00
  Baldwin mean gen-to-0.92:    6.38
  Control mean gen-to-0.92:    6.38

F1 learning-acceleration gate (K' = 10):       FAIL  signal_delta = +0.000
  Baldwin elite mean (L=25):              0.000
  Schema-prior baseline mean (L=25):      0.000

Comparative gate (Baldwin vs Lamarckian + 4):  PASS  margin +0.38
  Baldwin mean gen-to-0.92:    6.38
  Lamarckian mean gen-to-0.92: 2.75

Aggregator decision: STOP ❌
```

The aggregator emits STOP per its 3-gate logic. The user-facing verdict is reframed below per the structural finding.

#### Per-seed convergence speed (gen first reaches `best_fitness >= 0.92`)

| Seed | Baldwin | Lamarckian | Control |
|---|---|---|---|
| 42 | 2 | 2 | 2 |
| 43 | 4 | 3 | 4 |
| 44 | 3 | 3 | 3 |
| 45 | 9 | 6 | 9 |
| 46 | 9 | 2 | 9 |
| 47 | 13 | 2 | 13 |
| 48 | 8 | 2 | 8 |
| 49 | 3 | 2 | 3 |
| **mean** | **6.38** | **2.75** | **6.38** |

**Baldwin and Control are bit-identical for every seed.** Same gen-to-0.92 across all 8 seeds.

#### F1 K' sensitivity

K' = 10 (gate default): all seeds elite=0, baseline=0 — too small a budget for either arm to score on this task.

K' = 25 (Risk 4 fallback):

| Seed | F1 elite | F1 baseline | F1 signal |
|---|---|---|---|
| 42 | 0.280 | 0.000 | +0.280 |
| 43 | 0.000 | 0.000 | +0.000 |
| 44 | 0.000 | 0.000 | +0.000 |
| 45 | 0.000 | 0.000 | +0.000 |
| 46 | 0.520 | 0.000 | +0.520 |
| 47 | 0.040 | 0.000 | +0.040 |
| 48 | 0.040 | 0.000 | +0.040 |
| 49 | 0.000 | 0.000 | +0.000 |
| **mean** | **0.110** | **0.000** | **+0.110** |

At K' = 25 the elite shows real signal (mean +0.11; two seeds reach 0.28 and 0.52). But this is the *same* elite as the Control arm's elite (per the bit-identity finding below) — so the F1 signal is "evolved 8-field elite vs schema-prior baseline genome", NOT a Baldwin-vs-Control comparison.

#### Lamarckian rerun (n=8) — confirms framework integrity

All 8 Lamarckian seeds reach `best_fitness = 1.00`. Per-seed gen-to-0.92: `[2, 3, 3, 6, 2, 2, 2, 2]`, mean **2.75**. Pop-mean trajectory dramatically higher than Baldwin/Control (gen-2 mean 0.696 vs 0.349). The new 4 seeds (46-49) all reach 1.00 by gen 2.

The n=4 subset (seeds 42-45) gives gen-to-0.92 `[2, 3, 3, 6]`, mean 3.5 — broadly consistent with M3's published `[3, 4, 4, 7]` mean 4.5 (some seed-level variance from the M3 → M4 → M4.5 code revisions but the framework path is sound).

#### Forensic: Baldwin elite hyperparameters (audit A5 partial answer)

| Seed | actor_lr | critic_lr | gamma | ent_coef | weight_init_scale | entropy_decay_episodes | actor_hidden_dim | actor_num_layers |
|---|---|---|---|---|---|---|---|---|
| 42 | 7.8e-04 | 2.7e-04 | 0.908 | 1.2e-02 | 1.465 | 579 | 251 | 2 |
| 43 | 4.9e-04 | 1.5e-04 | 0.913 | 1.6e-03 | 1.365 | 1359 | 244 | 2 |
| 44 | 3.4e-04 | 4.1e-04 | 0.922 | 4.5e-04 | 1.180 | 657 | 198 | 3 |
| 45 | 8.0e-04 | 2.9e-04 | 0.926 | 1.4e-03 | 1.509 | 942 | 192 | 1 |
| 46 | 5.9e-04 | 1.0e-05 | 0.909 | 3.0e-03 | 1.237 | 1655 | 216 | 2 |
| 47 | 6.8e-04 | 2.9e-04 | 0.937 | 3.5e-04 | 1.016 | 1548 | 190 | 2 |
| 48 | 4.2e-04 | 7.0e-05 | 0.909 | 6.7e-04 | 0.795 | 705 | 231 | 2 |
| 49 | 9.1e-04 | 4.3e-04 | 0.921 | 6.3e-03 | 0.751 | 453 | 149 | 3 |

**TPE explored ALL 8 fields broadly; no fields pinned at defaults.** Notable patterns:

- `actor_hidden_dim` (default 64): TPE chose [149, 251] — strongly prefers wider actors than the default
- `entropy_decay_episodes` (default 500): consistently HIGHER (453-1655 across all 8 seeds) — confirms M4's finding that TPE wants slower entropy decay
- `weight_init_scale` (default 1.0): chose [0.751, 1.509] — full schema range explored
- `actor_num_layers` (default 2): mostly 2, with 1 and 3 each appearing once
- All 4 hyperparam knobs (actor_lr, critic_lr, gamma, entropy_coef) explored within bounds

**Audit A5 partially answered**: arch knobs ARE actively explored, NOT pinned at defaults. M4's hypothesis (arch knobs may have larger effects within K=50) is supported by the fact that TPE pushes `actor_hidden_dim` well above the brain's default 64. But this exploration happens identically in Baldwin and Control — so the hyperparam choices are independent of the Baldwin question.

#### Wall-time

| Phase | Wall |
|---|---|
| Pilot launch → all 3 arms done | ~4h 6min (parallel; each arm individually 3.5-4h) |
| F1 evaluator (K'=10 + K'=25) | ~2 min |
| Aggregator | \<1 min |
| **Total** | **~4h 6min** |

Longer than the smoke's 3h projection because early-stop fired on only 1/8 Baldwin seeds (seed-44 at gen 14). 6/8 Baldwin seeds and 7/8 Control seeds ran the full 20-gen budget.

### Analysis

#### CRITICAL OBSERVATION — Baldwin and Control are bit-identical

Per-seed best fitness at every generation, per-seed elite hyperparameters, and per-seed convergence speed are all **byte-identical** between the Baldwin and Control arms across all 8 seeds. The only observable difference between the two arms' artefacts is that Baldwin's `lineage.csv` has non-empty `inherited_from` cells from gen 1 onwards, while Control's `inherited_from` is empty for all rows.

**Why this happens** (proven structurally):

1. Identical 8-field schema (verified at YAML load: same names, types, bounds, log_scale flags, in same order)
2. Identical TPE seed (`--seed 42` for both Baldwin and Control invocations)
3. TPE's `ask()` is a pure function of seed + history → both arms get the same parameter vectors at every generation
4. Fitness function is deterministic in `(seed, genome)` — per-episode seed via `derive_run_seed(seed, ep_idx)` is a pure function (verified at [seeding.py](../../packages/quantum-nematode/quantumnematode/utils/seeding.py)) — so identical genome → identical fitness
5. `inherited_from` is metadata only: the loop writes it to `lineage.csv` but never feeds it back to `optimizer.tell()` (see [loop.py](../../packages/quantum-nematode/quantumnematode/evolution/loop.py)) or to any child's brain construction (see [encoders.py](../../packages/quantum-nematode/quantumnematode/evolution/encoders.py))

⇒ TPE's posterior is identical for both arms ⇒ same `(genome, fitness)` pairs ⇒ same elite genome each generation ⇒ same trajectory.

#### What this finding actually means

The Baldwin Effect — in evolutionary biology — relies on lifetime learning shaping which genomes get selected for the next generation. The biological mechanism requires three ingredients:

1. **Phenotypic plasticity within a generation**: each individual learns during its lifetime ✅ (K=50 PPO training)
2. **Heritable genome that's NOT directly modified by learning** ✅ (hyperparameter genome; learned weights die with the individual)
3. **Selection pressure on outcomes**: individuals that learn faster reproduce more ❌

In this framework's current Baldwin abstraction, **ingredient 3 is missing**. Selection happens via the optimizer (TPE), which operates on `(genome, fitness)` pairs and ignores the `inherited_from` lineage trace. There's no mechanism by which "child of a high-performing elite" gets a selection advantage.

The framework's current `BaldwinInheritance` is more accurately described as "lineage trace bookkeeping": it labels which children share hyperparameters with the prior elite (as identified in the lineage CSV), but doesn't actually bias future sampling toward elite descendants. The metaphor breaks down — there's no "inheritance" in the evolutionary sense; just a label.

#### Why audit A5 partial signal doesn't rescue Baldwin's verdict

TPE did discover that wider actors (hidden_dim 149-251) outperform the default 64. That's a real finding about the testbed, and it validates audit A5's hypothesis (arch knobs matter under K=50). But this discovery is independent of the Baldwin mechanism — the same discovery happens in Control with identical convergence trajectory.

The 8-field schema choice (option b head-to-head from design Decision 1) was the right call for answering audit A5. It just turned out that audit A5 was orthogonal to the Baldwin question; the Baldwin question's blocker is at a different layer of the abstraction.

### Iteration-step-1 verdict

**Aggregator literal output**: STOP (per its 3-gate logic).

**Reframed verdict**: **iteration step 1 finding — the framework's *current* Baldwin abstraction is mechanically null vs Control under matched conditions.** This isn't a measurement flaw; the schema-equalisation closed audit A1 perfectly. It's a property of the abstraction: lineage is metadata, not selection feedback.

A final STOP from this iteration alone would be too strong a claim. We can't conclude "the Baldwin Effect doesn't exhibit on this testbed" — we can only conclude "this abstraction can't test for it." A different abstraction with selection feedback could either confirm or refute the Baldwin Effect; M4.5 doesn't speak to that question.

**Decision 6 reinterpretation**: the pre-registered STOP semantic was scoped to "Baldwin's *redesigned gates* STOP after fixing the audit findings". M4.5 closed all five audit findings; what we found is that the framework *abstraction itself* is the wrong substrate for the test. Decision 6 doesn't preclude iterating on a different abstraction.

## Iteration step 2 — M4.6 (LSTMPPO + klinotaxis + pursuit predators, TPE)

### Hypothesis

A Baldwin abstraction with explicit selection feedback (i.e. the optimiser's `tell()` receives a different scalar in Baldwin vs Control rather than just a different `inherited_from` metadata column) will break M4.5's bit-identity result and produce a measurable Baldwin signal at n=8 seeds on the same predator-arm testbed.

### Method (planning + smoke phase only — no full pilot ran)

Three abstraction candidates were enumerated (informed by Fernando 2018 + Chiu 2024 literature search and a structured options-comparison sub-agent pass):

- **B3 — Learning-gain composite fitness**: Baldwin's effective fitness = `fit_K=50 + α · (fit_K=50 − fit_K=25)` with α ≈ 0.5. Selects on "improved during lifetime" not just "ended high." Direct continuous analogue of the Hinton-Nowlan 1987 mechanism.
- **B6 — Cost-of-learning fitness penalty**: `effective_fit = final_fit − α · episodes_to_threshold`. Selects on "needed fewer episodes to reach success." Same Baldwin spirit, less noisy on small K.
- **Truncated-K hybrid**: Baldwin selects on `fit_K=25` while Control selects on `fit_K=50`. Cleanest single-difference design — no composite weighting decision, no intermediate-checkpoint surgery.

### Pre-flight smoke (gen-0 population at seed 42, K=10/25/50)

Before committing to a 5-day implementation + ~9-14h pilot wall-time investment, ran a 3-minute smoke (`tmp/m4_6_b3_smoke.py`) measuring per-genome (fit_K=10, fit_K=25, fit_K=50) triples for the 12-genome gen-0 population deterministically reproduced from the M4.5 Baldwin pilot's seed 42. The smoke uses fresh-brain measurements at each K (cross-brain Δ rather than within-trajectory Δ), consistent with the existing F1 evaluator's pattern.

| genome_idx | fit_K=10 | fit_K=25 | fit_K=50 | Δ_25→50 | Δ_10→25 |
|---|---|---|---|---|---|
| 0-5, 7-9 | 0.000 | 0.000 | 0.000 | +0.000 | +0.000 |
| 6 | 0.000 | 0.000 | 0.080 | +0.080 | +0.000 |
| 10 | 0.000 | 0.000 | 0.440 | +0.440 | +0.000 |
| 11 | 0.000 | 0.080 | 0.840 | +0.760 | +0.080 |

**Decision-rule statistics:**

- Spearman(fit_K=50, Δ_25→50): **r = 1.000** → B3 is provably rank-preserving on this gen-0 population. Composite reorders nothing relative to raw fit_K=50; TPE's quantile splits identical → bit-identity at α-any.
- Non-zero genomes by K: K=10: **0/12**; K=25: **1/12**; K=50: **3/12**. Truncated-K hybrid loses TPE bootstrap (11/12 genomes at fitness 0 at K=25 → no posterior signal). B6's "episodes-to-first-non-zero" distribution: only 3 distinct values across 12 genomes (25, 50, 999) → poor selection gradient.
- Mean fitness: K=10: 0.000; K=25: 0.007; K=50: 0.113. **Random gen-0 hyperparameter genomes mostly produce un-trainable PPO configurations** — only 3/12 show any post-K=50 signal.

### Analysis — why all three abstractions failed at the smoke stage

The smoke didn't refute B3/B6/truncated-K via implementation flaws; it refuted them via the **substrate's fitness landscape**. Three findings:

1. **Δ_learn collapses to genome quality on a single task.** With 11/12 genomes at fitness 0 across K=10 → K=25, the "learning gain" signal Δ_25→50 is mathematically `fit_K=50 - 0 = fit_K=50` for those genomes. Spearman r=1.000 between Δ and raw fitness is structural, not coincidental. Any abstraction selecting on "learning rate" in this regime selects on genome quality in disguise.
2. **TPE bootstrap requires non-zero fitness variance.** 1/12 non-zero at K=25 is too sparse for TPE's KDE to update meaningfully. Truncated-K and B6 both fail here — the selection gradient collapses.
3. **The substrate has no Baldwin axis.** A single fixed task's optimal strategy is "innate good behaviour for THIS task" — exactly the opposite of what Baldwin selects for. Plasticity is a cost, not a benefit, in a single deterministic environment. Fernando 2018 and Chiu 2024 (the only published successful Baldwin demonstrations on hyperparameter/initial-weight evolution) both use **task distributions**, not single tasks. Our setup omits the literature's required ingredient.

### What it would take to fix the substrate

Fernando/Chiu requirements vs our framework status:

| Pre-requisite | Fernando/Chiu | Our framework | Cost to add |
|---|---|---|---|
| Variable environment / task distribution | Required | ❌ Single task | ~3-4 days (multi-task aggregation wrapper + env-variant configs + validator + tests) |
| Genome encodes innate behaviour | Initial weights + hyperparams | ⚠️ Just `weight_init_scale` (1 scalar magnitude knob) | ~1-5 days depending on depth (per-layer init scales → 3D Baldwin axis is ~2 days; full initial-weight evolution would re-open RQ1's TPE-vs-CMA-ES decision) |
| Lifetime learning that uses innate bias | PPO/SGD | ✅ Have it | 0 |
| Selection on outcome of learning across tasks | Mean fitness over distribution | ❌ Single-task fitness | (Same as #1) |
| Baldwin vs Lamarckian arm comparability | Same task distribution per arm | ✅ Existing strategies fine | 0 |

Total Path A-minimal investment: ~5 days impl + ~9-14h pilot wall + ~1 day overhead = **~7 days for a multi-task Baldwin retry**.

### Decision — STOP

After three iterations (M4 → M4.5 → M4.6) totalling significant project time, the Baldwin question is closed with a **STOP verdict** for Phase 5. Reasoning:

1. **Marginal scientific value is incremental, not novel.** A successful Baldwin demonstration would reproduce Fernando 2018's published result on a TPE substrate. Useful confirmation, not a new contribution.
2. **No technical blocker.** The `BaldwinInheritance` strategy code, F1 evaluator, 4-way aggregator, and 8-field pilot configs are all shipped and tested. M5 and M6 don't depend on a Baldwin GO verdict.
3. **M5 (co-evolution) intrinsically introduces task variation** via co-evolving predators — the same multi-task pressure Fernando/Chiu rely on, but emerging naturally from the ecological dynamics rather than via a hand-designed multi-task harness. M5 may demonstrate Baldwin as a side-effect without a dedicated milestone.
4. **The 7-day Path A-minimal investment is better spent on M5 directly.** Co-evolution dynamics are richer biological research than reproducing Hinton-Nowlan on a modern substrate.
5. **The 3-iteration arc produces a defensible STOP**: M4 (audit), M4.5 (audit closures + structural finding), M4.6 (substrate-constraint diagnosis). Each iteration narrowed the diagnosis. Closing now is honest scientific output, not a punt.

### What this PR (M4.6) ships

1. **Three abstraction candidates enumerated and ruled out** (B3, B6, truncated-K) via pre-flight smoke
2. **Smoke evidence** (`artifacts/logbooks/015/m4_6_smoke/`) demonstrating per-genome (fit_K=10, fit_K=25, fit_K=50) triples + Spearman analysis
3. **Substrate constraint diagnosis** (single fixed task → no Baldwin axis → all selection-feedback abstractions collapse to genome quality) cross-referenced to published literature (Fernando 2018, Chiu 2024)
4. **STOP verdict + iteration closure** in this logbook + tracker + roadmap
5. **No framework code changes** — same as M4.5, all work is investigation + analysis + documentation
6. **Deferred follow-up**: optional M4.7 to revisit Baldwin with multi-task infrastructure, gated on M5's co-evolution either (a) demonstrating Baldwin serendipitously or (b) producing infrastructure that makes M4.7 cheap to scaffold

## Conclusions

1. **(M4.5)** The framework's `BaldwinInheritance` abstraction is mechanically a no-op vs Control under matched 8-field schemas + matched seeds. Bit-identity verified empirically across all 8 seeds and explained structurally (5 conditions ⇒ identical genome populations).
2. **(M4.5)** Audit A1 closure mechanism (matched schemas + matched seeds + the aggregator's |Δ| ≤ 0.05 pre-flight check) works perfectly — the schema-shift confounder is fully eliminable.
3. **(M4.5)** The redesigned F1 evaluator (paired K'-train + L-eval, schema-prior baseline) is a clean apples-to-apples comparison and provides a reusable measurement tool for any future Baldwin design.
4. **(M4.5)** Lamarckian rerun extends M3's published numbers cleanly to n=8 — framework integrity confirmed across the M3 → M4 → M4.5 code revisions.
5. **(M4.6)** Three selection-feedback abstractions (B3 learning-gain composite, B6 cost-of-learning, truncated-K hybrid) ruled out via pre-flight smoke. All collapse to "rank by genome quality" in a single-task K=50 substrate because gen-0 fitness is mostly zero — the learning-rate signal cannot exceed the genome-quality signal when 11/12 random hyperparameter genomes don't train at all at K=25.
6. **(M4.6)** The substrate, not the abstraction, is the constraint. Single-task K=50 PPO has no Baldwin axis (optimal strategy = innate-task-specific behaviour). Published Baldwin demonstrations on hyperparameter evolution (Fernando 2018, Chiu 2024) use task distributions to create the selection pressure for general learners.
7. **(M4.6)** The Baldwin Effect question is **closed for Phase 5 with a STOP verdict**. Reproducing Fernando 2018's result on a TPE substrate is incremental rather than novel; M5 (co-evolution) will likely produce richer scientific results in the same project time.

## Next Steps

- **M5 (co-evolution arms race)** proceeds. Co-evolving predators intrinsically introduce environmental non-stationarity for prey — the same multi-task pressure Fernando/Chiu rely on. Watch for Baldwin signal as a side-effect (genome-encodes-learning-bias signature in M5 logbooks) — if it appears, the question is answered for free.
- **M6 (transgenerational memory)** uses Lamarckian inheritance (M3) as its substrate; not gated on M4.
- **Optional M4.7 (deferred)**: Revisit Baldwin with explicit multi-task infrastructure, gated on M5's outcome. Triggered if M5 produces multi-task aggregation infrastructure that makes M4.7 a small marginal investment, OR if M5 doesn't surface a Baldwin signal serendipitously and the question's marginal value rises. Estimated cost: ~5-7 days (~3-4 days impl reusing M5's varying-env machinery + ~9-14h pilot + ~1 day overhead).
- **Phase 5 tracker**: M4.6.1-M4.6.7 ticked with STOP verdict; M4 row in tracker + roadmap reframed to closed.

## Data References

### M4.5 pilot artefacts (this PR)

- Pilot output: `evolution_results/baldwin_retry_{baldwin,control,lamarckian}_lstmppo_klinotaxis_predator/seed-{42..49}/` (gitignored; staged under `artifacts/logbooks/015/m4_5_pilot/{baldwin,control,lamarckian}/seed-N/` for the logbook record)
- Aggregator output: [artifacts/logbooks/015/m4_5_pilot/summary/](../../artifacts/logbooks/015/m4_5_pilot/summary/)
  - `summary.md` — aggregator's verdict + per-seed table
  - `convergence.png` — 4-curve plot
  - `convergence_speed.csv` — per-seed table
  - `f1_learning_acceleration.csv` — F1 evaluator output (K'=10 + K'=25)
  - `elite_hyperparams_baldwin.csv` — forensic decoded elite hyperparams
- Aggregator output (K'=25 sensitivity): [artifacts/logbooks/015/m4_5_pilot/summary_kprime25/](../../artifacts/logbooks/015/m4_5_pilot/summary_kprime25/)
- Hand-tuned baseline (reused from M2.11): `evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline/seed-{42..45}.log` (n=4 only)

### Configs

- [Baldwin pilot YAML (8-field)](../../configs/evolution/baldwin_lstmppo_klinotaxis_predator_retry_pilot.yml)
- [Control pilot YAML (matching 8-field)](../../configs/evolution/control_lstmppo_klinotaxis_predator_retry_pilot.yml)
- [Lamarckian rerun YAML (4-field, M3)](../../configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml)

### Scripts

- F1 evaluator: [scripts/campaigns/baldwin_f1_postpilot_eval.py](../../scripts/campaigns/baldwin_f1_postpilot_eval.py)
- Aggregator: [scripts/campaigns/aggregate_baldwin_retry_pilot.py](../../scripts/campaigns/aggregate_baldwin_retry_pilot.py)
- Campaign scripts: [scripts/campaigns/phase5_baldwin_retry_baldwin.sh](../../scripts/campaigns/phase5_baldwin_retry_baldwin.sh), [phase5_baldwin_retry_control.sh](../../scripts/campaigns/phase5_baldwin_retry_control.sh), [phase5_baldwin_retry_lamarckian_rerun.sh](../../scripts/campaigns/phase5_baldwin_retry_lamarckian_rerun.sh)

### M4.6 smoke artefacts (this PR)

- Smoke output: [artifacts/logbooks/015/m4_6_smoke/](../../artifacts/logbooks/015/m4_6_smoke/)
  - `m4_6_b3_smoke.csv` — per-genome (fit_K=10, fit_K=25, fit_K=50) triples for the 12-genome gen-0 population at seed 42
  - `m4_6_b3_smoke.out.txt` — full smoke output with per-genome timings (~3min wall on local machine)

The smoke script itself (~150 LOC of throwaway investigation code) is not version-controlled. It deterministically reproduces the M4.5 Baldwin pilot's gen-0 12-genome population at seed 42 by re-instantiating `OptunaTPEOptimizer` with the same `(seed, bounds, population_size)` and calling `ask()` once — TPE is seeded so this is bit-deterministic. For each recovered genome, it calls `LearnedPerformanceFitness.evaluate` three times with K=10/25/50 (fresh brain each time, same per-seed RNG seed), recording the resulting fitness. Fully reproducible from this description — see `LearnedPerformanceFitness.evaluate` at [packages/quantum-nematode/quantumnematode/evolution/fitness.py](../../packages/quantum-nematode/quantumnematode/evolution/fitness.py) and `OptunaTPEOptimizer.ask` at [packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py](../../packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py).

### OpenSpec change (M4.5)

- [openspec/changes/archive/2026-05-03-add-baldwin-retry/](../../openspec/changes/archive/2026-05-03-add-baldwin-retry/) (M4.5 work, merged)

### M4.6 OpenSpec change

M4.6 ships as a STOP closure with no framework code changes — investigation + analysis + documentation only. No new OpenSpec change is opened (the M4.6 plan + smoke evidence + STOP rationale lives in this logbook + the phase5 tracker). The "scaffold OpenSpec change for M4.6" task from the M4.5 tracker is explicitly cancelled in favour of this minimal-overhead closure.

### Forward-references

- Audit findings A1-A5 are documented in detail in [logbook 014 § Audit](014-baldwin-inheritance-pilot.md)
- Optional M4.7 (deferred Baldwin retry with multi-task infrastructure) is scoped in the phase5 tracker M4.7 row, gated on M5's outcome
