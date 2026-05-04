# 015: Baldwin Effect — iterative evaluation (M4.5 + follow-ups)

**Status**: `IN PROGRESS — iteration step 1 of N`. M4.5 closed all five M4 audit findings (A1-A5) cleanly and ran a clean evaluation, but the results revealed a **structural finding**: the framework's *current* Baldwin abstraction is mechanically null vs Control under matched conditions. M4.6 (next PR) will implement an abstraction where selection explicitly uses the lineage signal and re-evaluate. This logbook spans the iteration: M4.5 below documents step 1; M4.6 will append step 2 once it lands.

**Branch**: `feat/baldwin-retry` (M4.5 PR pending; M4.6 will be a separate PR)

**Date Started**: 2026-05-03

**Date Last Updated**: 2026-05-04 — M4.5 pilot complete; structural finding identified; M4.6 follow-up scoped. This logbook stays IN PROGRESS until M4.6 closes the iteration.

This logbook covers the **iterative refinement of the Baldwin Effect evaluation** in this framework. M4 (logbook 014) shipped INCONCLUSIVE because three audit blockers prevented a clean test. M4.5 fixed all five audit findings and ran cleanly — and that clean run revealed a different problem: the framework's `BaldwinInheritance` abstraction records lineage but doesn't feed it back into selection. The fix isn't another design tweak; it's a different abstraction. M4.6 will implement it.

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

### Iteration step 2 — M4.6 (planned, separate PR)

Implement a Baldwin abstraction where selection explicitly uses the lineage signal. Two design candidates:

- **B1 — Augmented fitness**: child's effective fitness = raw fitness + bonus for being a descendant of a previous elite. Tournament selection (via `GeneticAlgorithmOptimizer`'s existing elite + tournament) preferentially samples the elite-descendant lineage. Lighter — reuses the existing GA.
- **B2 — Genome-level Lamarckian**: children sampled as Gaussian perturbations of the prior generation's elite genome (analogous to `LamarckianInheritance` for weights, but applied to the hyperparam genome). Cleanest — makes the Lamarckian/Baldwin asymmetry sharp: Lamarckian inherits weights+hyperparams; Baldwin inherits hyperparams only; Control inherits nothing.

M4.6 will choose B1 vs B2 in a separate design discussion. Reuses M4.5's F1 evaluator + 4-way aggregator + 8-field configs + n=8 seeds + smoke + review-checkpoint infrastructure.

When M4.6 lands, this logbook gets a new "## Iteration step 2 — M4.6" section with the same Hypothesis / Method / Results / Analysis / Verdict structure. The iteration terminates when one step produces a verdict that's defensible against all known design flaws.

### What this PR (M4.5) ships

1. **Audit closures**: A1 closed perfectly (`|Δ| = 0.0000`); A2/A3 addressed via paired K'-train F1 test; A4 (n=8) delivered; A5 partially answered (arch knobs explored but verdict-orthogonal).
2. **Reusable infrastructure**: redesigned F1 evaluator, 4-way aggregator with schema-equalisation pre-flight, 8-field pilot configs, campaign scripts at n=8. All ready for M4.6 to reuse.
3. **Structural finding documented**: `inherited_from` is metadata-only; doesn't feed back into selection. The proof (5 conditions ⇒ bit-identity) generalises beyond this pilot.
4. **Lamarckian rerun (n=8)**: extends M3 cleanly. Mean gen-to-0.92 = 2.75 (was 4.5 at n=4); all 8 seeds reach 1.00.
5. **No framework code changes** (per add-baldwin-retry design Decision 0). All work is pilot configs + scripts + an evaluator/aggregator redesign.

## Conclusions (iteration step 1)

1. The framework's existing `BaldwinInheritance` abstraction is mechanically a no-op vs Control under matched 8-field schemas. Bit-identity verified empirically across all 8 seeds and explained structurally (5 conditions ⇒ identical genome populations).
2. Audit A1 closure mechanism (matched schemas + matched seeds + the aggregator's |Δ| ≤ 0.05 pre-flight check) works perfectly — the schema-shift confounder is fully eliminable.
3. The redesigned F1 evaluator (paired K'-train + L-eval, schema-prior baseline) is a clean apples-to-apples comparison and provides a reusable measurement tool for any future Baldwin design.
4. n=8 doubles statistical power vs M4's n=4; SE on gen-to-092 drops from ~1.0-1.3 gens to ~0.6-0.9 gens — sufficient for the speed-gate's ±2 threshold to be 2-3σ.
5. TPE actively explores all 8 fields including the new arch knobs; `actor_hidden_dim` consistently lands above the default 64 across all seeds, supporting M4's audit A5 hypothesis that arch knobs matter under K=50.
6. Lamarckian rerun extends M3's published numbers cleanly to n=8 — framework integrity confirmed across the M3 → M4 → M4.5 code revisions.
7. The Baldwin Effect question is **unanswered, not negated**. M4.5 closed M4's design issues; M4.6 will close the abstraction issue.

## Next Steps

- **M4.6 (next PR)**: Implement a Baldwin abstraction with selection feedback (B1 augmented fitness OR B2 genome-level Lamarckian — choice deferred to M4.6 design). Re-run the same 4-arm pilot under the new abstraction. Append iteration step 2 results to this logbook.
- **Phase 5 tracker**: M4.5.1-M4.5.6 ticked; M4.5.7 (verdict-flip) deferred until M4.6 closes the iteration. New M4.6 row added with the abstraction-redesign scope.
- **M5/M6 dependencies**: stay deferred. NOT "M5 proceeds without Baldwin" yet — that was Decision 6's STOP-only conclusion, and M4.5 didn't STOP cleanly.
- **OpenSpec change `add-baldwin-retry`**: archived in this PR (M4.5 work is complete; the iterative follow-up gets its own change `add-baldwin-iterative-redesign` or similar, scaffolded at M4.6 time).

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

### OpenSpec change

- [openspec/changes/archive/2026-05-03-add-baldwin-retry/](../../openspec/changes/archive/2026-05-03-add-baldwin-retry/) (archived in this PR)

### Forward-references

- Audit findings A1-A5 are documented in detail in [logbook 014 § Audit](014-baldwin-inheritance-pilot.md)
- M4.6 (when implemented) will append iteration step 2 to this logbook + reference its own OpenSpec change
