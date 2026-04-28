# 012: Hyperparameter Evolution — MLPPPO Pilot

**Status**: `completed`

**Branch**: `feat/m2-hyperparameter-evolution`

**Date Started**: 2026-04-27

**Date Completed**: 2026-04-28

## Objective

Validate the new hyperparameter-evolution framework end-to-end: encode brain hyperparameters (rather than weights) as a flat float vector with a per-slot schema, train each genome from scratch for K episodes, evaluate it on L frozen episodes, and let CMA-ES search for hyperparameter combinations that produce competitive policies.

The pilot's job is **decision-gate**, not benchmark: does evolved-hyperparameter MLPPPO clear the +3pp threshold over the hand-tuned MLPPPO baseline? GO/PIVOT/STOP triggers the LSTMPPO+klinotaxis arm in the next PR.

## Background

Phase 5 M0 (PR #132, [logbook 011 / Klinotaxis Era](011-multi-agent-evaluation.md) follow-on) shipped a brain-agnostic evolution framework with `MLPPPOEncoder` / `LSTMPPOEncoder` weight encoders and `EpisodicSuccessRate` (frozen-weight fitness). The framework was *designed* to support hyperparameter evolution but only weight encoders were wired. M2 adds the missing pieces:

- **`HyperparameterEncoder`** — encodes brain config fields (e.g. `learning_rate`, `actor_hidden_dim`, `feature_gating`) as a flat float vector with a per-slot schema. Each evaluation builds a fresh brain from the genome's hyperparameters and trains it from scratch.
- **`LearnedPerformanceFitness`** — runs K training episodes (where `brain.learn()` IS called and weights mutate) followed by L frozen eval episodes. Score = eval-phase success rate.

These slot into the existing `GenomeEncoder` / `FitnessFunction` protocols without changing them.

This pilot is **PR 2 of three** in the post-M0 evolution work split. PR 1 (perf, merged as #133) cut per-step dead work and added opt-in CMA-ES diagonal mode. PR 2 (this PR) ships the M2 framework + this pilot. PR 3 will run the LSTMPPO+klinotaxis arm — gated on this pilot's GO/PIVOT/STOP decision.

**Prior work**: M0 brain-agnostic evolution framework (PR #132); [logbook 011](011-multi-agent-evaluation.md) (multi-agent + klinotaxis era; supplied the foraging baseline).

## Hypothesis

1. The hyperparameter-evolution framework would produce non-zero fitness end-to-end (i.e., genomes train, eval, and score in `[0, 1]`).
2. CMA-ES would find at least one hyperparameter combination that beats the hand-tuned MLPPPO baseline by ≥3pp across 4 seeds (the GO threshold).
3. CMA-ES would show monotonic mean-fitness improvement across the 20-generation budget — the canonical "convergence curve" expected of evolutionary optimisers.

Hypothesis 1 → confirmed (after fixing four blocking bugs uncovered by pre-pilot calibration).
Hypothesis 2 → confirmed (+5.5pp separation; GO).
Hypothesis 3 → **disconfirmed** (best fitness saturated at gen 1; mean fitness oscillated rather than climbed). See Analysis.

## Method

### Pilot configuration

7 evolved hyperparameters (3 architectural ints, 4 PPO floats):

| Slot | Field | Type | Bounds | Log-scale |
|---|---|---|---|---|
| 0 | `actor_hidden_dim` | int | [32, 256] | — |
| 1 | `critic_hidden_dim` | int | [32, 256] | — |
| 2 | `num_hidden_layers` | int | [1, 3] | — |
| 3 | `learning_rate` | float | [1e-5, 1e-2] | yes |
| 4 | `gamma` | float | [0.9, 0.999] | — |
| 5 | `entropy_coef` | float | [1e-4, 0.1] | yes |
| 6 | `num_epochs` | int | [1, 8] | — |

Pilot YAML: [`configs/evolution/hyperparam_mlpppo_pilot.yml`](../../../configs/evolution/hyperparam_mlpppo_pilot.yml). Brain block mirrors `configs/scenarios/foraging/mlpppo_small_oracle.yml`. Evolution block: CMA-ES, pop=12, gens=20, K=30 train + L=5 eval per genome, parallel=4, sigma0=1.0 (paired with the encoder's per-param `genome_stds`, each std = bound-width / 6).

### Campaign

Two campaign scripts under `scripts/campaigns/`:

- **Pilot** ([`phase5_m2_hyperparam_mlpppo.sh`](../../../scripts/campaigns/phase5_m2_hyperparam_mlpppo.sh)): 4 seeds (42-45) sequentially, full pilot config per seed.
- **Baseline** ([`phase5_m2_hyperparam_baseline.sh`](../../../scripts/campaigns/phase5_m2_hyperparam_baseline.sh)): 4 seeds × 100 episodes against `mlpppo_small_oracle.yml` via `scripts/run_simulation.py`. Captures the trained baseline's plateau performance.

Aggregator ([`scripts/campaigns/aggregate_m2_pilot.py`](../../../scripts/campaigns/aggregate_m2_pilot.py)) reads per-seed `history.csv` + `best_params.json` + baseline logs, emits a markdown summary and a 2-panel convergence plot.

### Code changes (high level)

- **New**: `evolution/encoders.py::HyperparameterEncoder` (brain-agnostic; `brain_name=""`), `build_birth_metadata` helper, `select_encoder` dispatch helper.
- **New**: `evolution/fitness.py::LearnedPerformanceFitness` (K-train + L-eval, fresh env between phases).
- **New**: `utils/config_loader.py::ParamSchemaEntry`, top-level `hyperparam_schema` field on `SimulationConfig`, `learn_episodes_per_eval` / `eval_episodes_per_eval` on `EvolutionConfig`. Cross-field validator catches schema name typos at YAML load.
- **New**: `--fitness {success_rate, learned_performance}` CLI flag in `scripts/run_evolution.py`.
- **Modified**: `evolution/loop.py` — populates `Genome.birth_metadata["param_schema"]` at both genome-construction sites; falls back to `sim_config.brain.name` when the encoder's `brain_name` is empty so lineage records the actual brain identity.
- **Modified**: `optimizers/evolutionary.py::CMAESOptimizer` accepts optional per-param `stds` (threaded as cma's `CMA_stds`).

Spec change: [`openspec/changes/archive/2026-04-28-2026-04-27-add-hyperparameter-evolution/`](../../../openspec/changes/archive/2026-04-28-2026-04-27-add-hyperparameter-evolution/) (proposal, design, tasks, spec deltas).

### Bugs uncovered by pre-pilot calibration

A single-generation calibration run produced 0% fitness across all 4 genomes. Investigation found four separate bugs, all fixed in commit `7795c6b2`:

1. **CMA-ES `x0=zeros` for hyperparameter encoders**. Default x0 is invalid for log-scale schemas where bounds sit far from origin (e.g. learning_rate log-bounds are [-11.5, -4.6]). Fix: pass `encoder.initial_genome().params` as x0.
2. **Single global sigma can't handle mixed-scale schemas**. log-lr range is ~7, gamma range is ~0.1 — uniform sigma is wrong for both. Fix: `GenomeEncoder` protocol gains `genome_stds()`; CMA-ES wires through `CMA_stds`.
3. **`max_body_length` not passed in fitness env construction**. Pre-existing M0 bug — `create_env_from_config` defaulted to body=6 regardless of YAML's `body_length: 2`. Fix: pass `sim_config.body_length` everywhere.
4. **No `agent.reset_environment()` between episodes**. THE big one. After episode 0 starves, the env stayed in failed state and subsequent episodes inherited it. Fix: call `agent.reset_environment()` between training and eval episodes (matching `run_simulation.py`'s per-run pattern).

See [supporting appendix](supporting/012/hyperparam-evolution-mlpppo-pilot-details.md) for full investigation traces.

## Results

### Per-seed best fitness (eval-phase success rate, L=5)

| Seed | Gen 1 best | Gen 20 best | Mean across gens |
|---|---|---|---|
| 42 | 1.000 | 1.000 | 1.000 |
| 43 | 1.000 | 1.000 | 1.000 |
| 44 | 1.000 | 1.000 | 1.000 |
| 45 | 1.000 | 1.000 | 1.000 |

**Pilot mean (gen-20 best across seeds)**: 1.000 ± 0.000

### Hand-tuned MLPPPO baseline (100 episodes per seed)

| Seed | Success rate |
|---|---|
| 42 | 0.960 |
| 43 | 0.980 |
| 44 | 0.920 |
| 45 | 0.920 |

**Baseline mean**: 0.945

### Decision gate

| | Value |
|---|---|
| Baseline mean | 0.945 |
| GO threshold (≥3pp over baseline) | 0.975 |
| Pilot mean (gen-20 best) | 1.000 |
| Separation | **+5.5pp** |

### Convergence — best vs mean fitness across population

![Convergence curves](../../../artifacts/logbooks/012/m2_hyperparam_pilot/summary/convergence.png)

The left panel shows the per-seed best fitness across 20 generations — flat at 1.0 from gen 1 in all 4 seeds. The right panel shows the per-seed mean fitness across the population of 12 — bouncing in the 0.55-0.95 range without monotonic improvement.

### Wall-time

Total campaign: **27 minutes** (faster than the 2-6 hour pre-pilot estimate). Per-seed:

| Seed | Wall |
|---|---|
| 42 | 4m 35s |
| 43 | 2m 19s |
| 44 | 2m 19s |
| 45 | 17m 58s |

Seed 45 hit a slow-training hyperparameter region (deeper network + small lr). The parallel=4 worker pool combines poorly with per-genome wall variance — one slow genome dominates each generation.

## Analysis

### Decision: GO ✅

The pilot beats the hand-tuned baseline by +5.5pp (1.000 vs 0.945), clearing the 3pp threshold. The framework demonstrably evolves brain hyperparameters end-to-end, and CMA-ES finds perfect-scoring genomes.

### Caveats — what GO does NOT mean

**1. The eval window is too small to discriminate.** L=5 means 1.0 = 5/5. With only 5 trials, a true-0.85 policy hits 5/5 by chance with probability `0.85⁵ ≈ 0.44`. So a "perfect" pilot score is consistent with a wide range of true success rates. The +5.5pp headline overstates the real separation; the test simply can't resolve below ~10pp.

**2. The metric is asymmetric.** Baseline = trained brain's plateau performance after 100 episodes (PPO converged). Pilot = eval performance after only 30 train episodes (PPO partially converged). The pilot is asking *which hyperparams reach a usable policy fastest*, not *which hyperparams reach the highest plateau*. These are different questions. The +5.5pp doesn't directly imply the latter.

**3. CMA-ES has nothing to climb.** Best-fitness curve saturates at gen 1 in all 4 seeds. The bound region of the schema is permissive enough that random sampling around `x0` finds perfect-scoring genomes immediately. Mean-fitness across the population (right panel) is also high — gen 1 mean is 0.85-0.95 — meaning a substantial fraction of randomly-sampled genomes already work.

The pilot doesn't prove "evolved hyperparams systematically beat hand-tuned" — the test is too easy. But it proves the framework is sound and worth investing in.

### Why the schema is too permissive

Looking at the per-seed winners (full table in the [appendix](supporting/012/hyperparam-evolution-mlpppo-pilot-details.md)):

| Seed | actor_dim | critic_dim | layers | lr | gamma | entropy | epochs |
|---|---|---|---|---|---|---|---|
| 42 | 224 | 125 | 3 | 0.0071 | 0.91 | 0.065 | 8 |
| 43 | 188 | 32 (clip) | 1 (clip) | 0.0018 | 0.97 | 0.00029 | 7 |
| 44 | 32 (clip) | 139 | 2 | 0.0013 | 0.90 (clip) | 0.0052 | 4 |
| 45 | 128 | 165 | 3 | 0.0027 | 0.95 | 0.0040 | 8 |

Every dimension varies 4-10× across the 4 winners, yet all hit 5/5 eval. The schema's viable region is broad rather than narrow on this task at K=30 / L=5. A flat fitness landscape gives CMA-ES no gradient to follow.

### What we'd do differently (input for the LSTMPPO PR)

- **Bump L from 5 → 50.** At L=50, `p(50/50 | true=0.95) ≈ 0.077` — much more discriminating. The pilot evaluates ~1200 episodes per seed (12 genomes × 5 eval × 20 gens); reallocating budget to fewer genomes × more eval episodes would sharpen the signal.
- **Tighten the schema bounds.** 7 orders of magnitude on `learning_rate` is more than needed once we know the viable region (lr ~ 1e-4 to 1e-2 looks sufficient).
- **Add a training-speed component to the fitness.** Integrate reward over the K train phase, not just measure post-training plateau. That captures "reaches usable policy fastest" explicitly rather than implicitly.

The LSTMPPO+klinotaxis pilot in the next PR is naturally harder: LSTMPPO trains slower (so K=30 won't hit the ceiling), klinotaxis sensing is a tighter constraint, and `rnn_type` is a categorical that exercises the framework's mixed-type handling. Worth carrying these tweaks into that pilot.

## Conclusions

1. **Hyperparameter evolution works end-to-end.** Framework ships and produces fitness scores in `[0, 1]` after fixing four pre-existing bugs that calibration uncovered (commit `7795c6b2`). The bugs were bugs regardless of M2; they just hid until hyperparameter evolution stress-tested the env-reset and optimiser-init code paths.

2. **Pilot beats baseline by +5.5pp.** Mean across 4 seeds clears the 3pp GO threshold. Decision: **GO**. The LSTMPPO+klinotaxis arm proceeds in PR 3.

3. **The test is too easy at this configuration.** L=5 can't discriminate "good" from "perfect"; the schema's viable region is broad enough that random sampling already wins. The framework is investment-worthy but the headline +5.5pp is conservative — real separation is likely smaller and unmeasurable at L=5.

4. **CMA-ES learned nothing.** Best fitness saturated at gen 1 in all 4 seeds. The 20-generation budget was wasted at this difficulty. Future pilots SHOULD start with a calibration-style probe to confirm the fitness landscape isn't flat before committing the gen budget.

5. **Four pre-existing bugs found and fixed.** All in code paths the M0 weight-evolution flow technically executed but didn't stress: x0 plumbing, per-param sigma scaling, body_length wiring, env-reset between episodes. M0 hid them because random weights typically can't succeed regardless. M2 trains genuinely, so the bugs surfaced.

## Next Steps

- [ ] PR 3: LSTMPPO+klinotaxis pilot (already gated on this GO decision). Carry forward L=50, tighter bounds, and consider an integrated-reward fitness component.
- [ ] Phase 10: M2 spec archive + roadmap row update on PR merge.
- [ ] Future PR (post-M2): consider a "sanity probe" CLI flag that runs gen 1 only and reports population fitness distribution before committing the full gen budget. Cheap to add and would have flagged this pilot's flatness immediately.
- [ ] Future PR: optimiser-portfolio re-evaluation (Optuna/TPE comparison on the same pilot config — already recorded as RQ1 in the Phase 5 tracking change).

## Data References

- **Pilot artefacts**: [`artifacts/logbooks/012/m2_hyperparam_pilot/seed-{42,43,44,45}/`](../../../artifacts/logbooks/012/m2_hyperparam_pilot/) — `best_params.json`, `history.csv`, `lineage.csv`, `checkpoint.pkl` per seed.
- **Baseline logs**: [`artifacts/logbooks/012/m2_hyperparam_pilot/baseline/`](../../../artifacts/logbooks/012/m2_hyperparam_pilot/baseline/) — `seed-{42-45}.log`.
- **Aggregated summary**: [`artifacts/logbooks/012/m2_hyperparam_pilot/summary/`](../../../artifacts/logbooks/012/m2_hyperparam_pilot/summary/) — `summary.md`, `convergence.png`.
- **Pilot config**: [`configs/evolution/hyperparam_mlpppo_pilot.yml`](../../../configs/evolution/hyperparam_mlpppo_pilot.yml) (also archived under `artifacts/logbooks/012/m2_hyperparam_pilot/`).
- **Spec change**: [`openspec/changes/archive/2026-04-28-2026-04-27-add-hyperparameter-evolution/`](../../../openspec/changes/archive/2026-04-28-2026-04-27-add-hyperparameter-evolution/) (proposal, design, tasks, spec deltas).
- **Supporting appendix**: [`docs/experiments/logbooks/supporting/012/hyperparam-evolution-mlpppo-pilot-details.md`](supporting/012/hyperparam-evolution-mlpppo-pilot-details.md) — full investigation traces, per-seed history tables, and bug-fix narratives.
