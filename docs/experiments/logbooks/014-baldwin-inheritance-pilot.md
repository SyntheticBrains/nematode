# 014: Baldwin Inheritance — M4 (LSTMPPO + klinotaxis + pursuit predators, TPE)

**Status**: `complete` — M4 **STOP ❌**. Speed gate FAIL (Baldwin = control mean 8.50, margin +0.00 vs need ≥2); Genetic-assimilation gate FAIL (F1 innate-only mean 0.000, baseline 0.170, margin -0.17 vs need ≥+0.10); Comparative gate trivially PASS. Baldwin Effect does not manifest as evolutionary acceleration on this codebase + arm. M5 (Co-evolution) is unblocked independently (its dependency is M0 + M1, not M4); M6 (transgenerational memory) needs an alternative substrate since Baldwin's hyperparameter genome alone produces zero useful priors here.

**Branch**: `feat/m4-baldwin-evolution` (PR pending)

**Date Started**: 2026-05-03

**Date Last Updated**: 2026-05-03 — M4 closed STOP. Baldwin's mean gen-to-0.92 (8.50) exactly matches control's (8.50); F1 innate-only is at floor (0.0 across all 4 seeds). Lamarckian rerun reproduces M3 exactly (mean 4.50). The negative result is published: see § Decision and § Conclusions.

This logbook covers Phase 5 M4 in a single PR: framework, pilot, two re-run controls (M3 lamarckian + M3 from-scratch on the M4 code revision), F1 innate-only post-pilot evaluation, and the **STOP** verdict. The headline finding is **the Baldwin Effect does not manifest as evolutionary acceleration on this codebase + arm: TPE-evolved learnability hyperparameters (including the new `weight_init_scale` and `entropy_decay_episodes` knobs) produce convergence speed identical to a 4-field schema without those knobs, AND the elite hyperparameter genome alone (without the K=50 train phase) produces zero successful episodes — no genetic assimilation observed.**

## Objective

Test whether **per-genome Baldwin inheritance** — recording the prior generation's elite genome ID in the lineage CSV but NOT propagating any trained weights — accelerates evolutionary convergence over the M3 from-scratch control on the M3 predator-arm configuration. The hyperparameter genome continues to evolve via TPE; the elite-ID lineage trace exists so post-pilot analysis can identify which prior-gen elite each child shares hyperparameters with via TPE's posterior. Each child trains from-scratch.

The Baldwin Effect (in evolutionary biology): lifetime learning guides genetic evolution toward genomes that learn fast, *without* the genome ever encoding the learned policy directly. M3 (logbook 013) demonstrated the Lamarckian alternative — propagating trained weights — accelerates convergence by ~5.25 generations. M4 asks whether the more biologically-realistic trait-only signal can produce comparable acceleration.

## Background

M3 (logbook 013) shipped Lamarckian inheritance and proved it accelerates convergence by ~5.25 generations on the LSTMPPO + klinotaxis + 2 pursuit-predator arm under TPE. Lamarckian is biologically inaccurate — acquired weights are not heritable in real organisms. M4 demonstrates the complementary mechanism (Baldwin Effect) on the same predator arm and tests whether the Baldwin signal is large enough on this codebase to be a credible substrate for M6 (transgenerational memory) work.

A second motivation: M3's lamarckian seeds saturated at fitness 1.00 by gen 3-7 and ran another 13-17 wasted generations. M4 runs ≥3 arms × 4 seeds × ~20 gens, so the new `--early-stop-on-saturation N` flag (also shipped in this PR) cuts wall-time on the saturating arms.

The rest of the framework (M0 + M2.12 + M3) is reused unchanged: `HyperparameterEncoder` for the genome, `OptunaTPEOptimizer` as the base optimiser, and `LearnedPerformanceFitness` for the K-train + L-eval flow. M4 ships:

- A new `BaldwinInheritance` strategy (trait-only — no per-genome weight checkpoints, but the prior-gen elite ID is recorded in `lineage.csv`'s `inherited_from` column).
- A new `kind() -> Literal["none","weights","trait"]` method on the `InheritanceStrategy` Protocol so the loop branches on intent rather than `isinstance` checks. Pure-additive Protocol extension.
- A two-guard split in the loop's post-tell block: `_inheritance_records_lineage()` wraps `select_parents` + `_selected_parent_ids` update (fires for both Lamarckian and Baldwin); `_inheritance_active()` wraps the GC step (fires only for Lamarckian, since Baldwin writes no checkpoints to GC).
- A new evolvable `weight_init_scale` LSTMPPO brain field (default 1.0, byte-equivalent to standard PPO init): scales the orthogonal-init `gain` for the actor's hidden Linears + critic's Linears. The actor's output-layer `gain=0.01` (standard PPO trick) is preserved unchanged.
- A new `--early-stop-on-saturation N` loop flag exiting when `best_fitness` plateaus for N consecutive generations.

## Hypothesis

1. **The framework's Baldwin path works mechanically**: lineage gen-0 rows have empty `inherited_from`; gen-1+ rows all share the prior-gen's elite ID; no `inheritance/` directory is ever created.
2. **Speed gate (Baldwin vs control)**: Baldwin's mean-generation-to-best ≥ 0.92 is at least 2 generations earlier than the M3-control rerun (the 4-field schema + TPE arm on the M4 code revision).
3. **Genetic-assimilation gate (F1 innate-only vs hand-tuned baseline)**: Baldwin's elite genome, evaluated with K=0 (no learning) and L=25 frozen-eval episodes, beats the run_simulation.py-driven hand-tuned baseline by at least 0.10pp on average across seeds. This is the textbook Baldwin signature: the genome alone produces useful priors over policies, even without the K=50 train phase.
4. **Comparative gate (Baldwin vs Lamarckian)**: Baldwin's mean-generation-to-best ≥ 0.92 is within 4 generations of the M3-lamarckian rerun (sanity tripwire — Baldwin doesn't have to beat Lamarckian, but mustn't be much worse).

## Method

### Architecture

LSTMPPO + klinotaxis sensing + 2 pursuit predators (the M3 predator arm). 47k brain parameters fixed; only the 6 hyperparam_schema fields evolved per genome.

### Evolutionary configuration

| Field | Value |
|---|---|
| Optimiser | TPE (Optuna) — closed RQ1 in M2.12, carried into M4 |
| Population size | 12 |
| Generations | 20 (with `early_stop_on_saturation: 5` to cut waste on saturating arms) |
| K (train episodes per eval) | 50 |
| L (frozen-eval episodes per eval) | 25 |
| Parallel workers | 4 |
| Seeds | 42, 43, 44, 45 |

### Evolved hyperparameter schema

Six fields (M3 control's 4 + 2 new innate-bias knobs):

| Field | Bounds | New in M4? |
|---|---|---|
| actor_lr | [1e-5, 1e-3] log | from M3 control |
| critic_lr | [1e-5, 1e-3] log | from M3 control |
| gamma | [0.9, 0.999] | from M3 control |
| entropy_coef | [1e-4, 0.1] log | from M3 control |
| weight_init_scale | [0.5, 2.0] | NEW (innate-bias knob) |
| entropy_decay_episodes | [200, 2000] | NEW (existing brain field, newly exposed for evolution) |

### Arms

Three evolution arms (run via campaign scripts under `scripts/campaigns/phase5_m4_*.sh`) plus one post-hoc forensic step:

| Arm | Config | Output |
|---|---|---|
| **Baldwin pilot** | `configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml` (6-field schema, `inheritance: baldwin`) | `evolution_results/m4_baldwin_lstmppo_klinotaxis_predator/seed-{42-45}/` |
| **Lamarckian rerun** | `configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml` (M3 config) on M4 revision | `evolution_results/m4_lamarckian_lstmppo_klinotaxis_predator/seed-{42-45}/` |
| **Control rerun** | `configs/evolution/lamarckian_lstmppo_klinotaxis_predator_control.yml` (M3 config) on M4 revision | `evolution_results/m4_control_lstmppo_klinotaxis_predator/seed-{42-45}/` |
| **Hand-tuned baseline** | M2.11's existing run (optimiser-independent) — no re-run needed | `evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline/` |
| **Baldwin F1 innate-only** | post-hoc: `scripts/campaigns/baldwin_f1_postpilot_eval.py` reads each Baldwin seed's gen-N elite `best_params.json`, instantiates the brain, runs L=25 frozen-eval (K=0) | `artifacts/logbooks/014/m4_baldwin_pilot/summary/f1_innate_only.csv` |

The M2.11 + M2.12 hand-tuned baseline reuses M2.11's published artefacts unchanged — those numbers are optimiser- and inheritance-independent so the M4 run revision is irrelevant. The Lamarckian and control configs ARE rerun on the M4 revision so the 4-arm comparison shares one code revision (no Python/dep/machine drift between M3's published numbers and M4's).

## Results

### Three-gate decision (from aggregator)

| Gate | Computation | Result | Margin |
|---|---|---|---|
| **Speed** (Baldwin vs control) | `mean_gen_baldwin_to_092 + 2 ≤ mean_gen_control_to_092` | **FAIL** | +0.00 (need ≥2) |
| **Genetic-assimilation** (F1 vs baseline) | `mean_f1_baldwin > mean_baseline + 0.10` | **FAIL** | -0.17 (need ≥+0.10) |
| **Comparative** (Baldwin vs Lamarckian) | `mean_gen_baldwin_to_092 ≤ mean_gen_lamarckian_to_092 + 4` | **PASS** | +0.00 |

**Hand-tuned baseline mean**: 0.170 (run_simulation.py, 100 episodes/seed, mean across 4 seeds — reused unchanged from M2.11).

### Per-seed convergence speed + F1 innate-only

| Seed | Baldwin gen-to-0.92 | Lamarckian gen-to-0.92 | Control gen-to-0.92 | F1 innate-only |
|---|---|---|---|---|
| 42 | — (saturated at 0.88, gen 7) | 3 | — (saturated at 0.84, gen 6) | 0.000 |
| 43 | 8 | 4 | 5 | 0.000 |
| 44 | 7 | 4 | 5 | 0.000 |
| 45 | 3 | 7 | 8 | 0.000 |
| **mean** | **8.50** (with seed 42 → 16 fallback) | **4.50** | **8.50** (with seed 42 → 16 fallback) | **0.000** |

Excluding seed 42's never-reached fallback: Baldwin = [8, 7, 3] mean 6.0 vs Control = [5, 5, 8] mean 6.0. Same mean. Even on the 3 seeds where both reached 0.92, Baldwin and control are statistically indistinguishable; the per-seed variance dominates any directional signal.

### Per-seed final fitness (best_fitness at last evaluated generation)

| Seed | Baldwin | Lamarckian | Control |
|---|---|---|---|
| 42 | 0.88 (gen 7) | 1.00 (gen 7) | 0.84 (gen 6) |
| 43 | 0.96 (gen 15) | 1.00 (gen 11) | 0.96 (gen 9) |
| 44 | 0.92 (gen 11) | 1.00 (gen 10) | 0.96 (gen 9) |
| 45 | 0.92 (gen 7) | 1.00 (gen 11) | 0.96 (gen 12) |
| **mean** | **0.920** | **1.000** | **0.930** |

Lamarckian rerun reproduces M3's published numbers exactly: all 4 seeds reach 1.0; mean gen-to-0.92 of 4.5. Confirms the M4 code revision is byte-equivalent for the M3 lamarckian path (the kind() Protocol method is pure-additive; the early-stop monitor is opt-in; the weight_init_scale brain field defaults to 1.0).

### Wall-time

| Phase | Wall-time |
|---|---|
| Hand-tuned baseline (4 seeds × 100 episodes) | ~2 min (reused from M2.11) |
| Baldwin pilot (4 seeds × 7-15 gens × pop 12 × K=50/L=25) | ~50 min total wall (3 arms in parallel) |
| Lamarckian rerun | (parallel with Baldwin) |
| Control rerun | (parallel with Baldwin) |
| F1 post-pilot (4 seeds × 25 episodes × K=0) | ~3 min |
| Total | **~55 min wall** with 3 arms in parallel + 1 F1 post-hoc |

`early_stop_on_saturation: 5` fired across most arm-seed combos: Baldwin saturated at gens 7-15; Lamarckian at 7-11; control at 6-12. None reached the 20-gen budget — early-stop saved roughly half the per-seed wall-time on the saturating arms.

**Convergence plot**: ![M4 convergence](../../../artifacts/logbooks/014/m4_baldwin_pilot/summary/convergence.png)

## Analysis

### The two new evolvable knobs were genuinely tested but didn't help

A central concern from design Risk 1 was that TPE might converge on `weight_init_scale ≈ 1.0` and `entropy_decay_episodes ≈ 500` (the brain defaults), making the new fields uninformative and explaining a speed-gate failure as "field never tested" rather than "field tested and found ineffective". The data rules this out:

| Seed | weight_init_scale | entropy_decay_episodes |
|---|---|---|
| 42 | 1.22 | 1465 |
| 43 | 1.33 | 1284 |
| 44 | 1.07 | 1022 |
| 45 | 0.57 | 1562 |
| (default) | 1.00 | 500 |

TPE explored the schema range for both fields. `weight_init_scale` evolved values across [0.57, 1.33] — substantially off the 1.0 default; seed 45's 0.57 is at the schema's lower edge [0.5, 2.0]. `entropy_decay_episodes` consistently evolved upward (1022-1562 across all seeds vs the 500 default), suggesting TPE wants slower entropy decay than M3's brain default. So the speed-gate FAIL is a real result: TPE found non-default values for these knobs, and those values produce convergence speed identical to the M3 control's 4-field schema. The two new knobs offer no exploitable signal on this arm.

### F1 innate-only is at floor

All 4 Baldwin seeds produced exactly 0 successful episodes when their elite hyperparameter genome was instantiated and evaluated with K=0 (no training). The hand-tuned baseline produces ~17% — meaning the Baldwin elite genome alone is *worse* than hand-tuning, not better. This is the textbook NEGATIVE Baldwin signal: the genome encodes hyperparameter values that are good when COMBINED with K=50 training, but the genome alone (without training) produces no useful behaviour. There is no genetic assimilation here — the hyperparameter genome is just a learning-rate-and-friends specification, not a behavioural prior.

The 0.0 vs 0.17 baseline gap is interesting in the other direction too: random-init brains under TPE-evolved hyperparams perform WORSE than random-init brains under hand-tuned hyperparams. Plausible explanation: TPE's evolved hyperparams are finely tuned to interact with the K=50 train phase's dynamics; without that training, the random-init policy's initial action distribution (driven partly by the evolved `entropy_coef` and `weight_init_scale`) is a poor explore-vs-exploit balance for frozen-eval.

### Lamarckian's seed-42 rescue replicated; Baldwin did NOT rescue seed 42

M3's headline anecdote was Lamarckian rescuing seed 42 (control saturated at 0.88, lamarckian reached 1.00). The M4 reruns reproduce this exactly: Lamarckian seed 42 reaches 1.00 at gen 7; M4-control seed 42 saturates at 0.84. Baldwin seed 42 also fails to rescue — saturating at 0.88 (same as M3 control's seed 42). So the rescue mechanism is specifically the **trained-weight transfer**, not the elite-ID information flow that Baldwin replicates.

This isolates the source of M3's speed lift: the +5.25 gen acceleration M3 reported was *causally driven by the bit-exact transfer of trained weights*, not by any TPE-posterior effect downstream of "the prior gen had a winner". When the trait-only signal (Baldwin) replicates the elite-ID flow without the weights, the speed lift evaporates entirely.

### Schema confounder ruled out

The Baldwin schema (6 fields) is a strict superset of the control schema (4 fields). If the speed-gate failure were caused by the M4 control rerun underperforming the M3 control (e.g. because of the kind() Protocol refactor or the early-stop flag), Baldwin might still look fast in absolute terms. The M4 control rerun produces mean gen-to-0.92 of 8.50, vs M3's published 9.75 — slightly faster, plausibly explained by the early-stop flag preventing seeds 43/44 from accumulating the late-gen flat plateau that pulled M3's published mean up. So the M4 control is, if anything, a stronger comparison than M3's, not a weaker one — strengthening the Baldwin-vs-control finding.

## Decision

**STOP ❌.** Baldwin inheritance, as implemented (trait-only — elite-ID lineage flow, no weight transfer), does NOT accelerate evolutionary convergence on the M3 predator arm. The two new evolvable knobs (`weight_init_scale`, `entropy_decay_episodes`) were genuinely explored by TPE but produce no exploitable signal beyond the M3 control's 4-field schema. The elite genome alone (F1 innate-only with K=0) produces zero successful episodes — no genetic assimilation observed.

The negative result is itself informative: M3's Lamarckian acceleration (+5.25 gens) is causally driven by the bit-exact transfer of trained weights, not by any optimiser-posterior side-effect of the elite-ID lineage flow that Baldwin replicates. M3's speed lift requires the actual trained policy, not just the genome that produced it.

## Conclusions

1. **The Baldwin Effect does not manifest on this codebase + arm.** Speed gate FAIL with Baldwin and control producing identical mean gen-to-0.92 (8.50). The richer 6-field hyperparameter schema offers no acceleration over the M3 control's 4-field schema, even though TPE genuinely explored the new fields' bounds.
2. **Genetic assimilation is not observed.** The elite hyperparameter genome alone, evaluated with K=0 frozen-eval, produces 0 successful episodes — *worse* than the hand-tuned baseline's 17%. Whatever the TPE-evolved hyperparams are doing, they don't encode behavioural priors that work without the K=50 train phase.
3. **M3's Lamarckian acceleration is causally weight-transfer, not lineage-flow.** Baldwin replicates M3's elite-ID lineage flow without the weight transfer; the speed lift evaporates entirely. This isolates the source of M3's +5.25 gen finding to the bit-exact trained-weight propagation.
4. **The Lamarckian rerun reproduces M3 exactly.** All 4 lamarckian seeds reach 1.00 with mean gen-to-0.92 of 4.50, matching M3's published numbers. Confirms the M4 code revision (kind() Protocol + early-stop monitor + weight_init_scale brain field) is byte-equivalent for the M3 lamarckian path — no regression introduced.
5. **The framework changes are still useful.** The kind()-based two-guard split, the BaldwinInheritance strategy, the early-stop flag, and the weight_init_scale brain field are all clean, validated additions to the framework. They're the substrate any future trait-flow strategy (M5 co-evolution, M6 transgenerational memory) will plug into. Baldwin's STOP doesn't block their reuse — it just informs the framing of M5 and M6.

## Next Steps

- **M5 (Co-evolution arms race)** is **unblocked**. M5's dependencies are M0 + M1 (per the Phase 5 tracker), neither of which depends on M4's outcome. The framework's evolution loop, hyperparam encoder, and lineage tracking are reused unchanged.
- **M6 (transgenerational memory)** needs to be re-evaluated. The original framing assumed Baldwin's success would prove that the genome encodes inheritable priors; with M4 STOP, that path is closed. M6 should be re-scoped to use **Lamarckian** as its substrate (or a new mechanism like direct brain-state transfer) since M3 proved that the trained-weight pathway DOES propagate useful behaviour. The "transgenerational memory" framing makes more sense with weight transfer than with hyperparameter-only.
- **Future PR** (low-priority): consider whether to add a "dynamic Baldwin" variant where the K-train budget itself is evolvable (per-genome K from a schema entry). Current Baldwin holds K=50 fixed; a richer Baldwin might evolve K to test whether shorter or longer training amortises better against the saturation gate. Out of scope for M4.
- **Future PR** (cleanup): the F1 evaluator script (`scripts/campaigns/baldwin_f1_postpilot_eval.py`) is generic — it can re-evaluate any pilot's elite genome with K=0. Could be promoted to a general-purpose `evolution_postpilot_frozen_eval.py` if other arms want F1 metrics. Out of scope for M4 (the script is named for its specific use case but is not Baldwin-coupled).

## Data References

- **Baldwin pilot artefacts**: [`evolution_results/m4_baldwin_lstmppo_klinotaxis_predator/seed-{42,43,44,45}/`](../../../evolution_results/m4_baldwin_lstmppo_klinotaxis_predator/) — `best_params.json`, `history.csv`, `lineage.csv`, `checkpoint.pkl` per seed. NO `inheritance/` subdirectory (Baldwin is mechanically a no-op on weight IO).
- **Lamarckian rerun artefacts**: [`evolution_results/m4_lamarckian_lstmppo_klinotaxis_predator/seed-{42,43,44,45}/`](../../../evolution_results/m4_lamarckian_lstmppo_klinotaxis_predator/) — same file set as Baldwin, plus `inheritance/genome-*.pt` (Lamarckian elite checkpoint).
- **Control rerun artefacts**: [`evolution_results/m4_control_lstmppo_klinotaxis_predator/seed-{42,43,44,45}/`](../../../evolution_results/m4_control_lstmppo_klinotaxis_predator/) — same file set as Baldwin, no `inheritance/`.
- **Hand-tuned baseline**: [`evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline/`](../../../evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline/) — reused from M2.11.
- **F1 innate-only forensic CSV**: [`artifacts/logbooks/014/m4_baldwin_pilot/summary/f1_innate_only.csv`](../../../artifacts/logbooks/014/m4_baldwin_pilot/summary/f1_innate_only.csv).
- **Aggregator outputs**: [`artifacts/logbooks/014/m4_baldwin_pilot/summary/`](../../../artifacts/logbooks/014/m4_baldwin_pilot/summary/) — `summary.md` (3-gate verdict), `convergence.png` (4-curve plot), `convergence_speed.csv` (per-seed gen-to-092 + F1).
- **Configs**: [`configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml`](../../../configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml).
- **Campaign scripts**: [`scripts/campaigns/phase5_m4_baldwin_lstmppo_klinotaxis_predator.sh`](../../../scripts/campaigns/phase5_m4_baldwin_lstmppo_klinotaxis_predator.sh), [`phase5_m4_lamarckian_rerun.sh`](../../../scripts/campaigns/phase5_m4_lamarckian_rerun.sh), [`phase5_m4_control_rerun.sh`](../../../scripts/campaigns/phase5_m4_control_rerun.sh).
- **F1 evaluator**: [`scripts/campaigns/baldwin_f1_postpilot_eval.py`](../../../scripts/campaigns/baldwin_f1_postpilot_eval.py).
- **Aggregator**: [`scripts/campaigns/aggregate_m4_pilot.py`](../../../scripts/campaigns/aggregate_m4_pilot.py).
- **OpenSpec change**: [`openspec/changes/add-baldwin-evolution/`](../../../openspec/changes/add-baldwin-evolution/).
- **Supporting appendix**: [`docs/experiments/logbooks/supporting/014/baldwin-inheritance-pilot-details.md`](supporting/014/baldwin-inheritance-pilot-details.md) — per-seed trajectory tables, evolved-hyperparameter distributions, F1 innate-only forensic discussion.
