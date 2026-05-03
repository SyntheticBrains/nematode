# 014: Baldwin Inheritance — M4 (LSTMPPO + klinotaxis + pursuit predators, TPE)

**Status**: `complete` — see § Decision below.

**Branch**: `feat/m4-baldwin-evolution` (PR pending)

**Date Started**: 2026-05-03

**Date Last Updated**: 2026-05-03 — M4 closed. See § Decision below for the verdict.

This logbook covers Phase 5 M4 in a single PR: framework, pilot, two re-run controls (M3 lamarckian + M3 from-scratch on the M4 code revision), F1 innate-only post-pilot evaluation, and verdict. The headline finding is **TBD pending pilot completion** — to be filled when the aggregator emits its summary.

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

**TBD — to be filled when the aggregator emits its summary.**

The aggregator's three gate outputs + per-seed table will be inserted here, sourced from `artifacts/logbooks/014/m4_baldwin_pilot/summary/summary.md`.

## Analysis

**TBD — to be filled when the aggregator emits its verdict.**

## Decision

**TBD — GO / PIVOT / STOP per the aggregator's three-gate verdict.**

## Conclusions

**TBD — filled in after Analysis.**

## Next Steps

**TBD — depends on the verdict:**

- GO: M5 (Co-evolution arms race) is unblocked (its dependency is M0 + M1 — already satisfied). M6 (transgenerational memory) starts on the Baldwin config since the genetic-assimilation gate proves the genome encodes useful priors.
- PIVOT: document the partial signal, recalibrate the Baldwin gates against M4's data, and decide whether to re-run with widened schema bounds before committing to M5/M6.
- STOP: the predator arm is the wrong testbed for Baldwin OR the richer schema offers no signal. Re-evaluate before M6 commits to Baldwin as a substrate.

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
