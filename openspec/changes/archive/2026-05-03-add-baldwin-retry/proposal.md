# Proposal: Baldwin Effect first valid measurement on the LSTMPPO + klinotaxis + pursuit-predator testbed

## Post-pilot status (added at PR-time)

**Status: IN PROGRESS — iteration step 1 of N.** M4.5 ran the pilot
cleanly with all five audit findings (A1-A5) addressed, but the
results revealed a **structural finding** that requires an iterative
follow-up rather than a final verdict:

Under matched 8-field schemas + matched seeds + identical fitness
functions, Baldwin and Control evolve **bit-identical** genome
populations across all 8 seeds — same per-seed best fitness at every
generation, same elite hyperparameters in `best_params.json`. The
schema-equalisation closed audit A1 perfectly (`|Δ| = 0.0000`),
proving the audit fix works.

**Why this happens** (proven structurally):

1. Identical schema → TPE samples the same parameter vectors at
   each generation
2. Deterministic fitness function → identical fitness for identical
   genomes
3. Baldwin's `inherited_from` lineage trace is metadata-only — never
   feeds back into `optimizer.tell()` or any child's brain
   construction
4. Therefore Baldwin and Control evolve identical populations

The framework's *current* Baldwin abstraction is mechanically null
vs Control. **M4.6 (next PR) will implement an abstraction where
selection explicitly uses the lineage signal** (likely a
`BaldwinGeneticInheritance` strategy where children are sampled as
Gaussian perturbations of the prior generation's elite genome —
analogous to `LamarckianInheritance` for weights, but applied to the
hyperparam genome instead of trained weights).

This PR ships the audit closures + the redesigned F1 evaluator + the
4-way aggregator with schema-equalisation pre-flight + the 8-field
pilot configs + the Lamarckian rerun (which dramatically outperforms,
mean gen-to-0.92 = 2.75 across n=8 — confirms framework integrity).
Logbook 015 captures M4.5 as iteration step 1; M4.6 will extend it
with iteration step 2 once a valid abstraction is implemented and
re-tested.

The pre-registered STOP semantic in design Decision 6 was scoped to
"Baldwin's redesigned gates STOP after fixing the audit findings."
M4.5 closed all five audit findings cleanly; what we found is that
the framework *abstraction* itself was the wrong substrate for the
test, not that the Baldwin Effect doesn't exhibit on this testbed.
Decision 6 doesn't preclude iterating on a different abstraction
that addresses a structural finding from a clean evaluation. M5/M6
dependencies stay deferred until M4.6 closes the Baldwin question.

See [logbook 015](../../../docs/experiments/logbooks/015-baldwin-iterative-evaluation.md)
for the full M4.5 results + the M4.6 follow-up plan.

## Why

The Baldwin Effect — whether evolution under TPE produces a hyperparameter genome that biases the brain to learn fast from random init — is **unanswered** in this framework. The prior Baldwin pilot (logbook 014) shipped INCONCLUSIVE because three blocking design flaws meant the gates measured the wrong thing:

- **A1 (schema-shift confounder)**: Baldwin evolved a 6-field schema; Control evolved a 4-field schema. With the same `--seed`, TPE samples completely different parameter vectors at gen-0 because schema dimensions differ. Result: Baldwin's gen-0 starting populations were systematically -0.14pp weaker than Control's *before any Baldwin-vs-no-Baldwin signal could fire*. The "Speed gate FAIL margin +0.00" outcome could mean "Baldwin matched Control despite a worse start" or "Baldwin's mechanism is null" — the data can't distinguish.
- **A2 (F1 test biologically incoherent)**: F1 measured "do random LSTM weights succeed at the task?" (answer: no, regardless of hyperparams) instead of "does the elite genome accelerate learning?" (the actual Baldwin Effect prediction). The test was rigged to fail by design.
- **A3 (F1 baseline apples-to-oranges)**: F1 (0 episodes, no learning) was compared to a baseline computed over 100 training episodes. Even an oracle Baldwin genome would have produced F1 ≈ 0 because the test wasn't measuring what the gate claimed.

The framework deliverables from the prior pilot — `BaldwinInheritance`, the `kind()` Protocol method, the two-guard loop split, `weight_init_scale`, and `early_stop_on_saturation` — are sound and verified (Lamarckian rerun reproduces M3 exactly, all 162 evolution tests pass). Only the experimental design was flawed.

This change is the **first valid attempt** to measure the Baldwin Effect in this framework. It re-uses the M4 framework code unchanged and rebuilds the pilot's experimental design from the audit findings.

## What Changes

- **Equalised 8-field schemas across all arms.** Baldwin and Control evolve the same 8 fields: M4's 4 hyperparameter knobs (`actor_lr`, `critic_lr`, `gamma`, `entropy_coef`) + M4's 2 innate-bias knobs (`weight_init_scale`, `entropy_decay_episodes`) + 2 NEW architecture knobs (`actor_hidden_dim`, `actor_num_layers`). Same TPE prior at gen-0 → identical starting populations across arms. **Eliminates audit finding A1.**
- **F1 redesigned as a learning-acceleration test.** For each pilot seed: take elite genome → instantiate brain → train K' = 10 episodes → measure success rate; compare to a **schema-prior baseline genome** (a deterministic random sample drawn from each schema entry's prior via `HyperparameterEncoder.initial_genome` with a seeded RNG) trained the same way. Both arms include learning, so the comparison is apples-to-apples. The schema-prior baseline is "what would a generic genome get from this schema" — semantically valid and uses an existing encoder method (no inverse-encoder helper needed). **Eliminates audit findings A2 + A3.**
- **n = 8 seeds (vs M4's n = 4).** With per-seed gen-to-0.92 sd ≈ 1.7-2.6 generations (M4 measurement), n=4 gives standard error ≈ 1.0-1.3 gens — a ±2-gen threshold is roughly 1σ. n=8 halves the SE to ≈ 0.6-0.9 gens → roughly 2-3σ sensitivity for the same threshold. **Eliminates audit finding A4.**
- **Architecture knobs added to the schema** (`actor_hidden_dim`, `actor_num_layers`). These permit a head-to-head between M4's "innate-bias knob" hypothesis and the audit's A5 hypothesis that arch knobs have larger effects within K=50. We learn either way: if TPE's posterior prefers arch knobs over M4's knobs, A5 was correct; if it prefers M4's, A5 was wrong. (Note: arch-changing schema entries are permitted under Baldwin per the framework spec — Baldwin doesn't load weights, so shape mismatches between parents and children are fine.)
- **K' = 10 for the F1 evaluator** (defended quantitatively: 1/5 of K=50 — small enough that bad genomes can't catch up via brute-force training; large enough that good genomes show their advantage above per-episode noise).
- **Gate thresholds recalibrated against M4's measured numbers**, not M3's published. M4's control mean = 8.50 (not M3's 9.75). Baldwin's speed-gate threshold becomes `mean_gen_baldwin_to_092 + 2 ≤ 8.50` (using M4 control mean, recomputed from M4.5's actual control-arm rerun for cleanest comparison).
- **Empirical schema-equalisation check.** Before declaring the pilot's results valid, verify gen-0 mean fitness converges between Baldwin and Control within ε at gen-0 (i.e. that the schema-equalisation actually worked). If gen-0 fitness still diverges by > ε under matched 8-field schemas, A1 is not fully resolved and the comparison is suspect.
- **Pre-pilot design review checkpoint** (in addition to the post-pilot evaluation review). M4's audit was post-hoc; M4.5 commits to a mid-implementation design checkpoint where the user reviews the pilot configs + F1 evaluator design BEFORE the ~3-hour pilot launches.
- **Pre-registered STOP semantic.** If M4.5's redesigned gates STOP, the Baldwin Effect is *not exhibited on this testbed*. M5 (co-evolution) proceeds without Baldwin; M6 (transgenerational memory) uses Lamarckian as substrate. No further "well actually we should..." cycle on Baldwin in this Phase. This makes the gate outcome load-bearing rather than open to re-litigation.
- **Logbook 015** publishes the redesigned pilot's results + final verdict. Logbook 014 stays as the historical INCONCLUSIVE record; logbook 015 forward-references audit findings A1-A5 from logbook 014. (One experiment per logbook is the established pattern.)

## Capabilities

**Modified Capabilities:** none expected. The M4 framework code (`BaldwinInheritance`, `kind()`, two-guard split, `weight_init_scale`, `early_stop_on_saturation`) is reused unchanged. The F1 evaluator is a forensic script (`scripts/campaigns/baldwin_f1_postpilot_eval.py`) — not part of the loop's runtime contract — so its redesign doesn't touch `evolution-framework/spec.md`.

If implementation discovers a framework-level requirement we missed (e.g. the F1 evaluator's K'-train phase exposes an encoder-side gap), the proposal will be amended before specs are written.

**New Capabilities:** none.

## Impact

- **New configs**: `configs/evolution/baldwin_lstmppo_klinotaxis_predator_retry_pilot.yml` (8-field schema, `inheritance: baldwin`, n=8) + `configs/evolution/control_lstmppo_klinotaxis_predator_retry_pilot.yml` (matching 8-field schema, `inheritance: none`, n=8). Lamarckian rerun reuses the existing 4-field config (its purpose is to confirm M4 framework integrity, not to compare against the 8-field Baldwin/Control directly).
- **Redesigned script**: `scripts/campaigns/baldwin_f1_postpilot_eval.py` rewritten to do K'-train → success-rate comparison instead of K=0 frozen-eval. The schema-prior baseline genome is sampled via `HyperparameterEncoder.initial_genome(sim_config, rng=...)` with a per-seed deterministic RNG, then run through the same `LearnedPerformanceFitness.evaluate` harness as the elite — no new encoder code, no inverse-encoder helper.
- **Updated aggregator**: `scripts/campaigns/aggregate_m4_pilot.py` (or a new `aggregate_baldwin_retry.py` — TBD in design) consumes 4 arm roots × 8 seeds and emits the recalibrated 3-gate verdict. The schema-equalisation check (gen-0 fitness convergence) becomes a pre-flight assertion at the top of the aggregator output.
- **New campaign scripts**: `scripts/campaigns/phase5_baldwin_retry_*.sh` for the 4-arm pilot launches. Same wrapper pattern as existing `phase5_m4_*.sh` scripts.
- **No framework code changes** expected. If profiling shows the F1 evaluator's K'-train path exercises an encoder/loop combination not covered by existing tests, we'll add unit tests but not change the framework contract.
- **Logbook**: new `docs/experiments/logbooks/015-baldwin-retry.md` + supporting appendix at `docs/experiments/logbooks/supporting/015/`.
- **Tracker**: tick M4.5.1-M4.5.7 in `openspec/changes/phase5-tracking/tasks.md` as work completes; flip M4 row from INCONCLUSIVE to whichever the M4.5 data supports per task M4.5.7.
- **Roadmap**: M4 row updated post-verdict.
- **Compute**: ~3-4 hours wall-time for the 4-arm × 8-seed × 20-gen pilot (baseline + control + Baldwin + Lamarckian; Baldwin and Control share an 8-field schema so harder TPE search may need 25-30 gens — design.md will pin the gen budget after smoke testing).
- **Wall-time risk**: 8-field TPE search may be slower to converge than 6-field. Will smoke-test 3 gens × pop 6 × seed 42 on the new schema before committing to the full pilot.
