# Design: Baldwin Effect first valid measurement

## Post-pilot status (added at PR-time)

**Status: IN PROGRESS — iteration step 1 of N.**

The pilot ran cleanly, all five audit findings were addressed, and
the schema-equalisation closed audit A1 perfectly (`|Δ| = 0.0000`).
But the results revealed that the framework's *current* Baldwin
abstraction is mechanically null vs Control under matched conditions:
identical schema + identical seed + deterministic fitness function +
metadata-only `inherited_from` ⇒ Baldwin and Control evolve
bit-identical genome populations across all 8 seeds.

**Decision 6 reinterpretation** (post-pilot): the pre-registered STOP
semantic was scoped to "Baldwin's *redesigned gates* STOP after
fixing the audit findings." M4.5 closed all five audit findings
cleanly; the finding is that the framework *abstraction itself* is
the wrong substrate for the test, not that the Baldwin Effect
doesn't exhibit on this testbed. Decision 6 doesn't preclude
iterating on a different abstraction that addresses a structural
finding from a clean evaluation.

**M4.6 (next PR) follow-up**: implement a Baldwin abstraction where
selection explicitly uses the lineage signal — likely
`BaldwinGeneticInheritance` (children sampled as Gaussian
perturbations of the prior generation's elite genome, analogous to
`LamarckianInheritance` for weights but applied to the hyperparam
genome). M4.6 will reuse this PR's F1 evaluator + 4-way aggregator +
8-field pilot configs + n=8 seeds + smoke + review-checkpoint
infrastructure.

See [logbook 015](../../../docs/experiments/logbooks/015-baldwin-iterative-evaluation.md)
for the M4.5 results + M4.6 follow-up plan + the bit-identity proof.

## Context

The prior Baldwin pilot (logbook 014) shipped INCONCLUSIVE because three blocking design flaws (audit findings A1-A3) meant the gates measured the wrong thing. The framework deliverables — `BaldwinInheritance`, `kind()` Protocol method, two-guard loop split, `weight_init_scale`, `early_stop_on_saturation` — are sound and verified. This change is the first valid attempt to measure the Baldwin Effect on the LSTMPPO + klinotaxis + pursuit-predator testbed.

The work is pilot-shaped: configs + scripts + an F1 evaluator redesign + a logbook. No framework code changes are expected. The audit findings are the change's spine; each design decision below maps back to one or more findings.

## Goals / Non-Goals

**Goals:**

- Produce a definitive GO / PIVOT / STOP verdict on the Baldwin Effect for this testbed, with all three M4 audit blockers fixed
- Default to reusing M4 framework code unchanged so the comparison is to the framework's published behaviour, not to a freshly-edited surface (escape hatches for genuine bugs and framework gaps documented in Decision 0)
- Pre-register the STOP semantic so the verdict is load-bearing, not open to re-litigation
- Surface design problems mid-implementation (smoke + user review) rather than post-pilot, breaking the M4 pattern that wasted ~3 hours of pilot wall

**Non-Goals:**

- Sweep multiple K' values for the F1 test. K' = 10 is committed to (defended in Decision 3); a curve sweep is out of scope and reserved for a follow-up PR if the Baldwin signal is positive but ambiguous.
- Evolve quantum-brain hyperparameters or arms beyond the LSTMPPO + klinotaxis + pursuit-predator testbed. Other testbeds may benefit from a separate Baldwin-style pilot but that's not this change.
- Refactor framework code that's working correctly. Opinion-driven cleanup ("would be nicer if...") is rejected; it stays out of M4.5 and gets a separate task in the tracker if worth doing. (See Decision 0 for the bug-fix and gap-fix escape hatches that ARE allowed.)
- Investigate alternate inheritance mechanisms (multi-elite Baldwin, transgenerational memory). Those are separately scoped milestones.
- Re-run the M4 pilot's exact 6-field schema for direct comparison. M4's data is preserved in logbook 014 + `artifacts/logbooks/014/`; M4.5's purpose is to do the experiment correctly, not to reproduce M4.

## Decisions

### Decision 0: Framework reuse policy

**Default**: reuse M4 framework code unchanged. M4.5 is pilot-shaped; the framework deliverables (`BaldwinInheritance`, `kind()` Protocol method, two-guard loop split, `weight_init_scale`, `early_stop_on_saturation`) are already verified — M4 PR #139 shipped them, all 162 evolution tests pass, the Lamarckian rerun reproduces M3 exactly.

**Escape hatch — bug fix**: if M4.5 implementation discovers a genuine bug in the M4 framework code (spec says X, code does Y), fix it as part of this PR with a new task group + a corresponding spec delta entry. M4.5 is the natural place to land the fix because we're already in `evolution-framework` code. Document the fix clearly in logbook 015's § Method so the attribution is correct ("M4.5 caught and fixed M4 bug Z" rather than "M4.5 added behaviour Z"). Do not silently fold a bug fix into a pilot config change.

**Escape hatch — gap fix**: if M4.5 implementation surfaces a framework gap that the pilot needs to land cleanly (e.g. a new validator, a new method on the `InheritanceStrategy` Protocol, a new field on `EvolutionConfig`), follow this sequence: (1) pause; (2) amend `proposal.md` to declare the framework scope expansion; (3) get user approval on the amended proposal; (4) add the new requirement to the specs delta; (5) implement under a new task group. One round-trip with the user before building, not after.

**Out of scope — refactor**: any framework change motivated by "this would be cleaner if..." or "we should restructure X" is rejected from M4.5 even if implementation tempts us toward it. M4.5 stays pilot-shaped. Refactor opportunities surfaced during M4.5 implementation get logged as separate tracker items for future PRs.

**Why this matters**: the proposal's "modified capabilities = none" claim is a *default*, not a hard rule. Framing it as a hard rule would force shipping a known framework bug if one is discovered, which is worse than fixing it. Framing it as "anything goes" loses the M4.5-stays-pilot-shaped discipline. The bug/gap/refactor categorisation gives concrete triggers for breaking the default position.

### Decision 1: 8-field schema (M4's 4 + M4's 2 + 2 NEW arch knobs)

**Choice**: Baldwin and Control evolve the same 8 fields:

| # | Field | Source | Range |
|---|---|---|---|
| 1 | `actor_lr` | M4 control | [1e-5, 1e-3] log-scale |
| 2 | `critic_lr` | M4 control | [1e-5, 1e-3] log-scale |
| 3 | `gamma` | M4 control | [0.9, 0.999] |
| 4 | `entropy_coef` | M4 control | [1e-4, 1e-1] log-scale |
| 5 | `weight_init_scale` | M4 Baldwin | [0.5, 2.0] |
| 6 | `entropy_decay_episodes` | M4 Baldwin | [200, 2000] |
| 7 | `actor_hidden_dim` | NEW | [64, 256] (powers-of-2 nearby — exact rounding TBD per smoke test) |
| 8 | `actor_num_layers` | NEW | [1, 3] |

**Alternative considered**: Option (a) from earlier alignment — swap M4's 2 innate-bias knobs out, swap arch knobs in (6 fields total). Cheaper TPE search; lower compute.

**Rationale**: The audit's A5 finding hypothesised that M4's knobs may not be optimal for K=50 budget. A head-to-head schema (option b) tests A5 directly: if TPE's posterior pins `weight_init_scale` and `entropy_decay_episodes` at defaults while exploring arch knobs, A5 was correct. If TPE explores all 6 non-baseline knobs comparably, A5 was overstated. We learn either way. Option (a) would force us to bet on which knobs are right, and a wrong bet costs another full pilot (~3-4h wall) for the redo. Option (b) costs ~50% more TPE budget upfront (8-field search vs 6-field), bounded at one pilot.

**Architecture-changing fields are permitted under Baldwin** per the framework's published spec (`Inheritance Strategy` requirement, `Baldwin permits architecture-changing schema entries` scenario): Baldwin doesn't load weights, so shape mismatches between parents and children are fine — each child constructs a fresh brain at the genome's evolved architecture.

**Trade-off**: 8-field TPE search may need 25-30 generations to saturate (vs M4's 20). Smoke test pins the gen budget before the full pilot. If smoke shows TPE is still climbing at gen 20, full pilot uses 30; if saturated by gen 15, full pilot uses 20.

### Decision 2: Schemas equalised across Baldwin and Control arms

**Choice**: Baldwin and Control share the identical 8-field schema. Only the `evolution.inheritance` field differs (`baldwin` vs `none`). Lamarckian rerun keeps M4's 4-field schema (its purpose is reproducing M3 framework integrity, not direct comparison to Baldwin/Control).

**Alternative considered**: Run a separate "narrow" Control with M4's 4-field schema for backwards comparison.

**Rationale**: Audit finding A1 — schema-shift confounder — is the binding constraint. Identical schemas across compared arms means TPE samples the same parameter vectors at gen-0 under the same seed → identical starting populations. A separate narrow control is interesting for cross-version comparison but doesn't advance the Baldwin question and doubles compute. Skip.

**Pre-flight verification (audit-A1 closure check)**: The aggregator's first output line is the **first-evaluated generation** fitness convergence check (the from-scratch generation in which gen-0 children are evaluated). Note that the framework's two CSVs use different generation-indexing conventions: `lineage.csv` is 0-indexed (the first-evaluated generation has `generation == 0` rows), but `history.csv` is 1-indexed (the first-evaluated generation is the file's first data row, labelled `generation = 1`). The aggregator SHALL pull from `history.csv`'s first data row (or equivalently `lineage.csv`'s `generation == 0` rows) — both refer to the same set of evaluations. The output line is labelled "gen-0" for shorthand throughout the rest of this design + spec, but the implementation reads from whichever CSV is convenient:

```text
Schema-equalisation check (audit A1):
  Baldwin first-gen mean best_fitness:  X.XXX
  Control first-gen mean best_fitness:  Y.YYY
  Δ:                                     ±Z.ZZZ
  Status:                                PASS (|Δ| ≤ 0.05) | FAIL (|Δ| > 0.05)
```

If `|Δ| > 0.05` the aggregator emits a clear warning and the verdict is forced to INCONCLUSIVE regardless of the gates' outcomes — A1 is unresolved and no Baldwin-vs-Control claim can be made. The 0.05 threshold is the largest cross-arm first-evaluated-generation deviation we'd accept as "schema-equalisation worked"; it's tighter than M4's measured |Δ| = 0.14.

### Decision 3: F1 redesign — K'-train learning-acceleration test

**Choice**: For each pilot seed:

1. Take Baldwin's elite genome (top fitness, lex-tie-broken on genome ID).
2. Instantiate a fresh brain at the elite's evolved hyperparameters (including the arch fields).
3. Train K' = 10 episodes via `LearnedPerformanceFitness.evaluate` (which reads K from `sim_config.evolution.learn_episodes_per_eval` — the script copies/mutates `sim_config` to set this to K' before invoking).
4. Measure success rate over L = 25 frozen-eval episodes (same L as M4); L is plumbed via `sim_config.evolution.eval_episodes_per_eval = L` (or via the protocol's `episodes` kwarg as a fallback per [fitness.py:478-482](packages/quantum-nematode/quantumnematode/evolution/fitness.py#L478-L482)).
5. Repeat with a **schema-prior baseline genome**: sample one float per schema slot via `HyperparameterEncoder.initial_genome(sim_config, rng=np.random.default_rng(seed))` — a deterministic per-seed random sample from each schema entry's prior distribution (uniform-in-bounds for floats; log-uniform for log-scale floats; uniform-int for ints; etc., per [encoders.py:658-692](packages/quantum-nematode/quantumnematode/evolution/encoders.py#L658-L692)).
6. Baldwin signal = elite's L-eval success rate − baseline's L-eval success rate.

**Alternative considered**:

- **K' values**: K' = 5 (more aggressive — small test budget) or K' = 25 (closer to K). Sweep all three.
- **Baseline construction**: Two alternatives to the schema-prior approach:
  - *Brain-config defaults* (encode the LSTMPPOBrainConfig field defaults like `actor_lr=0.0005` into a Genome). Most faithful to "what the brain would do without any evolution," but `HyperparameterEncoder` has no `encode()` method — only `decode()`. Would need ~30 LOC of inverse-encoder helper to map defaults to genome params per slot.
  - *Schema-mid baseline* (mean of each schema entry's prior — geometric mean for log-scale, arithmetic mean otherwise). Cleanest statistical baseline (no per-seed randomness in the baseline) but requires a new helper on the encoder.

**Rationale**: K' = 10 sits at the sweet spot between two failure modes:

- **Too small (K' < 5)**: per-episode noise dominates. With L = 25 evals and binomial sampling, even a 0.5 success rate has ±0.10 standard error per seed → can't discriminate Baldwin from baseline at the gate threshold.
- **Too large (K' > 25)**: brute-force training catches up. The test reverts to "trained-policy quality" which K=50 already measures — no new information about innate bias.

K' = 10 is 1/5 of K = 50 — the elite genome must show its advantage with one-fifth the training budget for the "innate prior" claim to hold. A sweep is out of scope per the Non-Goals (reserved for follow-up if signal is positive but ambiguous).

For the baseline, the schema-prior approach was chosen over brain-config-defaults and schema-mid alternatives because: (a) it uses an existing encoder method (no new code), (b) the per-seed deterministic random sample is a defensible statistical baseline ("what would a generic genome from this schema do"), and (c) seeding with the per-seed seed makes the comparison reproducible and isolated from any other RNG state.

**Symmetry**: both elite and baseline genomes go through the same K' train + L eval harness, eliminating audit findings A2 (test now measures learning-acceleration, not random-LSTM behaviour) and A3 (both arms include learning).

### Decision 4: n = 8 seeds (vs M4's n = 4)

**Choice**: Run all 4 arms (Baldwin, Control, Lamarckian rerun, hand-tuned baseline) at n = 8 seeds: 42, 43, 44, 45, 46, 47, 48, 49.

**Alternative considered**: n = 6 (compromise compute), n = 12 (3-sigma sensitivity).

**Rationale**: M4's per-seed gen-to-0.92 sd was 1.7-2.6 generations (excluding seed 42 which never reached 0.92 in either Baldwin or Control). At n = 4, standard error of the mean ≈ sd / √n ≈ 1.0-1.3 generations — the speed-gate threshold of ±2 was roughly 1σ. At n = 8, SE ≈ 0.6-0.9 generations → 2-3σ sensitivity for the same ±2 threshold. n = 12 would push to 3-4σ but doubles the compute (6-8h wall) for diminishing return on a binary GO/STOP decision. Settle at n = 8.

**Compute impact**: ~doubles per-arm wall-time. With 3 arms in parallel (Baldwin + Control + Lamarckian) at parallel=4, expect ~2.5-3 hours per arm wall (vs M4's ~70 min at n = 4). Total pilot wall ~3-4 hours including baseline + F1 post-pilot.

### Decision 5: Gate thresholds recalibrated against M4's measured numbers

**Choice**:

- **Speed gate** (Baldwin vs Control): `mean_gen_baldwin_to_092 + 2 ≤ mean_gen_control_to_092`. Threshold of +2 holds; reference comes from M4.5's own Control arm rerun (not M4's published 8.50, since the schema is now 8-field).
- **F1 learning-acceleration gate** (Baldwin elite vs synthetic baseline): `mean_f1_baldwin > mean_f1_baseline + 0.05` after K' = 10 training. Threshold of +0.05 means the elite genome's learning is at least 5pp better than a defaults-genome's learning at K' = 10. Tighter than M4's +0.10 because both sides now include the K' train phase, so noise is symmetric (was asymmetric in M4's 0-vs-100 framing).
- **Comparative gate** (Baldwin vs Lamarckian): retained from M4: `mean_gen_baldwin_to_092 ≤ mean_gen_lamarckian_to_092 + 4`. Lamarckian rerun's value comes from the same M4.5 run, not M4's 4.50.

**Alternative considered**: Keep M4's exact thresholds (speed +2, F1 +0.10, comparative +4).

**Rationale**: Speed +2 stays because the threshold is sd-bounded, not application-bounded. F1 changes from +0.10 to +0.05 because the test changed: M4's F1 was a 0-to-baseline comparison where the baseline floor was 0.17; M4.5's F1 is a within-test comparison where the baseline is whatever K' = 10 buys a defaults genome (likely lower, so a smaller absolute gap is meaningful). Comparative +4 retained — this gate's interpretation hasn't changed.

### Decision 6: Pre-registered STOP semantic

**Choice**: If M4.5's redesigned gates STOP, the Baldwin Effect is *not exhibited on this testbed*. The downstream consequences:

- M5 (co-evolution) proceeds without Baldwin in its pipeline. M5's substrate is Lamarckian or no-inheritance.
- M6 (transgenerational memory) uses Lamarckian as substrate (per the original Phase 5 tracker's M6 dependencies, M6 was already gated on M3 GO + (M4 or M5); M4.5 STOP makes M5 the dependency).
- No further Baldwin pilot in this Phase. Future Baldwin work would require a new tracker entry with explicit re-justification (e.g. a different testbed where Baldwin's mechanism is more likely to exhibit).

**Rationale**: The audit found that M4 was not pre-registered for STOP — when the gate output was STOP, we audited and found design flaws. This is the right behaviour for a flawed experiment, but it sets up an infinite regress where any STOP outcome triggers another "well actually" cycle. M4.5's design is the audit fix; if it STOPs after fixing all five audit findings, the question is settled for this testbed.

**This is not "no audit ever again"**. Mid-pilot crashes, framework bugs, or post-pilot data anomalies still trigger investigation. What's pre-registered is that gate semantics are load-bearing **conditional on the experiment running cleanly**.

### Decision 7: Logbook 015 (separate from 014)

**Choice**: Publish the redesigned pilot's results + verdict as `docs/experiments/logbooks/015-baldwin-retry.md`. Logbook 014 remains the historical INCONCLUSIVE record; logbook 015 forward-references audit findings A1-A5 from logbook 014.

**Alternative considered**: Append M4.5 results to logbook 014 as a new section.

**Rationale**: One experiment per logbook is the established pattern (logbooks 011, 012, 013, 014). Logbook 014's current verdict (INCONCLUSIVE due to design flaws) is a finding in its own right; overwriting it would lose that historical record. Logbook 015 is the M4.5 experiment's home.

## Risks / Trade-offs

\[**Risk 1**: 8-field TPE search may not converge within 20 gens.\] → Smoke test 3 gens × pop 6 × seed 42 BEFORE the full pilot to estimate convergence rate. If smoke shows TPE is still climbing at gen 20, full pilot uses 30; if saturated by gen 15, full pilot uses 20. Adds ~2 min to the smoke (already planned) and clarifies the gen budget before committing to the ~3-hour pilot wall.

\[**Risk 2**: Schema-equalisation check fires (gen-0 mean Δ > 0.05).\] → Aggregator forces verdict to INCONCLUSIVE with a clear "audit A1 not resolved" message. M4.5 would need a separate investigation into what's still confounding gen-0 across arms (could be: TPE seed plumbing, encoder initialisation, fitness-function determinism). Mitigation: smoke-test the schema-equalisation property at smoke scale (1 seed, 1 gen) before full pilot — if smoke fails, fix before the 3-hour pilot.

\[**Risk 3**: TPE pins the 2 NEW arch knobs at fixed values, contradicting the head-to-head Decision 1.\] → Check during pilot via best_params.json forensic inspection. If arch knobs are uniformly pinned at one value across all 8 seeds, that's evidence the schema's arch-knob ranges are too narrow OR that arch knobs don't matter for K=50 (which itself is informative; documented as a finding in logbook 015 even under STOP).

\[**Risk 4**: F1 K' = 10 produces near-zero fitness across the board (both elite and baseline), making the +0.05 threshold meaningless.\] → Smoke-test the F1 evaluator on a single Baldwin elite (smoke pilot output) BEFORE the post-pilot evaluation. If F1 ≈ 0 for both elite and baseline, K' = 10 is too small for this task — fall back to K' = 25 in the post-pilot evaluation only (does not require re-running the main pilot). Document the K' calibration in logbook 015 § Method.

\[**Risk 5**: Wall-time blows out beyond 4 hours.\] → Worst case: 4 arms × 8 seeds × 30 gens × parallel=4 ≈ 8 hours of wall for the pilot alone. Mitigation: run arms sequentially with `early_stop_on_saturation: 5` (already applied to all M4 arms; saved roughly half the per-seed wall on saturating arms). If wall projects > 6h after smoke, reduce n to 6 (lower statistical power but still clear of M4's n = 4 floor).

\[**Risk 6**: Audit A2/A3 fix exposes a different design flaw we haven't anticipated.\] → Pre-pilot design review checkpoint catches design issues before pilot launches. Post-pilot evaluation review (the standing M4-lessons-learned rule) catches data-interpretation issues before logbook 015 ships.

\[**Risk 7**: Baldwin signal is genuinely null on this testbed.\] → That's the pre-registered STOP outcome. Phase 5 proceeds with M5; Baldwin re-enters the roadmap only if a different testbed motivates re-test. Logbook 015 publishes the negative finding without spin.

## Migration Plan

Not applicable — this is a research pilot, not a deployed system. The new pilot configs + F1 evaluator script + logbook are purely additive. Rollback = revert the PR; no live-system impact.

## Open Questions

- **K' = 10 vs K' = 25 calibration**: deferred to F1 smoke (Risk 4). If K' = 10 produces near-zero on both arms, K' = 25 becomes the effective design choice and proposal/design are amended in a follow-up commit before the post-pilot eval runs.
- **Gen budget (20 vs 25 vs 30)**: deferred to smoke test (Risk 1). Pinned in the campaign script after smoke runs.
- **Whether to keep the M4 4-field Lamarckian rerun**: yes for now (proves M4 framework integrity is unchanged across the M4.5 pilot's setup), but if the gen-0 schema-equalisation check shows the Lamarckian arm is irrelevant to the Baldwin-vs-Control comparison (which is the main question), we may drop it from the verdict computation in the aggregator and note its results in the appendix instead.
