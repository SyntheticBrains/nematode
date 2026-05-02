## Context

M3 shipped Lamarckian inheritance — every child of generation N+1 is warm-started from the prior generation's elite parent's trained weights. The framework now has a proven `InheritanceStrategy` Protocol with a no-op (`NoInheritance`) and weight-flow (`LamarckianInheritance`) implementation. M4 adds the Baldwin Effect: trait inheritance — the genome flows, the weights do not — and tests whether evolution under TPE produces a genome whose hyperparameters bias the brain to learn fast from random init.

The stakes are scientific (Baldwin signal viability for M6 transgenerational memory work) and engineering (a clean Protocol-extension pattern for future strategies M5/M6 will add). The codebase constraints carry over from M3: TPE is the base optimiser; LSTMPPO + klinotaxis + pursuit predators is the only non-saturated arm; K=50 train + L=25 eval is the per-genome fitness budget.

The framework as built already runs evolution-of-learning-hyperparameters from-scratch when `inheritance: none + hyperparam_schema is set + learn_episodes_per_eval > 0` — that's the M3 control arm. So a strict Baldwin implementation must be more than "M3 with `inheritance: none`": M4 needs (a) richer learnability schema (knobs the M3 control did not evolve) AND (b) a learning-blocked F1 control that proves the bias is *innate* (genetic), not just learned within the K=50 episodes.

## Goals / Non-Goals

**Goals:**

- Add a `BaldwinInheritance` strategy that records elite-parent lineage without per-genome weight checkpoints, so the loop's lineage CSV traces the trait-inheritance pattern even though no `inheritance/` directory is created.
- Extend the `InheritanceStrategy` Protocol with a `kind() -> Literal["none","weights","trait"]` method so loop branching is explicit and future strategies (M5 co-evolution, M6 transgenerational memory) can plug in cleanly.
- Add an evolvable `weight_init_scale` LSTMPPO brain field (defaulting to 1.0 so existing runs are byte-equivalent) so the Baldwin pilot can test whether init-scale is an innate-bias knob TPE can profitably evolve.
- Add an `--early-stop-on-saturation N` loop flag so M4's 3-arm pilot doesn't waste compute on already-saturated runs.
- Run a 4-arm comparison (Baldwin-rich pilot / Lamarckian rerun / M3-control rerun / Baldwin F1 innate-only) and decide GO/PIVOT/STOP via three explicit gates (speed, genetic-assimilation, comparative).
- Publish logbook 014 with the verdict and Baldwin-vs-Lamarckian discussion.

**Non-Goals:**

- Multi-elite Baldwin (round-robin or tournament selection of multiple parents). The single-elite-broadcast contract from M3 carries over; multi-elite is reserved for M5/M6.
- Architecture-evolving Baldwin (evolving `actor_hidden_dim`, `lstm_hidden_dim`, etc.). Baldwin technically permits this (no per-genome weight load → shape mismatches are fine), but the M4 pilot keeps fixed architecture for fair comparison with the M3 control.
- New optimisers. M4 uses TPE per RQ1's resolution.
- New brain architectures. M4 uses LSTMPPO + klinotaxis + pursuit predators (the M3 arm).
- Interleaved learning-blocked control cohort (the original M4.2 task draft proposed this). User chose post-hoc `baldwin_f1_postpilot_eval.py` instead — keeps the loop unchanged.
- Quantum brain Baldwin runs. M4 stays classical.
- F2/F3 multi-generation transgenerational memory tests. Reserved for M6.

## Decisions

### Decision 1: Extend the `InheritanceStrategy` Protocol with `kind()` rather than special-case Baldwin in the loop

**Choice**: Add `kind() -> Literal["none", "weights", "trait"]` to the Protocol. `_inheritance_active()` becomes `kind() == "weights"`. A new helper `_inheritance_records_lineage()` returns `kind() != "none"`. Baldwin uses the same code path as `NoInheritance` for weight IO and the same code path as `LamarckianInheritance` for elite-ID lineage tracking.

**Alternative considered**: Special-case `BaldwinInheritance` in the loop with a dedicated `_baldwin_resolve_per_child` method alongside `_resolve_per_child_inheritance`. Faster to ship (~1 hour saved), but every future strategy (M5 co-evolution, M6 transgenerational memory) would also need its own special case. Within 2 milestones the loop becomes a tangle of `isinstance` branches.

**Rationale**: M3 deliberately left this seam ([M3 design.md openspec/changes/archive/2026-05-02-add-lamarckian-evolution/design.md] noted future-work strategies "would need a Protocol extension or special-case path"). Doing the Protocol extension now (a) makes Baldwin a 50-line strategy rather than a 200-line loop refactor, (b) tests that the M3 Protocol shape actually scales, and (c) front-loads the architectural work so M5/M6 inherit a clean extension point. The cost is ~1 hour of refactoring `_inheritance_active` call sites.

### Decision 2: `BaldwinInheritance` records `inherited_from` even though it has no weight substrate

**Choice**: When `kind() == "trait"`, the loop tracks the prior generation's elite genome ID (top fitness, lex-tie-broken — same selection rule as Lamarckian) and writes it to the lineage CSV's `inherited_from` column for every child of the next generation. No file IO; no checkpoint substrate.

**Alternative considered**: Leave `inherited_from` empty for Baldwin runs (since no weights actually flow). Cleaner from a "data should reflect physical reality" standpoint.

**Rationale**: TPE's posterior is biased toward observed-good hyperparameter regions, so the prior generation's elite genome IS the conceptual parent of the next generation's children — even though the underlying optimiser is what propagates that bias, not a code path in our loop. Recording the elite ID makes the lineage CSV uniform across `LamarckianInheritance` and `BaldwinInheritance`, and the post-pilot analysis can actually trace which elite genome each child shares hyperparameters with via TPE's posterior. Empty `inherited_from` would lose information that's trivially derivable.

### Decision 3: `weight_init_scale` is the only NEW brain field; `entropy_decay_episodes` is exposed via schema only

**Choice**: Add `weight_init_scale: float = 1.0` (validator `[0.1, 5.0]`) to LSTMPPO brain config. Implementation: at construction, after layers are built but before training, multiply each `nn.Linear.weight` and LSTM weight tensor's std by this scale. `entropy_decay_episodes` already exists in `_reservoir_lstm_base.py:159`; just reference it in the Baldwin pilot's `hyperparam_schema`.

**Alternative considered**: Add several richer-schema fields (`value_loss_coef`, `lr_warmup_episodes`, etc.). Could plausibly produce a larger Baldwin signal.

**Rationale**: Smaller schema = faster TPE convergence and clearer attribution. The M3 control + 4 fields took 4-9 generations to saturate; Baldwin + 6 fields should be analyzable within the same 20-gen budget. Two extra fields are enough to test the principle without making the gen-by-gen analysis hard to interpret. The user explicitly picked these two fields after seeing the alternatives.

The choice of `weight_init_scale` ∈ `[0.5, 2.0]` schema bounds (vs the wider `[0.1, 5.0]` validator bounds) reflects the same principle: TPE samples uniformly within the schema bounds, so a tighter range concentrates exploration on plausible values. The validator is wider so the field can still be set explicitly to extreme values for non-Baldwin investigations.

### Decision 4: `--early-stop-on-saturation N` triggers on `best_fitness` not `mean_fitness`

**Choice**: Track the previous generation's `best_fitness`. If `current_best_fitness <= previous_best_fitness` for N consecutive generations, exit the loop. Reset counter on any positive improvement.

**Alternative considered**: Trigger on `mean_fitness` (population-level saturation) or on a moving-average plateau detector.

**Rationale**: `best_fitness` is the metric the speed gate uses (`mean_gen_to_092` is computed from per-generation `best_fitness` ≥ 0.92). Triggering on the same metric as the gate keeps the truncation honest. Population-mean saturation lags best-fitness saturation by several generations and would let the loop run longer than necessary. Moving-average plateau detection is cleaner statistically but adds tunable parameters (window size, plateau threshold) — one knob (N consecutive non-improving generations) is enough.

### Decision 5: Bump `CHECKPOINT_VERSION` 2 → 3, persist `_gens_without_improvement` and `_baldwin_elite_id`

**Choice**: Add two fields to the checkpoint pickle: `gens_without_improvement: int` (the early-stop counter) and `baldwin_elite_id: str` (the prior-generation elite ID for Baldwin lineage). Bump `CHECKPOINT_VERSION` to 3. M3 (v2) checkpoints cannot be resumed under M4 — clear error advised.

**Alternative considered**: Reuse the existing `selected_parent_ids` field for `baldwin_elite_id` (since Baldwin's "selected parent" is a single ID). Avoid the version bump.

**Rationale**: Reusing the field would make the pickle ambiguous — `selected_parent_ids` has a different semantic meaning (Lamarckian: parents of the about-to-evaluate generation; Baldwin: elite ID of the just-evaluated generation, used for lineage only). Two distinct fields keep the resume code path explicit. Same pattern as M2 → M3 (v1 → v2).

### Decision 6: F1 control is post-hoc, not interleaved

**Choice**: New script `scripts/campaigns/baldwin_f1_postpilot_eval.py` reads each Baldwin seed's `best_params.json` after the pilot completes, instantiates the brain via the encoder, and runs L=25 frozen-eval (K=0) episodes via `EpisodicSuccessRate.evaluate`. Writes `f1_innate_only.csv`.

**Alternative considered**: Add `control_cohort_interval` and `control_cohort_fraction` config fields per the original M4.2 task draft. Every Nth generation, evaluate a fraction of the population with `learn_episodes_per_eval: 0` and tag those rows in lineage.

**Rationale**: Post-hoc is mechanically simpler (no loop change, no lineage schema change, no extra worker tuple element), and F1 is conceptually a separate forensic step — "given the elite genome the pilot found, can it produce useful behaviour without the K=50 train phase?" — not part of the evolutionary search itself. The interleaved cohort would also drag wall-time on every generation; post-hoc adds ~5 min total.

### Decision 7: Lamarckian and M3-control reruns share configs with M3, ride M4 revision for confounder-freedom

**Choice**: M4 reruns the M3 lamarckian YAML (`configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml`) and the M3 control YAML (`configs/evolution/lamarckian_lstmppo_klinotaxis_predator_control.yml`) using new campaign scripts (`phase5_m4_lamarckian_rerun.sh`, `phase5_m4_control_rerun.sh`). Same configs, different output directories, M4 code revision. The hand-tuned baseline reuses M2.11's existing run unchanged (optimiser- and inheritance-independent).

**Alternative considered**: Reuse M3's existing artefacts (`evolution_results/m3_lamarckian_lstmppo_klinotaxis_predator/` etc.) directly — they're on the published `convergence_speed.csv` already.

**Rationale**: M3 published its numbers under M3's code revision. M4 introduces a Protocol method, an early-stop monitor, and a `weight_init_scale` brain field that defaults to 1.0. The M3 numbers were *probably* reproducible under M4 (defaults are byte-equivalent), but the only honest way to compare arms is to rerun all of them on the same code revision. M3 made the same choice (M3 reran M2.11's hand-tuned baseline rather than reusing the pre-M3 numbers — see [openspec/changes/archive/2026-05-02-add-lamarckian-evolution/tasks.md] task 9.6).

## Risks / Trade-offs

\[Risk 1: TPE converges on `weight_init_scale ≈ 1.0` and `entropy_decay_episodes ≈ 800` (the M3 defaults) for all 4 seeds, making the new fields uninformative\]
→ Mitigation: Canary check at gen 5 — if all 4 seeds cluster within ±5% of defaults, the speed gate will likely fail, and the next iteration should widen schema bounds (e.g., `weight_init_scale ∈ [0.1, 5.0]` and `entropy_decay_episodes ∈ [50, 5000]`) before scaling to longer runs. Documented in the logbook's Risk section so the verdict captures whether the fields were genuinely tested.

[Risk 2: F1 innate-only is at floor — the elite genome's hyperparameters can't produce useful behaviour without K=50 training]
→ Mitigation: The genetic-assimilation gate is calibrated at "+0.10pp over hand-tuned baseline" (i.e. F1 innate-only must beat baseline 0.17 by ≥10pp → 0.27). If pilot data shows F1 hovers near baseline for ALL seeds, the gate may need to drop to "+0.03pp" or be retired in favour of "F1 > random init" (assuming random init ≈ 0.05). The decision and its calibration are documented in the logbook so future Baldwin work can recalibrate against M4's data.

[Risk 3: Early-stop fires at different generations across arms, complicating the cross-arm fitness-vs-generation plot]
→ Mitigation: Aggregator pads truncated histories with the final value (carry-forward) so all curves span 1..max-gen for plotting. The speed gate uses first-reach-0.92 which is robust to truncation. The unaffected arms (e.g. control arm that never reaches 0.92) just keep running until the 20-gen budget is exhausted.

[Risk 4: TPE posterior collapse — Baldwin-rich saturates by gen 3-4 with low diversity in evolved hyperparameters across seeds]
→ Mitigation: This would actually be evidence FOR genetic assimilation (single basin of attraction = strong Baldwin signal). If the speed gate fails because saturation is too fast (no room to be 2 gens faster than control), fall back to comparing F1 innate-only across arms — a cleaner signal anyway. The logbook captures whichever pattern the data shows.

\[Risk 5: `weight_init_scale` implementation bugs — multiplying init std at construction may interact with PyTorch's existing init logic in non-obvious ways\]
→ Mitigation: Unit test asserts that `weight_init_scale=1.0` produces tensors bit-identical to current init (no-op) and `weight_init_scale=2.0` produces tensors with std exactly 2× the baseline. Pre-pilot smoke (3 gens × pop 6, single seed) under Baldwin will catch any cascading effects on training stability.

\[Risk 6: The Protocol extension is not actually as clean as expected — adding `kind()` may surface other `isinstance(strategy, NoInheritance)` checks elsewhere in the codebase\]
→ Mitigation: Pre-implementation grep for `isinstance.*NoInheritance` and `isinstance.*LamarckianInheritance` to find every site. Plan-level estimate: 2-3 sites total (`_inheritance_active`, possibly the resume check, possibly the GC step). All refactor to `kind()` switches in the same commit.

## Migration Plan

**Deploy**:

1. Land the framework changes (Protocol + BaldwinInheritance + weight_init_scale + early-stop + validators + spec delta) in a single commit series.
2. Run the pre-pilot smoke (3 gens × pop 6 × seed 42) on a tiny config to validate mechanical correctness. Assert: no `inheritance/` directory created for Baldwin; lineage `inherited_from` populated for Baldwin gen-1+ rows; early-stop counter increments correctly.
3. Run the 3-arm pilot (Lamarckian rerun + Baldwin pilot + M3-control rerun) in parallel via the three campaign scripts.
4. Run the F1 post-pilot evaluator.
5. Run the aggregator to produce the verdict.
6. Publish logbook 014.
7. Update tracker (M4.1-M4.7 → ✅) and roadmap (M4 row → ✅).

**Rollback**:

The PR is rollback-by-revert. M3 callers continue to work — `inheritance: none` and `inheritance: lamarckian` paths are byte-equivalent except for the pure-additive `kind()` Protocol method.

The only caller-visible breaking change is `CHECKPOINT_VERSION` 2 → 3. M3 (v2) checkpoints cannot be resumed under M4. Rollback restores M3 checkpoint compatibility automatically. Production runs on `main` would not be affected because `main` is currently at M3 (no in-flight M3 runs would even attempt to resume under M4 unless the user explicitly merged then re-ran).

## Open Questions

None at plan-approval time. The following are deliberate non-decisions deferred to implementation:

- Exact LSTM weight init multiplication implementation (PyTorch idiom: `nn.init.normal_(p, std=...)` vs in-place tensor scaling). To be decided when writing `weight_init_scale` application code; consult existing init patterns in the brain module.
- Whether the early-stop log message should include the elapsed wall-time. Decided when writing the log line.
- Whether the F1 evaluator script should reuse `EpisodicSuccessRate` directly or wrap it for the K=0 frozen-eval. To be decided when writing the script; reuse is preferred per the "reused, not rebuilt" principle.
