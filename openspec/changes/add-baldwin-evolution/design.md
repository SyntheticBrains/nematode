> **Post-pilot status (added after PR review):** The framework changes designed below shipped successfully; the science verdict is **INCONCLUSIVE**, not the originally-anticipated GO/PIVOT/STOP. The M4 pilot's literal aggregator output was STOP, but a post-pilot audit found three blocking design flaws in the evaluation (schema-shift confounder, biologically incoherent F1 test, apples-to-oranges F1 baseline). The framework deliverables (every Decision below) are validated; the Risks section's R1 was correctly canaried; but the experimental design itself was inadequate for a definitive Baldwin verdict. M4.5 (a follow-up) will redesign the comparison. See `docs/experiments/logbooks/014-baldwin-inheritance-pilot.md` § Audit and the M4.5 task list in `openspec/changes/2026-04-26-phase5-tracking/tasks.md`. The decisions below stand for what they were intended to do (build the framework substrate); the parts about "the M4 pilot proves Baldwin works/doesn't work" need to be read with the audit in mind.

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

### Decision 2: `BaldwinInheritance` records `inherited_from` even though it has no weight substrate, reusing `_selected_parent_ids`

**Choice**: When `kind() == "trait"`, the loop tracks the prior generation's elite genome ID (top fitness, lex-tie-broken — same selection rule as Lamarckian) and writes it to the lineage CSV's `inherited_from` column for every child of the next generation. No file IO; no checkpoint substrate.

`BaldwinInheritance.select_parents` returns `[best_genome_id]` (single-element list, same shape as Lamarckian's single-elite case). The loop's existing `self._selected_parent_ids: list[str]` attribute carries the elite ID forward — no new instance attribute is needed. The per-child resolver `_resolve_per_child_inheritance` is refactored to a three-branch switch on `kind()`:

- `kind() == "none"` → returns `(None, None, "")` (existing behaviour).
- `kind() == "trait"` → returns `(None, None, parent_id)` where `parent_id = self._selected_parent_ids[0] if self._selected_parent_ids else ""`. No checkpoint paths computed.
- `kind() == "weights"` → existing logic (compute `child_capture_path`, look up parent's checkpoint, fall back on missing file).

The post-`tell` block at [loop.py:525-535](packages/quantum-nematode/quantumnematode/evolution/loop.py#L525-L535) splits into two distinct guards because `select_parents` (which populates `_selected_parent_ids`) needs to run for both Lamarckian and Baldwin, while the GC step runs only for Lamarckian:

```python
# Strategy lineage tracking (Lamarckian + Baldwin).
if self._inheritance_records_lineage():  # kind() != "none"
    next_selected = self.inheritance.select_parents(gen_ids, list(fitnesses), gen)
    self._selected_parent_ids = next_selected

# Weight-IO GC (Lamarckian only).
if self._inheritance_active():  # kind() == "weights"
    self._gc_inheritance_dir(gen - 1, [])
    self._gc_inheritance_dir(gen, self._selected_parent_ids)
```

Without this split, the M3 single-guard pattern would skip `select_parents` for Baldwin (because `_inheritance_active()` is False under `kind() == "trait"`), `_selected_parent_ids` would stay empty, and the per-child trait branch would emit empty `inherited_from` for every Baldwin row — silently breaking the lineage scenario.

**Alternative considered**: Add a new `self._baldwin_elite_id: str | None` attribute parallel to `_selected_parent_ids`. Then have `BaldwinInheritance.select_parents` return `[]` (no checkpoint to track) and route the elite ID through the new attribute.

**Rationale**: The new-attribute design adds a redundant data structure — `_baldwin_elite_id` would always equal `_selected_parent_ids[0]` for the single-elite case (which is the only configuration shipping). Reusing `_selected_parent_ids` keeps the loop's parent-tracking code path uniform across strategies and makes the kind()-switch in `_resolve_per_child_inheritance` the single point of difference. The post-pilot analysis can trace which elite genome each child shares hyperparameters with via TPE's posterior — empty `inherited_from` would lose information that's trivially derivable.

### Decision 3: `weight_init_scale` is the only NEW brain field; `entropy_decay_episodes` is exposed via schema only

**Choice**: Add `weight_init_scale: float = 1.0` (validator `[0.1, 5.0]`) to `LSTMPPOBrainConfig` ([brain/arch/lstmppo.py](packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py)). Implementation respects the existing orthogonal-init pattern at [lstmppo.py:471-488](packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py#L471-L488): the actor's hidden Linear layers and the critic's Linear layers use `nn.init.orthogonal_(weight, gain=np.sqrt(2))` — multiply that `gain` by `weight_init_scale`. The actor's output layer uses `gain=0.01` deliberately (standard PPO trick for stable initial policy) and SHALL NOT be scaled. The LSTM/GRU module (`self.rnn` at [lstmppo.py:349](packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py#L349)) is not touched by `_initialize_weights` and uses PyTorch's default — out of scope for this knob. `entropy_decay_episodes` already exists in `LSTMPPOBrainConfig` (default 500 at [lstmppo.py:102](packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py#L102)); just reference it in the Baldwin pilot's `hyperparam_schema`.

**Alternative considered**: Add several richer-schema fields (`value_loss_coef`, `lr_warmup_episodes`, etc.). Could plausibly produce a larger Baldwin signal.

**Rationale**: Smaller schema = faster TPE convergence and clearer attribution. The M3 control + 4 fields took 4-9 generations to saturate; Baldwin + 6 fields should be analyzable within the same 20-gen budget. Two extra fields are enough to test the principle without making the gen-by-gen analysis hard to interpret. The user explicitly picked these two fields after seeing the alternatives.

The choice of `weight_init_scale` ∈ `[0.5, 2.0]` schema bounds (vs the wider `[0.1, 5.0]` validator bounds) reflects the same principle: TPE samples uniformly within the schema bounds, so a tighter range concentrates exploration on plausible values. The validator is wider so the field can still be set explicitly to extreme values for non-Baldwin investigations.

### Decision 4: `--early-stop-on-saturation N` triggers on `best_fitness` not `mean_fitness`

**Choice**: Track the previous generation's `best_fitness`. If `current_best_fitness <= previous_best_fitness` for N consecutive generations, exit the loop. Reset counter on any positive improvement.

**Alternative considered**: Trigger on `mean_fitness` (population-level saturation) or on a moving-average plateau detector.

**Rationale**: `best_fitness` is the metric the speed gate uses (`mean_gen_to_092` is computed from per-generation `best_fitness` ≥ 0.92). Triggering on the same metric as the gate keeps the truncation honest. Population-mean saturation lags best-fitness saturation by several generations and would let the loop run longer than necessary. Moving-average plateau detection is cleaner statistically but adds tunable parameters (window size, plateau threshold) — one knob (N consecutive non-improving generations) is enough.

### Decision 5: Bump `CHECKPOINT_VERSION` 2 → 3, persist `_gens_without_improvement` + `_last_best_fitness`

**Choice**: Add two fields to the checkpoint pickle: `gens_without_improvement: int` (the early-stop counter) and `last_best_fitness: float | None` (the previous-generation best, needed so the post-resume comparison correctly detects whether the resumed generation is an improvement). Bump `CHECKPOINT_VERSION` to 3. M3 (v2) checkpoints cannot be resumed under M4 — clear error advised. Baldwin's elite ID rides on the existing `selected_parent_ids` field per Decision 2; no separate `baldwin_elite_id` key needed.

**Alternative considered**: No version bump — recompute the early-stop counter from the loaded `optimizer.history` on resume rather than persisting it explicitly.

**Rationale**: Recomputing from history requires the resume code to walk the entire history with the comparison rule and re-derive the counter. Cheap (O(generations)) but adds a code path that has to stay consistent with the in-loop counter logic — easy to drift. Persisting both `gens_without_improvement` (int) and `last_best_fitness` (float | None) explicitly is two extra fields in the pickle and zero extra logic. Same pattern as M3's `selected_parent_ids` addition (M2 → M3 was v1 → v2 for this exact reason).

### Decision 6: F1 control is post-hoc, not interleaved

**Choice**: New script `scripts/campaigns/baldwin_f1_postpilot_eval.py` reads each Baldwin seed's `best_params.json` after the pilot completes, instantiates the brain via the encoder, and runs L=25 frozen-eval (K=0) episodes via `EpisodicSuccessRate.evaluate`. Writes `f1_innate_only.csv`.

**Alternative considered**: Add `control_cohort_interval` and `control_cohort_fraction` config fields per the original M4.2 task draft. Every Nth generation, evaluate a fraction of the population with `learn_episodes_per_eval: 0` and tag those rows in lineage.

**Rationale**: Post-hoc is mechanically simpler (no loop change, no lineage schema change, no extra worker tuple element), and F1 is conceptually a separate forensic step — "given the elite genome the pilot found, can it produce useful behaviour without the K=50 train phase?" — not part of the evolutionary search itself. The interleaved cohort would also drag wall-time on every generation; post-hoc adds ~5 min total.

### Decision 7: Lamarckian and M3-control reruns share configs with M3, ride M4 revision for confounder-freedom

**Choice**: M4 reruns the M3 lamarckian YAML (`configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml`) and the M3 control YAML (`configs/evolution/lamarckian_lstmppo_klinotaxis_predator_control.yml`) using new campaign scripts (`phase5_m4_lamarckian_rerun.sh`, `phase5_m4_control_rerun.sh`). Same configs, different output directories, M4 code revision. The hand-tuned baseline reuses M2.11's existing run unchanged (optimiser- and inheritance-independent).

**Alternative considered**: Reuse M3's existing artefacts (`evolution_results/m3_lamarckian_lstmppo_klinotaxis_predator/` etc.) directly — they're on the published `convergence_speed.csv` already.

**Rationale**: M3 published its numbers under M3's code revision. M4 introduces a Protocol method, an early-stop monitor, and a `weight_init_scale` brain field that defaults to 1.0. The M3 numbers were *probably* reproducible under M4 (defaults are byte-equivalent), but the only honest way to compare arms is to rerun all of them on the same code revision. M3 made the same choice (M3 reran M2.11's hand-tuned baseline rather than reusing the pre-M3 numbers — see [openspec/changes/archive/2026-05-02-add-lamarckian-evolution/tasks.md] task 9.6).

## Risks / Trade-offs

\[Risk 1: TPE converges on `weight_init_scale ≈ 1.0` and `entropy_decay_episodes ≈ 500` (the brain defaults) for all 4 seeds, making the new fields uninformative\]
→ Mitigation: Canary check at gen 5 — if all 4 seeds cluster within ±5% of defaults, the speed gate will likely fail, and the next iteration should widen schema bounds (e.g., `weight_init_scale ∈ [0.1, 5.0]` and `entropy_decay_episodes ∈ [50, 5000]`) before scaling to longer runs. Documented in the logbook's Risk section so the verdict captures whether the fields were genuinely tested.

[Risk 2: F1 innate-only is at floor — the elite genome's hyperparameters can't produce useful behaviour without K=50 training]
→ Mitigation: The genetic-assimilation gate is calibrated at "+0.10pp over hand-tuned baseline" (i.e. F1 innate-only must beat baseline 0.17 by ≥10pp → 0.27). If pilot data shows F1 hovers near baseline for ALL seeds, the gate may need to drop to "+0.03pp" or be retired in favour of "F1 > random init" (assuming random init ≈ 0.05). The decision and its calibration are documented in the logbook so future Baldwin work can recalibrate against M4's data.

[Risk 3: Early-stop fires at different generations across arms, complicating the cross-arm fitness-vs-generation plot]
→ Mitigation: Aggregator pads truncated histories with the final value (carry-forward) so all curves span 1..max-gen for plotting. The speed gate uses first-reach-0.92 which is robust to truncation. The unaffected arms (e.g. control arm that never reaches 0.92) just keep running until the 20-gen budget is exhausted.

[Risk 4: TPE posterior collapse — Baldwin-rich saturates by gen 3-4 with low diversity in evolved hyperparameters across seeds]
→ Mitigation: This would actually be evidence FOR genetic assimilation (single basin of attraction = strong Baldwin signal). If the speed gate fails because saturation is too fast (no room to be 2 gens faster than control), fall back to comparing F1 innate-only across arms — a cleaner signal anyway. The logbook captures whichever pattern the data shows.

\[Risk 5: `weight_init_scale` implementation bugs — scaling the orthogonal `gain` may interact with PyTorch's existing init logic in non-obvious ways\]
→ Mitigation: Unit test asserts that `weight_init_scale=1.0` produces tensors bit-identical to current init (no-op for the actor's hidden Linears, the actor's small-init output layer, the critic's Linears, and the LSTM/GRU defaults). For `weight_init_scale=2.0` assert the actor's hidden Linears and the critic's Linears were initialised with `gain=2*np.sqrt(2)` (i.e. doubled `gain`) while the actor's output layer's `gain=0.01` is unchanged and the LSTM/GRU's parameters are unchanged from PyTorch defaults. Pre-pilot smoke (3 gens × pop 6, single seed) under Baldwin will catch any cascading effects on training stability.

\[Risk 6: The Protocol extension is not actually as clean as expected — adding `kind()` may surface other `isinstance(strategy, NoInheritance)` checks elsewhere in the codebase\]
→ Mitigation: Pre-implementation grep performed during review found exactly 1 production-code site (`_inheritance_active` at loop.py:308) and 1 docstring reference (inheritance.py:120). Both refactor to `kind()` switches in the same commit. The resume validation and GC step do NOT use isinstance — they read `_inheritance_active()` indirectly.

### Gate calibration against M3 published numbers

The aggregator's three gates are calibrated against M3's published `convergence_speed.csv` so the verdict has empirical grounding:

- **Speed gate** `mean_gen_baldwin_to_092 + 2 ≤ mean_gen_control_to_092`: M3 published `mean_gen_control_to_092 = 9.75`; passing the gate means Baldwin reaches 0.92 in ≤ 7.75 mean generations. M3's lamarckian arm reached 4.5 — substantial headroom. A Baldwin signal that is ~half the strength of Lamarckian (around gen 7) still passes. Calibrated as "noticeably faster than from-scratch" without setting an unrealistic bar.
- **Genetic-assimilation gate** `mean_f1_baldwin > mean_baseline + 0.10`: M3 published `mean_baseline = 0.170` (run_simulation.py 100-episode hand-tuned). Passing the gate means F1 innate-only ≥ 0.27. The threshold is calibrated as "genome alone (no learning) produces noticeably better policies than the hand-tuned baseline" — modest but clearly above noise.
- **Comparative gate** `mean_gen_baldwin_to_092 ≤ mean_gen_lamarckian_to_092 + 4`: at M3's published lamarckian 4.5, this means Baldwin ≤ 8.5. Note this is essentially redundant with the speed gate (control 9.75 → speed-gate-passing Baldwin ≤ 7.75 → automatically ≤ Lamarckian + 4 if Lamarckian rerun reproduces M3's 4.5). The comparative gate's job is a sanity tripwire if Lamarckian's M4 rerun underperforms its M3 published numbers.

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
