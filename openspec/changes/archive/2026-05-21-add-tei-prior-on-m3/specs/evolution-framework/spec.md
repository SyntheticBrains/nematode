# Evolution Framework — Delta for `add-tei-prior-on-m3` (M6.13)

## ADDED Requirements

### Requirement: Composed Lamarckian + Transgenerational Inheritance Strategy

The system SHALL provide a fifth `InheritanceStrategy` implementation, `LamarckianTransgenerationalInheritance`, that composes the M3 weight-inheritance path (per-genome `.pt` warm-start + capture + GC) with the M6.9+ substrate-flow path (F0 substrate extraction + decayed cascade applied via `brain.tei_prior`). The class SHALL live in a new module `quantumnematode/evolution/lamarckian_transgenerational_inheritance.py` so the two parent strategies (`LamarckianInheritance`, `TransgenerationalInheritance`) remain auditable in isolation.

The class SHALL provide the four `InheritanceStrategy` Protocol methods with the following per-method behaviour:

- `select_parents(gen_ids, fitnesses, generation)` — IDENTICAL semantics to `LamarckianInheritance.select_parents` with `elite_count=1` (top-fitness, lex-tie-broken on genome_id; multi-elite reserved for future work).
- `assign_parent(child_index, parent_ids)` — IDENTICAL semantics to `LamarckianInheritance.assign_parent` (round-robin; with single elite, every child broadcasts from the same parent — M3 pattern).
- `checkpoint_path(output_dir, generation, genome_id)` — IDENTICAL semantics to `LamarckianInheritance.checkpoint_path`: returns `output_dir / "inheritance" / f"gen-{generation:03d}" / f"genome-{genome_id}.pt"`. Returns the canonical `.pt` path, NOT the `.tei.pt` substrate path (the substrate path is owned by the F0 substrate-extraction pipeline as a separate concern).
- `kind()` — returns the new Literal value `"weights+transgenerational"`.

The `InheritanceStrategy.kind()` Protocol Literal SHALL widen from `Literal["none", "weights", "trait", "transgenerational"]` to `Literal["none", "weights", "trait", "transgenerational", "weights+transgenerational"]`. The widening SHALL be backwards-compatible: every existing strategy continues to return the same value it returned pre-M6.13; only the new composed class adds the new value.

#### Scenario: composed strategy returns weights+transgenerational kind

- **WHEN** an instance of `LamarckianTransgenerationalInheritance` is constructed with `elite_count=1`
- **THEN** `instance.kind()` SHALL return the string `"weights+transgenerational"`
- **AND** the return value SHALL be a member of the widened `InheritanceStrategy.kind()` Protocol Literal
- **AND** `isinstance(instance, InheritanceStrategy)` SHALL evaluate `True` under `runtime_checkable`

#### Scenario: composed strategy mirrors Lamarckian's select_parents

- **WHEN** `select_parents` is called with `gen_ids=["g0", "g1", "g2"]`, `fitnesses=[0.5, 0.8, 0.6]`, `generation=0` on both `LamarckianTransgenerationalInheritance(elite_count=1)` and `LamarckianInheritance(elite_count=1)`
- **THEN** both SHALL return `["g1"]` (the top-fitness genome)
- **AND** under a fitness tie (e.g. `fitnesses=[0.8, 0.8, 0.6]`), both SHALL return `["g0"]` (lex-tie-broken on genome_id)

#### Scenario: composed strategy mirrors Lamarckian's assign_parent round-robin

- **WHEN** `assign_parent` is called repeatedly across child indices `[0, 1, 2, 3]` with `parent_ids=["pA", "pB"]` on `LamarckianTransgenerationalInheritance(elite_count=2)`
- **THEN** the returned parent IDs SHALL be `["pA", "pB", "pA", "pB"]` (round-robin, matching Lamarckian's behaviour)

#### Scenario: composed strategy returns the canonical .pt checkpoint path

- **WHEN** `checkpoint_path(output_dir=Path("/output"), generation=2, genome_id="abc")` is called on `LamarckianTransgenerationalInheritance(elite_count=1)`
- **THEN** the returned path SHALL equal `Path("/output/inheritance/gen-002/genome-abc.pt")` (the canonical Lamarckian path, NOT `.tei.pt`)
- **AND** the returned path SHALL be identical to what `LamarckianInheritance.checkpoint_path` would return for the same inputs

#### Scenario: composed strategy rejects elite_count != 1 like Lamarckian

- **WHEN** `LamarckianTransgenerationalInheritance(elite_count=0)` or `LamarckianTransgenerationalInheritance(elite_count=-1)` is constructed
- **THEN** construction SHALL raise `ValueError` with a clear message stating `elite_count` MUST be `>= 1`
- **AND** the validator on `EvolutionConfig` SHALL reject `inheritance_elite_count != 1` when `inheritance: weights+transgenerational` (mirroring the Lamarckian single-elite-broadcast contract)

### Requirement: Composed Inheritance Loop Integration

When `inheritance: weights+transgenerational` is active, the `EvolutionLoop` SHALL run BOTH the weight-IO code path AND the substrate-flow code path in parallel for every F1+ child. Specifically:

- The `_inheritance_active()` predicate SHALL widen to return `True` when `kind() in {"weights", "weights+transgenerational"}` (gates weight-IO: capture, GC, warm-start).
- The `_substrate_inheritance_active()` predicate SHALL widen to return `True` when `kind() in {"transgenerational", "weights+transgenerational"}` (gates F0 substrate extraction + F1+ `tei_prior_source` plumbing).
- The `_resolve_per_child_inheritance` switch SHALL gain a fifth branch for `kind == "weights+transgenerational"` that returns the same `(parent_warm_start_path, child_capture_path, parent_id)` tuple shape as the existing `"weights"` branch (functionally identical; future divergence of the composed semantics would happen here).
- The `_expected_kind` dict in `EvolutionLoop.__init__` SHALL gain a `"weights+transgenerational": "weights+transgenerational"` entry so the kind-mismatch defensive check accepts the new strategy.

The F1+ workers SHALL receive BOTH `warm_start_path_override` (resolved from the new `_resolve_per_child_inheritance` branch) AND `tei_prior_source` (computed by `_compute_tei_prior_source`, unchanged from M6.9+) in their `LearnedPerformanceFitness.evaluate` call.

#### Scenario: composed mode fires both weight-IO and substrate-flow paths

- **GIVEN** an `EvolutionConfig` with `inheritance: weights+transgenerational` and `transgenerational.enabled: true`
- **WHEN** the loop runs for ≥ 2 generations
- **THEN** every F1+ child's `LearnedPerformanceFitness.evaluate` call SHALL receive `warm_start_path_override` pointing to the F0 elite's `.pt`
- **AND** the same `evaluate` call SHALL receive `tei_prior_source` pointing to the F0-extracted substrate `.tei.pt`
- **AND** the F0 substrate-extraction pipeline SHALL fire at end of gen 0 (substrate is extracted from the F0 elite's brain)
- **AND** F1+ children SHALL also write per-child `.pt` checkpoints for inheritance to F2+ (the M3 cascade pattern)

#### Scenario: composed mode preserves M3 and M6.9+ byte-equivalence on their own paths

- **GIVEN** a run with `inheritance: lamarckian` (M3 baseline) under any seed
- **WHEN** the run executes after M6.13 ships
- **THEN** the loop's behaviour SHALL be byte-equivalent to the pre-M6.13 Lamarckian path (no GC suppression, no substrate extraction, no `tei_prior_source` computation, no widened-predicate side-effects)
- **AND** the same byte-equivalence SHALL hold for `inheritance: transgenerational` (M6.9+ pure-TEI path — no warm-start, no weight capture, F0 substrate-extraction internal GC fires as before)
- **AND** the same byte-equivalence SHALL hold for `inheritance: baldwin` and `inheritance: none`

#### Scenario: F0 .pt files survive substrate extraction under composed mode

- **GIVEN** a composed-mode run with `population_size: 8`
- **WHEN** gen 0 completes
- **THEN** the main-loop GC pass (running at end of gen 0 after `select_parents`, before substrate extraction) SHALL fire under composed mode — `_inheritance_active()` widens to `True` for `kind() == "weights+transgenerational"` — and SHALL keep only the F0 elite's `.pt` in `gen-000/` (deleting the other 7)
- **AND** the F0 substrate-extraction pipeline SHALL THEN run, loading the surviving elite `.pt`, writing `.tei.pt`
- **AND** the inline GC loop inside `_run_f0_substrate_extraction` SHALL be SKIPPED under composed mode (`_combined_inheritance_active()` returns `True`) so the elite's `.pt` is NOT deleted by it
- **AND** the net effect at the moment gen 1 begins SHALL be IDENTICAL to a pure-M3 run: exactly one F0 `.pt` survives, indexed by the elite's genome_id, alongside the `.tei.pt` substrate artefact
- **AND** under non-composed `inheritance: transgenerational` mode, the main-loop GC SHALL NOT fire (`_inheritance_active()` is False) but the inline GC inside `_run_f0_substrate_extraction` SHALL still fire (no behaviour change for M6.9+ — the inline GC is the load-bearing pure-TEI cleanup)

#### Scenario: kind() Protocol method gates the widened loop behaviour

- **GIVEN** an `InheritanceStrategy` instance whose `kind()` returns one of the five legal values
- **WHEN** the loop computes `_inheritance_active()` and `_substrate_inheritance_active()`
- **THEN** both SHALL evaluate `True` when `kind() == "weights+transgenerational"` (composed mode runs both paths)
- **AND** `_inheritance_active()` SHALL evaluate `True` when `kind() == "weights"` (Lamarckian-only weight-IO path; substrate predicate is False)
- **AND** `_substrate_inheritance_active()` SHALL evaluate `True` when `kind() == "transgenerational"` (M6.9+ pure-TEI substrate-flow path; weight-IO predicate is False)
- **AND** both predicates SHALL evaluate `False` when `kind() == "none"` or `kind() == "trait"` (no weight-IO; no substrate flow)
- **AND** `_inheritance_records_lineage()` SHALL evaluate `True` when `kind() == "weights+transgenerational"` (composed mode populates the lineage CSV's `inherited_from` column at F1+, same as the four existing non-`none` kinds — the existing `kind() != "none"` predicate widens through naturally; no separate widening needed for lineage tracking)

### Requirement: M6.13 Cross-Arm Primary Verdict (Reframed)

The M6.13 aggregator SHALL produce a cross-arm primary verdict using the reframed pair `tei_weights − weights_only` (not the M6.9+ pair `tei_on − control`). The statistical machinery SHALL be IDENTICAL to M6.9+'s n=4-noise-aware verdict (one-sided Wilcoxon signed-rank + 80% bootstrap CI with non-overlapping criterion). The primary verdict SHALL be GO iff:

1. The `tei_weights` arm passes its per-arm gate (F1 ≥ 40% × F0, F2 ≥ 25% × F0, F3 ≥ 15% × F0, monotone non-increasing) in ≥ 2 of 4 seeds, AND
2. The paired-seed delta `tei_weights − weights_only` on F1+ retention is statistically distinguishable from zero via BOTH:
   - one-sided Wilcoxon signed-rank with p < 0.10, AND
   - ≥ 5 percentage points absolute mean delta AND non-overlapping 80% bootstrap confidence intervals (1000 resamples per seed).

Both checks (Wilcoxon AND bootstrap) MUST agree on direction (both positive in favour of `tei_weights`). A bare 5pp threshold without statistical agreement SHALL NOT pass the primary verdict (n=4 is noise-bounded — same constraint as M6.9+).

Two secondary verdicts SHALL also be emitted:

- `weights_only − control` (M3 re-reproduction at K_test — should match PR-A's +17.5pp scaled to the chosen K_test).
- `tei_weights − control` (composed-arm vs floor — weaker claim than the primary; useful as a sanity check that the composed arm is at least as good as the floor).

#### Scenario: GO when both Wilcoxon and bootstrap agree on tei_weights superiority

- **WHEN** the four paired-seed deltas `tei_weights − weights_only` are `[+0.08, +0.10, +0.07, +0.09]` (mean +0.085)
- **AND** Wilcoxon one-sided p < 0.10
- **AND** the 80% bootstrap CI of the mean delta is `[+0.04, +0.13]` (non-overlapping with zero)
- **AND** `tei_weights` passes its per-arm gate
- **THEN** the cross-arm primary verdict SHALL be GO
- **AND** the aggregator SHALL emit `m614_frequency_prior_trigger.md` documenting the M6.13 GO finding AND recommending the M6.14 frequency-prior ablation scaffold

#### Scenario: STOP when tei_weights ≈ weights_only (substrate prior inert)

- **WHEN** the four paired-seed deltas are `[+0.01, +0.02, -0.01, +0.02]` (mean +0.01)
- **AND** Wilcoxon p > 0.10 (insignificant)
- **THEN** the cross-arm primary verdict SHALL be STOP
- **AND** the aggregator SHALL emit `m613_null_finding_note.md` documenting that the substrate prior is inert under retraining

#### Scenario: STOP when tei_weights < weights_only (substrate interferes)

- **WHEN** the four paired-seed deltas are `[-0.05, -0.08, -0.04, -0.07]` (mean -0.06)
- **THEN** the cross-arm primary verdict SHALL be STOP
- **AND** the aggregator SHALL surface "substrate INTERFERES with M3" in `summary.md` (a useful negative result distinct from the inert-substrate STOP)

### Requirement: M6.13 Pre-Declared Pilot Pivot Table

When the M6.13 aggregator runs in `--mode pilot`, it SHALL emit `pilot_pivot_decision.md` classifying the observed pilot outcome against six pre-declared pivot rows (binding BEFORE the full campaign is unblocked). The six rows SHALL be:

| Pilot observation | Pre-declared pivot |
|---|---|
| `tei_weights F1+ ≈ weights_only F1+` (abs delta < 2pp) | STOP — substrate prior inert under retraining. M6.13 hypothesis falsified. |
| `tei_weights F1+ > weights_only F1+` by ≥ 5pp at K_test | GO — full campaign at K_test only. |
| `tei_weights F1+ > weights_only F1+` by 2-5pp at K_test | K-sensitivity pilot: rerun at K=500 and K=1500 to map dose-response. +6 wall-h cap. |
| `tei_weights F1+ < weights_only F1+` (Δ < -2pp) | STOP — substrate INTERFERES with M3. Document; future-work substrate-policy alignment. |
| `weights_only F0 ≈ F1` (M3-saturation at K_test) | PIVOT — K_test too large. Drop to K=500 and rerun pilot. +3 wall-h cap. |
| `tei_weights F1` collapses to ~0 (< 0.05) | STOP — substrate destabilises PPO update. Investigate clamp / freeze-during-update as future work. |

The pivot classification SHALL be deterministic — exactly one row SHALL match any given pilot outcome. The aggregator SHALL emit the matching row's recommended action (STOP / GO / K-sensitivity / PIVOT) verbatim into `pilot_pivot_decision.md`.

#### Scenario: pilot pivot table classifies substrate-inert outcome correctly

- **WHEN** the pilot retention table shows `tei_weights` F1-F3 mean retention `0.66` and `weights_only` F1-F3 mean retention `0.67` (Δ ≈ -1pp)
- **AND** the aggregator runs in `--mode pilot`
- **THEN** `pilot_pivot_decision.md` SHALL classify this as row 1 (substrate-inert) and recommend STOP
- **AND** the `summary.md` SHALL cross-reference the pivot decision

#### Scenario: pilot pivot table classifies K-sensitivity outcome correctly

- **WHEN** the pilot retention table shows `tei_weights` Δ vs `weights_only` of +3.5pp at K_test
- **AND** the aggregator runs in `--mode pilot`
- **THEN** `pilot_pivot_decision.md` SHALL classify this as row 3 (K-sensitivity pivot) and recommend rerun at K=500 and K=1500

#### Scenario: pilot pivot table classifies clear-GO outcome correctly

- **WHEN** the pilot retention table shows `tei_weights` Δ vs `weights_only` of +7pp at K_test
- **AND** the aggregator runs in `--mode pilot`
- **THEN** `pilot_pivot_decision.md` SHALL classify this as row 2 (clear GO) and recommend full campaign at K_test only

### Requirement: M3-Headroom Tripwire (T3')

The pre-flight calibration smoke SHALL include a new M3-headroom tripwire T3' that confirms the chosen K_test sits below M3 saturation AND above the from-scratch control floor. T3' SHALL evaluate two conditions on the smoke-pass data:

- `weights_only F1+ ≤ 0.95 × weights_only F0` (M3 has headroom — F1+ hasn't ceiling'd at K_test)
- `weights_only F1+ ≥ 1.2 × control F1+` (M3 is doing useful work — F1+ retention exceeds the from-scratch floor by 20%)

Both conditions MUST pass before the M6.13 pilot is unblocked. If T3' fails on the high end (`F1+ > 0.95 × F0`), the calibration smoke SHALL drop K_test to a smaller value (per the design.md § D5 procedure) and rerun. If T3' fails on the low end (`F1+ < 1.2 × control F1+`), K_test is too small for M3 to do useful work and SHALL be bumped.

The existing M6.9+ tripwires T1, T2, T4 (F0 envelope, substrate diversity, substrate magnitude) SHALL continue to apply unchanged.

#### Scenario: T3' passes when K_test sits in the headroom band

- **GIVEN** a smoke-pass result with `weights_only F0 = 0.60`, `weights_only F1 = 0.50`, `control F1 = 0.30`
- **WHEN** T3' is evaluated
- **THEN** the headroom check `0.50 ≤ 0.95 × 0.60 = 0.57` SHALL pass
- **AND** the floor check `0.50 ≥ 1.2 × 0.30 = 0.36` SHALL pass
- **AND** T3' SHALL emit a PASS verdict; the pilot SHALL be unblocked

#### Scenario: T3' fails when K_test is at M3 saturation

- **GIVEN** a smoke-pass result with `weights_only F0 = 0.60`, `weights_only F1 = 0.59`, `control F1 = 0.30`
- **WHEN** T3' is evaluated
- **THEN** the headroom check `0.59 ≤ 0.95 × 0.60 = 0.57` SHALL fail
- **AND** T3' SHALL emit a FAIL verdict with a recommendation to drop K_test toward 500
- **AND** the pilot SHALL NOT be unblocked

#### Scenario: T3' fails when K_test is too small for M3

- **GIVEN** a smoke-pass result with `weights_only F0 = 0.60`, `weights_only F1 = 0.30`, `control F1 = 0.28`
- **WHEN** T3' is evaluated
- **THEN** the floor check `0.30 ≥ 1.2 × 0.28 = 0.336` SHALL fail
- **AND** T3' SHALL emit a FAIL verdict with a recommendation to bump K_test toward 1500
- **AND** the pilot SHALL NOT be unblocked

## MODIFIED Requirements

### Requirement: Inheritance Strategy

The system SHALL provide an `InheritanceStrategy` Protocol in `quantumnematode/evolution/inheritance.py` with four methods (`select_parents`, `assign_parent`, `checkpoint_path`, `kind`) and FIVE concrete implementations: `NoInheritance` (the default no-op), `LamarckianInheritance(elite_count: int = 1)` (per-genome weight inheritance), `BaldwinInheritance` (per-genome trait inheritance — recorded in lineage, no weight checkpoints written), `TransgenerationalInheritance` (per-genome substrate inheritance — F0-extracted bias-network MLP threaded through F1+ via `brain.tei_prior`), and `LamarckianTransgenerationalInheritance(elite_count: int = 1)` (composed weights + substrate inheritance — M3 warm-start path + M6.9+ substrate-flow path running in parallel for every F1+ child).

The `kind()` method SHALL return one of FIVE string literals so the loop can branch on intent rather than `isinstance` checks: `"none"` (no inheritance configured — `NoInheritance`), `"weights"` (per-genome trained-weight checkpoints flow between generations — `LamarckianInheritance`), `"trait"` (per-genome elite-parent ID flows in lineage but no weight checkpoints are captured — `BaldwinInheritance`), `"transgenerational"` (F0-extracted substrate flows in lineage AND is threaded into F1+ workers; per-genome weight checkpoints NOT captured — `TransgenerationalInheritance`), or `"weights+transgenerational"` (BOTH weight checkpoints AND substrate flow in parallel — `LamarckianTransgenerationalInheritance`). The loop SHALL gate weight-IO code paths (capture, GC, warm-start) on `kind() in {"weights", "weights+transgenerational"}` and SHALL gate F0-substrate-extraction + F1+ `tei_prior_source` plumbing on `kind() in {"transgenerational", "weights+transgenerational"}`. The loop SHALL gate elite-ID lineage tracking on `kind() != "none"`.

When `kind() in {"weights", "weights+transgenerational"}` is active, the `EvolutionLoop` SHALL: (1) capture each genome's post-K-train brain weights to a per-genome `.pt` file via `LearnedPerformanceFitness`'s `weight_capture_path` kwarg; (2) after each generation completes its `optimizer.tell` call, ask the strategy to select parents from the prior generation's `(genome, fitness)` pairs; (3) before the next generation's fitness evaluation, warm-start each child's brain from its assigned parent's checkpoint via `LearnedPerformanceFitness`'s `warm_start_path_override` kwarg; (4) garbage-collect any per-genome checkpoint that is not selected as a parent for the next generation. Disk usage retained **after** step (4)'s GC pass SHALL be at most `2 * inheritance_elite_count` `.pt` files (the surviving parents from the just-finished generation plus the about-to-evaluate generation's surviving parent). The transient peak between step (1) and step (4) reaches `population_size` files for the in-flight generation; that's expected and is not bounded by this rule.

The `EvolutionLoop` SHALL persist the resolved `inheritance` value (the literal string, not the strategy instance) and the selected parent IDs in its checkpoint pickle dict, so that resume-time validation can reject mismatched inheritance settings (see "Resume rejects mismatched inheritance setting" scenario below).

The strategy SHALL be selectable via `evolution.inheritance: Literal["none", "lamarckian", "baldwin", "transgenerational", "weights+transgenerational"]` in YAML and overridable via the `--inheritance` CLI flag on `scripts/run_evolution.py`. The `evolution.inheritance_elite_count` field is structurally `int >= 1` (default 1) but the validator SHALL reject any value other than 1 when `inheritance: lamarckian` OR `inheritance: weights+transgenerational` (both use single-elite-broadcast only — multi-elite parent selection is reserved for future strategies). The `inheritance_elite_count` field is unused under `inheritance: baldwin` (Baldwin is conceptually single-elite by construction). When `inheritance: none` (the default), the loop, fitness, and lineage code paths SHALL be byte-equivalent to a frozen-weight evolution baseline.

#### Scenario: Lamarckian inheritance is selectable via config and CLI

- **GIVEN** an `EvolutionConfig` with `inheritance: lamarckian` and `inheritance_elite_count: 1`
- **WHEN** the loop runs for ≥ 2 generations
- **THEN** every child genome of generation N (for N ≥ 1) SHALL have its brain loaded with the previous generation's selected-parent weights via `load_weights` BEFORE the K train phase begins
- **AND** the parent SHALL be the prior generation's single highest-fitness genome (broken by genome ID lexicographic order on ties)
- **AND** the CLI flag `--inheritance lamarckian` SHALL override the YAML field with the same behaviour
- **AND** `--inheritance none` SHALL force the no-op path even when the YAML sets `lamarckian`

#### Scenario: Composed inheritance is selectable via config and CLI

- **GIVEN** an `EvolutionConfig` with `inheritance: weights+transgenerational`, `inheritance_elite_count: 1`, and `transgenerational.enabled: true`
- **WHEN** the loop runs for ≥ 2 generations
- **THEN** every F1+ child genome SHALL have BOTH (a) its brain loaded with the F0 elite's weights via `load_weights` AND (b) the F0-extracted substrate threaded into `brain.tei_prior` via `tei_prior_source`, in that order, BEFORE the K_test train phase begins
- **AND** the CLI flag `--inheritance weights+transgenerational` SHALL override the YAML field with the same behaviour

#### Scenario: kind() Protocol method gates loop behaviour (widened)

- **GIVEN** an `InheritanceStrategy` instance
- **WHEN** the loop calls `strategy.kind()`
- **THEN** the return value SHALL be exactly one of `"none"`, `"weights"`, `"trait"`, `"transgenerational"`, or `"weights+transgenerational"`
- **AND** `NoInheritance.kind()` SHALL return `"none"`
- **AND** `LamarckianInheritance.kind()` SHALL return `"weights"`
- **AND** `BaldwinInheritance.kind()` SHALL return `"trait"`
- **AND** `TransgenerationalInheritance.kind()` SHALL return `"transgenerational"`
- **AND** `LamarckianTransgenerationalInheritance.kind()` SHALL return `"weights+transgenerational"`
- **AND** the loop's `_inheritance_active()` helper (which decides whether to compute weight checkpoint paths and run the GC step) SHALL evaluate `strategy.kind() in {"weights", "weights+transgenerational"}`
- **AND** the loop's `_substrate_inheritance_active()` helper SHALL evaluate `strategy.kind() in {"transgenerational", "weights+transgenerational"}`
- **AND** the loop's `_inheritance_records_lineage()` helper (which decides whether to write `inherited_from` in lineage rows AND whether to call `select_parents` to update `_selected_parent_ids`) SHALL evaluate `strategy.kind() != "none"`

#### Scenario: First generation runs from-scratch under any inheritance strategy

- **GIVEN** any inheritance config (`none` / `lamarckian` / `baldwin` / `transgenerational` / `weights+transgenerational`) with any `inheritance_elite_count`
- **WHEN** generation 0 evaluates
- **THEN** every gen-0 child SHALL be from-scratch — `LearnedPerformanceFitness.evaluate` SHALL be invoked with `warm_start_path_override=None` AND `tei_prior_source=None` for every gen-0 genome
- **AND** the `lineage.csv` `inherited_from` column SHALL be empty for every gen-0 row regardless of strategy
- **AND** the gen-0 fitness scores SHALL be bit-for-bit identical across `inheritance` settings with the same seed (modulo the side-effect substrate extraction + `save_weights` writes that capture F0 artefacts for F1+; fitness arithmetic is unaffected)

#### Scenario: Per-genome weight checkpoints are captured and garbage-collected

- **GIVEN** a Lamarckian run with `inheritance_elite_count: 1`, `population_size: 12`, `generations: 5` (generations 0-4 inclusive)
- **WHEN** the run completes
- **THEN** during generation N's evaluation, exactly 12 `.pt` files SHALL have been written under `<output_dir>/inheritance/gen-{N:03d}/`
- **AND** after generation N's `optimizer.tell` returns and the strategy selects the next-generation parent, the GC step SHALL delete all 11 non-selected files in `gen-{N:03d}/`; additionally when N ≥ 1 it SHALL delete all remaining files in `gen-{N-1:03d}/` (whose children have just finished evaluating, so those parent checkpoints are no longer needed). For N = 0 the second clause no-ops because no `gen-{-1}/` directory exists.
- **AND** at the moment generation N+1's evaluation begins, exactly one file SHALL exist in `gen-{N:03d}/` (the selected parent for the about-to-evaluate children)
- **AND** when the run completes after generation 4 (the final generation), the loop SHALL still run `select_parents` on gen 4's results and the GC step SHALL still delete gen 3's surviving parent — so the only surviving file SHALL be the selected parent of the final generation, under `inheritance/gen-004/`. This file is intentionally NOT deleted by GC.
- **AND** if the run terminates via `early_stop_on_saturation`, the same invariant holds: the loop runs `select_parents` + GC for the final-evaluated generation BEFORE the early-stop break.

#### Scenario: Composed mode F0 .pt GC coordination

- **GIVEN** a composed-mode run with `population_size: 8` and `inheritance: weights+transgenerational`
- **WHEN** gen 0 completes
- **THEN** the main-loop Lamarckian GC pass SHALL fire FIRST (running at end of gen 0 after `select_parents`, BEFORE F0 substrate extraction) — `_inheritance_active()` widens to `True` under composed mode — and SHALL keep only the F0 elite's `.pt` in `gen-000/` (deleting the other 7)
- **AND** the F0 substrate-extraction pipeline SHALL THEN run, loading the surviving elite `.pt` and writing the `.tei.pt` substrate
- **AND** the F0 substrate-extraction pipeline's internal GC pass SHALL be SUPPRESSED under composed mode (the kind-conditional `_combined_inheritance_active()` predicate returns `True`, skipping the internal GC) so the elite's `.pt` is NOT deleted by it
- **AND** the net effect at the moment gen 1 begins SHALL be IDENTICAL to a pure-M3 run: exactly one F0 `.pt` survives, indexed by the elite's genome_id, alongside the `.tei.pt` substrate
- **AND** under non-composed `inheritance: transgenerational` mode, the main-loop GC SHALL NOT fire (`_inheritance_active()` is False for that kind) but the inline GC inside `_run_f0_substrate_extraction` SHALL still fire — the inline GC remains load-bearing for the pure-TEI cleanup

#### Scenario: Inheritance requires a training phase

- **GIVEN** a YAML with `evolution.inheritance` set to `lamarckian` / `baldwin` / `transgenerational` / `weights+transgenerational` AND `evolution.learn_episodes_per_eval: 0`
- **WHEN** the YAML is loaded via `load_simulation_config`
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL state that inheritance requires a non-zero train phase
- **AND** the message SHALL point the user to either set `learn_episodes_per_eval > 0` or set `inheritance: none`

#### Scenario: Composed mode requires F1+ retraining

- **GIVEN** a YAML with `evolution.inheritance: weights+transgenerational` AND a `lawn_schedule` entry where `generation > 0` and `ppo_train_episodes: 0`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError` naming the offending entry
- **AND** the error message SHALL state that composed mode REQUIRES F1+ retraining (the prior must act on actual training to be a prior) and that `ppo_train_episodes: 0` at F1+ is reserved for pure-TEI (`inheritance: transgenerational`)
- **AND** the message SHALL point the user to either set `ppo_train_episodes > 0` for every F1+ entry under composed mode, or set `inheritance: transgenerational` for the pure-TEI K=0 arm

#### Scenario: Inheritance is mutually exclusive with static warm-start

- **GIVEN** a YAML with `evolution.inheritance` set to any non-`none` value AND `evolution.warm_start_path: /some/path.pt`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL state that `warm_start_path` (run-wide static checkpoint) and `inheritance` (per-genome dynamic checkpoint) both load weights into the same brain slot before the K train phase, and that exactly one MAY be set
- **AND** the message SHALL point the user to drop one of the two

#### Scenario: Inheritance requires hyperparameter encoding

- **GIVEN** a YAML with `evolution.inheritance` set to any non-`none` value AND `hyperparam_schema is None`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError`
- **AND** the error message SHALL state that inheritance over weight encoders would double-count weights as both genome and substrate
- **AND** the message SHALL point the user to either drop `inheritance` or add a `hyperparam_schema`

#### Scenario: Lamarckian/composed inheritance incompatible with architecture-changing schema entries

- **GIVEN** a YAML with `evolution.inheritance: lamarckian` OR `evolution.inheritance: weights+transgenerational` AND a `hyperparam_schema` containing at least one entry whose `name` references a brain-config field that changes tensor shapes (e.g. `actor_hidden_dim`, `lstm_hidden_dim`, `rnn_type`, `actor_num_layers`)
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError` naming the offending fields
- **AND** the error SHALL share the same `_ARCHITECTURE_CHANGING_FIELDS` denylist that the existing static warm-start uses (single source of truth)
- **AND** the message SHALL explain that per-genome checkpoints cannot be loaded into a child whose architecture differs from the parent's
- **AND** this rejection SHALL apply to `lamarckian` AND `weights+transgenerational` (both use the warm-start path); `baldwin` and `transgenerational` do NOT load weights and therefore SHALL accept architecture-changing schema entries

#### Scenario: Multi-elite inheritance is rejected for Lamarckian and composed modes

- **GIVEN** a YAML with `evolution.inheritance: lamarckian` OR `evolution.inheritance: weights+transgenerational` AND `evolution.inheritance_elite_count: 2`
- **WHEN** the YAML is loaded
- **THEN** loading SHALL raise a Pydantic `ValidationError` stating that `inheritance_elite_count` MUST be 1 under both modes
- **AND** the message SHALL state that multi-elite parent selection is not currently supported and that the field exists structurally so future strategies can populate it
- **AND** the `Field(default=1, ge=1)` constraint SHALL still permit values >1 in the schema; the rejection is enforced by the model validator only for the affected inheritance values
- **AND** the rule SHALL NOT apply when `inheritance: baldwin` or `inheritance: transgenerational` — both are conceptually single-elite by construction
- **AND** a separate validator SHALL also reject `inheritance_elite_count > population_size` with a distinct error — strategy-independent

#### Scenario: Inheritance defaults preserve frozen-weight evolution byte-equivalently

- **GIVEN** any existing evolution YAML with no `inheritance:` key under `evolution:`
- **WHEN** loaded and run via `scripts/run_evolution.py`
- **THEN** `EvolutionConfig.inheritance` SHALL be `"none"` and `EvolutionConfig.inheritance_elite_count` SHALL be `1`
- **AND** the loop SHALL construct a `NoInheritance` strategy whose `kind()` returns `"none"`
- **AND** no `inheritance/` directory SHALL be created under the output directory
- **AND** `LearnedPerformanceFitness.evaluate` SHALL be invoked with `warm_start_path_override=None` and `weight_capture_path=None` for every genome
- **AND** the `lineage.csv` `inherited_from` column SHALL be empty for every row

#### Scenario: Resume from checkpoint preserves selected parent IDs

- **GIVEN** a Lamarckian or composed run that wrote a checkpoint at end of generation N with `_selected_parent_ids = [pid_a]`
- **WHEN** the run resumes via `--resume <checkpoint>`
- **THEN** the loaded loop's `_selected_parent_ids` SHALL equal `[pid_a]`
- **AND** generation N+1's children SHALL warm-start from `inheritance/gen-{N:03d}/genome-{pid_a}.pt`
- **AND** if that file is unexpectedly missing on resume, the affected children SHALL fall back to from-scratch with a `logger.warning` (the loop SHALL NOT crash)

#### Scenario: Checkpoint version compatibility

- **GIVEN** a checkpoint pickle file whose `checkpoint_version` field is older than the current `CHECKPOINT_VERSION`
- **WHEN** the loop attempts to load it via `--resume`
- **THEN** loading SHALL raise the existing version-mismatch error
- **AND** the user SHALL be advised to start the run fresh (no automated converter is provided)

#### Scenario: Resume rejects mismatched inheritance setting

- **GIVEN** a checkpoint produced under one inheritance setting AND a resume invocation whose resolved `EvolutionConfig.inheritance` is different
- **WHEN** the loop attempts to resume
- **THEN** loading SHALL raise a clear error stating that the resumed run's inheritance setting differs from the original and that mid-run inheritance changes are not supported
- **AND** the message SHALL list both the checkpoint's recorded inheritance and the resolved current value
- **AND** the rejection SHALL fire BEFORE the loop reaches the first generation iteration
- **AND** this rejection SHALL apply to `--resume` invocations only — for fresh runs, `--inheritance` overrides the YAML field normally

#### Scenario: CLI rejects inheritance + --fitness success_rate at startup

- **GIVEN** a YAML or CLI invocation with `evolution.inheritance != "none"` AND `--fitness success_rate`
- **WHEN** `scripts/run_evolution.py` parses arguments and resolves the `EvolutionConfig`
- **THEN** the script SHALL exit with code 1 BEFORE constructing the optimizer or the loop
- **AND** the error message SHALL state that inheritance writes per-genome weight checkpoints or records elite-parent lineage from a trained-elite-fitness signal after each train phase, and `EpisodicSuccessRate` is frozen-weight with no train phase
- **AND** the message SHALL point the user to `--fitness learned_performance` or to setting `inheritance: none`
