# Proposal: TEI-as-Prior-on-M3 (M6.13)

## Why

M6.9+ PR-A (logbook 019, merged 2026-05-19) decisively falsified the **pure-TEI K=0 hypothesis** (cross-arm `tei_on − control` delta = −49pp across three substrate variants) but produced two net-positive deliverables: (a) the sensory-conditional bias-network substrate framework, and (b) M3-on-the-new-env validation at **+17.5pp** delta vs from-scratch control (aggregator's `cross_arm_verdict.csv`). The 2025 wet-lab TEI literature (Kaletsky eLife 105673; mammalian-TEI 2025) clarifies that biological TEI is a **low-bandwidth single-circuit excitability prior**, not a transmitted policy — meaning the M6.9+ K=0 framing was structurally wrong and the wet-lab-aligned question was never tested.

Logbook 019 § Follow-ups pre-registered the reframe: does a small inherited substrate **prior**, combined with M3-pattern weight inheritance + K>0 retraining, **accelerate** F1+ learning vs M3 alone at K below saturation? **GO** would give the strongest scientific claim available — "substrate adds measurable value on top of trained weights" — matching the wet-lab mechanism. **STOP** would be a mechanistically-grounded null (PR-A's null floor + M6.13's null acceleration = "TEI doesn't transfer in either form on this RL substrate"), valuable for Phase 5 synthesis.

## What Changes

**Composed inheritance strategy (M6.13.1)**. New `LamarckianTransgenerationalInheritance` class (`evolution/lamarckian_transgenerational_inheritance.py`, ~120 LoC) implementing weight inheritance + substrate inheritance composition. `kind() == "weights+transgenerational"` adds a fifth Literal value to the existing four-way `InheritanceStrategy` dispatch. Per-method behaviour:

- `select_parents` / `assign_parent` / `checkpoint_path` — IDENTICAL to `LamarckianInheritance` (single-elite top-1, lex-tie-broken, round-robin per child, canonical `.pt` path).
- F1+ children warm-start from F0 elite's `.pt` (M3 path) AND the F0-extracted substrate flows in via `tei_prior_source` (PR-A path).

A composed class — not extension of `TransgenerationalInheritance` — keeps the two parent strategies auditable in isolation; the existing pure-TEI path (`TransgenerationalInheritance.assign_parent` returns `None`) MUST remain byte-equivalent to PR-A.

**Config validator relaxation (M6.13.2)**. Widen `EvolutionConfig.inheritance` Literal to include `"weights+transgenerational"`. Relax `_validate_inheritance` pairing rule: `transgenerational.enabled=True` now valid with EITHER `transgenerational` OR `weights+transgenerational` (was: only `transgenerational`). Add new sub-rule: under composed mode, `lawn_schedule.ppo_train_episodes` MUST be > 0 for every F1+ entry (opposite of pure-TEI which uses K=0; composed mode REQUIRES retraining for the prior to act as a prior).

**Loop integration (M6.13.3)**. Extend `_expected_kind` dict; widen `_inheritance_active()` and `_substrate_inheritance_active()` predicates to include `"weights+transgenerational"`; add a fifth branch in `_resolve_per_child_inheritance` (functionally identical to the existing `"weights"` branch — same checkpoint path, warm-start resolution, parent-ID assignment); kind-conditional GC suppression in the F0 substrate-extraction pipeline so F0 `.pt` files survive for F1 warm-start (the main-loop Lamarckian GC then keeps only the elite, identical to M3 net effect).

**Three-arm campaign + reframed verdict (M6.13.4)**. New campaign `tei_prior_m613_{tei_weights, weights_only, control}` mirroring PR-A's structure. **Primary verdict reframed** to `tei_weights − weights_only` (was: `tei_on − control` in PR-A). Same statistical machinery: GO iff `tei_weights` passes per-arm gate AND paired-seed delta satisfies Wilcoxon p\<0.10 AND ≥5pp absolute delta with non-overlapping 80% bootstrap CIs. K_test calibrated to sit below M3 saturation (~2000) via a new T3' tripwire (`weights_only F1+ ≤ 0.95 × F0 AND ≥ 1.2 × control F1+`).

**Out of scope** (deferred to future OpenSpec changes):

- Frequency-prior ablation (~4-element vector substrate). Deferred to M6.14 if M6.13 produces a positive signal. Defers the substrate-richness ablation to a sequential follow-up rather than bundling.
- M7 NEAT closure of M5 (independent multi-week infra).
- M8 Phase 5 synthesis logbook (pure integration writing).
- Substrate redesign / env redesign / reward redesign. M6.13 is **composition-only**; reuses PR-A's bias-network MLP, M6.10 env (15×15 grid / 4 predators / damage=25 / `min_food_predator_distance=4`), `gradient_proximity` reward mode, and LSTMPPO `tei_prior` hook.

## Capabilities

### New Capabilities

None — M6.13 extends existing capabilities; no new spec files.

### Modified Capabilities

- `evolution-framework`: add composed `LamarckianTransgenerationalInheritance` strategy; widen `InheritanceStrategy.kind()` Protocol Literal to include `"weights+transgenerational"`; widen `_inheritance_active()` + `_substrate_inheritance_active()` predicates; add new branch in `_resolve_per_child_inheritance`; kind-conditional GC suppression in F0 substrate-extraction pipeline; reframe primary cross-arm verdict to `tei_weights − weights_only`; pre-declared M6.13 pilot pivot table. M3 / M6.9+ pure-TEI paths remain byte-equivalent.
- `configuration-system`: widen `EvolutionConfig.inheritance` Literal to include `"weights+transgenerational"`; relax `_validate_inheritance` pairing rule (substrate-enabled now accepts both `transgenerational` and `weights+transgenerational`); add F1+ K>0 sub-rule under composed mode; relax three-arm YAML structural pairing requirement (PR-A's pairing block extended to permit the composed arm).

## Impact

- **New code**: ~270 LoC across 1 new module (`lamarckian_transgenerational_inheritance.py`) + 4 new YAML configs + 1 new launcher shell + 1 new aggregator (`aggregate_m613_pilot.py`, ~1000 LoC, fork of M6.9+ aggregator).
- **Modified code**: 5 existing files — `evolution/inheritance.py` (+5; Protocol Literal widening), `utils/config_loader.py` (+60; Literal + validator), `evolution/loop.py` (+80 / −5; predicates + branch + GC suppression), `evolution/__init__.py` (+1; re-export), `scripts/run_evolution.py` (+6; factory).
- **No env / reward / brain code changes.** M6.13 is an inheritance-composition change only.
- **InheritanceStrategy Protocol**: `kind()` return-type Literal widens by one value. Backwards-compatible: all four existing kinds remain unchanged; the protocol's runtime-checkable contract is unaffected.
- **Existing inheritance modes** (`none`, `lamarckian`, `baldwin`, `transgenerational`): no behaviour change. The PR-A pure-TEI arm uses `kind() == "transgenerational"` (M6.9+ path); M6.13's composed arm uses the new `kind() == "weights+transgenerational"`.
- **Compute envelope**: K_test calibration smoke ~2-4 wall-h; pilot ~3 wall-h; full ~14-18 wall-h; post-hoc ~1 wall-h. Total M6.13 ≈ **~20-25 wall-h**. Matches PR-A footprint with headroom for the calibration pivot.
- **Decision-gate evaluation**: per-(arm, seed) survival_rate retention table + per-arm cross-seed verdict + reframed cross-arm primary verdict (`tei_weights − weights_only`). User-reviewed before logbook 020 publication (per `feedback_logbook_review_before_verdict.md`).
- **Tests**: +40 cases across composed-strategy unit (+16), loop-smoke (+6), config-validator cross-product (+4), aggregator (+14).
- **Risk**: low for framework code (additive composition; M3 and M6.9+ regression tests gate byte-equivalence on existing paths); calibration-tripwire-gated for compute spend.
