# Design: TEI-as-Prior-on-M3 (M6.13)

## Context

PR-A (M6.9+) merged on 2026-05-19 with two verdicts: STOP ⛔ on pure-TEI K=0 (cross-arm `tei_on − control` delta = −49pp across three substrate variants) and SHIPPED ✅ on the framework + M3-on-the-new-env validation (+17.5pp delta vs from-scratch control, per the aggregator's `cross_arm_verdict.csv`). The 2025 wet-lab TEI literature (Kaletsky eLife 105673; mammalian-TEI 2025) clarified that biological TEI is a *low-bandwidth single-circuit excitability prior*, not a *transmitted policy*. PR-A's K=0 framing was structurally wrong; the wet-lab-aligned question was never tested. Logbook 019 § Follow-ups pre-registered this as the M6.13 reframe.

M6.13 asks: **does a small inherited substrate prior, combined with M3-pattern weight inheritance + K>0 retraining, accelerate F1+ learning vs M3 alone?**

The campaign reuses PR-A's bias-network substrate, M6.10 env (15×15 / 4 predators / damage=25 / `min_food_predator_distance=4`), `gradient_proximity` reward mode, LSTMPPO `tei_prior` hook, aggregator scaffolding, and tripwire infrastructure. No new substrate code; no env/reward/brain changes. The only production-code surface is **inheritance composition** — a new strategy class + widened predicates + relaxed validator.

**Constraints (rationale)**:

1. *Reuse, don't refactor*. PR-A's framework is the validated foundation; M6.13's question is "does composing existing pieces produce a new signal." Adding a new strategy class beside the existing four keeps the M3 and M6.9+ pure-TEI paths byte-equivalent (regression tests gate this).
2. *Composition not subclassing*. The new `LamarckianTransgenerationalInheritance` is its own class (not a subclass of either parent strategy). Composition by inheritance/mixin would conflate the strategies; both `LamarckianInheritance` and `TransgenerationalInheritance` must remain auditable in isolation for the M3 + M6.9+ regression tests to be meaningful.
3. *K_test calibration is load-bearing*. The hypothesis is "substrate prior accelerates K>0 retraining" — true acceleration only manifests below M3's saturation point. T3' tripwire enforces this BEFORE pilot is unblocked.
4. *Pre-declared pivots prevent post-hoc audit*. M6's failure mode was running the full campaign then auditing; PR-A's six-row D6 table caught D6 row 2 (substrate inert) cleanly. M6.13's pivot table mirrors that pattern.

**Stakeholders**: Phase 5 tracker M6.13 line; logbook 020 (this work); roadmap M6.9+/M6.13 row; downstream M6.14 (frequency-prior ablation, gated on M6.13's outcome).

## Goals / Non-Goals

**Goals**

- Compose the M3 (Lamarckian) weight-inheritance path with the M6.9+ substrate-flow path so F1+ children warm-start from F0 weights AND apply the F0-extracted substrate as a logit prior during K_test retraining.
- Produce a defensible verdict on whether the substrate prior accelerates K_test retraining vs M3 alone, at a K_test value calibrated to sit below M3 saturation.
- Reuse the M6.9+ aggregator scaffolding with a reframed primary verdict (`tei_weights − weights_only`); preserve the n=4-noise-aware statistical machinery (Wilcoxon + bootstrap CI).
- A binding pilot pivot table that pre-declares STOP / GO / K-sensitivity-pivot / calibration-pivot outcomes against six observed patterns.

**Non-Goals**

- New substrate forms. The frequency-prior ablation is M6.14's question, gated on M6.13's outcome.
- Env / reward / brain redesign. M6.10 audit-corrected env stays.
- Substrate-clamp sensitivity sweep. Deferred — if M6.13 produces the row-4 outcome (substrate interferes), this becomes the M6.13a follow-up.
- Spec-sync of the modified-capability deltas to main specs at archive time. Per the PR-A / M4 / M6 INCONCLUSIVE precedent: delta requirements stay in the change archive until either M6.13 produces a GO (then sync the composed strategy) or M6.14+ supersedes the design.

## Decisions

### D1. New `LamarckianTransgenerationalInheritance` class — composition, not extension

A new class beside `LamarckianInheritance` and `TransgenerationalInheritance`. `kind() == "weights+transgenerational"` adds a fifth Literal value to the `InheritanceStrategy.kind()` Protocol contract. Per-method behaviour:

- `select_parents(gen_ids, fitnesses, generation)` — IDENTICAL to `LamarckianInheritance.select_parents` with `elite_count=1` (top-fitness, lex-tie-broken on genome_id). Multi-elite extension reserved for future work, same validator as Lamarckian.
- `assign_parent(child_index, parent_ids)` — IDENTICAL to `LamarckianInheritance.assign_parent` (round-robin; with single elite, every child broadcasts from the same parent — M3 pattern).
- `checkpoint_path(output_dir, generation, genome_id)` — IDENTICAL to `LamarckianInheritance.checkpoint_path`: returns `output_dir / "inheritance" / f"gen-{generation:03d}" / f"genome-{genome_id}.pt"` (canonical `.pt`, NOT `.tei.pt`). The substrate path is OWNED by the F0 substrate-extraction pipeline (separate concern; same path-builder pattern already in place from PR-A).
- `kind()` — returns `"weights+transgenerational"`.

**Why composition class, not extension**: `TransgenerationalInheritance.assign_parent` returns `None` (no weight-IO). Changing it would break the M6.9+ pure-TEI arm (no warm-start in that flow). Mixin/inheritance would conflate strategies. A composed class makes the M6.13 path orthogonal: M3 and M6.9+ pure-TEI regression tests gate byte-equivalence on those paths; the new composed path has its own dedicated tests.

**Frozen-substrate invariant under retraining**: PR-A's brain rollout-buffer assertion (`_check_tei_prior_snapshot` in `lstmppo.py`) already validates the substrate does not mutate mid-window. Under composed mode the substrate's `bias_network` parameters are `requires_grad=False` (per PR-A) — they are NEVER trained; the M3 PPO update touches only `brain.state_dict()`, leaving `brain.tei_prior` (which holds the substrate object reference) untouched. This is the load-bearing invariant for M6.13.

### D2. Validator relaxation — pairing contract widening

Three validator changes in `_validate_inheritance` on `EvolutionConfig`:

1. **Widen `EvolutionConfig.inheritance` Literal** from `Literal["none", "lamarckian", "baldwin", "transgenerational"]` to `Literal["none", "lamarckian", "baldwin", "transgenerational", "weights+transgenerational"]`.
2. **Relax substrate-enabled pairing**: `transgenerational.enabled=True` was valid ONLY with `inheritance == "transgenerational"`. Widen to: VALID with EITHER `transgenerational` OR `weights+transgenerational`. The `transgenerational.enabled=False` case (PR-A's three-arm parity) is unchanged — still requires `inheritance == "none"`.
3. **Add F1+ K>0 sub-rule under composed mode**: `inheritance == "weights+transgenerational"` requires `lawn_schedule.ppo_train_episodes > 0` for every entry where `generation > 0`. This is the *opposite* of pure-TEI (which uses K=0 to test the floor); composed mode REQUIRES retraining for the prior to act as a prior.

**Cross-product validation matrix** (positive cells; everything else MUST raise):

| inheritance | transgenerational.enabled |
|---|---|
| `none` | False / None |
| `lamarckian` | None |
| `baldwin` | None |
| `transgenerational` | True |
| **`weights+transgenerational`** | **True** ← NEW |

The validator's existing single-elite rule (`elite_count != 1` rejects when `lamarckian`) extends to also reject under `weights+transgenerational`. The existing architecture-changing-fields rule (`actor_hidden_dim`, `lstm_hidden_dim`, etc. cannot be in `hyperparam_schema` under Lamarckian) also applies under composed mode (the warm-start path is the same constraint).

### D3. Loop integration — predicate widening + GC suppression

Three loop changes:

1. **`_expected_kind` dict** in `EvolutionLoop.__init__` gains `"weights+transgenerational": "weights+transgenerational"` entry. The kind-mismatch defensive check in `__init__` then validates the strategy instance returns the expected kind for the config value.
2. **`_inheritance_active()` predicate** widens to `kind() in {"weights", "weights+transgenerational"}`. **`_substrate_inheritance_active()` predicate** widens to `kind() in {"transgenerational", "weights+transgenerational"}`. The fact that BOTH predicates fire under composed mode is what makes the existing loop machinery do the right thing for free: F1+ workers receive both `warm_start_path_override` (from the weight-IO path) AND `tei_prior_source` (from the substrate-flow path) without any new plumbing.
3. **New branch in `_resolve_per_child_inheritance`** for `kind == "weights+transgenerational"` — functionally IDENTICAL to the existing `"weights"` branch (same `checkpoint_path` calls, same `assign_parent` resolution, same warm-start path lookup, same warning-on-missing-parent fallback). The branch exists for switch exhaustiveness; future divergence in the composed semantics would happen here.
4. **GC suppression in F0 substrate-extraction pipeline**. The actual gen-0 GC ordering is: (a) main-loop GC at `loop.py:~1649` runs FIRST (immediately after `select_parents`); when `_inheritance_active()` is True it keeps only `next_selected` in `gen-000/` — under composed mode this widens to True and keeps the elite's `.pt`, deleting the other 7. (b) F0 substrate extraction at `loop.py:~1705` runs SECOND, loading the elite's surviving `.pt`, writing `.tei.pt`, then the inline GC at `loop.py:~863-866` deletes every remaining `.pt`. Step (b)'s inline GC is what must be suppressed under composed mode — without the skip, the elite's `.pt` (kept by step (a)) gets deleted before F1 children can read it. **Fix**: kind-conditional guard around `loop.py:~863-866`: `if not self._combined_inheritance_active(): <inline GC loop>`. Under pure-TEI (`kind() == "transgenerational"`), step (a)'s main-loop GC does NOT fire (`_inheritance_active()` is False), so step (b)'s inline GC remains load-bearing for pure-TEI — the skip is composed-mode only.

### D4. Three-arm campaign + reframed verdict

| Arm | inheritance | F1+ K | F1+ inheritance | What it tests |
|---|---|---|---|---|
| `tei_weights` | `weights+transgenerational` | K_test | F0 weights + decayed substrate prior | M6.13 hypothesis |
| `weights_only` | `lamarckian` | K_test | F0 weights only | M3 baseline at K_test |
| `control` | `none` | K_test | nothing | Environmental floor at K_test |

**Primary verdict** (reframed from PR-A): GO iff (a) `tei_weights` passes the per-arm gate (F1≥40%, F2≥25%, F3≥15% of F0 training-time fitness; monotone non-increasing in ≥2/4 seeds) AND (b) `tei_weights − weights_only` paired-seed F1-F3 retention delta satisfies BOTH one-sided Wilcoxon p\<0.10 AND ≥5pp absolute mean delta with non-overlapping 80% bootstrap CIs.

**Secondary verdicts**: `weights_only − control` (M3 re-reproduction at K_test; should match PR-A's +17.5pp scaled to K_test); `tei_weights − control` (composed-arm vs floor; weaker claim than the primary).

**Three-arm YAML structural pairing**: same omit-the-block contract as PR-A. `weights_only` and `control` arms MUST omit `transgenerational:` entirely (PR-A's `_validate_inheritance` rule rejects `lamarckian + transgenerational.enabled=False`). `tei_weights` arm includes the full `transgenerational:` block with `enabled: true` (same content as PR-A's `tei_on` arm).

### D5. K_test calibration — one-knob smoke chain

**Goal**: pick the smallest K where `weights_only F1+` shows learning but has not yet ceiling'd.

**Procedure** (capped at 2 passes):

1. Pass 1: `weights_only` smoke at K=1000, 1 seed × pop 6 × 2 gens.
   - F1 ≥ 0.95 × F0 → ceiling'd; drop to K=500. Pass 2.
   - F0 < F1 < F0 + 0.05 OR `F1 > F0 + 0.05` → headroom; K=1000 is the test point. STOP.
   - F1 < 0.80 × F0 → too low; bump to K=1500. Pass 2.
2. Pass 2: re-run at adjusted K. Same decision rule. If neither pass lands a clean envelope, the PR-A env doesn't support M6.13 testing and this is a calibration-null verdict.

**Tripwires at smoke completion** (mirror PR-A's T1-T4):

| # | Tripwire | Pass criterion |
|--:|---|---|
| T1' | F0 envelope | `0.30 ≤ mean F0 survival_rate ≤ 0.70` (already validated by PR-A's pass-6; re-verify on composed-arm config) |
| T2' | Substrate diversity | Pairwise CoV > 5% across calibration seeds' `bias_network` state_dicts (PR-A T2 — re-run at chosen K_test) |
| **T3'** | **M3-headroom (NEW)** | `weights_only F1+ ≤ 0.95 × F0` AND `weights_only F1+ ≥ 1.2 × control F1+` |
| T4' | Substrate magnitude | Mean abs bias_network output > 0.1 (PR-A T4) |

T3' is the load-bearing M6.13 tripwire — it operationalises "K_test sits below M3 saturation AND above the control floor."

### D6. Pre-declared pilot pivot table (binding BEFORE full campaign)

| Pilot observation | Pre-declared pivot |
|---|---|
| `tei_weights F1+ ≈ weights_only F1+` (Δ < 2pp) | STOP — substrate prior inert under retraining. M6.13 hypothesis falsified. Logbook 020 documents the null. |
| `tei_weights F1+ > weights_only F1+` by ≥ 5pp at K_test | GO — full campaign at K_test only. |
| `tei_weights F1+ > weights_only F1+` by 2-5pp at K_test | K-sensitivity pilot: rerun at K=500 and K=1500 to map dose-response. Decide GO/STOP at the K with largest Δ. +6 wall-h cap. |
| `tei_weights F1+ < weights_only F1+` | STOP — substrate INTERFERES with M3. Document; recommends substrate-policy alignment as a follow-up. |
| `weights_only F0 ≈ F1` (M3-saturation at K_test) | PIVOT — K_test too large. Drop to K=500 and rerun pilot. +3 wall-h cap. |
| `tei_weights F1` collapses to ~0 | STOP — substrate destabilises PPO update. Investigate clamp / freeze-during-update as future work. |

### D7. Compute envelope

| Layer | Compute | Wall-h |
|---|---|--:|
| Smoke unit tests | local | \<1 |
| K_test calibration smoke | 1-2 seeds × 1-2 passes | ~2-4 |
| Pilot | 1 seed × pop 8 × 4 gens × 3 arms | ~3 |
| Full campaign (if pilot GO) | 4 seeds × pop 16 × 4 gens × 3 arms | ~14-18 |
| Post-hoc evaluator + aggregator | analysis only | ~1 |
| **Total M6.13** | | **~20-25** |

Matches PR-A's footprint with headroom for the calibration pivot. M6.14 (frequency-prior ablation, gated on M6.13 outcome) would add ~12-15 wall-h.

## Risks / Trade-offs

| # | Risk | Mitigation |
|--:|---|---|
| R1 | Substrate cascade interferes with retraining gradient — frozen bias-network outputs bias PPO toward whatever the prior favours, regardless of whether that direction is correct for the warm-start child's policy. May surface as row-4 STOP even when substrate IS aligned. | Brain's `_check_tei_prior_snapshot` already validates substrate doesn't mutate mid-window (PR-A invariant). Document the frozen-substrate-during-PPO contract in code. Defer clamp-sensitivity sweep to M6.13a if row 4 fires — don't bundle. |
| R2 | F0 `.pt` GC coordination bug between substrate-extraction-pipeline internal GC and main-loop Lamarckian GC. | Loop-smoke test asserts F0 elite's `.pt` exists post-extraction; second test asserts main-loop GC fires at end of F0 select_parents (Lamarckian pattern). Reuse existing `keep_ids` parameter shape — no new GC code, only a kind-conditional skip in the extraction pipeline. |
| R3 | K_test calibration miscalibration — pilot reveals M3-saturation at chosen K. Forces calibration + pilot redo. | T3' tripwire at smoke catches this BEFORE pilot. Pivot table row 5 catches it post-pilot. Hard-cap at 1 calibration redo + 1 pilot redo (+9 wall-h overflow allowed). Beyond that: STOP and document. |
| R4 | Validator-relaxation accidentally permits an invalid pairing (e.g. `baldwin + transgenerational.enabled=True`). | Cross-product validator test grid enumerates all `(inheritance, transgenerational.enabled)` cells; asserts pass/reject set explicitly. PR-A's existing matrix extends with two new cells (`weights+transgenerational, True` accepts; `weights+transgenerational, False/None` rejects). |
| R5 | F1+ warm-start weight value-head bias differs sharply from F0 elite — substrate prior is calibrated for F0 elite's baseline, may be off for warm-start child. | Under single-elite-broadcast (the only configuration shipped), every F1 child warm-starts from the F0 elite whose substrate they're inheriting — calibration target matches warm-start source by construction. Loop-smoke test asserts warm-start path and substrate-source genome_id match under composed mode. |
| R6 | n=4 cross-arm verdict noise-dominated (same risk as PR-A R7). | Wilcoxon p\<0.10 AND non-overlapping 80% bootstrap CIs — both must agree on direction. Bare 5pp threshold explicitly rejected at n=4. |
| R7 | M6.13 GO at K_test but null at higher K (substrate prior only helps in the narrow under-saturation regime). | Documented as a feature, not a bug — K-sensitivity is part of the dose-response question. Pivot row 3 (2-5pp Δ at K_test) explicitly triggers the K-sweep variant. |

## Migration Plan

The change is **additive** — no breaking changes to existing inheritance modes. `EvolutionConfig.inheritance: "none"` (default) preserves byte-equivalence with all pre-M6.13 paths. M3 (`lamarckian`), Baldwin (`baldwin`), and M6.9+ pure-TEI (`transgenerational`) regression tests gate byte-equivalence.

**Rollout order (matches commit grouping)**:

1. Scaffold M6.13 strategy class + Protocol widening. Default `kind()` set unchanged for existing strategies; new kind added.
2. Relax validator pairing rules. Existing positive cells unchanged; one new positive cell added.
3. Loop integration — extend predicates + new branch + kind-conditional GC suppression. Existing modes unaffected by guard widening.
4. Three-arm YAML configs + launcher shell + T3' tripwire.
5. Aggregator fork + M6.13 pivot table.
6. (Post-experiment) Logbook 020 + tracker tick + roadmap tick + archive.

**Rollback**: each commit is self-contained. If any downstream commit fails, the preceding state is byte-equivalent to pre-change `main` for all non-composed inheritance modes.

## Open Questions

None blocking implementation. Two honest gaps:

1. **K_test choice not biology-anchored.** Wet-lab TEI doesn't have an "episodes-of-retraining" knob. K_test is a project-internal calibration knob; T3' provides the empirical floor (M3 must show measurable learning but not saturate).
2. **Substrate-prior clamp value (6.0 from PR-A's pilot-3 lift) is the same under composed mode.** PR-A pilot-3 showed the clamp lift did NOT unlock signal at K=0 — but at K>0 the dynamics differ (PPO can consume the prior as initialization noise). Clamp sensitivity is the M6.13a follow-up if R1 fires.
