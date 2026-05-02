# Proposal: Baldwin-Effect Inheritance + Early-Stop-on-Saturation

## Why

Phase 5 M3 shipped Lamarckian inheritance (per-genome warm-start of trained weights) and proved it accelerates convergence by ~5.25 generations on the predator arm (logbook 013). Lamarckian is biologically inaccurate — acquired weights are not heritable in real organisms. The complementary biological mechanism is the **Baldwin Effect**: lifetime learning guides genetic evolution toward genomes that learn fast, *without* the genome ever encoding the learned policy. M4 demonstrates this on the same predator arm and tests whether the Baldwin signal is large enough on this codebase to be a credible substrate for M6 (transgenerational memory).

A second motivation: M3's lamarckian seeds saturated at fitness 1.00 by gen 3-7 and ran another 13-17 wasted generations. M4 runs ≥3 arms × 4 seeds × ~20 gens, so a `--early-stop-on-saturation N` flag would meaningfully cut wall-time. Bundling it with M4 keeps the loop changes coherent and avoids a separate one-flag PR.

## What Changes

- **Add `BaldwinInheritance` strategy** to `evolution/inheritance.py`: trait-only inheritance — no per-genome weight checkpoints, but the prior generation's elite genome ID is recorded as `inherited_from` for every child of the next generation, so the lineage CSV captures the evolutionary trace.
- **Extend `InheritanceStrategy` Protocol** with `kind() -> Literal["none", "weights", "trait"]` so the loop branches on intent rather than `isinstance` checks. `NoInheritance.kind() → "none"`, `LamarckianInheritance.kind() → "weights"`, `BaldwinInheritance.kind() → "trait"`. `_inheritance_active()` becomes `kind() == "weights"` so Baldwin no-ops on the weight-IO path while still recording lineage.
- **Add `weight_init_scale` LSTMPPO brain field** (NEW): scales `nn.Linear` and LSTM weight init std at construction. Default 1.0 (byte-equivalent to current), validator bounds `[0.1, 5.0]`. Genuine innate-bias knob the M3 control could not evolve.
- **Add `--early-stop-on-saturation N` loop flag**: `EvolutionConfig.early_stop_on_saturation: int | None = None`. When set, the loop exits if best_fitness has not improved for N consecutive generations. CLI override mirrors `--algorithm` / `--inheritance`. Persists `_gens_without_improvement` in the checkpoint pickle. Bumps `CHECKPOINT_VERSION` 2 → 3.
- **Extend `EvolutionConfig.inheritance` Literal** to `["none", "lamarckian", "baldwin"]`. Validator: Baldwin requires `learn_episodes_per_eval > 0`, `warm_start_path is None`, and `hyperparam_schema is not None` (same rules as Lamarckian). Baldwin does NOT require the architecture-changing-fields denylist (no per-genome weight load → shape mismatches are fine), but the M4 pilot keeps the same fixed architecture as M3 for fair comparison.
- **Extend `--inheritance` CLI choices** to include `"baldwin"`.
- **Ship Baldwin-rich pilot config** at `configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml`: 6-field schema (M3 control's actor_lr, critic_lr, gamma, entropy_coef + new `weight_init_scale` and `entropy_decay_episodes` — the latter already exists in the brain config and just needs schema exposure). Same brain/env/budget/seeds as M3 control, with `inheritance: baldwin` and `early_stop_on_saturation: 5`.
- **Ship 3 campaign scripts** (Baldwin pilot, Lamarckian rerun on M4 revision, M3-control rerun on M4 revision — last two are confounder-free reruns of M3's existing configs so the 3-arm comparison shares one code revision).
- **Ship F1 post-pilot evaluator** at `scripts/campaigns/baldwin_f1_postpilot_eval.py`: takes each Baldwin seed's gen-N elite `best_params.json`, runs L=25 frozen-eval (K=0) episodes, writes `f1_innate_only.csv`. Tests genetic assimilation: does the elite hyperparameter genome alone (without learning) beat random-init?
- **Ship 4-way aggregator** at `scripts/campaigns/aggregate_m4_pilot.py`: produces 4-curve plot, per-seed table, summary.md with three gates — speed (Baldwin vs control: +2 gens), genetic-assimilation (F1 vs hand-tuned baseline: +0.10pp), comparative (Baldwin vs Lamarckian: within +4 gens). GO if all three; PIVOT if speed only; STOP otherwise.
- **Publish logbook 014** at `docs/experiments/logbooks/014-baldwin-inheritance-pilot.md` with the pilot data, decision-gate verdict, and Baldwin-vs-Lamarckian discussion.

Not BREAKING. M3 callers continue to work — `inheritance: none` and `inheritance: lamarckian` paths are byte-equivalent to M3 except for the `kind()` Protocol method addition, which is a pure-additive Protocol extension.

## Capabilities

### New Capabilities

None — M4 extends the existing `evolution-framework` capability rather than introducing new ones.

### Modified Capabilities

- `evolution-framework`: extends the existing "Inheritance Strategy" requirement with Baldwin support (new `kind()` Protocol method, `BaldwinInheritance` implementation, Baldwin-specific validators, lineage semantics). Adds a sibling "Baldwin Inheritance Strategy" requirement covering Baldwin-specific scenarios (no `inheritance/` directory, trait-flow lineage, validator rules). Adds a new "Early Stop on Saturation" requirement covering the loop flag, config field, monitoring logic, and resume support. Updates the existing "Multi-elite inheritance is rejected" scenario to clarify the rule applies only to `inheritance: lamarckian`.

## Impact

**Code**:

- `packages/quantum-nematode/quantumnematode/evolution/inheritance.py`: +`BaldwinInheritance` class + `kind()` method on Protocol and all 3 impls.
- `packages/quantum-nematode/quantumnematode/evolution/loop.py`: refactor `_inheritance_active` to use `kind()`; add elite-ID tracking for Baldwin's `inherited_from`; add `_gens_without_improvement` counter + early-stop check; bump `CHECKPOINT_VERSION` 2 → 3 + persist new field.
- `packages/quantum-nematode/quantumnematode/utils/config_loader.py`: extend `inheritance` Literal; add `early_stop_on_saturation` field; extend `_validate_inheritance` for Baldwin rules.
- `packages/quantum-nematode/quantumnematode/brain/arch/_reservoir_lstm_base.py` (or wherever LSTMPPO config lives): add `weight_init_scale` field + apply at construction.
- `packages/quantum-nematode/quantumnematode/evolution/__init__.py`: re-export `BaldwinInheritance`.
- `scripts/run_evolution.py`: extend `--inheritance` choices; extend strategy construction; add `--early-stop-on-saturation` CLI.

**Configs**:

- NEW: `configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml`.

**Scripts**:

- NEW: `scripts/campaigns/phase5_m4_baldwin_lstmppo_klinotaxis_predator.sh`.
- NEW: `scripts/campaigns/phase5_m4_lamarckian_rerun.sh`.
- NEW: `scripts/campaigns/phase5_m4_control_rerun.sh`.
- NEW: `scripts/campaigns/baldwin_f1_postpilot_eval.py`.
- NEW: `scripts/campaigns/aggregate_m4_pilot.py`.

**Tests** (under `packages/quantum-nematode/tests/quantumnematode_tests/evolution/`):

- Modify `test_inheritance.py` — `BaldwinInheritance` unit tests + `kind()` for all 3 impls.
- Modify `test_loop_smoke.py` — Baldwin smoke + assert no `inheritance/` directory + `inherited_from` populated.
- Modify `test_config.py` — 3 Baldwin validator tests.
- NEW: `test_early_stop.py` — 4 cases (never improves / improves then stalls / monotonic improvement / resume preserves counter).
- NEW: `test_weight_init_scale.py` — brain construction with the field, std scales correctly, default 1.0 is no-op.

**Spec**:

- `openspec/specs/evolution-framework/spec.md`: 1 modified requirement + 2 new requirements + scenario clarification (synced from this change's deltas at archival time).

**Docs**:

- NEW logbook 014.
- Updated tracker (M4.1-M4.7 ticked, status flipped to `complete`).
- Updated roadmap (M4 row to ✅).

**External dependencies**: none. No new packages, no API changes, no breaking changes to existing scripts or configs.

**Resume compatibility**: `CHECKPOINT_VERSION` 2 → 3. M3 (v2) checkpoints cannot be resumed under M4 — clear error advised. Same handling as M2 → M3 transition.

**Wall-time**: pilot (3 arms × 4 seeds × 20 gens × pop 12 × K=50/L=25 × parallel=4) ≈ 3 hours total when arms run sequentially; ~50-60 min if all 3 arms run on different machines or backgrounded together. Early-stop should reduce this by ~30-50% on the saturating arms.
