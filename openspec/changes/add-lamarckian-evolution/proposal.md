## Why

Phase 5 M2 closed with the LSTMPPO + klinotaxis + pursuit-predator arm as the first non-saturated fitness landscape ([logbook 012](../../../docs/experiments/logbooks/012-hyperparam-evolution-mlpppo-pilot.md)) — under TPE the population climbs from gen-1 means of 0.5–0.7 to a 0.92–1.00 ceiling over 6–12 generations across 4 seeds. M3 is the natural next question: does inheriting trained weights from selected parents accelerate that climb so children start near the ceiling and reach it sooner?

The framework substrate already exists. M0 shipped `WeightPersistence` for MLPPPO and LSTMPPO; M2.10 shipped warm-start fitness via `evolution.warm_start_path` (one fixed checkpoint, all genomes). M3 generalises that to **per-genome, per-generation** warm-starting where the parent is *selected from the prior generation's fitness*, not a static file. The hyperparameter genome continues to evolve via TPE — weights flow as a side-channel substrate keyed by parent `genome_id`. This is biologically Lamarckian (acquired traits inherited) and unlocks M4 (Baldwin Effect) and M6 (transgenerational memory), both gated on M3 GO.

## What Changes

- **NEW** `InheritanceStrategy` Protocol in `quantumnematode/evolution/inheritance.py` with `select_parents`, `assign_parent`, and `checkpoint_path` methods. Two implementations: `NoInheritance` (no-op default) and `LamarckianInheritance(elite_count=1)` (top-K-by-fitness, broadcast to children of next generation).
- **MODIFIED** `LearnedPerformanceFitness.evaluate` gains two optional kwargs (`warm_start_path_override`, `weight_capture_path`) that thread per-genome warm-start and post-K-train weight capture through file IO. The `FitnessFunction` Protocol shape is unchanged — kwargs default to `None` and are only passed when inheritance is active.
- **MODIFIED** `EvolutionLoop.run` calls `strategy.select_parents` after each generation's `optimizer.tell`, threads parent checkpoint paths into the next generation's eval args, and garbage-collects per-genome checkpoints whose parents weren't selected. Steady-state disk usage is `2 * inheritance_elite_count` files. `CHECKPOINT_VERSION` bumps to 2 to persist `_selected_parent_ids` for resume safety.
- **MODIFIED** `LineageTracker` schema gains an `inherited_from` column (parent ID this child warm-started from, or empty string). `history.csv` is unchanged so existing aggregator scripts keep working.
- **MODIFIED** `EvolutionConfig` gains `inheritance: Literal["none", "lamarckian"] = "none"` and `inheritance_elite_count: int = 1`. Five validators reject invalid combinations (no train phase, conflict with static `warm_start_path`, missing `hyperparam_schema`, architecture-changing schema fields, elite_count > population_size).
- **NEW** `--inheritance {none,lamarckian}` CLI flag in `scripts/run_evolution.py` mirroring the existing `--algorithm` override pattern.
- **NEW** pilot config `configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml` (verbatim clone of M2.12's TPE predator config + `inheritance: lamarckian`), campaign script `scripts/campaigns/phase5_m3_lamarckian_lstmppo_klinotaxis_predator.sh`, and a within-experiment from-scratch control script (re-runs M2.12 under M3's revision so the comparison is confounder-free).
- **NEW** aggregator `scripts/campaigns/aggregate_m3_pilot.py` producing the lamarckian-vs-control two-curve plot, per-seed convergence-speed table, and GO/PIVOT/STOP verdict against the M3 decision gate.
- **NEW** logbook `docs/experiments/logbooks/013-lamarckian-inheritance-pilot.md` documenting the pilot result.

No breaking changes. With `inheritance: none` (the default) the loop, fitness, and lineage code paths are byte-equivalent to M2.12.

## Capabilities

### Modified Capabilities

- `evolution-framework`: adds an "Inheritance Strategy" requirement covering the `InheritanceStrategy` Protocol, `LamarckianInheritance` semantics, and the loop's per-genome weight checkpoint capture / GC contract; extends "Learned-Performance Fitness" with the per-genome `warm_start_path_override` and `weight_capture_path` kwargs; extends "Evolution Configuration Block" with `evolution.inheritance` and `evolution.inheritance_elite_count` fields and their validator rejection rules; extends "Lineage Tracking" with the new `inherited_from` CSV column.

### New Capabilities

None. M3 extends `evolution-framework` rather than introducing a new capability — inheritance is a strategy *within* the existing evolution loop. All `evolution.*` config rules live in `evolution-framework` (per the existing "Evolution Configuration Block" requirement), not in `configuration-system`.

## Impact

- **Code**: new module `quantumnematode/evolution/inheritance.py`; modifications to `evolution/loop.py`, `evolution/fitness.py`, `evolution/lineage.py`, `evolution/__init__.py`, `utils/config_loader.py`, `scripts/run_evolution.py`.
- **Configs**: new `configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml`. No existing configs change.
- **Scripts**: two new campaign scripts + one new aggregator under `scripts/campaigns/`. M2.11's run_simulation.py-driven baseline script is reused as-is.
- **Tests**: new `test_inheritance.py` and `test_weight_capture.py`; modifications to `test_loop_smoke.py`, `test_config.py`, `test_lineage.py` under `packages/quantum-nematode/tests/quantumnematode_tests/evolution/`.
- **Docs**: new logbook `013-lamarckian-inheritance-pilot.md`; updates to `openspec/changes/2026-04-26-phase5-tracking/tasks.md` (tick M3.1–M3.8, flip M3 status to `complete`) and `docs/roadmap.md` (M3 row → ✅ complete).
- **Dependencies**: none. Uses existing `WeightPersistence` round-trip, `OptunaTPEOptimizer`, and `HyperparameterEncoder`.
- **Downstream milestones unblocked**: M4 (Baldwin Effect, gated on M3 GO) and M6 (transgenerational memory, gated on M3 + M4-or-M5).

Detailed implementation plan: `/Users/chris/.claude/plans/agreed-with-all-including-memoized-abelson.md` (approved).
