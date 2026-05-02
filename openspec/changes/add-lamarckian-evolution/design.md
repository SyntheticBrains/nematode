## Context

Phase 5 M2 closed with the LSTMPPO + klinotaxis + pursuit-predator arm as the first non-saturated fitness landscape — under TPE the population climbs from gen-1 means of 0.5–0.7 to a 0.92–1.00 ceiling over 6–12 generations across 4 seeds ([logbook 012](../../../docs/experiments/logbooks/012-hyperparam-evolution-mlpppo-pilot.md)). M3 piggybacks on that exact arm to ask whether inheriting trained weights from selected parents accelerates the climb so children start near the ceiling and reach it sooner.

Three pieces of M0/M2 framework already do most of the work: `WeightPersistence` (M0, brain weight round-trip), `LearnedPerformanceFitness` (M2, K-train + L-eval phase split), and `evolution.warm_start_path` (M2.10, single fixed checkpoint warm-starting all genomes). M3 generalises the last one from "static, run-wide" to "per-genome, per-generation, parent-selected".

## Goals / Non-Goals

**Goals:**

- Allow each child of generation N+1 to warm-start its brain from a checkpoint of a *selected parent* from generation N, before the child's own K=50 training episodes.
- Keep the hyperparameter genome evolving via TPE (M2.12); weights flow as a side-channel substrate, not as part of the genome.
- Keep `inheritance: none` (the default) byte-equivalent to M2.12 — no new files written, no GC, no checkpoint version bump effects, no measurable wall-time regression on M2 pilots.
- Bound disk usage to `O(1)` per run (max `2 * inheritance_elite_count` `.pt` files at steady state) via per-generation garbage collection.
- Provide a single, narrow Protocol (`InheritanceStrategy`) so M4 (Baldwin) and tournament/roulette/soft-elite variants are pluggable additions rather than refactors.

**Non-Goals:**

- Inheritance for weight encoders (MLPPPO/LSTMPPO weight evolution). Validator rejects the combination — Lamarckian on top of weight evolution would double-count weights as both genome and substrate.
- Inheritance across sessions / runs. Inheritance state (selected-parent IDs, the `inheritance/` directory) is per-run.
- Architecture-changing schemas under inheritance. Same shape-mismatch reasoning as M2.10's static warm-start; rejected by the existing `_ARCHITECTURE_CHANGING_FIELDS` denylist.
- Genome-level inheritance (TPE prior bias toward best parents). TPE's posterior already does this implicitly; piling another bias on top would conflate two evolutionary signals.

## Decisions

### Decision 1: Inheritance is a Protocol-pluggable strategy, not a flag inside `LearnedPerformanceFitness`

`InheritanceStrategy` Protocol with three methods (`select_parents`, `assign_parent`, `checkpoint_path`) and two implementations (`NoInheritance`, `LamarckianInheritance`). The strategy lives in a new `evolution/inheritance.py` module and is constructed by `scripts/run_evolution.py` from the resolved `EvolutionConfig`, then injected into `EvolutionLoop`. Fitness functions stay frozen: they receive optional `warm_start_path_override` and `weight_capture_path` kwargs but have no awareness of strategy semantics.

**Why over an inline inheritance flag:** parent-selection rule is the policy that varies between Lamarckian / Baldwin / tournament / roulette / soft-elite. Each is a different `InheritanceStrategy` implementation behind the same Protocol — adding M4's `BaldwinInheritance` is a new file plus a `Literal` extension, not a fitness rewrite. Mirrors how `EvolutionaryOptimizer` (CMA-ES / GA / TPE) already plugs into the loop.

**Alternatives considered:** (a) A boolean `evolution.lamarckian: bool` flag with the parent-selection rule hard-coded into the loop. Rejected — locks the loop to one rule, makes M4 a refactor. (b) Subclassing `LearnedPerformanceFitness` into `LamarckianFitness`. Rejected — fitness function would need access to inter-generation state (selected parent IDs from the prior gen), forcing the loop to thread state through fitness calls; cleaner to keep fitness stateless per-genome and put cross-genome bookkeeping in the loop.

### Decision 2: Weight transfer via per-genome side-effect file IO, not return-type changes

`LearnedPerformanceFitness.evaluate` gains two optional kwargs (`warm_start_path_override`, `weight_capture_path`). The first short-circuits the existing `evolution_config.warm_start_path` load (validator makes the two mutually exclusive). The second writes the post-K-train weights to disk via `save_weights` immediately after the train loop and before the eval phase. The `FitnessFunction` Protocol shape is unchanged — kwargs default to `None`, `EpisodicSuccessRate` doesn't accept them, and the loop only passes them when inheritance is active.

**Why over changing the return type:** the `FitnessFunction` Protocol returns `float`. Returning `tuple[float, dict]` (or similar) would touch every implementor, every test, and the worker tuple in `_evaluate_in_worker`. Side-effect file IO is exactly how M2.10's warm-start already flows in the same function — symmetric capture path keeps the round-trip readable. Per-genome torch.save adds ~10 ms/genome (LSTMPPO checkpoints are ~hundreds of KB), negligible vs the ~60-second K=50 train phase.

**Alternatives considered:** (a) Return a `dict[genome_id, weights]` from the worker. Rejected — multiprocessing pickle overhead grows with population size; disk is cheaper and persists across resume. (b) Use a shared `multiprocessing.Manager()` dict. Rejected — adds a manager process, cross-platform fragility, no resume support.

### Decision 3: M3 ships single-elite only; multi-elite reserved for M4

`LamarckianInheritance(elite_count: int = 1)`. M3 ships single-elite-broadcast only — every child of generation N+1 inherits from the single best parent of generation N. The validator rejects `inheritance_elite_count != 1` when `inheritance: lamarckian`, so multi-elite paths are unreachable in M3 even though the field exists structurally.

The implementation supports `elite_count > 1` semantically (round-robin broadcast: `parent_ids[child_index % len(parent_ids)]`) so M4-or-later can lift the validator without an algorithm rewrite — but the path is documented behaviour, not shipped behaviour, in M3.

**Why ship single-elite only:** the pilot's job is decision-gate, not optimisation. Single-elite is the strongest possible inheritance signal — if Lamarckian doesn't help when every child gets the best-known parent, it won't help with sampling-style parent selection either. Pilot result interpretation is also clean: any acceleration over M2.12 can be attributed unambiguously to inherited weights, not to an elite-count tuning knob. Restricting at the validator (rather than just defaulting) prevents an experimenter from accidentally enabling an untested code path.

**Alternatives considered:** Tournament selection (`tournament_k=3`), fitness-proportionate ("roulette") sampling, soft-elite (each child draws independently from the top-k). All are listed in the module docstring as future-work strategies behind the same Protocol — adding any of them post-pilot is a new file, not a refactor.

### Decision 4: Generation 0 runs from-scratch identically to M2.12

`select_parents([], 0) → []`; `assign_parent` with empty parent IDs returns `None` for every child. The first generation has no prior gen to inherit from, so all gen-0 children get fresh-init weights — same as M2.12. The inheritance benefit only kicks in from gen 1 onwards, when gen-0 children have completed their K=50 train phases and produced checkpoints worth inheriting.

**Why this matters:** confirms the pilot's Floor metric (M3 gen-1 mean ≥ control gen-3 mean) is measuring real inheritance work — gen-0 is identical between arms, so any gen-1 lift is purely from the warm-start.

### Decision 5: Garbage collection runs at end of each generation, retains exactly the parent set

After each generation completes:

- Delete every file in `inheritance/gen-{gen-1}/` whose ID is NOT in the OLD selected-parent set (those checkpoints' children have just finished evaluating).
- Delete every file in `inheritance/gen-{gen}/` whose ID is NOT in the NEW selected-parent set.

Steady-state disk usage: `2 * inheritance_elite_count` files. With `elite_count=1` and ~hundreds of KB per LSTMPPO checkpoint, total inheritance footprint is ~1 MB throughout a 20-generation run.

**Why GC at end-of-generation rather than lazy:** simple to reason about (you can inspect `inheritance/` mid-run and see exactly the surviving parents); resume-safe (orphans from a killed generation are scrubbed on the next iteration's tail); no cross-generation reference counting.

### Decision 6: Lineage CSV gains an `inherited_from` column; `history.csv` is unchanged

`LineageTracker` schema extends to record which single parent ID each child inherited weights from (or empty string for from-scratch children, including all of gen 0). `history.csv` (consumed by `aggregate_m2_pilot.py`) is unchanged so existing aggregator scripts keep working. M3's new `aggregate_m3_pilot.py` reads `history.csv` from both arms head-to-head and doesn't need `inherited_from` for the decision verdict — that column exists for forensic post-pilot analysis (e.g. plotting the ancestry tree of a successful seed).

### Decision 7: Checkpoint version bumps to 2; `_selected_parent_ids` persists for resume

`CHECKPOINT_VERSION` at `loop.py:56` bumps from 1 to 2. The pickle dict gains a `selected_parent_ids: list[str]` key. `_load_checkpoint` already raises on version mismatch — old pickled checkpoints from M2 cannot be resumed under M3 directly, which is the correct behaviour (M3 needs to start fresh or a converter script needs writing later if anyone wants to resume an M2 run under inheritance, which nobody is asking for).

On resume, the GC scan that runs at the next iteration's tail naturally re-cleans any orphaned files the killed run left behind. If a parent file is unexpectedly missing on resume, the loop falls back to from-scratch for that one child with a `logger.warning` — the pilot operator sees a single warning line rather than a hard crash.

## Risks / Trade-offs

**Risk 1 — Inherited transients may need to be unlearned**: Each child inherits parent weights but its OWN evolved hyperparameters (different `actor_lr`, `gamma`, `entropy_coef`). If a child's `entropy_coef` is much higher than the parent's, the K=50 episodes may UNDO the inherited policy before fitting the new schedule. Canary: the pilot's Floor metric (gen-1 mean ≥ control gen-3 mean). Mitigation if seen: tighten schema bounds on entropy/gamma, or document Lamarckian as conditional on narrow schemas.

**Risk 2 — TPE posterior collapse onto the elite parent's hyperparameters**: TPE prefers near-observed-good points. Broadcasting weights from one parent while TPE simultaneously narrows around that parent's hyperparams may shrink diversity faster than M2.12, raising premature-convergence risk. Symptom: saturation by gen 3-4 at a different (lower?) ceiling than M2.12. Mitigation: check spread of evolved hyperparameters across seeds in the aggregator output; if collapsed, document and consider raising `inheritance_elite_count` in a follow-up milestone (M3 itself rejects elite_count > 1 at validation).

**Risk 3 — Per-genome torch.save under multiprocessing**: Four workers writing into the same `inheritance/gen-NNN/` directory may serialise on inode locks. Likely fine on APFS/ext4 for ~hundreds-of-KB LSTMPPO checkpoints, but if generation wall-time inflates noticeably vs M2.12's ~14 min/seed, profile the save call. Mitigation: per-worker subdirectories + atomic rename.

**Risk 4 — Comparison-arm confounders if M3 reuses M2.12's published numbers**: M2.12's results were produced on an earlier revision; comparing across PRs invites Python/dep/machine drift. Mitigation: M3 ships a within-experiment from-scratch control script that re-runs the M2.12 TPE config under the M3 revision so lamarckian-vs-control is confounder-free.

## Migration Plan

No migration. Default `inheritance: none` reproduces M2.12 behaviour byte-for-byte. Old checkpoint files (CHECKPOINT_VERSION=1) cannot be resumed under M3 — they raise the existing version-mismatch error and the user re-starts the run. Acceptable because (a) no in-flight M2 runs exist, and (b) M3 changes the on-disk loop state shape, not just the inheritance feature.

Rollback: revert the PR. The `inheritance/` directory under existing `evolution_results/` paths is harmless if left behind (no other code reads it).

## Open Questions

None blocking. The two pilot decisions worth flagging as "we'll learn during implementation":

1. Whether `aggregate_m3_pilot.py` should include a hyperparameter-spread-across-seeds plot (the canary for TPE posterior collapse, Risk 2). Cheap to add — leaning yes, but waiting until the first pilot run to see what shape best surfaces the signal.
2. Whether the logbook's GO criterion should weight the Speed metric (gens-to-0.92) more heavily than the Floor metric (gen-1 mean ≥ control gen-3 mean), or treat them as equal-weight as the plan currently does. This is a write-up decision after we see the numbers, not a code decision.
