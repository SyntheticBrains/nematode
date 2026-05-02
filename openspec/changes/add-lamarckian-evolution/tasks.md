# Tasks: M3 — Lamarckian Inheritance Pilot (Predator Arm)

Single PR. Order is dependency-first: Protocol + config + validators land before the loop hook so the framework is testable in isolation, then the pilot config + campaign + run + logbook close the milestone.

## 1. InheritanceStrategy Protocol + implementations

- [ ] 1.1 Create `packages/quantum-nematode/quantumnematode/evolution/inheritance.py` with the `InheritanceStrategy` Protocol (`select_parents`, `assign_parent`, `checkpoint_path`).
- [ ] 1.2 Implement `NoInheritance` (default no-op): `select_parents` returns `[]`, `assign_parent` returns `None`, `checkpoint_path` raises `NotImplementedError` (it should never be called when the strategy is no-op).
- [ ] 1.3 Implement `LamarckianInheritance(elite_count: int = 1)`: `select_parents` sorts prior gen by fitness desc and breaks ties on `genome_id` lexicographic order; `assign_parent` returns `parent_ids[child_index % len(parent_ids)]` or `None` when `parent_ids == []`; `checkpoint_path` returns `output_dir / "inheritance" / f"gen-{generation:03d}" / f"genome-{genome_id}.pt"`.
- [ ] 1.4 Re-export `InheritanceStrategy`, `NoInheritance`, `LamarckianInheritance` from `quantumnematode/evolution/__init__.py`.
- [ ] 1.5 Module docstring lists tournament / roulette / soft-elite as future strategies behind the same Protocol (so M4's BaldwinInheritance has a documented home to slot into).

## 2. EvolutionConfig + validators

- [ ] 2.1 Add `inheritance: Literal["none", "lamarckian"] = "none"` and `inheritance_elite_count: int = Field(default=1, ge=1)` to `EvolutionConfig` in `packages/quantum-nematode/quantumnematode/utils/config_loader.py` (after `warm_start_path`).
- [ ] 2.2 Add a `model_validator(mode="after")` on `EvolutionConfig` rejecting: `inheritance != "none"` AND `learn_episodes_per_eval == 0`; `inheritance != "none"` AND `warm_start_path is not None`; `inheritance_elite_count > population_size`. Error messages match the wording in the spec scenarios.
- [ ] 2.3 Extend `SimulationConfig._validate_hyperparam_schema` (or add a sibling `model_validator`) to reject: `inheritance != "none"` AND `hyperparam_schema is None`; `inheritance != "none"` AND any field in `hyperparam_schema` is in `_ARCHITECTURE_CHANGING_FIELDS` (reuse the existing constant — single source of truth).

## 3. LearnedPerformanceFitness kwargs

- [ ] 3.1 Add `warm_start_path_override: Path | None = None` and `weight_capture_path: Path | None = None` kwargs to `LearnedPerformanceFitness.evaluate` in `packages/quantum-nematode/quantumnematode/evolution/fitness.py`. Signature stays back-compatible (defaults are `None`); `EpisodicSuccessRate.evaluate` does not accept the kwargs.
- [ ] 3.2 In the warm-start branch (currently the conditional that calls `load_weights(brain, evolution_config.warm_start_path)`), prefer `warm_start_path_override` when set; fall through to `evolution_config.warm_start_path` otherwise. The validator (task 2.2) makes the two mutually exclusive at YAML load time, so in practice only one is ever set.
- [ ] 3.3 After the K-train loop completes and BEFORE the L-eval phase begins, if `weight_capture_path is not None`, call `save_weights(brain, weight_capture_path)`. Create the parent directory with `mkdir(parents=True, exist_ok=True)` first.

## 4. EvolutionLoop hook + GC + checkpoint v2

- [ ] 4.1 Add `inheritance: InheritanceStrategy = NoInheritance()` constructor kwarg to `EvolutionLoop` in `packages/quantum-nematode/quantumnematode/evolution/loop.py`. Initialise `self._selected_parent_ids: list[str] = []`.
- [ ] 4.2 In the per-child loop that builds `eval_args`, compute `child_capture_path = strategy.checkpoint_path(self.output_dir, gen, gid)` (only when `not isinstance(strategy, NoInheritance)`) and `parent_warm_start = strategy.checkpoint_path(self.output_dir, gen - 1, parent_id)` where `parent_id = strategy.assign_parent(idx, self._selected_parent_ids)`. Pass `None` for both when the strategy is no-op (preserves M2 behaviour).
- [ ] 4.3 Extend the `_evaluate_in_worker` tuple to carry the two new paths; forward them into `LearnedPerformanceFitness.evaluate` via the new kwargs.
- [ ] 4.4 After `optimizer.tell` returns and BEFORE `self._prev_generation_ids = gen_ids`, call `next_selected = strategy.select_parents(list(zip(genomes, fitnesses)), gen)`.
- [ ] 4.5 Implement GC: delete every file in `<output_dir>/inheritance/gen-{gen-1:03d}/` whose ID is NOT in `self._selected_parent_ids` (the OLD selected set — those parents' children just finished evaluating); then delete every file in `<output_dir>/inheritance/gen-{gen:03d}/` whose ID is NOT in `next_selected`. Use `Path.glob` + `unlink(missing_ok=True)`.
- [ ] 4.6 Update `self._selected_parent_ids = next_selected` before incrementing `_generation`.
- [ ] 4.7 Bump `CHECKPOINT_VERSION` from 1 to 2. Add `selected_parent_ids` to the checkpoint pickle dict in `_save_checkpoint`; restore it in `_load_checkpoint`.
- [ ] 4.8 Add a `Path.exists()` guard before passing `parent_warm_start` to a worker; if missing, log a `logger.warning` naming the missing genome ID and pass `None` instead (from-scratch fallback). Spec scenario "Resume from checkpoint preserves selected parent IDs" requires this.

## 5. Lineage CSV schema

- [ ] 5.1 Add `inherited_from` to `CSV_HEADER` in `packages/quantum-nematode/quantumnematode/evolution/lineage.py`.
- [ ] 5.2 Add `inherited_from: str = ""` parameter to `LineageTracker.record`. The loop passes `parent_id or ""` (empty for gen 0, no-inheritance, or fallback).
- [ ] 5.3 Confirm `aggregate_m2_pilot.py` reads `history.csv` only (not `lineage.csv`) — no changes needed there. If anything reads `lineage.csv`, update it to handle the new column.

## 6. CLI wiring

- [ ] 6.1 Add `--inheritance {none,lamarckian}` argparse flag to `parse_arguments` in `scripts/run_evolution.py` (mirrors `--algorithm`).
- [ ] 6.2 Thread the override into `_resolve_evolution_config` (`if args.inheritance is not None: overrides["inheritance"] = args.inheritance`). Pydantic re-validation catches invalid combinations at startup.
- [ ] 6.3 In `main()`, after `evolution_config` is resolved, construct the strategy: `LamarckianInheritance(elite_count=evolution_config.inheritance_elite_count)` if `evolution_config.inheritance == "lamarckian"` else `NoInheritance()`. Pass `inheritance=strategy` into the `EvolutionLoop` constructor.
- [ ] 6.4 Update the existing comment near `run_evolution.py:336-345` that says "Lamarckian inheritance, which is out of scope" — note that Lamarckian is now in scope via `evolution.inheritance`, but the existing weight-encoder + LearnedPerformanceFitness guard stays (the validator now also rejects `inheritance: lamarckian` + weight encoder for the same reason).

## 7. Tests

- [ ] 7.1 New `packages/quantum-nematode/tests/quantumnematode_tests/evolution/test_inheritance.py`: `test_no_inheritance_returns_empty_parents`; `test_lamarckian_select_parents_returns_top_k_by_fitness`; `test_lamarckian_select_parents_breaks_ties_lexicographically`; `test_lamarckian_assign_parent_round_robin`; `test_lamarckian_assign_parent_returns_none_for_empty_parent_list`; `test_checkpoint_path_format_round_trips`.
- [ ] 7.2 New `test_weight_capture.py`: `test_learned_performance_writes_capture_file_when_path_set` (mocked `save_weights`, assert called once with correct path BEFORE the eval phase); `test_learned_performance_loads_warm_start_override` (write a real checkpoint, evaluate with override, assert the loaded brain's first-step action under fixed sensory input matches a brain `load_weights`'d from the same file).
- [ ] 7.3 Modify `test_loop_smoke.py`: `test_loop_with_lamarckian_inheritance_3_gens` (3-gen × pop 4 lamarckian smoke; assert `lineage.csv` gen-1+ rows have non-empty `inherited_from`, `inheritance/gen-002/` has exactly 1 file); `test_inheritance_directory_garbage_collection` (4-gen smoke; assert `inheritance/gen-000/`, `gen-001/`, `gen-002/` are gone; only `gen-003/` survives with 1 file).
- [ ] 7.4 Modify `test_config.py` (or create `test_inheritance_validators.py` if config tests live elsewhere — confirm during implementation): four validator-rejection tests covering tasks 2.2 + 2.3, all asserting Pydantic `ValidationError` with a substring of the spec-mandated error text.
- [ ] 7.5 Modify `test_lineage.py`: `test_lineage_records_inherited_from` — write known parent IDs via `LineageTracker.record(..., inherited_from="parent-abc")`, read back, assert column round-trips. Existing test `inherited_from`-empty for no-inheritance runs.
- [ ] 7.6 Confirm full evolution test suite passes: `uv run pytest packages/quantum-nematode/tests/quantumnematode_tests/evolution/ -v` — including all M2 tests on `inheritance: none` paths (which must be byte-equivalent).

## 8. Pilot config + campaign scripts

- [ ] 8.1 New `configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml` — verbatim clone of `configs/evolution/hyperparam_lstmppo_klinotaxis_predator_pilot_tpe.yml` with `inheritance: lamarckian` and `inheritance_elite_count: 1` added to the `evolution:` block. Same K=50 / L=25 / 4 seeds (42-45) / parallel=4 / pop=12 / 20 gens.
- [ ] 8.2 New `scripts/campaigns/phase5_m3_lamarckian_lstmppo_klinotaxis_predator.sh` — clone of `scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis_predator_tpe.sh` with `OUTPUT_ROOT=evolution_results/m3_lamarckian_lstmppo_klinotaxis_predator` and the new CONFIG path. Header docstring re-targeted at M3.
- [ ] 8.3 New `scripts/campaigns/phase5_m3_lamarckian_lstmppo_klinotaxis_predator_control.sh` — within-experiment from-scratch control. Re-runs the M2.12 TPE config under the M3 revision so the lamarckian-vs-control comparison is confounder-free. `OUTPUT_ROOT=evolution_results/m3_lamarckian_lstmppo_klinotaxis_predator_control`. Same 4 seeds.
- [ ] 8.4 No new run_simulation.py-driven baseline — reuse `scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis_predator_baseline.sh` (optimiser- and inheritance-independent).

## 9. Aggregator + decision logic

- [ ] 9.1 New `scripts/campaigns/aggregate_m3_pilot.py` taking `--lamarckian-root`, `--control-root`, `--baseline-root`, `--seeds 42 43 44 45`, `--output-dir`. Reads `history.csv` from each seed under each arm.
- [ ] 9.2 Generate two-curve plot: lamarckian mean ± std vs control mean ± std vs generation, with the run_simulation.py baseline as a horizontal line. Save as `convergence.png` under `--output-dir`.
- [ ] 9.3 Per-seed table: generation at which best fitness first reaches ≥ 0.92 (write as `convergence_speed.csv`).
- [ ] 9.4 Decision verdict per the spec / plan: GO if both `mean_gen_lamarckian_to_092 + 4 ≤ mean_gen_control_to_092` AND `mean_gen1_lamarckian ≥ mean_gen3_control`. PIVOT if exactly one. STOP if neither. Write to `summary.md`.
- [ ] 9.5 Optional: hyperparameter-spread-across-seeds plot as the canary for design.md Risk 2 (TPE posterior collapse). Decide during implementation based on what the pilot data looks like.

## 10. Pilot run + logbook

- [ ] 10.1 Run the lamarckian campaign: `OUTPUT_ROOT=evolution_results/m3_lamarckian_lstmppo_klinotaxis_predator scripts/campaigns/phase5_m3_lamarckian_lstmppo_klinotaxis_predator.sh`. Expected wall-time: ~50 min (4 seeds × ~12-14 min/seed at parallel=4, comparable to M2.12).
- [ ] 10.2 Run the control campaign in parallel: `scripts/campaigns/phase5_m3_lamarckian_lstmppo_klinotaxis_predator_control.sh`. Expected wall-time: same as M2.12 (~50 min).
- [ ] 10.3 Run aggregator: `uv run python scripts/campaigns/aggregate_m3_pilot.py --lamarckian-root evolution_results/m3_lamarckian_lstmppo_klinotaxis_predator --control-root evolution_results/m3_lamarckian_lstmppo_klinotaxis_predator_control --baseline-root evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline --seeds 42 43 44 45 --output-dir artifacts/logbooks/013/m3_lamarckian_pilot/summary`.
- [ ] 10.4 Archive per-seed artefacts under `artifacts/logbooks/013/m3_lamarckian_pilot/{lamarckian,control}/seed-{42,43,44,45}/`.
- [ ] 10.5 Write logbook `docs/experiments/logbooks/013-lamarckian-inheritance-pilot.md` mirroring [logbook 012](../../../docs/experiments/logbooks/012-hyperparam-evolution-mlpppo-pilot.md)'s structure: Objective, Background, Hypothesis, Method, Results (per-seed best fitness tables for both arms; convergence plots; speed/floor metrics), Analysis (gens-to-0.92, gen-1 floor; whether either of design.md's Risks 1/2 manifested), Conclusions, Next Steps. State GO/PIVOT/STOP decision and what M4 inherits.
- [ ] 10.6 Optional supporting appendix at `docs/experiments/logbooks/supporting/013/lamarckian-inheritance-pilot-details.md` for full per-genome trajectories or hyperparameter-spread analysis.

## 11. Tracker + roadmap updates

- [ ] 11.1 Tick `M3.1`–`M3.8` and any added sub-tasks in `openspec/changes/2026-04-26-phase5-tracking/tasks.md`. Note: M3 task list there currently mentions both MLPPPO and LSTMPPO pilot YAMLs — update to reflect the predator-arm-only scope decided during planning (single config, no MLPPPO companion).
- [ ] 11.2 Flip M3 status header in the tracker to `complete` with a one-paragraph summary mirroring M2's: decision verdict, seed-level results, optimiser/inheritance settings, headline metric.
- [ ] 11.3 Flip the M3 row in `docs/roadmap.md` Phase 5 Milestone Tracker to `✅ complete` with a one-sentence outcome blurb.
- [ ] 11.4 If M3's GO unblocks M4 (Baldwin Effect) cleanly, note it in the tracker's M4 row dependencies. If GO is conditional or PIVOT, document the implication for M4's scope.

## 12. Quality gates

- [ ] 12.1 `openspec validate --changes add-lamarckian-evolution --strict` passes.
- [ ] 12.2 `uv run pre-commit run -a` clean.
- [ ] 12.3 Full test suite passes: `uv run pytest packages/quantum-nematode/tests/ -v` (including non-evolution tests for regression).
- [ ] 12.4 Open PR with link to logbook 013, the GO/PIVOT/STOP verdict in the description, and a checkbox for any deferred follow-up (e.g. tournament strategy, M4 prep) noted in design.md's Open Questions.
