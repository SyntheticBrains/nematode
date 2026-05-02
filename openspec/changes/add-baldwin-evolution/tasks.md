## 1. Protocol extension

- [ ] 1.1 Add `kind() -> Literal["none", "weights", "trait"]` method to the `InheritanceStrategy` Protocol in `packages/quantum-nematode/quantumnematode/evolution/inheritance.py`. Include in the docstring an explanation of the three semantics (no-op / weight-flow / trait-flow) and that the loop branches on this value.
- [ ] 1.2 Implement `NoInheritance.kind() -> "none"` and `LamarckianInheritance.kind() -> "weights"` to maintain existing behaviour.
- [ ] 1.3 Pre-implementation grep: find every `isinstance(.*, NoInheritance)` and `isinstance(.*, LamarckianInheritance)` call site in the codebase. Document the list (expected: `_inheritance_active` in loop.py at line ~308; possibly resume validation; possibly the GC step's no-op guard).
- [ ] 1.4 Refactor `_inheritance_active()` in `loop.py` from `not isinstance(self.inheritance, NoInheritance)` to `self.inheritance.kind() == "weights"`. Add a sibling helper `_inheritance_records_lineage()` returning `self.inheritance.kind() != "none"`.
- [ ] 1.5 Refactor any other `isinstance(...)` checks found in 1.3 to use `kind()`.
- [ ] 1.6 Add `test_inheritance_kind` test family to `test_inheritance.py`: assert all three impls return their declared literal; assert the literal is exactly one of the three (no leakage of new values).

## 2. BaldwinInheritance implementation

- [ ] 2.1 Implement `BaldwinInheritance` class in `inheritance.py`. No-arg constructor. `select_parents() -> []`, `assign_parent() -> None`, `checkpoint_path() -> None`, `kind() -> "trait"`. Module docstring updated to describe trait inheritance alongside the existing no-op and weight-flow descriptions.
- [ ] 2.2 Add elite-ID tracking to `EvolutionLoop`: a new `self._baldwin_elite_id: str | None = None` instance attribute. After each generation's `optimizer.tell` (the same point Lamarckian's `select_parents` runs), if `kind() == "trait"`, compute the prior generation's elite genome ID (top fitness, lex-tie-broken) and store it in the attribute. The next generation's per-child loop reads this attribute when `_inheritance_records_lineage()` is true and `kind() != "weights"` (Lamarckian uses its existing `inherited_from_per_child` array; Baldwin broadcasts the single elite ID).
- [ ] 2.3 Re-export `BaldwinInheritance` from `packages/quantum-nematode/quantumnematode/evolution/__init__.py`.
- [ ] 2.4 Add Baldwin unit tests to `test_inheritance.py`: kind value; no checkpoint path; no-op select/assign; elite-ID computation matches `LamarckianInheritance.select_parents` for the same input (single-elite case).

## 3. Brain field: weight_init_scale

- [ ] 3.1 Add `weight_init_scale: float = Field(default=1.0, ge=0.1, le=5.0)` field to the LSTMPPO brain config (likely `_reservoir_lstm_base.py:159` area). Field comment explains it's an evolvable innate-bias knob; default 1.0 is byte-equivalent to existing init.
- [ ] 3.2 At brain construction, after layers are built but before training, apply the scale: iterate over `nn.Linear.weight` and LSTM weight tensors and multiply std by `weight_init_scale`. Idiom TBD during implementation; consult existing brain-init patterns.
- [ ] 3.3 Add `test_weight_init_scale.py` with: (a) `weight_init_scale=1.0` produces tensors bit-identical to the M3-vintage init under same seed (no-op); (b) `weight_init_scale=2.0` produces tensors with std exactly 2× baseline; (c) field validates rejects values \<0.1 or >5.0.

## 4. Early stop on saturation

- [ ] 4.1 Add `early_stop_on_saturation: int | None = Field(default=None, ge=1)` field to `EvolutionConfig` in `packages/quantum-nematode/quantumnematode/utils/config_loader.py`.
- [ ] 4.2 Add `--early-stop-on-saturation N` CLI flag to `scripts/run_evolution.py` mirroring the existing `--algorithm` / `--inheritance` override pattern. argparse rejects values \<1.
- [ ] 4.3 Add `self._gens_without_improvement: int = 0` and `self._last_best_fitness: float | None = None` instance attributes to `EvolutionLoop`. After each generation's `optimizer.tell`, compare the current generation's `best_fitness` to `self._last_best_fitness` and update accordingly: improvement → reset counter to 0, set `_last_best_fitness`; no improvement → increment counter.
- [ ] 4.4 If `early_stop_on_saturation is not None` and `_gens_without_improvement >= early_stop_on_saturation`, log "Early-stop: best_fitness has not improved for N generations (last improvement at gen X)" and break out of the main loop. The break MUST occur AFTER the GC step for Lamarckian (preserves the surviving-elite-checkpoint semantics from the M3 spec).
- [ ] 4.5 Persist `_gens_without_improvement` and `_last_best_fitness` in the checkpoint pickle. Bump `CHECKPOINT_VERSION` 2 → 3.
- [ ] 4.6 Add `test_early_stop.py`: (a) never improves → fires at gen N; (b) monotonic improvement → never fires; (c) improves-then-stalls → counter resets correctly; (d) resume preserves the counter (write checkpoint at gen N with counter=2, resume, assert counter loaded as 2).

## 5. Config + CLI for Baldwin

- [ ] 5.1 Extend `EvolutionConfig.inheritance` Literal in `config_loader.py` to `Literal["none", "lamarckian", "baldwin"]`.
- [ ] 5.2 Extend `_validate_inheritance` for Baldwin: same rules as Lamarckian for `learn_episodes_per_eval > 0`, `warm_start_path is None`, `hyperparam_schema is not None`. Skip the architecture-changing-fields rejection (Baldwin doesn't load weights). Skip the `inheritance_elite_count != 1` rejection (Baldwin doesn't use the field; document this in the field comment).
- [ ] 5.3 Extend `--inheritance` CLI flag choices in `scripts/run_evolution.py` to include `"baldwin"`.
- [ ] 5.4 Extend the strategy construction in `scripts/run_evolution.py` `main()` (currently a 2-branch `if/else` for `lamarckian` vs `none`): add the third arm `case "baldwin": return BaldwinInheritance()`.
- [ ] 5.5 Extend the existing CLI guard at startup that rejects `inheritance != "none" + --fitness success_rate`. Update the error message to mention Baldwin's reason ("Baldwin records elite-parent lineage from the trained-elite-fitness signal — frozen-eval has no train phase to produce that signal").
- [ ] 5.6 Add 3 Baldwin validator tests to `test_config.py`: (a) Baldwin + `learn_episodes_per_eval=0` raises; (b) Baldwin + `warm_start_path` raises; (c) Baldwin + no `hyperparam_schema` raises. Plus a positive test: (d) Baldwin + arch-changing field is ALLOWED (the documented difference vs Lamarckian).
- [ ] 5.7 Add a smoke-style test to `test_loop_smoke.py`: 3-gen × pop 4 Baldwin run on a tiny config; assert no `inheritance/` directory is created; lineage gen-0 rows have empty `inherited_from`; lineage gen-1+ rows all have the same non-empty `inherited_from` (the prior-gen elite ID).

## 6. Spec sync to evolution-framework

- [ ] 6.1 Confirm the spec delta at `openspec/changes/add-baldwin-evolution/specs/evolution-framework/spec.md` covers all three new requirement scenarios (Baldwin Inheritance, Early Stop, modified Inheritance Strategy) per the `openspec validate --strict` pass that landed with the proposal.
- [ ] 6.2 Update the existing `openspec/specs/evolution-framework/spec.md` "Inheritance Strategy" requirement directly is NOT done in this PR — the delta sync runs at archival time per the openspec-archive-change skill workflow.

## 7. Pre-pilot smoke

- [ ] 7.1 Run a Baldwin smoke at `--generations 3 --population 6 --seed 42` (single seed, full K=50/L=25) on the Baldwin pilot YAML. Mirrors M3's task 9b smoke. Wall-time estimate: ~85s. Output to `evolution_results/m4_smoke_baldwin/`.
- [ ] 7.2 Verify smoke artefacts: lineage.csv has 18 rows (3 gens × 6 pop); gen-0 rows have empty `inherited_from`; gen-1+ rows have non-empty `inherited_from` equal to the prior gen's top-fitness genome ID; history.csv has 3 rows; NO `inheritance/` directory exists.
- [ ] 7.3 Run a 4-gen smoke that exercises the early-stop flag (e.g. `--early-stop-on-saturation 2 --generations 10 --population 4`) on a tiny config; verify the loop terminates before gen 10 if best_fitness plateaus.

## 8. Pilot configs

- [ ] 8.1 Create `configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml`. Structurally identical to `configs/evolution/lamarckian_lstmppo_klinotaxis_predator_control.yml` (the M3 control YAML with same brain/env/budget/seeds/TPE) with three diffs: (a) `inheritance: baldwin`, (b) add `weight_init_scale` and `entropy_decay_episodes` to `hyperparam_schema`, (c) add `early_stop_on_saturation: 5`.
- [ ] 8.2 Header comments explain the Baldwin framing (no weight inheritance, evolves richer learnability schema), point to the comparable arms (M3 lamarckian + M3 control), and note the new evolvable knobs.

## 9. Campaign scripts

- [ ] 9.1 Create `scripts/campaigns/phase5_m4_baldwin_lstmppo_klinotaxis_predator.sh`. 4-seed (42-45) loop, output to `evolution_results/m4_baldwin_lstmppo_klinotaxis_predator/`. Mirrors M3's lamarckian campaign script structure.
- [ ] 9.2 Create `scripts/campaigns/phase5_m4_lamarckian_rerun.sh`. Reuses M3's `lamarckian_lstmppo_klinotaxis_predator_pilot.yml`, output to `evolution_results/m4_lamarckian_lstmppo_klinotaxis_predator/`. Same 4 seeds. Header comment explains why the rerun (confounder-free 4-arm comparison on M4 revision).
- [ ] 9.3 Create `scripts/campaigns/phase5_m4_control_rerun.sh`. Reuses M3's `lamarckian_lstmppo_klinotaxis_predator_control.yml`, output to `evolution_results/m4_control_lstmppo_klinotaxis_predator/`. Same 4 seeds. Header comment explains the rerun rationale.
- [ ] 9.4 The hand-tuned baseline reuses M2.11's existing run (`evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline/`) — no script needed; just point the aggregator at it.

## 10. Run pilot

- [ ] 10.1 Run all three campaign scripts in parallel. Wall-time estimate: ~3 hours total when sequential, ~50-60 min when fully parallelised (each script uses 4 internal workers; user's machine should have ≥8-12 cores available for full parallel; otherwise stagger). Early-stop should reduce per-arm wall-time on saturating arms.
- [ ] 10.2 Verify each seed's `lineage.csv` has 20 generations (or fewer if early-stop fired) and the Baldwin arm's output directory does NOT contain an `inheritance/` subdirectory.

## 11. F1 post-pilot evaluator

- [ ] 11.1 Create `scripts/campaigns/baldwin_f1_postpilot_eval.py`. Args: `--baldwin-root` (the M4 Baldwin output root), `--seeds 42 43 44 45`, `--episodes 25` (default L=25), `--output-dir`. Reads each seed's `best_params.json` (the gen-N elite hyperparameter genome), instantiates the brain via `HyperparameterEncoder.decode`, and runs L frozen-eval episodes via `EpisodicSuccessRate.evaluate`.
- [ ] 11.2 Write `f1_innate_only.csv` with columns `seed, success_rate, elite_genome_id` (4 rows for the standard pilot).
- [ ] 11.3 Smoke test: run the script against the smoke output from task 7.1; verify it produces a CSV with one row containing a numeric success rate in [0.0, 1.0].

## 12. Aggregator + verdict

- [ ] 12.1 Create `scripts/campaigns/aggregate_m4_pilot.py`. Args: `--lamarckian-root`, `--baldwin-root`, `--control-root`, `--baseline-root`, `--f1-csv`, `--seeds`, `--output-dir`. Mirrors `aggregate_m3_pilot.py` structure but extended to 3 evolution arms + the F1 CSV.
- [ ] 12.2 Compute three gates: (a) Speed (Baldwin vs control): `mean_gen_baldwin_to_092 + 2 <= mean_gen_control_to_092`; (b) Genetic-assimilation (F1 vs baseline): `mean_f1_baldwin > mean_baseline + 0.10`; (c) Comparative (Baldwin vs Lamarckian): `mean_gen_baldwin_to_092 <= mean_gen_lamarckian_to_092 + 4`. Verdict: GO if all three; PIVOT if speed only; STOP otherwise.
- [ ] 12.3 Produce 4-curve plot (Baldwin + Lamarckian + control + baseline horizontal line; F1-innate-only as separate per-seed marker). Aggregator pads truncated histories with carry-forward of the final value so all curves span 1..max-gen for plotting.
- [ ] 12.4 Produce per-seed table: gen-to-0.92 for the three running arms + F1 innate-only success rate for Baldwin.
- [ ] 12.5 Produce `summary.md` with all three gate computations, per-seed tables, and the verdict.
- [ ] 12.6 Aggregator unit tests under `packages/quantum-nematode/tests/quantumnematode_tests/campaigns/` (or similar): test the 3-gate decision logic with synthetic histories (verdict cases: GO / PIVOT / STOP / PIVOT-with-failed-comparative).

## 13. Logbook 014

- [ ] 13.1 Create `docs/experiments/logbooks/014-baldwin-inheritance-pilot.md` with the standard logbook structure (Status / Background / Hypothesis / Method / Results / Analysis / Conclusions / Next Steps / Data References). Mirrors logbook 013's style. Include the M3-vs-M4 framing (weight-flow vs trait-flow) and the genetic-assimilation discussion.
- [ ] 13.2 Create `docs/experiments/logbooks/supporting/014/baldwin-inheritance-pilot-details.md` with per-seed tables, per-arm fitness curves, evolved-hyperparameter distributions, and the F1 innate-only forensic discussion.
- [ ] 13.3 Copy pilot artefacts to `artifacts/logbooks/014/m4_baldwin_pilot/{baldwin,lamarckian,control,baseline,summary}/` mirroring the M3 layout. Include each seed's `best_params.json`, `history.csv`, `lineage.csv`, `checkpoint.pkl`, and (for Lamarckian) the surviving final-gen elite under `inheritance/`.
- [ ] 13.4 Update `docs/experiments/README.md` to add the M4 logbook to the active-experiments table.

## 14. Tracker + roadmap

- [ ] 14.1 Tick M4.1-M4.7 in `openspec/changes/2026-04-26-phase5-tracking/tasks.md`. Flip M4 status from `not started` to `complete` with the same detail-level summary M3's status got.
- [ ] 14.2 Flip M4 row in `docs/roadmap.md` from 🟢 unblocked to ✅ complete with a one-sentence summary of the verdict.

## 15. PR + archive

- [ ] 15.1 Run `uv run pre-commit run -a` clean.
- [ ] 15.2 Run `uv run pytest -m "not nightly" --tb=short -q` clean (M3 tests must pass unchanged).
- [ ] 15.3 Open draft PR via `gh pr create --draft`. Title: "feat(m4): Baldwin Effect inheritance pilot + early-stop + weight_init_scale". Body summarises the verdict, the three gates' outcomes, and links logbook 014.
- [ ] 15.4 After review approval, archive the OpenSpec change via the `openspec-archive-change` skill: syncs the spec delta into `openspec/specs/evolution-framework/spec.md` and moves `openspec/changes/add-baldwin-evolution/` to `openspec/changes/archive/YYYY-MM-DD-add-baldwin-evolution/`.
