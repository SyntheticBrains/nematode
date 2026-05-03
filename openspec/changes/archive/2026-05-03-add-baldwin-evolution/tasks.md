## 1. Protocol extension

- [x] 1.1 Add `kind() -> Literal["none", "weights", "trait"]` method to the `InheritanceStrategy` Protocol in `packages/quantum-nematode/quantumnematode/evolution/inheritance.py`. Include in the docstring an explanation of the three semantics (no-op / weight-flow / trait-flow) and that the loop branches on this value.
- [x] 1.2 Implement `NoInheritance.kind() -> "none"` and `LamarckianInheritance.kind() -> "weights"` to maintain existing behaviour.
- [x] 1.3 Pre-implementation grep: find every `isinstance(.*, NoInheritance)` and `isinstance(.*, LamarckianInheritance)` call site in the codebase. Confirmed: 1 production-code site (`_inheritance_active` at loop.py:308) and 1 docstring reference (inheritance.py:120). Resume validation and GC step do not use isinstance — they read `_inheritance_active()` indirectly.
- [x] 1.4 Refactor `_inheritance_active()` in `loop.py` from `not isinstance(self.inheritance, NoInheritance)` to `self.inheritance.kind() == "weights"`. Add a sibling helper `_inheritance_records_lineage()` returning `self.inheritance.kind() != "none"`.
- [x] 1.5 Refactor any other `isinstance(...)` checks found in 1.3 to use `kind()`. Pre-implementation grep confirms the only other site is a docstring reference at `inheritance.py:120` ('guards with `isinstance(strategy, NoInheritance)`') — updated the docstring text to reflect the `kind()`-based gate (now: 'guards with `strategy.kind() == "none"`').
- [x] 1.6 Add `test_inheritance_kind` test family to `test_inheritance.py`: assert all three impls return their declared literal; assert the literal is exactly one of the three (no leakage of new values). Note: BaldwinInheritance kind test added in task 2.5.

## 2. BaldwinInheritance implementation

- [x] 2.1 Implement `BaldwinInheritance` class in `inheritance.py`. No-arg constructor. `select_parents(gen_ids, fitnesses, generation) -> [best_genome_id]` (single-element list — top fitness, lex-tie-broken on `genome_id`; same selection rule as `LamarckianInheritance` with elite_count=1). `assign_parent() -> None` (Baldwin doesn't warm-start). `checkpoint_path() -> None` (no on-disk substrate). `kind() -> "trait"`. Module docstring updated to describe trait inheritance alongside the existing no-op and weight-flow descriptions.
- [x] 2.2 Refactor `_resolve_per_child_inheritance` at `loop.py:362-382` to a three-branch switch on `self.inheritance.kind()`. Existing `not self._inheritance_active()` early-return at line 362-363 becomes `if kind() == "none": return None, None, ""`. New middle branch: `if kind() == "trait": parent_id = self._selected_parent_ids[0] if self._selected_parent_ids else ""; return None, None, parent_id`. Existing weight-IO logic at lines 364-382 becomes `if kind() == "weights": ...`. Update the docstring to document all three branches.
- [x] 2.3 Split the M3 post-`tell` block at `loop.py:525-535` into two distinct guards (the M3 single-guard `if inheritance_on:` would skip Baldwin entirely under M4's redefined `_inheritance_active()`). New structure: (a) `if self._inheritance_records_lineage():` runs `next_selected = self.inheritance.select_parents(gen_ids, list(fitnesses), gen)` and `self._selected_parent_ids = next_selected` — fires for both Lamarckian and Baldwin; (b) `if self._inheritance_active():` runs the two `_gc_inheritance_dir` calls — fires for Lamarckian only. Removed the `inheritance_on = self._inheritance_active()` local at `loop.py:453` since it was only used in one place (the post-tell guard) and that guard is now two distinct guards.
- [x] 2.4 Re-export `BaldwinInheritance` from `packages/quantum-nematode/quantumnematode/evolution/__init__.py`.
- [x] 2.5 Add Baldwin unit tests to `test_inheritance.py`: kind value `"trait"`; `checkpoint_path` returns None; `assign_parent` returns None; `select_parents` returns `[best_genome_id]` matching `LamarckianInheritance(elite_count=1).select_parents` byte-for-byte for the same input (verifies the shared selection rule).

## 3. Brain field: weight_init_scale

- [x] 3.1 Add `weight_init_scale: float = Field(default=1.0, ge=0.1, le=5.0)` field to `LSTMPPOBrainConfig` in `packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py` (around line 102, alongside `entropy_decay_episodes`). Field comment explains it's an evolvable innate-bias knob that scales the orthogonal-init `gain` for the actor's hidden Linear layers and the critic's Linear layers; the actor's output layer's small-init `gain=0.01` is preserved (standard PPO trick); the LSTM/GRU module is unaffected. Default 1.0 is byte-equivalent to existing init.
- [x] 3.2 Modify `_initialize_weights()` at `lstmppo.py:471-488` to thread `self.config.weight_init_scale` into the orthogonal-init gain. Specifically: change `nn.init.orthogonal_(module.weight, gain=np.sqrt(2))` to `nn.init.orthogonal_(module.weight, gain=np.sqrt(2) * self.config.weight_init_scale)` for the actor's hidden Linears (via `init_linear`) and the critic's Linears. The actor's output-layer init at `gain=0.01` (lines 484-488) MUST remain unchanged. The LSTM/GRU at `self.rnn` is not touched by `_initialize_weights` and remains on PyTorch's default init.
- [x] 3.3 Add `test_lstmppo_weight_init_scale.py` (8 tests) covering: (a) `weight_init_scale=1.0` produces tensors bit-identical to a paired same-seed reference (no-op for actor hidden Linears, actor output layer, critic Linears, and LSTM/GRU); (b) `weight_init_scale=2.0` produces actor-hidden and critic Linear weights with std exactly 2× scale=1.0 (rel tolerance 1e-5); (c) actor output layer std unchanged; (d) RNN params unchanged; (e+f) field validator rejects values \<0.1 or >5.0; (g) inclusive boundary values 0.1 and 5.0 accepted; (h) default 1.0.

## 4. Early stop on saturation

- [x] 4.1 Add `early_stop_on_saturation: int | None = Field(default=None, ge=1)` field to `EvolutionConfig` in `packages/quantum-nematode/quantumnematode/utils/config_loader.py`.
- [x] 4.2 Add `--early-stop-on-saturation N` CLI flag to `scripts/run_evolution.py` mirroring the existing `--algorithm` / `--inheritance` override pattern. CLI override applied via `_resolve_evolution_config` overrides dict; explicit non-positive value rejected with a clear `SystemExit` message before reaching the loop.
- [x] 4.3 Added `self._gens_without_improvement: int = 0` and `self._last_best_fitness: float | None = None` instance attributes to `EvolutionLoop`. Counter-update placed IMMEDIATELY after `optimizer.tell` and BEFORE the lineage-tracking guard, with explicit gen-1 bootstrap handling for the `None` case (treat as improvement, counter stays 0, bootstrap value recorded).
- [x] 4.4 Placed the early-stop check at the END of the main loop body, AFTER `self._generation += 1` and the `checkpoint_every` save. `_generation` reflects the post-evaluation increment value so resume continues from the correct point. `last_improvement_gen` computed from `_generation - _gens_without_improvement` (the gen index where the counter last reset); falls back to `"no improvement observed"` if `_last_best_fitness is None`.
- [x] 4.5 Persisted `_gens_without_improvement` (int) and `_last_best_fitness` (float | None) in the checkpoint pickle. Bumped `CHECKPOINT_VERSION` 2 → 3. The existing post-loop `_save_checkpoint()` already fires after `break` (control flows through `finally:` then to the post-loop save site) so no extra save call inside the loop is needed.
- [x] 4.6 Added `test_early_stop.py` (4 tests): flat trajectory + N=3 → fires after gen 4; monotonic improvement → never fires; improves-then-stalls + N=5 → counter resets correctly, never fires; resume preserves both `gens_without_improvement` and `last_best_fitness`. Plus updated `test_loop_smoke.py::test_checkpoint_contains_required_keys` to expect the new pickle keys.

## 5. Config + CLI for Baldwin

- [x] 5.1 Extended `EvolutionConfig.inheritance` Literal in `config_loader.py` to `Literal["none", "lamarckian", "baldwin"]`.
- [x] 5.2 Extended `_validate_inheritance` for Baldwin: rules 1, 2, 4 apply to any non-`none` inheritance. Rule 3 (`inheritance_elite_count != 1`) restricted to `lamarckian` only — Baldwin ignores the field. Validator messages updated to be Baldwin-aware. The `_validate_hyperparam_schema` architecture-changing-fields rejection also restricted to `lamarckian` only (Baldwin doesn't load weights so shape mismatches are fine).
- [x] 5.3 Extended `--inheritance` CLI flag choices in `scripts/run_evolution.py` to include `"baldwin"`. Updated help text to describe the trait-flow vs weight-flow distinction.
- [x] 5.4 Extended the strategy construction to include `BaldwinInheritance()` for the `"baldwin"` arm. Added explicit `InheritanceStrategy` type annotation on the `inheritance` local so pyright accepts the three-branch construction.
- [x] 5.5 Updated the CLI guard's error message at `scripts/run_evolution.py` to cover both Lamarckian (TypeError on weight_capture_path kwarg) and Baldwin (signal collapse on frozen-eval) failure modes when `inheritance != "none" + --fitness success_rate`.
- [x] 5.6 Added 5 Baldwin validator tests to `test_config.py`: (a) Baldwin + `learn_episodes_per_eval=0` raises; (b) Baldwin + `warm_start_path` raises; (c) Baldwin + no `hyperparam_schema` raises; (d) Baldwin + arch-changing field is ALLOWED (documented difference vs Lamarckian); (e) Baldwin ignores `elite_count != 1` but still enforces `> population_size`.
- [x] 5.7 Added Baldwin smoke test to `test_loop_smoke.py`: 3-gen × pop 4 Baldwin run via a new `_make_baldwin_loop` helper. Asserts no `inheritance/` directory; gen-0 rows have empty `inherited_from`; gen-1+ rows share a single non-empty `inherited_from` value (the prior-gen elite ID).

## 6. Spec sync to evolution-framework

- [x] 6.1 Confirmed the spec delta at `openspec/changes/add-baldwin-evolution/specs/evolution-framework/spec.md` covers all three new requirement scenarios (Baldwin Inheritance, Early Stop, modified Inheritance Strategy) per the `openspec validate --strict` pass that landed with the proposal.
- [x] 6.2 Update the existing `openspec/specs/evolution-framework/spec.md` "Inheritance Strategy" requirement directly is NOT done in this PR — the delta sync runs at archival time per the openspec-archive-change skill workflow.

## 7. Pre-pilot smoke

- [x] 7.1 Ran the Baldwin smoke at `--generations 3 --population 6 --seed 42` on the Baldwin pilot YAML. Output to `evolution_results/m4_smoke_baldwin/`. Wall-time ~90s.
- [x] 7.2 Verified smoke artefacts: lineage.csv has 18 rows + header (3 gens × 6 pop = 18 ✓); gen-0 rows have empty `inherited_from` ✓; gen-1 children all share `inherited_from = 8fee7485...` (the gen-0 elite with fitness 0.84) ✓; gen-2 children all share `inherited_from = 14e14e32...` (the gen-1 elite — all gen-1 children had fitness 0.0 so lex-tie-break selected the lex-first ID, confirming the selection rule) ✓; history.csv has 3 rows ✓; NO `inheritance/` directory exists ✓.
- [x] 7.3 Ran a 4-gen smoke exercising `--early-stop-on-saturation 2 --generations 10 --population 4` on the small MLPPPO config. History trajectory: `[0.0, 0.0, 0.5, 0.0, 0.0]` — counter walked 0 (bootstrap), 1, 0 (improvement at gen 3), 1, 2 → fired after gen 5 ✓. Loop terminated at 5 gens (well before the 10-gen budget).

## 8. Pilot configs

- [x] 8.1 Created `configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml`. Structurally identical to `configs/evolution/lamarckian_lstmppo_klinotaxis_predator_control.yml` (the M3 control YAML with same brain/env/budget/seeds/TPE) with three diffs: (a) `inheritance: baldwin`, (b) added `weight_init_scale` (bounds [0.5, 2.0]) and `entropy_decay_episodes` (bounds [200, 2000]) to `hyperparam_schema`, (c) added `early_stop_on_saturation: 5`. YAML loads cleanly under all validators.
- [x] 8.2 Header comments explain the Baldwin framing (trait-only inheritance, no weight checkpoints), enumerate the three diffs vs the M3 control YAML, document the new evolvable knobs, and embed the GO/PIVOT/STOP decision-gate framing.

## 9. Campaign scripts

- [x] 9.1 Created `scripts/campaigns/phase5_m4_baldwin_lstmppo_klinotaxis_predator.sh`. 4-seed (42-45) loop, output to `evolution_results/m4_baldwin_lstmppo_klinotaxis_predator/`. Header comments document the 4-arm comparison structure and the F1 post-pilot evaluator.
- [x] 9.2 Created `scripts/campaigns/phase5_m4_lamarckian_rerun.sh`. Reuses M3's `lamarckian_lstmppo_klinotaxis_predator_pilot.yml`, adds `--early-stop-on-saturation 5` at runtime to match the Baldwin pilot's behaviour. Output to `evolution_results/m4_lamarckian_lstmppo_klinotaxis_predator/`.
- [x] 9.3 Created `scripts/campaigns/phase5_m4_control_rerun.sh`. Reuses M3's `lamarckian_lstmppo_klinotaxis_predator_control.yml`, adds `--early-stop-on-saturation 5` at runtime. Output to `evolution_results/m4_control_lstmppo_klinotaxis_predator/`.
- [x] 9.4 The hand-tuned baseline reuses M2.11's existing run (`evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline/`) — no script needed; the aggregator points at it via `--baseline-root`.

## 10. Run pilot

- [x] 10.1 Ran all three campaign scripts in parallel. Total wall-time ~67 min (3 arms × 4 seeds, with internal parallel=4 + 3 scripts in background → ~12 worker procs against 10 cores). Early-stop fired on every arm-seed combo (none reached the 20-gen budget); saved roughly half the per-seed wall.
- [x] 10.2 Verified per-seed `lineage.csv` row counts match early-stop generation counts; Baldwin arm's output directory does NOT contain an `inheritance/` subdirectory across any seed (Baldwin is mechanically a no-op on weight IO).

## 11. F1 post-pilot evaluator

- [x] 11.1 Created `scripts/campaigns/baldwin_f1_postpilot_eval.py` with all required args (`--baldwin-root`, `--config`, `--seeds`, `--episodes`, `--output-dir`).
- [x] 11.2 Implemented per-seed flow: (a) read `best_params.json` from `<baldwin_root>/seed-N/<latest_session>/best_params.json`; (b) load YAML for the `hyperparam_schema`; (c) construct a synthetic `Genome` with `birth_metadata=build_birth_metadata(sim_config)` so the encoder's schema-required `param_schema` key is populated; (d) call `EpisodicSuccessRate().evaluate(...)` which internally invokes `encoder.decode` for brain construction.
- [x] 11.3 Writes `f1_innate_only.csv` with columns `seed, success_rate, elite_genome_id`. Also prints the mean F1 success rate to stdout for easy aggregation against the baseline.
- [x] 11.4 Smoke-tested against the Baldwin pre-pilot smoke output: produced a valid CSV with success_rate in [0.0, 1.0] and confirmed the script handles the per-seed `seed-N/<session>/` directory structure produced by the campaign script.

## 12. Aggregator + verdict

- [x] 12.1 Created `scripts/campaigns/aggregate_m4_pilot.py` with all required args (`--baldwin-root`, `--lamarckian-root`, `--control-root`, `--baseline-root`, `--f1-csv`, `--seeds`, `--output-dir`). Mirrors the M3 aggregator's structure (helper functions reused from the same idiom) but extended to 3 evolution arms + the F1 CSV.
- [x] 12.2 Implemented three-gate verdict computation in `_compute_verdict`: (a) Speed (Baldwin vs control): `SPEED_GAIN_GENERATIONS = 2`; (b) Genetic-assimilation (F1 vs baseline): `F1_OVER_BASELINE_THRESHOLD = 0.10`; (c) Comparative (Baldwin vs Lamarckian): `COMPARATIVE_GAP_GENERATIONS = 4`. All thresholds are module-level constants with header-comment calibration notes referencing M3's published numbers (control mean 9.75, baseline 0.17, lamarckian 4.5). Verdict: GO if all three; PIVOT if speed only with at least one of assimilation/comparative failing; STOP if speed fails.
- [x] 12.3 4-curve plot: Baldwin + Lamarckian + control mean ± std band, hand-tuned baseline + 0.92 target as horizontal reference lines. F1 innate-only success rates as separate per-seed star markers (offset to x = max_gen + 0.5 so they don't overlap the trajectory curves). Aggregator's `_stack` truncates to the shortest within-arm history (handles early-stop heterogeneity); the cross-arm alignment then truncates all to the shortest arm.
- [x] 12.4 Per-seed table written to `convergence_speed.csv` with columns `seed, baldwin_gen_to_092, lamarckian_gen_to_092, control_gen_to_092, f1_innate_only_success_rate`. Empty cells for seeds that never reach the threshold.
- [x] 12.5 `summary.md` includes the three gate computations (PASS/FAIL + margin), per-seed table, and the verdict + summary text.
- [x] 12.6 Aggregator unit tests skipped — same pattern as the M3 aggregator (`scripts/campaigns/aggregate_m3_pilot.py` ships without unit tests; verdict logic is validated by smoke-running the aggregator against real pilot output post-run, which is task 12 of the pilot run group). The `# pragma: no cover` directive on the aggregator file documents this convention. Adding tests now would require a sys.path hack to import from `scripts/campaigns/` — not worth the maintenance burden for a forensic post-pilot script with a single linear `_compute_verdict` decision tree.

## 13. Logbook 014

- [x] 13.1 Created `docs/experiments/logbooks/014-baldwin-inheritance-pilot.md` with full structure (Status / Background / Hypothesis / Method / Results / Analysis / Decision / Conclusions / Next Steps / Data References). Verdict STOP documented with all three gate margins; analysis discusses why Baldwin failed (weight transfer is what causally drives M3's acceleration; trait flow alone doesn't replicate it).
- [x] 13.2 Created `docs/experiments/logbooks/supporting/014/baldwin-inheritance-pilot-details.md` with per-seed final-fitness tables, per-arm gen-to-0.92 trajectories, decoded evolved-hyperparameter values per seed (showing TPE genuinely explored `weight_init_scale` 0.57-1.33 + `entropy_decay_episodes` 1022-1562), F1 innate-only forensic discussion (why 0.0 across all seeds), wall-time breakdown, schema confounder check, cross-arm code-revision check.
- [x] 13.3 Copied pilot artefacts to `artifacts/logbooks/014/m4_baldwin_pilot/{baldwin,lamarckian,control,baseline,summary}/` mirroring the M3 layout. 60 files total: per-seed `best_params.json` + `history.csv` + `lineage.csv` + `checkpoint.pkl` for all three arms; Lamarckian's surviving final-gen elite under `inheritance/`; baseline `seed-N.log` files; aggregator's `summary.md` + `convergence.png` + `convergence_speed.csv` + `f1_innate_only.csv`.
- [x] 13.4 Added M4 logbook entry to `docs/experiments/README.md` active-experiments table with the STOP verdict + key per-arm numbers + the "negative result isolates source of M3's lift to weight transfer" framing.

## 14. Tracker + roadmap

- [x] 14.1 Ticked M4.1-M4.9 in `openspec/changes/2026-04-26-phase5-tracking/tasks.md` (M4.8 + M4.9 added during planning). Flipped M4 status from `not started` to `complete` with the STOP verdict + per-gate margins. M4.2's scope-change note used the M3.4-style strikethrough pattern.
- [x] 14.2 Flipped M4 row in `docs/roadmap.md` from 🟢 unblocked to ✅ complete with the STOP verdict summary.

## 15. PR + archive

- [x] 15.1 Ran `uv run pre-commit run` on the changed files clean.
- [x] 15.2 Ran `uv run pytest -m "not nightly" --tb=short -q` clean — 2333 tests pass (28 new tests added across the M4 framework groups; M3 paths byte-equivalent).
- [x] 15.3 Opened draft PR #139 ("feat(m4): Baldwin Effect framework + INCONCLUSIVE pilot + audit"). Reviewed across 5 rounds + CI debugging (the gc.collect() autouse fixture in `evolution/conftest.py` was the eventual root-cause fix for the cma-library reference cycle that was causing CI runner-agent timeouts). Merged to main as commit `fef7610f`.
- [x] 15.4 Archived after merge. Originally deferred during the M4 PR per the audit downgrade (the `add-baldwin-evolution` change was kept open as substrate for M4.5). Now archived because the M4.5 follow-up will land as a separate, narrowly-scoped OpenSpec change (`add-baldwin-retry`) on its own branch — extending this change would have repeated task groups + muddled the audit trail. Archive happens here in a small standalone PR so M4.5's scope stays clean.

## 16. Post-pilot audit + INCONCLUSIVE downgrade (added in this PR)

- [x] 16.1 Conducted post-pilot audit on the literal STOP verdict from the aggregator. Found three blocking design flaws (A1: schema-shift confounder; A2: F1 test biologically incoherent; A3: F1 baseline apples-to-oranges) plus two significant issues (A4: n=4 seeds underpowered; A5: chosen knobs may not be optimal for K=50).
- [x] 16.2 Updated logbook 014 (`docs/experiments/logbooks/014-baldwin-inheritance-pilot.md`) to reflect the INCONCLUSIVE verdict: replaced the Decision/Conclusions/Next Steps sections with audit-grounded text; preserved the literal aggregator STOP output for traceability; added § Audit section with the five findings; added M4.5 follow-up plan to § Next Steps.
- [x] 16.3 Updated supporting appendix (`docs/experiments/logbooks/supporting/014/baldwin-inheritance-pilot-details.md`) with the schema-shift evidence (per-arm gen-0 best-fitness table showing Baldwin -0.14pp deficit), the F1 design-failure analysis, the sample-size discussion, and the knob-choice rationale for M4.5.
- [x] 16.4 Updated `docs/experiments/README.md` M4 entry from "STOP" to "completed (framework shipped) — INCONCLUSIVE on the science" with audit summary.
- [x] 16.5 Updated `docs/roadmap.md` M4 row to "✅ framework shipped, ⚠️ INCONCLUSIVE on the science" with audit summary + M4.5 forward reference.
- [x] 16.6 Updated tracker (`openspec/changes/2026-04-26-phase5-tracking/tasks.md`): M4 status flipped from STOP to INCONCLUSIVE with audit summary; added M4.10 task for the audit; added a new M4.5 section with 7 sub-tasks scoping the proper retry.
- [x] 16.7 Updated this OpenSpec change's tasks (this file): M4.2/M4.3/M4.5/M4.9 task notes amended to reference the audit findings; task 15.4 changed to DEFERRED (not archiving the change in this PR).
