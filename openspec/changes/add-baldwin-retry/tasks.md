# Tasks: Baldwin Effect first valid measurement

## 1. Spec self-review (gate before implementation)

- [ ] 1.1 Run `openspec validate add-baldwin-retry` and confirm clean.
- [ ] 1.2 Re-read `proposal.md` + `design.md` + `specs/evolution-framework/spec.md` end-to-end and check for: (a) ambiguity in the F1 evaluator scenario's CLI contract, (b) drift between proposal's "modified capabilities = none" and the actual MODIFIED requirement (the F1 evaluator paragraph + new scenario IS a spec-level change to the existing `Baldwin Inheritance Strategy` requirement; this is consistent with Decision 0's escape hatch — document the reasoning so reviewers don't see a contradiction).
- [ ] 1.3 Cross-check Decision 1's schema bounds (`actor_hidden_dim ∈ [64, 256]`, `actor_num_layers ∈ [1, 3]`) against the existing `LSTMPPOBrainConfig` field validators. Confirm the bounds fit within the brain config's accepted range and that the encoder's `genome_dim` calculation handles 8 fields cleanly.
- [ ] 1.4 Cross-check Decision 5's gate thresholds against M4's measured numbers (control mean = 8.50 from logbook 014; M4.5 will recompute against its own control arm rather than M4's number). Confirm the +0.05 F1 threshold is defensible against expected post-K'-train variance — record the reasoning in this task description for the post-pilot audit.

## 2. F1 evaluator redesign

- [ ] 2.1 Rewrite `scripts/campaigns/baldwin_f1_postpilot_eval.py` per the new spec scenario. Remove the K=0 frozen-eval path entirely; replace with the paired K'-train + L-eval comparison defined in the F1 evaluator scenario. Use `LearnedPerformanceFitness.evaluate` for both elite and baseline runs (don't reach into the loop's internals — go through the same fitness harness the pilot uses).
- [ ] 2.2 Synthesise the baseline genome via `HyperparameterEncoder.initial_genome(sim_config).params` so the baseline is the encoder's defaults under the same schema (apples-to-apples with elite per the design's symmetry constraint).
- [ ] 2.3 Add CLI arguments `--k-prime` (default 10) and `--episodes` (default 25) per the spec scenario. Reject `<= 0` at argparse-time with a clear error message. Honour the existing `--baldwin-root`, `--config`, `--seeds`, `--output-dir` arguments unchanged.
- [ ] 2.4 Implement append-mode CSV writing per the spec. Output is `f1_learning_acceleration.csv` with columns `seed, k_prime, episodes, elite_success_rate, baseline_success_rate, signal_delta`. If the file exists, append (don't overwrite); if not, create with header. Re-running with K' = 25 after K' = 10 SHALL produce a CSV containing both sets of rows.
- [ ] 2.5 Pin per-seed RNG seed across elite and baseline evaluations so the only between-arm difference is the genome. Use the seed argument as the run seed for both `LearnedPerformanceFitness.evaluate` calls within a seed.
- [ ] 2.6 Add unit tests at `packages/quantum-nematode/tests/quantumnematode_tests/scripts/test_baldwin_f1_postpilot_eval.py` covering: (a) argparse rejects `--k-prime 0` and `--episodes 0`; (b) `initial_genome().params` round-trips through the encoder cleanly under an 8-field schema; (c) append-mode CSV preserves prior rows when invoked twice with different K' values; (d) per-seed RNG plumbing produces deterministic results across re-runs.
- [ ] 2.7 Run `uv run pre-commit run --files scripts/campaigns/baldwin_f1_postpilot_eval.py packages/quantum-nematode/tests/quantumnematode_tests/scripts/test_baldwin_f1_postpilot_eval.py` clean.

## 3. Pilot configs (8-field schema, n = 8 seeds)

- [ ] 3.1 Create `configs/evolution/baldwin_lstmppo_klinotaxis_predator_retry_pilot.yml`. Base on the existing `configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml` (M4) with these diffs: extend `hyperparam_schema` from 6 to 8 fields per Decision 1's table (add `actor_hidden_dim ∈ [64, 256]` and `actor_num_layers ∈ [1, 3]`); keep `inheritance: baldwin` + `early_stop_on_saturation: 5`. Document the schema diff inline as a comment.
- [ ] 3.2 Create `configs/evolution/control_lstmppo_klinotaxis_predator_retry_pilot.yml`. Identical to the Baldwin config except `inheritance: none`. Schema is the SAME 8 fields per Decision 2 (audit A1 closure).
- [ ] 3.3 The Lamarckian rerun reuses the existing `configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml` (4-field schema) — its purpose is reproducing M3 framework integrity under the M4.5 code revision, not direct comparison to Baldwin/Control. No new YAML needed.
- [ ] 3.4 The hand-tuned baseline reuses M2.11's existing artefacts (no re-run needed; reproducible per logbook 013 § 9.6).
- [ ] 3.5 Validate both new YAMLs parse cleanly: `uv run python -c "from quantumnematode.utils.config_loader import load_simulation_config; load_simulation_config('configs/evolution/baldwin_lstmppo_klinotaxis_predator_retry_pilot.yml')"` (and same for control). Confirm the schema-equalisation property at YAML level: both YAMLs SHALL have identical `hyperparam_schema` blocks (modulo the `inheritance` field).

## 4. Campaign scripts

- [ ] 4.1 Create `scripts/campaigns/phase5_baldwin_retry_baldwin.sh`. Pattern matches `scripts/campaigns/phase5_m4_baldwin_lstmppo_klinotaxis_predator.sh` (M4) but extends to n = 8 seeds (42-49) and points at the new 8-field config.
- [ ] 4.2 Create `scripts/campaigns/phase5_baldwin_retry_control.sh`. Same structure as the Baldwin script but uses the new control config (8-field, `inheritance: none`).
- [ ] 4.3 Create `scripts/campaigns/phase5_baldwin_retry_lamarckian_rerun.sh`. Re-uses M3 lamarckian config (existing 4-field) under the M4.5 code revision; n = 8 seeds. Output to a distinct directory so M4 lamarckian artefacts are preserved.
- [ ] 4.4 Each campaign script SHALL document its expected wall-time per the design's compute estimate (~2.5-3 hours per arm at n = 8 with parallel = 4 + early-stop saturation enabled).
- [ ] 4.5 Run `uv run pre-commit run --files scripts/campaigns/phase5_baldwin_retry_*.sh` clean.

## 5. Aggregator updates

- [ ] 5.1 Decide: extend `scripts/campaigns/aggregate_m4_pilot.py` (consume the existing 4-arm interface) OR create a new `scripts/campaigns/aggregate_baldwin_retry_pilot.py`. Recommend the new file — the M4 aggregator is now archival-bound (logbook 014 references it); a new file keeps M4's audit trail clean. Document the choice as a decision comment at the top of the new aggregator.
- [ ] 5.2 Implement the schema-equalisation pre-flight check per Decision 2. The aggregator's first output SHALL be the gen-0 fitness convergence check; if `|Δ| > 0.05` between Baldwin and Control gen-0 means, FORCE the verdict to INCONCLUSIVE with a clear "audit A1 not resolved" message and skip the gates entirely.
- [ ] 5.3 Implement the recalibrated 3-gate verdict per Decision 5: speed gate `+2`, F1 gate `+0.05`, comparative gate `+4`. Reference values come from the M4.5 pilot's own arms (not M4's published numbers). GO if all three; PIVOT if speed only; STOP otherwise.
- [ ] 5.4 Update the aggregator's per-seed table to handle the F1 evaluator's new CSV format (reads `f1_learning_acceleration.csv` with the new columns; filter by `k_prime` if multiple K' rows exist — default to the K' = 10 rows for the headline gate).
- [ ] 5.5 Generate the 4-curve convergence plot (Baldwin + Control + Lamarckian + baseline as horizontal line) and write to `convergence.png` in the output directory. The convergence-curve treatment matches the M4 aggregator's pattern (mean ± std band per arm).
- [ ] 5.6 Write `summary.md` with: schema-equalisation check result, per-arm gen-to-0.92 means + per-seed table, 3 gates with margins + verdict. Same format as M4's summary.md for diff-comparability.
- [ ] 5.7 Add unit tests for the aggregator's gate logic at `packages/quantum-nematode/tests/quantumnematode_tests/scripts/test_aggregate_baldwin_retry.py`: (a) schema-equalisation check fires correctly above and below the 0.05 threshold; (b) GO / PIVOT / STOP verdict logic produces the right output for each gate combination; (c) the K' filter on F1 CSV reads the right rows when multiple K' values are present.

## 6. Pre-pilot smoke (gate before full pilot launch)

- [ ] 6.1 Smoke-test the schema-equalisation property: 1 seed, 1 generation each for Baldwin and Control under the new 8-field configs. Confirm gen-0 best fitness is bit-identical between the two arms (both should produce the same TPE samples since seed + schema are now identical). If gen-0 differs, debug before committing to the full pilot.
- [ ] 6.2 Smoke-test the F1 evaluator: take the smoke-pilot's Baldwin elite, run `baldwin_f1_postpilot_eval.py` with K' = 10 and K' = 25 in succession. Confirm CSV append-mode preserves both sets of rows. Confirm both elite and baseline produce non-zero success rates at K' = 10 (per Risk 4 — if both are near zero, fall back to K' = 25 as the default before full pilot launch).
- [ ] 6.3 Smoke-test 8-field TPE convergence rate: 3 generations × pop 6 × seed 42 on the 8-field Baldwin config. Estimate whether full-pilot needs 20 / 25 / 30 gens per Risk 1. If TPE is still climbing rapidly at gen 3, plan for 30; if saturated, plan for 20. Pin the gen budget in the campaign scripts before launching the full pilot.
- [ ] 6.4 Smoke-test wall-time projection: extrapolate from smoke per-gen wall to full pilot wall. If projection > 6 hours, drop n from 8 to 6 per Risk 5 (and document the trade-off in the campaign script comments).

## 7. Pre-pilot user review checkpoint

- [ ] 7.1 Hand the user: (a) the two new pilot YAMLs, (b) the rewritten F1 evaluator script, (c) the smoke results from §6, (d) the projected wall-time and gen budget. Wait for explicit approval before launching the full pilot. This is the design checkpoint promised in the proposal — surfaces config / evaluator design issues BEFORE the ~3-hour pilot wall instead of after.

## 8. Run full pilot

- [ ] 8.1 Launch all three arms (Baldwin + Control + Lamarckian rerun) in parallel via the three campaign scripts. Total expected wall ~3-4 hours. Each arm runs n = 8 seeds with early-stop on saturation enabled.
- [ ] 8.2 Monitor pilot progress (per-seed wall-time, early-stop firing, any crashes). If any seed crashes, inspect the log; do NOT silently retry — investigate the failure mode first (e.g. the M4 PR's gc.collect() conftest fixture exists in main and should keep memory pressure bounded; if a crash isn't memory-related, dig deeper).
- [ ] 8.3 On pilot completion, run the F1 evaluator on the Baldwin pilot's seed roots: `scripts/campaigns/baldwin_f1_postpilot_eval.py --baldwin-root <baldwin_root> --config <baldwin_yaml> --k-prime 10 --episodes 25 --output-dir <out>`. Then run again with `--k-prime 25` to provide the K' = 25 fallback measurement (per Risk 4 mitigation; both sets coexist in the CSV).
- [ ] 8.4 Run the aggregator on all four arm roots + the F1 CSV: `scripts/campaigns/aggregate_baldwin_retry_pilot.py --baldwin-root <...> --control-root <...> --lamarckian-root <...> --baseline-root <...> --f1-csv <...> --output-dir <...>`. Inspect `summary.md` output for the gates' verdict + the schema-equalisation check.

## 9. Post-pilot REVIEW CHECKPOINT (gate before logbook 015)

- [ ] 9.1 Compile a complete post-pilot working summary covering: setup, smoke results, pilot wall-times, schema-equalisation check status, per-seed gen-to-0.92 + F1 numbers for both K' = 10 and K' = 25, 3-gate verdict + margins, robustness checks (population-mean trajectory, seed-by-seed elite hyperparam analysis to test whether arch knobs were preferred over M4 knobs per Risk 3, F1 sensitivity across K' values). Propose a verdict but explicitly mark it as PROPOSED, not COMMITTED.
- [ ] 9.2 **Present the working summary to the user** with a message: "Aggregator says VERDICT. Schema-equalisation check: PASS / FAIL. Gate margins: a / b / c. Robustness checks show: … Before I write logbook 015, do you want me to proceed with VERDICT, or reconsider?" This is the post-pilot review checkpoint promised in the proposal.
- [ ] 9.3 Wait for explicit user approval ("ship verdict X" or equivalent) BEFORE writing the logbook's Decision / Conclusions / Next Steps sections. Do NOT auto-finalise.

## 10. Logbook 015 (post-approval)

- [ ] 10.1 Create `docs/experiments/logbooks/015-baldwin-retry.md` (main logbook) and `docs/experiments/logbooks/supporting/015/baldwin-retry-details.md` (supporting appendix). Mirror the structure of logbook 014 (Objective / Background / Hypothesis / Method / Results / Analysis / Conclusions / Next Steps / Data References).
- [ ] 10.2 Forward-reference audit findings A1-A5 from logbook 014 in the Background section. Make explicit which audit finding each M4.5 design change addresses (Decision 1 → A5, Decision 2 → A1, Decision 3 → A2 + A3, Decision 4 → A4, Decision 5 → recalibration).
- [ ] 10.3 Record the schema-equalisation check result. If PASS, document the actual `|Δ|` for transparency. If FAIL, the verdict is INCONCLUSIVE and the logbook explains what's still confounding gen-0.
- [ ] 10.4 Document both K' = 10 and K' = 25 F1 results. The gate uses K' = 10; K' = 25 is documented as the calibration sensitivity check (per Risk 4).
- [ ] 10.5 Document the head-to-head finding from Decision 1: which of the 8 fields TPE explored vs pinned at defaults across seeds. This answers audit finding A5 — were arch knobs the right call, or did M4's knobs suffice once the experimental design was correct?
- [ ] 10.6 Write the Decision section per the user-approved verdict from §9. If GO: M5 / M6 unblocked with Baldwin in the substrate. If PIVOT: scope a follow-up. If STOP: pre-registered downstream consequences (M5 proceeds without Baldwin, M6 uses Lamarckian) per Decision 6.
- [ ] 10.7 Copy artefacts to `artifacts/logbooks/015/baldwin_retry_pilot/{baldwin,control,lamarckian,baseline,summary}/`. Same layout as logbook 014's artefact tree for cross-logbook diff-comparability.
- [ ] 10.8 Update `docs/experiments/README.md` index with the new logbook entry.
- [ ] 10.9 Append a one-line "Update" pointer at the top of logbook 014 linking to logbook 015 (per the M4 OpenSpec design's promise: "When M4.5 lands, append a one-line 'Update' at the top of logbook 014 pointing to logbook 015").

## 11. Tracker + roadmap update

- [ ] 11.1 Tick M4.5.1 through M4.5.7 in `openspec/changes/2026-04-26-phase5-tracking/tasks.md`. Each task description should reference the M4.5 design decision or task group that fulfilled it (e.g. M4.5.1 → Decisions 1 + 2 + task group 3).
- [ ] 11.2 Update `openspec/changes/2026-04-26-phase5-tracking/tasks.md` M4 row from INCONCLUSIVE to whatever the M4.5 verdict supports (per task M4.5.7).
- [ ] 11.3 Update `docs/roadmap.md` M4 row. If GO: confirm the row is now ✅ complete with M4.5 as the corrective re-run. If STOP: the row gets a "STOP after M4.5 retry — Baldwin not exhibited on this testbed; M5 proceeds without Baldwin in pipeline" annotation.

## 12. Pre-PR verification

- [ ] 12.1 Run `uv run pre-commit run -a` clean.
- [ ] 12.2 Run `uv run pytest -m "not nightly" --tb=short -q` clean. Confirm no regression on existing tests; new tests in §2.6 + §5.7 are passing.
- [ ] 12.3 Run `openspec validate add-baldwin-retry` clean.
- [ ] 12.4 Self-review the branch diff one last time. Check for: any milestone refs (M4 / M4.5 / Phase 5 / RQ) that leaked into implementation code or non-campaign-script docstrings (per the existing memory rule); any TODO / FIXME / XXX left in code; any temporary debug print / breakpoint statements.
- [ ] 12.5 Commit any final cleanup; push branch.

## 13. Open PR

- [ ] 13.1 Open draft PR via `gh pr create --draft`. Title: "feat(baldwin-retry): first valid Baldwin Effect measurement on the predator testbed (verdict: VERDICT)". Body summarises: the audit findings A1-A5 that motivated M4.5; the design changes that addressed each; the pilot results + verdict; links to logbook 015. Awaiting user approval to push.

## 14. Post-merge

- [ ] 14.1 After PR merges, archive the OpenSpec change via `openspec-archive-change` skill. This includes the delta sync from `specs/evolution-framework/spec.md` into `openspec/specs/evolution-framework/spec.md` (the F1 evaluator paragraph + new scenario).
- [ ] 14.2 Confirm tracker + roadmap reflect the post-merge state. Phase 5 milestone progress checkpoint per the standing tracker convention.
