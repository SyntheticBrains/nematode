# Tasks: Baldwin Effect first valid measurement

## 1. Spec self-review (gate before implementation)

- [x] 1.1 Ran `openspec validate add-baldwin-retry` and `openspec validate --all` — both clean (`evolution-framework`, `add-baldwin-retry`, `phase5-tracking` all ✓).
- [x] 1.2 Re-read all four artifacts. F1 evaluator CLI contract is unambiguous after round-1 fix B2 (sim_config plumbing pattern explicitly described). Proposal's "modified capabilities = none expected" is consistent with the actual MODIFIED requirement via Decision 0's escape hatch (correcting M4's design-flawed F1 spec is a "bug fix" per the policy, not a new framework feature) — the qualifier "expected" in proposal § Capabilities leaves room for the bug-fix scope.
- [x] 1.3 Verified schema bounds against `LSTMPPOBrainConfig`. Both new arch knobs (`actor_hidden_dim`, `actor_num_layers`) exist as `int` fields with no Pydantic Field constraints. Bound `actor_hidden_dim ∈ [64, 256]` covers 4× the field default (64); the only existing dim validator is `lstm_hidden_dim < 2` rejected at line 129, well below our range. Bound `actor_num_layers ∈ [1, 3]` covers ±1 around the default (2); checked the constructor at lstmppo.py:373 — `actor_num_layers=1` produces a standard 1-hidden-layer MLP (loop iterates 0 times); `actor_num_layers=3` produces 3 hidden Linears. Both extremes safe. `HyperparameterEncoder.genome_dim` is `len(sim_config.hyperparam_schema)` — trivially returns 8.
- [x] 1.4 Cross-checked Decision 5's gate thresholds. Speed gate `+2` is roughly 2-3σ at n=8 per Decision 4's SE arithmetic (SE ≈ 0.6-0.9 gens). **F1 gate `+0.05` is ~1.4σ at n=8 (per-seed binomial sd at L=25 ≈ 0.1 at p=0.5; SE ≈ 0.035 across n=8 seeds; +0.05 ≈ 1.4σ).** This is tighter than the speed gate's σ multiplier — a truly null Baldwin effect would pass with ~10% probability under noise. The +0.05 threshold trades type-II for type-I error: tighter (+0.10 ≈ 3σ) would fail to detect a real-but-small Baldwin signal at K'=10 (which is an inherently small-budget test where the maximum elite-vs-baseline lift is bounded by what 10 episodes can buy). Defensible with this caveat recorded for the post-pilot audit.

## 2. F1 evaluator redesign

- [x] 2.1 Rewrote `scripts/campaigns/baldwin_f1_postpilot_eval.py` to the paired K'-train + L-eval comparison via `LearnedPerformanceFitness.evaluate`. K and L plumbed via a `sim_config` copy (`model_copy(update={"evolution": evolution.model_copy(update={"learn_episodes_per_eval": k_prime, "eval_episodes_per_eval": episodes})})`) — extracted into the `_build_sim_config_for_kprime` helper. K=0 frozen-eval path removed.
- [x] 2.2 Schema-prior baseline genome synthesised via `encoder.initial_genome(sim_config, rng=np.random.default_rng(seed))`, then mutated to a stable `genome_id` before evaluation. The same per-seed RNG seeds the env's per-episode trajectory in both runs (same seed → same env init), so the only between-arm difference is the genome.
- [x] 2.3 CLI: added `--k-prime` (default 10) and `--episodes` (default 25). Both reject `<= 0` at argparse-time via `parser.error(...)`. `--baldwin-root`, `--config`, `--seeds` (default n=8: 42-49), `--output-dir` preserved.
- [x] 2.4 Append-mode CSV (`open("a", newline="")`): `f1_learning_acceleration.csv` with header `seed, k_prime, episodes, elite_success_rate, baseline_success_rate, signal_delta`. Header written only if the file doesn't yet exist. Re-running with K'=25 after K'=10 appends new rows without losing the prior K'=10 rows (verified by `test_csv_append_mode_preserves_prior_rows`).
- [x] 2.5 Per-seed RNG: same `seed` passed to both `LearnedPerformanceFitness.evaluate` calls within a seed. Verified by `test_initial_genome_same_seed_produces_identical_params` and `test_initial_genome_different_seeds_produce_different_params`.
- [x] 2.6 Unit tests at `packages/quantum-nematode/tests/quantumnematode_tests/evolution/test_baldwin_f1_postpilot_eval.py` (11 tests, all passing): (a) CLI rejection for `--k-prime 0/-5/0` and `--episodes 0`; (b) `initial_genome` returns 8-dim params under an 8-field schema + decode round-trip; (c) append-mode CSV preserves prior rows; (d) per-seed RNG determinism + seed-sensitivity; (e) `_resolve_session` handles both direct and nested layouts; (f) `_build_sim_config_for_kprime` mutates K + L correctly + rejects no-evolution config. Tests use `_make_8field_sim_config` to extend M4's 6-field YAML to 8 fields in-memory (M4.5's 8-field YAML is created in task group 3).
- [x] 2.7 Pre-commit clean on both files (mdformat, markdownlint, ruff check + format, pyright, pytest hook all ✓).

## 3. Pilot configs (8-field schema, n = 8 seeds)

- [ ] 3.1 Create `configs/evolution/baldwin_lstmppo_klinotaxis_predator_retry_pilot.yml`. Base on the existing `configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml` (M4) with these diffs: extend `hyperparam_schema` from 6 to 8 fields per Decision 1's table (add `actor_hidden_dim ∈ [64, 256]` and `actor_num_layers ∈ [1, 3]`); keep `inheritance: baldwin` + `early_stop_on_saturation: 5`. Document the schema diff inline as a comment.
- [ ] 3.2 Create `configs/evolution/control_lstmppo_klinotaxis_predator_retry_pilot.yml`. Identical to the Baldwin config except `inheritance: none`. Schema is the SAME 8 fields per Decision 2 (audit A1 closure).
- [ ] 3.3 The Lamarckian rerun reuses the existing `configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml` (4-field schema). Primary purpose: provide the comparative-gate baseline at n=8 (Baldwin vs Lamarckian per design Decision 5 — `mean_gen_baldwin_to_092 ≤ mean_gen_lamarckian_to_092 + 4`). Secondary purpose: reproducibility check on the n=4 subset (seeds 42-45) — Lamarckian numbers should match M3's published `[3, 4, 4, 7]` mean 4.50, confirming the M4.5 code revision is byte-equivalent for the M3 path. No new YAML needed.
- [ ] 3.4 The hand-tuned baseline reuses M2.11's existing artefacts (no re-run needed; reproducible per logbook 013 § 9.6).
- [ ] 3.5 Validate both new YAMLs parse cleanly: `uv run python -c "from quantumnematode.utils.config_loader import load_simulation_config; load_simulation_config('configs/evolution/baldwin_lstmppo_klinotaxis_predator_retry_pilot.yml')"` (and same for control). Confirm the schema-equalisation property at YAML level: both YAMLs SHALL have identical `hyperparam_schema` blocks (modulo the `inheritance` field).

## 4. Campaign scripts

- [ ] 4.1 Create `scripts/campaigns/phase5_baldwin_retry_baldwin.sh`. Pattern matches `scripts/campaigns/phase5_m4_baldwin_lstmppo_klinotaxis_predator.sh` (M4) but extends to n = 8 seeds (42-49) and points at the new 8-field config.
- [ ] 4.2 Create `scripts/campaigns/phase5_baldwin_retry_control.sh`. Same structure as the Baldwin script but uses the new control config (8-field, `inheritance: none`).
- [ ] 4.3 Create `scripts/campaigns/phase5_baldwin_retry_lamarckian_rerun.sh`. Re-uses M3 lamarckian config (existing 4-field) under the M4.5 code revision; n = 8 seeds (42-49) — the n=8 sweep provides the comparative-gate baseline at apples-to-apples sample size with Baldwin/Control (per task 3.3). Output to a distinct directory so M4 lamarckian artefacts are preserved.
- [ ] 4.4 Each campaign script SHALL document its expected wall-time per the design's compute estimate (~2.5-3 hours per arm at n = 8 with parallel = 4 + early-stop saturation enabled).
- [ ] 4.5 Run `uv run pre-commit run --files scripts/campaigns/phase5_baldwin_retry_*.sh` clean.

## 5. Aggregator updates

- [ ] 5.1 Decide: extend `scripts/campaigns/aggregate_m4_pilot.py` (consume the existing 4-arm interface) OR create a new `scripts/campaigns/aggregate_baldwin_retry_pilot.py`. Recommend the new file — the M4 aggregator is now archival-bound (logbook 014 references it); a new file keeps M4's audit trail clean. Document the choice as a decision comment at the top of the new aggregator.
- [ ] 5.2 Implement the schema-equalisation pre-flight check per Decision 2. The aggregator's first output SHALL be the first-evaluated-generation fitness convergence check (read from `history.csv` first data row OR `lineage.csv` rows where `generation == 0` — both refer to the same evaluations; note `history.csv` labels this as `generation = 1` while `lineage.csv` labels it as `generation = 0` per the framework's existing indexing conventions). If `|Δ| > 0.05` between Baldwin and Control first-gen means, FORCE the verdict to INCONCLUSIVE with a clear "audit A1 not resolved" message and skip the gates entirely.
- [ ] 5.3 Implement the recalibrated 3-gate verdict per Decision 5: speed gate `+2`, F1 gate `+0.05`, comparative gate `+4`. Reference values come from the M4.5 pilot's own arms (not M4's published numbers). GO if all three; PIVOT if speed only; STOP otherwise.
- [ ] 5.4 Update the aggregator's per-seed table to handle the F1 evaluator's new CSV format (reads `f1_learning_acceleration.csv` with the new columns; filter by `k_prime` if multiple K' rows exist — default to the K' = 10 rows for the headline gate).
- [ ] 5.5 Generate the 4-curve convergence plot (Baldwin + Control + Lamarckian + baseline as horizontal line) and write to `convergence.png` in the output directory. The convergence-curve treatment matches the M4 aggregator's pattern (mean ± std band per arm).
- [ ] 5.6 Write `summary.md` with: schema-equalisation check result, per-arm gen-to-0.92 means + per-seed table, 3 gates with margins + verdict. Same format as M4's summary.md for diff-comparability.
- [ ] 5.7 Add unit tests for the aggregator's gate logic at `packages/quantum-nematode/tests/quantumnematode_tests/evolution/test_aggregate_baldwin_retry.py`: (a) schema-equalisation check fires correctly above and below the 0.05 threshold; (b) GO / PIVOT / STOP verdict logic produces the right output for each gate combination; (c) the K' filter on F1 CSV reads the right rows when multiple K' values are present. (Same convention as task 2.6 — tests live under `evolution/`.)

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
