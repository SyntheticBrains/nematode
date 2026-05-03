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

- [x] 3.1 Created `configs/evolution/baldwin_lstmppo_klinotaxis_predator_retry_pilot.yml` with the 8-field schema per Decision 1: 4 hyperparam knobs (actor_lr, critic_lr, gamma, entropy_coef) + 2 innate-bias knobs (weight_init_scale, entropy_decay_episodes) + 2 NEW arch knobs (actor_hidden_dim ∈ [64, 256], actor_num_layers ∈ [1, 3]). Inheritance: baldwin + early_stop_on_saturation: 5. Documented the audit findings + design decisions inline.
- [x] 3.2 Created `configs/evolution/control_lstmppo_klinotaxis_predator_retry_pilot.yml`. Identical to Baldwin config except `inheritance: none`. Schema is byte-identical (same 8 fields, same bounds, same order — verified at YAML load via the schema-equalisation check below).
- [x] 3.3 Lamarckian rerun reuses the existing `configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml` (4-field schema) — no new YAML needed. Primary purpose: comparative-gate baseline at n=8; secondary: M3 reproducibility on the n=4 subset.
- [x] 3.4 Hand-tuned baseline reuses M2.11's existing artefacts (4 seeds, 42-45). Note for logbook 015 + aggregator: baseline horizontal line on the convergence plot SHALL be annotated `(n=4 seeds 42-45)` so the n-asymmetry vs the n=8 pilot arms is explicit.
- [x] 3.5 Both new YAMLs parse cleanly via `load_simulation_config`. Verified the schema-equalisation property at YAML level: both YAMLs have IDENTICAL `hyperparam_schema` blocks (8 fields, byte-identical name/type/bounds/log_scale tuples in the same order). Inheritance field is the only diff.

## 4. Campaign scripts

- [x] 4.1 Created `scripts/campaigns/phase5_baldwin_retry_baldwin.sh` — patterned after M4's Baldwin script with two diffs: 8-field config + n=8 seeds (42-49). OUTPUT_ROOT is `evolution_results/baldwin_retry_baldwin_lstmppo_klinotaxis_predator` (distinct from M4's directory).
- [x] 4.2 Created `scripts/campaigns/phase5_baldwin_retry_control.sh`. Same structure as the Baldwin script but uses the matching 8-field control config + `--inheritance none`. OUTPUT_ROOT distinct from Baldwin's.
- [x] 4.3 Created `scripts/campaigns/phase5_baldwin_retry_lamarckian_rerun.sh`. Reuses M3's existing lamarckian config (4-field) under the M4.5 code revision; n=8 seeds. Documented the schema-asymmetry rationale inline (Lamarckian doesn't participate in audit-A1 schema-equalisation; sole role is comparative-gate baseline at n=8). Adds `--early-stop-on-saturation 5` at runtime since the M3 config doesn't set it. OUTPUT_ROOT distinct from M4 Lamarckian rerun's.
- [x] 4.4 Each script's header documents expected wall-time per the design's compute estimate (~2.5-3 hours per arm at parallel=4 with early-stop enabled; total pilot wall ~3-4 hours with all three arms in parallel + F1 post-pilot).
- [x] 4.5 Pre-commit clean on all three scripts (mdformat n/a; bash scripts pass the file-end + large-files + tests hooks). Scripts marked executable via `chmod +x`.

## 5. Aggregator updates

- [x] 5.1 Created new `scripts/campaigns/aggregate_baldwin_retry_pilot.py` (chose new-file over extending M4's aggregator — logbook 014 references `aggregate_m4_pilot.py` by name; modifying it would invalidate that audit trail). Decision documented in the script's module docstring.
- [x] 5.2 Implemented `_check_schema_equalisation` — reads `history.csv` first data row per seed (the framework writes 1-indexed history.csv where the first row labels `generation = 1`; equivalent to `lineage.csv` `generation == 0`), averages across seeds, and forces verdict to INCONCLUSIVE if `|Δ| > 0.05` between Baldwin and Control means. Tests verify pass at `|Δ| = 0.02` and fail at `|Δ| = 0.15`.
- [x] 5.3 Implemented `_compute_verdict` with the recalibrated 3 gates: speed `+2`, F1 `+0.05` (was `+0.10` in M4), comparative `+4`. Reference values are the M4.5 pilot's own arm means (not M4's published). 4-way verdict: INCONCLUSIVE if schema-equalisation fails; GO if all three gates pass; PIVOT if speed only; STOP otherwise. Tests cover all 16 gate combinations.
- [x] 5.4 Implemented `_read_f1_csv(f1_csv, k_prime=...)` — reads `f1_learning_acceleration.csv` with new columns (`seed, k_prime, episodes, elite_success_rate, baseline_success_rate, signal_delta`), filters by `--k-prime` (default 10). Tests verify K' = 10 and K' = 25 rows can coexist + filter selects the requested rows.
- [x] 5.5 Implemented `_plot_convergence` — 4-curve plot (Baldwin + Lamarckian + Control mean ± std bands; baseline horizontal line annotated `(n=4 seeds 42-45)` per round-2 review M5; F1 elite + baseline scatter markers at the right margin distinguishing elite from baseline). Writes `convergence.png` to the output directory.
- [x] 5.6 Implemented `_format_summary` writing `summary.md` with the schema-equalisation header (table + status), 3 gates with margins, verdict + verdict-text, per-seed table including F1 elite + baseline + signal columns. INCONCLUSIVE path skips the gates section + emits the audit-A1-not-closed text directly.
- [x] 5.7 Unit tests at `packages/quantum-nematode/tests/quantumnematode_tests/evolution/test_aggregate_baldwin_retry.py` (11 tests, all passing in 0.26s): schema-equalisation check (3 tests covering pass/fail/at-threshold with fp arithmetic note); verdict logic (4 tests covering INCONCLUSIVE / GO / PIVOT / STOP across gate combinations); F1 CSV K' filter (3 tests + missing-file error path); helpers exposure.

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
