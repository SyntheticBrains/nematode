# Tasks

## 1. Implement the level-agnostic detector

- [ ] 1.1 In [`benchmark/convergence.py`](../../../packages/quantum-nematode/quantumnematode/benchmark/convergence.py), add the level-agnostic plateau path to `detect_convergence`: convergence via a two-block no-trend test (final trailing block vs the equal-size preceding block agree within `band`); onset via the earliest run whose rolling-mean (window `W`) success reaches within `band` of the converged level `L`; return `None` when still trending. Remove the `min_success_rate` level-coupling from the gate.
- [ ] 1.2 Expose the tunables (`band`, `tail_frac`, smoothing `W`) as keyword args with defaults derived from block sampling-noise (`band` ≈ 2.5× the two-block 1σ at the expected n; documented inline), in ONE place.
- [ ] 1.3 Keep the function signature shape backward-compatible; update the docstring to describe the level-agnostic semantics + the band derivation.
- [ ] 1.4 Audit every consumer of `detect_convergence` / `post_convergence_success_rate` (grep `packages/` + `scripts/`); confirm the new path returns equivalent results on high-band inputs and update any caller assumptions. **Confirmed in review:** `detect_convergence` has exactly one production caller (`analyze_convergence` at convergence.py:590, no kwargs) → `calculate_post_convergence_metrics` → `ConvergenceMetrics` → tracker/CSV/experiment-JSON; the `min_success_rate` kwarg is passed by NO caller (the other `min_success_rate` hits are the unrelated `BenchmarkValidationRules`), so removing/repurposing it from the gate has zero external blast radius. The `fallback_window=10` last-N fallback in `calculate_post_convergence_metrics` is left unchanged (out of scope) — the level-agnostic path keeps it off the ranked path for converged arms.

## 2. Tests

- [ ] 2.1 **Regression (high band):** a ≥0.85 plateau sequence (with a warm-up ramp) converges at an equivalent onset and yields `post_convergence_success_rate` equal-within-ε to the legacy logic — pins T4/Phase-5 behaviour.
- [ ] 2.2 **Sub-saturation plateau:** a stable ~0.45 Bernoulli plateau (after a warm-up) is detected as converged and reports ≈0.45 (the legacy detector returns `None` here — assert the regression is fixed).
- [ ] 2.3 **Still-trending:** a monotonically-climbing sequence that has not flattened by the end returns `None` (flagged, not mis-scored).
- [ ] 2.4 **Flat-at-zero warm-up is not a plateau:** a long all-fail prefix followed by ignition does NOT report the zero region as the plateau (onset is after ignition; converged level reflects the final block).
- [ ] 2.5 **Cross-check:** detected `post_convergence_success_rate` agrees within sampling noise with the full-window mean on a homogeneous-warm-up synthetic set.
- [ ] 2.6 **Migrate the two existing old-semantics tests** (they encode level-coupling and WILL break — by design): `test_detect_convergence_never_converges` (alternating 0/1) and `test_analyze_convergence_no_convergence` (random ~50%) currently assert a stationary ~50% sequence is `None` / `converged=False`. Under level-agnostic semantics a stable 50% **success-rate** IS a converged plateau, so re-conceive "never converges" as a genuinely **still-trending** sequence (monotonic climb that has not flattened by the end) and assert `None` there; add/repoint a positive assertion that the stationary-50% case now converges and reports ≈50%. Keep `test_post_convergence_fallback_to_last_n` (the fallback still exists for genuinely-non-converged runs).

## 3. Validation against real data + run-length

- [ ] 3.1 Re-run the prototype over the existing T7 C3 per-seed run data; confirm all converged arms detect (no spurious nulls for steady plateaus) and the per-seed plateau rates match the final-window means.
- [ ] 3.2 Confirm the longer-run canary (MLP + Transformer @ extended budget) plateaus and detects cleanly — i.e. the residual late-trend nulls at 1200ep resolve at the budget the n≥8 cells will use. **Validated:** at **2400 episodes** MLP (quartiles 40/86/87/91) and Transformer (26/45/63/63) both plateau (Q3≈Q4) and detect cleanly (onset ~722/727, post_conv 89%/58%); the steady arms (CfC/LSTM/connectome) already plateau by ~ep 600 at 1200ep. **n≥8 PPO budget = 2400 episodes.** (Confirms the 1200ep final-quarter numbers underestimated the still-improving arms: MLP ~89–93% and Transformer ~58–62% at full convergence vs 80/52 — ranking order unchanged.)

## 4. Spec + gates

- [ ] 4.1 Land the `architecture-comparison-protocol` MODIFIED delta (level-agnostic plateau detection + the cross-check scenario).
- [ ] 4.2 `openspec validate add-level-agnostic-convergence-metric --strict` passes.
- [ ] 4.3 Targeted pre-commit (ruff / pyright / markdownlint) on changed files; full `uv run pytest -m "not nightly"` for the benchmark package.
