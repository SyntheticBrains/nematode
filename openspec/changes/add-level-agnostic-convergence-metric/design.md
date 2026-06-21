## Context

`detect_convergence` ([`benchmark/convergence.py`](../../../packages/quantum-nematode/quantumnematode/benchmark/convergence.py)) declares convergence at the earliest 10-run window whose **binary-success variance < 0.05** and **mean ≥ 0.5**. On 0/1 data those two conditions co-occur only in a ≈10/10 window, so the detector implicitly measures *"reaches near-100% streaks."* The ranked metric `post_convergence_success_rate` (the `architecture-comparison-protocol` capability) averages full-clear success from that onset to the run end.

At the **T4 grid** band (top arms 73–84%) this fired reliably. At the **T7 continuous** integrated-C3 band — deliberately locked to a *learnable, sub-saturation* range (≈35–80%) so architectures spread — it does not: empirically, on the locked count2 cell (per-seed, 1200 ep) `detect_convergence` returns `None` for the sub-50% arms. The metric does **not** become null — `calculate_post_convergence_metrics` falls back to the **last-10-run mean** (`fallback_window=10`), so the realised ranked values are a high-variance n=10 estimate: connectome reads **20%** (true plateau ~48%), Transformer **60%** (true ~42%), CfC 50% (true ~49%, ok by luck), while MLP (82%, conv_run 471) and LSTM (52%, conv_run 627) fire. This (a) corrupts the ranking *order* via ±20–30pp last-10 noise, (b) silently uses the **fixed last-N window the spec explicitly forbids**, and (c) mislabels 3 of 5 arms `converged=False` despite stable plateaus. The detector conflates **converged** (the policy stopped improving) with **near-100%** (the policy is excellent); they must be decoupled.

A second observation from the prototype: at 1200 ep MLP and Transformer are *still mildly improving* (MLP's entropy schedule anneals through ep 800), which a correct detector should flag — informing the n≥8 run length.

## Goals / Non-Goals

**Goals:**

- A **level-agnostic** plateau detector: detect a converged plateau at *any* absolute success level (40% or 95%), decoupling convergence from level.
- Preserve T4/Phase-5 numbers: a high-band arm converges at the same kind of onset and yields the same `post_convergence_success_rate` (regression-guarded).
- Distinguish a *stable intermediate plateau* (report it) from a *still-trending* run (flag it), robustly against the per-episode binary-sampling noise that is irreducible at intermediate success probabilities.
- Keep the statistics layer (paired-seed Wilcoxon + 80% bootstrap + BH-FDR) and the sub-metric set unchanged.

**Non-Goals:**

- No change to reward/env/brains. No change to the ranked-metric *name* or the BH-FDR family.
- Not a sample-efficiency metric (warm-up length stays descriptive, per the existing spec).
- Not a re-tune of the difficulty band — the sub-saturation band is intended.

## Decisions

### D1 — Detect *trend*, not *variance*, using coarse blocks

The naive flatness test (small variance of the rolling mean) fails: a rolling mean of a stable p≈0.5 policy has irreducible std ≈ √(p(1−p)/W) ≈ 0.07 at W=50, which exceeds any flatness tolerance tight enough to exclude a real trend. So a stable intermediate plateau looks "non-flat."

**Decision:** establish convergence by comparing two large tail blocks — the final `tail_frac` of runs (level `L`) against the immediately-preceding block of the same size (`prev`) — and declaring converged when `|L − prev| ≤ band`. Block means over `t = tail_frac·N` runs have sampling std ≈ √(p(1−p)/t); at N=1200, t=300 that is ≈0.029, so two blocks differ by noise ≈0.04 (1σ). `band = 0.10` is ≈2.5σ: a truly flat plateau passes ~99% of the time, while a residual trend > 0.10/quartile is correctly flagged. *Alternative considered:* linear-slope fit on the rolling mean — equivalent in spirit but with a less interpretable threshold; the two-block test is simpler and its threshold is calibrated directly against block sampling noise.

### D2 — Onset anchored on the converged level (level-agnostic)

**Decision:** once converged, the plateau onset is the first run whose **rolling-mean success** (window `W=50`) reaches within `band` of the converged level `L` (`rm ≥ L − band`). This adapts to each arm's warm-up length (slow igniters get a later onset) and is purely relative to `L`, so it works at any band. The metric is `mean(raw_success[onset:])`. Validated onsets on the real data are sensible (e.g. CfC ~100–180, connectome ~370–630, MLP ~360–720). *Alternative considered:* a fixed post-warmup window (e.g. last 50%) — simpler but mis-measures heterogeneous warm-ups (the exact reason the spec rejected fixed-window for T4's long-warm-up arms); kept only as the cross-check (D4).

### D3 — Still-trending runs return `None` (flagged, not mis-scored)

**Decision:** when `|L − prev| > band` (the run is still climbing/oscillating at the end), return `None` — the existing "ran but never converged → flag, don't mis-compare" semantics. The harness already surfaces a shrunken `n`. **Operationally**, the n≥8 cells run enough episodes that all arms plateau (the prototype shows 1200 ep leaves MLP/Transformer still creeping); any run still flagged at the chosen length is reported, not silently dropped. *Note:* this is strictly better than the status quo, where `None` triggered the **last-10-run fallback** in `calculate_post_convergence_metrics` — a high-variance n=10 estimate that silently replaced the plateau with a fixed-last-N window. With the level-agnostic path a converged plateau is detected directly, so the fallback only fires for genuinely-non-converged runs (which are then flagged for extend-and-rerun, not ranked on a noisy 10-sample). The tiny `fallback_window=10` is left as-is (out of scope) but is now rarely on the ranked path.

### D4 — Final-window-mean agreement cross-check

**Decision:** record the final-window mean (plateau mean over a post-warmup tail) alongside the detected `post_convergence_success_rate`; for a homogeneous-warm-up arm set (the T7 six) they must agree within noise. This is the auditable guarantee that the detector is not introducing bias, and it is the fallback metric if a detector regression is ever suspected.

### D5 — Backward compatibility via defaults

**Decision:** the new path is the default behaviour of `detect_convergence`; `min_success_rate` is removed from the convergence gate (it was the level-coupling), but the function keeps its signature shape and adds the new tunables (`band`, `tail_frac`, smoothing `W`) with defaults that reproduce sensible T4-band behaviour. A regression test asserts high-band (≥0.8) sequences converge at an equivalent onset and yield an equal-within-ε `post_convergence_success_rate` vs the legacy logic.

## Risks / Trade-offs

- **[Detector flags many seeds at too-short a run length]** → the n≥8 cells use enough episodes to plateau (validated by a longer-run canary: confirm MLP/Transformer converge + detect cleanly at the chosen length); the final-window-mean cross-check (D4) catches any residual disagreement.
- **\[`band`/`W`/`tail_frac` are tuned to the T7 band\]** → defaults are derived from sampling-noise math (D1), not fit to T7; the regression test (D5) and the cross-check (D4) bound the risk; constants live in one place and are documented.
- **[Changing a shared detector perturbs other consumers]** → `detect_convergence` is used by the benchmark/plateau path; the regression test pins high-band behaviour, and the change is additive (new path, same outputs on high-band inputs). Grep-audit all callers in tasks.
- **[Spec edit to a frozen methodology capability]** → the metric *name* and statistical family are unchanged; only the plateau-detection definition is generalised, recorded as a `MODIFIED` delta with a scenario, and the legacy behaviour is a documented special case.

## Migration Plan

1. Implement the level-agnostic path in `detect_convergence` + a regression test pinning high-band behaviour.
2. Re-derive any cached Phase-5 ranked numbers if needed (expected: unchanged).
3. Land the spec delta. The T7 n≥8 ranking (separate change) consumes the corrected metric.

## Open Questions

- Exact n≥8 run length: set from the longer-run canary so every arm plateaus (resolved during implementation, recorded in tasks).
