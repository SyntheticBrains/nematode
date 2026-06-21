## Why

The cross-architecture ranked metric `post_convergence_success_rate` is computed by `detect_convergence`, which only declares convergence when a sliding window of binary full-clear outcomes has variance `< 0.05` **and** mean `>= 0.5`. On binary data those two conditions can be met together only by a near-100% (≈10/10) window, so the detector effectively measures *"does the arm reach near-perfect success streaks?"* That held at the T4 grid band (top arms 73–84%, where 10/10 streaks occur), but it **breaks at the T7 continuous band**, whose integrated-C3 cell is deliberately locked to a *learnable, sub-saturation* range (≈35–80% full-clear) so the architectures actually spread. There, mid- and low-band arms never produce a low-variance high-mean window, so `detect_convergence` returns `None`. The metric does **not** then become null — `calculate_post_convergence_metrics` silently **falls back to the mean of the last 10 runs** (`fallback_window=10`) — which is three failures at once: (1) a tiny **n=10** estimate of a ~50% plateau has ≈16pp sampling std, so the ranked value swings ±20–30pp purely by which 10 episodes ended the run; (2) it is exactly the **fixed last-N window the spec forbids** (the spec rejects fixed-last-N because it mis-measures heterogeneous warm-ups); (3) the arm is mislabeled **`converged=False`** despite sitting on a stable plateau. Empirically on the locked count2 cell (per-seed, 1200 ep): connectome's ranked value reads **20%** against a true plateau of ~48% (a bad last-10 streak), Transformer reads **60%** against ~42% — corrupting the ranking *order*, not just the values, and flagging 3 of 5 arms non-converged. The detector conflates *converged* (the policy stopped improving) with *near-100%* (the policy is excellent); these must be decoupled before the T7 n≥8 ranking runs.

## What Changes

- Add a **level-agnostic plateau-convergence detector** to `benchmark/convergence.py`: detect the plateau onset from where a **smoothed (rolling-mean) success rate flattens** — stable in slope/variance over a sustained window — *regardless of the plateau's absolute level*, then average raw full-clear success from that onset. A converged 40% arm and a converged 95% arm are both detected; only genuinely non-stationary (still-climbing / oscillating) runs return `None`.
- `post_convergence_success_rate` is now well-defined at any plateau level. The legacy near-100%-streak behaviour is preserved as a special case (a 95% arm still converges at the same kind of onset), so T4-era numbers are unaffected.
- Amend the `architecture-comparison-protocol` capability: the ranked metric's plateau-detection is specified as **level-agnostic** (decoupled from absolute success), and a **final-window mean cross-check** (the plateau mean over a post-warmup tail window) is recorded as the required agreement test for homogeneous-warm-up arm sets such as the T7 six.
- No change to the statistics layer (paired-seed Wilcoxon + 80% bootstrap + BH-FDR) or to the sub-metric set — only how the per-seed plateau scalar is derived.

## Capabilities

### New Capabilities

<!-- none — this refines an existing methodology capability + its detector -->

### Modified Capabilities

- `architecture-comparison-protocol`: the ranked-metric plateau detection (`post_convergence_success_rate`) is respecified to be **level-agnostic** — convergence (stationarity of the smoothed success rate) is decoupled from absolute success level, so the metric is valid across any learnable band (e.g. the T7 continuous sub-saturation band), with a final-window-mean agreement cross-check.

## Impact

- **Code:** `packages/quantum-nematode/quantumnematode/benchmark/convergence.py` (`detect_convergence` — new level-agnostic path; the strict-window behaviour kept as the high-band special case), and its callers that populate `post_convergence_success_rate` in the experiment-tracking JSON.
- **Spec:** `openspec/specs/architecture-comparison-protocol/spec.md` (ranked-metric plateau-detection scenario).
- **Consumers:** `scripts/analysis/weight_search_architecture_ranking.py` reads the resulting per-seed scalar unchanged; the T7 n≥8 ranking depends on this fix landing first.
- **No** change to reward, env, brains, or the statistical-test layer; existing T4/Phase-5 rankings recompute to the same numbers (regression-guarded).
