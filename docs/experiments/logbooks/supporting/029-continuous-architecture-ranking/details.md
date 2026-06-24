# 029 — Continuous-Substrate Architecture Ranking: Supporting Detail

Appendix to [029-continuous-architecture-ranking.md](../../029-continuous-architecture-ranking.md).
Raw analysis outputs in this folder: `ranking-per-seed.csv`, `per-behaviour.csv`,
`cross_arch_pairwise.json`, `ga_c3_results.json`.

## Per-seed plateau-tail success (n=8; final ranking)

| arch | mean ± sd | converged | per-seed (success %) |
|---|---|---|---|
| mlpppo | 89.0 ± 7.3 | 7/8 | 95.5 93.3 91.6 92.4 93.1 92.6 79.9 73.7 |
| cfcppo | 75.8 ± 9.7 | 7/8 | 55.3 72.9 84.5 83.1 78.9 81.7 66.2 83.8 |
| transformerppo | 74.0 ± 7.7 | 6/8 | 77.2 74.5 70.1 86.1 78.9 75.2 57.3 73.1 |
| lstmppo | 60.1 ± 15.8 | 5/8 | 24.7 67.1 73.4 66.9 57.1 50.8 61.1 79.6 |
| connectomeppo | 52.2 ± 9.0 | 8/8 | 53.7 45.1 51.6 51.6 68.3 35.3 59.4 52.6 |
| feedforwardga | 15.0 ± 16.4 | n/a | 50.0 30.0 17.5 2.5 0.0 5.0 0.0 15.0 |

Budgets: all seeds at 6000ep except the 5 that were still climbing at 6000ep
(`cfcppo` s2/s7, `lstmppo` s1/s5, `connectomeppo` s1), re-run at 8000ep. `converged` = the level-agnostic
`detect_convergence` returned a non-null onset (`feedforwardga` has no training-plateau notion → n/a).

## Pairwise statistics (BH-FDR, one-sided Wilcoxon a>b, 80% bootstrap CI)

Full table in `cross_arch_pairwise.json`. Significant (q < 0.05) except: **cfcppo vs transformerppo**
(Δ+1.8, q=0.527 — the #2 tie) and **lstmppo vs connectomeppo** (Δ+7.9, q=0.205 — LSTM's high variance).
All five MLP comparisons and all five GA comparisons are q ≤ 0.024 \*\*\*.

## Budget-iteration story (why uniform 6000ep + an 8000ep top-up)

The arms learn at very different *rates* on this hard cell, so the budget was raised once the convergence
audit showed slow-climbers. Each step re-used the same locked configs + metric; only `--runs` changed.

| budget | what was run | finding |
|---|---|---|
| 1200ep | initial 5-arm n=4 + canaries | discrimination confirmed; difficulty locked (count2). MLP-leads. |
| 2400ep | re-measure n=4 (post #250 metric) | **two 1200ep artifacts overturned**: connectome is mid-pack (not a "35 structural ceiling"), Transformer is the robust #2 — both were *under-trained*, not weak. |
| 3000ep | per-arch tuning + n=8 ranking | tuning lifted LSTM/CfC modestly; n=8 ranking computed. **Dig-in: fallback contamination** (sub-50% plateaus → null → noisy last-10 mean) — fixed by ranking on the plateau tail + the real `converged` flag. |
| 4000ep | dial-able arms (MLP/Tfmr/conn) | MLP → 8/8 converged; **Transformer rose 67 → 75** (it was still climbing at 3000ep). |
| **6000ep** | **uniform re-run, all 5 PPO arms** | **CfC rose 65 → 76** (all its non-converged seeds had been climbing) — moves CfC from #3 to the #2 tie. The fairness fix that mattered. |
| 8000ep | 5 still-climbing seeds (B1) | all 5 converged; most *settled slightly lower* than their 6k tail (the 6k "climbing" tails caught upward fluctuations) → confirms the plateau-tail metric was already sound. Order unchanged; CfC 76.3 → 75.8. |

Net: the order **MLP ≫ {CfC ≈ Transformer} > LSTM > connectome ≫ GA** is robust across all budget levels;
the budget work mattered for CfC's #2 placement and for retiring the slow-climber lower-bound caveat.

## Per-arch recipes (tuned on the integrated cell — recipes do not port between cells)

| arch | key recipe |
|---|---|
| mlpppo | flat `entropy 0.08`, lr 3e-4 (the documented 0.08→0.02/800ep schedule was a **no-op** — `mlpppo` never implemented it; see below) |
| cfcppo | `entropy 0.01`, lr 3e-4 (more entropy made it *worse* — lost the high basin) |
| transformerppo | `entropy 0.005`, lr **1e-4** (lr 3e-4 collapses — learn-then-collapse) |
| lstmppo | flat `entropy 0.01`, lr **2e-4** (the lr+entropy fix lifted the floor; residual wobble is intrinsic) |
| connectomeppo | `entropy 0.005` (drift-lock) + predator + thermo projections |

## Methodology findings (the dig-ins)

1. **Fallback contamination (fixed via #250 + the harness).** The pre-fix metric returned null on
   sub-50% plateaus and fell back to a last-10-run mean — e.g. a Transformer seed read 100% (a lucky
   final streak) vs its 72% plateau; connectome read 20% vs ~48%. The ranking now uses the plateau-tail
   mean (equals `post_convergence_success_rate` within ~3–5pp for converged seeds — the cross-check) and
   reads the *real* `convergence_run` flag.
2. **Slow-climber under-ranking (fixed by uniform budget).** At 3000–4000ep the slow learners were still
   climbing; ranking them against a plateaued MLP under-ranked them. Uniform 6000ep fixed it (CfC +11pp).
3. **Phantom MLP schedule (→ #253/#254).** The MLP config set `entropy_coef_end`/`entropy_decay_episodes`,
   but `mlpppo` has no entropy schedule, so the keys were silently dropped and MLP ran flat `entropy 0.08`
   the whole time (it is still the leader). A load-time warning for dropped brain-config keys was added
   (#253); the wider dead-key debt (`normalize_advantages` on 60+ mlpppo configs, never implemented) is
   tracked in #254.

## GA champion eval

The GA (FeedforwardGA) trains via `run_evolution.py` (population weight-search, no per-run
`--track-experiment` JSON), so it is scored by its evolved champion's full-clear rate over a 40-episode
frozen eval (`t7_ga_champion_eval.py`), the analogue of the PPO post-convergence success. Per-seed
0–50% (mean 15.0, sd 16.4) — gradient-free search does not solve the integrated lethal cell, reproducing
the T4 ~0–collapse (optimiser-fundamental, action-space-agnostic). Runs discrete (4-way) via the env's
discrete-action fallback (no continuous head; no code change).
