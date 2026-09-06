# L4 panel: results as computed by the registered harness

Panel launched 2026-09-06 from commit `648854b4` (pinned state `e9380fe4`, [launch.md](launch.md));
56 runs, seven arms × seeds 1–8, 3000 episodes, 2 h 2 min on 16 workers; one registered
extension ([launch.md](launch.md) § Extensions). Analysis: `scripts/analysis/l4_panel.py` over
[\_manifest.txt](_manifest.txt) → [panel.json](panel.json), [per-seed.csv](per-seed.csv),
[curves.csv](curves.csv). The recipe, the budget and every value below the arms ran with are the
pilots' ([pilot-3-notes.md](pilot-3-notes.md), [probe-4-robustness.md](probe-4-robustness.md));
the yardstick's configuration is probes 5–6's ([probe-6-frozen-readout-yardstick.md](probe-6-frozen-readout-yardstick.md)).

## Plateau-tail full-clear success (%), seeds 1–8

| arm | mean | per seed (1…8) | converged |
|---|---|---|---|
| wt_frozen | 8.2 | 36.9, 0.8, 7.7, 14.1, 1.1, 0.4, 2.8, 2.0 | 8/8 |
| wt_hebbian | 30.2 | 78.3, 1.9, 3.3, 64.4, 0.0, 0.0, 26.5, 67.3 | 8/8 |
| wt_plastic | 17.8 | 0.0, 70.5, 28.4, 1.9, 27.5, 9.2, 0.7, 4.5 | 8/8 (seed 3 at 4500) |
| rn_frozen | 7.8 | 8.7, 4.8, 4.5, 2.9, 4.8, 15.2, 21.1, 0.4 | 8/8 |
| rn_hebbian | 13.7 | 12.1, 44.1, 0.0, 24.1, 5.1, 6.8, 12.0, 5.6 | 8/8 |
| rn_plastic | 11.2 | 13.6, 6.3, 7.1, 2.1, 2.0, 46.9, 0.3, 11.3 | 8/8 |
| mlp_plastic | 1.1 | 1.9, 0.0, 0.4, 0.0, 1.5, 0.4, 3.6, 0.7 | 8/8 |

## The registered family (paired seeds, one-sided Wilcoxon, 80% bootstrap CI, BH-FDR α = 0.05)

| test | contrast | mean Δ | 80% CI | p | q | +seeds | result |
|---|---|---|---|---|---|---|---|
| T1 | wt_plastic − rn_plastic | +6.6 | −5.5 … +20.7 | 0.371 | 0.495 | 4/8 | fail |
| T2 | wt_plastic − wt_frozen | +9.6 | −3.3 … +22.9 | 0.230 | 0.495 | 5/8 | fail |
| T3 | wt_plastic − wt_hebbian | −12.4 | −34.2 … +10.7 | 0.727 | 0.727 | 4/8 | fail |
| T4 | (wt_plastic − wt_frozen) − (rn_plastic − rn_frozen) | +6.2 | −7.0 … +21.4 | 0.371 | 0.495 | 4/8 | fail |

Band test (wt_plastic − mlp_plastic): +16.8, CI +6.9 … +28.0, PASS — by construction, since the
yardstick sits at chance (its peak tracked action densities are in the tens of thousands: a
collapsed representation, as probes 5–6 established, not a policy).

**Verdict: `sanity_floor_fail`** (T3 fails; T4 agrees with T1). Not `robustness`, so no
sensitivity pass. Ensemble invariance: T1 positive on 4 of 8 seeds, T4 on 4 of 8 — no dynamics
claim is admissible.

## Descriptive pairs of interest (uncorrected)

| pair | mean Δ | 80% CI | +seeds |
|---|---|---|---|
| wt_hebbian − rn_hebbian | +16.5 | +0.4 … +31.8 | 5/8 |
| wt_frozen − rn_frozen | +0.4 | −5.7 … +6.5 | 4/8 |
| wt_plastic − mlp_plastic | +16.8 | +6.9 … +28.0 | 6/8 |

Per-behaviour sub-metrics (means): the wild-type Hebbian floor leads on evasion (21% against
6% frozen) and foods (4.75 against 3.25); the wild-type plastic arm sits between (12%, 4.05).

## Reading (descriptive; the logbook carries the interpretation)

- **Outcomes are fixed points, seeded by the initial weights.** Every arm's plateau is reached
  within a few hundred episodes and depends more on the seed than on the rule. The wild-type
  Hebbian floor — no reward at all — reaches 78%, 64% and 67% on three seeds and 0% on two.
- **Reward modulation shifts the plastic arm's distribution upward but not reliably.** It
  produces the two best learning curves in the project (70.5% and 28.4%, genuine climbs) and it
  also takes a 37% initial policy to zero on seed 1. At n = 8 that is a null on every registered
  test.
- **The wiring signal is in the floors.** Reward-free Hebbian alignment finds better fixed
  points on the real wiring than on its degree-matched scramble (+16.5, 5/8 seeds), while the
  frozen floors are indistinguishable (+0.4). This is descriptive and uncorrected, and it is the
  thread the follow-ups pull.
- **Action densities** stay low on every connectome arm (peak ≤ 20, median 3–5): the connectome
  lands on poor fixed points on some seeds but never collapses the way the dense MLP does.
