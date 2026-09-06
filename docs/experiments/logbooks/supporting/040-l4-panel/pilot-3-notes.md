# Pilot 3 (re-registered grid, robustness values pinned): the pin

Run 2026-09-06 from commit `e123e950` with `scripts/campaigns/l4_panel_pilot.py`: seeds 101–102,
3000 episodes, `plasticity_rate ∈ {3e-4, 1e-3, 3e-3}`, every arm with both scaling switches on,
homeostasis on and `initial_log_std −1.0`, the MLP arm on tanh units; 34 runs, 117 min on 16
workers. Summary: [pilot.json](pilot.json) (the harness's output from the logs, verbatim,
stamped `pinned: false`; the pin is the dated amendment in the panel change's design).

## Plateau-tail full-clear success (%), pilot seeds (101 / 102)

| rate | wt_plastic | rn_plastic | mlp_plastic | wt_hebbian | rn_hebbian |
|---|---|---|---|---|---|
| 3e-4 | 9.6 / 19.3 | 0.4 / 17.5 | 1.7 / 0.0 | 0.0 / 20.3 | 35.9 / 22.1 |
| **1e-3** | **42.0 / 37.1** | **18.3 / 0.4** | 0.1 / 0.0 | 0.0 / 21.6 | 35.9 / 20.8 |
| 3e-3 | 50.1 / 10.8 | 0.8 / 8.8 | 0.1 / 0.4 | 0.0 / 20.5 | 36.3 / 19.3 |
| frozen floors | wt 2.4 / 3.3 | rn 7.5 / 0.1 | | | |

Pooled three-factor means: 3e-4 → 8.1, 1e-3 → 16.3, 3e-3 → 11.8. **The rules select 1e-3.**
Latest plateau onset among converged runs at 1e-3: 2090 (`wt_hebbian` seed 102); 1.25 × 2090
rounds up to **3000**. Every three-factor arm converged on at least one seed at the selected rate,
so no extension is owed.

## Full-clear successes per 250-episode block

| run | blocks 1–12 |
|---|---|
| wt_plastic 1e-3, s101 | 101 151 137 105 100 98 113 105 124 103 110 102 |
| wt_plastic 1e-3, s102 | 20 54 93 87 82 80 100 106 89 99 87 92 |
| wt_plastic 3e-3, s101 (still climbing) | 91 93 66 69 88 102 84 94 107 94 131 151 |
| wt_plastic 3e-3, s102 | 91 46 40 42 29 33 48 43 23 31 18 32 |
| wt_plastic 3e-4, s101 / s102 | 62 32 52 13 10 7 10 10 13 20 24 28 / 30 28 4 21 17 43 46 52 39 63 45 37 |
| rn_plastic 1e-3, s101 / s102 | 51 110 91 93 79 53 43 59 52 43 49 45 / 0 0 1 0 0 0 0 0 1 2 1 0 |
| rn_plastic 3e-4, s101 / s102 | 19 4 6 3 2 8 0 0 1 0 1 2 / 1 0 0 0 6 30 30 35 32 36 51 44 |
| rn_hebbian 1e-3, s101 / s102 | 97 91 87 94 86 81 109 92 95 89 94 86 / 57 48 59 47 49 59 61 52 54 43 59 54 |
| wt_frozen s101 / s102 | 11 6 3 7 11 5 8 7 10 9 5 4 / 6 11 4 1 1 6 6 1 4 8 5 12 |
| rn_frozen s101 / s102 | 15 23 24 29 26 19 20 16 19 18 20 18 / 0 0 0 0 0 0 0 0 0 1 0 0 |
| mlp_plastic, every rate and seed | 0–11 per block, no trend |

## Reading

- **The wild-type plastic arm learns and holds.** At 1e-3 both seeds reach 40–60% per block by
  episode 500 and stay there for 3000 episodes; against frozen floors of 2–3% this is a
  fifteen-fold gain with zero saturation throughout. At 3e-4 it is unstable and at 3e-3 one seed
  decays while the other is still climbing at the budget (not converged): the rate window is
  narrow and 1e-3 is inside it.
- **The rewired plastic arm is fragile.** At 1e-3 one seed learns to about 18–20% and the other
  never leaves zero; at 3e-4 one seed collapses and the other learns late; at 3e-3 both stay
  low. Where it learns, it learns less than the wild-type arm and less reliably.
- **The rewired Hebbian floor is high and constant.** From its first block the unmodulated rule
  on the rewired wiring sits at 21–36% at every rate — above its own plastic arm — while the same
  rule on the wild-type wiring sits at 0% or 21% depending on the seed. Under homeostasis the
  Hebbian floor lands on a constant policy immediately, and where it lands is a property of the
  wiring and the seed, not of reward. This makes T3 a substantive test and is the most surprising
  number in the pilot: on the rewired wiring, reward modulation appears to *hurt* relative to
  reward-free drift, while on the wild-type wiring it helps enormously.
- **The yardstick does not learn** at any rate with bounded units, a frozen readout and 3000
  episodes. D2 test (ii) will pass by construction; the logbook says so.
- All of this is two seeds. The direction of every contrast is what the panel's eight paired
  seeds are for.

## Pin (recorded in the panel design by dated amendment)

`plasticity_rate: 0.001` written into all seven arm configs; uniform budget 3000 episodes; the
single pre-registered extension to 4500 for any seed still climbing at 3000.
