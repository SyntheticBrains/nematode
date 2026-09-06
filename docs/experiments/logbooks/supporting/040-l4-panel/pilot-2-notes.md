# Pilot 2 (re-registered grid, centred modulator): what the rules select, and what the curves show

Run 2026-09-06 from commit `8159232c` with `scripts/campaigns/l4_panel_pilot.py`: seeds 101–102,
3000 episodes, `plasticity_rate ∈ {3e-4, 1e-3, 3e-3}`, both scaling switches on in every arm,
34 runs, 51 min on 16 workers. Summary: [pilot.json](pilot.json) (the harness's output from the
logs, verbatim, stamped `pinned: false`).

## Plateau-tail full-clear success (%), pilot seeds (101 / 102)

| rate | wt_plastic | rn_plastic | mlp_plastic | wt_hebbian | rn_hebbian |
|---|---|---|---|---|---|
| 3e-4 | 7.6 / 7.3 | 3.7 / 8.8 | 0.0 / 0.0 | 0.0 / 3.6 | 7.5 / 7.3 |
| 1e-3 | 5.7 / 10.1 | 5.7 / 16.8 | 0.0 / 0.0 | 0.0 / 3.9 | 7.5 / 7.3 |
| 3e-3 | 12.7 / 2.4 | 9.7 / 2.5 | 5.1 / 5.1 | 0.0 / 3.9 | 8.8 / 7.2 |
| frozen floors | wt 2.7 / 7.9 | rn 8.0 / 2.8 | | | |

Pooled three-factor means: 3e-4 → 4.6, 1e-3 → 6.4, 3e-3 → 6.2. **The rules select 1e-3.** Latest
plateau onset among converged runs at 1e-3: 500 (`rn_plastic` seed 102); 1.25 × 500 rounds to
2000 after the floor. **The rules give a budget of 2000.**

## What the curves and the telemetry show

Full-clear successes per 250-episode block (12 blocks):

| run | blocks 1–12 |
|---|---|
| wt_plastic 1e-3, s101 | 18 13 17 18 22 25 20 21 23 12 17 14 |
| wt_plastic 1e-3, s102 | 28 30 34 36 31 32 18 26 22 31 19 26 |
| wt_plastic 3e-3, s101 | 26 40 31 36 28 29 37 27 37 35 25 35 |
| wt_plastic 3e-3, s102 | 11 4 1 5 8 3 11 5 5 5 5 8 |
| rn_plastic 1e-3, s102 | 6 9 31 28 50 41 35 41 33 44 38 44 |
| wt_frozen s101 / s102 | 8 7 6 3 5 6 8 10 6 5 7 8 / 12 17 13 17 21 17 18 21 14 16 19 24 |
| rn_frozen s101 / s102 | 29 22 18 23 24 16 20 24 24 22 20 18 / 6 4 5 4 6 4 9 9 7 8 7 6 |
| mlp_plastic 3e-4 and 1e-3, both seeds | all zero |
| mlp_plastic 3e-3, s101 / s102 | 6 15 12 18 14 16 17 15 11 14 13 11 / 5 16 12 18 14 15 15 15 11 14 13 11 |

- **The plastic connectome arms sit near their frozen floors.** Per seed at 1e-3 the wild-type
  gains +3.0 and +2.2 points over its frozen floor; the rewired arm −2.3 and +14.0. The one clear
  learning curve is `rn_plastic` 1e-3 seed 102 (from 6 per block to the 40s, onset 500). Where the
  wild-type arm is above its floor it is above from the first block (3e-3 seed 101: 26 per block
  against 8 frozen), a fast effect that then holds flat.
- **Saturation grows through the run at every rate.** Terminal-step saturated fraction in five
  slices of the run: wt 1e-3 s102 `0.00 0.02 0.20 0.20 0.22`; wt 3e-3 s101 `0.05 0.16 0.16 0.18 0.17`; rn 1e-3 s102 `0.00 0.00 0.02 0.12 0.11`. A fifth of the synapses are on the ±3 bound by
  the plateau tail at the selected rate: the weight decay (`0.001` per unit rate) is too weak to
  hold the random walk, and the plateau is measured on a partly clamped substrate.
- **The Hebbian floors saturate to a constant policy almost immediately** (their block sequences
  are identical across rates); the wild-type Hebbian floor sits below the frozen floor.
- **The MLP yardstick either explodes or dies.** At 3e-4 and 1e-3 its trace scale climbs to
  `1e7` and it never clears the task; at 3e-3 its units die early (mean absolute change falls to
  `1e-7`) and it settles at a constant policy whose plateau is identical across seeds.
- The modulator behaves as designed: `σ ≈ 0.6–0.7`, centre `−0.03` to `−0.07`, stable through
  the run on every connectome arm.

## Pin

Pending: the recipe and the budget are the registered rules' outputs above; the pin is a dated
amendment to the panel change's design and is not made in this file.
