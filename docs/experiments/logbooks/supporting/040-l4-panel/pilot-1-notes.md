# Pilot 1 (registered grid): outcome and why it did not pin a recipe

Run 2026-09-05/06 from commit `3494428a` with `scripts/campaigns/l4_panel_pilot.py`
(seeds 101–102, 3000 episodes, `plasticity_rate ∈ {0.003, 0.01, 0.03}`, 34 runs, 40 min on
16 workers). Summary: [pilot-1-registered-grid.json](pilot-1-registered-grid.json).

## Plateau-tail full-clear success (%), pilot seeds

| rate | wt_plastic | rn_plastic | mlp_plastic | wt_hebbian | rn_hebbian |
|---|---|---|---|---|---|
| 0.003 | 7.1 | 10.6 | 5.1 | 3.7 | 8.3 |
| 0.01 | 5.3 | 5.8 | 5.1 | 6.9 | 10.1 |
| 0.03 | 7.6 | 5.2 | 5.1 | 1.9 | 9.8 |
| frozen floors | wt 5.3 | rn 5.4 | | | |

Every arm sits at the frozen floors' level. Every run "converged" at episode 1: flat from the
start. The pooled rule would have selected 0.003 on the strength of one rewired seed at 21%.

## What the per-update telemetry showed (terminal step of each episode)

| run (rate 0.01, seed 101 unless noted) | saturated fraction, mean / max | mean abs weight change |
|---|---|---|
| wt_plastic | 0.12–0.14 / 0.91–0.95 | 2.7e-2 |
| rn_plastic (0.003) | 0.24–0.27 / 0.91–0.93 | 7.0e-3 |
| wt_hebbian (seed 102) | 0.95–0.96 / 0.98 | 2.5e-5 (nothing left to move) |
| mlp_plastic | 0.25 in episode 1, then 0.00 | 3.7e-2 in episode 1, then 1e-5 falling to 7e-8 |

The prediction error at the terminal step is about −10 (the death penalty against a baseline
near −0.3), so a single update at rate 0.01 moves weights by order 1 against an initialisation
scale of about 0.3. The connectome arms random-walk into the ±3 bound; the Hebbian floor, whose
update never changes sign for a co-active pair, ends with 96% of synapses clamped. The MLP takes
one −10 kick through its ReLU layers in episode 1, after which its units are dead and its updates
vanish — its behaviour is then identical across rates (38 of 3000 episode lines differ between
0.003 and 0.01 at one seed) and its plateau-tail value identical across seeds.

## Diagnostic probes (not part of the registration; seed 101)

60 episodes, terminal-step means over the last 10:

| arm | rate | saturated | mean abs change |
|---|---|---|---|
| wt_plastic | 1e-3 | 0.000 | 1.9e-3 |
| wt_plastic | 1e-4 | 0.000 | 1.5e-4 |
| wt_plastic | 1e-5 | 0.000 | 9.2e-6 |
| mlp_plastic | 1e-3 | 0.000 | 2.6e-6 |
| mlp_plastic | 1e-4 | 0.000 | 3.6e-6 |
| mlp_plastic | 1e-5 | 0.000 | 1.8e-6 |

600 episodes, full-clear successes per 100-episode block:

| arm | rate | blocks 1–6 | saturated (last 20) |
|---|---|---|---|
| wt_plastic | 1e-4 | 3 6 8 22 12 15 | 0.002 |
| wt_plastic | 1e-3 | 6 11 6 9 7 7 | 0.066 |
| mlp_plastic | 3e-3 | 1 6 3 5 8 3 | 0.000 |
| mlp_plastic | 1e-3 | 1 6 3 6 8 3 | 0.000 |
| wt_frozen (pilot log, first 600) | — | 4 3 4 3 1 2 | — |

The connectome learns at 1e-4 (three to five times the frozen rate by episode 400–600, negligible
saturation) and is already drifting toward the bound at 1e-3. The MLP has no workable shared rate:
lethal at 0.01, negligible at 3e-3 and below (its two probe trajectories are near-identical), with
a per-weight trace magnitude roughly a thousand times smaller than the connectome's. A matched
*rate* is therefore not a matched *rule* across substrates.

## Consequence

The registered grid is two orders of magnitude too hot, and no single rate serves both
substrates. Ratified with Chris (2026-09-06): the rule's scaling is made substrate-invariant in
its own change before the panel; the grid is then re-registered by dated amendment and the pilot
re-run. Recipe and budget stay unpinned until then. This file and the summary JSON are kept so the
registered pilot's outcome is on the record.
