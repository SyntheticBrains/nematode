# Launch record: F.0a, NPE recovers C.1's mechanistic baseline

**Registered 2026-10-25**, change `add-npe-recovery`. This record is committed before any training simulation runs. A gate registered here is never re-read under a different rule. Anything else is reported beside it as description.

## The model

`quantumnematode.evaluation.controllers.BatchedMechanistic` has 20 parameters, in `MECHANISTIC_PARAMETERS` order:

- **switching:** dwelling → roaming at exp(a0 + a_intake·I + a_satiety·S) per second, and roaming → dwelling at exp(b0 + b_intake·I + b_satiety·S);
- **runs:** speed v_s, and curvature κ_w·tanh(lateral) + σ_s·z;
- **reversals:** r_s · exp(−β_r·tanh(dC/dt)) · exp(edge·loss) · search;
- **omega turns:** o_s · exp(−β_o·tanh(dC/dt)) · search;
- **search off food:** 1 + A·exp(−t_off/τ).

The delta hazard is the primitives' minimum.

## Priors (none from search data)

NPE works on z ∈ [0, 1]²⁰, mapped through each prior (`npe_recovery.PRIOR`):

| Parameters | Prior |
|---|---|
| a0, b0 | uniform in [ln 10⁻⁴, ln 10⁻¹] (rates per second) |
| a_intake, a_satiety, b_intake, b_satiety | uniform [−5, 5] |
| v_dwell, v_roam | uniform [0.005, 0.20] mm/s |
| sigma_dwell, sigma_roam | uniform [5, 60] °/s |
| kappa_weathervane | uniform [−50, 50] °/s |
| r_dwell, r_roam, o_dwell, o_roam | log-uniform [10⁻³, 0.5] /s |
| beta_reversal, beta_omega, edge | uniform [0, 5] |
| search_amplitude | uniform [0, 20] |
| search_tau_s | log-uniform [10, 3,000] s |

## The simulator and summary

- **The simulator:** 200 animals on `scheer_lawn`, `uniform_lawn_scheer` and `food_removal_lopezcruz` (the MLP arm's assay configs), through the batched evaluator and the grader.
- **The summary:** the 11 wild-type statistics and local search's presence bound. Each is z-scored over the training simulations where it is defined. An undefined entry is set to −5, with a 0/1 flag beside it: 24 inputs.

## Seeds

- **Prior draws:** `default_rng(5401).random((10100, 20))`. Rows 0–9,999 are the training simulations and rows 10,000–10,099 the truths.
- **Animals:** simulation k's are `1_000_000 + 200·k + i`.
- **The flows:** torch seeds 5402 and 5403. Posterior samples for truth j are drawn with seed `flow_seed·1000 + j`.
- **Posterior-predictive simulations:** k = 10,100 + 400·flow + 20·truth + draw.
- **Prior-predictive simulations:** 400 draws from `default_rng(5404)`, at k = 10,900 onward.
- **The pilot:** animals `2_000_000 + 200·k + i`.

Every scheme is disjoint from every band.

## NPE

`sbi` 0.27 `NPE` with a neural-spline flow (`density_estimator="nsf"`) and sbi's default training settings, trained on the 10,000 training simulations, once per flow seed. Each truth's posterior is sampled 1,000 times.

## The gates

Both are required, for each flow. Either failing, for either flow, fails F.0a.

1. **Calibrated:** over the 100 truths × 20 parameters, the pooled coverage of the central 90% posterior intervals lies in **[0.80, 0.97]**.
2. **Informative:**
   - for truths 0–19, simulate at the first 20 posterior samples;
   - for each truth, take the median over its draws of D between its own simulation and the draw's, over the 11 statistics, z on the bands' scales at n = 200, capped at 10 (undefined in both counts 0, undefined in one counts the cap);
   - the median of those 20 medians is at most **1.42**, W's minimum.

**Reported, not gated:**

- the same D at 20 prior draws per truth;
- each parameter's coverage and contraction (1 − posterior SD × √12, on the unit box);
- coverage on the 25 truths whose own simulation has the lowest D to the bands.

**Why `gate_preflight.py` does not apply:** there are no panel levels, and the informative gate cannot be read before a flow is trained. The pilot measures cost.

## The pilot (band-free; no gate read)

64 simulations from the prior's first rows, on the pilot seed scheme, 16 workers:

| Measure | Value |
|---|---|
| median seconds per simulation, under 16-way contention | 20.0 |
| wall seconds per simulation | 1.29 |
| simulations registered | 11,300 (10,100 plus 1,200 predictive) |
| projected wall time | **4.05 h** |
| simulations with at least one undefined statistic | 81% |

The projection is under the 6-hour bar the design set for bringing the run back to the maintainer. The design's "about 8%" for the predictive simulations understated them: there are 1,200 (11%), counted here.
