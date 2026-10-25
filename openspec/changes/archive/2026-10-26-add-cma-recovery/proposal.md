## Why

C.1 is fitted by sep-CMA on the benchmark's distance, like C.2 ([roadmap D38](../../../docs/roadmap.md), as amended 2026-10-25). F.0's remaining control is sep-CMA's: a fit that cannot recover worms it generated itself cannot be trusted on real worms. The maintainer split it in two:

- **F.0b-i**, this change, recovers C.1's model;
- **F.0b-ii** builds C.2's encoder and recovers it.

A planning probe found sep-CMA moving towards a near-wild-type truth, though not yet converged:

- the truth was one of F.0a's prior worms with a floor of 0.62 and a prior-predictive D of 6.94;
- over 30 generations of 16, the population's median D fell from 6.91 to 4.30;
- the re-simulated mean reached a gain fraction of 0.51 while σ was still shrinking.

The probe is uncommitted. This change's pilot measures the same thing on its own seeds and goes into the record.

The maintainer's decisions of 2026-10-25:

- **4 truths × 2 restarts** (decided for F.0b earlier);
- **the budget:** a pilot sets each fit's generation cap, and a registered plateau rule stops a fit early;
- **the gate:** for each truth, the restart with the lower final objective is re-simulated and scored by its gain fraction, (prior D − D) / (prior D − floor), and the median over the 4 truths is at least **0.75**;
- **the start:** each fit starts at the centre of the parameter box with σ₀ = 0.2, as C.1's real fit will.

## What Changes

- **`scripts/analysis/cma_recovery.py`**, reusing F.0a's simulator, distance and pool. Its stages:

  - `preflight`: the truths' floors and stability, the oracle, the unfitted start, and reachable gain;
  - `pilot`: one fit on the pilot truth to 300 generations, which sets the cap and measures the sustained cost;
  - `run`: the 8 registered fits, resumable each generation;
  - `read`: the gate, refused until the launch record is committed and unchanged.

- **The objective**, per D37: D on the benchmark's 11 statistics, local search included, between a candidate's simulation (200 fresh animals on the three assays) and the truth's observation, on the bands' n = 200 scales.

- **Truths:** F.0a's prior simulations near the wild type, taken in a registered order. A candidate is kept only if its first 10 fresh re-simulations leave every statistic's definedness unchanged and its gain is reachable.

- **The preflight** (protocol principles 4 and 6):

  - the oracle, re-simulations of each truth scored as a fit, must reach a median gain of at least 0.85;
  - the unfitted start, the box centre re-simulated, must score a median gain of at most 0.25.

- **Reported beside the gate:**

  - the other restart's gain, the spread that sizes W;
  - parameter errors;
  - the plateau generation;
  - each fit's trajectory.

- **F.0 clears per model** (the maintainer's decision, a dated tracker amendment): once F.0b-i passes, C.1 may be fitted to real worms. C.2 waits for F.0b-ii.

## Capabilities

### Modified Capabilities

- `fit-recovery`: adds sep-CMA's recovery of C.1's model.

## Impact

- **Code:** `scripts/analysis/cma_recovery.py`, with tests. No package code changes.
- **Records:**
  - `docs/experiments/logbooks/supporting/112-cma-recovery/`:
    - `launch.md` in two parts: Part A before the preflight, and Part B after the preflight and pilot, holding the cap and the truths;
    - `preflight.json`, `pilot.json` and `recovery.json`;
  - Logbook 112.
- **Cost:** about 27 s of wall time per generation of 16 on 16 workers:
  - the pilot's 300 generations take about 2.3 h;
  - the registered fits take at most 8 × the cap, and about 18 h at a cap of 300;
  - the preflight's re-simulations add about 15 minutes.
- **Seeds:**
  - CMA seeds 5410–5418 in F.0's band;
  - animals at `50_000_000 + 200·k + i`, and the pilot's at `80_000_000 + 200·k + i`;
  - the planning probe's `32_000_000` and `33_000_000`, recorded as used.
