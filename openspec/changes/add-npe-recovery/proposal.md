## Why

D38 fits C.1 by neural posterior estimation (NPE) and C.2 by sep-CMA, and requires each method to pass a **parameter-recovery positive control** first (F.0). The plan review's finding A6 is the reason. A wild-type-only NPE can leave pathways on a ridge, and then the posterior reports the prior. A fit that cannot recover parameters it generated itself cannot be trusted on real worms.

The maintainer's decisions of 2026-10-25:

- **two changes, NPE first.** This change, F.0a, builds C.1's mechanistic baseline and recovers it by NPE. F.0b builds C.2's encoder and recovers it by sep-CMA.
- **NPE by `sbi`;**
- **gates in calibration** for NPE, and in statistic space for sep-CMA;
- **sep-CMA at 4 truths × 2 seeds** (F.0b).

## What Changes

- **C.1's mechanistic baseline**, `BatchedMechanistic`, a closed-loop controller with 20 parameters on the primitives:
  - **a roaming/dwelling state.** Its switching hazards depend log-linearly on intake and satiety, after Roberts 2016 and Flavell 2013.
  - **per-state runs,** with weathervane steering: curvature biased by the head-sweep difference.
  - **reversals and omega turns.** Klinokinesis scales their hazards down as dC/dt rises (Pierce-Shimomura 1999). A falling food-contact reading raises the reversal hazard: the edge response behind Scheer's head-poke reversals. After food is lost, the hazards are raised and decay with time off food (a search term).

  It is the model C.1 fits. Mutant encodings are C's, registered before C.1 fits.
- **Priors without search data** (D37): box or log-uniform ranges from published magnitudes and the band data's lawn statistics. No López-Cruz value enters a prior.
- **NPE recovery** (`scripts/analysis/npe_recovery.py`):
  - the simulator is the batched evaluator at 200 animals on the three wild-type assays, summarised by the benchmark's 11 wild-type statistics plus presence;
  - 10,000 training simulations drawn from the prior;
  - two NPE flows trained with different seeds;
  - simulation-based calibration on 100 held-out truths from the prior.
- **The registered gates**, both required for each flow:
  1. **calibrated:** the pooled 90% posterior-interval coverage over 100 truths × 20 parameters lies in **[0.80, 0.97]**;
  2. **informative:** over 20 truths, simulations at 20 posterior draws reproduce each truth's statistics with a **median D of at most 1.42**, W's minimum.

  Calibration alone would pass a posterior equal to the prior, the failure A6 named. Reported beside the gates, not gated:
  - the same D at 20 prior draws, so the gain over the prior is visible;
  - each parameter's coverage and contraction (posterior SD over prior SD), naming which parameters the benchmark constrains;
  - coverage on the quarter of truths nearest the wild type.
- **`sbi`** pinned in the `analysis` extra.

## Capabilities

### New Capabilities

- `mechanistic-baseline`: C.1's controller and its parameters.
- `fit-recovery`: the recovery protocol and gates for the fitting methods. This change adds NPE; F.0b adds sep-CMA.

## Impact

- **Code:**
  - `quantumnematode/evaluation/controllers.py` gains `BatchedMechanistic`;
  - `scripts/analysis/npe_recovery.py`;
  - `packages/quantum-nematode/pyproject.toml` and `uv.lock` gain `sbi`.
- **Records:**
  - `docs/experiments/logbooks/supporting/111-npe-recovery/`: `launch.md`, committed before the run, with the pilot's cost; the simulations' statistics; the calibration;
  - Logbook 111.
- **Cost:** about 10,100 simulations at an estimated 12 s each, about 34 CPU-hours: 2–3 hours on 16 worker processes. The pilot measures it, and the run is registered over two hours, so the spec review and a pilot gate it. Training the flows takes minutes.
- **Seeds:**
  - simulation k's animals are `1_000_000 + 200·k + i`, i < 200, disjoint from every band;
  - truths and the prior draws use `default_rng(5401)`; the flows use 5402 and 5403 (F.0's band, 5401–5464);
  - a pilot on simulations 0–63 of a separate scheme, `2_000_000 + 200·k + i`;
  - smokes on 5991–5999.
