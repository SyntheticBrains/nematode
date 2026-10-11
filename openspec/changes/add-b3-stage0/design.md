## Overview

B.3 stage 0 asks whether the connectome, like D.1's MLP, takes the dwelling state on lawns where its
untrained policy does not. It reuses D.1's cell, instrument and registered reading. Only the learner
changes, and its entropy bonus is calibrated first.

## Decisions

### Decision A: The learner

The connectome PPO brain as Phase 8b's earlier cells ran it:
`connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_emmons_reversal.yml`.

- **Wiring:** Emmons 2024, the wild type, with gap junctions and the strict chemical mask.
- **Sensing:** klinotaxis.
- **Settling:** to depth 4.
- **Action:** signed speed.

Everything else is D.1's lawn cell:

- 4 lawns of 2.5 mm, with the worm starting on one;
- intake 2% of a cell per step, and a movement cost of 0.013 per mm;
- turns up to π per step, 720 steps per episode, 3,000 episodes per run;
- no state-shaping reward.

There is **no satiety input**. The connectome has no sensory slot for internal state; satiety reaches
it through B.3's modulators. D.1 found that reading satiety did not change the MLP's dwelling share.

**Arms:** the connectome learner, and its untrained floor (`freeze_updates`).

### Decision B: The entropy calibration

A pilot at two levels, on disjoint seeds 9301–9304:

- **0.005**, D.1's MLP lawn setting. At 0.05 the MLP's noise alone looked like roaming.
- **0.05**, the connectome's point-worm default. The connectome's mean action is settled rates
  through an anatomical readout, bounded near ±1, and it may need more exploration to learn.

The registration takes the level where the connectome passes the learning gate and takes the
dwelling state. If both do, it takes the one with the larger dwelling share over its floor. If
neither does, stage 0 stops for a decision before B.3.

**The pilot, 2026-10-11** (seeds 9301–9304):

| level | gate | dwelling over floor, per seed |
|---|---|---|
| 0.005 | passes, +1.44 [+1.12, +1.77] | −0.01, +0.01, +0.93, +0.73 |
| 0.05 | passes barely, +0.14 [+0.04, +0.24] | none |

**0.005 is registered.** The spread is bimodal: seeds either dwell or roam throughout. So the panel is
sized at **19** seeds (minimum 0.278, MDE 0.277): D.1's 2001–2016, so the learners pair, plus
2017–2019.

### Decision C: The registered reading

D.1's reading: **the learner's on-lawn dwelling share minus its floor's**, paired by seed, classified
by `mc.classify` at a minimum of 2/3 of the pilot's reference (at the chosen level), behind the intake
gate.

- **Seeds:** D.1's, 2001–2016.
- **Size:** 16 unless the pilot's spread demands more.
- **Beside it,** never as verdicts:
  - the connectome against D.1's MLP, paired by seed, from D.1's committed per-run readings;
  - bouts, the depletion and dwelling-end statistics, and action noise.

## Risks

- **The connectome may not learn the cell, or may not dwell.** The pilot exists to find out. A
  failure stops B.3 for a decision, since B.3 reads modulator effects against this baseline.
- **A different entropy from the MLP's** would make the connectome-against-MLP pairing a comparison
  of learners under different settings. That is said beside it, not hidden.
