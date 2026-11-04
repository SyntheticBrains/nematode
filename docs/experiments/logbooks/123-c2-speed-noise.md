# 123: Phase 10 — C.2's Slow Speed Noise: Built and Checked, and Its Positive Control Fails on a Fit That Does Not Run

**Status**: completed (description; nothing here is a registered verdict). **Slow speed noise is built and every check of the mechanism passes. Its positive control fails as registered, because the stage-2 answer it was read on does not run.**

**The mechanism** (`encoder_speed_noise: ou`): the encoded brain's speed output gets slow Ornstein–Uhlenbeck noise, with a fitted timescale.

- The two runners agree on every sampled action.
- The process has the right variance and lag-1 autocorrelation.
- PPO scores the speed by its exact conditional density.
- With the option off, every existing connectome output is byte-identical.

**The positive control.** On C.2's stage-2 answer (Logbook 122), no timescale moves the roaming speed's 90th percentile by more than 0.06 band scales, against a bar of 3.

The answer explains it. Its speed output is pinned at zero (pre-squash median −3.9), and its curvature at the maximum (+9.5). Its worm sits still, spins, and moves only by reversals and delta turns, whose kinematics are fixed from data. No speed noise can reach a speed the squash holds at zero.

With both outputs moved mid-range by hand, slow noise does reach the window statistics: the roaming 90th percentile rises 3.7 band scales at τ = 30 s.

**For the next change,** the problem to fix first is the run mode, not the speed spread. The fit found a worm that does not run, with every behaviour carried by the reversal hazard. W would compare wirings through that one channel. The control is re-run on the first fit whose speed output responds, before the readiness restarts rely on slow noise.

> | Check | Result |
> |---|---|
> | Identity: every existing connectome record, slow noise off | **20 of 20 byte-identical** |
> | Parity, slow noise on: state (50 steps), sampled actions (50), trajectories (300), wild type, Scheer's lawn, 32 seeds | pass |
> | An animal alone and in a batch of 64; a phase start crossed; two tampered controllers fail | pass |
> | The process: variance 1, lag-1 autocorrelation ρ; the first step and outputs 2–5 equal independent noise | pass |
> | PPO: every ratio 1 before an update; the exact conditional and its gradient; τ held | pass |
> | **Positive control on the stage-2 answer** (roaming p90 to rise ≥ 3 band scales) | **fails**: at most +0.02; its speed output unresponsive (pre-squash median −3.89) |
> | Nested at τ = 0.25 s (every kinematics statistic within 2 scales of independent noise) | pass, at most 0.36 |
> | Descriptive: runs moved mid-range by hand | roaming p90 +3.7 scales at 30 s |

**Date**: 2026-11-04.

**OpenSpec change**: `add-c2-speed-noise`.

**Records**: [`supporting/123-c2-speed-noise/`](supporting/123-c2-speed-noise/):

- `identity.json`;
- `positive_control.json`;
- `diagnosis/`, each record beside its script as run:
  - `stage2_answer_variants`: the answer's three variants, which started this change;
  - `actions_profile_old` and `actions_profile_new`: its actions and pools on each wiring;
  - `runs_mid_range`: runs moved mid-range.

## Objective

C.2's staged fit missed the kinematics family on its 10-s window speeds. The plan was a state-dependent per-second std.

The first diagnosis (`diagnosis/stage2_answer_variants.json`, 200 animals of seed block 88,000,000) ruled that out. Variant A is the answer as fitted. Variant B is the same with the speed output's noise cut from log-SD +0.116 to −4. Every kinematics statistic stayed in place, for example the dwelling median at 0.0153 in both.

Ten independent seconds average out inside a 10-s window, so what the statistics need is speed variation that persists across seconds. The maintainer chose:

- slow noise with a fitted timescale;
- PPO support with the exact conditional likelihood;
- a box that contains independent noise;
- encoder only, with the timescale held under PPO.

The same diagnosis graded the mutants on the answer for the first time, and all three failed:

- tph-1 changes the roaming fraction by +0.002 (z −5.3);
- pdfr-1 by +0.146 (z +4.4);
- tax-4 by +0.40, the wrong direction and capped at z 10.

On the corrected AIA/ADE wiring, the same parameters moved a lot: roaming fraction 0.38 → 0.66, D_fit 3.7 → 5.4. That was expected, since the answer was fitted on the earlier wiring.

## The mechanism

- **The process:** x_t = ρ·x\_{t−1} + √(1−ρ²)·ε_t, on the speed output's standardised noise, with ρ = exp(−1 s/τ). Here ε_t is that second's own draw, so no random number is added and the other four outputs are unchanged.
  - x_0 = ε_0 at each episode's start.
  - It carries across an assay's phases.
- **The timescale:** the encoder's 197th parameter (133rd per ligand), `timescale.speed_noise`, log-scaled over 0.25–300 s.
  - At 0.25 s, ρ = 0.018, which is independent noise in practice.
  - The box centre is 8.7 s, at ρ = 0.891.
  - A 196-entry vector is a prefix of a 197-entry one.
- **Under PPO:**
  - The buffer stores each step's previous pre-squash speed and state.
  - The update recomputes the previous mean under the current parameters, so the speed's density is N(m(s_t) + ρ·(u\_{t−1} − m(s\_{t−1})), σ²(1−ρ²)).
  - The timescale is a held buffer.

## The checks

- **Process** (`test_slow_speed_noise.py`): over 20,000 steps at ρ = 0.9, variance 1 and lag-1 autocorrelation 0.9. An episode's first step equals the independent sampler bit for bit, and outputs 2–5 do at every step.
- **PPO:** before an update every log-prob equals the stored one. Under changed parameters the evaluator is the analytic exact conditional, and its gradient through the previous mean is −ρ times the speed's score. A short PPO run updates, with finite losses, and leaves τ unchanged.
- **Parity** (`test_batched_parity.py`): the encoded wild type with slow noise on Scheer's small lawn, on seeds 5001–5032, meets the layered parity on features, means and derivatives, the sampled actions themselves (within 1e-5), and trajectories over 300 steps. Further checks:
  - one animal alone matches it inside a batch of 64;
  - a 100-s phase start is crossed with the noise state carried;
  - a controller that restarts the noise at the phase, and one that draws it independently, both fail.
- **Refusals:** slow noise without the encoder, in a validated config or one derived by `model_copy`. A weight file from the other mode is refused in both directions.
- **Identity** (`identity.json`): 20 of 20 jobs byte-identical to a baseline recorded before the change: every batched and per-step connectome job, and the built brains of the per-connection configs and the encoded one.

## The positive control (`positive_control.json`)

The stage-2 answer on the wiring it was fitted on was graded on 200 animals of seed block 88,100,000. The same animals ran with independent noise and with slow noise at each timescale.

| τ (s) | Dwelling median | Dwelling p90 | Roaming median | Roaming p90 | Roaming p90 rise, band scales |
|---|---|---|---|---|---|
| independent | 0.0153 | 0.0356 | 0.0340 | 0.0518 | — |
| 0.25 | 0.0152 | 0.0354 | 0.0343 | 0.0521 | 0 |
| 2 | 0.0152 | 0.0354 | 0.0342 | 0.0519 | −0.04 |
| 8.7 | 0.0154 | 0.0359 | 0.0343 | 0.0520 | −0.01 |
| 30 | 0.0154 | 0.0358 | 0.0345 | 0.0522 | +0.02 |
| 100 | 0.0152 | 0.0355 | 0.0342 | 0.0520 | 0.00 |
| 300 | 0.0151 | 0.0357 | 0.0344 | 0.0518 | −0.06 |

Real worms: dwelling median 0.0128 and p90 0.0282; roaming median 0.0373 and p90 0.0733.

**The rise fails:** at most +0.02 band scales against 3. **The nesting passes:** at 0.25 s every kinematics statistic is within 0.36 scales of independent noise. **The answer's speed output does not respond:** its pre-squash median over one 1,800-step reference episode (seed 88,100,200) is −3.89, outside the ±2 the revised requirement reads the control within. The record carries all three; its graded points were re-run once, when the reading was revised, and repeated exactly.

## Why: the answer does not run

`diagnosis/actions_profile_old.txt` covers four 1,800-s episodes on Scheer's small lawn, on the wiring the answer was fitted on:

- **The run speed is pinned at zero:** the sampled raw speed's median is −0.999, about 0.0001 mm/s.
- **The curvature is pinned at the maximum:** its raw value is 1.0 at every percentile.

| Output | Pre-squash mean (5th / 50th / 95th percentile) |
|---|---|
| speed | −4.32 / −3.88 / −3.43 |
| curvature | +5.32 / +9.47 / +11.98 |
| reversal hazard | −1.95 / +0.16 / +3.22 |
| omega hazard | −4.21 / −3.61 / −2.09 |
| delta hazard | −3.56 / −2.84 / −2.10 |

**The network is not saturated.** The pooled motor rates vary: VB −0.97 to −0.55, DB −0.99 to −0.75, VA −0.99 to +0.12, DA −1.00 to −0.20. It is the readout that drives speed and curvature off the ends of the squash. Its weights are up to ±3 on pools in ±1, with a bias of up to ±3. Only the reversal hazard stays in the squash's responsive range.

So the worm moves only by its bouts: 30% of seconds in reversals and 20% in delta turns, in this profile. In the graded window the grader reports a reversal share of 0.209, against real worms' 17.3%.

A reversal on a lawn backs at a fixed 0.069 mm/s. That is why the answer's roaming p90 tops out near 0.052 mm/s, against the real 0.073. It is also why it fails the held-out off-food statistics: onset rate z +5.9, direction flips z +6.1. On the corrected wiring (`actions_profile_new.txt`) the pinning is the same, with reversals 47% of seconds.

## Description: runs moved mid-range (`diagnosis/runs_mid_range.json`)

As a check of the mechanism, not a re-read of the control: the speed and curvature readout biases were moved, outside the box, by +3.89 and −9.87. That puts each output's pre-squash median over a reference episode at 0. Independent noise and four timescales were graded on 200 animals of seed block 88,300,000.

| τ (s) | Dwelling median | Dwelling p90 | Roaming median | Roaming p90 | Small-lawn roaming fraction | Time on the lawn |
|---|---|---|---|---|---|---|
| independent | 0.0340 | 0.0598 | 0.0458 | 0.0710 | 0.661 | 0.009 |
| 2 | 0.0280 | 0.0561 | 0.0445 | 0.0733 | 0.581 | 0.013 |
| 8.7 | 0.0224 | 0.0569 | 0.0422 | 0.0755 | 0.413 | 0.021 |
| 30 | 0.0192 | 0.0467 | 0.0494 | 0.0859 | 0.220 | 0.018 |
| 100 | 0.0160 | 0.0389 | 0.0433 | 0.0784 | 0.148 | 0.040 |

Once runs carry speed, slow noise moves every window statistic. The roaming p90 rises 3.7 band scales at 30 s, and the dwelling median falls by half by 100 s.

This worm leaves the lawn, though: it spends 1–4% of the assay on it. So these on-lawn windows are few and the values are indicative only. The table shows the mechanism reaches the statistics. It does not show what slow noise does for a fit.

## What this establishes, and what it does not

- **Established:** slow speed noise is built, byte-identical when off, the same in both runners, and scored exactly under PPO. It reaches the window statistics when the speed output responds.
- **Not established:** that it helps a fit. The registered control failed on the only fit available, which does not use its run speed. The control is re-run on the first fit whose speed output responds, before the readiness restarts rely on it.
- **Found:** the stage-2 answer is degenerate. It holds the lawn by not running, and every behaviour passes through the reversal hazard. Its mutants all fail, as do its held-out off-food statistics.

## For the next change

1. **The run mode first.**
   - Report each output's pre-squash range at the box centre and along a fit.
   - Keep speed and curvature responsive, for example with a smaller readout box or a readout on normalised pools.
   - Make stage 1 hold the lawn by running slowly rather than not at all: either the kinematics in its objective, or a run-share check in its bar.
2. **Then the positive control,** on the first fit whose speed output responds.
3. **Then the staged procedure registered and the readiness restarts,** with the mutants graded at every restart. The maintainer's choice for the mutants, held out or fitted, waits for those restarts.

For W: slow noise comes from outside the network, so every W arm gets the same handle on the kinematics family. W's registration reads that family with this in mind.

## Corrections and lessons

- **Check what an answer's outputs are doing before designing a fix for its statistics.** The speed misses looked like a missing spread. The first diagnosis showed per-second noise did not matter; only the action profile showed the speed was pinned at zero, which no noise can fix. A per-output pre-squash range is a two-minute check and comes first next time.
- **A positive control needs an answer that can express it.** The control was registered on the stage-2 answer without checking that its speed output responded. The revised requirement makes that a condition of reading it.
