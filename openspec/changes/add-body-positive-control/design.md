## Overview

C.1d asks whether the strongest available learner forages through the body, and whether what it
produces moves like a worm. Its answer gates C.1e. It freezes the body's last free parameter, the
steering gain, so that every later arm runs the same body.

## Decisions

### Decision A: The body's parameters

| parameter | value | source |
|---|---|---|
| period | 3.33 s | crawling frequency 0.30 ± 0.02 Hz (Fang-Yen et al. 2010, *PNAS* 107:20323, Table 1) |
| wavelength | 0.65 body lengths | 0.65 ± 0.03 body lengths (Fang-Yen et al. 2010) |
| drag anisotropy `c_n / c_t` | 10 | 9.4 ± 0.6 on wet agar (Shen et al. 2012, *Biophys J*); 222.0 / 22.1 measured directly (Rabets et al. 2014, *Biophys J*) |
| peak curvature `A₀` | 18 body-lengths⁻¹ | see below |
| steering gain `B₀` | calibrated (Decision C) | no direct measurement |

**Peak curvature.** Typical crawling postures have amplitude A/q ≈ 1 at wavevector qL ≈ 9, and
Ω-shapes have A/q ≈ 2 (Bilbao et al., "Navigation of *C. elegans* in three-dimensional media", arXiv
1609.03452). The sourced wavelength gives qL = 2π / 0.65 ≈ 9.7, in agreement.

Peak curvature is 18, which spreads the drive across the full posture range:

| drive | amplitude factor | peak κL | posture |
|---|---|---|---|
| zero (neutral) | 0.5 | about 9 | the sourced crawl (W-shape) |
| full | 1 | 18 | Ω-shape |
| minimum | 0 | 0 | straight |

The brain can therefore damp, keep or exaggerate the crawl, which C.3's omega-turn geometry needs.

**A defect in C.1c, fixed here.** The head switch flips when the bend crosses ±θ (θ = 0.5), so the
relayed wave spanned ±θ, not the ±1 the amplitude mapping assumes. Every posture was therefore half as
curved as intended. The relay now divides the wave by θ, so that it spans ±1. Measured on the body at
the sourced parameters:

| drive | peak κL | speed (body lengths/s) |
|---|---|---|
| neutral | 7.9 | 0.109 |
| full | 15.7 | 0.08 |

The neutral peak is close to the sourced 9; the switch's exponential wave shape accounts for the
difference. The neutral speed sits just below C.3's speed band. Very large amplitudes are less
efficient, so speed falls at full drive. The body is not tuned toward the band, which is what this
control validates against, and the trained MLP chooses its own amplitude.

**Two consequences of the larger postures**, measured when the defaults changed:

- **Convergence.** At 20 sub-steps the motion is within about 4% of fully converged at neutral drive,
  and within about 11% at high amplitude, against the 1% C.1c measured at its smaller placeholder
  postures. The half-step check (Decision E) reads what this does to the instruments on trained
  policies.
- **Heading.** A body started straight settles within about 10 steps into a straight crawl along a
  stable heading. A step is 1.5 periods, so successive steps sample opposite phases of the head's
  swing, and the heading the brain senses alternates by about ±0.06 rad from step to step: a wobble,
  not a drift.

### Decision B: The reversal threshold

The wave runs tail-to-head only when the direction channel is below **−0.5**, for every arm. Under a
sign threshold, any policy whose direction sits near zero would flip its wave at random from step to
step. The threshold makes a reversal a deliberate output.

If an untrained policy's direction were centred, its per-step reversal probability would be
`Φ(−artanh(0.5) / σ)`: about 0.07 for the MLP (σ 0.37) and 0.29 for the connectome (σ 1.0). **It is not
centred.**

**Measured, 2026-10-07.** Twelve untrained MLP seeds (1480–1491, outside both registered bands), one
evaluation episode each, through the body at the hard350 config:

- The direction channel's mean is a per-seed draw from the initial weights, from −0.70 to +0.48.
- So the untrained reversal probability ranges from **0.02 to 0.82 per step**, median about 0.27.
- Five of the twelve seeds reverse on more than half their steps.

The MLP keeps its standard initialisation, for three reasons:

- every earlier MLP arm used it, including C.0's signed-speed reversal arm, which learned;
- the frozen floor is paired by seed, so each seed's starting bias is in its own floor;
- centring it would make this control a different MLP from the one it stands for.

The pilot reports each seed's untrained and trained reversal fraction, so a gain is never chosen on a
seed that learned only to stop reversing.

**The worm's reversal rate, checked 2026-10-07.** Spontaneous reversal frequency is
condition-dependent, not a constant. Zhao et al. (2003, J Neurosci 23:5319) find that humidity,
gravidity and touch all move it. Off food on wet agar it is several per minute, about 5 in secondary
summaries, and it falls as local search gives way to global search over roughly 15 minutes (Gray,
Hill & Bargmann 2005, PNAS 102:3184). The primary figures were not open to check the exact values.

"One to a few a minute" is therefore an order of magnitude, not a band. A one-step reversal at one to
five a minute is a per-step fraction of about 0.08 to 0.4. The registration cites it as that order of
magnitude, and reversal fraction stays descriptive, as Decision E has it.

### Decision B′: A reversal is brief *(added 2026-10-08, after the first pilot)*

**What the first pilot showed.** It ran with reversals uncapped (32 runs, seeds 1501–1504). Each
trained seed settled on one gait. Trained reversal fractions were near 0 or near 1, never between.
Seed 1502 started biased toward reversing (0.72 untrained) and at gains 1 and 2 crawled **backward on
100% of its steps**. At gain 1 it had the lowest plateau, 17%. A worm's reversals are brief, one to three
head swings, and followed by forward crawling. A body that can crawl backward indefinitely lets a seed's
starting bias choose a gait no worm uses.

**The cap.** A reversal runs tail-to-head for at most **one step** (5 worm-seconds, 1.5 head swings),
then the body crawls forward for at least **one step** before it may reverse again. Both limits are
`BodyParams` fields. A held reversal request therefore alternates, reversed and forward. The body
records each step's executed direction, which the reversal-fraction instrument reads.

**What it gives up.** At one step only short reversals exist. The long reversals that precede an omega
turn in a pirouette, three or more head swings, cannot be produced. That is recorded as a kinematic
condition, to revisit at C.3's behavioural validation, where pirouettes are graded.

**Consequence for the calibration.** The cap changes the body every gain runs through, so the first
pilot no longer calibrates it. That pilot is kept as evidence ([its readings are committed beside the
registration](../../../docs/experiments/logbooks/supporting/083-body-control/)) and the pilot re-runs
under the cap on the same seeds, same configs, same rule.

### Decision B″: A segment's wave is damped, never silenced *(added 2026-10-08, after the second pilot)*

**What the capped pilot showed.** Under the reversal cap no seed locked into a gait: trained reversal
fractions were 0.004–0.047. But the trained MLP silenced 2.7–5 of the 12 segments per step. At gain 1
the head segment's wave sat below a quarter of the peak on 94–98% of steps for seeds 1501 and 1502,
with mean amplitude 0.01–0.03. Only 0–5% of wall-clear steps had every segment undulating, so frequency
and wavelength could not be read on three of sixteen runs. Where they could, they read 0.315 Hz and
0.62 body lengths.

**The floor.** A segment's wave amplitude now spans **[0.25, 1]** of the peak, not [0, 1]. Neutral drive
still gives 0.5, so the sourced neutral crawl is unchanged. Positive drive keeps its old mapping, and
negative drive damps down to 0.25 rather than 0. In forward crawling the wave propagates along the
whole body through proprioceptive coupling: posterior bending follows anterior bending (Wen et al.
2012). A body in which the head can stop undulating while the rest crawls is not the worm's. The floor
applies to every arm, the connectome's included, since the drive is the same for both.

**Consequence for the calibration.** The floor changes the body every gain runs through, so the pilot
re-runs a third time, on the same seeds, configs and rule. Both earlier pilots are kept as evidence:
[pilot-uncapped.json](../../../docs/experiments/logbooks/supporting/083-body-control/pilot-uncapped.json)
and [pilot-capped.json](../../../docs/experiments/logbooks/supporting/083-body-control/pilot-capped.json),
with their kinematics. Both chose gain 1.

### Decision C: The steering calibration, a rule fixed first

**The pilot.** MLP-PPO through the body at **B₀ ∈ {0.5, 1, 2, 4}**, on hard350 with reversal on, 3,000
episodes, **seeds 1501–1504**, learning and frozen at each gain, 32 runs. The MLP has no connectome;
Emmons 2024 applies to the connectome arms that follow.

**The rule.** Choose the B₀ with the highest mean plateau success. Differences under 5 points are ties,
broken toward the smaller gain, the gentler steering. If no B₀ beats the frozen floor at any seed, the
pilot is C.1d's diagnosis, and no control runs.

**After it.** The chosen gain is frozen in the body's defaults. Its neighbours' plateaus are reported
as the sensitivity check (D18). Only MLP-PPO runs, so no wiring result informs the choice.

### Decision D: The positive control

MLP-PPO through the body at the chosen B₀, on hard350 with reversal on, 3,000 episodes, **seeds
1505–1512**, learning and frozen.

**Passes** if:

- the learning arm beats its frozen floor, paired by seed, with the 80% interval of the plateau
  difference above zero;
- every seed's plateau success is at least 30%, the episode metric's competence level.

**The fallback**, fixed before the control runs. If the floor gate passes but competence fails, a
gate-only pilot lengthens the episode, at 500, then 700 and then 1,000 steps. It runs the learning arm
only, on **seeds 1513–1516**. The shortest length at which every seed reaches competence is chosen,
and the control re-runs there, learning and frozen, on fresh **seeds 1517–1524**. The cell's meaning moves, which C.1e already treats as a new reference frame.

If the floor gate fails, C.1d fails, and the diagnosis is its deliverable.

**Sequencing** *(added 2026-10-08)*. In both earlier pilots one of four seeds sat under the competence
bar at the chosen gain, so the fallback is likely. The fallback's gate-only pilot (learning arm, 500,
700 and 1,000 steps, seeds 1513–1516, 12 runs) therefore launches together with the control, as one
campaign. Only the order changes. The fallback pilot is read only if the control reads `fallback`;
otherwise it is reported beside the control as description. The re-run on seeds 1517–1524 still waits
for the control's reading.

### Decision E: The kinematic instruments

Each trained control run's final weights are evaluated for 10 episodes with posture capture. The body
records each sub-step's curvature and head position, every 0.25 worm-seconds at 20 sub-steps.

- **Undulation frequency.** Half the rate of the mid-body curvature's crossings of its own mean, over
  each unbroken stretch of forward-running, wall-clear steps. A crossing counts only after the curvature has left a ±0.31 κL band around that mean (C.3's
  adopted band rule).
- **Wavelength.** From the phase lag of curvature along the body, in body lengths. Each segment's
  delay is read against its neighbour's and summed down the body, since a lag read against the head
  wraps past a period once the wave needs longer than one to arrive.
- **Speed.** The head's net displacement per worm-second, in body lengths per second, from each
  episode's second step on (the first has no recorded starting pose).
- **Reversal fraction.** The share of steps run tail-to-head.

**Wall exclusion.** Steps whose head lies within 1 mm of a wall are excluded (H.3's margin), and the
exclusion is recorded.

**Bands** (C.3's adopted thresholds): frequency 0.2–0.45 Hz, wavelength 0.5–0.8 body lengths, speed
0.12–0.3 body lengths per second. A control that forages but sits outside a band is recorded as a
kinematic condition on every later body result. It is not a foraging failure.

**Undulating steps only** *(added 2026-10-08, after the first pilot)*. The trained policies switched
individual segments' waves on and off: at gain 1, seed 1501's head segment ran at a mean amplitude of
0.06 against about 0.9 at mid-body, and amplitudes spanned 0 to 1 between steps. A silenced segment
holds still at its steering offset and crosses its mean only when the drive changes between steps. The
crossing delays read that as a wave, and the pooled wavelength read 0.36 body lengths against the body's
0.66. Frequency and wavelength are therefore read only on **undulating** steps: forward, wall-clear, and
with every segment's wave amplitude at least **0.25** of the peak. The share of steps that qualify is
reported beside. Speed and reversal fraction stay on every wall-clear step. Real-worm analyses likewise
read the wave on forward runs.

**Half-step check.** The same weights are re-evaluated at 40 sub-steps. Each instrument's mean over the
control's learning runs must agree within 10% between the two. Reversal fraction must agree within
10% or 0.01 absolute, whichever is larger, since a fraction near zero has no stable relative error.
Means are compared, not single runs: the two sub-step counts drive the policy down different
trajectories, so single-run readings also differ by sampling.

Before any run, the body alone at full drive moves 9% slower at 40 sub-steps than at 20 (Decision A's
11% non-convergence at high drive), close to the bar. A control that spends its steps at high drive
can fail the check on speed. That would be a property of the body's integration, recorded as such.

**What the instruments can and cannot show.** The frequency and wavelength are largely set by the
generator's fixed parameters, so reading them mostly confirms the generator and its drive modulation.
Speed and reversal fraction are emergent. The eigenworm check needs Stephens' basis, which is not
vendored, and stays with C.3.

## Risks

- **The body is slower than the point worm**, and hard350's 350 steps may be too few. The fallback
  handles it.
- **The reversal rate's reference is recalled, not checked.** It is verified before the registration
  cites it.
