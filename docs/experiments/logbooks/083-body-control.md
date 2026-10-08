# 083: MLP-PPO Learns Hard350 Through the Kinematic Body, at 500 Steps (Phase 8b C.1d)

**Status**: completed — **passes, through the registered fallback.** MLP-PPO forages through the
25-number body drive and the 12-segment kinematic body, far above its frozen floor and with every seed
competent, once the episode is 500 steps rather than hard350's 350.

| control, MLP-PPO through the body | plateau | floor | plateau − floor, 80% CI | seeds ≥ 30% | verdict |
|---|---|---|---|---|---|
| 350 steps, seeds 1505–1512 | 41.7% | 0.0% | +41.7 [36.9, 46.3] | 7 of 8 | `fallback` |
| **500 steps, seeds 1517–1524** | **92.5%** | **0.6%** | **+91.9 [90.2, 93.6]** | **8 of 8** | **`passes`** |

Its posture is the sourced crawl: **0.300 Hz** and **0.645 body lengths**, both in band. Its speed,
**0.122 body lengths per second**, sits on the band's 0.12 floor: 0.117 at twice the sub-steps, so the
registered reading is **at the band edge**. The half-step check passes.

**The body's last free parameter is frozen: steering gain 2.** Two body properties were added on the
way, each on a pilot's finding: **a reversal lasts one step**, and **a segment's wave is damped, never
silenced**. C.1e may register, on the 500-step cell.

**Date**: 2026-10-08.

**OpenSpec change**: `add-body-positive-control`.

**Pre-registration**: [supporting/083-body-control/launch.md](supporting/083-body-control/launch.md),
committed before any scored run, after the spec review and a half-step preflight.

## Objective

C.1a–c gave the connectome brain a body: an anatomical motor-to-muscle readout, a 25-number drive, and
a 12-segment kinematic body under resistive-force theory. Before any wiring is read through it, a
learner known to learn the cell must learn it through the body (protocol principle 4). MLP-PPO learns
hard350 with reversal on as a point worm at 97.1% ([Logbook 081](081-body-prerequisites.md)). If it
cannot forage through the body, nothing read later from a connectome through it is interpretable.

The roadmap also asks for the control to be a kinematic pass, read with C.3's adopted instruments, not
only a foraging one.

## Method

### The body's parameters

The crawl's period (0.30 Hz) and wavelength (0.65 body lengths) are Fang-Yen et al. 2010's, the drag
anisotropy (10) Shen et al. 2012's and Rabets et al. 2014's, and the peak curvature (18 body-lengths⁻¹
at full drive) puts the neutral drive at the sourced crawl and full drive at an Ω-shape (Bilbao et
al.). C.1c's relayed wave spanned ±0.5 instead of ±1, halving every posture; it is fixed. The reversal
threshold is −0.5 on the direction channel for every arm.

### Three calibration pilots

The steering gain has no direct measurement. A pilot of MLP-PPO at gains 0.5, 1, 2 and 4, learning and
frozen, on seeds 1501–1504 (32 runs), chooses it by a rule fixed first: the highest mean plateau, ties
within 5 points to the smaller gain. The pilot ran three times, because the first two each found a way
the body let the policy move like no worm, and each fix changed the body every gain runs through:

| pilot | body | chosen gain | what it found |
|---|---|---|---|
| uncapped | as C.1c left it | 1 | each seed locked into one gait; seed 1502 crawled **backward on every step** |
| capped | reversals last one step | 1 | the MLP **silenced 2.7–5 of 12 segments per step**, the head on 94–98% of steps for two seeds |
| floored | wave amplitude ≥ 0.25 of the peak | **2** | below |

- **A reversal is brief.** A worm reverses for one to three head swings and resumes forward crawling. A
  body that can crawl backward indefinitely let a seed's starting bias choose a gait no worm uses. A
  reversal now lasts at most one step (5 worm-seconds, 1.5 head swings) and is followed by at least one
  forward step.
- **A segment's wave is damped, never silenced.** In forward crawling, bending propagates along the
  whole body through proprioceptive coupling (Wen et al. 2012). A body whose head could stop undulating
  while the rest crawled is not the worm's. A segment's wave amplitude now spans [0.25, 1] of the peak,
  neutral still 0.5.

On the final body the gain moved from 1 to 2. With silencing gone the policy steers through the
dorsal–ventral bias, which needs more gain:

| gain | mean plateau | seeds 1501–1504 | floor |
|---|---|---|---|
| 0.5 | 5.2% | 4.9, 8.5, 5.5, 1.7 | 0% |
| 1 | 15.4% | 11.7, 17.5, 3.5, 28.8 | 0% |
| **2** | **37.7%** | 50.0, 34.9, 32.8, 32.9 | 0% |
| 4 | 39.2% | 46.3, 44.4, 26.3, 40.0 | 0% |

Gain 4 is +1.6 points, a tie, so 2 is chosen and frozen as the body's default for every arm. Its
neighbours, for D18's sensitivity check: gain 1 is −22.3 points, gain 4 +1.6.

### The control

MLP-PPO, width 64, through the body at gain 2 on hard350 with reversal on, 3,000 episodes, learning and
frozen, **seeds 1505–1512**. **Passes** if the plateau beats the frozen floor paired by seed (80%
interval above zero) and every seed reaches 30%. A passing floor with a seed under 30% sends it to the
**fallback**: a gate-only pilot of the learning arm at 500, 700 and 1,000 steps on seeds 1513–1516
chooses the shortest length at which every seed reaches 30%, and the control re-runs there on **seeds
1517–1524**. The pilot put three of four seeds within 5 points of the bar, so the fallback pilot
launched with the control and was read only if the control read `fallback`.

### The kinematic instruments

Each learning run's final weights are evaluated frozen for 10 held-out episodes with sub-step posture
capture:

- **frequency**, from mid-body curvature's crossings of its mean, a crossing counting only after
  leaving a ±0.31 κL band;
- **wavelength**, from crossing delays between adjacent segments summed down the body;
- **speed**, the head's displacement per worm-second;
- **reversal fraction**, the share of steps the body ran tail-to-head.

Frequency and wavelength are read on undulating, forward, wall-clear steps; steps within 1 mm of a wall
are excluded. The bands are frequency 0.2–0.45 Hz, wavelength 0.5–0.8 body lengths, speed 0.12–0.3
body lengths per second. The **half-step check** re-reads the same weights at 40 sub-steps: every
instrument's mean must agree within 10%. Each band is read at both sub-step counts, and a reading that
differs between them is **at the band edge**.

## Results

### The control at 350 steps: `fallback`

The floor gate passes, +41.7 points [36.9, 46.3] over 0% floors. Seven seeds reach 30.4–57.3%; seed
1507 reaches 26.9%. Competence fails by one seed.

### The fallback pilot: 500 steps

| length | seeds 1513–1516 |
|---|---|
| **500** | 73.5, 93.7, 99.1, 89.2 |
| 700 | 99.9, 99.2, 100.0, 100.0 |
| 1,000 | 100.0, 99.7, 99.7, 100.0 |

500 is the shortest length at which every seed is competent.

### The control at 500 steps: `passes`

| seed | 1517 | 1518 | 1519 | 1520 | 1521 | 1522 | 1523 | 1524 |
|---|---|---|---|---|---|---|---|---|
| plateau | 96.1 | 86.5 | 87.1 | 96.8 | 95.6 | 89.1 | 93.7 | 94.8 |
| floor | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7 | 3.9 |

Mean plateau **92.5%** against a **0.6%** floor: **+91.9 points [90.2, 93.6]**, every seed above 86%.

### Kinematics, on the passing control

| instrument | 20 sub-steps | 40 sub-steps | band | reading |
|---|---|---|---|---|
| frequency | 0.300 Hz | 0.300 Hz | 0.2–0.45 | **in** |
| wavelength | 0.645 body lengths | 0.645 | 0.5–0.8 | **in** |
| speed | 0.122 body lengths/s | 0.117 | 0.12–0.3 | **edge** |
| reversal fraction | 0.009 | 0.008 | descriptive | |

The half-step check passes: speed moves −4.6%, everything else agrees to three places. 97–100% of
wall-clear steps undulate. Per seed, speed ranges 0.115–0.128 and reversal fraction 0.001–0.030
([per-seed.csv](supporting/083-body-control/per-seed.csv)). The 350-step control reads the same: 0.300
Hz, 0.649 body lengths, speed 0.123 / 0.116 (edge), reversal fraction 0.011.

## Analysis

**The body is learnable, and slower than the point worm.** The same learner, the same cell, the same
3,000 episodes: 97.1% as a point worm, 41.7% through the body at 350 steps, 92.5% at 500. The body
covers ground at about 0.12 body lengths per second, near the bottom of the worm's band, so a cell sized
for the point worm's reach is too short for it. Lengthening the episode restores competence without
changing anything else about the task.

**The posture is the sourced crawl, and that is mostly by construction.** Frequency and wavelength are
set by the body's generator, so their readings confirm the generator and show the trained drive does not
distort it. Speed and reversals are what the policy makes of the body.

**Speed sits on the band's floor.** At 20 sub-steps it reads inside the band; at 40, slightly below.
The more converged reading is lower, so the body's true crawl is probably just under the worm's band.
That is recorded as a kinematic condition on every later body result.

**The trained MLP rarely reverses.** 0.009 of steps is about one reversal every nine minutes. A worm
reverses several times a minute off food, a per-step fraction of roughly 0.08–0.4 (Zhao et al. 2003;
Gray, Hill & Bargmann 2005). Reversal is not graded here, and hard350 does not reward it: forward
klinotaxis reaches the food. It is recorded beside.

## What this establishes, and what it does not

- **Established**: MLP-PPO learns hard350 through the 25-number body drive and the kinematic body, at
  500 steps, far above its frozen floor with every seed competent. **The body's positive control passes**,
  and C.1e may register.
- **Established, with its condition**: the cell is hard350's food layout at **500 steps**, not 350. That
  moves the cell's meaning, which C.1e already treats as a new reference frame: no delta against 029,
  block V or any point-worm result.
- **Established**: the body's parameters are frozen for every arm: steering gain 2, a one-step reversal
  with a one-step refractory period, a wave floor of 0.25.
- **Not established**: anything about the connectome through the body. That is C.1e.
- **A warning for C.1e's preflight**: at 500 steps the MLP reaches 86–97%, and the fallback pilot's
  700- and 1,000-step arms reached 99–100%. Wiring contrasts are unreadable where both learning arms
  reach 90%. If the connectome also learns this cell that well, the saturation gate will bite; C.1e's
  gate preflight must read it on connectome pilots at 500 steps before anything else.

## Biological fidelity

**Right as built**: the period, wavelength, drag anisotropy and crawl amplitude are measured values. A
brief reversal and a wave that propagates the whole body are the worm's. The motor-to-muscle map is
anatomy.

**Fine for now, and recorded as conditions:**

- **Reversals are short only.** At one step, 1.5 head swings, the long reversals that precede an omega
  turn cannot occur. C.3 grades pirouettes, and revisits this there.
- **The wave floor's value, 0.25 of the peak, is not measured.** Wen et al. 2012 justify a floor above
  zero, not its value.
- **The steering gain is a calibration**, not a measurement.
- **Speed sits on the band's floor**, probably just under it once the integration converges.
- **The trained control reverses far less often than a worm.** The task does not ask it to.

## Artefacts

- [launch.md](supporting/083-body-control/launch.md): the registration.
- [pilot-uncapped.json](supporting/083-body-control/pilot-uncapped.json),
  [pilot-capped.json](supporting/083-body-control/pilot-capped.json),
  [pilot.json](supporting/083-body-control/pilot.json), each with its `-kinematics.json`: the three
  calibration pilots.
- [control.json](supporting/083-body-control/control.json),
  [fallback-pilot.json](supporting/083-body-control/fallback-pilot.json),
  [control-500.json](supporting/083-body-control/control-500.json): the gates and verdicts.
- [kinematics.json](supporting/083-body-control/kinematics.json),
  [kinematics-350.json](supporting/083-body-control/kinematics-350.json): the instruments, the half-step
  check and the band readings.
- [per-seed.csv](supporting/083-body-control/per-seed.csv): every control seed's plateau, floor and
  kinematics.
- Reproduce: `scripts/analysis/body_control.py {pilot,control,fallback,kinematics}`, as in the launch
  record. The pilot and campaign directories and the worktree's session records are archived off-repo.
