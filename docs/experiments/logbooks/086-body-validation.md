# 086: Through the Body, the Crawl Passes Its Body Checks and a Weathervane Survives Without the Head-Sweep; No Turn Has an Omega's Posture (Phase 8b C.3)

**Status**: completed — **registered grading, every arm readable.** On C.1e's trained runs, through the
kinematic body:

| reading | wild type | chemical-only null | MLP-PPO |
|---|---|---|---|
| frequency, Hz | 0.300 **pass** | 0.300 **pass** | 0.300 **pass** |
| wavelength, body lengths | 0.650 **pass** | 0.650 **pass** | 0.650 **pass** |
| variance in four eigenworms | 98.1% **pass** | 98.1% **pass** | 98.0% **pass** |
| half-step agreement | **agrees** | **agrees** | **agrees** |
| speed, body lengths/s | 0.113 **partial** | 0.113 **partial** | 0.116 **partial** |
| episodes with a ≥ 20 s forward bout | 100% **pass** | 100% **pass** | 100% **pass** |
| weathervane (035) | **PRESENT**, learned | **PRESENT**, learned | **PRESENT**, learned |
| klinokinesis (035) | PRESENT_PARTIAL, not learned | PRESENT_PARTIAL, magnitude learned | PRESENT_PARTIAL, magnitude learned |

"Learned" means the arm's per-seed statistic minus its floor's has an 80% interval above zero.

**The control is readable, and its weathervane survives.** Without the synthetic head-sweep, the MLP still
curves toward the gradient (PRESENT, +0.015 thresholded, learned over its floor). By the registration
that is clean evidence **the body's own motion carries a weathervane**. The sweep adds to it: the MLP
minus the control is +0.027 [+0.016, +0.040].

**No counted turn has an omega's posture.** The trained arms turn the head line past 135° within one
head swing 2–5 times per worm-minute. Of about 807,000 such turns across all seven arms, none reaches
the real postures' deep-bend tail. They are steering pivots.

**The turn-rate bias is far weaker than a real worm's.** Every arm turns more often heading
down-gradient than up it, but the ratio is 1.03–1.24, against the literature's 1.5–3.0.

**Date**: 2026-10-10.

**OpenSpec change**: `add-body-validation`.

**Pre-registration**: [supporting/086-body-validation/launch.md](supporting/086-body-validation/launch.md),
committed at 3fec9b4f before the control's training and before any scored evaluation, after the spec
review, the gate preflight and two pilots.

## Objective

C.3 asks whether what a trained worm does through the body looks like *C. elegans*, at the level of
posture and of behaviour, graded against thresholds fixed in advance. The runs are C.1e's
([Logbook 085](085-body-wiring.md)); one control is new.

## Method

**The runs.** C.1e's panel at the 500-step hard350 cell through the frozen kinematic body (steering
gain 2, one-step reversals, wave floor 0.25): the wild type and the chemical-only null under PPO, their
frozen runs, and MLP-PPO, seeds 1801–1864. The MLP's floor is each seed's untrained policy, since C.1e
trained no frozen MLP.

**The control.** C.1e's MLP config with `chemotaxis_mode: derivative`, nothing else changed: no
synthetic lateral sample. It was trained learning and frozen, seeds 1801–1816, 3,000 episodes, from a
worktree at the registration commit. All 32 runs succeeded in 1 h 53 min, against the registration's
1.5 h estimate: at 16 workers a run took about 60 minutes, against 38 at the pilot's 8.

**The evaluation.** Each run's final weights were evaluated frozen for 30 held-out episodes (run
indices from 1,000,000), with posture captured at every sub-step (4 Hz) and behaviour at every step.
The graded arms were evaluated again at 40 sub-steps. 416 runs took 31.5 minutes on 16 workers. A
second, identical run, adding the per-run CSV, reproduced the JSON byte for byte.

**The instruments.**

| instrument | how |
|---|---|
| frequency | mid-body curvature crossings, each counted after leaving ±0.31 κL |
| wavelength | adjacent-segment delays summed down the body |
| speed | head displacement, steps within 1 mm of a wall excluded |
| eigenworm variance | each posture's 12 segment angles interpolated to 100 mean-removed tangent angles, projected on the vendored WormPose basis (Stephens et al. 2008), pooled |
| amplitude | each posture's radius in the first two eigenworms' plane |
| omega turns | the head line, 0.2 body lengths to the head, turning more than 135° net between consecutive zero-crossings of the head segment's curvature; each also read for the third eigenworm's peak across the swing |
| bias curves | Logbook 035's harness on the behaviour captures, 1 mm wall margin, `θ_sharp` 0.45, both slope families |

## Results

### Body checks

Every graded arm passes frequency, wavelength and the eigenworm clause, and agrees with itself at 40
sub-steps. No reading sits on a band's edge.

| arm | frequency 20 / 40 sub-steps | wavelength 20 / 40 | speed 20 / 40 | reversal fraction 20 / 40 |
|---|---|---|---|---|
| wild type | 0.300 / 0.300 | 0.650 / 0.650 | 0.113 / 0.109 | 0.004 / 0.004 |
| chemical-only null | 0.300 / 0.300 | 0.650 / 0.650 | 0.113 / 0.109 | 0.008 / 0.007 |
| MLP-PPO | 0.300 / 0.300 | 0.650 / 0.651 | 0.116 / 0.110 | 0.014 / 0.014 |

The frequency and wavelength are the generator's (0.30 Hz, 0.65 body lengths), as the design expected.
The eigenworms capture 98.0–98.1% of the body's posture variance, more than the real postures' 96.5%: a
travelling sine on 12 segments is simpler than a real worm's shape. A body rocking in place would pass
this clause too, so it grades the shapes, not the crawl.

### Behaviour readings

**Speed** is partial on every arm, 0.113–0.116 body lengths/s against a pass band from 0.12. It is
partial at 40 sub-steps too, so it is not on an edge. Logbook 083 found the body's speed at the band's
floor; trained policies do not move it.

**Forward bouts** pass on every arm, every episode, nearly by construction, as registered: an episode is
2,500 worm-seconds and trained reversal fractions are 0.4–1.4%.

**The bias curves:**

| arm | klinokinesis, thresholded | klinokinesis, threshold-free | weathervane, thresholded | weathervane, all steps |
|---|---|---|---|---|
| wild type | 1.031 [1.018, 1.042] PARTIAL | 1.018 [1.011, 1.025] REPRODUCED | +0.046 [+0.044, +0.047] REPRODUCED | +0.161 [+0.154, +0.168] REPRODUCED |
| chemical-only null | 1.059 [1.046, 1.071] PARTIAL | 1.039 [1.031, 1.047] REPRODUCED | +0.045 [+0.043, +0.047] REPRODUCED | +0.157 [+0.150, +0.164] REPRODUCED |
| MLP-PPO | 1.238 [1.151, 1.336] PARTIAL | 1.118 [1.075, 1.160] REPRODUCED | +0.049 [+0.045, +0.053] REPRODUCED | +0.228 [+0.214, +0.243] REPRODUCED |
| derivative control | 1.407 [1.210, 1.611] REPRODUCED | 1.226 [1.123, 1.335] REPRODUCED | +0.015 [+0.011, +0.019] REPRODUCED | +0.032 [+0.026, +0.039] REPRODUCED |

Klinokinesis is a ratio with null 1; the weathervane is a slope with null 0.

**Why the trained arms' thresholded klinokinesis reads PARTIAL.** It is the one statistic 035's harness
grades against a magnitude: the down/up turn-rate ratio's literature range, 1.5–3.0 (a real worm turns
about twice as often heading down-gradient; Pierce-Shimomura et al. 1999). The trained arms' ratios are
significant in the right direction but sit well below that range. The other three statistics are graded
on sign alone. The launch record called all four "sign-only"; that described this statistic wrongly. The
verdicts reported here are the harness's own, and none is re-read.

The derivative control's thresholded ratio, 1.41 [1.21, 1.61], overlaps the range and reads REPRODUCED.
Without the spatial sweep it leans harder on turning, as on the point worm (Logbook 035).

### What learning added: each arm against its floor, paired by seed

| arm − floor | klinokinesis | klinokinesis, threshold-free | weathervane | weathervane, all steps |
|---|---|---|---|---|
| wild type − frozen (64) | −0.009 [−0.033, +0.015] | +0.003 [−0.008, +0.014] | **+0.041** [+0.039, +0.043] | **+0.151** [+0.144, +0.158] |
| null − frozen (64) | +0.026 [−0.012, +0.061] | **+0.028** [+0.011, +0.043] | **+0.041** [+0.039, +0.043] | **+0.148** [+0.141, +0.155] |
| MLP − untrained (64) | −0.483 [−0.829, −0.140] | **+0.077** [+0.014, +0.139] | **+0.045** [+0.040, +0.051] | **+0.225** [+0.208, +0.243] |
| control − frozen (16) | −0.842 [−1.926, +0.181] | **+0.209** [+0.047, +0.371] | **+0.014** [+0.011, +0.018] | **+0.031** [+0.026, +0.037] |

Bold: the interval is above zero.

- **The weathervane is learned in every arm**, with p ≤ 0.002 on every pair.
- **Klinokinesis barely is.**
  - The wild type's turn-rate bias does not differ from its frozen runs' on either statistic.
  - The null and the MLP add a small threshold-free bias.
  - On the thresholded ratio, the MLP's untrained policy (1.72) and the control's floor (2.25) read
    higher than their trained arms. The floors keep only 25–40% of their transitions after the wall
    margin (the trained arms keep 75–81%), and their ratios are noisy. That is why the MLP's difference
    is negative.

**The floors' own verdicts**, reported beside as registered:

- The connectome's frozen runs read a weathervane PRESENT, with slopes of +0.004 and +0.010 against the
  trained arms' +0.046 and +0.161.
- The MLP's untrained policy and the control's floor read PARTIAL.

The paired rule, adopted after the cost pilot showed exactly this lean, attributes to learning only what
the trained arms add over their floors.

### The control

**Gate**: readable.

- The learning arm beats its frozen floor on all 16 seeds' paired plateaus: mean 10.2% against 0.0%, with
  the interval's lower bound at +7.3.
- No seed is competent; six plateau at 0%. C.1d's gate reads `fallback`, and the registered gate is the
  floor alone.

**The reading.** The MLP's weathervane minus the control's, paired by seed over 1801–1816:

| slope | MLP − control | one-sided Wilcoxon p |
|---|---|---|
| thresholded (deciding) | **+0.027** [+0.016, +0.040] | 0.003 |
| all steps | +0.239 [+0.200, +0.279] | < 0.001 |

The interval lies above zero and the control shows a weathervane (PRESENT, both slopes REPRODUCED).
The registered outcome is **"the body carries a weathervane and the sweep strengthens it."**

- **The first half is clean.** A weathervane surviving in the control is the outcome the registration
  fixed as unaffected by the control's weaker foraging. The control's slope is learned over its own
  floor (+0.014 [+0.011, +0.018]), and its floor reads only PARTIAL.
- **The second half carries the registered caveat.** The sweep's added slope cannot be told apart from
  the control foraging worse: 10% against the MLP's 70.8%.

**Beside it, Logbook 035's point worm**, for scale and not as a delta:

- Removing the sweep there cut the thresholded slope from +0.027 to +0.002, a geometric residual
  that fell to PARTIAL once walls were excluded.
- Through the body, the control's slope is +0.015, about seven times the point worm's residual, and
  REPRODUCED with walls excluded.
- What survives without the sweep is larger through the body than on the point worm.

### Omega turns and amplitude, reported

| arm | head-line turns > 135° per worm-minute | median heading change | with an omega's posture |
|---|---|---|---|
| wild type | 2.14 | 149° | 0 of 151,859 |
| chemical-only null | 2.00 | 149° | 0 of 143,383 |
| MLP-PPO | 5.27 | 154° | 0 of 346,860 |
| derivative control | 4.12 | 150° | 0 of 81,134 |
| floors (wild type, null, MLP untrained, control) | 0.29, 0.31, 0.32, 0.94 | 143–145° | 0 |

An omega's deep bend loads the third eigenworm. The real postures' 99th percentile of its magnitude is
10.6. On one pilot wild-type run, the 99th percentile of the body's own postures was 6.16, and no counted
turn in any arm reaches 10.6. **The body makes no omega posture.** What the Wormlight heading
criterion counts here is steering: the policy turns the head line through about 150° within one half
period (1.75 s). Trained arms do this 2–5 times per worm-minute, their untrained floors about 0.3 times.

**Amplitude** in the first two eigenworms' plane:

| | median | 5th–95th percentile |
|---|---|---|
| trained arms | 5.75–6.33 | 3.67–8.86 |
| floors | 5.63–5.69 | 3.99–7.60 |
| real postures | 5.07 | 2.30–7.87 |

The trained body bends a little more deeply than a real worm, and rarely as shallowly. The amplitude is
set by the body's peak curvature and drive, so this describes the calibration.

## What this establishes, and what it does not

- **Established**:
  - Through this body, every trained arm undulates at the generator's frequency and wavelength.
  - Every trained arm holds worm-like shapes on the pinned eigenworm basis, and its readings converge
    at half the time step.
  - Its speed reads partial, at 0.11–0.12 body lengths/s.
- **Established**: every trained arm, connectome or MLP, learns a weathervane through the body, well
  above its floor.
- **Established**: without the synthetic head-sweep, the MLP still learns a weathervane through the body.
  **The body's own motion carries one.** On the point worm, the same control left only a geometric
  residual.
- **Established, as a finding about the body**: it makes no omega postures. Its sharp reorientations
  are steering pivots of normal shape, at 2–5 per worm-minute when trained.
- **Not established**: how much of the MLP's weathervane the head-sweep supplies. The +0.027 difference
  cannot be told apart from the control's weaker foraging, as registered.
- **Not established**: a worm-sized klinokinesis. Every arm's turn-rate ratio (1.03–1.24) sits below the
  literature's 1.5–3.0. The wild-type connectome's bias equals its frozen runs'; the null and the MLP add
  a small threshold-free one.
- **Not established**: anything about the wiring. The arms are graded side by side, and differences
  between them are described, not tested. The wild type and the null read alike on every body check
  and curve.
- **Not established**: anything for thermotaxis. Logbook 036's curves are out of scope (M.9).

## Biological fidelity

- **The body checks pass by construction.** The frequency and wavelength are the generator's settings,
  inside Fang-Yen et al. 2010's crawling ranges. The eigenworm clause is passed more easily by a sine
  than by a worm.
- **Speed sits just below the crawling band** (0.12–0.30 body lengths/s, Ramot et al. 2008 via
  Wormlight). It is a property of the body's drag and wave, not of the learner.
- **Reorientation through this body is steering, not omega turns or pirouettes.**
  - A real worm's sharp turns are omega turns and reversal–omega pirouettes (Gray et al. 2005); here
    long reversals cannot occur and no posture approaches an omega's.
  - So klinokinesis, which the point worm read off sharp turns, has only pivots to act through, and the
    weak, partly unlearned klinokinesis is unsurprising.
  - D.1 and any later body need an omega-capable posture if klinokinesis is to be read as the worm
    does it.
- **The weathervane is the most worm-like behaviour here.** Real weathervaning is a gradual curve
  driven by the head's lateral sampling during undulation (Iino & Yoshida 2009). That a weathervane
  survives the removal of the synthetic lateral sample is consistent with the body's motion supplying
  some of that sampling. Which part of the swing a 5 s sensing step samples is not analysed here.
- **The derivative control forages poorly through the body** (10% plateau). On the point worm it
  reached 98.7–100%. Through this body the spatial sweep does most of the work of foraging.

## Next

- **C.3 closes on these readings.** The tracker item is met.
- **For D.1's patchy lawns and any later body**, two conditions carry forward:
  - an omega-capable posture, if reorientation is to be read as the worm does it;
  - the paired floor rule, wherever an untrained policy can lean.
- **The weathervane's mechanism through the body**, meaning which part of the head's swing the sensor
  samples and whether a shorter sensing step strengthens it, is a candidate question, not registered
  here.
- **Next in 8b is B.3 + D.1**, then S8b.

## Artefacts

- [launch.md](supporting/086-body-validation/launch.md): the registration.
- Pilots, which are calibrations, not C.3 readings:
  - [evaluation-pilot.json](supporting/086-body-validation/evaluation-pilot.json): the evaluation
    cost pilot on C.1e's pilot runs;
  - [control-preflight.json](supporting/086-body-validation/control-preflight.json): the control's
    gate preflight.
- [validation.json](supporting/086-body-validation/validation.json): every arm's readings, grades,
  half-step check, bias curves, the floor comparison and the control.
- [per-run.csv](supporting/086-body-validation/per-run.csv): one row per evaluated run.
- Reproduce: `scripts/analysis/body_validation.py --logs campaigns/c1e-panel/logs --logs campaigns/c3-control/logs --out-dir <dir> --out validation.json --csv per-run.csv`.
  The control's campaign directory, the evaluation captures and the worktree's session records are
  archived off-repo.
