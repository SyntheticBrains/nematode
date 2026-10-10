## Overview

C.3 grades the body's posture and behaviour on runs that already exist: C.1e's panel (seeds
1801–1864), whose learning arms are the wild type, the chemical-only null and MLP-PPO, and whose
frozen floors are the untrained policies. One control is trained. Every threshold is fixed in the
registration before any reading.

## Decisions

### Decision A: What is read

Each run's final weights are evaluated frozen for **30 held-out episodes**, with posture capture
(every sub-step, 4 Hz) and behaviour capture (every step), at the run's own 20 sub-steps, then at 40
for the half-step check.

| arm | runs | role |
|---|---|---|
| wild type, PPO | 64 | graded |
| chemical-only null, PPO | 64 | graded; does the behaviour depend on the wiring? |
| MLP-PPO | 64 | graded; does it depend on the learner? |
| frozen floors (wild type, null) | 64 + 64 | the bias curves' null |
| MLP, untrained policy | 64 | the MLP's floor (C.1e trained no frozen MLP) |
| **MLP, derivative mode** (new), learning and frozen | 16 + 16 | the weathervane's specificity control, and its floor |

**The control.** Logbook 035 trained an MLP with `chemotaxis_mode: derivative`, which has no spatial
head-sweep, and its weathervane collapsed: the sensing arms' weathervane was sensor-driven. Through
the body the question returns: klinotaxis sensing still uses a synthetic lateral sample, apart from
the body's own head swing. The control is the C.1e MLP config in derivative mode, entropy 0.004,
500 steps, 3,000 episodes, **seeds 1801–1816**, paired by seed with the MLP arm, learning and frozen
(32 runs, about 1.5 hours on 16 workers, from its pilot). **If the control's learning arm does
not beat its frozen floor on the 16 seeds' paired plateaus (all present, 80% interval above zero,
C.1d's control gate), or a seed is missing, it is recorded unreadable**, and the weathervane's specificity is
reported as untested, never inferred.

**Out of scope: Logbook 036's thermotaxis curves.** The body has no thermal cell; that is M.9.

### Decision B: The body checks, which the generator largely sets

Frequency, wavelength and the eigenworm spectrum are largely properties of the body's generator, so
reading them confirms the generator and that the trained drive does not distort it. They are graded
as **body checks**, apart from the behaviour readings.

| check | pass | partial | source |
|---|---|---|---|
| frequency | 0.20–0.45 Hz | 0.10–0.60 Hz | Wormlight checkpoint 1 |
| wavelength | 0.50–0.80 body lengths | 0.40–1.00 | Wormlight |
| variance the first four eigenworms capture | ≥ 85% | ≥ 70% | Wormlight; Stephens et al. 2008 |
| half-step agreement | every instrument within 10% (reversal fraction 10% or 0.01) | | C.1d |

**Eigenworms.** Each sub-step's posture is the body's 12 segment angles, interpolated to 100 tangent
angles head to tail with the mean removed, as the basis expects. It is projected on the WormPose
basis, and the variance captured is pooled over postures. The real postures give the reference: their
first four modes capture 96.46% (Wormlight's in-sample check). **Wormlight's caveat applies**: a body
rocking in place passes the eigenworm clause, so it grades how worm-like the shapes are, not whether
the worm crawls.

**Amplitude** is reported beside, not graded: each posture's radius in the plane of the first two
eigenworms (Stephens et al.'s amplitude; its angle there is the undulation's phase), against the same
statistic on the 6,655 real postures (median 5.07, 5th–95th percentile 2.30–7.87), as distributions.
A posture's peak |κL| was the first choice and is not used: on real postures, differentiating tracked
angles amplifies the tracking noise (a median of 20, against a crawl's ~9). The body's amplitude is
set by its peak curvature and drive, so a mismatch would describe the calibration, not the learner.

### Decision C: The behaviour readings, which are emergent

| reading | pass | partial | source |
|---|---|---|---|
| speed | 0.12–0.30 body lengths/s | 0.06–0.50 | Wormlight |
| episodes with a forward bout of 20 s or more (nearly by construction: an episode is 2,500 worm-seconds and trained reversal fractions are about 1%) | ≥ 80% | ≥ 50% | Wormlight |
| klinokinesis (reorientation rate down/up the gradient) | REPRODUCED / PARTIAL / ABSENT, sign-only | | Logbook 035 |
| weathervane (curving rate vs bearing) | REPRODUCED / PARTIAL / ABSENT, sign-only | | Logbook 035 |

*(Note added 2026-10-10, after the evaluation: "sign-only" was wrong for one statistic. 035's harness
grades the thresholded klinokinesis ratio against a literature range, 1.5–3.0, and the other three on
sign alone. The harness's own verdicts are what Logbook 086 reports; none is re-read.)*

- **Omega turns** are reported, not graded. A head swing is the interval between two consecutive
  zero-crossings of the head segment's curvature. An omega turn is the world-frame angle of the line
  from the midline at 0.2 body lengths to the head changing, net, by more than 135° across one head
  swing (Wormlight; Gray et al. 2005). Reported: their rate per worm-minute and the heading change
  across each. The body may rarely or never make one, since steering acts within a 5 s step and long
  reversals cannot occur; a rate near zero is then a finding about the body, not an instrument
  failure.
- **Omega posture, added after the cost pilot.** The pilot (C.1e's pilot runs, seeds 1701–1702)
  showed the opposite of that expectation: trained arms turn the head line past 135° within one head
  swing 2–4 times per worm-minute, while their postures stay ordinary (the third eigenworm's peak
  across a turn has median 3.3 and never reaches the real postures' 99th percentile, 10.6). The
  heading criterion counts steering pivots as well as omega turns. So each counted turn is also read
  for its posture: the peak |a3| across its swing, against the real postures' 99th percentile of
  |a3|; a real omega's deep bend loads the third eigenworm (Stephens et al. 2008). Reported: the
  share of turns that reach it. Omega turns remain reported, not graded, so no grade is chosen after
  the pilot.
- **Reversals** are reported as before. A reversal lasts one step by construction, so long reversals
  and pirouettes cannot occur, and none is graded.
- **The bias curves** come from Logbook 035's harness on the evaluations' behaviour captures, all 30
  episodes per run (`--tail-runs 30`), with the 1 mm wall margin (H.3), `θ_sharp` fixed at 035's 0.45,
  and its sign-only grading. A step through the body is 5 worm-seconds, so per-step heading changes run
  larger than the point worm's; the harness's threshold-free companions (the turn-magnitude ratio and
  the all-step weathervane slope) are reported beside. **What learning added** is each arm's
  per-seed statistic minus its floor's, paired by seed with an 80% interval: a bias is attributed to
  learning only where the interval lies above zero, and the floors' own verdicts are reported beside.
  The cost pilot's frozen wild-type floor already leaned on two seeds, so a binary "the floor shows
  no bias" rule was replaced by this one before any scored reading.
- **The control's reading**: the weathervane slope of the derivative control against the MLP arm,
  paired by seed. A weathervane that survives without the synthetic sweep is the body's own; one that
  collapses was the sweep's. Reported as an effect size with its interval, as 035 did.

Speed and wavelength repeat C.1d's readings on new runs. Speed sits at the band's floor, and C.1d's
`edge` rule (a reading that differs between 20 and 40 sub-steps) applies.

### Decision D: The data

`data/posture/` vendors both files with their licences, sources, pinned commits and SHA-256s in
`PROVENANCE.md`:

- `EigenWorms.csv` from `iteal/wormpose` (BSD-3-Clause, Copyright 2020 OIST), at the commit Wormlight
  pinned. Its identity with Stephens et al. 2008's basis is inferred, not stated by the file, and the
  provenance says so.
- `shapes.csv` from the OIST Physics of Behavior tutorials v1.0 (CC BY 4.0).

### Decision E: Grading

Each arm gets a grade per check and per reading. Nothing is a contrast between arms: the null and the
MLP are graded alongside the wild type, and differences between them are described, not tested. The
derivative control is the one paired comparison, reported as an effect size. There is no
`unreadable` state; an instrument that cannot be read on an arm, for want of undulating steps or
forward bouts, is recorded as such.

## Risks

- **The body checks will probably pass by construction.** That is the point of separating them.
- **The weathervane may be the synthetic sweep's.** The control is there to say so. If it is, the
  body's turning carries klinokinesis at most, and the logbook says that plainly.
- **The control forages worse than the MLP through the body.** Its pilot (seeds 1701–1704) plateaued
  at 0–25%, against C.1e's MLP at 70.8%. A weathervane that collapses in the control therefore cannot
  be told apart from weaker foraging; only one that survives is clean. The registration fixes that
  asymmetry before the control runs.
- **Cost.** The control's training is 32 runs, about 1.5 hours on 16 workers, near the review
  threshold, so it launches only after the spec review. Evaluation is 416 runs × 30 episodes, with a
  second pass at 40 sub-steps for the 192 graded runs; its cost is measured on a pilot of a few runs
  before the registration.
