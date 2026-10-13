## Overview

Two follow-ups to M.9, both read from runs that already exist. Decided with the user on 2026-10-14,
before the S8b synthesis.

## Decisions

### Decision A: What is evaluated

- **Weights:** the trained wild type and gap-only null from M.9's panel, seeds 2301–2332, each
  run's `final.pt`, resolved from its log as C.3 does (`body_control.final_weights`).
- **Floors:** each seed's untrained policy (`weights = None`), as C.3's floors are.
- **Episodes:** **30 held-out episodes per run**, seeded away from training's arenas, with weights
  frozen. These are C.3's settings.
- **The cell:** M.9's, unchanged: target 20, 500 steps, the kinematic body, the synthetic head-sweep.
- **Runs:** 4 arms × 32 seeds = 128 evaluations, in parallel. About an hour.

### Decision B: What is captured and how it is graded, fixed before any reading

- **Capture:** `capture_behaviour_modality: thermotaxis`. The drive is `−|T − Tc|` with
  `Tc` = 20 °C, its one-step change, and the direction toward comfort. These are Logbook 036's
  definitions.
- **Grading:** Logbook 036's four statistics, with the same harness. It is the same as C.3's except
  for the modality.
  - **Statistics:** klinokinesis thresholded and threshold-free; weathervane thresholded and on all
    steps.
  - **Labels:** each statistic is graded against `data/thermotaxis/behavioural_bias_signatures.json`
    (sign-only) as REPRODUCED, PARTIAL or ABSENT. Each strategy gets a combined label.
  - **Thresholds:** `theta_sharp` 0.45 and a 1.0 mm wall margin, C.3's.
- **Floor comparison,** C.3's rule: each trained arm's statistic minus its floor's, paired by seed and
  oriented so that positive is the reference's direction. Learning added a bias where the 80%
  interval lies above zero.
- **Beside, described only:** the wild type minus the gap-only null on each statistic, paired by
  seed, with a two-sided p from `operating_point_surface.two_sided`.

**What "met" means.** The tracker item is C.3's open thermotaxis reading. It is met when both arms
are graded and the labels and floor comparisons are reported, whatever they show, as C.3 was. No
outcome is a verdict on M.9's wiring reading.

### Decision C: This cell is not Logbook 036's assay

The logbook states these differences, so the grade is not read as 036's:

- **Geometry.** The field is linear across the plate, 20 °C at the centre and 1.25 °C/mm. The
  comfort band (|T − Tc| ≤ 5 °C) is 8 mm wide through the middle, and the danger zones lie beyond
  8 mm from the centre. 036 graded migration up a 16 → 20 °C gradient.
- **Task.** The worm also forages, and thermal pressure is mostly avoiding the danger zones. Much of
  the time the worm sits where the drive changes little.
- **No specificity control.** 036 and C.3 each had a derivative-sensing control that needs no head
  sweep. A thermal one through the body would need training and is out of scope. C.3 found the food
  weathervane survives without the sweep.

### Decision D: The speed–stability addendum to Logbook 092

Description from the panel's committed logs, no verdict. These readings were first computed on
2026-10-14 while planning this change; they are recorded here, not fixed in advance. The registered reading stays as read. The
readings are the wild type minus the gap-only null, paired by seed, with two-sided p:

- `auc_success` over the first 3,000 episodes, before any collapse;
- the best 500-episode block's success;
- episodes to that block;
- `auc_success` on the seeds where neither learner collapsed, and over their first 3,000 episodes;
- each collapse's timing.

The addendum corrects 092's "part speed, part stability" to what the split shows. The JSON is
committed beside 092.

## Risks

- **The thermal drive may barely vary.** If the worm stays in the comfort band, the per-seed
  statistics may be noisy or not finite. The harness drops non-finite values and reports how many it
  dropped.
- **Wall exclusion.** C.3 kept 75–81% of transitions in trained arms but only 25–40% in floors. The
  floors' readings are thinner, and the logbook reports the fractions kept.
