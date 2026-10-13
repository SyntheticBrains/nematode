# 093: Through the Body, the Trained Connectome Weathervanes Toward Its Thermal Setpoint but Shows No Klinokinesis (Phase 8b C.3 thermotaxis)

**Status**: completed. C.3's open thermotaxis reading is graded, on M.9's trained bodies.

| arm, seeds 2301–2332 | klinotaxis (weathervane) | klinokinesis |
|---|---|---|
| wild type, trained | **PRESENT**: slope +0.0119 [+0.0111, +0.0129] | ABSENT: ratio 0.935 |
| gap-only null, trained | **PRESENT**: slope +0.0067 [+0.0057, +0.0077] | ABSENT: ratio 0.950 |
| wild type, untrained floor | ABSENT: slope −0.0065 | ABSENT: ratio 0.948 |
| gap-only null, untrained floor | ABSENT: slope −0.0072 | ABSENT: ratio 0.960 |

The table gives the thresholded statistics, with 80% intervals. The threshold-free statistics agree
in every arm.

**Beside it, never read as verdicts:**

- **Learning builds the weathervane.** Each trained arm beats its untrained floor on both weathervane
  statistics, p < 10⁻⁸. Neither learns klinokinesis. On the threshold-free klinokinesis statistic the
  trained wild type moves slightly *against* the real worm's direction (−0.020 against its floor).
- **The wild type weathervanes more strongly than the gap-only null:** +0.0052 [+0.0038, +0.0067] on
  the thresholded slope (p = 4 × 10⁻⁵), and +0.0117 on all steps (p = 0.0014). This fits its faster
  learning in Logbook 092, but nothing here shows that it causes it.

This is the same split Logbook 036 found on the point worm: the weathervane reproduced, klinokinesis
absent. It also matches C.3's chemotaxis through the body, where the weathervane was learned and the
wild type's klinokinesis was not.

**Date**: 2026-10-14.

**OpenSpec change**: `add-thermal-body-validation`. The arms, the grading and the floor comparison
were fixed in its design before any evaluation ran.

## Objective

C.3 (Logbook 086) validated chemotaxis through the kinematic body and left thermotaxis open: "Not
established: anything for thermotaxis. Logbook 036's curves are out of scope (M.9)." M.9 (Logbook 092)
then trained 32 wild-type and 32 gap-only-null bodies on the thermal cell, but its design never
graded their thermotaxis.

**This logbook asks whether the trained body worms thermotax with the real worm's two strategies:**

- **klinokinesis:** turn more often when moving away from the setpoint;
- **weathervaning:** curve toward it.

## Method

**The runs.** M.9's trained weights (seeds 2301–2332), each run's `final.pt`, with weights frozen. Each
seed's untrained policy serves as its floor, as in C.3. Every run plays **30 held-out episodes**, seeded
away from training's arenas, on M.9's cell:

- target 20, 500 steps;
- the kinematic body;
- the synthetic head-sweep.

That is 128 evaluations, which took 3 minutes in parallel.

**The capture.** The evaluation harness now takes the capture modality. With `thermotaxis` it
records:

- the drive, the setpoint error `−|T − Tc|`, with `Tc` = 20 °C;
- the drive's one-step change;
- the direction toward comfort.

These are Logbook 036's definitions. A test checks that every captured drive equals `−|T − Tc|` at
the head.

**The grading** is Logbook 036's harness with C.3's thresholds (`theta_sharp` 0.45, a 1.0 mm wall
margin), against `data/thermotaxis/behavioural_bias_signatures.json`. That reference set is sign-only,
from Luo et al. 2014, Ryu & Samuel 2002 and Clark et al. 2007.

- **Statistics:** four in all. Klinokinesis is read as the turn ratio while moving away from comfort
  against toward it, thresholded and threshold-free. The weathervane is read as the slope of curving
  rate against bearing to comfort, thresholded and on all steps.
- **Labels:** each statistic is REPRODUCED, PARTIAL or ABSENT, and each strategy gets a combined
  label.
- **Floor comparison:** each trained arm minus its floor, paired by seed and oriented to the
  reference's sign. Learning added a bias where the 80% interval lies above zero.
- **Beside:** the wild type minus the gap-only null, with two-sided p.

**This cell is not Logbook 036's assay.**

- **Geometry.** The field is linear, 20 °C at the plate's centre and 1.25 °C/mm along x. The comfort
  band (within 5 °C of 20) is 8 mm wide through the middle, and the danger zones lie beyond 8 mm from
  the centre. Logbook 036 graded migration up a 16 → 20 °C gradient.
- **Task.** The worm also forages, and its thermal pressure is mostly avoiding the danger zones.
- **No specificity control.** There is no thermal derivative-sensing control through the body, which
  would need training. C.3 found that the food weathervane survives without the head-sweep.

## Results

### Grades per arm

| arm | weathervane, thresholded | weathervane, all steps | klinokinesis ratio | klinokinesis, threshold-free | transitions kept from the walls |
|---|---|---|---|---|---|
| wild type, trained | +0.0119 REPRODUCED | +0.0434 REPRODUCED | 0.935 ABSENT | 0.943 ABSENT | 90% |
| gap-only null, trained | +0.0067 REPRODUCED | +0.0317 REPRODUCED | 0.950 ABSENT | 0.957 ABSENT | 88% |
| wild type, floor | −0.0065 ABSENT | −0.0142 ABSENT | 0.948 ABSENT | 0.964 ABSENT | 44% |
| gap-only null, floor | −0.0072 ABSENT | −0.0154 ABSENT | 0.960 ABSENT | 0.975 ABSENT | 42% |

**The floors lean away from comfort.** Both untrained floors curve slightly *away* from it, and their
intervals exclude zero. Untrained bodies spend more time near the walls, and only 42–44% of their
transitions survive the wall margin, so their readings are thinner and may carry a geometric bias.
The floor comparison below is measured from those floors.

### What learning added

Trained minus floor, oriented to the reference:

| arm | weathervane | weathervane, all steps | klinokinesis | klinokinesis, threshold-free |
|---|---|---|---|---|
| wild type | **+0.0184** [+0.0171, +0.0197] | **+0.0576** [+0.0548, +0.0605] | −0.012 [−0.031, +0.005] | −0.020 [−0.026, −0.013] |
| gap-only null | **+0.0139** [+0.0122, +0.0157] | **+0.0471** [+0.0425, +0.0520] | −0.010 [−0.029, +0.008] | −0.018 [−0.028, −0.006] |

**Both wirings learn to weathervane toward comfort** (p < 10⁻⁸). **Neither learns klinokinesis**, and
the threshold-free reading moves slightly away from the real worm's direction.

### The two wirings, beside

Wild type minus the gap-only null, oriented to the reference, with two-sided p:

| statistic | difference | 80% CI | p |
|---|---|---|---|
| weathervane | **+0.0052** | [+0.0038, +0.0067] | 4 × 10⁻⁵ |
| weathervane, all steps | **+0.0117** | [+0.0075, +0.0158] | 0.0014 |
| klinokinesis | −0.015 | [−0.025, −0.006] | 0.068 |
| klinokinesis, threshold-free | −0.014 | [−0.021, −0.006] | 0.029 |

## What this establishes, and what it does not

**Established:**

- **Through the body, the trained connectome thermotaxes by weathervaning.** Its curving rate leans
  toward the thermal setpoint, as the real worm's does in sign. Its untrained floor's does not.
- **It does not show klinokinesis.** It does not turn more often when heading away from comfort.

**Described, with that standing:**

- **The wild type weathervanes more strongly than the gap-only null.** In Logbook 092 the wild type
  also learns faster and peaks higher. A stronger steering bias is one candidate for why, but this
  reading shows only that the two go together.
- **The pattern repeats across substrates.** The point worm (036), chemotaxis through the body (C.3)
  and thermotaxis through the body all reproduce the weathervane and not klinokinesis. The reward and
  the learner seem to favour steering over the pirouette strategy, whatever the sense or the body.

**Not established:**

- **Magnitudes against the real worm.** The reference is sign-only.
- **That the weathervane is sensor-specific here.** Without a derivative control it could partly be
  geometry, and the floors' negative slopes show that geometry can bias this reading.
- **Thermotaxis in Logbook 036's assay.** This cell's comfort band and foraging task differ.

## Biological fidelity

- **Real worms use both strategies in thermotaxis.** Luo et al. 2014 show klinokinesis-like run
  modulation and weathervaning. This body reproduces only the second.
- **The left–right signal is synthetic.** The weathervane is read off the head-sweep's lateral
  difference, not the body's own head swing. C.3 showed, for food, that the body's motion carries a
  weathervane without it.
- **The setpoint is the cultivation temperature (20 °C),** as in the real worm. The thermal circuit's
  memory of it is not modelled; `Tc` is fixed.

## Next

- **C.3's thermotaxis reading is graded.** The tracker records it.
- **S8b, Phase 8's synthesis,** comes next.
- **Candidates, none scheduled:**
  - a thermal derivative control through the body, to test whether the weathervane is
    sensor-specific;
  - why no substrate learns klinokinesis.

## Artefacts

- [thermotaxis.json](supporting/093-thermal-body-validation/thermotaxis.json): every arm's four
  statistics, per seed and graded, the wall exclusion, the floor comparisons and the wiring
  comparison.
- Each run's capture is archived off-repo.

**To reproduce:**

```bash
uv run python scripts/analysis/thermal_body.py thermotaxis --logs campaigns/m9-panel/logs \
  --seeds $(seq 2301 2332) --target 20 --out-dir /tmp/m9-thermo --workers 16 --out thermotaxis.json
```
