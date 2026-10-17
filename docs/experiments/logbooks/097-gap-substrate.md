# 097: Log-Compressed Counts, Not Ohmic Coupling, Carry the Corrected Gap Model (Phase 9 G.0-A)

**Status**: completed. **The registered rule chose log-compressed counts.** On both block-V cells, ohmic
gap coupling at its calibrated scale was eligible but learned more slowly than log-compressed counts, by
more than each cell's margin.

G.0 found that the wild type's lead over the gap-only null rides on the count-scaled gap model
([Logbook 095](095-gap-operating-point.md)). G.0-A picks the corrected model that block G will run on.
Ohmic coupling and log-compressed counts were the two candidates, and the pilot judged them on
learnability alone. It never read the gap between the wild type and the null.

| cell | ohmic − log, `auc_success` (80% CI) | margin δ | ohmic shown slower | seeds with ohmic behind |
|---|---|---|---|---|
| thermal, target 35 | −0.058 [−0.087, −0.029] | 0.0239 | **yes** | 14 / 16 |
| hard350 | −0.048 [−0.065, −0.030] | 0.0143 | **yes** | 13 / 16 |

Both cells were readable and both models were eligible on both. The choice is decided by speed alone.

**Date**: 2026-10-18.

**OpenSpec change**: `add-ohmic-gap-coupling`.

**Pre-registration**: [supporting/097-gap-substrate/launch.md](supporting/097-gap-substrate/launch.md),
committed at e01787d0 before any pilot run, with pre-launch review fixes at dd05a82d. The ohmic scale was
calibrated by the maintainer on 2026-10-17, also before any pilot run, after the registered probe stopped
the first launch (below).

## The question

G.0 read `artefact`: under unit gap weights the wild type's lead over the gap-only null drops from
+0.104 to +0.003. The model's additive coupling scales each junction by its synapse count, so a few
high-count pairs dominate. Two corrections were candidates:

- **Ohmic coupling**: current proportional to the potential difference, implemented as
  `(I + L)⁻¹` in the settling map, `L` the gap Laplacian. It averages, so it adds no loop gain at any
  conductance. This is the more physical model.
- **Log-compressed counts**: the existing additive coupling with each count replaced by `log1p(count)`.

**Which corrected model is the better instrument for block G?** The pilot measured how fast each one
learns. It did not measure which one gives the larger wiring effect, so that answer could not steer
the choice.

## Method

**The probe, at no run cost.** On untrained brains, with the 64 fixed inputs of
`sign_scale_calibration.py` and the pilot's 16 seeds, both cells ([probe.json](supporting/097-gap-substrate/probe.json)):

| brain | action spread across inputs | saturated fraction |
|---|---|---|
| additive counts | 0.238 | 0.058 |
| additive log counts | 0.041 | 0.000 |
| gap junctions off | 0.025 | 0.000 |
| ohmic × 1 (the built conductances) | 0.0003 | 0.000 |
| ohmic × 0.1 | 0.0078 | 0.000 |
| **ohmic × 0.03** | **0.0162** | 0.000 |
| ohmic × 0.01 | 0.0213 | 0.000 |

(Thermal cell shown; hard350 agrees within 0.001.)

The registered stop rule required ohmic to keep a quarter of the additive model's spread. **It stopped
the first launch.** At the built conductances, ohmic coupling averages the input away.

The probe also showed that the yardstick itself was the artefact. The log-count and gap-off brains
learn as well as the count model, yet both fall far below a quarter of its spread. The count model's
large spread is its gap amplification.

The maintainer amended the rule before any pilot run. The reference became the gap-off brain, a known
learner, and the scale became the largest on the grid that keeps at least half its spread on both cells.
**That scale is ×0.03.** At it the median neuron keeps 93% of its own drive. ALA, the most strongly
coupled neuron, keeps 29%.

**The pilot.** Both block-V cells (thermal at target 35, and hard350) on Emmons 2024, at 3,000 episodes,
on seeds 9871–9886. Each cell had 11 arms:

- each model's wild type and gap-only null, learning and frozen;
- the additive-count wild type, learning, for description;
- the MLP-PPO control, learning and frozen.

That makes 352 runs, all of which exited 0. They took 4.9 hours at 16 workers, against the 6 hours
estimated.

**The rule**, fixed before the pilot:

1. The cells are readable if the MLP beats its frozen floor on both.
2. A model is eligible on a cell if its wild type and its null each beat their floors and do not both
   plateau at or above 90%.
3. Ohmic is chosen if eligible on both cells, unless on either cell it is **shown slower** than log.
   That means the paired 80% interval of `auc_success(ohmic) − auc_success(log)` on the wild type lies
   entirely below −δ.
4. Otherwise log is chosen if eligible on both cells, otherwise neither.

The registered harness `gap_substrate_selection.py select` was run from the pilot's worktree at
dd05a82d. Its output is [selection.json](supporting/097-gap-substrate/selection.json).

## Results

**Readability.** The MLP control beat its frozen floor on both cells. Its plateaus were 71.2% on the
thermal cell and 95.4% on hard350. The saturation bar is part of eligibility, not readability, and is
not applied to the MLP.

**Eligibility: gate evidence only.** Plateaus are in percent, and every frozen floor is 0.0%:

| cell | model | wild type plateau | null plateau | saturated | eligible |
|---|---|---|---|---|---|
| thermal | log | 70.0 | 69.2 | no | ✓ |
| thermal | ohmic | 61.1 | 54.1 | no | ✓ |
| hard350 | log | 79.2 | 77.0 | no | ✓ |
| hard350 | ohmic | 75.2 | 75.4 | no | ✓ |

These are the readings the eligibility rule uses. The difference between the wild type and the null in
each row is not a registered reading. It is the contrast the next panel exists to measure, on 64 seeds
and with its own minimum, so it is not interpreted here.

**Speed, the deciding reading.** Ohmic was shown slower than log on both cells, as in the table at the
top: by 0.058 on the thermal cell, where the whole interval lies below −0.0239, and by 0.048 on hard350,
below −0.0143. The registered operating characteristic gave ohmic about a one-in-three chance of being
chosen if it were slower by 0.05. **The choice is log-compressed counts.**

**Beside it, as description.** Each corrected wild type against the additive-count wild type, on
`auc_success`:

| cell | log − additive | ohmic − additive |
|---|---|---|
| thermal | −0.009 [−0.043, +0.027] | −0.067 [−0.108, −0.025] |
| hard350 | +0.016 [−0.004, +0.037] | −0.032 [−0.046, −0.017] |

Log compression costs the learner nothing measurable on either cell. Ohmic coupling at ×0.03 costs it
on both cells.

## Interpretation

Ohmic coupling is the more physical model, but here it is the slower learner. At the calibrated scale it
still damps the action spread to about two-thirds of the gap-off brain's. The averaging that removes the
count artefact also smooths away some of the sensory contrast the policy reads. The calibration took the
largest scale that kept half the gap-off spread; smaller scales would keep more, but by then the gap
junctions would barely couple at all. In this settling map, therefore, ohmic coupling trades learnability
for physical fidelity. Log compression keeps every junction coupling additively, removes most of the
dominance of the high-count pairs, and learns as well as the count model.

This reading does not say that ohmic coupling is the wrong model of electrical synapses. It says only
that, inside a single-step settling map trained by PPO for 3,000 episodes, it is the worse instrument.
Logbook 080 reached the same order on the leaky dynamics, which already couple ohmically (thermal 20%
against 64%). The more physical models of gap junctions keep losing to the additive ones on learnability.
That is worth carrying into any future dynamics rung.

## What follows

- **The corrected model is `gap_weight_mode: log_count`**, with additive coupling. The ohmic option
  stays in the brain, refused outside the settling map, for any later reading that needs it.
- **Next: the point-worm contrast on log-compressed counts.** It runs in its own change, on seeds
  3901–3964, with a minimum of 0.0577 (Logbook 082's), on Emmons 2024. Its D24 table and shortcut audit
  are restated for the log-count null. No new D24 definitions are needed, because the coupling is still
  additive.
- **092's body contrast** is re-run on log counts after M.1 reads out (the 2026-10-17 sequencing
  amendment).
- G.0(e)'s 16-seed log-count level (+0.012 [−0.023, +0.046], description) is not a substitute for the
  64-seed contrast. It predicts a small or absent wiring effect, which the STOP branch then decides.

## Data

- [supporting/097-gap-substrate/launch.md](supporting/097-gap-substrate/launch.md) — the registration.
- [supporting/097-gap-substrate/probe.json](supporting/097-gap-substrate/probe.json) — the untrained
  probe and the calibration grid.
- [supporting/097-gap-substrate/selection.json](supporting/097-gap-substrate/selection.json) — the
  selection, with each arm's per-seed plateau and floor and each paired difference.
- Raw logs of the smoke and pilot campaigns: archived off-repo.
