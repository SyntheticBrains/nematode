# 099: On Log-Compressed Gap Counts the Wild Type Learns No Faster Than Its Gap-Only Null (Phase 9 G.0-A)

**Status**: completed. **`no_wiring_effect`.** On the corrected gap model, the wild type's lead over the
gap-only null is +0.003 `auc_success`, with an 80% interval of [−0.009, +0.016]. The interval includes
zero and lies well inside the ±0.0577 minimum.

| reading | value (80% CI) | q | state | verdict |
|---|---|---|---|---|
| `gap(log)`, `auc_success`, wild type − null, 64 seeds | **+0.0032** [−0.0093, +0.0158] | 0.43 | `no_move` | **`no_wiring_effect`** |

Both wirings learn to the same level: plateaus of 70.5% for the wild type and 70.9% for the null. Every
seed beats its frozen floor of 0.0% on both wirings, and neither arm is near the 90% bar.

The gap-placement lead the project carried from Logbook 082 to 092 does not exist on the point worm
under log-compressed counts. That is the point-worm half of the phase's STOP condition. **G.1–G.3b do not
register on the point worm.** Whether the STOP branch is taken depends on 092's body contrast, re-run on
log counts.

**Date**: 2026-10-19.

**OpenSpec change**: `add-logcount-gap-contrast`.

**Pre-registration**: [supporting/099-logcount-contrast/launch.md](supporting/099-logcount-contrast/launch.md),
committed at 96d1d9bf before any panel run. Its verdict map names both cells' joint outcomes.

## The question

G.0 found that the wild type's lead over the gap-only null on the thermal cell rode on count-scaled gap
coupling ([Logbook 095](095-gap-operating-point.md)). G.0-A chose log-compressed counts as the corrected
model, on learnability alone ([Logbook 097](097-gap-substrate.md)). **On the corrected model, does the
wild type still learn faster than its gap-only null?**

## Method

The thermal cell at target 35 on Emmons 2024, at block V's point, with `gap_weight_mode: log_count` and
3,000 episodes. The arms are the wild type and the gap-only null, each learning and frozen, on seeds
3901–3964. Those seeds are disjoint from every committed band on this cell's gap contrasts. The configs
are G.0-A's four, unchanged. All 256 runs exited 0.

The run took 3.6 hours at 16 workers, including some sharing of the machine with test runs, against the
3.7 estimated.

The registered harness `logcount_contrast.py score` was run from the panel's worktree at 96d1d9bf. It
reads `gap(log)` on `auc_success` with an 80% bootstrap interval and the two-sided Wilcoxon as q (a
family of one), and classifies it by `mc.classify` at m = 0.0577. Its output is
[panel.json](supporting/099-logcount-contrast/panel.json), with the per-seed values in
[per-seed.csv](supporting/099-logcount-contrast/per-seed.csv).

## Results

**The reading.** `no_move`: the 80% interval [−0.0093, +0.0158] includes zero and lies inside ±0.0577,
and q = 0.43. The verdict is **`no_wiring_effect`**. The wild type leads on 35 of 64 seeds.

**Gates.** Readable. Each wiring beats its frozen floor on all 64 seeds, with a learning − frozen plateau
of +70.5 points [68.6, 72.1] for the wild type and +70.9 [69.4, 72.4] for the null. The highest plateau is
19 points under the bar.

**Sensitivity achieved.** The per-seed spread is σ = 0.079, tighter than the 0.115 the registration was
sized at. The detectable effect for the two-sided test at 80% power is therefore about 0.028. A wild-type
lead of half the minimum would have been seen.

**Beside, never verdicts:**

- **Episodes to 30% success.** The wild type reaches 30% success 29 episodes sooner than the null, with an
  80% interval of [−21, +76] and q = 0.12. The interval spans zero.

- **Against G.0's levels on Emmons**, all read on `auc_success` on this cell:

  | level | lead | 80% CI | seeds |
  |---|---|---|---|
  | count weights | +0.104 | [0.087, 0.123] | 64 |
  | unit weights | +0.003 | [−0.012, +0.018] | 64 |
  | log counts, G.0 | +0.012 | [−0.023, +0.046] | 16 |
  | **log counts, here** | **+0.003** | [−0.009, +0.016] | 64 |

  Both corrections, removing the counts and compressing them, leave the same near-zero gap.

- **One wild-type seed barely learned.** On seed 3903 the wild type's plateau is 4.0% against the null's
  81.2%, the panel's largest gap at −0.35. Without it the mean is +0.009, still inside the minimum. The
  seed stays in the reading, as registered.

- **The censoring rule's choice** is the censored metric, because both wirings cross 30% on almost every
  seed (98% and 100%). The reading is fixed on `auc_success` whatever this field says.

- **The D24 audit**, committed before launch ([table.json](supporting/099-logcount-contrast/table.json)).
  The log-count null still creates a median of five sensor–motor gap pairs, and drops the wild type's
  count–degree correlation from 0.19 to about zero. The null therefore carries a shortcut the wild type
  lacks, and the two still learn alike.

## Interpretation

On the point worm, the wild type's gap-junction advantage was a property of how the model scaled gap
junctions, not of where they are.

- **Under count-scaled additive coupling** the lead was large: +0.104 on Emmons, +0.096 on Cook (082),
  and it survived embodiment (092).
- **Under the two corrections tried, it is gone.** Unit weights give +0.003, and log-compressed counts
  give +0.003, with an interval that excludes any lead above 0.016.

The count model let a few high-count pairs between low-degree neurons couple far above 1: ALA–CAN at
89.7. Rewiring moved those pairs, and that was what the learner felt. With the counts compressed, placing
the gap junctions where the animal has them earns nothing on this cell.

What this does not show:

- That gap placement is irrelevant to learning in every model. The two corrections here keep the
  coupling additive. Ohmic coupling was not chosen, because it learned more slowly (Logbook 097), and it
  was not read as a contrast.
- Anything about the body. That is the other half of the STOP condition, and it is read next.

## What follows

- **Under the registered map**, G.1–G.3b are not licensed on the point worm, and the STOP condition stays
  open until the body cell reads on log counts.
- **092's body contrast on log counts** registers next. M.1 has read (Logbook 098), so it is free to
  register; its horizon is for its own change to fix. The joint outcomes are already fixed:
  - **The body also shows no effect:** the STOP branch is taken. The phase's deliverable becomes the
    statement that the project's one surviving wiring effect belonged to the electrical-synapse model,
    with the controls that make that citable.
  - **The body shows an effect:** whether G.1–G.3b move to the body is a dated decision then.
- **The decomposition** of Logbooks 074–092 is restated once the body has read.

## Data

- [supporting/099-logcount-contrast/launch.md](supporting/099-logcount-contrast/launch.md): the
  registration.
- [preflight.json](supporting/099-logcount-contrast/preflight.json): the gate preflight on G.0-A's
  pilot.
- [table.json](supporting/099-logcount-contrast/table.json): D24's matched statistics.
- [panel.json](supporting/099-logcount-contrast/panel.json) and
  [per-seed.csv](supporting/099-logcount-contrast/per-seed.csv): the reading, the gates and every seed.
- Raw logs and session records of the panel: archived off-repo.
