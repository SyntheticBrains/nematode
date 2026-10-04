# 077: On Block V's Thermal Cell the Panel Saturates, and Its Lead Disappears Against Nulls With the Wild Type's Gap Junctions (Phase 8b A.6t)

**Status**: completed — **registered verdict: unreadable.** Every level saturates. Both learning arms
plateau at 94–96% full-clear success, above the registered 90% bar, so neither primary interaction may
be read. The registration imported that gate from the hard350 panels, where A.6's arms plateaued at
72–76%, without checking that block V's thermal point sits above it. That is a registration error, and
it is stated rather than corrected after the fact.

**Reported beside, as registered, and not a verdict.** At 128 seeds the three gaps are:

| wild type against | `auc_success` gap | 80% CI | episodes to competence | 80% CI |
|---|---|---|---|---|
| the current null | **+0.089** | [+0.064, +0.112] | **+202** | [+124, +279] |
| the chemical-only null | +0.010 | [−0.014, +0.032] | −3 | [−71, +63] |
| the gap-held null | −0.007 | [−0.028, +0.014] | **−71** | [−130, −12] |

**Against either null that keeps the wild type's gap junctions, block V's thermal lead is not there.**
The gap-held null shares the current null's chemical graph exactly, and differs from it only in its gap
junctions. Against it the wild type is level on `auc_success` and slower to competence by 71 episodes,
an interval that excludes zero. On hard350 about half the lead survived the same control. On this cell,
at this saturation, none of it is visible.

**Date**: 2026-10-05.

**OpenSpec change**: `add-thermal-null-strength`, which adds the requirement that a control repeated on
another cell is sized and judged on that cell's own effect.

**Pre-registration**:
[supporting/077-thermal-null-strength/launch.md](supporting/077-thermal-null-strength/launch.md),
committed before any scored run.

## Objective

[Logbook 074](074-null-strength-control.md) and [075](075-gap-only-split.md) found that about half of
block V's `auc_success` lead over the degree-preserving null on hard350 came from that null's rewired
gap junctions. Block V's second cell, the thermal-plus-foraging cell at target 20, had only A.1's
initialisation control behind it ([Logbook 070](070-init-sharing-control.md)). This panel runs both
controls on that cell.

## Method

**Four wirings**, each learning and frozen, under PPO at block V's committed thermal point: the
edge-order draw, the pooled readout, depth 4, target 20. They are the wild type, the current
degree-preserving null, the chemical-only null and the gap-held null. **Seeds 385–512 (128)**, fresh,
3,000 episodes each, 1,024 runs. All 1,024 succeeded, in 16.4 hours of run time on 16 workers and 18.4
hours elapsed, the difference mostly about 75 minutes the machine spent asleep.

**Two primary interactions**, corrected together:

- **`combined`**: the gap against the chemical-only null minus the gap against the current null. It
  holds gap placement, gap strength and autapses together, as A.6 did.
- **`split`**: the gap against the gap-held null minus the gap against the current null. It holds the
  gap junctions alone on the current null's exact chemical graph, as A.6's split did.

**The minimum** is 0.0553 `auc_success`, two-thirds of A.1's thermal effect, the cell's own. **The
gates** are the hard350 panels': each learning arm must beat its frozen floor, and a level where both
learning arms plateau at or above 90% full-clear success is unreadable.

## Results

### Gates

| level | wild type plateau / floor | null plateau / floor | beats floor | saturated |
|---|---|---|---|---|
| full | 94.0% / 0.2% | 93.8% / 0.4% | yes | **yes** |
| chemical | 94.0% / 0.2% | 94.4% / 1.5% | yes | **yes** |
| gap-held | 94.0% / 0.2% | 96.0% / 1.2% | yes | **yes** |

Every arm learns, far above its frozen floor. Every level saturates. **The panel is unreadable**, and no
state or verdict is assigned to either interaction.

### The interactions, as numbers only

They are reported because the analysis computed them, and they carry no verdict.

| interaction | mean | 80% CI | q |
|---|---|---|---|
| `combined` | −0.079 | [−0.100, −0.058] | < 0.001 |
| `split` | −0.096 | [−0.113, −0.078] | < 0.001 |

Both are negative: the wild type stands worse against each narrower null than against the current one.
The `split` mean is 121% of the `combined` mean.

### Achieved sensitivity, beside the registered one

The registered proxy put the interaction spread at 0.27–0.30. The achieved spreads are 0.18
(`combined`) and 0.15 (`split`), so the achieved detectable effects are about 0.040 and 0.034, below
the 0.055 minimum. As the protocol requires, this is reported and never used to re-read anything.

## Why the gate fired, and what it does and does not protect against

The saturation gate exists because at a ceiling, time-to-competence compresses: two arms that both
reach 94% can look tied when one would have learned faster with more room. A gap that vanishes at
saturation might be that compression rather than the wiring.

**One observation bears on that here, and it is post hoc.** The gap against the current null is
+0.089 `auc_success` at the same saturation. That is A.1's thermal effect (+0.083) almost exactly, so
the instrument resolved a gap of the expected size at this ceiling on these seeds. That makes ceiling
compression a less likely reason for the other two gaps to vanish. **It does not license a verdict**:
the gate was registered, it fired, and reading past it now would be choosing the rule after seeing the
data.

**A.1 applied no saturation gate on this cell**, and block V's own thermal readings were made on the
efficiency axis at this same point. So every committed thermal result sits at this saturation too.
That conditions those results; it does not invalidate them.

## What this establishes, and what it does not

- **Established:** at block V's committed thermal point, every arm of every null learns, and all of
  them saturate. The registered verdict is unreadable.
- **Described, not established:** against nulls that keep the wild type's gap junctions, block V's
  thermal lead is not visible, and against the gap-held null the wild type is slower to competence.
  If a non-saturating thermal panel confirms this, block V's thermal advantage would be entirely the
  current null's rewired gap junctions, where on hard350 it was about half.
- **Not established:** anything about the chemical wiring on thermal. A null result here would need a
  readable panel.

## Registered consequences, and the decision this leaves

The registration named no follow-up for an unreadable panel. The options are the maintainer's:

1. **A non-saturating thermal panel.** Register a thermal point below the 90% bar, for example a
   higher food target, which is how block V moved this cell off target 10. It needs its own operating
   point checked first, since block V's thermal claims were made at target 20.
2. **Record the described result and move on.** Carry "unreadable; described, the thermal lead is not
   visible against nulls with the wild type's gap junctions" into block V's conditions, and let the
   combined paper's thermal half say so.

**Block V's thermal condition**, until one of those is done: on the thermal cell, block V's advantage
over the degree-preserving null is +0.089 `auc_success` at 128 seeds, at a saturated operating point.
A control against nulls with the wild type's gap junctions was unreadable by its registered gate, and
described, showed no lead.

## Artefacts

- [launch.md](supporting/077-thermal-null-strength/launch.md): the registration.
- [control.json](supporting/077-thermal-null-strength/control.json): gates, gaps, interactions, the
  censoring rule's choice, the drift record and the reading.
- [per-seed.csv](supporting/077-thermal-null-strength/per-seed.csv): every arm's plateau and floor, the
  three gaps and the two interactions, per seed.
- Reproduce: `scripts/analysis/thermal_null_strength.py --campaign <campaign> --out-dir <dir> --out control.json --csv per-seed.csv`. The campaign directory is archived off-repo.
