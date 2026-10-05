# 078: On Block V's Thermal Cell, the Null's Rewired Gap Junctions Carry About 84% of the Lead (Phase 8b A.6t follow-up)

**Status**: completed — **`gap_junctions`; the remaining lead `lead_below_minimum`.** At a readable
thermal point (target 35), holding a null's gap junctions at the wild type's moves block V's wiring gap
toward the null by more than the registered minimum. Against that gap-held null the wild type still
leads, significantly but by less than the minimum.

| registered reading | mean | 80% CI | q | verdict |
|---|---|---|---|---|
| **split**: gap(gap-held) − gap(current) | **−0.0865** `auc_success` | [−0.097, −0.075] | < 0.001 | **gap_junctions** |
| **lead**: wild type vs gap-held null | **+0.0164** | [+0.006, +0.027] | 0.028 | **lead_below_minimum** |

Beside them: block V's effect at this point is **+0.103** `auc_success` [+0.092, +0.113] and **+404
episodes to competence**. Against the gap-held null the wild type is **+73 episodes** [+26, +122].

**About 84% of block V's thermal `auc_success` lead over the degree-preserving null came from that null's
rewired gap junctions. A chemical-wiring lead remains, smaller than the registered minimum.** On hard350
the same control left about half ([Logbook 075](075-gap-only-split.md)).

**Date**: 2026-10-06.

**OpenSpec change**: `add-thermal-split`, which adds the requirement that a difficulty pin for a
follow-up is chosen by the gates alone, under a rule fixed before the pilot that decides it.

**Pre-registration**: [supporting/078-thermal-split/launch.md](supporting/078-thermal-split/launch.md),
committed before any scored run, after a spec review and the gate preflight.

## Objective

[Logbook 077](077-thermal-null-strength.md) ran this question at block V's thermal point (target 20)
and was unreadable: every level saturated. Described there, block V's thermal lead was not visible
against nulls with the wild type's gap junctions. This panel asks the split's question at a thermal
point where the gates can be read.

## Method

**Choosing the point.** A gate-only pilot on seeds 1001–1004 tried food targets 25, 30 and 40, then
35 under a rule written down before it ran: the lowest target at which every level is readable. Target
25 saturated, 30 sat within the preflight's 5-point margin, and 35 and 40 were readable, so 35 was
chosen ([pilot.json](supporting/078-thermal-split/pilot.json)).

**The panel.** The wild type, the current degree-preserving null and the gap-held null, each learning
and frozen, under PPO at block V's committed point except the target: edge-order draw, pooled readout,
depth 4, target 35. The gap-held null has the current null's chemical graph exactly and the wild type's
gap junctions, so against the current null it differs in its gap junctions alone, placement and
strength together. **Seeds 513–640 (128)**, 3,000 episodes, 768 runs, all succeeded in 13 hours.

**The readings.** `split` and `lead`, corrected together, each against a minimum of **0.0239
`auc_success`**: two-thirds of A.1's thermal effect at target 20, scaled by the wild type's own
`auc_success` ratio between the targets (0.312 / 0.721). The ratio uses the wild type alone.

## Results

### Gates

| level | wild type plateau / floor | null plateau / floor | beats floor | saturated |
|---|---|---|---|---|
| full | 69.7% / 0.0% | 56.7% / 0.0% | yes | no |
| gap-held | 69.7% / 0.0% | 68.9% / 0.0% | yes | no |

Both levels are readable, with 20 points to the 90% bar. Drift reads as PPO's positive control: PPO
writes the chemical matrix, and it moved.

### The registered readings

**`split` → `gap_junctions`.** Holding the gap junctions moves the wiring gap by −0.0865 `auc_success`
[−0.097, −0.075], about 3.6 times the minimum, and by −330 episodes [−374, −282].

**`lead` → `lead_below_minimum`.** Against the gap-held null the wild type leads by +0.0164 \[+0.006,
+0.027\] at q = 0.028. The lead is significant and below the minimum of 0.0239. The wild type is ahead
on 73 of 128 seeds, against 113 of 128 against the current null.

### Achieved sensitivity, beside the registered one

| reading | achieved sd | achieved MDE | registered range | minimum |
|---|---|---|---|---|
| `split` | 0.103 | 0.022 | 0.015–0.034 | 0.024 |
| `lead` | 0.098 | 0.021 | 0.018–0.041 | 0.024 |

The spread fell between the two proxy readings the registration gave. It is reported, and never used to
re-read a verdict.

### Episodes, reported beside

The censoring rule found the episode metric comparable (crossing rates 96–99%).

| wild type against | episodes to competence | 80% CI |
|---|---|---|
| current null | +404 | [+357, +451] |
| gap-held null | +73 | [+26, +122] |

## What this establishes, and what it does not

- **Established**: at target 35 on the thermal cell under PPO, most of block V's lead over the
  degree-preserving null, about 84% of it on `auc_success`, came from that null's rewired gap junctions.
  Holding them moves the gap by more than the registered minimum.
- **Established, with its condition**: a chemical-wiring lead remains against a null with the wild
  type's gap junctions, significant on both metrics (+0.016 `auc_success`, +73 episodes) and below the
  registered minimum on the primary. It may not be cited as a lead of at least the minimum.
- **Not established**: anything at target 20, block V's own thermal point. That remains Logbook 077's
  condition: unreadable, and described as no visible lead. Target 35 is a harder operating point, and
  the two are not read as each other.
- **Not separated**: gap placement against gap strength. The gap-held null holds both.

## Block V, restated with the thermal half

On hard350, against a null with the wild type's gap junctions, block V is +0.025 `auc_success` and +235
episodes, about half its lead over the degree-preserving null. On the thermal cell at target 35 it is
+0.016 `auc_success` and +73 episodes, about a sixth: a significant lead below the registered minimum.
**On both cells the null's rewired gap junctions carry most or half of block V's advantage, and a
smaller chemical-wiring lead in learning speed remains.**

## Artefacts

- [launch.md](supporting/078-thermal-split/launch.md), [pilot.json](supporting/078-thermal-split/pilot.json),
  [selection-rule.md](supporting/078-thermal-split/selection-rule.md): the registration and its pilot.
- [control.json](supporting/078-thermal-split/control.json): gates, gaps, the split, the censoring
  rule's choice, drift and the reading.
- [per-seed.csv](supporting/078-thermal-split/per-seed.csv): every arm's plateau and floor, both gaps
  and the split, per seed.
- Reproduce: `scripts/analysis/thermal_split.py --campaign <campaign> --out-dir <dir> --out control.json --csv per-seed.csv`. The pilot and campaign directories are archived off-repo.
