# 082: The Wild Type's Gap-Junction Advantage Is in Their Placement, Not Their Strengths (Phase 8b M.8)

**Status**: completed — **`placement`.** On the thermal cell at target 35, under PPO at block V's point,
a null whose gap junctions alone are rewired trails the wild type by +0.096 `auc_success`. When both
wirings may tune the strength of every gap junction they have, it still trails by +0.086. The
interaction, how much tuning changes the gap, is −0.010, inside the registered minimum.

| registered reading | mean | 80% CI | q | state |
|---|---|---|---|---|
| **base**: wild type vs gap-only null, fixed strengths | **+0.0959** `auc_success` | [+0.081, +0.112] | < 0.001 | `move_wt` |
| **lead**: the same, plastic strengths | **+0.0859** | [+0.065, +0.107] | < 0.001 | `move_wt` |
| **interaction**: lead − base | **−0.0100** | [−0.027, +0.005] | 0.85 | `no_move` |

Beside them, on episodes to competence: +370 at fixed strengths, +349 with plastic ones, and an
interaction of −21 [−91, +50].

**Given its own strengths to tune, the null does not close the gap. The advantage Logbooks 075 and 078
traced to the null's rewired gap junctions is in which neurons are coupled, not in how strongly.** The
tuning was real: across the panel the learned multipliers moved the median gap pair by about 25%, and
moved some pairs 20-fold or more.

**Date**: 2026-10-08.

**OpenSpec change**: `add-plastic-gap-junctions`, which adds the gap-only null
(`rewired_gap_junctions_only`) and plastic gap junctions (`plastic_gaps`) on the settling substrate.

**Pre-registration**: [supporting/082-plastic-gaps/launch.md](supporting/082-plastic-gaps/launch.md),
committed before any scored run, after the identity check, a pilot, the gate preflight and the spec
review.

## Objective

[Logbook 075](075-gap-only-split.md) on hard350 and [Logbook 078](078-thermal-split.md) on the thermal
cell found that the degree-preserving null's rewired gap junctions carry about half and about 84% of
block V's lead. A rewiring moves gap **placement** and gap **strength** together, because EM counts
travel with their edges. This panel asks which of the two the wild type's advantage lies in. It was
B.2b's question, which closed unreachable when its leaky substrate failed its positive control
([Logbook 080](080-across-step-state.md)), registered anew on the settling substrate.

## Method

**The two wirings.** The wild type, and the **gap-only null**: its chemical graph, autapses included,
is held exactly, and its gap junctions are rewired by the degree-preserving swap with their counts. It
differs from the wild type in gap placement and in each neuron's total gap strength, and in nothing
else.

**Plastic gaps.** `plastic_gaps` gives every existing gap pair a learnable positive multiplier,
`exp((P + Pᵀ)/2)` on its count, starting at 1, so each wiring starts from its own strengths and PPO can
tune them. No pair is created or removed, so placement stays fixed.

**The panel.** The thermal cell at target 35 under PPO at block V's point otherwise: edge-order draw,
pooled readout, depth 4, Cook 2019, settling dynamics. Each wiring learning with fixed and with plastic
gaps, and frozen. **Seeds 513–576 (64)**, 3,000 episodes. The wild type's fixed-gap learning and frozen
runs are reused from Logbook 078 under an identity check: six re-runs on this change's code matched bit
for bit ([identity.json](supporting/082-plastic-gaps/identity.json)). That leaves **256 new runs**,
all succeeded, in 7.1 hours from a separate worktree. The registration estimated 4.5 hours from
the pilot's run times; the panel ran about 1.6 times longer.

**The readings.** `base`, `lead` and `interaction`, corrected together, each against a minimum of
**0.0577 `auc_success`**, two-thirds of the 0.0865 Logbook 078's split found on this cell. The
registration's verdict map turns the three states into one verdict.

## Results

### Gates

| level | wild type plateau / floor | null plateau / floor | beats floor | saturated |
|---|---|---|---|---|
| fixed | 69.6% / 0.0% | 57.6% / 0.0% | yes | no |
| plastic | 68.7% / 0.0% | 59.4% / 0.0% | yes | no |

Both levels are readable, about 20 points below the 90% bar.

### The registered readings

**`base` → `move_wt`.** With fixed strengths the wild type leads the gap-only null by +0.0959
`auc_success` [+0.081, +0.112], on 58 of 64 seeds. That is close to Logbook 078's split, −0.0865 on 128
seeds, which measured the same gap junctions from the other side: holding them in the full null rather
than rewiring them alone in the wild type. Two constructions agree that the rewired gap junctions are
worth about 0.09 `auc_success` on this cell.

**`lead` → `move_wt`.** With plastic strengths the lead is +0.0859 [+0.065, +0.107], on 53 of 64 seeds.

**`interaction` → `no_move`.** −0.0100 \[−0.027, +0.005\]: tuning narrows the gap by at most 0.027 on its
interval, under half the minimum. It is positive on 35 of 64 seeds.

**Verdict: `placement`**, by the registered map: `base` excludes zero above, `interaction` is `no_move`
and `lead` is `move_wt`.

### The tuning was real

The plasticity check finds every plastic run differing from its fixed-gap twin, on all 64 seeds for
both wirings ([plasticity-panel.json](supporting/082-plastic-gaps/plasticity-panel.json)). How far the
multipliers moved, medians over seeds of each run's statistics over its existing gap pairs
([multipliers.json](supporting/082-plastic-gaps/multipliers.json)):

| wiring | median |log multiplier| | 5th–95th percentile | extremes | total strength vs start |
|---|---|---|---|---|
| wild type | 0.22 | 0.53–1.80 | 0.024–16.0 | 1.07 (0.95–1.39) |
| gap-only null | 0.22 | 0.54–1.69 | 0.055–13.2 | 1.03 (0.91–1.41) |

Both wirings retuned their strengths by about the same amount, with total strength nearly conserved.
So a null that could reweight its own junctions by a factor of two either way, and further on some
pairs, still trailed by about as much as with its counts fixed.

### Achieved sensitivity, beside the registered one

| reading | achieved sd | achieved MDE | registered | minimum |
|---|---|---|---|---|
| `base` | 0.094 | 0.029 | sd 0.1025, MDE about 0.032 | 0.0577 |
| `lead` | 0.131 | 0.041 | | 0.0577 |
| `interaction` | 0.101 | 0.031 | | 0.0577 |

The interaction's spread matched the proxy. Plasticity widened the lead's spread. Every reading could
detect the minimum. These figures are reported, and never used to re-read a verdict.

### Episodes, reported beside

The censoring rule found the episode metric comparable (crossing rates 95–98%).

| reading | episodes to competence | 80% CI |
|---|---|---|
| base | +370 | [+307, +434] |
| lead | +349 | [+262, +434] |
| interaction | −21 | [−91, +50] |

## What this establishes, and what it does not

- **Established**: on the thermal cell at target 35 under PPO, the wild type's gap-junction advantage
  over a null that rewires its gap junctions alone survives letting both wirings tune every gap
  strength they have. Tuning changes the gap by less than the registered minimum. **The advantage is in
  which neurons are coupled**, and a gap-junction claim on this cell is a placement claim.
- **Established, with its condition**: the null had real freedom. Its multipliers moved by about as much
  as the wild type's. They were tuned by PPO in 3,000 episodes, from starting strengths of 1, by the
  same learner that writes the chemical weights. A different optimiser, a longer run, or freedom to
  create pairs could tune further. Placement here means placement that this tuning does not substitute
  for.
- **Not established**: anything on hard350, where Logbook 075 found the gap junctions carry about half
  the lead. This panel ran the thermal cell, where they carry most of it. Nor anything through the
  body, since every run here is the point worm on Cook 2019, 8a's substrate.
- **Not separated**: gap placement from each neuron's total gap strength. A degree-preserving swap
  keeps each neuron's number of gap partners but moves the counts with the edges, so per-neuron totals
  change. The multipliers could have restored the wild type's totals and the gap did not close. That
  argues against totals, but no arm held totals while moving placement.

## Biological fidelity

A learnable multiplier per existing pair stands in for coupling that the worm modulates through innexin
expression (UNC-7, UNC-9 among them) and neuromodulation. Activity-dependent gap-junction weakening is
documented in the olfactory circuit (NMDAR-modulated RMG–AIB coupling, *Nat. Commun.* 2020). The model
keeps the EM placement fixed and tunes strength freely, symmetric and positive. That fits the question,
since placement is what EM measures and strength is what the worm can change. It does not model
rectifying junctions, which are directional, or heterotypic innexin pairs. Both are left for the
dynamics rung if it is reopened.

## Artefacts

- [launch.md](supporting/082-plastic-gaps/launch.md), [identity.json](supporting/082-plastic-gaps/identity.json),
  [preflight.json](supporting/082-plastic-gaps/preflight.json),
  [plasticity.json](supporting/082-plastic-gaps/plasticity.json): the registration, its identity check,
  and its pilot.
- [control.json](supporting/082-plastic-gaps/control.json): gates, gaps, the interaction, the censoring
  rule's choice and the reading.
- [per-seed.csv](supporting/082-plastic-gaps/per-seed.csv): every arm's plateau and floor, both gaps and
  the interaction, per seed.
- [plasticity-panel.json](supporting/082-plastic-gaps/plasticity-panel.json),
  [multipliers.json](supporting/082-plastic-gaps/multipliers.json): the plasticity check on the panel,
  and how far the multipliers moved.
- Reproduce: `scripts/analysis/plastic_gaps.py score --logs campaigns/m8-panel/logs --logs campaigns/a6t2-thermal-split/logs --out-dir <dir> --out control.json --csv per-seed.csv`;
  `plastic_gaps.py plasticity --panel --logs ...`; `plastic_gaps.py multipliers --logs campaigns/m8-panel/logs`.
  The pilot and panel campaign directories and the worktree's session records are archived off-repo.
