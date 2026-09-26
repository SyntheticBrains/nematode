# 074: Holding the Null's Gap Junctions and Autapses Halves Block V's Advantage — Below the Registered Minimum, and Significant on Both Learners (Phase 8 A.6)

**Status**: completed — **`below_minimum` on both learners under the registered map, with a
significant move in the same direction on each.** A.6 compared the wild type against a null that
rewires only the chemical graph and holds the wild type's gap junctions and autapses.

- **PPO at block V's committed point:** the wild type's lead shrinks from **+0.049** `auc_success`
  against the current null to **+0.022** [+0.007, +0.036] against the chemical-only null. The
  interaction is −0.028 [−0.041, −0.014], q = 0.019.
- **On block V's headline metric, episodes to competence,** the lead falls from **+372** [+192, +547]
  to **+94** [−81, +280], which **no longer excludes zero** at 32 seeds.
- **Under the reading learner** the null's lead grows from −0.064 to **−0.163**, an interaction of
  −0.099 [−0.151, −0.049], q = 0.019.

Neither interaction reaches the registered 2/3 minimum (0.041 and 0.140). Under the map that reading
is `below_minimum` and licenses no verdict of its own. Block V's claim is now **conditioned, not
dissolved**: against a null that keeps the wild type's gap junctions and autapses, a chemical-wiring
advantage remains on `auc_success`, about 44% of the lead measured against the current null. The
learning-speed form of the claim is unresolved at this panel's size.

**Date**: 2026-09-25 to 2026-09-26.

*(**Sharpened 2026-09-26**, [Logbook 075](075-gap-only-split.md): Against a null with the wild type's gap junctions (placement and strength), block V's PPO advantage on hard350 is **+0.025 `auc_success` [+0.012, +0.038]** and **+235 episodes to competence [+62, +409]**: about half of its `auc_success` lead over the degree-preserving null (+0.049) came from that null's rewired gap junctions, and a chemical-wiring advantage in learning speed remains.)*

**OpenSpec change**: `add-null-strength-control`. It extends `connectome-ppo-brain` with the
`rewired_chemical_only` wiring and restates what each null preserves, and it extends
`architecture-comparison-protocol` with "a null states every structural property it does not
preserve".

**Pre-registration**: [supporting/074-null-strength-control/launch.md](supporting/074-null-strength-control/launch.md),
written before any seed ran.

## Objective

The Wormlight review (PR #405) found that the degree-preserving null behind every wiring result here
differs from the wild type in more than which neurons connect:

- **Gap-junction strength.** Each junction's EM count is its coupling weight and travels with the
  edge. ALA's gap input is 232 in the wild type and about 3 on a null, and about half of all neurons'
  gap totals move by more than 50%.
- **Autapses.** The wild type's 38 are lost; each of three nulls checked had none.

A.1 controlled neither. [Logbook 067](067-l4-feature-ablations.md) had found the reading learner's
advantage consisted in the wild type's gap junctions costing it less than the rewired ones cost the
null. A.6 asks whether block V's wiring gap is in the chemical wiring, or partly in how the null
treats gap junctions and autapses.

## Method

Wild type, the current null (`rewired_degree_preserving`) and the chemical-only null
(`rewired_chemical_only`: the same chemical swap, with the gap junctions' pairs and counts and all 38
autapses held at the wild type's). Each arm ran learning and frozen, on hard350.

| learner | point | seeds | runs | wall clock |
|---|---|---|---|---|
| PPO | block V's committed point: `edge_order`, pooled readout, depth 4 | 305–336 (32) | 192 | 3.0 h |
| reading (`readout_only`) | A.2's centre, `edge_order` | 337–384 (48) | 288 | 8.2 h |

**The interaction.** For each learner the interaction is
`gap(wild type vs chemical null) − gap(wild type vs current null)`, on `auc_success` (registered, with
episodes reported beside). The wild type cancels, so it is the current null minus the chemical null,
seed by seed.

**The verdict rule.** Each interaction is read against 2/3 of the learner's committed reference
effect: 0.041 for PPO (A.1's +0.061) and 0.140 for the reading learner (A.2's −0.2105). q comes from
BH-FDR over the two interactions.

**It is a combined control.** Gap placement, gap strength and autapses are held together, and the two
nulls are different random chemical graphs at each seed. The existing null was pinned before the code
changed and is unchanged. Every statistic is A.2's, unmodified.

## Results

### The gates

Every learning arm beat its own floor, and no level saturates, so both learners are readable. The
reading learner's `w_chem` drift is **0.0** on every seed; under PPO the same check reads up to 1.33,
its positive control.

| level | PPO wild type | PPO null | reading wild type | reading null |
|---|---|---|---|---|
| current null (`full`) | 76.4 | 72.2 | 46.3 | 52.3 |
| chemical null | 76.4 | **73.4** | 46.3 | **61.7** |

The wild-type column is the same runs at both levels. **The chemical null learns better than the
current null on both learners**, by 1.2 points under PPO and 9.4 under the reading learner. Holding
the wild type's gap junctions and autapses helps a rewired graph.

### The primary

`auc_success`, with 80% intervals and BH-FDR q.

| learner | gap vs current null | gap vs chemical null | interaction | q | minimum | state | verdict |
|---|---|---|---|---|---|---|---|
| PPO | +0.0493 [+0.0362, +0.0621] | +0.0215 [+0.0074, +0.0358] | **−0.0278** [−0.0414, −0.0139] | 0.019 | 0.041 | `below` | **below_minimum** |
| reading | −0.0639 [−0.1186, −0.0082] | −0.1625 [−0.2306, −0.0954] | **−0.0986** [−0.1515, −0.0493] | 0.019 | 0.140 | `below` | **below_minimum** |

**Read against the registered map.** Both interactions are significant, and both intervals exclude
zero on the same side. Both are smaller than 2/3 of the committed effect, so both are `below` and
license no verdict of their own. The map reserved `gap_or_autapse` for a move of at least the minimum,
and this is not one.

**What the numbers say beside the map, as description.**

- **PPO.** Holding the null's gap junctions and autapses removes **56%** of the wild type's lead as
  measured in this campaign (0.0278 of 0.0493), and 46% of the committed reference. It falls short of
  2/3 either way.
- **Reading learner.** The null's lead grows 2.5-fold, which is 47% of the reference in absolute
  size.
- **Direction.** On both learners the move is toward the null: the current null's rewired gap
  junctions and lost autapses were costing it. This is the direction Logbook 067 found for the reading
  learner, now shown under PPO as well.

### Episodes, reported beside

| learner | gap vs current null | gap vs chemical null | interaction | q |
|---|---|---|---|---|
| PPO | +371.7 [+192.4, +546.8] | **+93.5 [−80.9, +279.7]** | −278.2 [−424.2, −112.7] | 0.029 |
| reading | −182.2 [−433.2, +73.3] | −404.2 [−680.1, −135.7] | −222.0 [−418.8, −31.6] | 0.029 |

**Block V's headline is a learning-speed claim**: the wild type reaches competence sooner. Measured in
episodes, against the chemical null, it is **+94 episodes with an interval spanning zero**, and 75% of
the lead against the current null is gone. The censoring rule would have chosen episodes for both
learners (crossing-rate spreads 0.0 and 0.042); `auc_success` was registered in its place before
launch.

### The hop probe on the chemical null, as description

A.2's measurement, repeated on the chemical-only null over eight rewirings
(`sensory-motor-hops-chemical-null.json`):

- **One-hop routes:** the chemical null manufactures **8.0** motor neurons one hop from a food sensor
  on average, against the current null's 9.25 and the wild type's 0.
- **Reach at two hops:** it reaches all 39 motor neurons, against the wild type's 26.

**A.2's depth mechanism therefore survives here.** It comes from the chemical rewiring, not the gap
junctions.

## Corrections made in the open

- **Two defects were fixed at spec review, before launch.** The first draft's `chemical` verdict could
  have reported a gap as "in the chemical wiring" where no gap existed; it now needs the attribution
  gate. The draft also called the sensitivity proxy conservative, although the two nulls are
  different random graphs at each seed.
- **The PR review of the roadmap change (PR #405) removed attributions this control cannot make.** It
  stopped any outcome being assigned to gap *strength* alone, where a combined control holds
  placement, strength and autapses together, and it qualified the autapse claim: zero autapses is what
  three seeds showed, not a guarantee.
- **None of the figures above changed after scoring.**

## What this establishes, and what it does not

**Establishes.**

- The current null's handling of gap junctions and autapses **carries a significant part of the
  wiring gap on both learners**, in the same direction on each: holding them at the wild type's moves
  the gap toward the null.
- Under PPO the wild type **still leads** a null that keeps its gap junctions and autapses, on
  `auc_success` (+0.022, the interval excluding zero): a chemical-wiring advantage exists.
- A.2's hop mechanism is chemical.
- The existing null is unchanged, so every committed result stands as read against it.

**Leaves unresolved.**

- **Whether block V's learning-speed advantage survives** against the chemical null: +94 episodes,
  with the interval spanning zero at 32 seeds.
- **Which of the held properties carries the move**: gap placement, gap strength or autapses. The
  gap-only null, which pairs exactly with the current null, separates the gap junctions' joint effect
  (placement and strength together) from the autapses'; telling placement from strength needs a
  further control that moves one without the other.

**Does not establish.**

- That block V's advantage dissolves. The registered map gives `below_minimum`, not `gap_or_autapse`,
  and a chemical advantage remains on the primary.
- That the move is caused by gap strength in particular.

## Registered consequences

- **A.6: met.** Both learners were read against the null-strength control and both carry a verdict:
  `below_minimum` on each.

- **Block V's standing condition is resolved into a quantified one**, stated in the same sentence as
  the claim at every citation site:

  > On hard350 under PPO, the wild type's advantage over its degree-preserving null is
  > +0.049 `auc_success` and +372 episodes. Against a null that also holds the wild type's gap
  > junctions and autapses it is **+0.022 [+0.007, +0.036]** and **+94 [−81, +280] episodes**, so the
  > learning-speed form of the claim is unresolved at 32 seeds.

  Dated notes carry this at the roadmap's A.1 and A.6, the tracker's A.1 and A.6, and Logbooks 067
  and 070.

- **The gap-only split.** Registration made it the follow-up *if the combined control moves*. Under
  the map it did not move: `below` is not a move, so the registered trigger did not fire. **A
  significant move of about half the effect, on both learners, is still a strong reason to run it.**
  It is recorded as *deferred-with-destination* to the 8a synthesis (S8a), which decides whether it
  runs before A.5.

- **A.5, the publication decision**, inherits the quantified condition. Block V is not cited without
  the chemical-null figures.

## Artefacts

Under [supporting/074-null-strength-control/](supporting/074-null-strength-control/):

- `launch.md`: the registration, with the table of what each null preserves.
- `control.json`: each level's gates (per-seed plateau and floor), both gaps and the interaction on
  both metrics with their q, the states and verdicts, the censoring rule's own choice, and the drift
  evidence.
- `per-seed.csv`: 80 rows, one per learner and seed.
- `sensory-motor-hops-chemical-null.json`: the hop probe on the chemical-only null.

Drivers:

- `scripts/analysis/null_strength_control.py`, which reuses `operating_point_surface.py`,
  `measured_prior_contrast.py` and `measured_prior_pilot.py`;
- `scripts/analysis/sensory_motor_hops.py --null rewired_chemical_only`;
- configs from `scripts/campaigns/generate_null_strength_configs.py`.

Raw campaign logs (480 runs across two campaign directories) are archived off-repo under A.0.
