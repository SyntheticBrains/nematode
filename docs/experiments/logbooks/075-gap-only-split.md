# 075: The Null's Rewired Gap Junctions Carry A.6's Move on Both Learners (Phase 8 A.6 follow-up)

**Status**: completed — **`gap_junctions` on both learners.** A.6 found that holding a null's gap
junctions and autapses at the wild type's moved the wiring gap toward the null. Holding the **gap
junctions alone**, on the current null's exact chemical graph, reproduces that move at or above the
registered 2/3 minimum.

| learner | move from holding the gap junctions | A.6's move | share | q |
|---|---|---|---|---|
| PPO | **−0.024** `auc_success` [−0.036, −0.012] | −0.028 | about 87% | 0.025 |
| reading learner | **−0.131** [−0.185, −0.074] | −0.099 | more than all of it | 0.021 |

In both learners the remainder — the autapses together with the chemical-graph difference in A.6's
null — is indistinguishable from zero.

**What block V's advantage becomes.** Under PPO, against a null with the wild type's gap junctions,
the wild type still leads:

- **+0.025** `auc_success` [+0.012, +0.038];
- **+235 episodes to competence** [+62, +409].

**About half of block V's `auc_success` lead against the current null came from that null's rewired
gap junctions. The rest is chemical wiring, and it shows in learning speed.**

**Date**: 2026-09-26.

**OpenSpec change**: `add-gap-only-split`, which extends `connectome-ppo-brain` with the
`rewired_gap_junctions_held` wiring and its exact-pairing guarantee.

**Pre-registration**: [supporting/075-gap-only-split/launch.md](supporting/075-gap-only-split/launch.md),
written before any new seed ran.

## Objective

[Logbook 074](074-null-strength-control.md) moved block V's wiring gap toward the null on both
learners, but it did so by holding a null's gap junctions **and** its autapses together, so it could
not say which carried the move. This split holds the gap junctions alone, and asks what share of A.6's
move they reproduce.

## Method

**The gap-held null** runs the current null's chemical swap unchanged and skips only the gap-junction
swap. Tests confirm that at each seed its chemical mask, its drawn chemical weights and its autapse
diagonal are bit-identical to the current null's, and its `g_gap` to the wild type's. Against the
current null it therefore differs in its gap junctions alone, **placement and strength jointly**.
Nothing here separates those two.

**The panel reuses A.6.** It uses A.6's seeds (305–336 for PPO, 337–384 for the reading learner) and
A.6's runs of the wild type, the current null and the chemical-only null. Only the gap-held arms are
new: 64 PPO runs and 96 reading runs, 58 minutes and 2 h 52 min.

**The reuse was licensed first.** One seed per reused arm was re-run with A.6's exact command line,
12 runs in all. **Every one matched A.6's run on all 3,000 `Run:` lines and on the final `w_chem`, bit
for bit**
([identity-ppo.json](supporting/075-gap-only-split/identity-ppo.json),
[identity-reading.json](supporting/075-gap-only-split/identity-reading.json)).

Scoring also recomputed A.6's move from the reused runs. **It matches A.6's committed per-seed CSV at
every seed on both learners** (`a6_reproduced`).

**The primary** is, per learner, `gap(gap-held) − gap(current)` on `auc_success`, the current null
minus the gap-held null on the same chemical graph. It is read against 2/3 of A.6's committed move:
0.0185 for PPO and 0.0657 for the reading learner. The two primaries share a BH-FDR family, and every
statistic is A.2's, unmodified.

## Results

### Gates

Every learning arm beats its floor, and no level saturates. The reading learner's `w_chem` drift is
**0.0** on every seed.

Plateau success of each null, against the wild type's 76.4 (PPO) and 46.3 (reading):

| null | PPO | reading |
|---|---|---|
| current | 72.2 | 52.3 |
| chemical-only (A.6) | 73.4 | 61.7 |
| **gap-held** | **74.5** | **65.9** |

**On both learners, the null with the wild type's gap junctions learns best of the three.**

### The primary

`auc_success`, with 80% intervals.

| learner | gap vs current null | gap vs gap-held null | gap junctions (primary) | q | minimum | verdict |
|---|---|---|---|---|---|---|
| PPO | +0.0493 [+0.0362, +0.0621] | +0.0250 [+0.0121, +0.0377] | **−0.0243** [−0.0362, −0.0119] | 0.025 | 0.0185 | **gap_junctions** |
| reading | −0.0639 [−0.1186, −0.0082] | −0.1952 [−0.2476, −0.1413] | **−0.1313** [−0.1853, −0.0736] | 0.021 | 0.0657 | **gap_junctions** |

Both interactions are significant, point toward the null, and pass the minimum, so the state is
`move_null`.

**Sensitivity achieved.** The paired spread gives a minimum detectable effect of 0.84× A.6's move
for PPO, against the registered 0.97×, and 1.11× for the reading learner, against 1.03×. Exact
pairing sharpened PPO but not the reading learner. Both verdicts cleared the minimum either way.

### The breakdown, as description

Everything is at the same seeds, so A.6's move splits exactly:

| learner | A.6's move | gap junctions | remainder (autapses + chemical-graph difference) |
|---|---|---|---|
| PPO | −0.0278 [−0.0414, −0.0139] | −0.0243 [−0.0362, −0.0119] | −0.0035 [−0.0166, +0.0101] |
| reading | −0.0986 [−0.1515, −0.0493] | −0.1313 [−0.1853, −0.0736] | +0.0327 [−0.0262, +0.0881] |

**The gap junctions carry about 87% of A.6's move under PPO and more than all of it under the reading
learner.** A.6's own move is uncertain, so the share is approximate. The remainder is
indistinguishable from zero on both learners, and it is not attributed to the autapses, since it mixes
them with the chemical-graph difference in A.6's null.

### Episodes, reported beside

| learner | gap vs current null | gap vs gap-held null | gap junctions | q | A.6's move | remainder |
|---|---|---|---|---|---|---|
| PPO | +371.7 | **+234.9 [+61.8, +408.7]** | −136.8 [−304.4, +40.8] | 0.51 | −278.2 | −141.3 [−308.0, +23.7] |
| reading | −182.2 | −552.8 [−774.6, −338.0] | −370.5 [−547.2, −186.7] | 0.011 | −222.0 | +148.5 [−48.7, +343.8] |

**On PPO's episodes to competence, the wild type still leads a null that has its gap junctions:** by
+235 episodes, with an interval clear of zero. Against A.6's chemical-only null the lead was +94, with
an interval spanning zero. On this metric the gap-junction term (−137) and the remainder (−141) each
span zero, so episodes do not attribute A.6's move. That leaves the learning-speed question in this
state:

- **Against a null with the wild type's gap junctions,** the lead is resolved.
- **Against a null that also keeps the autapses and draws a different chemical sample,** it is not.

## Corrections made in the open

- **The design said 384 reused runs; it is 480.** The count left out the chemical-only null's arms.
  This was corrected before launch, and all 480 had their drift evidence confirmed.
- **Two defects were fixed at spec review, before launch.**
  - The identity check compared only the `Run:` lines, although the drift check also reads the final
    weights.
  - `not_gap_junctions` said "mostly" where the bound only says "at least a third".
- **None of the figures above changed after scoring.**

## What this establishes, and what it does not

**Establishes.**

- On hard350, **A.6's move comes from the current null's rewired gap junctions**, placement and
  strength jointly, on both learners.
- A null with the wild type's gap junctions learns better than the current null on both learners.
- **Under PPO a chemical-wiring advantage remains against that null,** in `auc_success` (+0.025) and
  in episodes to competence (+235, with the interval excluding zero).
- The reuse of A.6's runs is licensed, and A.6's committed per-seed result reproduces from them.

**Leaves unresolved.**

- Whether **placement or strength** carries the gap-junction effect. This panel holds both.
- The learning-speed lead against A.6's chemical-only null, which also keeps the autapses and draws
  a different chemical sample (+94 episodes, the interval spanning zero). The episode remainder, −141,
  also spans zero.

**Does not establish.**

- That autapses matter or do not. The remainder mixes them with a chemical-graph difference.
- Anything beyond hard350, `edge_order` and the two learners' swept points.

## Registered consequences

- **A.6's follow-up: met.** Both learners read `gap_junctions`.

- **Block V's condition is sharpened.** It is stated in the same sentence as the claim at every
  citation site:

  > On hard350 under PPO, about half of the wild type's `auc_success` lead over its
  > degree-preserving null (+0.049; +372 episodes) came from that null's rewired gap junctions. Against a null
  > with the wild type's gap junctions, the lead is **+0.025 [+0.012, +0.038] `auc_success`** and
  > **+235 [+62, +409] episodes to competence**, a chemical-wiring advantage in learning speed.

  Dated notes carry it at the roadmap's A.1 and A.6, the tracker's A.1, A.6 and S8a, and Logbooks
  067, 070 and 074.

- **Placement against strength** is *deferred-with-destination* to the 8a synthesis. Answering it
  needs a control that moves one without the other, for instance gap junctions kept in place with
  their counts redistributed. S8a decides whether A.5 needs it.

- **The reading learner's result is sharpened too.** Logbook 067 found its wild-type advantage "in
  the wild type's gap junctions costing it less than the rewired ones cost the null". This split
  measures that cost directly: holding the gap junctions lets the null learn 13.6 points better.

## Artefacts

Under [supporting/075-gap-only-split/](supporting/075-gap-only-split/):

- `launch.md`: the registration, with the three-null table.
- `identity-ppo.json`, `identity-reading.json`: the identity check, per arm, covering `Run:` lines and
  final `w_chem`.
- `split.json`: every level's gates, gaps, the primary and the breakdown on both metrics, the drift
  evidence and the A.6 reproduction check.
- `per-seed.csv`: 80 rows, one per learner and seed.

Drivers:

- `scripts/analysis/gap_split.py` (`identity`, `evidence`, `score`), which reuses
  `null_strength_control.py`, `measured_prior_contrast.py`, `measured_prior_pilot.py` and
  `operating_point_surface.py`;
- configs generated by `scripts/campaigns/generate_null_strength_configs.py`.

Raw logs (160 new runs, 12 identity runs, and A.6's 480 reused) are archived off-repo under A.0.
