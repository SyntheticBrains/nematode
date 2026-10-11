# 088: Most Learners Leave Food Without Ever Roaming, So Patch-Leaving's Coupling to Roaming Cannot Be Read (Phase 8b D.1b)

**Status**: completed — **`unreadable`.** On patchy lawns of 1.5 mm, MLP-PPO reading its satiety learns
the cell and leaves lawns often. Nine of its 16 seeds never roam on a lawn, though, so only 7 seeds
take both states, against the 12 the registration required. Whether its leaving comes out of
roaming, as Scheer & Bargmann 2023 found in worms, is not read.

| gate | result |
|---|---|
| the learner learns (intake against its untrained floor) | **passes**: +1.16 [+1.04, +1.26] |
| each read seed leaves at least 10 times | 16 / 16 |
| each read seed takes both states (on-lawn roaming 5–95%) | **7 / 16**; 9 seeds roam under 5%, 6 of them never |
| at least 12 read seeds | **fails**: the panel is unreadable |

Described, never read as a verdict: the seven two-state seeds' band enrichment averages +0.025 (sd
0.12). Two sit near real worms' value (+0.23, +0.16), and five at zero or below.

**Date**: 2026-10-11.

**OpenSpec change**: `add-patch-leaving`.

**Pre-registration**: [supporting/088-patch-leaving/launch.md](supporting/088-patch-leaving/launch.md),
committed at 0502afdc before any panel run. Its likeliest verdict, no coupling, was stated in
advance; the panel did not reach a verdict at all.

## Objective

D.1 (Logbook 087) showed that a learner dwells on lawns where its untrained policy never does. D.1b
asked the next question: **does its patch-leaving come out of the roaming state, as a worm's does?**
Scheer & Bargmann 2023's 1,586 wild-type animals, read with D.1's classifier at the point worm's
step, roam 46.3% of the time in the 30 to 180 s before leaving a lawn, against 17.5% overall. That is
an enrichment of **+0.288**, or +0.227 with the authors' labels.

## Method

**The reference** is derived from the vendored windows alone
([scheer2023_leaving.json](../../../data/roaming_dwelling/scheer2023_leaving.json)), using the same
code as simulated worms:

- **Exits.** An exit is the first fully off-lawn window after an on-lawn run, so a flicker at the
  edge does not count. The authors' in-lawn and out-of-lawn masks overlap at the moment of leaving,
  and counting raw transitions doubles some exits. By this rule there are 212 sustained exits from
  184 animals.
- **The band.** The reading is the roaming share 30 to 180 s before each exit, minus the overall
  on-lawn share. It leaves out the last 30 s because leaving means moving, and the classifier reads
  movement as roaming: the last window alone reads +0.467.

**The cell** is D.1's (movement cost, slow depletion, a start on a lawn, `entropy_coef` 0.005, no
state-shaping reward), with 4 lawns of 1.5 mm. Two pilots shaped it:

| pilot | cell | what it showed |
|---|---|---|
| 1 | 8 lawns of 1.0 mm | learners learned and left, but 3 of 4 never roamed; a both-states gate was added |
| 2 | 4 lawns of 1.5 mm | every gate passed on all 4 seeds; enrichment near zero; **registered** |
| 2 | 6 lawns of 1.5 mm | one seed failed the both-states gate; enrichment near zero |

**The panel:**

- **Arms:** the learner and its untrained floor.
- **Seeds:** 2101–2116, 3,000 episodes each; 32 runs, all of which succeeded.
- **Evaluation:** each run frozen over 30 held-out episodes.

## Results

### Gates

The learner eats 1.82 per episode against its floor's 0.66: **+1.16 [+1.04, +1.26]**. Every seed
leaves at least 32 times over its 30 evaluation episodes (median 58.5).

**Both states, by seed:**

| on-lawn roaming | seeds |
|---|---|
| 0% | 2101, 2104, 2106, 2109, 2111, 2112 |
| under 5% | 2102 (0.4%), 2107 (0.3%), 2114 (0.3%) |
| 5–95% (read) | 2103 (9%), 2105 (75%), 2108 (19%), 2110 (67%), 2113 (25%), 2115 (81%), 2116 (17%) |

Seven read seeds is fewer than 12, so by the registration the panel is **unreadable** and nothing is
classified.

### Described beside

| reading | learner | floor | real wild type |
|---|---|---|---|
| on-lawn roaming | 18% (mean) | 98% | 17.5% (classifier) |
| exits per on-lawn hour (median) | 7.7 | 49.0 | 0.21 |
| last-window enrichment (mean) | −0.007 | −0.081 | +0.467 |
| intake per held-out episode | 1.88 | 0.67 | — |
| complete bouts on lawns (dwelling / roaming) | 7 / 2, over 16 seeds | 0 / 0 | — |

**The seven two-state seeds' band enrichment:** +0.229, −0.034, −0.033, +0.157, −0.021, −0.119, −0.004.
The mean is +0.025, with sd 0.12.

**The dwelling-end statistic carries no information here.** Roaming after dwelling is near zero on
grazed and fresh spots alike (0.001 and 0.002), because so few learners roam at all.

## What this establishes, and what it does not

**Established:**

- **Learners leave food without roaming.** In this cell, MLP-PPO learns to feed and to move between
  lawns, and most of its seeds do it without ever entering the roaming state. They travel slowly at
  dwelling speed, as the first pilot's learners did on smaller lawns. A worm leaves from roaming,
  and these learners leave from what the classifier reads as dwelling.
- **They leave far more often than worms:** a median of 7.7 exits per on-lawn hour, against 0.21.
  Their lawns are small and their stays short, so few bouts of either state complete on a lawn.

**Not established:**

- **Whether leaving is coupled to roaming,** in either direction. The registered reading was not
  reached. The seven two-state seeds are fewer than registered, and their scatter is described, not
  read.
- **That this is a property of learners in general.** It is one learner, the MLP, in one cell
  family. The movement cost charges per millimetre regardless of speed, so moving slowly costs no
  more energy than moving fast. Only the basal decay paid while travelling rewards speed, and with
  lawns 2 mm apart it is small.

## Biological fidelity

- **Real worms change state to leave; these learners do not have to.** Real roaming is a high-speed,
  high-arousal state that precedes leaving over minutes (Scheer & Bargmann 2023). In this cell,
  travelling slowly costs no more energy per millimetre than travelling fast. The learners' slow
  travel is that economics, not a worm-like state.
- **A cell that rewarded speed for travel would need a cost of time:** a basal decay large enough
  that slow travel between lawns starves the worm, or lawns far enough apart that it does. That
  changes the cell's economics, and it would be a new registration.
- **The instrument reads speed and turning,** so "dwelling" here includes slow travel. D.1 said this
  of stopped worms; D.1b finds that it also covers slow travellers.

## Next

- **D.1b closes unmet-with-reason.** The registered reading was unreadable because most learners
  leave without ever roaming.
- **Destination:** a cell with a cost of time, where travelling slowly between lawns does not pay. It
  is recorded, not scheduled. B.3 does not need it, since B.3's mutant references are within-lawn
  roaming fractions.
- **Next in 8b is B.3 stage 0:** the connectome on D.1's cell, with no modulators, against its floor.

## Artefacts

**The registration and the panel:**

- [launch.md](supporting/088-patch-leaving/launch.md) and [preflight.json](supporting/088-patch-leaving/preflight.json): the registration and its gate preflight.
- [panel.json](supporting/088-patch-leaving/panel.json): the gates, the leaving reading (unreadable, with each seed's gate) and each run's readings.
- [per-run.csv](supporting/088-patch-leaving/per-run.csv): one row per evaluated run.

**The pilots:**

- [pilot1-states.json](supporting/088-patch-leaving/pilot1-states.json), [pilot2-4lawns-states.json](supporting/088-patch-leaving/pilot2-4lawns-states.json), [pilot2-6lawns-states.json](supporting/088-patch-leaving/pilot2-6lawns-states.json).

**The real reference:** [scheer2023_leaving.json](../../../data/roaming_dwelling/scheer2023_leaving.json), with its provenance.

**To reproduce:**

```bash
uv run python scripts/analysis/lawn_states.py --logs campaigns/d1b-panel/logs --cell leaving15x4 \
  --seeds $(seq 2101 2116) --episodes 30 --leaving-minimum 0.10 --min-exits 10 --min-seeds 12 \
  --out panel.json --csv per-run.csv
```

The campaign directories and session records are archived off-repo.
