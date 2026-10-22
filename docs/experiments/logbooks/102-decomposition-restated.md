# 102: Block V's Decomposition, Restated on the Corrected Gap Model (Phase 9 G.0-A)

**Status**: completed. No runs: this logbook restates the readings of Logbooks 074–092 in the light of
095–101. **Every gap-junction component of block V's advantage was read under the model's
degree-normalised count weights. On the thermal cell, where it has been re-read, it is absent under
every correction tried. What stands is the chemical wiring's part: on hard350 the sensory–motor
boundary, on thermal a lead below the minimum, and nowhere an interior-wiring advantage.**

| cell | block V's lead against the degree-preserving null (count model) | the gap-junction part | what is left once the gap junctions are held |
|---|---|---|---|
| hard350, PPO | +0.049 `auc_success` (074) | about half: −0.024 [−0.036, −0.012] (075) — **not re-read on a corrected model** | +0.025, all of it the sensory–motor boundary (079); the interior ±0.014 |
| thermal t35, PPO | +0.103 [+0.092, +0.113] (078) | about 84%: −0.0865 [−0.097, −0.075] (078) — **absent on the corrected model** (095, 099, 101) | +0.016 [+0.006, +0.027], below the minimum (078) |
| thermal t20, through the body | gap-only null only: +0.126 [+0.067, +0.178] (092) | all of it — **absent on the corrected model** (100) | not read |

**Date**: 2026-10-22.

**OpenSpec change**: none. This is the restatement step of the tracker's G.0-A item.

## Why a restatement

The project's model of a gap junction divided each pair's synapse count by `√(d_i d_j)`, where `d` is
the binary gap degree. G.0 found that the wild type's lead over its gap-only null followed that coupling
scale (Logbook 095). The corrections since then all remove it on the thermal cell:

| gap model, thermal t35, Emmons 2024 | wild type − gap-only null (80% CI) | logbook |
|---|---|---|
| count weights, degree normaliser (the original) | +0.104 [+0.087, +0.123] | 095 |
| unit weights | +0.003 [−0.012, +0.018] | 095 |
| log-compressed counts | +0.003 [−0.009, +0.016] | 099 |
| strength normaliser (counts kept) | −0.009 [−0.026, +0.009] | 101 |
| ohmic coupling ×0.03 | +0.021 [+0.007, +0.035], unresolved and below the minimum | 101 |
| log-compressed counts, through the body (t20) | −0.022 [−0.045, +0.005] | 100 |

The strength normaliser keeps every count and its relative weight within each neuron, and changes only
the normaliser. That it removes the lead as completely as discarding the counts places the artefact in
the normaliser. Every reading in Logbooks 074–092 that involves a rewired gap layer was taken under it,
so each is restated below for what it still says.

## The readings, restated

**074 — the chemical-only null (hard350).** The wild type led a null that rewires the chemical graph and
holds the wild type's gap junctions and autapses by +0.022. In that comparison both arms carry the same
gap matrix, so the artefact is common to them. **It stands as read**, as a chemical-wiring lead under the
original gap model. 079 then located all of it at the sensory–motor boundary.

**075 — the gap-only split (hard350).** Holding the null's gap junctions at the wild type's moved the
gap by −0.024: about half of block V's lead on hard350. This component compares two gap placements under
the degree normaliser, the comparison now shown to be an artefact on the thermal cell. **It was not
re-read on a corrected model on hard350.** It is therefore not cited as a gap-placement effect. The
evidence of 095–101 is that it would not survive, but on this cell that is an inference, not a reading.

**078 — the thermal split.** The gap junctions carried about 84% of block V's thermal lead (−0.0865).
**This component is the artefact**: the same contrast on the corrected model reads nothing (099, 101). The
remaining +0.016, against a null with the wild type's gap junctions, involves no gap rewiring. It
**stands as read**: a chemical-wiring and autapse lead, significant but below the minimum.

**079 — the boundary null (hard350).** Against a null that holds the gap junctions, the autapses and
every chemical edge at the sensory–motor boundary, the wild type has no lead (−0.003 [−0.010, +0.003]).
Both arms carry the wild type's gap matrix. **It stands as read**: on hard350 the chemical part of block
V's advantage is the boundary, and no interior-wiring advantage is detectable. Its dependence on the gap
model was not tested. Since the gap layer is identical in both arms, the normaliser artefact does not
enter the contrast directly, though it sets the operating point both arms share.

**082 — plastic gap strengths (thermal t35, Cook 2019).** The gap-only null trailed by +0.096, and still
by +0.086 when both wirings could tune every gap strength. 082 read this as **placement**: not the
strength of each junction, but which neurons are coupled, or each neuron's total. **That reading is
withdrawn.** 082 showed that learning each pair's strength did not close the gap. It did not show that
the gap was about placement rather than the model's coupling scale. Both arms started from couplings set
by the degree normaliser, and over the run the learned multipliers moved the median pair by only about
25%. Under the corrected model there is no lead to attribute.

**092 — through the body (thermal t20, Cook 2019).** The gap-only null trailed by +0.126. Logbook 094
read this as the first wiring advantage to survive embodiment. **That reading is withdrawn.** Re-run on
log-compressed counts on Emmons 2024 with 128 seeds, there is no lead, early or late (100). The
substrate also changed, Cook to Emmons; Logbook 100 states that limit. The behavioural findings of 092
and 093 that do not depend on the wiring contrast stand: the two wirings crawl alike, and the weathervane
is learned in every arm.

## What Phase 8's headline becomes

Logbook 094 closed Phase 8 as *the connectome's measurable wiring advantage is gap-junction placement,
and it survives the body*. Restated:

> On hard350 under PPO, the wild-type connectome learns faster than its degree-preserving null. About
> half of that lead came from comparing two gap placements under a gap model whose degree normaliser
> lets a few high-count pairs couple far above 1. That comparison is an artefact wherever it has been
> re-read. The other half is the chemical wiring at the sensory–motor boundary. On the thermal cell, the
> chemical part is a lead below the minimum. **No interior-wiring advantage is detectable on either
> cell, and no wiring advantage of any kind is detectable through the body on the corrected gap model.**

The phase's STOP condition and its statement (Logbooks 100 and 101) follow from this.

## What remains open, and is not chased here

- **The boundary lead on hard350** (079) was read only on the original gap model. It is the project's
  one remaining chemical-wiring effect at the minimum's scale, and its re-reading on a corrected gap
  model is a candidate for a later phase.
- **075's gap component on hard350** is uncited rather than re-read.
- **The ohmic residue** (101) is not chased, by the maintainer's decision of 2026-10-21.

## Annotations

The headers of Logbooks 075, 078, 082, 092 and 094 carry a dated note pointing here. Nothing in their
registered readings is changed: they read what they read, under the model they read it on.
