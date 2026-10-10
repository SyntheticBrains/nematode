# 084: Through the Body, the Wiring Contrast Is Small Against Its Spread; the Panel Is Fixed at 64 Seeds (Phase 8b C.1e pilot)

**Status**: completed — **calibration.** The registered pilot for C.1e's wiring contrast through the
kinematic body fixes the panel's minimum at its floor, **0.0367 `auc_success`**, and its size at the
cap, **64 seeds**, where it can detect 0.050. Both wirings learn on every seed; no gate fails.

| pilot, PPO through the body, seeds 1701–1716 | mean plateau | floor |
|---|---|---|
| wild type | 57.3% (31.1–81.6) | 0.0% |
| chemical-only null | 54.6% (18.0–87.9) | 0.0% |

Described, never read as a verdict: the wild type leads by **+0.023 `auc_success`** [−0.025, +0.071],
p = 0.32, ahead on 9 of 16 seeds, with a per-seed spread of 0.161. Every wild-type seed reaches
competence against 14 of 16 null seeds (exact McNemar p = 0.5).

**Before the pilot**, the connectome could not learn through the body at all. A drive gain vector and a
dimension-matched entropy bonus fixed that. A frozen-wiring learner, the body's stand-in for the reading
learner, failed its floor and left C.1e.

**Date**: 2026-10-09.

**OpenSpec change**: `add-body-wiring-contrast` (in progress), with `add-body-drive-gain` before it.

**Pre-registration**: [supporting/084-body-wiring-pilot/launch.md](supporting/084-body-wiring-pilot/launch.md),
committed before any scored run, after the spec review and a gate preflight.

## Objective

C.1e asks whether the wild-type wiring helps a connectome forage through the body, against nulls that
differ from it only in the chemical wiring. Nothing committed measured a wiring effect on the body cell,
and the protocol requires the minimum to come from data on the same cell. This pilot measures it, and
its committed numbers size the panel.

## Method

**The cell.** hard350's food layout at 500 steps (C.1d's fallback), through the frozen kinematic body
(steering gain 2, one-step reversals, wave floor 0.25), on Emmons 2024 with reversal on, settling
dynamics at depth 4, 3,000 episodes, `entropy_coef` 0.004. A new reference frame: no delta against
029, block V or any point-worm result.

**The arms.** The wild type and the chemical-only null (D21's primary), each learning under PPO and
frozen. Seeds 1701–1716, 64 runs, all succeeded in 4.4 hours.

**What the pilot fixes.** The reference effect is the paired wild-type-minus-null `auc_success` mean.
The minimum is 2/3 of it, never below a judged floor of 0.0367 carried from the point worm. The panel's
seed count is the smallest n whose MDE (`2.487 × sd / √n`) reaches the minimum, between 16 and 64. The
gates are both learning arms beating their floors and the level not saturating.

## Before the pilot: four probes

All on seeds 1601–1604 at this cell ([probes.json](supporting/084-body-wiring-pilot/probes.json)):

| probe | change | wild type | chemical-only null |
|---|---|---|---|
| 1 | C.1 as merged | 0, 0, 0, 0 | 0, 0, 0, 0 |
| 2 | + a learnable gain on each drive output (D18) | 0, 0, 0, 0 | 0, 0, 62.1, 52.7 |
| 3 | + entropy bonus matched to the action width | 58.4, 76.1, 48.7, 83.5 | 27.7, 80.7, 18.8, 82.0 |
| 4 | + frozen wiring | 0, 0, 0, 0 | 0, 0, 25.3, 0 |

- **Probe 1.** PPO's entropy bonus drove every learner's exploration noise to its cap, σ = 7.4. The MLP
  answered with an unbounded mean. The connectome could not: its mean is settled rates through a
  normalised anatomical map, bounded near ±1, so its drive was noise.
- **Probe 2.** D18 specified a learnable gain vector over muscle groups, and C.1 had left it out. With
  it, learning appeared exactly where the gains outgrew the noise.
- **Probe 3.** The entropy bonus is summed over action dimensions, so 0.05, set for the point worm's two
  action numbers, pushed 12.5 times harder on the body's 25. At 0.004, the same per-dimension pressure,
  every seed of both wirings learned.
- **Probe 4.** PPO training only the drive gains and sensor gains over a fixed wiring did not learn. The
  gate preflight read it `fails_floor`, so the reading-learner half of C.1e closes
  unreachable-with-reason: within 3,000 episodes at these settings, the wiring read but not written
  does not carry foraging through the body.

## Results

### Gates

Both learning arms beat their 0% floors on every seed; neither approaches the 90% saturation bar. The
level is readable.

### What the pilot fixes

| quantity | value |
|---|---|
| reference effect, `auc_success` | +0.0229 |
| 2/3 of the reference | 0.0153 |
| **minimum** | **0.0367**, the floor (2/3 of the reference is below it) |
| the floor's share of the wild type's mean `auc_success` (0.386) | 9.5% |
| per-seed sd of the gap | 0.161 |
| **panel seeds** | **64, capped**: the MDE at 64 is **0.050**, above the minimum |

### Described beside

- `auc_success`: +0.023 [−0.025, +0.071], wild type ahead on 9 of 16 seeds.
- Episodes to 30% success: −87 [−281, +120].
- Competent seeds: wild type 16 of 16, null 14 of 16; McNemar p = 0.5.

The pilot is a calibration. These are not readings of the wiring.

### Cost

About 70 minutes per learning run and 60 per frozen run at 16 workers.

## What it means for the panel

**The spread, not the seed count, limits the panel.** Seeds differ widely in when they begin to learn,
and that drives per-seed gaps between −0.22 and +0.32. At the 64-seed cap the panel detects 0.050, so an
effect between the 0.0367 minimum and 0.050 will read `unresolved`, by the protocol's rule. Its likely
outcomes are:

- if the true effect is near the pilot's +0.023, an interval around [0.00, 0.05], spanning the minimum:
  `unresolved`;
- if it is 0.05 or more, a `move_wt`;
- a `no_move` only if the panel's mean lands within about ±0.011 of zero.

**The panel is therefore cut to what can be read.** It runs the primary contrast alone: the wild type
against the chemical-only null, learning and frozen, with the MLP beside, on 64 fresh seeds (about 20
hours). **The boundary-preserving null runs afterwards, only if the primary reads `move_wt`.** An
interior-wiring claim needs a wiring effect to locate. Without one, the boundary null's 128 runs would
answer a question that did not arise. The registered design's four-wiring panel would have cost about
29 hours for that. The panel is registered with this design before any panel run (Logbook 085's launch
record).

## What this establishes, and what it does not

- **Established**: the connectome learns the 500-step body cell under PPO on every seed, given the drive
  gain vector and a dimension-matched entropy bonus. Without either, it does not.
- **Established**: PPO training only the gains over a fixed wiring does not learn the cell within 3,000
  episodes. The reading learner has no body-drive equivalent here.
- **Not established**: any wiring effect through the body. The pilot's +0.023 is a calibration figure
  with an interval spanning zero; the panel, on fresh seeds, makes the reading.

## Biological fidelity

The gain vector restores D18's design: the map stays anatomy, and learning sets how strongly each
muscle group is driven, the same 25 gains for every wiring. Matching the entropy bonus to the action's
width is a learning setting, not a biological one, and applies to every body-drive arm alike. The
conditions Logbook 083 recorded on the body (short reversals only, an unmeasured wave floor and steering
gain, a speed at the band's floor) carry over unchanged.

## Artefacts

- [launch.md](supporting/084-body-wiring-pilot/launch.md): the registration.
- [probes.json](supporting/084-body-wiring-pilot/probes.json),
  [probe-preflight.json](supporting/084-body-wiring-pilot/probe-preflight.json): the four probes and the
  gate preflight on probe 3.
- [pilot.json](supporting/084-body-wiring-pilot/pilot.json): gates, gaps, the reference, the minimum, the
  panel's size, the competence frequency.
- [per-seed.csv](supporting/084-body-wiring-pilot/per-seed.csv): each seed's plateaus, floors and gaps.
- Reproduce: `scripts/analysis/body_wiring.py pilot --logs campaigns/c1e-pilot/logs --out-dir <dir> --out pilot.json --csv per-seed.csv`. The probe and pilot campaign directories are archived off-repo.
