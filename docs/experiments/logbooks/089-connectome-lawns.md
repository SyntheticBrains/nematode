# 089: The Connectome Dwells on Lawns Where Its Untrained Policy Never Does, Giving B.3 Its Baseline (Phase 8b B.3 stage 0)

**Status**: completed — **`dwells`.** On D.1's patchy-lawn cell, the connectome PPO brain (Emmons 2024,
settling to depth 4, no modulators) spends far more of its on-lawn time dwelling than its untrained
policy does.

| registered reading, seeds 2001–2019 | mean | 80% CI | q | state |
|---|---|---|---|---|
| connectome − untrained floor, on-lawn dwelling share | **+0.514** | [+0.398, +0.626] | 1.6 × 10⁻⁴ | `move_wt` → **dwells** |

The interval lies above the registered minimum of 0.278. Thirteen of 19 seeds dwell well above their
floor, and the other six stay near it, roaming throughout.

**Beside it, never read as verdicts:**

- **Against D.1's MLP,** paired over seeds 2001–2016: the connectome dwells slightly less, −0.15
  [−0.30, +0.01]. The interval reaches zero.
- **Fewer seeds take complete bouts of both states:** 4 of 19, against the MLP's 9 of 16.

**This is the baseline B.3's serotonin and PDF knockouts will be read against.**

**Date**: 2026-10-11.

**OpenSpec change**: `add-b3-stage0`.

**Pre-registration**: [supporting/089-connectome-lawns/launch.md](supporting/089-connectome-lawns/launch.md),
committed at efed6859 before any panel run, after the entropy pilot and the spec review.

## Objective

B.3 puts serotonin and PDF on the connectome and reads their knockouts against roaming and dwelling.
D.1 (Logbook 087) validated that readout with the MLP only. The phase protocol asks two things before
any modulator is ablated:

- **A positive control for the new component.** The connectome must show it can take the states at
  all.
- **A baseline at the new operating point.** It must be re-established before anything is ablated.

Stage 0 does both: the connectome on D.1's cell unchanged, against its untrained floor.

## Method

**The learner.** The connectome PPO brain on Emmons 2024, with gap junctions and the strict chemical
mask, reversal, klinotaxis sensing and settling to depth 4.

- **No modulators, and no satiety input.** Satiety reaches the connectome only through B.3's
  modulators.
- **Everything else is D.1's cell:**
  - 4 lawns of 2.5 mm, with the worm starting on one;
  - intake 2% of a cell per step, and a movement cost of 0.013 per mm;
  - turns up to π per step;
  - 720-step episodes, 3,000 per run;
  - no state-shaping reward.

**The entropy calibration.** A pilot on seeds 9301–9304 tried two levels:

| entropy | intake, learner against floor | gate | dwelling over floor, per seed |
|---|---|---|---|
| **0.005** | 3.15 against 1.71 | passes, +1.44 [+1.12, +1.77] | −0.01, +0.01, +0.93, +0.73 |
| 0.05 | 1.85 against 1.71 | passes, +0.14 [+0.04, +0.24] | none |

0.005 was registered, by the rule fixed in the design. At 0.05 the connectome barely learned and never
dwelt, as D.1's MLP did at that level.

**Sizing.** The pilot's spread was bimodal: seeds either dwelt or roamed throughout. Its +0.417
reference set the minimum at 0.278 and the panel at **19 seeds**, where the MDE is 0.277. Seeds
2001–2016 are D.1's MLP panel, so the two learners pair by seed.

**The panel.** The connectome and its floor, 38 runs, all succeeded. Each was evaluated frozen over 30
held-out episodes with D.1's instrument.

## Results

### Gate

The learner eats 3.34 per episode against its floor's 1.70: **+1.64 [+1.53, +1.75]**.

### The registered reading

**+0.514 [+0.398, +0.626]**, q = 1.6 × 10⁻⁴. The learner's on-lawn dwelling share is 0.53, against its
floor's 0.015.

| seed | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 |
|---|---|---|---|---|---|---|---|---|---|---|
| learner − floor | −0.00 | +0.45 | +0.28 | +0.99 | +0.99 | +0.97 | +0.03 | +0.52 | +0.43 | +0.78 |

| seed | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 |
|---|---|---|---|---|---|---|---|---|---|
| learner − floor | −0.01 | +0.72 | +0.98 | +0.11 | −0.01 | −0.02 | +0.97 | +0.62 | +0.99 |

The verdict is **dwells**. Achieved spread: sd 0.41 and MDE 0.23, against the pilot's 0.49 and 0.28.
Reported beside the minimum, never used to re-read it.

### Described beside

| reading | connectome | D.1's MLP (Logbook 087) | connectome's floor |
|---|---|---|---|
| on-lawn dwelling share, mean | 0.53 | 0.61 | 0.015 |
| seeds dwelling more than 0.278 above their floor | 13 / 19 | 14 / 16 | — |
| seeds with complete bouts of both states | 4 / 19 | 9 / 16 | 0 |
| median dwelling bout | 160 s | 354 s | — |
| median roaming bout | 188 s | 170 s | — |
| intake per held-out episode | 3.54 | 3.48 | 1.67 |
| roaming where grazed − where fresh | −0.12 (3 / 18 positive) | −0.18 (2 / 16 positive) | — |

- **Paired with the MLP.** Over seeds 2001–2016, the connectome minus the MLP dwells **−0.15 \[−0.30,
  +0.01\]** (one-sided p = 0.89 for the connectome dwelling more). The two learners ran on the same cell
  at the same entropy, so this is a comparison of learners, not of settings.
- **Depletion.** The grazed-against-fresh reading runs the wrong way, as in D.1, and for the same
  reason: dwelling is what grazes a cell. The dwelling-end statistic is near zero on most seeds,
  because few seeds switch states.
- **Action noise.** The learned speed noise is small, with a median log_std of −4.7.

## What this establishes, and what it does not

**Established:**

- **The connectome dwells.** On D.1's lawn cell, with no modulators, it takes the dwelling state on
  food that its untrained policy never takes, as D.1's MLP did. The readout reads it, so B.3 has its
  baseline.
- **Its entropy matters as the MLP's did.** At 0.05 it barely learns and never dwells. At 0.005 it
  does both.

**Not established:**

- **That the connectome dwells less than the MLP.** The paired interval reaches zero.
- **That its states are worm-like in timing or depletion.** Complete bouts of both states are rare,
  and the depletion reading is confounded, as in D.1.
- **Anything about the wiring.** Stage 0 has one wiring and no null. A wiring contrast on this cell
  would re-ask C.1e's unresolved question before B.3 needs it.

## Biological fidelity

- **The states here are an RL forager's.** Dwelling on this cell is the learner's best response to its
  economics: movement costs energy, and food depletes over minutes. In the worm, serotonin and PDF set
  the states (Flavell et al. 2013), and that is what B.3 adds.
- **The spread is individual.** Six seeds roam throughout, and thirteen mostly dwell. Real worms differ
  individually in roaming too (Stern et al. 2017), but nothing here says the variation has the same
  cause.
- **There is no satiety input,** by design.
- **The instrument passes its gate close to the bar and calls more roaming than its source's authors,
  as D.1 said.** Shares are compared across learners read by the same instrument, never against the
  authors' fractions.

## Next

- **B.3 stage 0 is met.** The baseline exists.

- **B.3 proper:**

  - **the modulators:** serotonin and PDF as global concentrations, with release from NSM and the
    PDF-1 neurons, and gating at the cited receptor cells (MOD-1 in AIY, RIF and ASI; PDFR-1 as
    cited);
  - **the primary reading:** acute knockouts of the trained modulator connectome, graded on Ji et al.
    2021's directions (*tph-1* roams more, *pdfr-1* less);
  - **beside it:** raised mutants, and the satiety-to-serotonin link.

  It starts with planning: the receptor sites' sources, the release model, and how the field gates
  the network.

## Artefacts

**The registration and the panel:**

- [launch.md](supporting/089-connectome-lawns/launch.md) and [preflight.json](supporting/089-connectome-lawns/preflight.json): the registration and its gate preflight.
- [panel.json](supporting/089-connectome-lawns/panel.json): the gate, the reading and its verdict, the pairing with D.1's MLP, and each run's readings.
- [per-run.csv](supporting/089-connectome-lawns/per-run.csv): one row per evaluated run.

**The entropy pilot:** [pilot-e005-states.json](supporting/089-connectome-lawns/pilot-e005-states.json), [pilot-e05-states.json](supporting/089-connectome-lawns/pilot-e05-states.json).

**To reproduce:**

```bash
uv run python scripts/analysis/lawn_states.py --logs campaigns/b3s0-panel/logs \
  --arms connectome_e005 connectome_e005_frozen --learner-arm connectome_e005 \
  --floor-arm connectome_e005_frozen --minimum 0.278 --seeds $(seq 2001 2019) --episodes 30 \
  --compare-mlp docs/experiments/logbooks/supporting/087-patchy-lawns/panel.json \
  --out panel.json --csv per-run.csv
```

The campaign directories and session records are archived off-repo.
