# 101: Strength-Normalised Gap Junctions Show No Wiring Effect; Ohmic Coupling Leaves a Small Lead Below the Minimum, Unresolved (Phase 9 G.0-B)

**Status**: completed. **Strength: `no_wiring_effect`. Ohmic ×0.03: `unresolved`.** Under the
registered map that pair is neither branch it names, so what follows was a dated decision. **The
maintainer decided on 2026-10-21 that the STOP statement stands, with ohmic's residue stated.**

| reading, 64 seeds, family of two | wild type − null, `auc_success` (80% CI) | q (BH) | state | verdict |
|---|---|---|---|---|
| `gap(strength)` | **−0.0089** [−0.0263, +0.0087] | 0.93 | `no_move` | **`no_wiring_effect`** |
| `gap(ohmic)` | **+0.0209** [+0.0067, +0.0345] | 0.12 | `unresolved` | **`unresolved`** |

**Strength normalisation**, the model that fixes the normaliser behind G.0's artefact, shows no wiring
effect.

**Ohmic coupling at its calibrated, weak conductance** shows a small wild-type lead.

- Its interval excludes zero, but the Wilcoxon does not reach significance once corrected (two-sided p
  0.06, q 0.12), so the classifier reads `unresolved`.
- The interval lies **wholly below the minimum**: its upper end, +0.035, is under 0.0577.

No model tested shows the lead the project carried, of 0.058 or more.

**Date**: 2026-10-21.

**OpenSpec change**: `add-gap-model-question`.

**Pre-registration**: [supporting/101-gap-model-question/launch.md](supporting/101-gap-model-question/launch.md),
committed at 4603358a before any panel run. The models, the cells and the decision to open the question
inside Phase 9 are the maintainer's, of 2026-10-20.

## The question

After the STOP condition was met on log-compressed counts (Logbooks 099 and 100), the maintainer opened
the modelling question inside Phase 9. **Under two principled gap models that change something other
than each pair's weight, does the wild type still learn faster than its gap-only null?**

The two models:

- **Strength normalisation** divides each pair by `√(s_i s_j)`, where `s` is the neuron's total gap
  weight. Every coupling is then at most 1.
- **Ohmic coupling at ×0.03** passes current in proportion to the potential difference, at the scale
  calibrated on untrained brains (Logbook 097).

## Method

The thermal cell at target 35 on Emmons 2024, 3,000 episodes, count weights. For each model, the wild
type and the gap-only null, each learning and frozen, on seeds 4201–4264: 512 runs, every one exiting 0.
The panel ran from 19:58 to 03:27, 7.5 hours, against the 7.3 estimated.

Before launch:

- **The strength probe passed**: an untrained action spread of 0.0374, against a floor of 0.0132.
- **Both models' gates read readable**: strength on its own pilot, ohmic on G.0-A's.

The registered harness `gap_model_question.py score` was run from the panel's worktree at 4603358a. It
corrects the two readings together (BH-FDR on the two-sided Wilcoxon) and classifies each at
m = 0.0577. Its output is [panel.json](supporting/101-gap-model-question/panel.json), with the per-seed
values in [per-seed.csv](supporting/101-gap-model-question/per-seed.csv).

## Results

**Gates.** Both models are readable on all 64 seeds.

| model | wild type plateau | null plateau | floors |
|---|---|---|---|
| strength | 66.7% | 68.3% | 0.0% |
| ohmic | 67.1% | 63.9% | 0.0% |

Neither model is near the 90% bar.

**The readings** (table above):

- **Strength: `no_wiring_effect`.** Its interval includes zero and lies inside ±0.0577. The wild type
  leads on 33 of 64 seeds.
- **Ohmic: `unresolved`.** The wild type leads on 37 of 64 seeds. The 80% interval excludes zero, but
  the corrected q, 0.12, does not reach 0.05. That is the shared classifier's band between the interval
  and the test, which the registration put at about 15% under a true null. Under the registered map,
  `unresolved` licenses nothing by itself.

**Sensitivity achieved.** The per-seed spread is 0.113 for strength and 0.088 for ohmic, against the
0.119 sized for. The detectable effect for the two-sided test at 80% power is about 0.040 and 0.031.

**Beside, never verdicts:**

- **Episodes to 30% success** follow the same pattern:
  - strength: the wild type is 53 episodes slower [−123, +11], q 0.84;
  - ohmic: the wild type is 83 episodes faster [+25, +144], q 0.16.
- **The two wild types learn alike.** Strength − ohmic is +0.005 [−0.012, +0.023].
- **The extremes** are single seeds where one learner barely took off, all of which stay in the
  reading:
  - strength: wild type 4234, at a 7.9% plateau (the gap −0.33);
  - ohmic: wild type 4237, at 3.5%;
  - ohmic: null 4238, at 0.4%.
- **The D24 audit** ([table.json](supporting/101-gap-model-question/table.json)):
  - Both nulls create about five sensor–motor gap pairs where the wild types have none.
  - The strength null re-normalises its hubs by much smaller strengths: ALA's total gap count is 34
    in the null's median draw, against 1,314 in the wild type.

## Interpretation

Across the four corrections now read on this cell, the wild type's lead over its gap-only null is:

| gap model | lead (80% CI) | source |
|---|---|---|
| count weights, `√(d_i d_j)` normaliser (the original) | +0.104 [+0.087, +0.123] | Logbook 095 |
| unit weights | +0.003 [−0.012, +0.018] | Logbook 095 |
| log-compressed counts | +0.003 [−0.009, +0.016] | Logbook 099 |
| strength normalisation | −0.009 [−0.026, +0.009] | here |
| ohmic coupling ×0.03 | +0.021 [+0.007, +0.035] | here |

The lead the project carried appears only under the original model, the one whose normaliser lets a few
high-count pairs couple near 90. Fixing the normaliser alone, which keeps every count and replaces only the degree-based denominator
with the total-strength one, removes it as completely as discarding the counts does. That localises the
artefact: **it was the normaliser**.

Ohmic coupling is the one model with a residue: a wild-type lead of about 0.02.

- It is a third of the minimum, and its interval excludes the minimum.
- It is not significant once the family is corrected.
- It is read at a weak conductance, where the median neuron keeps 93% of its own drive.

It may be a real small effect of placement under diffusive coupling. It may also be the shared
classifier's band under a true null. This panel cannot separate the two, and the registration licenses
no reading of it beyond `unresolved`.

What this does not show:

- that ohmic coupling at a stronger conductance shows nothing, or something; stronger scales failed the
  learnability probe (Logbook 097), so they cannot be read under this learner;
- anything through the body for these two models, since neither read `placement`.

## What follows

**Decided 2026-10-21 by the maintainer: the STOP statement stands, with the residue stated.** The
registered verdicts above are unchanged; the decision chooses what follows, not how they read. The
statement the phase ships:

> No gap model tested shows the wild type's gap-placement lead at the minimum. The lead the project
> carried from Logbook 082 to 092 came from the model's degree normaliser, which let a few high-count
> pairs couple far above 1. Fixing the normaliser alone removes it, as do unit weights and log counts.
> Under ohmic coupling at its calibrated, weak conductance, a wild-type lead below the minimum
> (+0.021 [+0.007, +0.035]) is unresolved.

Next, under G.0-A's item, the decomposition of Logbooks 074–092 is restated on the corrected model,
and then the 9a synthesis follows. The ohmic residue is not chased: at best a larger panel would read
`below_minimum`, which licenses nothing.

## Data

- [supporting/101-gap-model-question/launch.md](supporting/101-gap-model-question/launch.md): the
  registration.
- [probe.json](supporting/101-gap-model-question/probe.json): the strength model's untrained probe.
- [preflight.json](supporting/101-gap-model-question/preflight.json): both models' gates.
- [table.json](supporting/101-gap-model-question/table.json): D24, with the strength null's
  normalisers.
- [panel.json](supporting/101-gap-model-question/panel.json) and
  [per-seed.csv](supporting/101-gap-model-question/per-seed.csv): the readings and every seed.
- Raw logs of the smoke, pilot and panel campaigns: archived off-repo.
