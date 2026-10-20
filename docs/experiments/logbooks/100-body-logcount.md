# 100: Through the Body Too, the Corrected Gap Model Shows No Wiring Effect, and the STOP Condition Is Met (Phase 9 G.0-A)

**Status**: completed. **`no_wiring_effect`.** Through the kinematic body, on log-compressed gap
counts, the wild type's lead over the gap-only null over episodes 1–3,000 is −0.022, with an 80%
interval of [−0.045, +0.005]. The interval includes zero and lies inside the ±0.0577 minimum.

| reading, 128 seeds | value (80% CI) | q | state | verdict |
|---|---|---|---|---|
| `gap(log, body)`, `auc_success` over episodes 1–3,000, wild type − null | **−0.0218** [−0.0454, +0.0047] | 0.35 | `no_move` | **`no_wiring_effect`** |

With the point worm's `no_wiring_effect` ([Logbook 099](099-logcount-contrast.md)), **the phase's
STOP condition is met as registered**: the corrected model shows no wiring effect on either cell under
PPO.

The gap-placement advantage the project carried from Logbook 082 through 092 belonged to the
count-scaled electrical-synapse model. Under log-compressed counts it does not appear on the point worm
or through the body.

**Date**: 2026-10-20.

**OpenSpec change**: `add-body-logcount-contrast`.

**Pre-registration**: [supporting/100-body-logcount/launch.md](supporting/100-body-logcount/launch.md),
committed at 39140a20 before any panel run. The maintainer chose the hybrid horizon (6,000 episodes,
with the verdict on episodes 1–3,000), 128 seeds, floors on 16 seeds, and no count-weight arm, on
2026-10-19 before registration.

## The question

On log-compressed gap counts, the point worm shows no wiring effect (Logbook 099). The STOP condition
names a second cell: 092's body cell, where the count model's wild type led by +0.126 on Cook 2019.
**Through the body, on the corrected model, does the wild type still learn faster than its gap-only
null?**

## Method

**The cell.** 092's thermal body cell at target 20, through the kinematic body, on Emmons 2024 (D23)
with `gap_weight_mode: log_count`, 6,000 episodes. The arms are the wild type and the gap-only null,
learning on seeds 4001–4128 and frozen on 4001–4016. All 288 runs exited 0. The panel ran from 02:26 on
19 October to 10:33 on 20 October, 32.1 hours, inside the registered 24–34.

**The reading.** The registered harness `body_logcount_contrast.py score` was run from the panel's
worktree at 39140a20. It reads the per-seed gap in mean success over episodes 1–3,000, paired by seed,
with an 80% bootstrap interval and the two-sided Wilcoxon as q, and classifies it by `mc.classify` at m
= 0.0577. The gates are read on the same window. Its output is
[panel.json](supporting/100-body-logcount/panel.json), with the per-seed values in
[per-seed.csv](supporting/100-body-logcount/per-seed.csv).

## Results

**The reading.** `no_move`: the 80% interval [−0.0454, +0.0047] includes zero and lies inside ±0.0577,
and q = 0.35. The verdict is **`no_wiring_effect`**. The wild type leads on 62 of 128 seeds.

**Gates on the window.** Readable.

- Plateaus over episodes 2,251–3,000: 65.0% for the wild type and 65.9% for the null.
- Both frozen floors are 0.0%. Learning beats frozen on the 16 floor seeds by [+69.6, +79.9] points for
  the wild type and [+50.4, +64.8] for the null.
- The highest plateau is 24 points under the bar.

**Sensitivity achieved.** The per-seed spread is σ = 0.219, against the 0.242 the panel was sized on.
The detectable effect for the two-sided test at 80% power is therefore about 0.054, just under the
minimum.

**Beside, never verdicts:**

- **The late window agrees with the verdict.**

  | window | wild type − null | 80% CI | q |
  |---|---|---|---|
  | episodes 3,001–6,000 | −0.012 | [−0.038, +0.014] | 0.87 |
  | all 6,000 episodes | −0.017 | [−0.037, +0.006] | 0.40 |

  092's late-half lead of +0.137 on the count model does not appear.

- **Plateaus at 6,000:** 76.3% for the wild type and 78.4% for the null.

- **Episodes to a 30% rolling success rate.** The wild type gets there 120 episodes *later*, with an
  interval of [−233, −3]. The interval excludes zero, but q = 0.49, so the bootstrap and the Wilcoxon
  disagree. If anything it points toward the null.

- **Collapse (D25)**, by 6,000 episodes:

  | arm | collapsed | rate (95% Wilson) | onsets |
  |---|---|---|---|
  | wild type | 10 of 128 | 7.8% [4.3%, 13.8%] | 2,080–5,695 |
  | null | 6 of 128 | 4.7% [2.2%, 9.8%] | 4,764–5,395 |

  Pooled, that is 16 of 256 (6.3%), against 092's 8 of 64 on the count model. One wild-type onset,
  at episode 2,080, falls inside the window: the first onset seen before 3,858 under the unmodified
  learner on this cell. That run stays in the reading, as registered. In 092 the null collapsed more;
  here the wild type does.

- **Two wild-type seeds did not learn on the window:**

  - 4041, at 0% against the null's 85%, the panel's largest gap at −0.58;
  - 4054, at 0% against 12%.

  Without them the mean is −0.016, still inside the minimum. Both stay in the reading.

- **The D24 audit**, committed before launch ([table.json](supporting/100-body-logcount/table.json)).
  The null carries a median of four sensor–motor gap pairs where the wild type has none, and no
  count–degree correlation. It learns as well as the wild type all the same.

## Interpretation

The project's one wiring effect to survive Phase 8 was gap-junction placement. It was:

- +0.096 on the point worm (082);
- +0.134 on Emmons (G.0's anchor);
- +0.126 through the body (092).

All three were read under the model's count-scaled additive gap coupling, in which a few high-count
pairs between low-degree neurons couple far above 1. G.0 showed the point-worm lead followed that scale
(Logbook 095). On the corrected model it is gone in both places:

| cell | wild type − null | 80% CI |
|---|---|---|
| point worm (Logbook 099) | +0.003 | [−0.009, +0.016] |
| body, episodes 1–3,000 (here) | −0.022 | [−0.045, +0.005] |
| body, episodes 3,001–6,000 (here) | −0.012 | [−0.038, +0.014] |

Each interval excludes a lead at the minimum, and the body's points, if anywhere, toward the null.

That is the STOP condition registered in roadmap § Phase 9 Go/No-Go. Its stated consequence:

- the phase's deliverable is the statement that the project's one surviving wiring effect belonged to
  the electrical-synapse model, with the controls that make that citable;
- the phase after 9 inherits a modelling question, how electrical synapses should enter a learned
  connectome model, rather than a mechanism question;
- it is "not a reason to withhold the instrument, the variability baseline or the ranking's controls
  individually".

**The controls that make it citable**, all committed:

- the operating-point screen, unit weights and the outlier nulls (095);
- the corrected model, chosen by learnability alone with its wiring gap never read (097);
- both cells' contrasts on that model, each registered before running, with gates on their own windows,
  achieved spreads that resolve the minimum, and D24 audits (099, 100);
- collapse handled by a window fixed in advance, with the late window and collapse incidence beside.

What this does not show:

- **That the substrate switch is ruled out through the body.** This panel ran on Emmons; 092 ran on
  Cook. The count model's body lead on Emmons was not measured, because the maintainer declined that
  arm. The point worm's count lead reproduced on Emmons (+0.134), so the substrate is not expected to be
  what removes it, but through the body that rests on the point worm.
- **That every gap model removes it.** Both corrections read here keep the coupling additive. Ohmic
  coupling was not chosen, because it learned more slowly (097), and its contrast was not read.
- **Anything about gap junctions in the animal.** The result is about how a model represents them.

## What follows

- **The STOP branch is taken** as registered. How Phase 9 proceeds is the maintainer's dated decision,
  to be recorded in the tracker and the roadmap. It covers which of M.2, M.3, R.1 and R.2 continue,
  G.1–G.3b's deferral and destination, and the shape of the 9a deliverable.
- **The decomposition** of Logbooks 074–092 is restated on the corrected model, as G.0-A's item
  requires.

## Data

- [supporting/100-body-logcount/launch.md](supporting/100-body-logcount/launch.md): the registration.
- [preflight.json](supporting/100-body-logcount/preflight.json): the gate preflight on the pilot,
  on the window.
- [table.json](supporting/100-body-logcount/table.json): D24's matched statistics on the 128 draws.
- [panel.json](supporting/100-body-logcount/panel.json) and
  [per-seed.csv](supporting/100-body-logcount/per-seed.csv): the reading, the gates, the late windows,
  episodes to 30%, and the collapse statistic for every run.
- Raw logs of the smoke, pilot and panel campaigns: archived off-repo.
