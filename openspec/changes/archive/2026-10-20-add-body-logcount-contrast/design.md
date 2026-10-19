## Context

The point worm shows no wiring effect on log-compressed gap counts
([Logbook 099](../../../docs/experiments/logbooks/099-logcount-contrast.md)). The STOP condition (roadmap
§ Phase 9 Go/No-Go) needs the same on 092's body cell. 092's lead through the body was +0.126 over
6,000 episodes, on Cook 2019 with count weights and 32 seeds.

From its archived logs, per seed:

| window | mean lead | sd |
|---|---|---|
| episodes 1–3,000 | +0.115 | 0.242 |
| episodes 3,001–6,000 | +0.137 | 0.291 |

Late collapse follows the training stream, about one run in eight by 6,000 episodes
([Logbook 098](../../../docs/experiments/logbooks/098-collapse-remedies.md)). Under the unmodified
learner every observed onset is after episode 3,858: 092's eight, re-run in M.1(a), and the placebo's two
(3,913 and 5,625). Under the remedies, two came earlier: 2,396 under weight clipping and 3,668 under the
KL stop. The
maintainer's decisions of 2026-10-19:

- the hybrid horizon: run to 6,000 episodes and read the verdict on the first 3,000;
- 128 seeds;
- frozen floors on 16 seeds;
- no count-weight arm.

## Goals / Non-Goals

**Goals**

- Read the corrected model's wiring contrast through the body at the minimum, under a verdict map that
  names the joint outcome with the point worm.
- Report the late window and collapse incidence beside the verdict, so that a null result is not open to
  the objection that the effect is late.

**Non-goals**

- The chemical-only null (092's beside reading).
- A count-weight arm on Emmons.
- Kinematics.
- Any mechanism reading.
- The decomposition restatement, which follows this panel's reading.

## Decisions

### Decision A: The cell, the substrate and the configs

**The cell** is 092's thermal body cell:

- target 20, 500 steps an episode;
- `body_model: kinematic`, `allow_reversal: true`, `action_space: body_drive`;
- `entropy_coef` 0.004;
- settling depth 4;
- 6,000 episodes.

**Two keys change**, and nothing else:

- `connectome_source: emmons_2024_hermaphrodite`. D23 puts the whole phase on Emmons, and the point
  worm's contrast was read on Emmons.
- `gap_weight_mode: log_count`. This is the corrected model (Logbook 097).

The configs are the wild type and the gap-only null, each learning and frozen. They come from a new
generator, `scripts/campaigns/generate_body_logcount_configs.py`, which derives each from 092's committed
config as a stated delta and never overwrites. A loader test re-reads them, and the smokes confirm that
the body builds and steps on Emmons.

**Seeds.**

- Panel: 4001–4128 for the learning arms, and 4001–4016 for the frozen arms.
- Gate pilot: 9895–9898.
- Smokes: 9899–9900.

All are new bands, disjoint from every committed band.

**288 runs**: 256 learning and 32 frozen.

### Decision B: The reading, its window, and the verdict map, fixed before any run

**The reading.** `gap(log, body)` is the paired difference by seed, wild type − null, in `auc_success`
over **episodes 1–3,000**: each run's mean success over its first 3,000 episodes, from its per-episode
series (`collapse.success_series` and `collapse.auc_to`). It uses
`wp.paired_seed_wilcoxon_bootstrap`, with an 80% percentile interval and the two-sided Wilcoxon p as q
(a family of one), and is classified by `mc.classify` at **m = 0.0577**. The minimum was fixed for
`auc_success` over 6,000 episodes. It carries to the 3,000-episode window because 092's lead was nearly
the same over both: +0.115 to 3,000 and +0.126 to 6,000. It is the minimum 092 read
the body at, and the one the point worm used. The maintainer declined to re-anchor on the count model's
lead, because that lead is the artefact.

The window is fixed here and never moved. It ends before every collapse onset observed under the
unmodified learner on this cell, the earliest at episode 3,858. A collapse can still begin inside it, as
one did at 2,396 under weight clipping. Such a run stays in the reading, and the collapse statistic
reports its onset.

| state | verdict | with the point worm's `no_wiring_effect` |
|---|---|---|
| gates fail | **`unreadable`** | a dated decision |
| `move_wt` | **`placement`** | whether G.1–G.3b move to the body is a dated decision |
| `below`, either direction | **`below_minimum`** | a dated decision, with the direction recorded |
| `no_move` | **`no_wiring_effect`** | **the STOP branch is taken** |
| `move_null` | **`reversed`** | a dated decision |
| `unresolved` | **`unresolved`** | a dated decision, never a re-read under another rule or window |

**Beside, never verdicts:**

- the same gap over episodes 1–6,000 and over 3,001–6,000;
- D25's collapse statistic per arm, from `collapse.panel`: incidence with a Wilson interval, Kaplan–Meier
  time to collapse censored at 6,000, dips, and 092's own rule;
- each arm's plateau at 3,000 and at 6,000;
- `episodes_to_30pct_success`;
- the D24 audit.

If the late window points elsewhere than the verdict, the verdict stands and the disagreement is
reported.

### Decision C: Sizing and the operating characteristic

The spread is 092's per-seed sd over episodes 1–3,000, 0.242, measured on the count model and on Cook.
On log counts the point worm's spread fell by a third (0.119 to 0.079), but the panel is sized on the
committed upper value. The body's per-seed gaps are heavier-tailed than normal: 092 has learners that
never took off, such as seed 2301 at −0.15. The Wilcoxon q and the percentile bootstrap do not assume
normality, so the simulated table is a guide, not a guarantee.

At 128 seeds the standard error is about 0.021. The detectable effect for the two-sided test at 80%
power is 0.059, about the minimum.

The operating characteristic is simulated with the registered instrument: 600 draws of 128 normal seed
gaps at σ = 0.24 per row, each scored by `wp.paired_seed_wilcoxon_bootstrap` and classified by
`mc.classify` at m.

| true effect | `placement` | `below_minimum` | `no_wiring_effect` | `unresolved` | `reversed` |
|---|---|---|---|---|---|
| 0 | 0 | 0.05 (0.02 wild type, 0.03 null) | 0.79 | 0.15 | 0.01 |
| +0.02 | 0.04 | 0.12 | 0.60 | 0.23 | 0 |
| +0.0577, the minimum | 0.50 | 0.23 | 0.07 | 0.20 | 0 |
| +0.08 | 0.85 | 0.11 | 0 | 0.04 | 0 |
| +0.115, 092's early lead | 1.00 | 0 | 0 | 0 | 0 |

`unresolved` takes about 15% at a true zero, whatever the seed count. This is the shared classifier's
band, kept as on the point worm.

At 64 seeds a true null would read `no_wiring_effect` only about half the time. That is why the
maintainer chose 128.

### Decision D: Gates on the verdict's window

**The plateau on the window** is the instrument's final-quarter tail (`t7.plateau_tail`) applied to
episodes 1–3,000: mean success over episodes 2,251–3,000. The plateau over 4,501–6,000 is reported beside
it.

**The gates**, each on that window:

- **Floor:** each learning arm beats its wiring's frozen floor. That is the paired interval of learning −
  frozen plateau over the 16 floor seeds (4001–4016), and it must lie above zero.
- **Saturation:** the two learning arms' mean plateaus over all 128 seeds must not both be at 90% or
  more.

The frozen floors run on 16 seeds because every body floor in 092 was 0.0%. Any learning run whose
plateau on the window is 0% is listed beside the reading and stays in it.

**Before launch**, `body_logcount_contrast.py preflight` evaluates these gates on the pilot (seeds
9895–9898, all four arms at 6,000 episodes) on the same 3,000-episode window. Any level that fails its
floor, saturates, sits within 5 points of the bar, or has no evidence blocks the launch until it is
resolved. The pilot's gap is never read: on four seeds it licenses nothing. Its plateaus at 6,000 are
also reported, because 092's pilot sat near the bar there (86.1%).

### Decision E: D24 before launch

`gap_operating_point.matched_statistics` is computed for the two body configs, the wild type and the
null on each of the 128 panel seeds, and committed as `table.json` before any run. The gap matrix does
not depend on the body, so this should reproduce Logbook 099's pattern on these draws: five
sensor–motor gap pairs in the null where the wild type has none, and no count–degree correlation. It is
reported beside the verdict.

### Decision F: The harness

`scripts/analysis/body_logcount_contrast.py` exposes a `STEMS` table and four subcommands:

- `plan`;
- `preflight`, the gates on the pilot on the verdict's window;
- `table`;
- `score`, which produces the manifest, completeness, the windowed reading and gates, the verdict, the
  beside readings, the collapse statistic, a per-seed CSV and the panel JSON.

Its manifest uses Logbook 099's check, imported from `logcount_contrast.py` rather than copied:

- only runs whose `.exit` is 0 are read;
- a run must record the config its name claims, and one that does not is refused;
- every run, learning and frozen, must have all 6,000 episodes;
- a missing run is refused, never read as a failure.

The tests run on synthetic success series and manifests:

- the windowed reading (episodes after 3,000 cannot change it);
- every verdict row;
- the gates on the window;
- every refusal;
- the seeds' disjointness from committed bands.

## Risks / Trade-offs

- **The substrate differs from 092's.** 092 ran on Cook; this panel runs on Emmons, following D23. The
  count model's body lead on Emmons is not measured here, because the maintainer declined that arm.
  The point worm's count-model lead reproduced on Emmons (+0.134 in G.0's anchor), so the substrate is
  not expected to be what removes a lead. The logbook states the limit.
- **The plateau at 3,000 is mid-learning.** Body learners are still climbing at 3,000. The gate asks only
  that they beat a 0% floor, which 092's learners did well before then (a median of 1,397 episodes to
  30%).
- **Cost.** About 36 hours with the pilot. Body runs at 6,000 episodes are the most expensive in the
  project, and the late window is what the extra 3,000 episodes buy.
- **The null's shortcut.** The log-count null carries about five sensor–motor gap pairs, as every
  gap-only null has. The audit reports this beside the verdict.
