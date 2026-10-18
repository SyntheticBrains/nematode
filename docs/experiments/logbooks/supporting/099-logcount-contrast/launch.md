# G.0-A — the corrected model's point-worm contrast on log-compressed counts: registration and launch

**Registered 2026-10-18, before any panel run.** Change: `add-logcount-gap-contrast`. Tracker
`phase9-tracking` G.0-A; roadmap v4.5 § Phase 9 (G.0's entry, as amended 2026-10-17), D23 and D24.

## The question

G.0 found that the wild type's lead over the gap-only null on the thermal cell rode on count-scaled gap
coupling ([Logbook 095](../../095-gap-operating-point.md)). G.0-A then chose log-compressed counts as
the corrected model, by learnability alone and without reading the wiring gap
([Logbook 097](../../097-gap-substrate.md)). **On the corrected model, does the wild type still learn
faster than its gap-only null?** The only reading so far is G.0's 16-seed description of its log-count
level: +0.012 [−0.023, +0.046].

## The arms

The thermal cell at target 35 on Emmons 2024, with block V's point otherwise: edge-order draw, pooled
readout, depth 4, settling dynamics, 3,000 episodes. `gap_weight_mode: log_count`. The configs are
G.0-A's four, unchanged, in `configs/scenarios/thermal_foraging/`:

| arm | config |
|---|---|
| wild type, learning | `connectomeppo_small_continuous2d_thermal_klinotaxis_g0a_log_wt_t35.yml` |
| wild type, frozen | `connectomeppo_small_continuous2d_thermal_klinotaxis_g0a_log_wt_frozen_t35.yml` |
| gap-only null, learning | `connectomeppo_small_continuous2d_thermal_klinotaxis_g0a_log_gaponly_t35.yml` |
| gap-only null, frozen | `connectomeppo_small_continuous2d_thermal_klinotaxis_g0a_log_gaponly_frozen_t35.yml` |

Seeds **3901–3964** (64), the band reserved in the tracker's seed-band table. They are disjoint from
every committed band on this cell's gap contrasts: G.0's panel, anchor, smokes and gate pilot
(2401–2464, 9801–9816, 9851–9868); G.0-A's pilot and smokes (9871–9886, 9891–9892); and 082's Cook draws
(513–576). The harness checks this. **256 runs.**

The gap-only null holds the chemical graph, autapses included, and every neuron's gap degree. It moves
gap placement and, with the log counts travelling with their edges, each neuron's total gap strength.

## The reading

`gap(log)` on `auc_success`, wild type − null, paired by seed over the 64 seeds. It is computed with
`ops.score_level` and `ops.wiring_gap`, with an 80% percentile bootstrap interval. Its q is the
two-sided Wilcoxon p, a family of one, through `ops.apply_family_correction`. It is classified by
`mc.classify` at **m = 0.0577**, the carried minimum (two-thirds of the thermal split's 0.0865, Logbook
082). The maintainer chose that minimum on 2026-10-17. G.0's own minimum, 0.0894, was re-fixed from the
count model's lead, which is the artefact.

**Gates.** Each learning arm beats its wiring's frozen floor (the paired interval of learning − frozen
plateau lies above zero). The two learning arms do not both plateau at 90% or more (`mp.level_passes`).

| state of `gap(log)` | verdict | what it licenses |
|---|---|---|
| gates fail | **`unreadable`** | nothing; a dated decision in the tracker and the roadmap |
| `move_wt` | **`placement`** | G.1–G.3b register on log counts, with the minimum below |
| `below`, either direction | **`below_minimum`** | nothing by itself; the direction is recorded; a dated decision |
| `no_move` | **`no_wiring_effect`** | the point-worm half of the STOP condition; G.1–G.3b do not register until the body cell reads on log counts |
| `move_null` | **`reversed`** | nothing by itself; a dated decision |
| `unresolved` | **`unresolved`** | nothing by itself; a dated decision, never a re-read under another rule |

**The minimum a mechanism panel carries, under `placement`.** This panel is D23's anchor on the
corrected model. It is read in its own campaign and never inside G.1's panel, so G.1 needs no separate
anchor pilot.

- A lead in [0.058, 0.13] carries 0.0577.
- A lead above 0.13 re-fixes the minimum at two-thirds of the lead.
- A lead that clears 0.0577 but sits below 0.058 carries 0.0577, not two-thirds of itself.

**The two cells together**, fixed now so that neither is read in the other's light:

- **STOP** needs `no_wiring_effect` on both: here, and in 092's body contrast re-run on log counts after
  M.1.
- **`placement` here:** G.1–G.3b register on the point worm, whatever the body reads.
- **`no_wiring_effect` here and an effect through the body:** G.1–G.3b are not licensed on the point
  worm. Whether they move to the body is a dated decision then.
- **Any other combination** is a dated decision.

**Described beside, never verdicts:**

- episodes to 30% success;
- each arm's plateau and floor;
- the difference from G.0's 16-seed log level;
- the censoring rule's choice;
- the D24 audit below.

No run exceeds 3,000 episodes, so the collapse statistic does not apply.

## Sizing and the operating characteristic

**The spread.** On this cell the paired spread is σ ≈ 0.119 for G.0's count level (64 seeds), 0.109 for
its log level (16 seeds) and 0.094 for 082's Cook gap (64 seeds). At σ = 0.115 the standard error over
64 seeds is about 0.014. The detectable effect for the two-sided test at 80% power is 0.038–0.042, below
m.

**The operating characteristic.** It is simulated with the registered instrument: 1,000 draws of 64
normal seed gaps at σ = 0.115 per row, each scored by `wp.paired_seed_wilcoxon_bootstrap` and
classified by `mc.classify` at m.

| true effect | `placement` | `below_minimum` | `no_wiring_effect` | `unresolved` |
|---|---|---|---|---|
| 0 | 0 | 0.05 (0.02 wild type, 0.03 null) | 0.81 | 0.15 |
| +0.012 (G.0's 16-seed estimate) | 0 | 0.13 | 0.63 | 0.24 |
| +0.0577 (the minimum) | 0.50 | 0.47 | 0 | 0.03 |
| +0.08 | 0.94 | 0.06 | 0 | 0 |
| +0.104 (the count model's lead) | 1.00 | 0 | 0 | 0 |

**Why `unresolved` is common when there is no effect.** `no_move` needs the 80% interval to include
zero and q to be at least 0.05. A mean between about 1.28 and 1.96 standard errors from zero satisfies
neither condition, whatever the seed count. It is the shared classifier's rule, kept by the maintainer
on 2026-10-18 at the spec review.

The achieved spread is reported beside the reading and never used to re-read it.

## D24: what the null does not preserve, before any run

Computed with `logcount_contrast.py table` on the panel's 64 draws ([table.json](table.json)), using
each config's own brain. The wild type's matrix takes no seed; the null shows medians over seeds.

| wiring | max G | p99 | pairs > 1 | ALA total | CANL total | AVAL total | sensor–motor pairs | count–degree ρ | motors at 1 hop | at 2 hops |
|---|---|---|---|---|---|---|---|---|---|---|
| log-count wild type | 1.34 | 0.73 | 5 | 4.47 | 2.37 | 4.09 | 0 | 0.188 | 0 | 28 |
| log-count gap-only | 1.30 | 0.63 | 2 | 1.54 | 0.66 | 3.28 | 5 | −0.005 | 5 | 39 |

The null creates a median of five sensor–motor gap pairs where the wild type has none. Five motor
neurons therefore sit one hop from an injected sensor in the null and none in the wild type. It also
destroys the wild type's correlation between counts and degree products, and lowers the hubs' totals:
ALA's falls from 4.47 to 1.54. The gap therefore compares the wild type against a null that both moves
placement and carries a sensor–motor shortcut, as every gap-only null has since 082. This is stated
beside the verdict, never used to re-read it.

## Before launch

- **Gates on the same configs, on disjoint seeds**: `logcount_contrast.py preflight` on G.0-A's pilot
  (seeds 9871–9886), [preflight.json](preflight.json). The level reads **`readable`** and launch is
  **true**: plateaus 70.0% (wild type) and 69.2% (null), floors 0.0% for both, 20 points under the bar,
  16 of 16 seeds scored. Gate evidence only. On 16 seeds a difference between the arms licenses nothing,
  and none is read.
- **The cell's positive control**: G.0-A's MLP-PPO on this cell's configs and episodes learned to 71.2%
  against a 0.0% floor (Logbook 097). No MLP arm runs here.
- **Smokes**: none new. These four configs ran 8 smoke runs on 9891–9892 and 64 pilot runs, every one
  exiting 0.
- **Cost**: from the pilot's own records of these four configs (the tracked-experiment ID's start
  against its UTC end timestamp), median 23.1 minutes per learning run and 4.6 per frozen run at 16
  workers. 128 of each come to about **3.7 hours**, about 4 with the last wave.
- **Readiness review**: `/nematode-review-spec add-logcount-gap-contrast`, 2026-10-18. Fixed before
  registration:
  - the operating characteristic, first written from a normal approximation that missed the
    classifier's `unresolved` band, is now simulated with the instrument at the committed spread;
  - `below` with the null ahead was mapped to `reversed`, and is now `below_minimum` with its
    direction;
  - the two cells' joint outcomes are named;
  - the D24 table is built from G.0-A's configs, not G.0's generator;
  - the mechanism minimum's edge just below 0.058 is closed;
  - the cost is taken from the pilot's own records.

## Launch

After M.1(b)'s rounds release the machine, from a worktree at this commit. The flags and configs are
arrays, so the commands work pasted into bash or zsh.

```bash
git -C ../nematode-g0a checkout --detach <this commit>
cd ../nematode-g0a
uv sync --extra cpu --extra torch --extra pixel --extra analysis --extra qpu
T=configs/scenarios/thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis_g0a_log
OUT=(-- --theme headless --track-experiment --no-detailed-export --no-file-log)
cfgs=(); for a in wt_t35 wt_frozen_t35 gaponly_t35 gaponly_frozen_t35; do cfgs+=(--config "${T}_${a}.yml"); done
uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds 3901-3964 --runs 3000 --workers 16 \
  --output-dir ../nematode/campaigns/g0a-logcount "${OUT[@]}"

# Then, from the same worktree, so the registered harness reads the runs:
D=../nematode/docs/experiments/logbooks/supporting/099-logcount-contrast
uv run python scripts/analysis/logcount_contrast.py score --logs ../nematode/campaigns/g0a-logcount/logs \
  --out-dir build/g0a-logcount --out $D/panel.json --csv $D/per-seed.csv
```

**Failed runs**: a run that exits non-zero is re-run once on the same seed. `score` reads only a complete
panel, every arm on every seed, and refuses a log that does not record the config its name claims.

## Retention (A.0)

Committed: this record, `preflight.json`, `table.json`, and the panel's `panel.json` and `per-seed.csv`.
Archived off-repo: the panel campaign's raw logs and the worktree's session records.
