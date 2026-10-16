# G.0 — the gap model's operating point: registration and launch

**Registered 2026-10-17, before any panel run.** Change: `add-gap-operating-point`. Tracker
`phase9-tracking` G.0; roadmap v4.5 D23 (one substrate, Emmons 2024, re-anchored in its own pilot) and
D24 (the null a question reads, its matched statistics and its shortcut audit).

## The question

On the thermal cell at target 35 the wild type leads the gap-only null, whose chemical graph is held and
whose gap junctions are rewired with their counts (Logbook 082, Cook 2019, +0.096 `auc_success`). The
model couples gap junctions as `G_ij = count_ij / sqrt(d_i d_j)`, `d` the binary gap degree, added to
the settling map's tanh drive. That normaliser counts partners, not counts, so a few high-count pairs
between low-degree neurons couple far above 1: ALA–CAN at 89.7, and 98 pairs above 1 on Emmons. The
swap moves them. **Is the lead a property of the wiring, or of how this model scales gap junctions?**

## The arms

The thermal cell at target 35 on **Emmons 2024** (D23), PPO at block V's point otherwise: edge-order
draw, pooled readout, depth 4 unless stated, settling dynamics, 3,000 episodes. Configs are generated
as stated deltas by `scripts/campaigns/generate_gap_operating_point_configs.py`.

| level | wild type | null | gap seeds | floor seeds | role |
|---|---|---|---|---|---|
| `count` | learn + frozen | gap-only, learn + frozen | 2401–2464 | 2401–2464 | the reference |
| `unit` | learn + frozen | gap-only, learn + frozen | 2401–2464 | 2401–2464 | **deciding** |
| `outliers_scattered` | the `count` wild type | scattered, learn + frozen | 2401–2432 | 2401–2432 | **deciding** |
| `outliers_held` | the `count` wild type | held, learn + frozen | 2401–2432 | 2401–2416 | beside |
| `gap_off_vs_wt` | the `count` wild type | gap-off, learn + frozen | 2401–2432 | 2401–2416 | beside |
| `gap_off_vs_null` | gap-off | the `count` gap-only null | 2401–2432 | 2401–2416 | beside |
| `scale0p25`, `scale0p5`, `scale2`, `log_count` | learn | gap-only, learn | 2401–2416 | 2401–2416 | beside |
| `depth2`, `depth3`, `depth6` | learn | gap-only, learn | 2401–2416 | 2401–2416 | beside |

What each wiring holds and what it does not (D24):

- **gap-only null**: holds the chemical graph, autapses included, and every neuron's gap degree; moves
  gap placement and, with the counts travelling with their edges, each neuron's total gap strength.
- **`unit`** (both wirings): every existing pair weighted 1 before the normalisation; the binary degrees,
  and so the normalisers, are unchanged. The two wirings then differ in placement alone.
- **`outliers_scattered`**: the 26 pairs with count ≥ 20 are each swapped once with a partner no
  earlier swap used, and a fair coin decides which of the two new pairs carries the outlier's count, so
  it lands on a pair containing either endpoint with equal chance; only the 26 partners move with
  them. The chemical graph and every gap degree are held. It does not hold the outliers' endpoints'
  totals (ALA's total count of 1,314 falls to 265–1,070 across seeds) and creates no sensor–motor gap
  pair.
- **`outliers_held`**: the 26 outliers stay on their pairs with their counts; the other 1,069 pairs are
  rewired by the same degree-preserving swap, with the outliers forbidden as targets.
- **gap-off**: no gap junctions; one brain, since without them the wild type and the gap-only null
  coincide.
- **scale** and **`log_count`**: the placement of the wiring they sit on; the coupling × s, or each count
  replaced by `log1p(count)`.
- **depth**: the settling iterations, 2, 3 or 6 instead of 4.

The profile and depth levels have no frozen arms: their gates read the `count` level's floors on
2401–2416, which says whether each arm learns at all and nothing finer.

**896 runs**: 8 configs × 64 seeds, 4 × 32, 16 × 16. For `outliers_held` and gap-off the saturation
gate reads the 16 floor seeds while the gap reads 32; the per-seed CSV's plateau columns are blank on
2417–2432 for those levels.

## The readings

Four readings, each a paired `wild type − null` gap on `auc_success`, positive when the wild type
leads, corrected together (BH-FDR, per metric) and classified by `mc.classify` at the minimum *m*:
`gap(count)`, `gap(unit)`, `gap(outliers_scattered)`, `gap(outliers_held)`. Episodes to 30% success are
reported beside.

**The minimum** is fixed by D23's rule from the anchor (below): **m = 0.0894**, two-thirds of the
anchor's lead of 0.1341, because the lead sits above the [0.058, 0.13] range in which 082's 0.0577
would have been carried.

`gap(count)` must exclude zero above (its 80% interval's lower end above zero; its BH q is not read
for this check), or the panel reads **`no_gap_effect`** and nothing else is read. If the `count` level
fails a floor or saturates, the panel reads **`unreadable`**.
A deciding level that fails a floor or saturates is **open**, never read. Then each deciding reading
falls in one of three classes: `gap(unit)` **holds** (`move_wt`), **fails** (`no_move`, `below` or
`move_null`) or is **open** (`unresolved`, or unreadable); `gap(outliers_scattered)` **reproduces**
(`move_wt`), **does not** (`no_move`, `below` or `move_null`) or is **open**.

| `gap(unit)` ↓ / `gap(outliers_scattered)` → | does not | reproduces | open |
|---|---|---|---|
| **holds** | `survives` | `both` | `unresolved` |
| **fails** | `artefact` | `artefact` | `artefact` |
| **open** | `unresolved` | `unresolved` | `unresolved` |

- **`survives`** licenses G.1–G.3b at the committed scale and depth, cited in every claim.
- **`both`**: placement matters without the counts, and moving the few outliers alone costs a
  cell-scale lead. It licenses G.1–G.3b, with the outlier pairs registered as their own class in G.3.
- **`artefact`** re-aims block G (G.0-A): once the counts are removed the lead is gone, whatever the
  outliers do.
- **`unresolved`** licenses nothing by itself; how block G proceeds is a dated decision in the tracker
  and the roadmap.
- **`no_gap_effect`**: the effect did not reproduce on Emmons, which is itself D23's answer.

**Described beside, never verdicts**: `gap(outliers_held)`; gap-off against both count arms; the scale
profile and `log_count`; the depths; episodes to competence; the censoring rule's choice. No run here
exceeds 3,000 episodes, so the collapse statistic does not apply.

**Gates.** Each learning arm beats its wiring's frozen floor on the level's floor seeds (the paired
test's interval excludes zero above), and the two learning arms of a level do not both reach 90%.

**Sizing.** For the family's two-sided test at 80% power the detectable effect is about 2.8 σ/√n. On
Logbook 082's Cook gap (σ 0.0943 over 64 seeds) and on the anchor's own (σ 0.1095 over 16) it is 0.033–
0.038 at 64 seeds and 0.047–0.054 at 32, both well below *m*; at 16 seeds it is 0.066–0.077, which is
why the 16-seed levels are description. The achieved spread is reported beside and never used to
re-read.

## The predictor, registered before it is computed (Decision E)

On untrained brains, with `sign_scale_calibration.py`'s 64 fixed inputs: the mean fraction of the
**path set** — the injected sensors (the six food sensors and AFDL/R), the command interneurons
(AVA, AVB, AVD, AVE and PVC, left and right) and the readout motor neurons — at `|h| > 0.99`, null minus
wild type, per seed. **Predicted**: Spearman ρ with that seed's committed gap is positive (more saturated
path in the null, a larger wild-type lead), |ρ| ≥ 0.3; two-sided permutation p (20,000 permutations) and
an 80% bootstrap interval. States: **`predicts`** (ρ ≥ 0.3 and p < 0.05), **`no_prediction`** (the
interval inside ±0.3), **`unresolved`** otherwise. Computed first on Logbook 082's 64 committed
draws (Cook 2019, seeds 513–576, against `fixed_gap_auc_success`), after this record is committed; then on G.0(a)'s Emmons draws, the
anchor's 16 seeds against the anchor's own `count` gap, as the roadmap registers it; and on the
panel's 64 draws against the panel's `count` gap, once the panel reads, as description beside.

## The matched statistics and the shortcut audit (Decision F, D24)

Computed before launch ([table.json](table.json)), for every gap wiring at every seed it is read on (the
depth levels share the count matrices, and gap-off has none; a wild type's matrix takes no seed).
Medians over seeds:

| wiring | max G | p99 | pairs > 1 | ALA total | CANL total | AVAL total | sensor–motor pairs | count–degree ρ | motors at 1 hop | at 2 hops |
|---|---|---|---|---|---|---|---|---|---|---|
| `count` wild type | 89.67 | 5.13 | 98 | 232.0 | 112.0 | 12.46 | 0 | 0.188 | 0 | 28 |
| `count` gap-only | 57.88 | 5.22 | 122 | 4.7 | 1.2 | 10.01 | 4 | 0.005 | 4 | 38 |
| `unit` wild type | 1.00 | 0.41 | 0 | 1.3 | 0.4 | 2.59 | 0 | — | 0 | 28 |
| `unit` gap-only | 0.58 | 0.33 | 0 | 1.1 | 0.5 | 2.37 | 4 | — | 4 | 38 |
| `outliers_scattered` | 94.52 | 4.61 | 100 | 67.8 | 88.7 | 12.46 | 0 | 0.182 | 0 | 28 |
| `outliers_held` | 89.67 | 5.23 | 120 | 232.6 | 112.0 | 8.93 | 4 | 0.002 | 4 | 39 |
| `log_count` wild type / gap-only | 1.34 / 1.20 | 0.73 / 0.65 | 5 / 2 | 4.5 / 1.5 | 2.4 / 0.5 | 4.09 / 3.24 | 0 / 4 | 0.188 / 0.008 | 0 / 4 | 28 / 38 |
| ×0.25, ×0.5, ×2 wild type | 22.4, 44.8, 179.3 | 1.28, 2.57, 10.26 | 15, 41, 267 | 58, 116, 464 | 28, 56, 224 | 3.1, 6.2, 24.9 | 0 | 0.188 | 0 | 28 |
| ×0.25, ×0.5, ×2 gap-only | 14.7, 29.4, 117.7 | 1.37, 2.75, 10.99 | 16, 44, 280 | 1.0, 2.1, 8.3 | 0.2, 0.5, 1.9 | 2.6, 5.1, 20.4 | 4 | 0.008 | 4 | 38 |

**What the audit shows before any run.** Every gap-only null, under every weighting, creates about four
sensor–motor gap pairs where the wild type has none, so four motor neurons sit one hop from an injected
sensor in the null and none in the wild type; that is also true of `outliers_held`, whose rewired 1,069
pairs are where those pairs come from. The scattered null creates none. So `gap(unit)` compares a wild
type against a null that both lacks the counts' structure **and** carries a sensor–motor shortcut, and
`gap(outliers_scattered)` compares against a null that moves the outliers and nothing else of that
kind. The verdict reads as registered; this is stated beside it, never used to re-read it. The wild
type's counts are correlated with its pairs' degree products (ρ 0.19), which the swap destroys (ρ ≈ 0).

## Before launch

- **The anchor** (D23), seeds 9801–9816, the four `count` arms, 64 runs, every one exiting 0
  ([anchor.json](anchor.json)): the wild type leads by **0.1341** `auc_success`, 80% interval
  [0.101, 0.168], Wilcoxon p 0.0002; by 516 episodes to 30% success. Plateaus 71.2% and 49.1%, floors
  0.0% for both, neither saturated: readable. The effect reproduces on Emmons, larger than 082's 0.096
  on Cook. D23's case is `refixed`: **m = 0.0894**.
- **Gate pilot** (seeds 9861–9868, 64 runs: the `count` wild type learning and frozen, the four `unit`
  arms, the scattered null learning and frozen), configured exactly as the panel, so the deciding
  levels' gates are evaluated before launch. `unit` moves the operating point (the largest coupling
  falls from 89.7 to 1), so its floor and saturation could not be assumed.
- **Gate preflight** (`gap_operating_point.py preflight`, [preflight.json](preflight.json)), `count`
  on the anchor and `unit` and `outliers_scattered` on the gate pilot: all three
  **`readable`**, launch **true**: `count` plateaus 71.2% and 49.1% (16 seeds), `unit` 74.5% and 73.2%
  and `outliers_scattered` 62.9% and 67.0% (8 seeds each), every floor 0.0%, the highest plateau 15
  points under the 90% bar. The gate pilot's 64 runs all exited 0. Its plateaus were read for the
  gates and nothing else: on 8 seeds a difference between arms licenses nothing, and none is read.
- **Smokes**, seeds 9851–9852, every new level's configs at 20 episodes: 48 of 48 runs exited 0, and the
  scattered configs again after the scatter's fix. They check that the configs execute and are
  discarded; they are not a pilot. The beside levels' floors are read in the panel itself, under the
  rule that a level failing its floor is reported unreadable.
- **Cost**, from the anchor's own run times at 16 workers: 22.9 minutes a learning run and 5.1 a
  frozen one (about four minutes of the first wave overlapped an eight-worker test run). The panel's
  576 learning and 320 frozen runs come to about **16 hours**; the depth-6 arms run longer and the
  depth-2 arms shorter.
- **Readiness review**: `/nematode-review-spec add-gap-operating-point`, 2026-10-17. Fixed before
  launch: the scattered null always left each outlier's count on its first-sorted endpoint, so ALA kept
  its four largest junctions at every seed (now a fair coin, partners used once, re-smoked); the
  deciding levels' gates had only a smoke (now the gate pilot and its preflight); the launch commands
  were not zsh-safe (now arrays); the table was to be computed after the panel (now committed here); a
  reversed anchor and an unreadable `count` level are named; the Emmons predictor repeat reads the
  anchor's draws as the roadmap says; the sizing uses the family's two-sided test; a run counts only
  if it exited 0; the censoring rule's choice is labelled description.

## Launch

From the worktree at this commit (`git -C ../nematode-g0 checkout --detach <this commit>`). The flags
and configs are arrays, so the commands work pasted into bash or zsh.

```bash
cd ../nematode-g0
T=configs/scenarios/thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis_g0
OUT=(-- --theme headless --track-experiment --no-detailed-export --no-file-log)
run() {
  local seeds=$1; shift
  local cfgs=(); for s in "$@"; do cfgs+=(--config "${T}_${s}.yml"); done
  uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds "$seeds" --runs 3000 --workers 16 \
    --output-dir ../nematode/campaigns/g0-panel "${OUT[@]}"
}
run 2401-2464 count_wt_t35 count_wt_frozen_t35 count_gaponly_t35 count_gaponly_frozen_t35 \
  unit_wt_t35 unit_wt_frozen_t35 unit_gaponly_t35 unit_gaponly_frozen_t35
run 2401-2432 outliers_scattered_t35 outliers_scattered_frozen_t35 outliers_held_t35 gap_off_t35
run 2401-2416 outliers_held_frozen_t35 gap_off_frozen_t35 scale0p25_wt_t35 scale0p25_gaponly_t35 \
  scale0p5_wt_t35 scale0p5_gaponly_t35 scale2_wt_t35 scale2_gaponly_t35 log_count_wt_t35 \
  log_count_gaponly_t35 depth2_wt_t35 depth2_gaponly_t35 depth3_wt_t35 depth3_gaponly_t35 \
  depth6_wt_t35 depth6_gaponly_t35

# Then, from the main checkout: copy the worktree's records in, and score.
cd ../nematode
rsync -a --ignore-existing ../nematode-g0/experiments/ experiments/
rsync -a --ignore-existing ../nematode-g0/exports/ exports/
D=docs/experiments/logbooks/supporting/095-gap-operating-point
uv run python scripts/analysis/gap_operating_point.py score --logs campaigns/g0-panel/logs \
  --anchor $D/anchor.json --out-dir build/g0 --out $D/panel.json --csv $D/per-seed.csv
uv run python scripts/analysis/gap_operating_point.py predictor --source emmons_panel \
  --out $D/predictor-emmons-panel.json
```

The `plan` subcommand prints the same three groups from the harness's own tables.

## Retention (A.0)

Committed: this record, `anchor.json`, `table.json`, `preflight.json`, `predictor-cook.json`,
`predictor-emmons.json` (the anchor's draws), the panel's `panel.json` and `per-seed.csv`, and
`predictor-emmons-panel.json`. Archived off-repo: the smoke,
anchor, gate-pilot and panel campaigns' raw logs and the worktree's session records.
