# A.6's gap-only split: registration and launch

**Registered 2026-09-26, before any new seed ran.** Change: `add-gap-only-split`.

This is A.6's registered follow-up, run before the 8a synthesis. It fixes the reuse and its identity
check, the interaction, the minimum, the verdicts, the sensitivity and the breakdown before the
identity check or any new arm runs.

## The question

[Logbook 074](../../074-null-strength-control.md) held a null's gap junctions **and** autapses at the
wild type's. That moved the wiring gap toward the null by −0.028 `auc_success` under PPO and −0.099
under the reading learner. How much of that move do the **gap junctions** reproduce, holding placement
and strength together?

## Three nulls, and what each preserves

| property | current null (`rewired_degree_preserving`) | chemical-only null (A.6) | gap-held null (`rewired_gap_junctions_held`, new) |
|---|---|---|---|
| chemical degree per neuron | preserved | preserved | preserved |
| chemical graph at a given seed | the swap's sample | a different sample (autapses removed from the swap) | **the current null's, edge for edge** |
| autapses | lost (none in the three checked) | the wild type's 38 | **lost exactly as in the current null** |
| gap-junction degree per neuron | preserved | preserved | preserved |
| gap-junction pairs and counts (strength) | rewired, strength moves | the wild type's | **the wild type's** |

**The gap-held null pairs exactly with the current null.** It runs the same chemical swap over the
same list with the same generator, and skips only the gap-junction swap. Tests assert that `m_chem`,
the drawn `w_chem` and the autapse diagonal are bit-identical to the current null's at the same seed,
and `g_gap` to the wild type's.

The two therefore differ in their gap junctions alone, **placement and strength jointly**. This panel
cannot separate placement from strength; that needs a further control that moves one without the
other.

## The panel: A.6's seeds and runs, reused

| learner | seeds | reused from `campaigns/a6-*` | new | new runs |
|---|---|---|---|---|
| PPO | 305–336 | the wild type, the current null and the chemical-only null, each learning and frozen | the gap-held null, learning and frozen | 64 |
| reading | 337–384 | same | same | 96 |

The new configs differ from their current-null parents in `wiring` alone.

## The reuse is licensed first

The committed-baseline requirement governs the reuse.

**The identity check.** One seed per reused arm, the band's first (305 or 337), is re-run: six arms
per learner, 12 runs.

- **Fields compared:** everything the analysis reads from a run.
  - every `Run:` line, including status and foods, which is the series every instrument parses;
  - the final `w_chem`, compared **bit for bit**, which is what the drift check reads.
- **Command line:** A.6's exactly (below), `--track-experiment` included.
- **If anything differs,** the reused baseline is re-run in full and nothing is mixed. **No new arm
  launches until all 12 are identical.**
- The comparison is committed here as `identity-ppo.json` and `identity-reading.json`.

**Drift evidence.** It already resolves for every reused run: all 480, 192 PPO and 288 reading, were
checked with `gap_split.py evidence`. The experiment records and exports those runs depend on are kept
until the logbook commits.

**A.6's move is re-derived when the panel is scored.** Scoring recomputes it from the reused runs and
checks it seed by seed against [A.6's committed per-seed CSV](../074-null-strength-control/per-seed.csv).
The result is recorded as `a6_reproduced`.

## The interaction

Per learner, on `auc_success`, paired by seed with A.2's `interaction`:

```text
gap(wild type vs gap-held null) − gap(wild type vs current null)
```

The wild type is shared and the chemical graphs are identical, so this is the current null minus the
gap-held null on the same graph. Episodes are reported beside it, and the family is the two primaries
under BH-FDR.

## The minimum and the verdicts

The minimum is **2/3 of A.6's committed move**, taken from Logbook 074's `control.json`:

- PPO: **0.0185** (from −0.02779);
- reading: **0.0657** (from −0.09859).

States are B.1c's `classify`: q from the two-sided folded Wilcoxon, with an 80% bootstrap interval.

| state | verdict | reading |
|---|---|---|
| `move_null` | **gap_junctions** | holding the gap junctions alone reproduces at least 2/3 of A.6's move; they carry most of it, placement and strength jointly |
| `below` | **partial** | a significant move toward the null, short of 2/3 of A.6's |
| `no_move` | **not_gap_junctions** | no move detected, with the interval inside ±2/3 of A.6's move; the gap junctions reproduce less than 2/3 of it, so at least a third lies with the autapses or with the chemical-graph difference in A.6's null, which this panel cannot separate |
| `move_wt` | **opposite** | holding the gap junctions moves the gap toward the wild type |
| `unresolved` | **unresolved** | reported with both minimum detectable effects |

**Gates:** B.1c's.

- Every learning arm beats its own floor.
- No level saturates.
- The reading learner's `w_chem` does not drift (`honour_drift`).

## Sensitivity, stated honestly

A.6's per-seed interaction spread is the only committed proxy.

| learner | sd | n | minimum detectable effect | as a share of A.6's move |
|---|---|---|---|---|
| PPO | 0.0614 | 32 | 0.027 | 0.97× |
| reading | 0.2821 | 48 | 0.101 | 1.03× |

**At that power the split resolves only a gap-junction effect about as large as A.6's whole move.**

The proxy is **pessimistic**. A.6 differenced two nulls on different chemical graphs, and this panel
differences two on the same graph, which removes the graph-sampling variance. The achieved sensitivity
is reported beside the registered one. **No verdict is re-read against it.**

## The breakdown, as description

Everything is at the same seeds, so A.6's move splits exactly, seed by seed:

```text
A.6 move = [gap(gap-held) − gap(current)] + [gap(chemical-only) − gap(gap-held)]
              gap junctions, jointly           autapses + chemical-graph difference
```

- **The second term** is never attributed to the autapses. The chemical-only null's chemical graph is
  a different sample from the gap-held null's.
- **The gap junctions' share** is reported beside A.6's own uncertainty (PPO [−0.041, −0.014]), never
  as a precise fraction.

## Launch

The identity check comes first, with A.6's exact command line:

```bash
for half in ppo reading; do
  if [ $half = ppo ]; then seed=305; else seed=337; fi
  cfgs=(); for s in $(uv run python -c "import sys;sys.path.insert(0,'scripts/analysis');import gap_split as g;print(' '.join(g.REUSED_STEMS['$half']))"); do cfgs+=(--config configs/scenarios/foraging/$s.yml); done
  uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds $seed --runs 3000 --workers 16 \
    --output-dir campaigns/a6b-identity-$half \
    -- --theme headless --track-experiment --no-detailed-export --no-file-log
  uv run python scripts/analysis/gap_split.py identity --half $half \
    --a6-campaign campaigns/a6-$half --identity-campaign campaigns/a6b-identity-$half \
    --out docs/experiments/logbooks/supporting/075-gap-only-split/identity-$half.json
done
```

Then the new arms, but only if both identity checks report `all_identical`:

```bash
for half in ppo reading; do
  if [ $half = ppo ]; then seeds=305-336; else seeds=337-384; fi
  cfgs=(); for s in $(uv run python -c "import sys;sys.path.insert(0,'scripts/analysis');import gap_split as g;print(' '.join(s for s,(h,_) in g.NEW_ARMS.items() if h=='$half'))"); do cfgs+=(--config configs/scenarios/foraging/$s.yml); done
  uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds $seeds --runs 3000 --workers 16 \
    --output-dir campaigns/a6b-$half \
    -- --theme headless --track-experiment --no-detailed-export --no-file-log
done

uv run python scripts/analysis/gap_split.py score \
  --a6-ppo campaigns/a6-ppo --a6-reading campaigns/a6-reading \
  --split-ppo campaigns/a6b-ppo --split-reading campaigns/a6b-reading \
  --out-dir build/a6b --out build/a6b/split.json --csv build/a6b/per-seed.csv
```

## Artefact retention (A.0)

- **Committed:** the per-seed CSV, the analysis JSON, both identity-check files and this launch
  record.
- **Archived off-repo:** the raw logs.
- **Kept until Logbook 075 commits:** `campaigns/a6-*` and the reused runs' experiment records and
  exports.

## Cost

160 new runs plus 12 identity runs, at B.1b's measured per-run times: **about 3.5 hours.**
