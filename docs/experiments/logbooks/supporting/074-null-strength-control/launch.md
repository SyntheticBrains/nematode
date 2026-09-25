# A.6 — the null-strength control: registration and launch

**Registered 2026-09-25, before any seed ran.** Change: `add-null-strength-control`.

This is Phase 8 **A.6**, added by the Wormlight review (PR #405) and required before the 8a synthesis
and A.5. This record fixes the arms, the metric, the sensitivity, the minimum, the verdict map and
the gates before either campaign launches. **No seed in either band has been touched.**

## The question

Is block V's wiring gap in the **chemical wiring**, or partly in how the null treats gap junctions and
autapses?

## What each null preserves, and what it does not

Stated as the new protocol requirement asks, where the null is introduced:

| property | wild type | current null (`rewired_degree_preserving`) | chemical null (`rewired_chemical_only`) |
|---|---|---|---|
| chemical in- and out-degree per neuron | — | preserved exactly | preserved exactly |
| which neurons connect chemically | — | rewired | rewired (a different random sample at each seed) |
| gap-junction degree per neuron | — | preserved exactly | preserved exactly |
| gap-junction pairs | — | rewired | **the wild type's** |
| gap-junction strength per neuron (counts are coupling weights) | ALA's gap input 232 | **moves**: ALA's about 3, and about half of neurons move by more than 50% | **the wild type's** |
| autapses (38 in the wild type) | 38 | **lost**: the swap can remove them and never creates them; each of three nulls checked had none | **the wild type's**, all 38 with their counts |

**Every committed wiring result was read against the middle column:** Logbook 034, block V, and
Logbooks 070–073. [Logbook 067](../../067-l4-feature-ablations.md) found that under the reading
learner the wild type's advantage consisted in its gap junctions costing it less than the rewired ones
cost the null. That is one result these unpreserved properties could explain.

## The design

Wild type, the current null and the chemical null, each learning and frozen, on **hard350**.

| learner | point | seeds | runs |
|---|---|---|---|
| PPO | block V's committed point: `edge_order` draw, pooled readout, depth 4 | 305–336 (32) | 192 |
| reading (`readout_only`) | A.2's centre, `edge_order` | 337–384 (48) | 288 |

**Configs.** The wild-type and current-null arms are the committed block V and A.2-centre configs,
unchanged. The four chemical-null configs differ from their current-null parents in `wiring` alone,
and a test checks this through the real loader.

**Levels.** Two, `full` and `chemical`, each gated against its own frozen floors. The wild-type runs
serve both levels.

## The interaction

For each learner:

```text
gap(wild type vs chemical null) − gap(wild type vs current null)
```

It is paired by seed with A.2's `interaction`. The wild type cancels, so seed by seed the interaction
is the current null minus the chemical null. It is positive when the wild type stands better against
the chemical null.

**It is a combined control.** Gap placement, gap strength and autapses are held together, so a move
is attributed to them jointly and to none of them alone.

**The two nulls are different random chemical graphs at each seed.** Taking the autapses out of the
swap changes what it draws. This is no bias, since it averages over seeds, but it adds graph-sampling
variance.

**The gap-only null is the registered paired follow-up if this control moves.** Holding gap
junctions alone (`rewire_gap_junctions=False`) reproduces the current null's chemical graph exactly,
because the chemical swap runs first; a test asserts this. That makes it the first split between the
two causes.

## The metric

**`auc_success` is the primary on both learners**, registered here for two reasons:

- the verdict should rest on one metric across both learners;
- on hard350, A.1's data put `auc_success`'s sensitivity at or above the episode metric's.

`episodes_to_30pct_success` is reported beside it, with the censoring rule's own choice recorded next
to it.

## Sensitivity, from frozen committed data

The minimum detectable effect (MDE) is `2.487 × sd / √n`.

| learner | source of the spread | sd | n | MDE | reference effect | MDE ÷ reference |
|---|---|---|---|---|---|---|
| PPO | [A.1's per-seed hard350 interactions](../070-init-sharing-control/), `auc_success` | 0.0700 / 0.0715 | 32 | 0.031 / 0.032 | +0.0610 | **0.50 / 0.52** |
| reading | [B.1c's per-seed reading interactions](../073-measured-prior-contrast/), `auc_success` | 0.3873 / 0.4164 | 48 | 0.139 / 0.150 | −0.2105 | **0.66 / 0.71** |

**Neither direction is claimed for the proxy.** The wild type cancels in this interaction, which
would make its spread smaller than the source's. The two nulls being different random graphs at each
seed adds variance back.

## The registered minimum, in both directions

The minimum is **2/3 of each learner's committed reference effect**:

- **PPO: 0.0407** `auc_success`, from A.1's `edge_order` hard350 effect of +0.0610 (Logbook 070's
  `baseline_gap_mean`);
- **Reading: 0.1403**, from A.2's reading centre of −0.2105.

It is never a fraction of this campaign's own gap.

**States.** These are B.1c's `classify`, imported unchanged: q from the two-sided folded Wilcoxon,
BH-FDR corrected across **the two primary interactions**, and an 80% bootstrap interval.

| state | condition |
|---|---|
| `move_wt` | q < 0.05, interval above zero, mean ≥ minimum |
| `move_null` | q < 0.05, interval below zero, mean ≤ −minimum |
| `below` | q < 0.05, interval excludes zero, absolute mean < minimum |
| `no_move` | q ≥ 0.05, interval inside (−minimum, +minimum) and including zero |
| `unresolved` | anything else |

## The verdict map, per learner

| state | verdict | reading |
|---|---|---|
| `no_move` | **chemical** | Against a null holding the wild type's gap junctions (placement and strength) and autapses, the wiring gap moves by less than the minimum. The gap is in the chemical wiring *given those held in place*. |
| `no_move`, attribution gate failed | **no_gap_to_attribute** | Neither null is separated from the wild type in this campaign, so there is nothing to locate. |
| `move_null` | **gap_or_autapse** | The wiring gap moves toward the null by at least the minimum. It came at least partly from how the current null rewires gap junctions or drops autapses, jointly. The gap-only null splits them. |
| `move_wt` | **amplified** | The chemical null does worse than the current null by at least the minimum. The current null's rewiring was hiding part of the gap. |
| `below` | **below_minimum** | Licenses nothing on its own. |
| `unresolved` | **unresolved** | Reported with the MDE beside it. |

**The attribution gate.** `chemical` needs the gap against the chemical null to exclude zero on the
reference effect's side: above zero for PPO, below zero for the reading learner, where the null leads.
**Block V's claim is named only in a PPO `chemical` verdict that passes this gate.**

## Gates, read before the interaction

These are B.1c's gates, reused:

- **Floors.** Every learning arm must beat its own level's frozen floor. A level where either wiring
  fails, or both arms saturate at the 90% bar, is **unreadable**, and so is the learner.
- **Drift.** The reading learner's `w_chem` must not move against its floor on any scored seed, or
  that half is **void**. On PPO the same check must read large, as its positive control.

## Described, not registered

A.2's hop probe is run on the chemical null:

```text
scripts/analysis/sensory_motor_hops.py --null rewired_chemical_only
```

The chemical null keeps the wild type's gap junctions, which the propagating graph includes, so its
count of motor neurons one hop from a food sensor shows how much of A.2's depth mechanism survives
here. No verdict reads it.

## Launch

The runner takes one seed range per campaign. The family spans both learners, so both campaigns are
scored together.

```bash
arms() { uv run python -c "import sys;sys.path.insert(0,'scripts/analysis');import null_strength_control as n;print(' '.join(s for s,(h,_,_) in n.LEVELS_BY_STEM.items() if h=='$1'))"; }
for half in ppo reading; do
  if [ $half = ppo ]; then seeds=305-336; else seeds=337-384; fi
  cfgs=(); for s in $(arms $half); do cfgs+=(--config configs/scenarios/foraging/$s.yml); done
  uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds $seeds --runs 3000 --workers 16 \
    --output-dir campaigns/a6-$half \
    -- --theme headless --track-experiment --no-detailed-export --no-file-log
done

uv run python scripts/analysis/null_strength_control.py \
  --ppo-campaign campaigns/a6-ppo --reading-campaign campaigns/a6-reading \
  --out-dir build/a6 --out build/a6/control.json --csv build/a6/per-seed.csv
```

## Artefact retention (A.0)

- **Committed:** the parsed per-seed CSV, the analysis JSON and this launch record.
- **Archived off-repo:** the raw campaign logs.
- **Deletion:** a campaign directory is removed only after its CSV is committed. Any field that
  cannot be compared because its source is gone is named **uncompared**.

## Cost

B.1b measured the per-run times: PPO at 16.4 min learning and 10.5 frozen, the reading learner at
19.4 and 31.9. Both campaigns ran at 15.9× on 16 workers. That puts PPO's 192 runs at about **2.7
hours** and the reading learner's 288 at about **7.7**, **about 10.5 in all**.
