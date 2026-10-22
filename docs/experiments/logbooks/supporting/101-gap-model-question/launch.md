# G.0-B — the modelling question, strength-normalised and ohmic gap coupling: registration and launch

**Registered 2026-10-20, before any panel run.** Change: `add-gap-model-question`. Tracker
`phase9-tracking` G.0-B; roadmap v4.5 § Phase 9 Go/No-Go, as decided by the maintainer on 2026-10-20.

## The question

On log-compressed counts the corrected model shows no wiring effect on either cell, which meets the
phase's STOP condition (Logbooks [099](../../099-logcount-contrast.md) and
[100](../../100-body-logcount.md)). The maintainer chose to open the modelling question inside Phase
9 before the STOP statement ships.

Both corrections read so far, unit weights and log counts, shrink each pair's weight. Neither changes
the normaliser `√(d_i d_j)`, which counts partners rather than synapses and so let a few high-count
pairs couple far above 1. **Under two principled gap models that change something else, does the
wild type still learn faster than its gap-only null?**

## The models and the arms

The thermal cell at target 35 on Emmons 2024, at block V's point otherwise: edge-order draw, pooled
readout, depth 4, settling dynamics, 3,000 episodes, count weights.

- **Strength** (`gap_normalisation: strength`): each pair is divided by `√(s_i s_j)`, where `s` is
  the neuron's total gap weight on its own wiring. Every coupling is at most 1, and the matrix's
  spectral radius is 1. The configs are the four `*_g0b_strength_*_t35` from
  `generate_gap_model_configs.py`.
- **Ohmic** (`gap_coupling: ohmic`, `gap_weight_scale: 0.03`): current in proportion to the potential
  difference, at the scale calibrated on untrained brains (Logbook 097). The configs are G.0-A's four
  `*_g0a_ohmic_*_t35`, unchanged. G.0-A's pilot never read this model's wiring gap, and it stays
  unread.

For each model the panel runs the wild type and the gap-only null, each learning and frozen, on seeds
**4201–4264**. That is **512 runs**. The seeds are disjoint from every committed band, which the
harness checks.

**What the strength null does not preserve.** A degree-preserving swap keeps each neuron's partner
count but not its strength. Under `strength`, the null therefore also re-normalises every neuron by its
own new strength. The gap junctions of ALA, CANL and AVAL carry total counts of 1,314, 501 and 227 in
the wild type, against medians of 34, 7 and 177 over the null's 64 draws.

## The readings and the verdict map

**The readings.** `gap(strength)` and `gap(ohmic)` on `auc_success`, wild type − null, each paired by
seed over 64 seeds:

- 80% bootstrap intervals (`ops.wiring_gap`);
- **BH-FDR across the family of two** (`ops.apply_family_correction`, on the two-sided Wilcoxon p);
- each classified by `mc.classify` at **m = 0.0577**.

**Gates**, per model: each learning arm beats its wiring's frozen floor on all 64 seeds, and the two
learning arms do not both plateau at 90% or more (`mp.level_passes`).

| state, per model | verdict |
|---|---|
| gates fail | `unreadable` |
| `move_wt` | **`placement`** |
| `below`, either direction | `below_minimum`, with the direction recorded |
| `no_move` | **`no_wiring_effect`** |
| `move_null` | `reversed` |
| `unresolved` | `unresolved` |

**What the pair does to the STOP statement**, fixed now:

- **Either model reads `placement`: qualified.** The wiring effect depends on how electrical synapses
  are modelled. Each `placement` model registers its body contrast next, and whether G.1–G.3b reopen
  on it is a dated decision after the body reads.
- **Both read `no_wiring_effect`: the statement stands**, behind four corrections (unit weights, log
  counts, strength normalisation, and ohmic coupling). Ohmic's reading is stated at its operating
  point: at ×0.03 the median neuron keeps 93% of its own drive, and the untrained action spread is
  0.0176 against the gap-off brain's 0.0264. An ohmic null is a null at weak coupling, never at every
  conductance. The 9a synthesis follows.
- **Any other pair:** a dated decision.

**Described beside, never verdicts:**

- episodes to 30% success;
- each arm's plateau and floor;
- the two wild types against each other;
- the D24 audit, below.

No run exceeds 3,000 episodes, so the collapse statistic does not apply.

## Sizing and the operating characteristic

The spread is σ = 0.119, the largest committed on this cell (G.0's count level over 64 seeds).

The operating characteristic was simulated with the registered instrument: 500 draws of two models'
64 normal seed gaps at σ = 0.119, each scored by `wp.paired_seed_wilcoxon_bootstrap`, corrected
together by BH-FDR, and read by the harness's own `verdict` and `joint`.

| true effects (strength, ohmic) | statement stands | qualified | dated decision | strength alone reads |
|---|---|---|---|---|
| 0, 0 | 0.59 | 0 | 0.41 | `no_wiring_effect` 0.80, `unresolved` 0.17 |
| +0.012, +0.012 | 0.46 | 0 | 0.54 | `no_wiring_effect` 0.67, `unresolved` 0.25 |
| +0.08, 0 | 0 | 0.90 | 0.10 | `placement` 0.90 |
| +0.0577, 0 | 0 | 0.52 | 0.48 | `placement` 0.52, `below_minimum` 0.40 |
| +0.104, +0.104 | 0 | 1.00 | 0 | `placement` 1.00 |

**Why "stands" is only 0.59 when both models are null.** It needs `no_wiring_effect` on both models,
about 0.8 each. The shared classifier's `unresolved` band takes about 15% of each at a true zero,
whatever the seed count. The maintainer kept that classifier on 2026-10-18. A dated decision is
therefore the second most likely outcome under a true null. It is never a re-read under another rule.

The achieved spread is reported beside the reading and never used to re-read it.

## D24: what each null does not preserve, before any run

Computed with `gap_model_question.py table` on the 64 panel draws ([table.json](table.json)). Each
wild type's matrix takes no seed; for the nulls the table gives medians over draws.

| model, wiring | max G | p99 | pairs > 1 | ALA total | sensor–motor pairs | count–degree ρ | motors at 1 hop | at 2 hops |
|---|---|---|---|---|---|---|---|---|
| strength, wild type | 1.00 | 0.57 | 0 | 1.78 | 0 | −0.04 | 0 | 28 |
| strength, gap-only | 0.95 | 0.66 | 0 | 1.09 | 5 | −0.07 | 4 | 38 |
| ohmic ×0.03, wild type | 2.69 | 0.15 | 2 | 6.96 | 0 | 0.19 | 0 | 28 |
| ohmic ×0.03, gap-only | 1.64 | 0.15 | 2 | 0.12 | 5 | 0.01 | 4 | 38 |

As with every gap-only null since 082, both nulls create about five sensor–motor gap pairs where the
wild type has none. The ohmic matrix is the built conductance matrix at ×0.03, which then enters
`(I + L)⁻¹`. The audit is stated beside the readings and never used to re-read them.

## Before launch

- **The strength probe**, at no run cost ([probe.json](probe.json), seeds 9905–9912). The untrained
  action spread is 0.0374, against a floor of 0.0132 (half the gap-off brain's 0.0264), with no unit
  saturated. It **runs**. For reference: count weights read 0.167, log counts 0.041, ohmic 0.0176.

- **Smokes**: the strength configs on 9913–9914 at 20 episodes. All 8 runs exited 0. They were
  execution checks only, and were discarded.

- **Gates** ([preflight.json](preflight.json)), with launch **true**:

  - strength, on its own pilot (9905–9912, all four arms, configured as the panel, all 32 runs exiting
    0): **`readable`**, at 62.2% and 71.3%, floors 0.0%;
  - ohmic, on G.0-A's pilot arms (9871–9886, the same configs): **`readable`**, at 61.1% and 54.1%,
    floors 0.0%.

  These plateaus are gate evidence only. On 8 or 16 seeds a difference between the arms licenses
  nothing, and none is read.

- **Cost**, from the pilots' own records:

  - strength: 22.1 minutes a learning run and 4.4 frozen;
  - ohmic (G.0-A): 23.2 and 4.8.

  At 16 workers the 512 runs come to about **7.3 hours**.

- **Readiness review**: `/nematode-review-spec add-gap-model-question`, 2026-10-20. It found no
  blocking issue. Fixed before implementation:

  - the strength null's normaliser change is stated;
  - ohmic is read at its operating point;
  - the probe's instrument is named;
  - a zero-strength neuron is tested;
  - ohmic's cost is cited;
  - a cross-panel description was dropped.

## Launch

From a worktree at this commit. The flags and configs are arrays, so the commands work pasted into bash
or zsh.

```bash
git -C ../nematode-g0a checkout --detach <this commit>
cd ../nematode-g0a
uv sync --extra cpu --extra torch --extra pixel --extra analysis --extra qpu
cfgs=($(uv run python scripts/analysis/gap_model_question.py plan \
  | python3 -c "import json,sys; print(' '.join('--config '+c for c in json.load(sys.stdin)['configs']))"))
uv run python scripts/run_campaign.py "${cfgs[@]}" --seeds 4201-4264 --runs 3000 --workers 16 \
  --output-dir ../nematode/campaigns/g0b-panel \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# Then, from the same worktree, so the registered harness reads the runs:
D=../nematode/docs/experiments/logbooks/supporting/101-gap-model-question
uv run python scripts/analysis/gap_model_question.py score --logs ../nematode/campaigns/g0b-panel/logs \
  --out-dir build/g0b --out $D/panel.json --csv $D/per-seed.csv
```

**Failed runs**: a run that exits non-zero is re-run once on the same seed. `score` reads only a
complete panel, every arm of both models on every seed, and refuses a log that does not record the
config its name claims.

## Retention (A.0)

Committed: this record, `probe.json`, `preflight.json`, `table.json`, and the panel's `panel.json` and
`per-seed.csv`. Archived off-repo: the smoke, pilot and panel campaigns' raw logs and the worktree's
session records.
