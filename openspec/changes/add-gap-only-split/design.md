## Context

[Logbook 074](../../../docs/experiments/logbooks/074-null-strength-control.md) (A.6) moved block V's
wiring gap toward the null by holding a null's gap junctions and autapses together. The rewiring code
already holds each separately (`rewire_gap_junctions`, `preserve_autapses`).

Decisions already taken:

- the split runs before the 8a synthesis;
- it runs on both learners;
- it reuses A.6's seeds and runs, licensed by the identity check.

## Goals / Non-Goals

**Goals.** Measure how much of A.6's move the gap junctions reproduce, jointly, on an exactly paired
contrast.

**Non-Goals.**

- Separating gap placement from gap strength.
- A separately controlled autapse arm: the remainder is reported, not tested.
- New seeds.
- Thermal.

## Decisions

### Decision A: The gap-held null pairs exactly with the current null

`wiring: "rewired_gap_junctions_held"` calls
`rewire_degree_preserving(..., rewire_gap_junctions=False, preserve_autapses=False)`.

- **The chemical swap is unchanged.** It runs first, over the same list, with the same generator, so
  at the same `rewire_seed` its chemical edges and counts are the current null's exactly: the same
  swapped graph, and the same autapses lost.
- **Only the gap-junction swap is skipped**, so gap junctions keep the wild type's pairs and counts.

The pairing is asserted, not argued. At a fixed seed, `m_chem` must be bit-identical to the current
null's and `g_gap` bit-identical to the wild type's.

### Decision B: Reuse, licensed by the identity check

**What is reused.** A.6's arms at seeds 305–336 (PPO) and 337–384 (reading): the wild type, the
current null and the chemical-only null, each learning and frozen. The logs sit in `campaigns/a6-*`.

**The requirement.** "A committed baseline is reused only under a parsed-field identity check" asks
for one seed per reused arm, re-run on the current path and compared on every parsed field.

**How this change meets it.**

- **Seeds re-run:** the band's first seed, 305 or 337.
- **Arms:** all six reused arms on each learner, so 12 runs.
- **Output:** the re-runs go to a separate campaign directory.
- **Fields compared:** everything the analysis reads from a run.
  - Every `Run:` line of each log, status and foods included: the series `plateau_tail` and the
    efficiency instruments parse.
  - The run's final chemical matrix, `w_chem` in `weights/final.pt`, found through its experiment
    record, which is what the drift check reads. It is compared bit for bit on both learners.
- **Command line:** A.6's exactly — the same configs, `--runs 3000`, and the same output flags,
  `--track-experiment` included, without which no weights are exported. The launch record quotes it.
- **On a difference:** the reused baseline is re-run in full and nothing is mixed. The launch waits
  on this check.
- **Evidence:** the comparison is written to the supporting directory.

### Decision C: New arms and cost

| learner | seeds | new stems | runs |
|---|---|---|---|
| PPO | 305–336 | `…_hard350_rewired_gap_held_null`, `…_hard350_rewired_gap_held_null_frozen` | 64 |
| reading | 337–384 | `…_hard350_eprop_readout_only_rewired_gap_held_null`, `…_hard350_eprop_frozen_rewired_gap_held_null` | 96 |

Each new arm is derived from its current-null parent by changing `wiring` alone, from an explicit
stem table.

The total is 160 runs plus 12 identity runs, about 3.5 hours at B.1b's measured per-run times.

### Decision D: The primary, the minimum and the verdicts

**The primary interaction, per learner,** is `gap(gap-held) − gap(current)` on `auc_success`, paired
by seed with A.2's `interaction`. The wild-type arms are shared and the chemical graphs are identical,
so it is `current null − gap-held null` on the same graph.

**The minimum is 2/3 of A.6's committed move.** A.6's move is its own interaction (Logbook 074's
`control.json`): −0.02778 for PPO and −0.09857 for the reading learner, which gives minimums of 0.0185
and 0.0657. The question is what share of that move the gap junctions reproduce, so A.6's move is the
reference, not block V's effect.

**The states** are B.1c's `classify`, imported unchanged:

| state | verdict | reading |
|---|---|---|
| `move_null` | **gap_junctions** | holding gap junctions alone moves the gap toward the null by at least 2/3 of A.6's move: the gap junctions, placement and strength jointly, carry most of it |
| `below` | **partial** | a significant move toward the null, short of 2/3 of A.6's |
| `no_move` | **not_gap_junctions** | no move is detected, and the interval stays inside ±2/3 of A.6's move: the gap junctions reproduce less than 2/3 of it, so at least a third lies with the autapses or with the chemical-graph difference in A.6's chemical-only null, which this panel cannot separate |
| `move_wt` | **opposite** | holding gap junctions moves the gap toward the wild type |
| `unresolved` | **unresolved** | reported with both MDEs |

**Gates:** B.1c's. Every learning arm must beat its own floor, and no level may saturate. The reading
learner's `w_chem` must not drift on any scored seed (`honour_drift`).

**Family:** BH-FDR over the two primaries.

### Decision E: Sensitivity

The only committed proxy is A.6's own per-seed interaction spread: sd 0.0614 for PPO (n = 32) and
0.2821 for the reading learner (n = 48). That gives MDEs of 0.027 and 0.101, or 0.97× and 1.03× of
A.6's move. **At that power the split resolves only a gap-junction effect about as large as A.6's
whole move.**

**The proxy is pessimistic.** A.6's interaction differences two nulls built on different chemical
graphs; this one differences two nulls on the same graph, which removes the graph-sampling variance.
The achieved MDE, from this panel's own spread, is reported beside the registered figure. No verdict
is re-read against it.

### Decision F: The breakdown, as description

Everything is at the same seeds, so A.6's move splits exactly per seed:

```text
A.6 move = [gap(gap-held) − gap(current)]  +  [gap(chemical-only) − gap(gap-held)]
                gap junctions, jointly             autapses + chemical-graph difference
```

**The gap junctions' share is reported with A.6's uncertainty beside it.** A.6's move is itself
uncertain ([−0.041, −0.014] under PPO), so the share of it is never stated as a precise fraction; the
minimum uses the point estimate as a reference, which is all it needs to be.

The second term cannot be attributed to the autapses. The chemical-only null's chemical graph is a
different sample from the gap-held null's, so the term mixes the two causes. It is reported as that.

### Decision G: Reuse, not copies

`scripts/analysis/gap_split.py` holds this panel: stems, levels, the two-campaign manifest, the
identity comparator, the breakdown and the verdict map. It reuses:

- **A.6's module:** `null_strength_control`'s stem table and seeds, extended by one level;
- **A.2's module:** `score_level`, `learning_gates`, `interaction`, `wiring_gap`, `censoring_rates`,
  `choose_metric`, `apply_family_correction`, `substrate_drift`;
- **B.1c's module:** `classify`, `honour_drift`;
- **B.1b's module:** `require_complete`.

## Risks / Trade-offs

- **The split may come out unresolved.** The panel is powered, on the pessimistic proxy, only for the
  whole move. This is registered and accepted; a fresh, larger panel was the alternative and cost
  about six times as much.
- **The identity check could fail.** Then the reused baseline is re-run in full, and the cost grows
  to about 13 hours. Nothing in the execution path has changed since A.6 but documentation, so this is
  not expected. It is still checked, not assumed.
- **The reuse depends on more than the campaign directories.** The drift evidence for every reused
  run lives outside them, in `experiments/<id>.json` and `exports/<session>/weights/final.pt`. Those
  are kept too, until the logbook commits. A pre-launch check confirms drift evidence resolves for all
  384 reused runs, not only the 12 re-run for the identity check; a gap there would void the reading
  learner's reused arms.
