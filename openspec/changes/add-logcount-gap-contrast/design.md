## Context

G.0-A chose log-compressed counts (`gap_weight_mode: log_count`) as the corrected gap model
([Logbook 097](../../../docs/experiments/logbooks/097-gap-substrate.md)). Its selection pilot read
learnability only and never the gap between the wild type and the null. The one reading of that gap
on log counts is G.0's 16-seed description: +0.012 `auc_success` [−0.023, +0.046] on seeds 2401–2416.
At 16 seeds that licenses nothing.

The tracker's G.0-A item, as amended on 2026-10-17, runs two things in order:

1. The corrected model's point-worm contrast, on seeds 3901–3964 at a minimum of 0.0577.
2. 092's body contrast on the corrected model, after M.1 reads out.

Then the decomposition of Logbooks 074–092 is restated. G.1–G.3b run on the corrected model or are
deferred with a destination. The phase's STOP branch is: *the corrected model shows no wiring effect
on either cell under PPO*.

## Goals / Non-Goals

**Goals**

- Read the corrected model's wiring contrast on the point worm, at the program's minimum, under a
  verdict map fixed before any run.
- Commit D24's matched statistics and shortcut audit for the log-count null before launch.
- Fix G.1's minimum on the corrected model by D23's rule, if the contrast reads `placement`.

**Non-goals**

- The body contrast, which is the second step and its own change after M.1.
- The decomposition restatement, which waits for both cells.
- Ohmic coupling, which was not chosen.
- hard350 (Decision A).
- Any mechanism reading.

## Decisions

### Decision A: One cell, the pilot's own configs

The arms are the thermal cell at target 35 on Emmons 2024, with block V's point otherwise: edge-order
draw, pooled readout, depth 4, settling dynamics, 3,000 episodes. They are the wild type and the
gap-only null on log-compressed counts, each learning and frozen, using G.0-A's four configs
unchanged:

- `connectomeppo_small_continuous2d_thermal_klinotaxis_g0a_log_wt_t35.yml`
- `connectomeppo_small_continuous2d_thermal_klinotaxis_g0a_log_wt_frozen_t35.yml`
- `connectomeppo_small_continuous2d_thermal_klinotaxis_g0a_log_gaponly_t35.yml`
- `connectomeppo_small_continuous2d_thermal_klinotaxis_g0a_log_gaponly_frozen_t35.yml`

Reusing them means the G.0-A pilot, on seeds 9871–9886, is this panel's gate evidence, configured
exactly as the panel. Seeds 3901–3964 are disjoint from it, from G.0 (2401–2464, 9801–9816,
9851–9868) and from 082.

**Why only the thermal cell.** 0.0577 is a minimum fixed on this cell: two-thirds of the thermal
split's 0.0865, registered by Logbook 082. The STOP branch's two cells are the point worm and the body,
both on the thermal task, and the body's turn comes after M.1. hard350 has no registered minimum for a
gap-only contrast on Emmons. Reading it would need its own anchor. It is not part of the sequence the
maintainer fixed, and adding it here would double the cost without a decision that reads it. It
stays open for G.1's registration to raise.

**No MLP arm.** The cell's readability under this learner was established on 2026-10-18 by G.0-A's
MLP control, on this cell's configs and episodes: plateau 71.2% against a 0.0% floor on 16 seeds.
This panel's own gates check that both connectome arms learn.

### Decision B: The reading and the verdict map, fixed before any run

**The reading.** `gap(log)` on `auc_success`, wild type − null, paired by seed over 64 seeds. It is
computed with `ops.score_level` and `ops.wiring_gap`: the one-sided Wilcoxon and an 80% percentile
bootstrap interval. Its q is the family-of-one correction from `ops.apply_family_correction`, which is
the two-sided Wilcoxon p, the same code path G.0 used. It is classified by `mc.classify` at
**m = 0.0577**: `CARRIED_MINIMUM`, the value `gap_operating_point.py` already carries.

**Gates.** Each learning arm beats its wiring's frozen floor, meaning the paired interval of learning −
frozen plateau lies above zero. The two learning arms do not both plateau at 90% or more. This is
`mp.level_passes`, as in G.0.

| state of `gap(log)` | verdict | what it licenses |
|---|---|---|
| gates fail | **`unreadable`** | nothing; a dated decision in the tracker and the roadmap |
| `move_wt` | **`placement`** | G.1–G.3b register on log counts, with G.1's minimum fixed by D23's rule from this lead (below) |
| `below` with the wild type ahead | **`below_minimum`** | nothing by itself: a lead under the minimum; a dated decision |
| `no_move` | **`no_wiring_effect`** | the point-worm half of the STOP condition; G.1–G.3b do not register until the body cell reads on log counts |
| `move_null`, or `below` with the null ahead | **`reversed`** | nothing by itself; a dated decision |
| `unresolved` | **`unresolved`** | nothing by itself; a dated decision |

**D23 on the corrected model.** Under `placement` this panel is the anchor D23 asks for. It is the
corrected model's effect, read on Emmons in its own campaign and never inside G.1's panel. G.1 carries
0.0577 if the lead sits in [0.058, 0.13]. Otherwise G.1's minimum is re-fixed at two-thirds of the
lead, which is `anchor_minimum`'s rule. G.1 then needs no separate anchor pilot.

**Described beside, never verdicts:**

- episodes to 30% success, the censored metric;
- each arm's plateau and floor;
- the difference from G.0's 16-seed log level;
- the censoring rule's choice.

No run exceeds 3,000 episodes, so the collapse statistic does not apply.

### Decision C: Sizing and the operating characteristic

The paired spread on this cell is σ ≈ 0.094–0.110 (082's Cook gap over 64 seeds; G.0's anchor over 16
seeds). G.0's log level gives about 0.10. At 64 seeds the standard error is about 0.0125.

- The detectable effect for the two-sided test at 80% power is about 2.8σ/√n = 0.033–0.038, well below
  m.
- Under a normal approximation with σ = 0.10:

| true effect | `placement` | `below_minimum` | `no_wiring_effect` |
|---|---|---|---|
| 0 | 0 | 0.03 | 0.95 |
| +0.012, G.0's 16-seed estimate | 0 | 0.16 | 0.84 |
| +0.0577, the minimum | 0.50 | 0.50 | 0 |
| +0.08 | 0.96 | 0.04 | 0 |
| +0.104, the count model's lead | 1.00 | 0 | 0 |

Any remaining probability goes to `reversed`; at a true effect of 0 that is about 0.025. The
`no_wiring_effect` state also needs the 80% interval to lie inside ±m. Its half-width is about
0.016, so this holds whenever the mean is within about ±0.04. The achieved spread is reported beside
the reading and never used to re-read it.

### Decision D: D24 before launch

`gap_operating_point.table`'s `matched_statistics` is computed for the log-count wild type and for the
gap-only null on each of the 64 panel seeds, and committed as `table.json` before any run:

- max G, p99, pairs above 1;
- the ALA, CANL and AVAL totals;
- sensor–motor gap pairs;
- the count–degree ρ;
- motors at 1 and 2 hops.

G.0's table already shows the pattern on 2401–2464. The log-count gap-only null creates about four
sensor–motor gap pairs where the wild type has none, and destroys the count–degree correlation (ρ 0.19
to ≈ 0). The audit states what the null does not preserve, beside the verdict, and is never used to
re-read it.

### Decision E: Gates and cost before launch

**Gates.** `logcount_contrast.py preflight` evaluates the panel's level on the G.0-A pilot's four log
arms (seeds 9871–9886) through `gate_preflight`, and the preflight is committed. Those arms read 70.0%
and 69.2%, with floors at 0.0%, 20 points under the bar. Gate evidence is reported only as evidence.

**Cost.** It is estimated from the pilot's own logs, at the same cell, configs and episodes, and
stated in the launch record.

**Readiness review.** `/nematode-review-spec` runs before registration.

### Decision F: The harness

`scripts/analysis/logcount_contrast.py` is a thin module over G.0's helpers, with four subcommands:

- `plan`: the configs and seeds, from its own tables;
- `preflight`;
- `table`;
- `score`: manifest, completeness, gates, gap, classification, verdict, CSV.

Its manifest reads only runs whose `.exit` is 0 and whose log records the config its name claims. The
check is the same as `gap_substrate_selection.py`'s. A missing run is refused, never read as a failure.

The tests run on synthetic manifests:

- every verdict-map row;
- the D23 rule's cases;
- refusal of an incomplete or mislabelled panel;
- the seeds' disjointness from every committed G band.

## Risks / Trade-offs

- **The minimum sits above G.0's count-model anchor fraction.** G.0's panel used 0.0894, re-fixed from
  the count model's anchor. This panel uses 0.0577, the maintainer's choice on 2026-10-17. The count
  model's lead is the artefact, so re-anchoring from it would carry the artefact into the corrected
  model's bar. If the corrected model's lead is real but smaller than 0.0577, the panel reads
  `below_minimum` and the next step is a dated decision. It is never a re-read at a lower bar.
- **The null carries a sensor–motor shortcut** of about four pairs, as every gap-only null has since
  082\. The audit reports it beside the verdict; a cleaner null is G.3's question, not this panel's.
- **One cell.** A `no_wiring_effect` here does not trigger STOP by itself. The body cell decides,
  after M.1.
