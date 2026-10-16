## Context

G.0 asks one question before any mechanism is sought: **is the gap-placement effect a property of
the wiring, or of how this model scales gap junctions?** The roadmap fixes the arms and the three
verdicts (`survives`, `artefact`, `unresolved`); this design fixes how each arm is built and read.

Facts the design rests on (Emmons 2024 unless stated): 1,095 gap pairs; counts with a median of 2, 96
pairs at ≥ 10, 26 at ≥ 20, 9 at ≥ 50, 6 at ≥ 100 (ALA–CANL/R 401, ALA–PVDL/R 250, CANL–PVDL and
CANR–PVDR 100); the gap matrix enters the settling map as `preact = W_chemᵀh + Gᵀh` with
`G_ij = count_ij / sqrt(d_i d_j)`, `d` the binary gap degree (`connectome_ppo.py`, the gap build).

## Goals / Non-Goals

**Goals:** the arms the roadmap names, buildable and byte-identical at their defaults; a readout that
gives one of the three verdicts by a rule fixed before the panel runs; the matched-statistics table and
shortcut audit D24 requires.

**Non-Goals:** the totals-held, sizes-permuted, boundary-held and class nulls (G.1–G.3, their own
changes); any substrate change an `artefact` verdict would name (G.0-A).

## Decisions

### Decision A: The gap-weight options act before the normalisation, the scale after

`unit` and `log_count` replace each pair's count before `1/sqrt(d_i d_j)`, so the normalisers are the
wiring's own (the binary degrees are unchanged by either). `gap_weight_scale` multiplies the
normalised matrix, so ×0.25 is a quarter of today's coupling everywhere. Under a rewired null the counts
travel with their edges first and the mode then acts, so a unit-weight gap-only null differs from the
unit-weight wild type in placement alone. `plastic_gaps` multiplies whatever matrix the options give.
Rejected: re-normalising after `unit` to match today's row sums (it would re-introduce the count
structure the arm exists to remove).

### Decision B: The outliers are defined by count, and scattered by focal swaps

`gap_outlier_min_count` (default 20, 26 pairs on Emmons, 24 on Cook) names the outliers. **Held**: the
undirected swap draws both edges from the non-outlier pairs only, so every outlier keeps its place and
count. **Scattered**: each accepted swap draws its first edge from the outliers and its second from the
non-outliers, `swaps_per_edge × k` swaps for k outliers, so each outlier's count lands on a new pair and
each displaced partner moves once; degrees are preserved by construction. The count of moved
non-outlier pairs is reported. Rejected: permuting outlier counts onto other existing pairs with
placement held (that is G.1's sizes-permuted family, and it would test a different thing).

### Decision C: Gap-off is one arm

With gap junctions off, the wild type and the gap-only null are the same brain (the null differs in
gap junctions only), so gap-off is one learning arm, read against the count-weight wild type
("do the junctions help?") and the count-weight gap-only null ("do the null's junctions hurt?").

### Decision D: The panel and its readings

Thermal target 35 on Emmons 2024, depth 4 unless stated, PPO at block V's point otherwise:

| level | wild type | null | seeds | role |
|---|---|---|---|---|
| `count` | learn + frozen | gap-only, learn + frozen | 2401–2464 | the reference on the panel's own seeds |
| `unit` | learn + frozen | gap-only, learn + frozen | 2401–2464 | **verdict** |
| `outliers_scattered` | the `count` wild type | scattered null, learn + frozen | 2401–2432 | **verdict** |
| `outliers_held` | the `count` wild type | held null, learn + frozen (16) | 2401–2432 | beside |
| `gap_off` | one arm, learn + frozen (16) | — | 2401–2432 | beside (Decision C) |
| scale ×0.25, ×0.5, ×2; `log_count` | learn | gap-only, learn | 2401–2416 | beside (profile) |
| depth 2, 3, 6 | learn | gap-only, learn | 2401–2416 | beside (M.6's thermal depth check) |

About 896 runs. The two deciding levels carry frozen floors on every seed, so their gates read on the same seeds as their readings; the beside levels carry floors on a 16-seed subset (marked 16), and the harness gates each level on the seeds its floors cover. Readings, each a paired `wild type − null` gap on `auc_success`, classified by
`mc.classify` at the minimum `m` D23 fixes (0.0577, carried if the anchor's lead sits in [0.058, 0.13] and both
anchor levels are readable; ⅔ of the anchor's lead if the lead excludes zero but sits outside that
range; 0.0577 again if the anchor's interval includes zero, in which case the panel's own `gap(count)`
decides `no_gap_effect`), corrected together:
`gap(count)`, `gap(unit)`, `gap(outliers_scattered)`, `gap(outliers_held)`.

**Verdict map (fixed here, before any run):**

Each deciding reading is first put in one of three classes: for `gap(unit)`, **holds** (`move_wt`),
**fails** (`no_move`, `below` or `move_null` — the null level with or ahead of the wild type once counts
are removed) or **open** (`unresolved`); for `gap(outliers_scattered)`, **reproduces** (`move_wt`),
**does not** (`no_move`, `below` or `move_null`) or **open**. Every combination is named:

| `gap(unit)` ↓ / `gap(outliers_scattered)` → | does not | reproduces | open |
|---|---|---|---|
| **holds** | `survives` | `both` | `unresolved` |
| **fails** | `artefact` | `artefact` | `artefact` |
| **open** | `unresolved` | `unresolved` | `unresolved` |

- `survives` licenses G.1–G.3b at the committed scale and depth.
- `both` — placement matters without the counts, *and* moving a few count outliers alone costs a
  cell-scale lead — licenses G.1–G.3b, with the outlier pairs registered as their own class in G.3.
- `artefact` re-aims block G (the tracker's G.0-A): once counts are removed the lead is gone, whatever
  the outliers do.
- `unresolved` licenses nothing by itself; how block G proceeds is a dated decision in the tracker and
  the roadmap.

`gap(count)` must exclude zero above, or the panel reads `no_gap_effect` and nothing else is read
(the effect did not reproduce on Emmons). This is the roadmap's two-thirds rule in the project's
`classify` vocabulary: `m` is two-thirds of the committed reference effect.

**Described beside, never verdicts:** `gap(outliers_held)`; gap-off against both count arms; the scale
profile and `log_count`; the depths; episodes to competence; the censoring rule's choice; the collapse
statistic if M.1 has landed it, otherwise a note that no panel here exceeds 3,000 episodes.

### Decision E: The saturation predictor, registered before it is computed

On untrained brains (the probe of `sign_scale_calibration.py`: 64 fixed inputs, a unit saturated at
`|h| > 0.99`), for each of Logbook 082's 64 committed gap-only draws on Cook 2019 (seeds 513–576): the
saturated fraction over the injected sensors, the command interneurons (AVA, AVB, AVD, AVE, PVC) and
the readout motor neurons, null minus wild type. **Predicted**: Spearman ρ with that seed's committed
`auc_success` gap (`fixed_gap_auc_success` in Logbook 082's `per-seed.csv`) > 0 (more saturation on the path in the null, a larger wild-type lead), minimum
|ρ| 0.3, two-sided permutation p and an 80% bootstrap interval, as A.3 registered its predictor
([Logbook 079](../../../docs/experiments/logbooks/079-boundary-null.md)); repeated on the panel's
Emmons draws once they read. States: `predicts` (ρ ≥ 0.3, p < 0.05), `no_prediction` (interval inside
±0.3), `unresolved`.

### Decision F: The matched-statistics table and the shortcut audit

For every gap wiring at every panel seed (D24): the effective maximum and 99th percentile of
`G`, the effective per-neuron totals for ALA, CANL/R, PVDL/R, AVAL/R and AVBL/R, the number of
sensor–motor gap pairs, the count–degree Spearman correlation, and from `sensory_motor_hops.py` the
one- and two-hop sensor-to-motor reach over chemical plus gap edges on the thermal cell's injected
sensors. Each wiring's launch-record entry states what it does not preserve.

### Decision G: Order and gates

The anchor pilot runs first on 9801–9816 (wild type and gap-only, count weights, learning and frozen);
it fixes `m` by D23's rule and is the gate preflight's evidence for the `count` level. The other levels
are new wirings with no committed runs, so a two-seed smoke per new level (seeds 9851–9852, recorded in the tracker's seed-band
table; not a pilot, discarded) checks they execute, and their floors are read in the panel itself with the rule that
a level failing its floor is reported unreadable rather than read. The launch record is committed
before any panel run.

## Risks / Trade-offs

- **The effect may not reproduce on Emmons** (four gap pairs differ, in the layer under test) — then
  the panel reads `no_gap_effect`, which is itself D23's answer, and the roadmap's risk row applies.
- **A scattered swap can fail to place every outlier** when partners are scarce — the accepted-swap
  count is logged and reported; fewer than `k` accepted swaps is reported in the table, not hidden.
- **`unit` changes the operating point** (total gap drive falls by about the count mean) — that is the
  point of the arm, and the scale profile beside it shows how the lead moves with drive alone.
