# 065: Block V's Wiring Advantage Replicates on Fresh Rewirings (7a-ii V.4 / Phase 7)

**Status**: completed — **`specific_wiring_efficiency` on both cells**, and the open caveat closes.
The wild-type connectome reaches competence **+55.3%** sooner than its degree-preserving rewired null
on the thermal cell (309.22 against 691.50 episodes) and **+40.1%** sooner on the hard food-only cell
(827.75 against 1380.88), both against a registered **≥ 20%** minimum and both at **q = 0.000** on all
four efficiency metrics. Seeds **65–96** are fresh to both prior panels, so these are 32 rewired graphs
**no result in this project has used** — and both point estimates come in **above** their comparators
([V.1](057-wiring-premise-contrast.md)'s +35.4% over 64 seeds and
[V.3](058-wiring-premise-difficulty.md)'s +23.5% over 32) rather than shrinking toward the bar. Both
learning gates fire 32/32 and **no pre-update difference between the wirings was detected** on either
cell (+0.640, q = 0.164; +0.192, q = 0.327 — a failure to detect at 32 pairs, not a demonstration that
none exists); **no seed is censored** on either cell. **What this
does not do is separate the rewiring from the initialisation**: `rewire_seed` stays unset, as V.1 and
V.3 ran it, so a fresh seed moves the graph, the task draw and the initial weights together. That
stricter question remains open and unregistered.

**Branch**: `feat/l4-fresh-rewiring`.

**Date**: 2026-09-16.

**OpenSpec change**: `add-wiring-fresh-rewiring` (extends `plasticity-evaluation`: a replication states
the coupling it does and does not break, uses the original instrument unmodified, and fixes its
multi-panel disagreement rule in advance).

## Objective

Block V's learning-speed advantage is, after [L.0](064-l4-frozen-features.md), the **only surviving
wiring result in this project**. Every other reading of the wild-type connectome is negative:

| reading | finding |
|---|---|
| endpoint under gradient learning ([034](034-connectome-structure-controls.md)) | inert — no endpoint advantage |
| under local rules that **write** the wiring ([R.1c](061-l4-reduced-perturbation.md), [R.2](063-l4-eprop.md)) | actively harmful — every substrate-writing arm below the frozen control |
| as **fixed features** ([L.0](064-l4-frozen-features.md)) | indistinguishable from a degree-matched shuffle |
| **learning speed under PPO** ([V.1](057-wiring-premise-contrast.md), [V.3](058-wiring-premise-difficulty.md)) | **+35.4%** and **+23.5%** off time-to-competence |

That last row carried a defect worth taking seriously. `rewire_seed` is unset in every wiring config, so
each seed's rewired graph derives from its run seed — and **V.3's rewirings at seeds 1–32 are a subset of
V.1's at 1–64**. The project's two positive wiring results do not corroborate each other on independent
nulls; they **share** them. A shuffle that happened to be a poor graph at seed 7 was the same poor graph
at seed 7 in both records, so the second panel could not have caught it.

> *Does the advantage hold on rewired graphs no panel has used?*

## Method

Registered in `openspec/changes/add-wiring-fresh-rewiring` and committed **before any arm ran**; the
protocol is [`supporting/065-wiring-fresh-rewiring/launch.md`](supporting/065-wiring-fresh-rewiring/launch.md).

**No new configs.** All eight arms are the ones V.1 and V.3 ran, and tests assert each is byte-unchanged
since its panel ran — the thermal four against `431a4689`, the hard350 four against `48ba778c`. **256
runs**: 2 cells × 4 arms × seeds 65–96 at 3000 episodes, all succeeded.

**The instrument does not change, and a test asserts it.** `wiring_premise.py` and
`connectome_structure_efficiency.py` are byte-identical to `main`; the new
`scripts/analysis/wiring_fresh_rewiring.py` is a manifest builder and branch reporter that re-declares
none of the registered 20% minimum, the verdicts, the crossing floor or the arm mapping. A replication
that edits its own instrument cannot separate *"the instrument changed"* from *"the effect is not
there"*, and those are not separable after the fact.

Both comparators were re-scored through that unmodified harness before anything ran: V.3's panel
reproduces 892.03 against 1165.34 episodes, +23.5%, 21/32, q = 0.029 — its whole `efficiency` block
equal field for field — and V.1's pooled 64 reproduce +217.11, q = 0.001, +35.4%, 44/64 wild-better on
the primary. The instrument still reproduces both records it is replicating.

## Results

### Efficiency axis — the registered primary on both cells

| metric | cell | wild | rewired null | Δ | q | wild-better |
|---|---|---|---|---|---|---|
| `episodes_to_30pct_success` | thermal | **309.22** | **691.50** | **+382.28** | 0.000 | **24/32** |
| `auc_success` | thermal | 0.7917 | 0.6290 | +0.1627 | 0.000 | 27/32 |
| `auc_foods` | thermal | 18.179 | 16.803 | +1.376 | 0.000 | 25/32 |
| `episodes_to_90pct_foods_plateau` | thermal | 445.59 | 986.75 | +541.16 | 0.000 | 22/32 |
| `episodes_to_30pct_success` | hard_food | **827.75** | **1380.88** | **+553.13** | 0.000 | **28/32** |
| `auc_success` | hard_food | 0.4403 | 0.3610 | +0.0793 | 0.000 | 29/32 |
| `auc_foods` | hard_food | 17.393 | 16.784 | +0.609 | 0.000 | 31/32 |
| `episodes_to_90pct_foods_plateau` | hard_food | 999.25 | 1432.44 | +433.19 | 0.000 | 26/32 |

**Time-to-competence gain: +55.3% (thermal) and +40.1% (hard_food)**, both above the registered 20%
minimum. **Crossing rates are 100% in all four arms** — no seed sits at the 3000-episode cap on either
cell, so nothing is censored and no mean is taken on trust. Verdict **`specific_wiring_efficiency`** on
both cells. Per-seed values: [`per-seed-primary.csv`](supporting/065-wiring-fresh-rewiring/per-seed-primary.csv).

### Peak axis, and the gates and priors read before the contrast

| test | cell | contrast | role | Δ | q |
|---|---|---|---|---|---|
| V5 | thermal | `wt_ppo − rn_ppo` | secondary | +0.091 foods | 0.047 |
| V6 | thermal | `wt_ppo − wt_frozen` | **gate** | +16.864 foods, 32/32 | 0.000 |
| V7 | thermal | `rn_ppo − rn_frozen` | **gate_null** | +17.413 foods, 32/32 | 0.000 |
| V8 | thermal | `wt_frozen − rn_frozen` | **prior** | +0.640 | 0.164 |
| V9 | hard_food | `wt_ppo − rn_ppo` | **primary (peak)** | **+8.267 points, 29/32** | 0.000 |
| V10 | hard_food | `wt_ppo − wt_frozen` | **gate** | +76.617 points, 32/32 | 0.000 |
| V11 | hard_food | `rn_ppo − rn_frozen` | **gate_null** | +68.542 points, 32/32 | 0.000 |
| V12 | hard_food | `wt_frozen − rn_frozen` | **prior** | +0.192 | 0.327 |

Both cells' learning gates fire at 32/32 seeds, so both arms plainly learned and the contrast is
interpretable. **Neither untrained prior detects a pre-update difference between the wirings**, which
matters more here than usual, because a fresh-seed panel varies initialisation along with the graph. It
is worth being exact about what that buys: a non-significant prior at 32 pairs is a **failure to
detect** a pre-update difference, not evidence that none exists, and the panel was powered for the
contrast rather than for this check. It removes the crudest confound; it does not make the contrast
unconfounded.

**The thermal peak axis returned `saturated`** — 96.14% and 94.14% full clear, above the harness's 90%
ceiling. That is the registered, documented reason the efficiency axis is primary on that cell, and it
now holds outside the seeds it was decided on. **The hard_food peak axis improved on V.3**, returning
`specific_wiring` at +8.267 full-clear points where V.3 got `below_min_effect` at +4.05 against a +5.00
minimum.

## What this establishes, and what it does not

1. **The caveat closes.** Block V's positive is now **independent in rewiring**: 32 graphs fresh to both
   prior panels, on two cells, with both estimates above their comparators rather than shrinking. V.1's
   own panels spread 15 points (+46.4%, +32.6%, +31.8%), so +55.3% and +40.1% sit inside that
   instrument's demonstrated variability on the high side.
2. **It does not attribute the effect to the rewiring alone.** `rewire_seed` stays unset — deliberately,
   because it is the coupling V.1 and V.3 ran under and removing it would not be replicating them. So a
   fresh seed moves the graph, the task draw **and** the initial weights together. What is excluded is
   that the committed figures rode on a particular set of shuffles; what is **not** excluded is that
   initialisation contributes. Isolating the graph needs `rewire_seed` pinned across seeds, which is a
   different experiment that neither V.1, V.3 nor this change registered. **The priors at both cells are
   the only evidence here bearing on it: they detect no pre-update difference, which is a check rather
   than a decomposition, and at 32 pairs a failure to detect rather than a demonstration of absence.**
3. **The phase now holds one positive and one systematic negative, and they are about different things.**
   The wiring buys **learning speed under PPO** — a learner the animal cannot host. It buys nothing as an
   endpoint (034), nothing under rules that write it (R.1c, R.2), and nothing as fixed features (L.0).
   This replication strengthens the first without touching the second, and V.2 still offers **no
   mechanism**: 64 rewirings against four graph properties fixed in advance predicted nothing.
4. **No mechanism claim is made or implied.** That the advantage is real and reproducible on fresh graphs
   says the wild-type edges matter to PPO's search; it does not say which edges, or why.

## Two driver defects this panel exposed

Neither changed a number, and both were in the new reporting driver rather than in the committed
harnesses — but both would have misreported the result, so they are recorded.

**1. A verdict printed off the seed count.** At n pairs the smallest achievable one-sided exact p is
`2**-n`, so at the pilot's **4 pairs** it is 0.0625, above q = 0.05, and nothing can clear the
significance level. The harness duly returned `no_learning` on gates of +14.95 and +77.37 at 4/4, and
`degree_statistics` on efficiency gains of +48.1% and +27.7% — both **above** the registered minimum. The
driver mapped that straight through and printed **"does not replicate" for both cells**; taken at face
value on a pilot, it would have withdrawn V.1 and V.3. It now consults its own power arithmetic before
assigning any branch and withholds all of them when the level is unreachable. **This was the second
appearance of this defect class** — L.0's harness printed a void verdict on its own 4-seed pilot — so it
is now a test.

**2. The registered result came back as "unrecognised".** The branch map was keyed on `specific_wiring`,
but the efficiency axis — the registered primary on **both** cells — emits `specific_wiring_efficiency`.
The test meant to prevent exactly this fork had the harness's vocabulary **hand-copied into it**, so it
checked the driver against memory rather than against the harness. Deriving the vocabulary from the
harness source instead surfaced **three** unhandled verdicts: `specific_wiring_efficiency`,
`rewired_beats_wildtype` (significant in the **reverse** direction — now a failure branch stronger than
non-replication) and `inconclusive` (not significant with an interval not bracketing zero — not a
failure, since failing to place an effect is not placing it at zero). **The latter two were registered
after the data, and neither occurred**, so the registration cannot have shaped this result; it is
recorded here rather than left implicit. A hand-copied vocabulary is the fork it was meant to prevent.

## Registered consequences

Task 5.4 — the correction pass across V.1, V.3, [059](059-7a-shipment.md) and the roadmap had this
failed on both cells — **does not apply**: it replicated on both. Task 5.5 applies, and is discharged
above and in the roadmap: the caveat closes, block V's positive is independent in rewiring, and the
record stays explicit that rewiring and initialisation still vary together.

*(Promoted 2026-09-19 at the Phase 7 close, [Logbook 069](069-phase7-synthesis.md): this is no longer
an open caveat but a **standing condition**.)* Two things changed its weight. After
[L.1b](068-l1b-rate-calibration.md) found L.1's interaction reversing with the learning rate, block V
is the phase's strongest citable result, so the confound sits under the headline rather than beside
it. And [Dhiman 2026](https://arxiv.org/abs/2604.04033) reports the fly connectome's apparent
advantage **dissolving under shared initialisation plus a degree-preserving null** — the same control,
on the same kind of claim, in another organism. So the condition is now carried **in the same sentence
as the claim** at every citation site, the effect size is not to be quoted without it, and the control
is **Phase 8's first act**. It needs its own design decision first: "the same initialisation" has no
single meaning once the mask changes, since the init scale is `1/sqrt(chemical in-degree)` and a
degree-preserving rewiring preserves the degree sequence but not which neuron holds which degree.

## Artefacts

- [`supporting/065-wiring-fresh-rewiring/launch.md`](supporting/065-wiring-fresh-rewiring/launch.md) — the protocol, both instrument checks, the pilot
- [`supporting/065-wiring-fresh-rewiring/fresh_rewiring.json`](supporting/065-wiring-fresh-rewiring/fresh_rewiring.json) — the full reading, including the harness's own report
- [`supporting/065-wiring-fresh-rewiring/per-seed-primary.csv`](supporting/065-wiring-fresh-rewiring/per-seed-primary.csv) — one row per cell, arm and seed, with the censoring column
