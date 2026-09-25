## Context

The degree-preserving null has been this project's control since Logbook 034. The Wormlight review
found it differs from the wild type in gap-junction strength and in autapses as well as in placement,
and both were verified on the code (PR #405). A.1 shared the chemical weights between the two wirings
and left both of these properties alone.

Decisions already taken:

- **One combined null.** Gap junctions and autapses are held together, which is Wormlight's primary
  null, so the two projects compare. A split comes only if it moves.
- **Both learners.** PPO first, then the reading learner.

## Goals / Non-Goals

**Goals.** Read block V's wiring effect against a null that differs from the wild type only in
chemical placement, on both learners, with the reading fixed before launch.

**Non-Goals.**

- Separating gap strength from gap placement, or either from autapses: that is the follow-up if this
  control moves.
- A strength-preserving gap swap.
- Thermal.
- Changing the default null. Every committed result stays reproducible.

## Decisions

### Decision A: The chemical null, and a default that does not move

`rewire_degree_preserving(connectome, rng, swaps_per_edge=10, *, rewire_gap_junctions=True, preserve_autapses=False)`:

- **`preserve_autapses=True`:** self-loops are taken out of the directed swap's edge list before
  swapping and re-added unchanged afterwards. In- and out-degree are still exact, because each
  autapse contributes one to both and is untouched.
- **`rewire_gap_junctions=False`:** the undirected swap is skipped, so the gap junctions are the wild
  type's, placement and counts alike.
- **At the defaults**, the function takes the same draws in the same order as today. A test asserts
  the default null is unchanged at fixed seeds, edge by edge and count by count.

**The two nulls are different random chemical graphs at the same seed.** The chemical swap runs
first and the gap swap after it (`rewiring.py`), and taking the autapses out of the directed list
changes what the chemical swap draws. So at one seed the chemical null's chemical graph is a
different sample from the full null's, not the same graph with gap junctions and autapses restored.
This is no bias, since each null is a random draw and the difference averages over seeds, but it
adds graph-sampling variance to the interaction and it is stated as such. The exactly paired
alternative is `rewire_gap_junctions=False` alone: with autapses still swapped away, the chemical
swap draws exactly as the full null's does and reproduces its chemical graph, so the two nulls would
differ in their gap junctions and nothing else. It isolates gap junctions only, and the combined
null was chosen for comparability with Wormlight. **It is the registered first follow-up if the
combined control moves**, and the natural first split between the two causes.

`wiring: "rewired_chemical_only"` calls the function with both arguments flipped. Every place that
tests for the rewired value tests `wiring != "wild_type"` instead: the rewiring step, and the
measured prior's `rewired=` flag. Both nulls keep each neuron's chemical in-degree, so the per-neuron
placement of a measured prior and the fan-in draw apply unchanged. `rewire_seed` works as it does
today.

### Decision B: The panel

| learner | point | wirings (each learning and frozen) | seeds | runs |
|---|---|---|---|---|
| PPO | block V's committed hard350 point: `edge_order`, pooled readout, depth 4 | wild type, full null, chemical null | 305–336 (32) | 192 |
| reading | A.2's centre: `readout_only`, `edge_order` | same | 337–384 (48) | 288 |

**Configs.** The wild-type and full-null arms are committed configs, reused unchanged: block V's four
hard350 arms, and A.2's reading centre. The 4 new configs are the chemical-null arms, learning and
frozen, each derived from its full-null parent with `wiring` changed and nothing else. The committed
spellings are not uniform — PPO puts `_frozen` after the wiring tag, the reading family puts it
before — so the new stems follow each family's own order and come from an explicit table, not a rule:

| learner | arm | full-null parent (committed) | chemical-null stem (new) |
|---|---|---|---|
| PPO | learn | `…_hard350_rewired_null` | `…_hard350_rewired_chemical_null` |
| PPO | frozen | `…_hard350_rewired_null_frozen` | `…_hard350_rewired_chemical_null_frozen` |
| reading | learn | `…_hard350_eprop_readout_only_rewired_null` | `…_hard350_eprop_readout_only_rewired_chemical_null` |
| reading | frozen | `…_hard350_eprop_frozen_rewired_null` | `…_hard350_eprop_frozen_rewired_chemical_null` |

**Levels.** Two, `full` and `chemical`. The wild-type arms belong to both, since they are the same
runs, and appear under both levels in the manifest. Each level is gated against its own floors, so
the chemical null has its own frozen floor.

**Cost.** At B.1b's measured per-run times, about 2.7 hours for PPO and 7.7 for the reading learner.
The seed bands are fresh, checked by a test against every earlier band.

### Decision C: The interaction, the metric and the sensitivity

**The interaction per learner** is `gap(chemical) − gap(full)`, paired by seed with A.2's
`interaction`. The wild-type arms are shared, so it equals `full null − chemical null` seed by seed.
It is positive when the wild type's lead is larger against the chemical null.

**`auc_success` is the primary on both learners,** registered here with its reason:

- one metric for the verdict on both learners;
- on hard350, A.1's data put `auc_success`'s sensitivity at or above the episode metric's.

`episodes_to_30pct_success` is reported beside it, with the censoring rule's own choice recorded next
to it.

**Sensitivity** comes from frozen committed data, as `2.487 × sd / √n`:

| learner | source of the spread | sd | n | MDE | reference effect | MDE ÷ reference |
|---|---|---|---|---|---|---|
| PPO | A.1's hard350 per-seed interactions (Logbook 070) | ≈ 0.070 | 32 | ≈ 0.031 | +0.061 (A.1, `edge_order`) | ≈ 0.51 |
| reading | B.1c's reading per-seed interactions (Logbook 073) | 0.39–0.42 | 48 | 0.14–0.15 | −0.2105 (A.2 centre) | 0.66–0.71 |

**The proxy is not obviously conservative.** The wild-type noise cancels in this interaction, which
would make its spread smaller than A.1's. But the two nulls are different random chemical graphs at
each seed (Decision A), which adds graph-sampling variance back. The launch record states the exact
figures from the committed CSVs and does not claim the proxy errs in either direction.

### Decision D: Minimum, states, verdicts

**The minimum** is 2/3 of each learner's committed reference effect, in both directions: 0.041
`auc_success` for PPO and 0.140 for the reading learner. B.1c's `classify` is imported unchanged, so
the states and their three `unresolved` cases are the ones already registered and tested.

| state | verdict | reading |
|---|---|---|
| `no_move` | **chemical** | Against a null holding the wild type's gap junctions (placement and strength) and autapses, the wiring gap moves by less than the minimum. The wiring gap is in the chemical wiring *given those held in place*. Requires the attribution gate below; otherwise **no_gap_to_attribute**. |
| `move_null` | **gap_or_autapse** | The wiring gap shrinks toward the null by at least the minimum. Part of the advantage came from how the current null rewires gap junctions (placement and strength together) or drops autapses. This control cannot say which, and the follow-up splits them. |
| `move_wt` | **amplified** | The chemical null does worse than the full null by at least the minimum. The current null's rewiring was hiding part of the advantage. |
| `below` | **below_minimum** | Licenses nothing on its own. |
| `unresolved` | **unresolved** | Reported with the MDE beside it. |

**The verdicts speak of the wiring gap, not "the advantage".** Under PPO the gap is the wild type's
lead; under the reading learner it is the null's lead, since the reference effect is negative there.
Orientation is unchanged — a positive interaction means a better wild-type position — so on the
reading learner `move_null` means the null's lead grows against the chemical null.

**The attribution gate.** `chemical` attributes a gap, so there must be one to attribute: the gap
against the chemical null must have its interval excluding zero on the reference effect's side (above
zero for PPO, below for the reading learner). Where it does not, the verdict is
**no_gap_to_attribute**, reported with both levels' gaps: neither null is separated from the wild type
in this campaign, and the interaction says nothing about where a gap lives. Block V's claim is named
only in a PPO `chemical` verdict that passes this gate.

**The family** is the two primary interactions, one per learner, under BH-FDR with the two-sided
folded p.

### Decision E: Gates

These are B.1c's gates, reused:

- every learning arm must beat its own frozen floor, and a level where either wiring fails or both
  saturate is unreadable;
- the reading learner's chemical matrix must not drift on any scored seed, or that half is void
  (`honour_drift`).

### Decision F: Reuse

`scripts/analysis/null_strength_control.py` holds only this panel: stems, seeds, levels, the verdict
map, and scoring glue. It reuses:

- **A.2's module:** `score_level`, `learning_gates` with `floor_level`, `interaction`, `wiring_gap`,
  `censoring_rates`, `choose_metric`, `apply_family_correction`, `substrate_drift`;
- **B.1c's module:** `classify`, `honour_drift`;
- **B.1b's module:** the manifest format, through `build_manifest(arm_by_stem=…)` and
  `require_complete(levels=…)`.

The generator writes the four configs from this module's panel definition.

**Described, not registered:** A.2's hop probe (`scripts/analysis/sensory_motor_hops.py`) is run on the
chemical null. The graph the simulation propagates through includes gap junctions, which this null
keeps, so the count of motor neurons one hop from a food sensor shows how much of A.2's depth
mechanism survives here. It enters the logbook as description and no verdict reads it.

## Risks / Trade-offs

- **The combined control cannot attribute.** Stated in every verdict, and a split follows only if it
  moves.
- **The chemical null keeps autapses that the spec's current wording forbids in a rewired set.** The
  requirement is restated so each null says what it preserves, and the full null keeps its current
  behaviour.
- **Block V's reference comes from A.1, not from this campaign's own full-null gap.** This avoids
  sizing the bar on the result. The record reports the in-campaign gap beside it.
