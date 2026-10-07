## Overview

The question is whether the wild type's gap-junction advantage on the thermal cell is in **where** its
gap junctions are or in **how strong** they are. The maintainer chose three settings before this change
was written:

- a new gap-only null;
- the thermal cell at target 35 only, where gap junctions carried about 84% of the lead;
- learnable multipliers starting from each wiring's own strengths.

## Decisions

### Decision A: The gap-only null

`rewire_degree_preserving` gains `rewire_chemical: bool = True`. The gap-only null sets it false and
keeps `rewire_gap_junctions` true. The chemical graph, autapses included, is the wild type's exactly.
The gap junctions are rewired by the existing undirected swap, so each neuron keeps its gap degree, and
the counts travel with their edges.

Against the wild type, therefore, the null differs in **gap placement and in each neuron's total gap
strength**, and nothing else. This is the null whose gap effect Logbook 078 measured indirectly. 078
measured it as the move from the current null to the gap-held one, against a background of rewired
chemistry.

At its defaults the rewiring function draws exactly as before; a test pins the default null.

### Decision B: Plastic gap junctions

Under `plastic_gaps` only, the topology gains `gap_log_multiplier`, an `N × N` parameter initialised
at zero. Off, no parameter exists, so checkpoints and state dictionaries are unchanged. The coupling the
forward pass uses is `g_gap ⊙ exp((P + Pᵀ) / 2)`:

- symmetric by construction;
- positive wherever `g_gap` is;
- zero wherever `g_gap` is zero.

So no pair is created, and each wiring starts from its own strengths. Both wirings carry a parameter of
the same shape, so the trainable-parameter count cannot differ by wiring (the L.1 lesson). The
parameter joins `learnable_parameters` only under `plastic_gaps`, so the off path's optimiser and
random draws are unchanged.

`plastic_gaps` is refused in two places:

- **Under leaky dynamics**, which precomputes its implicit operator from the gap matrix once.
- **Under any learning rule but PPO**, since the plastic rules define no update for gap junctions.

No arm here needs either.

**A positive control for the new component.** On the pilot, each plastic-gap learning run must differ
from its fixed-gap twin at the same seed. If the multipliers received no gradient, the two runs would be
bit-identical, since Adam's update and the gradient clip see the same norm. A difference therefore shows
the plasticity acts. A unit test also pins a non-zero gradient on the multipliers.

### Decision C: The panel

The thermal cell at target 35, under PPO at block V's committed point otherwise: edge-order draw,
pooled readout, depth 4, Cook 2019. **Seeds 513–576 (64)**, the first 64 of Logbook 078's band.

| level | wild type | gap-only null |
|---|---|---|
| `fixed` | learning, fixed gaps (reused from 078); frozen (reused) | learning, fixed gaps; frozen |
| `plastic` | learning, plastic gaps; the same frozen floor | learning, plastic gaps; the same frozen floor |

A frozen arm learns nothing, so its gaps never move. One floor per wiring therefore serves both levels.
There are 192 new runs: the null's fixed learning and frozen arms, and both wirings' plastic learning
arms.

**Three readings, corrected together** (BH-FDR on the primary metric). In each, positive means the
wild type ahead.

- `base = gap(fixed)`: the wild type's lead over the gap-only null with fixed gaps.
- `lead = gap(plastic)`: the same lead once each wiring can tune its gap strengths.
- `interaction = gap(plastic) − gap(fixed)`.

**Primary metric**: `auc_success`, with episodes reported beside.

**The minimum is 0.0577**, two-thirds of the committed 0.0865 that holding the null's gap junctions
moved the lead on this cell (Logbook 078's split). It is the cell's own committed gap effect.

**The verdict map**:

| `base` | `interaction` | `lead` | verdict |
|---|---|---|---|
| present | moves toward the null | `no_lead` or `below` | **strength**: the gap advantage was in the strengths, which learning repairs |
| present | `no_move` | remains | **placement**: the advantage survives tuned strengths |
| present | moves toward the null | remains | **partly strength**: tuning repairs some but not all |
| present | moves toward the wild type | — | **placement amplified**: tuning helps the wild type more |
| absent | — | — | **no_gap_effect**: no base to attribute, and the other readings are reported only |

Any other combination reads **`mixed`**, reported without attribution. That covers, for example, a
present base with an unmoved interaction but a lead below the minimum, or an interaction that is
significant but below the minimum. Any `unresolved` reading makes the verdict `unresolved`.

**Base-effect gate.** `base` must exclude zero above, as A.3 required. Otherwise the verdict is
`no_gap_effect`.

**Gates.** Each learning arm beats its wiring's frozen floor, and the two learning arms of a level do
not both reach 90%. The gate preflight runs on the pilot before launch.

**Sizing.** The proxy is 078's split spread, sd 0.1025 over 128 seeds. At 64 seeds the detectable effect
is about 0.032, below the 0.0577 minimum. An interaction may spread more than one gap, and the achieved
spread is reported beside.

### Decision D: Reuse and pilot

**The identity check.** Four learning and two frozen wild-type runs from the band are re-run on this
change's code, and must match 078's logs bit for bit. That licenses reusing the wild type's fixed-gap
arms. B.2a ran the same check on seeds 513–516, but on earlier code.

**The pilot.** On **seeds 1401–1404**, all six arm types, 24 runs. It does four things:

- the gate preflight on both levels;
- the plasticity check (plastic and fixed runs differ at each seed);
- the cost estimate;
- a dry run of the analysis, with no reading printed.

### Decision E: Worktree

The campaign runs from a separate git worktree on this branch, so C.1's development in the main working
tree cannot change the code or configs a running campaign reads. This is the AGENTS.md rule against
branch switches during campaigns, kept by a second checkout.

## Risks

- **Strength learning may be slow at PPO's rate.** Multipliers that barely move would read as
  `placement` for the wrong reason. The plasticity check guards against inert gaps, and the logbook
  reports how far the multipliers moved on the panel's final weights.
- **The gap-only null may barely hurt the wild type with fixed gaps.** 078's gap effect was measured
  against rewired chemistry. With the chemistry held, the base reading could fall below the minimum,
  and then the panel reads `no_gap_effect`, a result in itself.
