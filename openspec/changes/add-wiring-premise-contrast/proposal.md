# The wiring contrast on a behaviour the circuit is wired for (7a-ii V.1)

## Why

Phase 7's MUST is whether the wild-type *C. elegans* wiring is load-bearing. Eight logbooks answer
no, and [Logbook 056](../../../docs/experiments/logbooks/056-l4-ladder-reread.md) has just sorted
what those answers are evidence about. Five of the 32 registered contrasts turned out to be
uninformative about both the wiring and the instrument, because the premise they rest on — that
learning finds a wild-type advantage — has never been demonstrated by any method tested.

**Every one of those results, and every wiring contrast in the project, ran on one cell.** The
integrated C3 cell is food chemotaxis *plus predator evasion plus thermotaxis*, 2400 steps an
episode. [Logbook 034](../../../docs/experiments/logbooks/034-connectome-structure-controls.md)
recorded the limitation itself, in its own words: "**Single task** (the continuous integrated-C3
cell); the result is for this multi-objective foraging/predator/thermotaxis demand, not a universal
statement."

And the deficit has a location. Both architecture rankings put the connectome **competitive on
foraging and behind on predator evasion** ([025](../../../docs/experiments/logbooks/025-weight-search-architecture-ranking.md),
[029](../../../docs/experiments/logbooks/029-continuous-architecture-ranking.md)) — the least
worm-like of the cell's three demands, and one the animal's circuit was under no pressure to solve.
The wiring contrast has only ever been measured with that component in the mix.

So the premise has never been tested where it has the best chance of holding: on a **single
behaviour the animal actually performs**, which this repository has already validated against the
real worm at the strategy level ([035](../../../docs/experiments/logbooks/035-realworm-chemotaxis-validation.md)
reproduces both klinokinesis and the klinotaxis weathervane;
[036](../../../docs/experiments/logbooks/036-realworm-thermotaxis-validation.md) reproduces the
thermotaxis weathervane), under an optimiser known to learn on this substrate.

**This also gates 7b.** Shipment 7b's MUST is a comparative cross-connectome sweep on exactly these
two behaviours — klinotaxis and thermotaxis — with PPO as secondary context. If the wiring contrast
is null on both *within one species, under a working optimiser, on the cell matched to the
behaviour*, then 7b's central measurement has no within-species signal to find and eight to twelve
weeks of pipeline would buy a null by construction. That is worth knowing for the cost of a few
hours of compute.

## What Changes

- **Two wiring contrasts under PPO**, one per behaviour, on the single-behaviour cells: the C1
  klinotaxis foraging cell (`max_steps: 800`) and the thermal cell, wild type against its
  degree-preserving rewired null, 16 paired seeds each.
- **A learning gate per cell, run and reported before its contrast is read.** A contrast against a
  null presupposes that the arm learns; on this substrate that has to be shown per cell, not
  assumed. Both wirings are gated against their own frozen-weights floor on the same seeds.
- **A saturation clause, registered in advance.** An easier cell can put both arms on the ceiling,
  where no contrast can resolve. The threshold and the named remedy are fixed before the pilot runs,
  not improvised after it.
- **A pilot on disjoint seeds** that measures per-run wall time on these exact cells and checks that
  the inherited C1 recipe converges on the connectome — a check the committed config header flags as
  still open — before any registered seed is spent.
- **An MLP reference arm per cell**, descriptive, so the record states what the cell's ceiling is
  rather than inferring it.
- **What each verdict licenses**, registered before any data exists, including what a positive would
  mean for the phase's headline and for 7b's gate.

Out of scope: any plasticity rule (this runs PPO only), the instrument ladder that follows it, the
7a shipment decision, and any change to a committed verdict.

## Capabilities

**Modified**: `architecture-comparison-protocol` (a wiring contrast is run on a cell matched to the
behaviour under claim, and gated on that cell showing learning).

## Impact

- New: four connectome configs (two rewired-null, two frozen-control), the campaign harness, the
  records under `supporting/057-wiring-premise-contrast/`, Logbook 057.
- Edited: the experiments index, `CHANGELOG.md`, tracker (new block V), roadmap.
- Compute: 160 registered runs plus a pilot, on cells measured at roughly a tenth of C3's cost.
