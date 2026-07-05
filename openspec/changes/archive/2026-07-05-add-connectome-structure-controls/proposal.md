# Add connectome-structure controls (degree-preserving rewired-null)

## Why

The continuous-substrate architecture ranking ([Logbook 029](../../../docs/experiments/logbooks/029-continuous-architecture-ranking.md))
placed the wild-type *C. elegans* connectome **5th of 6** under PPO weight search — beaten on all
three behaviours by a plain MLP (`MLP 89.0 ≫ … > connectome 52.2 ≫ GA 15.0`). For that connectome
result to be **credible and citable** in the Phase 6a synthesis, one question must be answered: is the
connectome's standing a property of its **specific *C. elegans* wiring**, or merely of its **degree /
sparsity statistics** (any graph with the same degree sequence would rank the same)?

The standard control is a **degree-preserving rewired null** (Dhiman 2026): compare the wild-type
connectome against rewired graphs that keep each neuron's in/out degree exactly but scramble *which*
neurons connect. A naive random-rewiring null is **not** sufficient — it destroys the degree sequence
and so tests a weaker, uninteresting hypothesis. The interpretation is informative in **both**
directions:

- **connectome ≫ its rewirings** → the specific wiring is genuinely better-than-random for these
  behaviours (even while it loses to the MLP), a positive statement about biological structure.
- **connectome ≈ its rewirings** → it is the connectivity *statistics*, not the wiring, that set
  performance — the 5th-place standing is not "about" the *C. elegans* circuit.

This control was decoupled from the T8 NEAT work into **Phase 6a** (the 2026-07 endgame re-scope): it
is a **PPO weight-search ablation on the existing connectome pipeline**, needing no NEAT / GPU /
env-vectorisation. It is the credibility gate for the connectome half of the 6a synthesis (T9a).

A sibling control — making the fixed **gap-junction weights learnable** (the frozen-electrical-synapse
fairness question) — is a tracked **fast-follow** (`T7.controls.learnable_gj`), deferred from this
change: it is secondary (lower expected signal — gap junctions are a minority of the wiring and the
connectome trails the MLP by a wide margin), less biologically faithful (worm gap-junction
conductances are far less plastic than chemical synapses), and it carries the only non-trivial
correctness risk (symmetry-preserving gradient flow). This change ships the **gating** rewired-null
clean; learnable-gj is revisited if the rewired-null result or the T9a review raises the frozen-gj
objection.

## What Changes

- **Rewired-null wiring option on the existing connectome brain.** Add a `wiring` selector (`wild_type`
  | `rewired_degree_preserving`) + a `rewire_seed` to `ConnectomePPOBrainConfig`. When rewired, the
  loaded `Connectome` is transformed in-memory — its chemical-synapse and gap-junction edge sets are
  replaced by **degree-preserving edge-swapped** ones (each neuron's in/out degree preserved exactly)
  — before the topology is built. Everything downstream (PPO recipe, dims, node set/order, masks,
  projections, critic) is byte-for-byte identical; only *which* neurons are wired changes.
- **A degree-preserving rewiring utility** in the `connectome/` package — a seeded double-edge-swap
  (directed configuration-model swaps for the chemical graph, undirected for the gap-junction graph),
  hand-rolled on the seeded numpy RNG (no new graph dependency), rejecting self-loops and multi-edges.
- **A config** mirroring the `T7.connectome.c3_integrated` cell verbatim (same reward / substrate /
  PPO recipe), changing only the new `wiring` flag.
- **A control-analysis harness** that computes the C3 plateau-tail ranked-success metric for wild-type
  vs each control across paired seeds and reports the paired-seed Wilcoxon + bootstrap CI + BH-FDR
  verdict (reusing the committed `architecture-comparison-protocol` statistics layer).
- **Tests** for the rewiring invariants (degree preserved, no self-loops/multi-edges, node set fixed,
  determinism under seed) and byte-identity when `wiring: wild_type`.

## Capabilities

**Modified**: `connectome-ppo-brain` — gains the degree-preserving rewired-null wiring option,
config-gated and defaulting to the current behaviour (wild-type wiring) so existing runs are
byte-identical.

**Added**: none — no new brain family, no new environment capability. This is a substrate/analysis
control that reuses the existing connectome PPO pipeline (matching the roadmap framing that these are
"PPO-based connectome ablations on the existing pipeline … need no NEAT infrastructure").

## Impact

- **Code**: `ConnectomePPOBrainConfig` (`wiring` + `rewire_seed` fields), the connectome brain's
  load→topology seam (apply the transform), a new `connectome/rewiring.py` utility, a new
  `scripts/analysis/connectome_structure_controls.py`, and tests. Default paths unchanged →
  byte-identical wild-type behaviour (guarded by a test).
- **Configs**: `configs/scenarios/foraging_predator_thermal/` gains the rewired-null cell alongside the
  existing `connectomeppo_small_continuous2d_combined_klinotaxis.yml`.
- **Docs / tracking**: resolves `T7.controls.rewired_null` + `T7.controls.logbook`; the verdict feeds
  the 6a synthesis (`T9a`, T9.2). `T7.controls.learnable_gj` remains open as a fast-follow. No
  behaviour change to any non-connectome arm.
