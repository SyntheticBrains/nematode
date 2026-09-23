## Why

Phase 8's **B.1c**, the last required item before shipment 8a, which ships on A.1, A.2 and B.1. It
asks D17's question: **do the animal's own synaptic weights make its wiring legible where random
ones did not?**

Every connectome brain in this project has been anatomically constrained in topology and randomly
initialised in weight. Block V's wiring effect, and every null before it (Logbook 034's
degree-statistics verdict among them), was read on random weights.

B.1a vendored the Creamer–Leifer–Pillow fitted weights (bioRxiv 2024.09.22.614271, a preprint). B.1b
(Logbook 072) showed they leave the pathway learnable under both learners at every scale tried, and
selected the magnitude-matched multiplier, 1.0, on both. B.1b also defined measured priors under the
per-neuron fan-in draw, which D17 requires for a PPO arm, because under PPO a measured prior is an
initialisation.

## What Changes

**The 2×3 on hard350.** Wiring {wild type, rewired null} × prior {random, measured,
measured-shuffled}, each arm learning and frozen, with every prior level gated against its own
frozen floor.

- **PPO:** under `per_neuron_fanin` with the pooled readout, 32 seeds (225–256).
- **The reading learner (`readout_only`):** at A.2's centre under `edge_order`, 48 seeds (257–304).
- **Size:** 960 runs, about 21 hours.
- **Configs:** only the 8 `measured_shuffled` configs are new. The random and measured levels are
  B.1b's committed configs, reused unchanged.

**Two registered interactions per learner, with the random prior as reference:**

1. **Measured × wiring:** does the measured prior move the wiring gap?
2. **Placement × wiring (measured against shuffled):** does it matter which synapse carries which
   fitted value? Without this contrast, a positive could be a fact about the value distribution, not
   about the wiring.

**Sensitivity and minimum effect:**

- **Sensitivity:** taken from B.1b's committed per-seed data. The panel detects about 0.7–0.8 of each
  learner's committed wiring effect.
- **Minimum:** a registered 2/3 of that effect, in both directions.
- **Verdicts:** read against a verdict map fixed before launch.

**A protocol requirement.** A measured-weight positive is reported as legibility only when its
placement-shuffled control does not move the contrast as far. Otherwise it is a value-distribution
effect.

## Capabilities

### Modified Capabilities

- `architecture-comparison-protocol`: adds the placement-control requirement.

## Impact

- **Code:** no brain changes.
  - New: `scripts/analysis/measured_prior_contrast.py`, holding the panel, the scorer and the verdict
    map.
  - The config generator is extended to write the shuffled arms.
  - Reused: A.2's gate, interaction, family correction and drift check, and B.1b's panel
    definitions.
- **Configs:** 8 new files under `configs/scenarios/foraging/`.
- **Records:** Logbook 073 with its supporting files; tracker B.1c; the roadmap's B.1 exit line, D17
  and the novelty map; the index row. Citation sites are conditioned where the result demands it:
  Logbook 034's degree-statistics verdict if the result is null, and block V's claim if measured
  weights hide the wiring.
