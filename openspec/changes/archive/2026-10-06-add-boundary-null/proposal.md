## Why

A degree-preserving rewiring can route an injected sensor straight onto a readout motor neuron. The
wild type has no motor neuron one hop from a food sensor; the current null has about nine and the
chemical-only null about eight (Logbooks 071, 074). A.2 found block V's wiring advantage depth-critical
— the null wins at a settling depth of two, where only those shortcuts reach the motor layer — and its
hop probe tied that to the shortcuts after the fact. Park (arXiv:2609.39248) found the same artefact in
an embodied fly connectome: nulls that keep every sensory-output and motor-input edge erased the
connectome's apparent difference. The 8b re-plan added a boundary-preserving null to the family as A.3,
reported beside the chemical-only primary (D21 as amended), and asked A.3 to register the hop
statistic as a predictor before reading it.

## What Changes

- **A boundary-preserving null.** `rewire_degree_preserving` gains `hold_boundary=(sensory, motor)`,
  which takes every chemical edge out of a sensory neuron or into a motor neuron out of the swap and puts
  it back unchanged. `ConnectomePPOBrain` gains `wiring: rewired_boundary_held`: the chemical-only null
  (gap junctions and autapses held) with the boundary held too, the sensory side being the 17 neurons
  any projection injects into and the motor side the 39 the readout pools. Byte-identical when off.
- **A boundary-null panel** on hard350 under PPO at block V's point: wild type, chemical-only null and
  boundary null, learning and frozen, 128 fresh seeds (641–768), 768 runs. Two readings, corrected
  together: how much holding the boundary moves the wild type's lead, and the lead over the boundary
  null itself, against 2/3 of A.6's committed lead over the chemical-only null.
- **A registered hop predictor** on committed data: across 48 seeds of A.2's and A.6's hard350 PPO
  panels, Spearman's rho between each seed's current-null one-hop count and its wiring gap, predicted
  negative, minimum |rho| 0.3. No training.

## Capabilities

**Modified**: `connectome-ppo-brain`, with one added requirement: a boundary-preserving rewired null.

## Impact

- `connectome/rewiring.py`: `hold_boundary`; `brain/arch/connectome_ppo.py`: the wiring value and
  `boundary_neurons()`
- `configs/scenarios/foraging/`: two boundary-null configs through the existing generator
- `scripts/analysis/boundary_null.py`, `scripts/analysis/hop_predictor.py`: new;
  `sensory_motor_hops.py` accepts the new null
- Tests: the null's guarantees, the pinned default null unchanged, the configs, the readings, the
  predictor's statistic and sources
- After the readout: Logbook 079, the tracker's A.3, D21's third null in the roadmap

## Breaking Changes

None. The default wiring and every existing null are byte-identical.
