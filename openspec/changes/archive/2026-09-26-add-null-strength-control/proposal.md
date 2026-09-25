## Why

Phase 8's **A.6**, added by the Wormlight review (PR #405) and required before the 8a synthesis and
A.5.

Every wiring result in this project (Logbook 034, block V, and Logbooks 070–073) compares the wild
type against a degree-preserving rewired null. That null differs from the wild type in two ways
besides which neurons connect. A.1 controlled neither.

- **Gap-junction strength moves with the edges.** The brain uses each gap junction's EM count as its
  coupling weight, normalised only by degree, and the undirected swap carries counts with the edges.
  The wild type concentrates coupling in a few hubs: ALA's gap input is 232, against at most 6.3 of
  chemical input on any neuron. On a null ALA's falls to about 3, and about half of all neurons' gap
  totals move by more than 50%.
- **Autapses are lost.** The wild type has 38. The swap never creates a self-loop but can remove one,
  and each of three nulls checked had none. The live requirement says the rewired edge sets contain
  no self-loops, so this is the specified behaviour, not a bug. What was never stated is that it
  makes the null differ from the wild type in more than degree.

Logbook 067 found that under the reading learner the wild type's advantage consisted in its gap
junctions costing it less than the rewired ones cost the null, and did not trace why. The phase's
most exposed claim, block V, has never been read against a null that holds these properties fixed.

## What Changes

- **A third wiring value, `rewired_chemical_only`.** It rewires the chemical graph with the same
  directed degree-preserving swap and holds the gap junctions and the 38 autapses at the wild type's
  placement and counts.

  - The rewiring function gains `rewire_gap_junctions` and `preserve_autapses`. Their defaults leave
    the current null byte-identical.
  - Both configuration and construction accept the new value. The measured prior and the fan-in draw
    apply to it, since chemical in-degree is preserved.

- **The rewired-null requirement is restated** to name what each null preserves and what it does not.

- **A protocol requirement:** a null states every structural property it does not preserve.

- **The control, on hard350:** wild type, the current null, and the chemical null, each learning and
  frozen, under two learners:

  - PPO at block V's committed point (edge-order draw, pooled readout, depth 4), 32 seeds;
  - the reading learner at A.2's centre, 48 seeds.

  That is 480 runs, about 10.5 hours. Only 4 configs are new.

- **The reading is fixed before launch.** One interaction per learner: the wild type's gap against
  the chemical null, minus its gap against the current null. It is scored on `auc_success`, against
  2/3 of each learner's committed wiring effect, with gates and drift first, and read through a
  verdict map.

- **A verdict attributes a gap only where one exists.** `chemical` also needs the wild type's gap
  against the chemical null to exclude zero; otherwise the verdict is `no_gap_to_attribute`.

- **The interpretation is combined.** A survival means the advantage is in the chemical wiring
  *given the wild type's gap junctions and autapses held in place*. A dissolution means it came at
  least partly from how the current null rewires gap junctions (placement and strength together) or
  drops autapses. This control cannot say which, and a split follows only if it moves.

## Capabilities

### Modified Capabilities

- `connectome-ppo-brain`: the rewired-null requirement gains the chemical-only value and states what
  each null preserves.
- `architecture-comparison-protocol`: adds the requirement that a null states what it does not
  preserve.

## Impact

- **Code:**
  - `connectome/rewiring.py` gains two keyword arguments, with the default path unchanged.
  - `brain/arch/connectome_ppo.py` gains the wiring literal, and every test for the rewired value
    becomes `wiring != "wild_type"`.
- **Scripts and configs:** `scripts/analysis/null_strength_control.py`, a generator, and 4 configs.
  A.2's, B.1b's and B.1c's helpers are reused.
- **Records:**
  - Logbook 074;
  - tracker A.6;
  - block V's standing-condition notes, resolved at every citation site: the roadmap's A.1, the
    tracker's A.1, and Logbooks 067 and 070.
