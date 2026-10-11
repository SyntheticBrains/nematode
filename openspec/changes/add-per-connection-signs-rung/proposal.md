## Why

Phase 8 asks of each substrate rung whether the wild-type wiring starts to matter once that piece is
real. Synapse signs are a piece the connectome brain still approximates. Two models exist today:

- **Random signs** (`synapse_signs: random`, the default): each sign is drawn.
- **Per-neuron atlas signs** (`synapse_signs: atlas`): a synapse is signed by its presynaptic cell's
  transmitter, so every glutamatergic synapse is excitatory.

H.2 vendored a per-connection sign table. It takes a cited physiology measurement first (51
connections), then Fenyves et al. 2020's expression-based polarity (1,676), then the per-neuron rule
(1,449), and gives no sign to the remaining 533.

- **Coverage is the same as the atlas's:** 3,176 of 3,709 chemical synapses.
- **What differs:** **342 signs**, the connections receptor expression or physiology flips. AWC → AIY,
  which is glutamatergic and inhibitory, is among them.
- **No brain reads the table yet.**

B.3 (Logbook 090) found that grounding signs changes what a learner makes of a circuit. That makes
this the substrate's nearest unrun fidelity step. It is also a cheap rung before the decision on M.9,
the thermal body cell.

## What Changes

- **`synapse_signs: per_connection`** on the connectome PPO brain. On the wild type, each synapse
  takes its per-connection sign. Drawn magnitudes and the RNG stream are unchanged, as under the atlas
  model.
- **Rewired nulls carry signs with their edges.** A null's edge keeps its presynaptic neuron and takes
  a new target in each swap. It carries the wild-type sign of the edge whose slot it took, as it
  already carries that edge's synapse count. Each neuron keeps its set of outgoing signs; only their
  placement differs. Under the atlas model this rule reproduces today's signs exactly.
- **The rewiring records each null edge's origin,** the wild-type edge whose slot it took. It is
  recorded without changing the draws.
- **Enforcement and every sign-reading path accept `per_connection`** wherever they accept `atlas`:
  - PPO's sign hold;
  - the plastic rules' projection;
  - the validators;
  - the checkpoint's sign buffer.
- **A registered rung** on the thermal cell at target 35 under PPO, run as a 2×3:
  - signs {random, atlas, per-connection} × wiring {wild type, chemical-only null}, each with a
    frozen floor;
  - three readings on the wild type's lead over the null: **the lead** under per-connection signs,
    **grounding** (per-connection − random) and **receptor correction** (per-connection − atlas),
    Holm-adjusted together.
  - At random signs that lead is +0.016 here, so the rung asks whether grounded signs make the
    chemical wiring matter. The cell's large wiring effect is gap placement, which signs do not
    touch.
- **Beside it:** B.3's two acute knockouts re-read on a modulated connectome with per-connection
  signs, on B.3's seeds, never as verdicts.

## Capabilities

**Modified**:

- `connectome-ppo-brain`: the per-connection sign model; signs travelling with a null's edges.
- `connectome-substrate`: the rewiring's edge origins.

## Impact

- `connectome/rewiring.py`: the optional origin record.
- `brain/arch/connectome_ppo.py`: the sign model, the null's signs, validators, and the three-factor
  hand-over.
- `scripts/campaigns/`: the generator for the rung's 12 arms and the per-connection modulated arms.
- `scripts/analysis/`: the rung's analysis, in the gate preflight's shape.
- Tests, a logbook, the tracker, the roadmap and the CHANGELOG.
