## Why

G.0 read **`artefact`** ([Logbook 095](../../../docs/experiments/logbooks/095-gap-operating-point.md)):
on the thermal cell at target 35 on Emmons 2024, the wild type leads the gap-only null by +0.104
`auc_success` with the model's count-scaled gap weights, and by +0.003 with every count set to 1. The
lead follows the gap scale and vanishes under log-compressed counts. The registered consequence is
G.0-A (roadmap v4.5 § Phase 9; tracker `phase9-tracking` G.0-A): the substrate change G.0 names lands
as its own change with a positive control, the corrected model's contrast is re-run, and the
decomposition of Logbooks 074–092 is restated.

The artefact has a mechanism in the form of the coupling, not only its size. In the settling map a gap
junction is **additive drive**, `+ Gᵀh`: a neuron receives its partners' activity whatever its own
state, so a symmetric pair coupled above about 1 is a self-sustaining latch (ALA–CAN couples at 89.7).
Real electrical synapses pass current in proportion to the **potential difference**, `g(v_j − v_i)`:
diffusive coupling, which pulls partners together and cannot drive a pair on its own at any
conductance. The maintainer named the change on 2026-10-17: **ohmic gap coupling in the settling map**,
chosen against **log-compressed counts** (the one candidate G.0 tested) in a gated pilot whose selection
rule is fixed before it runs.

The repo already has ohmic coupling, but only inside `dynamics: leaky`, where it couples membrane
potentials carried across steps through a precomputed implicit operator. That substrate learns the
thermal cell far worse than the settling map ([Logbook 080](../../../docs/experiments/logbooks/080-across-step-state.md):
20% against 64%), and 080 attributes most of the loss to the carried state. This change puts ohmic
coupling into the memoryless settling map that every registered reading uses.

## What Changes

- **Brain option** (`connectome-ppo-brain`) `gap_coupling`: `additive` (default, byte-identical) |
  `ohmic`. Under `ohmic`, each settling step's drive (chemical input, plus the modulator field's held
  current where present) is passed through `(I + L)⁻¹`, `L = D − G` the Laplacian of the brain's gap
  matrix (so `gap_weight_mode` and `gap_weight_scale` still set the conductances), before `tanh`: within
  each settling step the gap-coupled potentials take the steady state of `dv/dt = −v + drive − Lv`. The
  operator is symmetric, its eigenvalues lie in (0, 1], so it is stable and cannot latch for any
  conductances; it is computed once in double precision. The additive `Gᵀh` term is not added.
  Refused with gap junctions off, with `dynamics: leaky` (already ohmic), with `plastic_gaps` (the
  operator is fixed), and under the plastic learning rules (their eligibility reads the additive map).
- **The selection pilot** (seeds 9871–9878), configured as G.0-A's panels will be: on the thermal cell at
  target 35 and on hard350, both on Emmons 2024, each of `ohmic` and `log_count` with the wild type and
  the gap-only null, learning and frozen, and the MLP-PPO control learning and frozen: 160 runs. **The
  selection rule reads learnability only, never the wiring gap**, so choosing the model cannot choose
  the answer (design Decision C).
- **Configs** through a generator, each a stated delta, re-read by a loader test.
- **Analysis** (`scripts/analysis/gap_substrate_selection.py`): the pilot's gates and the selection rule.
- **Then, each under its own launch record**: the corrected model's point-worm contrast, wild type
  against the gap-only null on the thermal cell at target 35, 64 seeds (3901–3964), with G.0's beside
  levels re-read on the corrected model where they still apply; and 092's body contrast re-run on the
  corrected model **after M.1 reads out** (the maintainer's sequencing decision, 2026-10-17: D25 forbids
  a 6,000-episode body panel before then). The body re-run and the decomposition restatement are their
  own changes.

## Capabilities

**Modified**: `connectome-ppo-brain` — the `gap_coupling` option (ADDED requirement; every existing
scenario unchanged).

## Impact

- `brain/arch/connectome_ppo.py`: the option, its refusals, the operator, the two settling paths
  (single and batched; the body drive reads the same settled state).
- New configs, a generator, an analysis module and their tests.
- Tracker: G.0-A's items; the seed-band table gains 9871–9878 and 3901–3964.
- No existing reading changes: `additive` is the default and byte-identical.
