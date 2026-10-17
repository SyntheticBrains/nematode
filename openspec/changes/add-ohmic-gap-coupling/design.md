## Context

G.0 found the gap-junction placement lead on the thermal cell is carried by how the model couples gap
junctions: additive drive scaled by EM counts (Logbook 095). Additive coupling has no biophysical
counterpart: an electrical synapse passes current in proportion to the potential difference across it.
The maintainer chose (2026-10-17) to build ohmic coupling into the settling map and select between it and
log-compressed counts in a gated pilot, then re-run the corrected model's contrasts: the point worm now,
the body after M.1 reads out.

## Decisions

### Decision A: Ohmic coupling as an implicit step inside each settling iteration

Today each of the `K` settling iterations computes `h ← tanh(W_chemᵀh + Gᵀh + c)`, `c` the modulator
field's held current where present, from `h₀` the injected sensors. Under `gap_coupling: ohmic`:

```text
h ← tanh( M⁻¹ (W_chemᵀh + c) ),   M = I + L,   L = diag(G·1) − G
```

`M⁻¹ d` is the steady state of `dv/dt = −v + d − Lv`: within each iteration the gap-coupled potentials
equilibrate to the drive, each pair's current proportional to the difference of the two potentials and
to the pair's conductance. `L` is symmetric positive semi-definite for a non-negative symmetric `G`, so
`M⁻¹`'s eigenvalues lie in (0, 1]: the map cannot latch and stays stable at any conductance. `G` is
fixed, so `M⁻¹` is computed once, in double precision, and applied by one matrix product per iteration
(right-multiplied on a batch, as the leaky substrate already does). The sensors, the readout, the body
drive and the critic are unchanged.

**The conductances are the brain's gap matrix as built**: the counts divided by `sqrt(d_i d_j)`, with
`gap_weight_mode` and `gap_weight_scale` applied. So at the defaults the only difference from today is
the form of the coupling, and the pilot compares forms, not forms and magnitudes at once. Rejected for
this change: raw counts as conductances (it would change the form and the magnitude together; it can be
a later level of the profile).

### Decision B: Not the leaky substrate

`dynamics: leaky` already couples ohmically, through the same kind of operator, but carries membrane
potentials across steps; it learned the thermal cell at 20% against the settling map's 64% (Logbook
080), mostly from the carried state. This change keeps the settling map memoryless, so the corrected
model differs from every registered reading in the gap coupling's form alone. `gap_coupling: ohmic` is
refused under `dynamics: leaky`, which is ohmic already.

### Decision C: The selection rule, fixed before the pilot runs

The pilot (seeds 9871–9878, disjoint from every band) runs on both block-V cells on Emmons 2024, the
thermal cell at target 35 and hard350, 3,000 episodes: for each model (`ohmic` at the defaults;
`log_count`), the wild type and the gap-only null, each learning and frozen; and the MLP-PPO control,
learning and frozen. 2 cells × (2 models × 4 arms + 2 MLP arms) × 8 seeds = **160 runs**.

1. **The cells must be readable**: the MLP control beats its frozen floor on both cells (the paired
   80% interval of learning − frozen plateau above zero). If not, the pilot is unreadable and nothing is
   chosen.
2. **A model is eligible on a cell** when both its learning arms, the wild type and the gap-only null,
   beat their own frozen floors, and the two do not both plateau at or above 90%.
3. **Ohmic is chosen** if it is eligible on both cells and, on each cell, its wild type is not slower
   than log's wild type by more than **0.05** `auc_success`: the paired-by-seed 80% interval of
   `ohmic − log` has its lower end above −0.05.
4. Otherwise **log-compressed counts** are chosen if eligible on both cells.
5. Otherwise **neither**: the result is recorded and the substrate decision returns to the maintainer.

The rule reads only floors, saturation and the wild type's learning speed. **It never reads the gap
between the wild type and the null**, on either model, so the choice of model cannot be a choice of
answer; that gap is read only in the registered panels that follow. The 0.05 margin is about half of
Logbook 082's thermal gap effect (+0.096); a model slower than that is a worse instrument for the
readings it must carry.

### Decision D: Byte identity and the tests

`gap_coupling: additive` is the default and computes exactly what it computes today: the settling
paths are unchanged on that branch, and pinned digests (gap matrix, weights, the next draw, as G.0's
tests pin them) plus an output comparison hold it. Tests on `ohmic`: the operator equals a hand
computation on a small graph; a symmetric pair at conductance 100 with drive on one neuron settles both
within 1% of each other, and holds no activity with no drive (no latch); the single and batched
paths agree; every refusal fires, by the validator and again at build (`model_copy` skips validators).

### Decision E: What follows, each under its own registration

- **G.0-A's point-worm contrast** (seeds 3901–3964): the wild type against the gap-only null on the
  thermal cell at target 35, on the chosen model, learning and frozen, 64 seeds, its minimum fixed in
  its launch record before it runs; beside it, G.0's levels that still apply on the chosen model (the
  scattered and held outlier nulls, gap-off).
- **092's body contrast on the corrected model**, after M.1 reads out (D25), seeds fixed then.
- **The decomposition restated** (Logbooks 074, 075, 078, 079, 082, 092), once both have read.
- G.1–G.3b then register on the corrected model, or are deferred with a destination.

## Risks / Trade-offs

- **Ohmic coupling may learn slowly**, as the leaky substrate did. Then the rule picks log-compressed
  counts, and the fidelity question is recorded as open rather than forced.
- **Ohmic coupling may abolish any gap effect** on the corrected model. That is a legitimate reading of
  the next panel, not a reason to choose the other model; the selection rule cannot see it.
- **Eight seeds per arm** resolve only large differences in learning speed; the margin is set for an
  instrument choice, not a finding, and the choice is never cited as one.
- **The plastic learning rules and `plastic_gaps` are refused under ohmic**; extending them is a later
  change if a block needs it.

## Open Questions

- The point-worm contrast's minimum (Decision E): two-thirds of G.0's count-model anchor (0.0894), or of
  Logbook 082's committed effect (0.0577). Fixed in its launch record, before it runs.
