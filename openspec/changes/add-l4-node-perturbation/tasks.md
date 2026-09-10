# Tasks: an eligibility with the noise inside it

## 1. Perturbation on the seam

- [ ] 1.1 Add `plastic_perturbations` to the `PlasticTopology` Protocol, aligned like the other seam
  members and indexed along the axis complementary to the fan-in axis.
- [ ] 1.2 Implement it on `MLPTopology`: when enabled, draw `ξ ~ N(0, σ_node²)` per plastic unit in
  the trace-accumulating forward, **add it to the activity the layer passes on**, and accumulate
  `pre ⊗ ξ`. Off by default and byte-identical off.
- [ ] 1.3 Implement it on `ConnectomeTopology` the same way, on the settled hidden state.
- [ ] 1.4 Tests: the unit acts on its own perturbation (its activity differs by exactly the exposed
  value); disabled is bit-identical on both substrates; the vectors are aligned with the weights;
  perturbations are redrawn per step and are not carried between steps.

## 2. The eligibility mode

- [ ] 2.1 Add the eligibility mode and the noise scale to the plasticity mixin, with load-time
  bounds; refuse `node_perturbation` at zero noise and on a substrate that does not perturb — on the
  config and, duplicated, at brain construction, since `model_copy` skips validators.
- [ ] 2.2 Accumulate `pre ⊗ ξ` under the mode; leave every other term untouched.
- [ ] 2.3 Tests: the trace equals pre × perturbation and not pre × activity; `hebbian` is
  bit-identical; every refusal including the construction guard on a copied config; the mode
  composes with routing, consolidation, decorrelation, Dale's law and homeostasis in the registered
  order with the clamp last.

## 3. Clearance on the positive control

- [ ] 3.1 Add the variant as an arm of the positive-control harness over the declared grid
  `σ_node ∈ {0.01, 0.05, 0.2}`, reusing the registered pass rule, the reference arm and the floor
  arm unchanged.
- [ ] 3.2 Report the variant's gradient alignment beside its pass or fail, from the same runs.
- [ ] 3.3 Tests: the harness scores the variant under the same rule as the three-factor arm; any
  grid value passing counts as a pass; a variant passing with a near-zero alignment is flagged.
- [ ] 3.4 Commit the launch record — the grid, the pass rule, the clearance order and what each
  outcome licenses — then run it.
- [ ] 3.5 Records under `docs/experiments/logbooks/supporting/049-l4-node-perturbation/`:
  `launch.md`, `control.json`, `per-seed.csv`, `details.md`.

## 4. Documentation

- [ ] 4.1 `docs/architectures.md`: the perturbation, the eligibility mode, and that a new
  eligibility clears the control before any substrate arm.
- [ ] 4.2 `CHANGELOG.md`.
- [ ] 4.3 Tracker and roadmap updated with the outcome at close-out; if the variant passes, the
  clone assay is next and no panel runs before I.2.
