# Design: an eligibility with the noise inside it

## The defect, stated precisely

For a stochastic policy, the REINFORCE estimator of the reward gradient at a synapse is the reward
surprise times the derivative of the log-likelihood of what the network actually did. In the
node-perturbation form, each unit `j` adds its own perturbation `ξ_j` to its activity, and

```text
Δw_ij  ∝  δ · x_i · ξ_j
```

is an unbiased estimate of the gradient of expected reward, because `ξ_j` is exactly the part of
unit `j`'s output that the network could have done differently, and `δ` says whether doing it was
good. Every term is local to the synapse: pre-synaptic activity, the post-synaptic unit's own
perturbation, and a broadcast scalar.

Today's rule uses `Δw_ij ∝ δ · x_i · h_j`, with `h_j` the unit's deterministic activity and the only
stochasticity applied to the *action* after the forward pass. `h_j` is not a perturbation and
carries no information about a counterfactual, so the product correlates the reward surprise with
the network's ordinary activity rather than with any choice it made. That is reinforced correlation,
and it is why the measured alignment with the true gradient is +0.009.

**This is not fixable in the rule alone.** The eligibility is accumulated inside each topology's
forward pass, and no topology in this project has access to the action noise — it is added by the
brain after the forward returns. Whatever perturbation the trace is to carry must be generated
where the trace is built.

## What the seam gains

One optional behaviour, expressed as state a topology already owns:

- when **perturbation is enabled**, a trace-accumulating forward draws `ξ ~ N(0, σ_node²)` per
  plastic unit, adds it to that unit's activity, and accumulates `pre ⊗ ξ` instead of `pre ⊗ post`;
- the perturbation is exposed on the seam as `plastic_perturbations`, aligned like the other seam
  members, so a rule or a harness can read what was injected.

The perturbation is added to the activity that the network actually uses, not to a shadow copy:
the unit must *act* on its perturbation or the eligibility describes a counterfactual the network
never took, and the estimator is biased. That means perturbation changes the forward pass, which is
why it is off by default and why every arm that uses it is a new arm rather than a re-reading of an
old one.

**Why not reuse the action noise.** The obvious alternative — propagate the action-level noise back
into the trace — requires knowing each unit's contribution to the action, which is a credit
assignment problem and needs weight transport. That is exactly the non-locality this rule family
exists to avoid, and it would make the result uninteresting even if it worked.

## What the rule gains

`eligibility: node_perturbation` uses the seam's perturbation in place of the post-synaptic factor.
Nothing else changes: the modulator, both scaling switches, the decay, the mask, Dale's-law
projection, consolidation, routing, the homeostatic rescale and the clamp keep their order and
meaning. The mode is refused where the substrate exposes no perturbation, since the trace would
otherwise silently be zero.

The trace normalisation matters more here than before and is kept on: `ξ` has a different scale from
`h`, so a rate pinned against a Hebbian trace would mean something else against a perturbation
trace. With normalisation on, the rate is the same root-mean-square step per unit modulator in both
modes, which is what makes the two comparable at all.

## Configuration

| field | default | meaning |
|---|---|---|
| `plasticity_eligibility` | `hebbian` | `hebbian` or `node_perturbation` |
| `plasticity_node_noise` | `0.0` | `σ_node`, the per-unit perturbation, `≥ 0` |

`node_perturbation` with a zero noise is rejected at load: the trace would be identically zero and
the arm would look like a rule that learns nothing when it is a rule that was never given anything
to learn from. The mode is also rejected on a substrate whose topology does not implement
perturbation.

`σ_node` has no value to inherit. It is pinned by the control itself rather than by a separate
pilot: the positive control is cheap, deterministic and already the clearance gate, so the variant
runs it over a declared grid `σ_node ∈ {0.01, 0.05, 0.2}` and the pinned value is the one that
passes, ties to the smaller. A grid that fails at every value is a failure of the variant, and the
record says so rather than widening the grid.

## Clearance, in order, and what each step licenses

1. **The positive control** (I.0's, unchanged, at its registered pass rule). The variant must beat
   the cue-blind floor on ≥ 7 of 8 seeds and reach halfway to the optimum. **If it fails here it is
   not an instrument**, no connectome arm is built, and block I's next question is whether anything
   in this rule family can learn this task.
2. **The gradient alignment**, reported from the same runs. A pass with an alignment near zero would
   mean the variant learned by some route other than the one it was built for, and would be recorded
   as such rather than as a vindication of the theory.
3. **The clone assay**, which already gates any panel: does the variant hold a competent policy.
4. Only then a connectome arm, and only under I.2's statistic and metric.

Steps 1 and 2 are this change. Steps 3 and 4 are named here so the order is fixed in advance and
cannot be reordered after a good result.

## What the outcomes mean

- **Passes the control with a positive alignment** — the diagnosis in Logbook 048 is confirmed and
  repaired: the rule failed because its eligibility carried no counterfactual, and an eligibility
  that does makes it a learner. The seven reframed results become re-runnable questions rather than
  closed ones, and I.4's re-read has a working instrument to re-read them against.
- **Passes with a near-zero alignment** — something else is doing the work; recorded, and the
  variant is not yet the instrument it claims to be.
- **Fails** — the eligibility was not the whole defect. That is a substantive finding about this
  rule family on this task, it closes the most specific hypothesis block I had, and the 7a shipment
  decision is taken with the rule family characterised rather than with one more rung queued.

## Alternatives considered

- **Propagating action noise into the trace** — needs weight transport; rejected above.
- **Perturbing weights rather than units** — also unbiased, but the variance scales with the number
  of synapses rather than units, which on a 302-unit, 3,709-synapse substrate is an order of
  magnitude worse.
- **Removing the action noise when perturbation is on** — tempting, since perturbation already
  drives exploration, but it would change two things at once and make a comparison with the panels'
  arms impossible. The action noise stays at the arms' pinned value.
- **Testing on the connectome first** — the control is cheaper, deterministic, and the only place a
  null is unambiguous. That ordering is the whole point of block I.
