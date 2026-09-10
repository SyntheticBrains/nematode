# Node-perturbation clearance details

Run by `scripts/analysis/l4_rule_positive_control.py` at the registered budget: 8 seeds × 20,000 trials × 8 arm/parameter combinations (64 runs). `control.json` is the full output and `per-seed.csv` the
table. The control, its arms and its pass rule are I.0's, unchanged.

## Result: the variant **passes**, and passes for the reason it was built

| arm | mean | seeds above floor | alignment | result |
|---|---|---|---|---|
| `analytic` (reference) | −0.1361 | 8/8 | +0.454 pooled, +1.0 while learning † | passes |
| `hebbian` (floor) | −0.8753 | 3/8 | −0.010 | does not pass |
| `three_factor` @ 1e-4 / 1e-3 / 1e-2 | −0.79 / −0.75 / −0.75 | 1/8, 1/8, 2/8 | +0.031 pooled | does not pass |
| `node_perturbation` @ σ=0.01 | −0.7769 | 4/8 | +0.027 | does not pass |
| `node_perturbation` @ σ=0.05 | −0.4773 | 7/8 | +0.118 | does not pass |
| **`node_perturbation` @ σ=0.2** | **−0.1962** | **8/8** | **+0.263** | **passes** |

Floor −0.6909, optimum −0.1353, halfway threshold −0.4131.

† The reference's pooled figure is the one in `control.json`. It converges after roughly
5,000 trials, and past that its per-block gradient is numerically negligible so the cosine decays
into rounding noise; measured while it is still learning it is exactly +1.0, which is the
end-to-end check on the sign convention. No such dilution applies to the arms being judged: the
old rule never converges, and the variant's +0.263 is measured over a run that was still improving.

**The control is valid.** The analytic reference passes and the unmodulated floor does not, so
neither void condition fires and this is a fact about the variant. The `outcome` field in
`control.json` reads `fail` because it tracks the **three-factor** arm — I.0's question, whose
answer is unchanged; the variant's result is its own `passes` flag.

**The margin is large and the spread is small.** At σ = 0.2 the variant reaches −0.196, about 88%
of the way from the cue-blind floor to the optimum, on every one of eight seeds, with per-seed
scores spanning −0.184 to −0.213. The rule it replaces sits near −0.75 at every rate — *below* the
floor a policy ignoring the cue would reach.

## The dose-response is the evidence

| σ_node | mean score | alignment |
|---|---|---|
| 0.01 | −0.777 | +0.027 |
| 0.05 | −0.477 | +0.118 |
| 0.2 | −0.196 | +0.263 |

Performance and gradient alignment rise together across the grid. That is the signature of an
estimator whose variance falls as the perturbation grows: a larger `ξ` gives a better-conditioned
estimate of the reward gradient, the updates point more nearly along it, and the policy improves.
The registered branch for "passes with a near-zero alignment" — learned by some route other than
the one it was built for — **does not fire**: the passing arm's alignment is +0.263 against the old
rule's +0.009.

Recall that with trace normalisation on the learning step is roughly σ-invariant, so this is not a
learning-rate sweep in disguise. The grid varies **how much each unit jitters its own
pre-activation**, and the result says the network needs a substantial amount of it — σ = 0.2
against activities bounded in (−1, 1) — before the estimate is good enough to learn from.

## What this confirms

Logbook 048 diagnosed the failure as an eligibility carrying no counterfactual: `pre × post`
correlates reward surprise with the network's ordinary activity, which is reinforced correlation
and not a gradient estimate. Replacing the post-synaptic factor with the unit's own perturbation —
the part of its output it actually varied — makes the same rule, with the same modulator, the same
scaling, the same bound and the same recipe, into a learner on the same task. **Nothing else
changed.** That is as clean a confirmation of a diagnosis as this project has produced.

## What it does not license

- **Not a connectome arm.** The registered clearance order is control → alignment → **clone
  assay** → substrate arm. The variant has cleared the first two. A rule that learns from random
  weights may still take a competent policy apart, which is exactly what the three consolidation
  mechanisms did.
- **Not a panel.** Any panel still waits on I.2's statistic and graded metric.
- **Not a re-reading of Logbooks 040–047 on its own.** I.4 does that, and now has a working
  instrument to do it with.

## Limitations

- One task, one topology, one arrangement — the MLP yardstick with a frozen readout at the panels'
  pinned recipe. The connectome is a different substrate and the variant has not been run on it.
- σ = 0.2 is the only passing value in the declared grid, and it is the grid's largest. Whether a
  larger perturbation would do better, and at what cost to the policy the perturbation degrades, is
  untested; the grid was fixed in advance and was not widened after the result.
- The action noise remained at the arms' pinned value throughout, adding variance to `δ`
  uncorrelated with `ξ`. The variant therefore passes *despite* a noisier estimate than
  perturbation-only exploration would give.
- The alignment is measured against the immediate loss on a one-step task; a longer-horizon setting
  would not have this luxury.

## Facts

- 64 runs (8 seeds × 8 arm/parameter combinations: the reference, the floor, three rates of the original rule and three perturbation scales), single process, no environment.
- The launch record fixing the grid, the pass rule, the clearance order and each outcome's reading
  was committed before this run.
- A defect was found and repaired in the per-seed CSV: the writer's header row had not been updated
  alongside its data row, so the emitted columns were shifted by one. `control.json` was unaffected,
  being computed from the runs rather than the file, and the row data was intact, so the header was
  corrected in place rather than re-running. The writer is fixed.
