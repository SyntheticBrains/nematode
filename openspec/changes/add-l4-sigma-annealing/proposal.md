# Annealing the perturbation scale (7a-ii I.1b)

## Why

I.1 left one finding and one tension. The finding: an eligibility carrying each unit's own
pre-activation perturbation is the first mechanism in this sequence that learns — at σ = 0.2 it
covers 88% of the floor-to-optimum gap on 8 of 8 seeds with a gradient alignment of +0.263, where
the rule it replaces sits *below* the cue-blind floor
([049 records](../../../docs/experiments/logbooks/supporting/049-l4-node-perturbation/details.md)).

The tension: that same σ is the reason the variant failed the clone assay. The frozen-perturbation
control registered beside it settles what the failure means — with weights untouched (cosine 1.00)
the control still falls from 38.7 to **8.9**, already 9.1 in its first quarter, so σ = 0.2 jitter
alone destroys a competent policy and the learning arm's 12.0 sits *above* that baseline
([050 records](../../../docs/experiments/logbooks/supporting/050-l4-perturbation-clone-assay/details.md)).
The assay at that σ could not test retention because no competent policy survived to be retained.

**The σ that makes the rule learn is the σ that makes a competent policy unrunnable.** That is a
statement about a *constant* σ, and nothing in the estimator requires it to be constant. The
control's own dose-response already shows the two demands pulling apart along the grid: σ = 0.05
put 7 of 8 seeds above the floor without clearing the learning bar, while σ = 0.2 cleared it. A
schedule that explores at the σ which learns and decays toward the σ which can be run is the
obvious untested mechanism, and the tracker has named it as such since I.1 closed.

This registers that mechanism and the two gates it must clear, before it runs.

## What Changes

- **A schedule for the perturbation scale.** `plasticity_node_noise` becomes the *initial* σ;
  `plasticity_node_noise_final` and `plasticity_node_noise_anneal_episodes` add a decay toward a
  floor, advanced once per episode at the existing `prepare_episode` hook. Absent the new fields
  the scale is constant and every existing arm is byte-identical.
- **The scale coupling stated and tested.** The trace carries `pre ⊗ ξ` with `ξ ~ N(0, σ²)` drawn
  as `randn · σ`, so the update's magnitude scales with σ *linearly* — decaying σ shrinks the step
  as well as the exploration. `plasticity_normalise_trace`, which every panel arm enables, divides
  each tensor's Hebbian term by a running RMS of its own trace and therefore removes most of that
  coupling. The change does not add a compensation term; it **registers which of the two regimes an
  arm runs in** and requires the annealed arm to state it, so a result cannot later be attributed
  to a rate change that was never declared.
- **Two gates, in this order.** (1) The annealed schedule must still pass the rule's positive
  control — a schedule that cannot learn is not a mechanism. (2) Only then the clone assay, with
  the frozen-perturbation control run beside it as I.1's did, since the same argument that made
  that control necessary applies unchanged.
- The runs, records under `supporting/051-l4-sigma-annealing/`, tests, docs.

Out of scope: any panel (still gated on I.2's statistic and metric); the 2×2 re-run; a schedule
whose shape is fitted to the outcome — the shape and its parameters are fixed here, before the
result exists, and a failure is reported as a failure of that schedule rather than re-tuned.

## Capabilities

**Modified**: `learning-rules` (the perturbation scale gains a schedule), `plasticity-evaluation`
(the two gates the annealed arm clears).

## Impact

- New: `plasticity_node_noise_final`, `plasticity_node_noise_anneal_episodes`, arm configs, the
  supporting directory. Edited: the plasticity config and its validators, the two topologies'
  perturbation draw, `scripts/analysis/l4_rule_positive_control.py` (an annealed arm),
  `scripts/analysis/l4_consolidation_screen.py` (an arm entry), `CHANGELOG.md`.
- No default change: with the new fields unset, σ is constant and the arithmetic is unchanged.
