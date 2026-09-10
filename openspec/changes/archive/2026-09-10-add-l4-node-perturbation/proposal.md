# An eligibility with the noise inside it (7a-ii I.1)

## Why

The rule's positive control ran and the rule failed it ([Logbook 048](../../../docs/experiments/logbooks/048-l4-rule-positive-control.md)).
On a one-step association whose analytic reference closes 99.9% of the available gap on every seed,
the three-factor rule ends **below the cue-blind floor**, at every rate across two orders of
magnitude, with a live trace and a well-behaved modulator and a **gradient alignment of +0.009**.
The updates are not starved of reward information; they are not aimed. That is the measured form of
a defect the theory names precisely.

The eligibility is `pre × post` and the exploration noise is applied at the **action**, after the
forward pass. The three-factor rules that are unbiased policy-gradient estimators put the noise
**inside the eligibility**: the trace is pre-synaptic activity times the *perturbation* of the
post-synaptic unit, so that `Δw = η · δ · pre · ξ` correlates the reward surprise with the
perturbation that produced it (Williams 1992; node perturbation, Fiete & Seung 2006; the Frémaux &
Gerstner 2016 review). With deterministic units and output-only noise there is no such correlate:
an internal synapse cannot know which way to move to make the sampled action more likely, so the
rule reinforces whatever correlation structure is present and drifts.

Recon confirms the gap is structural rather than a parameter choice: **no topology in this project
sees the exploration noise**. It is added at the action head after the forward returns, so the
trace physically cannot contain it. Fixing this is a substrate change, not a rule tweak.

Ratified as block I's critical path: I.1 is what every queued substrate rung now waits on.

## What Changes

- **Per-unit perturbation on the plastic-topology seam**: a topology may, when asked, perturb each
  unit's activity by `ξ ~ N(0, σ_node²)` during the trace-accumulating forward and expose that
  perturbation. Off by default and byte-identical off, so every committed result stands.
- **An eligibility mode on the rule** (`eligibility: hebbian | node_perturbation`): the trace
  becomes `pre ⊗ ξ` rather than `pre ⊗ post`, making the update a policy-gradient estimator over
  the perturbations rather than reinforced correlation. Everything else — the modulator, the
  scaling switches, the bound, decay, homeostasis, Dale's law, consolidation, routing — is
  unchanged and composes as before.
- **Cleared on the rule's own positive control first.** The control built for I.0 is the instrument:
  the variant is run through it at the **same registered pass rule**, on the MLP yardstick, before
  any connectome arm exists. If the yardstick does not learn, the variant is not an instrument
  either and the record says so.
- **Then the clone assay**, which gates any panel, and only then a connectome arm.
- Records under `supporting/049-l4-node-perturbation/`, tests, docs.

Out of scope: re-running any panel; the connectome arms; B.3's receptor layer; I.2's statistic and
metric. This change asks one question — does an eligibility carrying the noise make this rule a
learner — and answers it on the smallest instrument that can answer it.

## Capabilities

**Modified**: `brain-architecture` (the seam's perturbation), `learning-rules` (the eligibility
mode and its clearance requirement).

## Impact

- New: the perturbation on both topologies, the eligibility mode, an arm in the positive-control
  harness, the supporting directory. Edited: `brain/arch/_topology.py`,
  `brain/arch/_mlp_topology.py`, `brain/arch/connectome_ppo.py`,
  `brain/arch/_plasticity_config.py`, `learning_rules/three_factor.py`,
  `scripts/analysis/l4_rule_positive_control.py`, `docs/architectures.md`, `CHANGELOG.md`.
- Defaults are byte-identical: `eligibility: hebbian` with perturbation off is today's code path
  and allocates nothing.
