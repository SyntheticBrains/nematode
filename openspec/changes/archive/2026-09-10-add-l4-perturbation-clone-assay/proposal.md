# The node-perturbation variant through the clone assay (7a-ii I.1, step 3)

## Why

The node-perturbation eligibility passed the rule's positive control
([Logbook 049 records](../../../docs/experiments/logbooks/supporting/049-l4-node-perturbation/details.md)):
at σ = 0.2 it covers 88% of the floor-to-optimum gap on 8 of 8 seeds with a gradient alignment of
+0.263, where the rule it replaces sits below the cue-blind floor. It is the first mechanism in
this sequence that works.

The clearance order fixed in that change's launch record, before the result existed, is: the
positive control, the gradient alignment, **the clone assay**, and only then a connectome arm.
Steps 1 and 2 are done. This is step 3, and it is not a formality: the three consolidation
mechanisms are the standing reminder that a rule can move a policy toward reward from random
weights and still take a competent one apart, which is the failure mode that gates the panel.

It is also the variant's **first evaluation on the connectome** — the perturbation was implemented and tested there in I.1, but every result so far came from the MLP yardstick on a synthetic task. The clone arm runs the real substrate, the real environment and a policy that is already competent.

## What Changes

- **Two arm configs**: the wild-type plastic clone arm with the variant's rule keys and nothing
  else changed — `plasticity_eligibility: node_perturbation` at the σ the control pinned — and a
  **frozen-perturbation control**, the same arm with `freeze_updates: true`, which applies the
  perturbation and writes no weight, so what the jitter alone costs a competent policy is measured
  rather than assumed.
- **A trajectory annotation**: each arm's plateau tail over its final quarter against its first,
  from the curves the harness already reads, so "held it and paid an exploration tax" is
  distinguishable from "destroyed it". Not a verdict change.
- **The arm joins the existing clone-assay harness registry**, whose pass rule, comparator and
  reporting are the ones registered with the clone-destruction diagnostic and used for the three
  consolidation mechanisms. Nothing about the assay changes; only an arm is added.
- The runs, records under `supporting/050-l4-perturbation-clone-assay/`, tests, docs. This closes into I.1's logbook rather than taking one of its own.

Out of scope: any panel (still gated on I.2's statistic and metric), the 2×2 re-run, and any
tuning of σ — the value is the one the control pinned, and re-tuning it here would make the assay
a search rather than a gate.

## Capabilities

**Modified**: `l4-plasticity-panel` (the assay's application to an eligibility variant).

## Impact

- New: one arm config, the supporting directory. Edited:
  `scripts/analysis/l4_consolidation_screen.py` (one registry entry and its docstring, since the
  harness now screens more than consolidation), `CHANGELOG.md`.
- No rule, substrate or default changes: the variant already exists and is off by default.
