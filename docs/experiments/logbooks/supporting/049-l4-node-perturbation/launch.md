# Node-perturbation clearance launch record

Written and committed before the registered run.

- **Date**: 2026-09-10
- **Pinned state**: `feat/l4-node-perturbation` at the commit implementing the perturbation, the
  eligibility mode and their tests.
- **The defect this repairs**: the rule failed its positive control with a gradient alignment of
  **+0.009** — a live trace and a well-behaved modulator, but updates near-orthogonal to the policy
  gradient. Its eligibility is `pre × post` with exploration noise applied at the action, so it
  correlates reward surprise with ordinary activity rather than with anything the network could
  have done differently.

## The variant

Each plastic unit's **pre-activation** is perturbed by `ξ ~ N(0, σ_node²)` — `h = tanh(a + ξ)`, so
the unit acts on its perturbation through its own nonlinearity — and the eligibility becomes
`pre ⊗ ξ`. On the connectome an independent `ξ` is injected at **every settling step** and the
exposed value is their sum; perturbing only the settled state would leave a synapse's effect
through later steps uncredited. `ξ` comes from a generator dedicated to it and seeded from the run
seed, so enabling perturbation shifts nothing else in the random stream. Perturbation state is
transient: cleared per episode, never persisted.

Everything else is unchanged and composes as before — the modulator, both scaling switches, decay,
mask, Dale's law, homeostasis, consolidation, routing, and the clamp last.

## Protocol

The rule's positive control (I.0's, unchanged) at its **registered pass rule**: 8 seeds ×
20,000 trials, scored on mean reward over the final 1,000 trials.

- **Pass** — beats the cue-blind floor (−0.6909) on ≥ 7 of 8 seeds **and** the mean reaches at
  least halfway to the optimum (−0.4131), at **any** `σ_node` in the declared grid.
- **Fail** — it does not.
- **Void** — the analytic reference does not pass, or the unmodulated floor does.

**Grid**: `σ_node ∈ {0.01, 0.05, 0.2}`, pinned by the control itself rather than a separate pilot.
A grid failing at every value is a failure of the variant; the grid is not widened afterwards.

With trace normalisation on, the Hebbian term is divided by the trace's running RMS and that RMS
scales with `σ_node`, so the learning step is roughly σ-invariant: **the grid sets behavioural
perturbation, not learning rate**. A `σ_node` that passes says the network tolerates that much
per-unit jitter and still learns from it. The action noise stays at the arms' pinned value, which
adds variance to `δ` uncorrelated with `ξ` — unbiased but noisier — accepted for comparability.

## Clearance order, fixed before any result

1. **This control.** If the variant fails here it is **not an instrument**, no connectome arm is
   built, and block I's next question is whether anything in this rule family learns this task.
2. **The gradient alignment**, from the same runs. A pass with a near-zero alignment means the
   variant learned by some route other than the one it was built for, and is recorded as that
   rather than as vindicating the theory.
3. **The clone assay**, which gates any panel.
4. Only then a connectome arm, and only under I.2's statistic and metric.

## What each outcome licenses

- **Pass with a positive alignment** — Logbook 048's diagnosis is confirmed and repaired: the rule
  failed because its eligibility carried no counterfactual. The seven reframed results become
  re-runnable questions, and I.4's re-read has a working instrument.
- **Pass with near-zero alignment** — something else did the work; recorded as such.
- **Fail** — the eligibility was not the whole defect. That closes the most specific hypothesis
  block I had, and the 7a shipment decision is taken with the rule family characterised.

## Command

```bash
uv run python scripts/analysis/l4_rule_positive_control.py \
  --out docs/experiments/logbooks/supporting/049-l4-node-perturbation/control.json \
  --csv docs/experiments/logbooks/supporting/049-l4-node-perturbation/per-seed.csv
```

## Results

Run 2026-09-10, 48 runs. **The variant passes**, at σ = 0.2: mean **−0.1962** against a floor of
−0.6909 and a halfway threshold of −0.4131, on **8 of 8 seeds**, with per-seed scores spanning
−0.184 to −0.213. The control is valid — the analytic reference passes, the unmodulated floor does
not. Performance and alignment rise together across the grid (−0.777/+0.027, −0.477/+0.118,
−0.196/+0.263), so the "passed by some other route" branch does not fire: the passing arm's
alignment is **+0.263** against the old rule's +0.009. The eligibility was the defect, and carrying
the perturbation repairs it. Full reading in `details.md`.
