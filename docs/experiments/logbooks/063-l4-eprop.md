# 063: e-prop Learns the Cell, and the Control Says the Wiring Did Not (7a-ii R.2 / Phase 7)

**Status**: completed — **`learns_without_the_substrate`**. Node perturbation was closed by
[R.1](060-l4-perturbation-scale.md), [R.1c](061-l4-reduced-perturbation.md) and
[R.1d](062-l4-frozen-readout.md), which between them left one suspect: not the dimension, not the
input map, not the output map, but **what the eligibility carries**. So e-prop — D1's named fallback —
replacing the perturbation with the network's own settling derivative and the draw's sign with a
broadcast learning signal. **Two arms reach competence on the connectome, the first in this
programme**: 17.570 and 13.675 foods of 20 at 52.61% and 40.22% full clear, against a shared frozen
floor of 2.353 and PPO's matched 18.945, on **16 of 16 seeds at q = 0.000**. **And the control added
before either ran says the wiring did not do it.** The better arm is the one whose chemical matrix is
**frozen**; letting the rule write it **costs 3.895 foods** (p 1.000), and every arm that writes the
substrate sits below the control — by 3.9, 4.2, 14.3, 14.6 and 15.9 foods. What learned this cell is an
**8-parameter linear readout over four pooled motor-class means, on frozen recurrent features**. So
059's gate is met **in letter and not in substance**: the substrate rungs stay gated, and the
~1.3–1.4× credited drift that was invariant across R.1c, R.1d and now R.2 is what the cost of writing
the wiring looks like. Two further findings travel with it: the arm the plausibility claim rested on,
**broadcast alignment, cannot work here for a structural reason that was measured before the campaign**
— a frozen readout is the path it would have aligned to; and at matched reach the **true** direction is
worth **+10.123 foods** while reach at a matched source is worth **−0.304**.

**Branch**: `feat/l4-eprop`.

**Date**: 2026-09-15.

**OpenSpec change**: `add-l4-eprop` (extends `plasticity-evaluation`: an eligibility derived from a
substrate's own dynamics states what its approximation drops; a rule whose update needs a per-unit
signal registers how that signal reaches each unit and tests it against the ablation and a
reach-matched control; a mechanism's positive control names both an arm that must pass and one that
must fail).

## Objective

Under the rule the connectome writes `w_chem` alone. Across three campaigns, **credited-synapse drift
sat at 1.37–1.42× the weight's own norm**: 302 perturbed units and 39, four readouts at two norms and
two directions. The rule is never starved of signal — it writes a great deal, in a direction that does
not help. e-prop changes what the eligibility carries:

```text
eps_ij = sum over settling steps s of  psi_j^(s+1) * h_i^(s)     psi = 1 - tanh^2
E_ij  <- decay * E_ij + M_chem ∘ (eps_ij * L_j)
g_k   = (u_k - mu_k) / sigma_k^2      u = the PRE-SQUASH Gaussian draw
L_j   = sum_k B_jk * g_k              B fixed for the run, persisted with the checkpoint
```

`eps` is the **local** part of `∂h_j/∂w_ij`; the paths through other units are dropped, and that
truncation *is* the method. No perturbation is drawn.

## What recon and stage 1 changed, before any arm ran

**The readout decides which units a true-gradient signal can reach.** It is a `(4, 2)` matrix over the
**mean-pooled** VB/DB/VA/DA classes, so `∂mu_k/∂h_j` is zero for 263 of 302 units — and e-prop drops
the multi-hop paths by which any of those 263 reaches the action. So symmetric feedback is not a broad
arm with true directions: it reaches only the readout pool, making it the e-prop analogue of R.1c's
`motor` set. The routing therefore varies **two** things, and the campaign crossed them; the fourth
cell of that 2×2 — true directions reaching all 302 — is one **the mechanism forbids**.

**Stage 1 then measured that the frozen readout is why the plausible arm cannot work.** Feedback
alignment needs the forward path to the output to come into alignment with the feedback matrix, and a
frozen readout cannot. Same task, same rule, same projection, 20,000 trials, 8 seeds:

| readout | rate 1e-4 | rate 1e-3 | rate 1e-2 | seeds above floor |
|---|---|---|---|---|
| **frozen** (`eprop_random`) | −0.8147 | −0.7097 | −0.7031 | 4/8, 6/8, 6/8 — fails at every rate |
| **plastic** (diagnostic, disjoint seeds 101–108) | −0.2726 | **−0.1353** | −0.1391 | 8/8 at every rate |

Floor −0.6909, optimum −0.1353: with the readout plastic it reaches the optimum **exactly**. The
control itself held — `analytic` −0.1361 and `hebbian` −0.8753, both unchanged from the committed
record to five decimals — and its two required expectations both held: `eprop_symmetric` reached
−0.1385 at the pinned rate (8/8 at every rate), and `eprop_scalar` did not pass.

**That retired this change's own exclusion of a plastic readout.** [Logbook 040](040-l4-panel.md)
measured that collapse **under the Hebbian rule**, where a plastic output layer's post-synaptic factor
is its own *output* and its rows self-amplify into saturation. Under e-prop the factor is its own
*error*: the readout's post-synaptic units **are** the action dimensions, so its learning signal is the
identity and its eligibility is its **exact gradient** — no projection, no truncation, no dropped
paths. Pinned against autograd on both substrates.

**And the pilot forced the control that decided the result.** On disjoint seeds 101–104,
`plastic_readout` reached 15.550 foods at 49.13% full clear. A plastic readout is an **8-parameter
linear map** over four pooled means, so "a local rule learns this substrate" and "a small linear
readout on frozen recurrent features learns this cell" predict the same success. `readout_only` freezes
`w_chem` and leaves the readout learning. **It was registered before any `readout_only` run existed**;
what was unknown was the sign.

## Results

Six learning arms and one shared frozen floor, seeds 1–16, 3000 episodes, **112 of 112 runs
succeeded**. Plateau-tail mean foods, each arm against the shared floor, paired one-sided, BH-FDR
across the six.

| arm | readout | `w_chem` | learning | frozen | shift | q | seeds | clear % | drift-cr | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **`readout_only`** | plastic | **frozen** | **17.570** | 2.353 | +15.22 | 0.000 | 16/16 | **52.61** | **0.00** | beats_floor |
| `plastic_readout` | plastic | plastic | 13.675 | 2.353 | +11.32 | 0.000 | 16/16 | 40.22 | 1.31 | beats_floor |
| `symmetric` | frozen | plastic | 13.408 | 2.353 | +11.06 | 0.000 | 16/16 | 15.39 | 1.38 | beats_floor |
| `random_motor` | frozen | plastic | 3.285 | 2.353 | +0.93 | 0.282 | 10/16 | 0.03 | 1.39 | no_improvement |
| `random` | frozen | plastic | 2.981 | 2.353 | +0.63 | 0.684 | 6/16 | 1.58 | 1.37 | no_improvement |
| `scalar` | frozen | plastic | 1.719 | 2.353 | **−0.63** | 0.636 | 8/16 | 0.00 | 1.38 | no_improvement |

Binding minimum 1.66 foods throughout (10% of the gap to PPO's matched 18.945, which exceeds the
absolute 1.0).

### The control inverts the headline

| contrast | what it isolates | result |
|---|---|---|
| `plastic_readout − readout_only` | **the substrate's own plasticity**, at a matched readout | **−3.895 foods, p 1.000** |
| `plastic_readout − random` | the readout, at a matched signal and matched reach | +10.694, p 0.000 |
| `symmetric − random_motor` | the signal's **direction**, at matched reach | +10.123, p 0.000 |
| `random − random_motor` | the signal's **reach**, at a matched source | −0.304, p 0.948 |

Every arm that writes the substrate sits **below** the readout-only control: `plastic_readout` −3.895,
`symmetric` −4.162, `random_motor` −14.285, `random` −14.589, `scalar` −15.851. The control's credited
drift is exactly **0.00**, which is the check that the withholding happened.

So the substrate's plasticity is not merely unnecessary on this cell — it is **actively harmful**, and
it costs more the more the rule is allowed to credit.

### What else the arms say

- **`scalar` sits below its own floor** (−0.63 foods, 0.00% clear). A dynamics-derived trace with no
  per-unit sign is destructive, not neutral, which is what the ablation existed to establish.
- **`symmetric` beats its floor at 13.408 foods and reaches only 15.39% clear** — below the 20% bar.
  High mean foods, few full clears. Against `random_motor` at matched reach it is worth **+10.123
  foods**, so where a true-gradient signal *can* reach, the direction is worth a great deal.
- **Reach buys nothing.** `random − random_motor` is −0.304 at p 0.948: extending an arbitrary
  projection from 39 units to 302 is worth nothing, or slightly negative.
- **The learned readouts are not PPO's.** `readout_only` ends at norm **5.326**, cosine **−0.04** to
  the anatomical default; `plastic_readout` at **4.054**, cosine **+0.108**. PPO's was 7.820 at −0.178.
  e-prop finds a third thing — roughly orthogonal to the anatomical prior rather than opposed to it,
  and at two-thirds of PPO's scale.
- **The registered spatial prediction failed to discriminate.** e-prop drops the multi-hop terms, so a
  learning arm's change should concentrate near the motor pool. `random`'s mean absolute change is
  0.0103 at hop 0 against 0.0065 at the unreachable units — a factor of **1.6**, with no monotone
  falloff (0.0132 at hop 1, 0.0042 at hop 4, 0.0106 at hop 7). The rule writes nearly uniformly over
  graph distance.

## The reading, and what it does not open

**059's third outcome is met in letter and not in substance**, so the third outcome was split on the
control's evidence — the same failure mode R.1c caught in itself, one rung further on: a verdict
condition weaker than the consequence attached to it. `learns_the_cell` now requires a competent arm
that **writes the substrate** and clears the readout-only control by the registered 1.0-food minimum.
No arm does.

- **The substrate rungs stay gated.** B.5, B.1, B.4 and B.4b each ask their question of a rule that
  writes the wiring, and no rule here does so to any benefit.
- **R.1b becomes runnable in a changed form**: wild type against its degree-preserving rewired null as
  **frozen features**, under the readout-only arm. That is a clean question — does the wild-type wiring
  supply better fixed features to a small learned readout than a shuffle does — and arguably closer to
  "is the wiring legible to a learner" than the original. It is registered fresh in its own change.
- **7b's gate is untouched.** It asks for a local rule that beats the null; nothing here beats a null,
  and the competent arms do not write the wiring at all.

## What this may not be cited as

- **Evidence that a local rule learns the connectome.** The competent arms learn with the connectome's
  weights frozen, and every arm that writes them does worse.
- **A result about e-prop in general.** One substrate, one cell, one truncation depth, 16 seeds, and a
  rule whose only large degree of freedom turned out to be an 8-parameter readout.
- **A result about feedback alignment.** One `B` per seed at one scale, and on a substrate whose frozen
  readout denies it the path it would align to — which is a statement about this configuration, not
  about the method.
- **A claim that the readout is all that matters anywhere.** The readout stands in for the entire motor
  periphery here; the substrate ladder's "a body" rung is what would replace it, and this result is
  partly a measurement of that gap.
- **A claim about the 2400-step C3 cell or the block-V cells**, neither of which is run here.

## Corrections on the record

- **`4.361` was carried in four places as R.1c's `motor` arm level.** It is the σ-calibration figure at
  σ 0.1 — a different run. The arm is **3.751** learning against **3.150** frozen, a **+0.601** shift
  on 7 of 8 seeds. Caught before the campaign, by a test that reads the committed record instead of
  trusting the constant.
- **The symmetric projection had to be derived live.** The anatomical contrast overwrites the readout
  *after* the topology is built, and a checkpoint can substitute it again, so a projection cached at
  construction is the transpose of a readout no arm runs with. Caught by the pooling-divisor test.
- **The harness announced the programme's end on a four-seed pilot.** At four seeds the exact
  one-sided paired test cannot reach the significance gate — its smallest reachable p is 2⁻⁴ = 0.0625,
  which BH across the arms pushes above it — so every arm read `no_improvement` whatever it did, and
  `does_not_learn — the programme stops` was printed from evidence that could not have said otherwise.
  A pilot now withholds the verdict and says why.

## Next Steps

- [ ] R.1b in its changed form: wild type against the rewired null as frozen features, under
  `readout_only`, on the block-V cells against the registered ≥ 20% time-to-competence bar.
- [ ] The substrate ladder's rungs stay gated on a rule that writes the wiring to any benefit.
- [ ] V.4, the fresh-rewiring panel, still the named caveat on block V's positive result.
