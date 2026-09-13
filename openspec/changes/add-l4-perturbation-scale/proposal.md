# The perturbation dimension: was the rule ever run at a scale it could work at? (R.1)

## Why

The rule's one success and all of its failures differ in a dimension nobody varied.

The positive control it passes is `Linear(K, 8) → tanh → Linear(8, 1)` — **8 perturbed units**, one
plastic layer ([`l4_rule_positive_control.py:91`](../../../scripts/analysis/l4_rule_positive_control.py#L91)).
Every MLP yardstick arm that failed ran `actor_hidden_dim: 64, num_hidden_layers: 2` with
`plastic_layers: hidden` — **128**. The connectome perturbs all **302** neurons, at **each of four
settling steps** ([`connectome_ppo.py:1142`](../../../packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py#L1142)).

Node perturbation estimates a gradient from a scalar reward and N perturbed units, and its learning
speed falls roughly as **1/N** (Werfel, Xie & Seung 2005). So the phase's central pattern — one-step
success, multi-step failure — is confounded with a 16× and a 38× difference in perturbation
dimension, and the scale axis has never been tested.

## What Changes

- **Two sweeps, in this order.** The arithmetic is tested where the rule demonstrably works before it
  is read off where it fails:

  1. **S1, the arithmetic.** The committed one-step control at `HIDDEN ∈ {8, 16, 32, 64, 128}`, 8
     seeds, 20 000 trials, scored by the control's own registered pass rule. This platform has **no
     capacity confound** — the task is solvable by 8 units and every further unit only adds noise —
     so N is isolated. It also yields a **rate**: trials-to-criterion, which is what 1/N predicts and
     what a pass/fail reading throws away.
  2. **S2, the rescue.** The MLP yardstick on the calibrated hard-food cell at
     `actor_hidden_dim ∈ {4, 8, 16, 32, 64}` — **8 to 128 perturbed units**, the same grid as S1 —
     learning arm against **its own** frozen control at each width, 8 seeds, 3000 episodes.

- **A capability arm per width in S2.** Width is capacity as well as perturbation dimension, and the
  prediction runs toward *small* N, so a width too small to hold the policy would produce a null that
  reads as evidence against the hypothesis. Each width therefore carries a **PPO arm**: a width whose
  PPO arm misses the registered floor is reported **uninterpretable**, not as a null. PPO is a
  capability floor, not a ceiling for the rule.

- **A pilot on disjoint seeds 101–104 before S2.** The 350-step budget was calibrated on the
  *connectome*; nothing says it leaves an MLP room. The pilot runs the two extreme widths, and a
  declared remedy applies if the platform has no room.

- **A derived budget, labelled as extrapolation.** If S1 fits trials-to-criterion against N, the fit
  says what budget 128 and 302 units would need, against what the panels actually spent. A five-point
  fit read outside its range is an **extrapolation and is recorded as one**, never as a measurement.

- **Three registered outcomes**, including the one where the arithmetic is confirmed and still does
  not rescue the multi-step task — which would make the multi-step failure a **second, independent**
  defect rather than the same one.

Out of scope: any connectome run (registered only if S1 and S2 both move); a reduced-perturbation
connectome variant; e-prop, which is R.2; and any change to the rule itself.

## Capabilities

**Modified**: `plasticity-evaluation` — a mechanism whose predicted effect depends on a platform
dimension has that dimension varied where the rule works before a failure elsewhere is attributed to
it, and a stochastic-gradient result records the perturbation dimension it was measured at.

## Impact

- New: `scripts/analysis/l4_perturbation_scale.py` and its tests; a width axis on the control harness;
  fifteen configs for S2, and three more only if the declared small-N alternative is used; records under `supporting/060-l4-perturbation-scale/`; Logbook 060.
- Edited: the experiments index, `CHANGELOG.md`, the tracker (new **R** block, R.1), the roadmap only
  if the reading changes.
- Compute: S1 is in-script and runs in minutes. S2 is **120 registered runs** plus a pilot — above the
  80 that [Logbook 059](../../../docs/experiments/logbooks/059-7a-shipment.md) estimated, because the
  capability arm was not in that sketch; the cell runs at 350 steps rather than the yardstick's 2400.
