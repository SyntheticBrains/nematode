# The connectome's perturbation dimension, and the 44% of it that cannot matter (R.1c)

## Why

[R.1](../../../docs/experiments/logbooks/060-l4-perturbation-scale.md) found the rule solving a
multi-step foraging cell at **8 perturbed units** and collapsing to **3.0% full clear at 128**. The
connectome perturbs **302 neurons at each of four settling steps — 1208 draws per scored decision** —
and there is no way to lower it: `node_noise` is applied to every pre-activation. So no connectome arm
follows from R.1, and **R.1b — the wiring contrast 7b's gate actually asks for — is blocked on this.**

Recon found something sharper than a missing knob. The motor readout mean-pools **only the 39
VB/DB/VA/DA motor neurons** ([`connectome_ppo.py:596`](../../../packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py#L596)),
and the eligibility is `E ← decay·E + M_chem ∘ (h_prev ⊗ perturbation)`, so a unit's perturbation
writes eligibility on every synapse onto it whether or not it can reach the readout. At settling step
`s` it can only reach the readout if it is within `depth − s` hops of that pool. Measured on the Cook
2019 graph at `forward_pass_depth: 4`:

| settling step | units that can still reach the readout |
|---|---|
| 1 | 277 of 302 (91.7%) |
| 2 | 247 of 302 (81.8%) |
| 3 | 109 of 302 (36.1%) |
| 4 | **39 of 302 (12.9%)** |

**672 of 1208 draws per decision are causally connected. The other 536 — 44.4% — cannot change the
action at all, and every one of them writes eligibility.** The largest hop budget is 3, at step 1, so
the **25** units at four hops or more can never contribute at any step — 1 at four hops, 7 at five, 7 at
six, 3 at seven or more, and 7 unreachable in the directed chemical graph — and their perturbation is
always noise. 277 + 25 = 302.

## What Changes

- **A declarable perturbation set on the connectome**, which is the mechanism R.1c exists to build:
  per-unit noise restricted to a declared set of neurons, with the set, the synapses it makes
  adaptable and the draws per decision recorded with every run. The plasticity config is **shared with
  the MLP**, which has no connectome to derive a set from, so anything but the default raises there —
  the guard `third_factor: pathway` already sets the precedent for.

- **A causal per-step mask**, the exact form: at step `s`, perturb only units within `depth − s` hops
  of the readout pool. This is **not** a dimension knob — it removes only draws that provably cannot
  influence the action, so no usable signal is lost. 1208 → **672** draws per decision.

- **A dimension axis nested by proximity to the readout**, so K moves while every arm's perturbed
  units can still affect behaviour — which a *random* subset would not guarantee, since the connectome's
  units differ enormously in their influence on the output:

  | mask | units | adaptable synapses | draws/decision |
  |---|---|---|---|
  | none — today | 302 | 3709 | 1208 |
  | causal per-step reach | 302 | 3709 | 672 |
  | ≤ 2 hops | 247 | 3242 | 988 |
  | ≤ 1 hop | 109 | 1476 | 436 |
  | **motor pool only** | **39** | **323** | **156** |

  The motor-only arm sits between the MLP's 32 units (which reached 53.6% full clear) and 64 (which was
  marginal), which is the informative place for it to be.

- **A settling-step axis** — all four steps against the last step only. This separates *units* from
  *draws per decision*, the ambiguity R.1's extrapolation flagged, and is the settling-step control
  R.1's mechanism claim was explicitly left qualified pending.

- **The hard-food cell under the rule on the connectome**, which does not exist as a config: every
  plastic connectome arm to date runs the 2400-step C3 cell.

Out of scope: the wiring contrast itself (R.1b, which this unblocks); e-prop (R.2); and any
re-registration of the substrate rungs (R.3).

## Capabilities

**Modified**: `connectome-substrate` (the perturbation set is declarable, and its size, adaptable-synapse
count and draws per decision travel with the run). **Modified**: `plasticity-evaluation` (a perturbation
that cannot reach the scored outcome is not counted as exploration, and a dimension claim separates
draws that can from draws that cannot).

## Impact

- New: the perturbation-set mechanism and its config surface; the hop-distance mask derived from the
  loaded connectome; configs for the cell and the arms; `scripts/analysis/l4_reduced_perturbation.py`
  and tests; records under `supporting/061-l4-reduced-perturbation/`; Logbook 061.
- Edited: the experiments index, `CHANGELOG.md`, the tracker (R.1c, and R.1b's block lifted or not),
  the roadmap only if the reading changes.
- Compute: connectome runs, measured at ~2.1 ms an environment step (settling included) **on the C3
  cell**, which carries predator and thermal modules this cell does not — so the resulting ~20–37
  minutes a run at 3000 episodes of up to 350 steps is **conservative**. The pilot calibrates it and the
  campaign is scheduled from the measurement, not from this estimate.
