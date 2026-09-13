# R.1c — a declarable perturbation set for the connectome: design

## What the substrate does today, and why 44% of it cannot help

The eligibility is `E ← decay·E + M_chem ∘ (h_prev ⊗ perturbation)`, so a unit's perturbation writes
eligibility on **every chemical synapse onto it**, regardless of whether that unit can influence the
action. The action comes from a mean-pool over the **39 VB/DB/VA/DA motor neurons** only. With
`forward_pass_depth: 4`, a perturbation injected at step `s` reaches the pool only if its unit is within
`4 − s` hops.

| settling step | hop budget | units that can reach the readout |
|---|---|---|
| 1 | 3 | 277 of 302 |
| 2 | 2 | 247 of 302 |
| 3 | 1 | 109 of 302 |
| 4 | 0 | **39 of 302** |

**672 of 1208 draws per decision are causally connected; 536 are not.** The largest budget is 3, at step
1, so the **25** units at four hops or more can never contribute at any step: 1 at four hops, 7 at five,
7 at six, 3 at seven or more, and 7 unreachable in the directed chemical graph. 277 + 25 = 302.

**A masked synapse stays put only because homeostasis cancels an unconditional decay, and the
cancellation is close but not exact.** The rule applies `− rate · weight_decay · weight` to **every**
plastic weight, trace or no trace, so a masked unit's incoming weights shrink on every update. The
homeostatic rescale returns each unit's incoming norm to its construction target, and both operations
are purely radial, so they cancel — measured on the recipe's own values (rate 1e-3, decay 1e-3) with
every trace held at zero:

| updates | incoming-norm change | largest single weight's excursion | 1 − cosine of the weight vector |
|---|---|---|---|
| 1 000 | 0.0000% | 1.8e-05 | 0 |
| 10 000 | 0.0000% | 1.9e-04 | 6e-08 |
| 100 000 | 0.0000% | 1.9e-03 | 2.6e-06 |

So the norm is held **exactly** and the direction to a cosine of **1 − 3e-05** extrapolated across a
full run's ~1.05M updates. What remains is float32 round-off, accumulating linearly in the largest
single-weight excursion — about 1.9e-02 over a run, roughly 9% of these weights' rms — while leaving
the vector the unit actually computes with unchanged. That is jitter, not decay and not learning.

**Without homeostasis it is decay.** The same bench run loses **2.0% of the norm per 20 000 updates**,
which compounds to a collapse over a run: a `motor` arm would shrink its 3,386 uncredited synapses and
be a covert global weight-decay experiment reported as a dimension result. So masked arms are **only
supported with homeostasis on**; a config validator refuses the combination, and the test asserts both
directions — norm conserved with it, norm lost without it — so the dependency is measured rather than
assumed.

**And it is measured in the campaign, not only on the bench.** Drift is reported separately for
**excluded** and **credited** synapses per arm, so the claim that excluded weights only jitter is
checked on the real substrate at the real scale rather than argued from this table. Note the residual is
a property of the rule that **every** plastic result already carries; what a restricted set changes is
that it becomes the *only* update those synapses receive.

Two distinct things follow, and the design keeps them apart:

- **A correctness fix.** Masking the 536 disconnected draws removes variance from the estimator and
  **no signal**, because those draws provably cannot move the outcome they are credited against.
- **A dimension knob.** Restricting the perturbed set further trades adaptable synapses for a smaller
  dimension, which is a real cost and not a free improvement.

## The arms

One cell: the **calibrated hard-food connectome cell** (`max_steps: 350`,
`target_foods_to_collect: 20`) — block V's cell, so a result here is directly comparable both to R.1's
MLP sweep and to the PPO reference already on the record.

| arm | perturbed set | units | adaptable synapses | draws/decision |
|---|---|---|---|---|
| `full` | every unit, every step — **today's behaviour** | 302 | 3709 | 1208 |
| `causal` | per-step reach mask | 277 | 3538 | **672** |
| `hop1` | within 1 hop of the readout pool | 109 | 1476 | 436 |
| `motor` | the readout pool itself | 39 | 323 | 156 |
| `motor_last` | the readout pool, **last settling step only** | 39 | 323 | **39** |

Realised causally-connected draws, measured from the built masks: `full` 672 of 1208, `causal` 672
of 672, `hop1` **366 of 436** — a hop-1 unit cannot reach the pool from the last step — `motor` 156
of 156, `motor_last` 39 of 39.

Each carries **its own frozen control** at the same σ and the **same mask**, freezing only the update —
R.1's lesson, so the perturbation's cost to the policy is matched across each pair and the contrast
measures the update alone. Eight seeds, 3000 episodes. **Ten arms, 80 runs.**

**Correction, 2026-09-13, before any arm ran.** The table above first gave `causal` as 302 units and
3709 adaptable synapses, on the reasoning that it removes only draws that cannot matter. The
implementation shows that is wrong in one respect: the **25** units at four hops or more are never
perturbed at *any* step, so their **171** incoming synapses are never credited either. `causal` is
therefore 277 units and 3538 synapses, coinciding with the units of a "within 3 hops" set.

What it gives up is credit those synapses could never have earned informatively — their units cannot
influence the action from any settling step — so the claim "no signal lost" still holds, while the
claim "every synapse stays adaptable" does not. Both are now stated.

`hop2` (247 units, 988 draws) is deliberately omitted: it is within 1.2× of `full` on the axis under
test and would spend 16 runs to interpolate a curve the other arms already bracket.

### The capability reference is already on the record

Block V ran PPO on this exact cell and substrate over **32 seeds**
([Logbook 058](../../../docs/experiments/logbooks/058-wiring-premise-difficulty.md)), so **no PPO arm is
re-run here**:

| arm (058, 32 seeds) | mean foods of 20 | full clear |
|---|---|---|
| wild type under PPO | **19.31** | — |
| wild type, frozen weights | **3.82** | **0.0%** |

So the reachable gap on this cell is **15.49 foods**, which fixes the relative minimum at **1.55 foods**
rather than leaving it to be computed later. Note 058's frozen arm does **not** perturb, so it is a
reference and not the comparator: every arm here carries its own frozen control at its own mask and σ,
which is the only matched null.

For scale, the MLP's frozen arms on the same cell ran 1.69–3.89 foods in R.1, beside this substrate's
3.82 — a small consistency check, not a comparison.

## The reading

**Plateau-tail mean foods** through I.2's graded family, learning against its own frozen control, paired
by seed, one-sided, BH-FDR across the five masks. Full-clear success recorded alongside; where it sits
at the floor the record says so rather than reporting a null.

**A shift counts only if significant *and* both**: at least **1.0 foods** of the cell's 20, and at least
**10%** of the reachable gap. Against 058's committed reference the gap is 19.31 − 3.82 = **15.49
foods**, so the relative minimum is **1.55 foods**; where an arm's own frozen mean differs from 3.82 the
harness recomputes it per arm and reports both.

**Drift** per arm — relative weight distance from its own frozen control — so starved-and-still is
distinguishable from writing-a-lot-in-a-worsening-direction, which is what separated the two regimes in
R.1.

**Every run records its perturbation set**: the declared mask, the unit count, the adaptable-synapse
count and the draws per decision. A dimension claim that does not carry those numbers is the thing R.1
had to reconstruct after the fact.

## Outcomes, fixed before the run

| verdict | test | what follows |
|---|---|---|
| `causal_mask_sufficient` | the `causal` arm beats its own frozen control by both minima | the strongest available result: the failure was **uninformative noise**, not scale, and nothing had to be given up to fix it. R.1b runs at the causal mask with all 3709 synapses adaptable |
| `dimension_reducible` | some reduced arm (`hop1`, `motor`, `motor_last`) beats its control by both minima, and `causal` does not | the dimension binds on the connectome as it does on the MLP. **R.1b is unblocked at the winning mask**, which becomes its registered dimension, and the record states which synapses that mask gives up |
| `not_reducible` | no arm beats its own frozen control | the connectome's failure is **not** the perturbation dimension. R.1b stays blocked, R.2 (e-prop) becomes the live path, and R.1's result stays bounded to the MLP |

A partial reading — `motor_last` working where `motor` does not, or the ordering non-monotone — is
recorded as **partial with the ordering stated**, not resolved toward the nearest verdict.

**A deliverable that does not depend on the learning arms.** The causal-reach table is a property of the
substrate at a given depth, computed from the loaded connectome, and it stands whatever the arms do. A
`not_reducible` verdict still ships the measurement that 44.4% of this substrate's perturbation is
credited against an outcome it cannot influence.

### Stop clauses — void until found

- **`full` does not reproduce the known failure.** Every plastic connectome result to date has it at or
  below its floor; if it learns here, something changed since 055 and nothing in the sweep is
  interpretable until that is found.
- **A mask does not mask.** Each arm asserts its realised draws-per-decision against the declared value,
  and that a masked unit's incoming synapses do not move across an episode. A mask that silently does
  nothing would make its arm a duplicate of `full` reported under another name.
- **The frozen floor is at the ceiling.** If the frozen arms already clear the cell, there is nothing to
  add and the finding is about the cell, not the rule.

## Honest prior

**`not_reducible` for `causal` alone, and better than even for `motor`.** The arithmetic is the reason to
say so in advance:

- the causal mask takes draws per decision from 1208 to 672 — a **1.8×** reduction;
- R.1's MLP curve needed **16×** (128 → 8 units) to move from 3.0% to 90.7% full clear, and 128 → 64 (a
  2× reduction) only reached 12.9%.

So a 1.8× reduction is very unlikely to be sufficient on its own, and the causal mask should be argued
for as **correctness** rather than as a knob. `motor` at 156 draws is a **7.7×** reduction and 39 units,
which sits between the MLP's 32 (53.6% full clear) and 64 (12.9%) — the informative place to be.

Against it: only **323 synapses** are adaptable at `motor`, against 3709 unmasked. There may simply be
too little left to express — though the frozen prior is a long way from competent on this cell (3.82 of
20 foods and **0.0% full clear** over 32 seeds), so there is a large gap to move into rather than a small
one. That is the risk the `hop1` arm exists to bracket, at 1476 synapses.

## Risks

- **Reduced masks buy dimension with adaptable synapses.** Unlike R.1's MLP sweep, where narrowing the
  network narrowed everything together, here the substrate is fixed and the mask only chooses who gets
  credit. A win at `motor` is a win for a **restricted learner**, and the record says so.
- **Nested-by-proximity is not the only ordering.** It is chosen because a random subset of 302 units
  would mostly pick units with little influence on a 39-unit readout, confounding "how many" with
  "which". The cost is that the arms differ in identity as well as count, which the record states rather
  than claims to have separated.
- **The masked arms are coupled to homeostasis** by the decay cancellation above. That is pinned and
  tested, but it means these arms cannot be re-run at `plasticity_homeostasis: false` without the mask
  becoming a decay manipulation, and the configs say so.
- **Eight seeds**, the same power limit every block-I and R result carries; hence both effect minima.
- **One cell.** A reduced dimension that works here says nothing about the 2400-step C3 cell, where the
  horizon is seven times longer and R.1 found the horizon interacting with the dimension.
- **The hop distances are computed on the directed chemical graph** and ignore gap junctions, which are
  frozen and bidirectional. A gap junction could carry influence the hop count misses, so the causal mask
  is **conservative in the wrong direction** — it may mask a draw that could have mattered through a gap
  path. The arm is registered with that stated, and the gap-inclusive variant is named as the follow-up
  rather than silently folded in.
