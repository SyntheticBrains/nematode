# Design: e-prop on the connectome, and what its eligibility can and cannot carry

## The mechanism, stated against the one it replaces

The three-factor rule is unchanged: `Δw = η·δ·E − η·λ_w·w`, `δ = r − b`. What changes is `E`.

| | node perturbation | e-prop |
|---|---|---|
| post-synaptic factor | `ξ_j`, the noise the unit acted on | `psi_j · L_j`, its activation derivative times a broadcast learning signal |
| pre-synaptic factor | `h_i` at the previous env step | `h_i` at each settling step, summed within the step |
| what makes it a gradient estimate | averaging over draws | the chain rule, truncated to the direct path |
| variance source | the draws themselves | the projection the learning signal arrives through |
| cost per decision | 1208 draws on the connectome | none |

On the settling recurrence `h^(s+1) = tanh(W^T h^(s))`:

```text
d h_j^(s+1) / d w_ij  =  psi_j^(s+1) * ( h_i^(s)  +  sum_k (d h_k^(s) / d w_ij) * W_kj )
```

e-prop keeps the first term and drops the sum — that truncation *is* the method, and it is what makes
the trace computable forward in time with no backward pass and no stored graph. On this substrate the
units carry no membrane state between settling steps (`h` is replaced, not leaked), so the trace has no
within-unit recursion to carry either: the eligibility is the sum over settling steps of
`psi_j^(s+1) · h_i^(s)`, and the decay across environment steps is the existing `trace_decay`.

## The truncation decides the arm set, not just the accuracy

The motor readout is a `(4, 2)` matrix over the **mean-pooled** VB/DB/VA/DA classes
([`connectome_ppo.py:684`](../../../packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py#L684)),
so the exact derivative of the action mean with respect to one unit's activity is

```text
d mu_k / d h_j  =  readout[k, class(j)] / |class(j)|      j in the 39-unit pool
                =  0                                     for the other 263 units
```

with everything beyond the pool reaching the action **only** through the multi-hop paths e-prop drops.
So symmetric feedback under this truncation is not "the true direction, broadly applied" — it is a
learning signal of exactly zero on 263 of 302 units, which makes it the e-prop analogue of R.1c's
`motor` perturbation set. That set is not a neutral comparator either: it was R.1c's **best** arm --
3.751 learning against 3.150 frozen, a shift of **+0.601** on 7 of 8 seeds.

The campaign therefore crosses the two things the routing varies:

| arm | signal source | credited breadth | readout | what it answers |
|---|---|---|---|---|
| `symmetric` | readout transpose | the 39-unit pool, **forced** | frozen | does the true direction help, where it can reach |
| `random_motor` | fixed random `B`, masked to the pool | the same 39 units | frozen | is it the direction, or is it the pool |
| `random` | fixed random `B` | all 302 | frozen | can a broadcast projection credit the far units usefully |
| `scalar` | none, `L_j = 1` | all 302 | frozen | did the per-unit signal do anything at all |
| `plastic_readout` | fixed random `B` | all 302 | **plastic** | can a broadcast projection work **at all**, once the path it would align to may move |
| `readout_only` | fixed random `B` | all 302 | **plastic** | how much of that is the substrate, and how much is an 8-parameter linear readout (`w_chem` frozen) |

The missing fourth cell — true directions reaching all 302 units — is **the cell the mechanism
forbids**, and stating it is part of the result rather than a gap in the design. `random_motor` exists
because without it a `symmetric` win reads as two claims at once and R.1c already measured the weaker
one. `scalar` is an ablation and not a candidate: if it matches `random`, the per-unit signal did
nothing and the result is about the dynamics-derived trace alone.

**The spatial prediction this makes is registered in advance.** e-prop's dropped terms are exactly the
multi-hop ones, so where a learning arm's weight change lands should concentrate near the motor pool —
and `random`, the only arm that can credit the far units, is where the prediction is testable. Reported
by hop distance using the same walk R.1c committed
([`_readout_hop_distances`](../../../packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py#L917)).

## What stage 1 changed, before any arm ran

The four arms above share a frozen readout, and stage 1 measured that this is not a neutral choice.
Feedback alignment works because the forward path to the output comes into alignment with the
feedback matrix; a frozen readout cannot, so `random` — the arm the plausibility claim rests on — is
**structurally unable to work** in the configuration all four share. Same task, same rule, same
projection, 20,000 trials:

| readout | rate 1e-4 | rate 1e-3 | rate 1e-2 | seeds above floor |
|---|---|---|---|---|
| **frozen** (`eprop_random`) | −0.8147 | −0.7097 | −0.7031 | 4/8, 6/8, 6/8 — fails at every rate |
| **plastic** (diagnostic) | −0.2726 | **−0.1353** | −0.1391 | 8/8 at every rate |

Floor −0.6909, optimum −0.1353. With the readout plastic the broadcast arm reaches the optimum
**exactly**, on every seed. That is not a marginal difference in degree.

**This change's original exclusion of a plastic readout was not supported by the evidence it cited.**
Logbook 040 measured a plastic readout collapsing under the **Hebbian** rule, where a plastic output
layer's post-synaptic factor is its own *output*, so its rows self-amplify toward whatever maximises
the action mean (density 1e17). Under e-prop the factor is its own *error*, and there is no such
loop: the readout's post-synaptic units **are** the action dimensions, so its learning signal is the
identity and its eligibility is

```text
E[k, c] <- decay * E[k, c] + score_k * pooled_c
```

which is the **exact gradient** of the action log-probability with respect to the readout — no
projection, no truncation, no dropped paths. It is the one place in this mechanism where nothing is
approximated, and it is pinned against autograd by test.

Two consequences carried deliberately:

- **The readout is excluded from the homeostatic rescale.** That rescale returns each unit's incoming
  norm to its construction value, and the readout's scale is part of what this arm asks about — R.1d
  measured **+4.51 foods from scale alone** with the direction held. Pinning it would leave the arm
  able to rotate the readout and not to resize it, which is half the question. The rule honours this
  through a per-tensor flag the topology supplies, so no other substrate changes.
- **The rule's weight bound still applies**, at 3.0 per entry. That allows a readout norm of 8.49
  against the **7.820** R.1d's PPO harvest reached, so it does not bind on the scale that mattered —
  stated because it is a bound the arm runs under, not because it is expected to matter.

## The learning signal is the part that is not a drop-in

`eps_ij` is unsigned with respect to the outcome. It is large where `w_ij` has leverage on `h_j` and
says nothing about whether more of `h_j` was good. Two readings of "a global learning signal" give
different rules, and the difference matters enough to register:

1. **One scalar reaching every unit.** Then `E_ij = eps_ij` and the update is `η·δ·eps` — every synapse
   with leverage moves the same way, gated only by how much leverage it has. That is reward-modulated
   Hebbian with an activation derivative in place of the post-synaptic rate. It is coherent, the panels
   have measured relatives of it, and **it is not a gradient estimator**; calling it e-prop would
   overstate it. It is the `scalar` arm.
2. **One error vector, broadcast through a fixed projection.** `L_j = Σ_k B_jk g_k`, `B` fixed and drawn
   once. No synapse computes `B`, nothing propagates backwards, and the same error vector reaches
   everything — which is what makes it broadcast. This is reward-based e-prop with feedback alignment,
   and it is the reading D1's fallback names.

### The score function is taken pre-squash

The continuous head samples a tanh-squashed Gaussian
([`continuous_sample_tanh_gaussian`](../../../packages/quantum-nematode/quantumnematode/brain/arch/_policy.py)),
and `∂ log π / ∂ mu_k = (u_k − mu_k) / sigma_k²` where `u` is the **pre-squash** draw: the squash's
log-determinant correction depends on `u` but not on `mu`, so it contributes nothing to this
derivative. `_continuous_action_step` already keeps that draw as `pre_tanh`, so the signal is available
without changing the head.

The **categorical** head is a different score function (`onehot − probs`). This change does not
implement it and the config layer **refuses** `eprop` on a discrete-action config rather than deriving
a signal from a head it was not written for.

### Where the signal is folded in

The trace update currently happens inside the forward pass, before the action exists. The score
function needs the sampled action. So the topology accumulates `eps` during settling and the brain folds
the signal in immediately after sampling:

```text
forward:              eps_ij = sum_s psi_j^(s+1) * h_i^(s)          (held, not yet credited)
after the action:     E      = decay * E + M ∘ (eps * L)            (one call, once per env step)
```

This is a new call on the plastic-topology seam rather than a new argument to `forward`, for two
reasons: the untraced batched forward (the PPO replay path) must keep touching nothing, and a rule
reading `plastic_post_activities` must continue to see the same vector the trace was built from.
**An env step whose forward ran but whose signal was never folded in must fail rather than credit a
stale `eps`** — the failure mode is silent and would read as a rule that learns slowly. `mlpppo` has no
call site for it in this change, so it refuses `eprop` at load rather than raising mid-episode.

## Why the rate carries over, which is load-bearing and was nearly left implicit

e-prop's raw trace has nothing like node perturbation's magnitude: `psi · h · L` against `h ⊗ ξ` at
`σ = 0.1`. The committed arm already runs **`plasticity_normalise_trace: true`**, which divides the
Hebbian term by `rho`, a running RMS of each tensor's trace over its edge set, so `plasticity_rate`
means *the root-mean-square step per unit modulator* rather than an absolute step. That is exactly what
makes `0.001` transferable across eligibilities, and it is why this change does not re-tune the rate —
together with `plasticity_normalise_modulator: true`, which bounds the third factor the same way.

Normalisation equalises scale, not distribution: `eps · L` is signed and concentrated where the RMS is
taken over the whole edge set. So a **rate check on the pilot's disjoint seeds** is registered — the
committed rate and one decade either side, four seeds, learning arm only — with the committed rate
standing unless it is visibly off. This is the same discipline as R.1c's σ calibration, which found a
carried-over value costing 31.5% of the arm's level, and it is what stops a `does_not_learn` reading
being a rate artefact.

## Stage 1's arms, and why `symmetric` must pass

On the committed contextual association there is one plastic layer, one forward pass, one scalar
Gaussian action. `eps_ij = psi_j · h_i`, `g = (a − mu)/sigma²`, and with `B = W_out^T` the product
`δ · eps_ij · L_j` **is** the REINFORCE gradient of expected reward with respect to that layer's
weights. The harness's `analytic` arm already runs gradient descent on the same topology, so the two
should agree in direction.

| arm | required | what a failure means |
|---|---|---|
| `symmetric` | passes | the implementation is wrong — the derivative, the score function, or the fold-in. Nothing downstream is interpretable |
| `random` | reported, not required | feedback alignment does or does not align on a 4-cue association. Informative either way, and a fail bounds what stage 2 could show |
| `scalar` | must **not** pass | a rule with no per-unit signal solving a cue-to-target association means the task is not discriminating, and the control is VOID |

The last row is the existing `hebbian` floor arm's logic applied to the new axis: the control's value
comes from something being required to fail.

## Stage 2's arms, and the one number they are read against

The cell is R.1c and R.1d's `hard350` connectome cell at `initial_log_std: -1.0` with the **committed
anatomical readout** — not R.1d's `anatomical_scaled`. Two reasons. The scaled readout's norm came from
a PPO harvest, so an arm using it inherits R.1d's "cannot satisfy the plausibility deliverable" flag,
and having a rule that *can* satisfy it is the whole point of e-prop. And the anatomical arm is the one
with a matched node-perturbation comparator already on the record.

The consequence is registered in advance: **R.1d puts a ceiling on what any rule writing `w_chem` alone
can reach through this readout**, and every perturbation set R.1c ran sat in a 2.4–4.4 food band. If
e-prop clears its floor but lands in that band, the reading is *the readout scale binds before the
credit-assignment rule does*, the follow-up is the readout-scale / action-noise interaction R.1d
surfaced, and it is **not** grounds for re-running e-prop at a scaled readout inside this change.

| arm | `freeze_updates` | routing | readout | runs |
|---|---|---|---|---|
| `eprop_symmetric` | no | `symmetric` | frozen | 16 |
| `eprop_random_motor` | no | `random_motor` | frozen | 16 |
| `eprop_random` | no | `random` | frozen | 16 |
| `eprop_scalar` | no | `scalar` | frozen | 16 |
| `eprop_plastic_readout` | no | `random` | **plastic** | 16 |
| `eprop_readout_only` | no | `random` | **plastic**, `w_chem` frozen | 16 |
| `eprop_frozen` | yes | — | frozen | 16 |

One frozen floor for all six: with updates frozen no weight moves, so neither the routing nor a
plastic readout can reach the behaviour, and a config declaring both a plastic readout and a freeze
is refused rather than reported as a plastic-readout floor. Asserted by an exact-key config test
rather than argued — the four frozen-readout configs differ from each other in the routing key
**alone**, `plastic_readout` differs from `random` in `plasticity_plastic_readout` alone, and the
frozen config differs from `random` in `freeze_updates` alone.

It is **not** R.1c's frozen floor: that arm ran at `plasticity_node_noise` 0.1, and the perturbation
enters the forward pass whether or not updates are frozen, so its floor is a *noisier* policy than this
one. Reusing it would compare against the wrong null, and the 16 runs are spent rather than saved.

## The verdict, and what each branch costs

Per arm, against the shared frozen control, paired by seed, one-sided, BH-FDR across the **six**
learning arms, with 80% bootstrap CIs — the statistics layer R.1c and R.1d used, unchanged. Both
registered minima apply as they did there: the absolute 1.0-food floor and the 10%-of-reachable-gap
minimum taken against PPO's matched **18.945**, with the larger binding.

| reading | test | what follows |
|---|---|---|
| `does_not_learn` | no learning arm beats the floor by the binding minimum | 059's first outcome fires: the programme **stops**, 7b proceeds under PPO after the power arithmetic, the plausibility claim is given up. The rule family has now failed with two independent eligibilities |
| `learns_below_competence` | a learning arm beats the floor; none reaches the 20% full-clear threshold | a result. **R.1b stays blocked and 7b's gate is untouched**, since block V's contrast is on time to competence. The readout-scale follow-up becomes the live question |
| `learns_the_cell` | a learning arm reaches competence **while writing the substrate**, clearing the readout-only control by at least the 1.0-food minimum | 059's stage 3 becomes runnable: the wiring contrast on the block-V cells against the ≥ 20% bar, registered fresh in its own change. **B.5, B.1, B.4 and B.4b become askable** under R.3 |
| `learns_without_the_substrate` | a learning arm reaches competence, and none does so while writing the substrate | 059's gate is met in **letter and not in substance**. Every substrate rung asks its question of a rule that writes the wiring, so **B.5, B.1, B.4 and B.4b stay gated**. R.1b becomes runnable in a changed form: wild type against its rewired null as **frozen features** under the readout-only arm |
| `void` | stage 1's `symmetric` arm fails, or its `scalar` arm passes | nothing in stage 2 is interpretable and no connectome run is spent |

The `void` row is why stage 1 is a stop clause and not a task: **no stage-2 run is launched until it
passes.**

### Why the third outcome needed splitting

*(Added after the campaign, on the control's evidence.)* 059 registered the gate as "a rule that
learns the hard-food cell", and that phrasing cannot distinguish two very different worlds on this
substrate. A plastic readout is an **8-parameter linear map** over four pooled motor-class means, so
an arm can learn this cell with the wiring frozen — and then "the cell was learned" says nothing
about the wiring, which is the only thing every substrate rung is about.

The split is not a post-hoc convenience: the **control that forces it was registered before any
`readout_only` run existed**, and its stated purpose was exactly this separation. What was unknown
in advance was the sign. The condition uses the registered absolute minimum rather than a new
threshold — the substrate's contribution must clear **1.0 food** over the control to count, the same
bar every arm's effect is held to.

This is the failure mode R.1c caught in itself, one rung further on: a verdict condition weaker than
the consequence attached to it. R.1c surfaced its own before the campaign; this one was surfaced by
the control after it, which is the next best thing and the reason the control was added.

## What this may not be cited as

- **A result about e-prop in general.** One substrate, one cell, one truncation, 16 seeds, and a rule
  that writes `w_chem` alone through a frozen anatomical readout R.1d has already shown to be part of
  the limit.
- **A result about feedback alignment.** `random` draws one `B` per seed at one scale; nothing here
  sweeps the projection, and a broad arm failing is not evidence that no projection works.
- **Evidence that the dynamics-derived eligibility beats perturbation**, unless `scalar` separates from
  the floor — otherwise the comparison is confounded with the learning signal.
- **A result about the true gradient direction**, unless `symmetric` separates from `random_motor`:
  matched on breadth is the only comparison that isolates the direction.
- **A claim that the substrate learned anything**, unless `plastic_readout` separates from
  `readout_only`: a plastic readout is an 8-parameter linear map over four pooled means, and on its
  own it says nothing about the wiring behind it.
- **A claim about the 2400-step C3 cell or the block-V cells**, neither of which is run here.
