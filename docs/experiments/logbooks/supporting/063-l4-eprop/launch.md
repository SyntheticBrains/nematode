# R.2 — e-prop: the registered protocol

Registered in `openspec/changes/add-l4-eprop`, reviewed and committed **before** the arms run.

## The question

Node perturbation is closed. [R.1](../../060-l4-perturbation-scale.md) found the rule solving a
multi-step cell at **8 perturbed units** and collapsing to 3.0% full clear at 128;
[R.1c](../../061-l4-reduced-perturbation.md) took the connectome from **1208 draws per scored decision
down to 39** and returned `not_reducible`; [R.1d](../../062-l4-frozen-readout.md) showed the frozen
readout is part of the limit — 3.751 → 9.639 foods — with **no arm reaching competence**.

What all three share is the argument for this one. **Credited-synapse drift sat at 1.37–1.42× the
weight's own norm in every one of them**: 302 perturbed units and 39, four readouts at two norms and
two directions. The rule is never starved of signal. It writes a great deal, in a direction that does
not help. The remaining suspect is **what the eligibility carries**.

Under node perturbation `E ← decay·E + M ∘ (h_prev ⊗ ξ)`: the draw `ξ` is the only thing telling a
synapse which way its post-synaptic unit moved, and its variance grows with the number of units
perturbed. **e-prop removes the draw**: the trace comes from the network's own settling derivative and
the sign comes from a broadcast learning signal.

## The mechanism

```text
eps_ij = sum over settling steps s of  psi_j^(s+1) * h_i^(s)     psi = 1 - tanh^2
E_ij  <- decay * E_ij + M_chem ∘ (eps_ij * L_j)
g_k   = (u_k - mu_k) / sigma_k^2        u = the PRE-SQUASH Gaussian draw
L_j   = sum_k B_jk * g_k                B fixed for the run, persisted with the checkpoint
```

`eps` is the **local** part of `∂h_j/∂w_ij`; the paths through other units are dropped, and that
truncation *is* the method. The trace is unsigned with respect to the outcome, so it is held during the
forward and credited the moment the action exists.

## What recon found, and why there are four arms

The motor readout is a `(4, 2)` matrix over the **mean-pooled** VB/DB/VA/DA classes, so
`∂mu_k/∂h_j = readout[k, class(j)] / |class(j)|` on the 39 pooled units and **exactly zero on the
other 263** — and e-prop drops the multi-hop paths by which any of those 263 reaches the action. So
**symmetric feedback is not a broad arm with true directions**: it reaches only the readout pool, which
makes it the e-prop analogue of R.1c's `motor` perturbation set — **3.751 learning against 3.150
frozen, a shift of +0.601 on 7 of 8 seeds, R.1c's best arm**. The routing therefore varies two things at once, and the campaign crosses them:

| arm | signal source | units it can reach | readout | what it answers |
|---|---|---|---|---|
| `symmetric` | the readout's transpose | the 39-unit pool — **forced** | frozen | does the true direction help, where it can reach |
| `random_motor` | fixed random `B`, masked to the pool | the same 39 | frozen | is it the direction, or is it the pool |
| `random` | fixed random `B` | all 302 | frozen | can a broadcast projection credit the far units usefully |
| `scalar` | none, `L_j = 1` | all 302 | frozen | did the per-unit signal do anything at all |

The fourth cell of that 2×2 — **true directions reaching all 302 units — is one the mechanism
forbids**, and the harness records it as forbidden rather than omitting it.

## The fifth arm, added after stage 1 and before any arm ran

The four above share a **frozen** readout, and stage 1 measured that this is not a neutral choice.
Feedback alignment works because the forward path to the output comes into alignment with the
feedback matrix; a frozen readout cannot. Same task, same rule, same projection, 20,000 trials:

| readout | rate 1e-4 | rate 1e-3 | rate 1e-2 | seeds above floor |
|---|---|---|---|---|
| **frozen** (`eprop_random`) | −0.8147 | −0.7097 | −0.7031 | 4/8, 6/8, 6/8 — fails at every rate |
| **plastic** (diagnostic, seeds 101–108) | −0.2726 | **−0.1353** | −0.1391 | 8/8 at every rate |

Floor −0.6909, optimum −0.1353. With the readout plastic the broadcast arm reaches the optimum
**exactly**. So `random` — the arm the plausibility claim rests on — is structurally unable to work in
the configuration all four share, and `plastic_readout` is the one arm stage 1 says can.

**The exclusion this change first registered was not supported by the evidence cited for it.** Logbook
040 measured a plastic readout collapsing under the **Hebbian** rule, where a plastic output layer's
post-synaptic factor is its own *output*, so its rows self-amplify toward whatever maximises the action
mean (density 1e17). Under e-prop the factor is its own *error*, and there is no such loop. The
readout's post-synaptic units **are** the action dimensions, so its learning signal is the identity and
its eligibility is

```text
E[k, c] <- decay * E[k, c] + score_k * pooled_c
```

the **exact gradient** of the action log-probability with respect to the readout — no projection, no
truncation, no dropped paths. It is the one place in this mechanism where nothing is approximated, and
it is pinned against autograd by test.

**The readout is excluded from the homeostatic rescale**, which returns each unit's incoming norm to
construction: its scale is part of what this arm asks about, since R.1d measured **+4.51 foods from
scale alone** with the direction held. The rule's weight bound still applies at 3.0 per entry, allowing
a readout norm of 8.49 against the **7.820** R.1d's PPO harvest reached, so it does not bind on the
scale that mattered.

## The control the pilot made necessary

The pilot on **disjoint seeds 101–104**, 12/12 runs clean, at the committed rate:

| arm | readout | learning | frozen | full clear % |
|---|---|---|---|---|
| `plastic_readout` | plastic | **15.550** | 1.829 | **49.13%** |
| `random` | frozen | 3.730 | 1.829 | 0.00% |

`plastic_readout` is the first arm in this programme to reach competence on the connectome, and
`random` sits inside R.1c's 2.4–4.4 band exactly as stage 1 predicted. The learned readout landed at
norm **3.991**, cosine **−0.208** to the anatomical default — between anatomical 1.414 and PPO's
7.820, and close to PPO's direction (−0.178).

**That result must not ship with its alternative explanation untested.** A plastic readout is an
**8-parameter linear map** over four pooled motor-class means, so "a local rule learns this substrate"
and "a small linear readout on frozen recurrent features learns this cell" predict the same success.
`readout_only` freezes `w_chem` and leaves the readout learning, so
**`plastic_readout − readout_only` is what the substrate's own plasticity contributes.** Nothing on
the record supplies it: R.1d's frozen arms froze everything. Its credited drift should read ~0, which
is the check that the withholding happened.

**The pilot's verdict is withheld, not reported.** At four seeds the exact one-sided paired test
cannot reach the significance gate at all — its smallest reachable p is 2⁻⁴ = 0.0625, which BH across
the arms pushes above it — so every arm reads `no_improvement` whatever it did. The harness printed
`does_not_learn — the programme stops` on that evidence before this was caught and fixed.

**Where the readout ends up is reported**, norm and cosine to the anatomical default, against R.1d's
two measured readouts — anatomical 1.414, PPO 7.820 at cosine −0.178 — so "e-prop rediscovers something
like PPO's decoding" and "it finds something else" are separable.

`random_motor` is what makes `symmetric` vs `random` readable: without it, a `symmetric` win reads as
"the true direction helped" and "crediting only the motor pool helped" at once, and R.1c already
measured the second. `scalar` is an **ablation, not a candidate** — reward-modulated Hebbian with an
activation derivative in place of the post-synaptic rate. If it matches `random`, the per-unit signal
did nothing and the result is about the dynamics-derived trace alone.

## The operating point, and why the rate carries over

R.1c's and R.1d's, unchanged: the `hard350` cell, `forward_pass_depth: 4`, `initial_log_std: -1.0`,
`plasticity_rate: 0.001`, `trace_decay: 0.9`, `plasticity_normalise_modulator`,
`plasticity_normalise_trace` and `plasticity_homeostasis` all on, the **committed anatomical readout**,
and `plasticity_node_noise: 0.0` with no perturbation set.

e-prop's raw trace magnitude is nothing like `h ⊗ ξ` at σ 0.1. **`plasticity_normalise_trace` is what
makes the rate transfer**: it divides the Hebbian term by a running RMS of the trace over its edge
set, so `plasticity_rate` is a root-mean-square step per unit modulator rather than an absolute step.
A **rate check on the pilot's disjoint seeds** was registered against exactly that — the committed
rate and one decade either side — with the committed rate standing unless it is visibly off, the
discipline R.1c's σ calibration established after a carried-over value cost 31.5% of the arm's level.
**It was not run as a separate sweep**: the pilot at the committed rate reached 15.550 foods of 20 at
49.13% full clear, which is not a rate visibly off, and stage 1 had already swept the same grid on the
same mechanism — the true-gradient arm passing at all three rates and the ablation at none. Recorded
as a deviation from the registered task, not as a completed one.

**Not R.1d's `anatomical_scaled` readout.** Its norm came from a PPO harvest, so an arm using it
inherits R.1d's "cannot satisfy the plausibility deliverable" flag — and having a rule that *can*
satisfy it is the whole point of e-prop. The anatomical arm is also the one with a matched
node-perturbation comparator on the record: **3.751** learning, **3.150** frozen.

## One frozen floor, and why it is not R.1c's

Four learning arms share **one** frozen control: with updates frozen no weight moves, so no routing
can reach the behaviour, and the projections draw from their own generator. Asserted by test, not
argued.

It is **not** R.1c's frozen arm. That ran at `plasticity_node_noise: 0.1`, and the perturbation enters
the forward pass whether or not updates are frozen, so its floor is a *noisier* policy than this one.
The 16 runs are spent rather than saved.

## Stage 1 is a stop clause, not a task

On the committed one-step contextual association there is one plastic layer, one forward pass and one
scalar Gaussian action, so `eps_ij = psi_j · h_i` and with `B = W_out^T` the update **is** the
REINFORCE gradient of that layer.

| arm | required | what a failure means |
|---|---|---|
| `eprop_symmetric` | **passes** | the implementation is wrong — the derivative, the score function or the fold-in. Nothing downstream is interpretable |
| `eprop_random` | reported, required of nothing | feedback alignment does or does not align on a 4-cue association; either way it bounds what stage 2 could show |
| `eprop_scalar` | **must not pass** | a rule with no per-unit signal solving a cue-to-target association means the task does not discriminate, and the control is VOID |

**No stage-2 run is launched until both hold**, and the scoring harness reads that control's own JSON
record rather than taking it on trust.

## The reading

Plateau-tail mean foods through I.2's graded family, each arm against the shared frozen control,
paired, one-sided, **BH-FDR across the six**. Both minima: **1.0 foods** of 20, and **10% of the
reachable gap** against PPO's matched **18.945**, the more demanding binding.

**Reported separately**: whether an arm **beats its floor**, and whether it **reaches competence** (20%
full clear). Only the second makes block V's time-to-competence contrast defined, and only the second
bears on R.1b.

**Credited-synapse drift** per arm — credited being the units the signal can reach — against R.1c's
**1.37–1.38×** and R.1d's **1.38–1.42×**. A third structural axis holding the same number is the
finding; a different number is a bigger one.

**Where the change lands, by hop distance to the motor pool.** e-prop drops the multi-hop terms, so a
learning arm's change should concentrate near the pool. `random` is where the prediction is testable —
it is the only arm whose signal reaches the far units at all. Registered before the run.

**The four matched contrasts**, descriptive and reported in every branch:
`symmetric − random_motor` isolates the direction at matched reach; `random − random_motor` isolates
reach at a matched source; `plastic_readout − random` isolates the readout at a matched signal and
matched reach; and `plastic_readout − readout_only` isolates the substrate's own plasticity at a
matched readout — the one a positive result needs to mean what it claims.

## Outcomes — [Logbook 059](../../059-7a-shipment.md)'s three, with its third split

| verdict | test | what follows |
|---|---|---|
| `does_not_learn` | no arm beats its floor by the registered minima | the programme **stops**; 7b proceeds under PPO after the power arithmetic; the plausibility claim is given up. The rule family has now failed with two independent eligibilities |
| `learns_below_competence` | an arm beats its floor; none reaches 20% full clear | a **result**. R.1b stays blocked and 7b's gate is untouched; the readout-scale follow-up becomes the live question |
| `learns_the_cell` | an arm beats its floor, reaches competence **and clears the readout-only control by the registered 1.0-food minimum** | the wiring contrast becomes runnable, registered fresh in its own change; B.5, B.1, B.4 and B.4b become askable |
| `learns_without_the_substrate` | an arm beats its floor and reaches competence; **none does so while writing the substrate** | 059's gate is met in **letter and not in substance**. Every substrate rung asks its question of a rule that writes the wiring, so **B.5, B.1, B.4 and B.4b stay gated**. R.1b becomes runnable in a changed form: wild type against its rewired null as **frozen features** under the readout-only arm |
| `void` | stage 1's required-pass arm failed, or its required-fail arm passed | nothing here is interpretable and **no connectome run is spent** |

**The split of 059's third outcome is POST HOC, added 2026-09-15 with the campaign's results already
in hand.** What is not post hoc is the control that forced it: `readout_only` was registered, built and
its purpose stated — separating "a local rule learns this substrate" from "a small linear readout on
frozen recurrent features learns this cell" — **before any `readout_only` run existed**, with the sign
unknown. The condition also uses the registered absolute minimum rather than a new threshold: the
substrate's contribution must clear **1.0 food** over the control, the same bar every arm's effect is
held to. The ordering, plainly: the arms ran, `plastic_readout − readout_only` came back at **−3.895
foods**, and only then was the outcome split. That is the failure mode R.1c caught in itself — a
verdict condition weaker than the consequence attached to it — surfaced here *after* the campaign
rather than before it, which is worse, and is the reason the control existed at all.

059's **three-active-week implementation bound** applies: the case for this programme over running 7b
under PPO is that it is cheaper. **Implementation started 2026-09-14**, so the bound falls on
**2026-10-05**.

### What this cannot be, whatever it returns

Not a result about e-prop in general (one substrate, one cell, one truncation, 16 seeds, `w_chem`
alone through a readout R.1d already showed to be part of the limit); not a result about feedback
alignment (one `B` per seed at one scale); not evidence the dynamics-derived eligibility beats
perturbation unless `scalar` separates from the floor; and not a result about the true gradient
direction unless `symmetric` separates from `random_motor`.

## Honest prior, revised by stage 1

**Before stage 1** this read `does_not_learn`, with `learns_below_competence` a real possibility: the
truncation is worst exactly where this substrate needs it most — 263 of 302 units reach the action
only through the paths it drops — and R.1d showed the readout scale already binds around 4 foods at
this operating point.

**Stage 1 moved it, and only for one arm.** The four frozen-readout arms are now *expected* to fail,
`random` for a structural reason that is measured rather than argued. `plastic_readout` is the live
question, and it is genuinely open: it reached the optimum on a one-step task with one plastic layer
and a 1-D action, which says the mechanism is sound and says nothing about 350 steps, 302 recurrent
units and a 2-D action. What pulls for it is that it removes the two limits this programme has
measured at once — R.1d's readout ceiling and the 1/N perturbation variance R.1 found decisive. What
pulls against it is the truncation, which is untouched: the far units are still credited only through
a projection that is not the gradient.

If `scalar` sits at the floor and `plastic_readout` does not, that is the cleanest evidence yet that
what the rule has been missing is a per-unit signal with somewhere to align.

## Reproduce

```bash
P=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop

# 1. stage 1 — the stop clause. VOID here means no connectome run is spent.
uv run python scripts/analysis/l4_rule_positive_control.py \
  --out docs/experiments/logbooks/supporting/063-l4-eprop/stage1-control.json \
  --csv docs/experiments/logbooks/supporting/063-l4-eprop/stage1-per-seed.csv

# 1b. the alignment diagnostic on DISJOINT seeds: is a frozen readout what stops `random` working
uv run python scripts/analysis/l4_rule_positive_control.py \
  --diagnostic-out docs/experiments/logbooks/supporting/063-l4-eprop/stage1-diagnostic.json

# 2. pilot on DISJOINT seeds — the 12 runs this record reports. At the committed rate; no separate
#    rate-check sweep was run (see above).
uv run python scripts/run_campaign.py \
  --config ${P}_plastic_readout.yml --config ${P}_random.yml --config ${P}_frozen.yml \
  --seeds 101-104 --runs 3000 --output-dir campaigns/eprop-pilot \
  -- --theme headless --track-experiment

# 3. the arms — 112 runs. `--track-experiment` is REQUIRED for the drift and hop columns.
uv run python scripts/run_campaign.py \
  $(for R in symmetric random_motor random scalar plastic_readout readout_only; do
      printf -- "--config %s_%s.yml " "$P" "$R"
    done) \
  --config ${P}_frozen.yml \
  --seeds 1-16 --runs 3000 --output-dir campaigns/eprop \
  -- --theme headless --track-experiment

# 4. score, with stage 1 as the gate
uv run python scripts/analysis/l4_eprop.py \
  --campaign campaigns/eprop --seeds 1-16 \
  --stage-one docs/experiments/logbooks/supporting/063-l4-eprop/stage1-control.json \
  --out docs/experiments/logbooks/supporting/063-l4-eprop/eprop.json \
  --csv docs/experiments/logbooks/supporting/063-l4-eprop/per-seed.csv
```
