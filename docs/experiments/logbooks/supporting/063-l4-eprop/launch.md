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

| arm | signal source | units it can reach | what it answers |
|---|---|---|---|
| `symmetric` | the readout's transpose | the 39-unit pool — **forced** | does the true direction help, where it can reach |
| `random_motor` | fixed random `B`, masked to the pool | the same 39 | is it the direction, or is it the pool |
| `random` | fixed random `B` | all 302 | can a broadcast projection credit the far units usefully |
| `scalar` | none, `L_j = 1` | all 302 | did the per-unit signal do anything at all |

The fourth cell of that 2×2 — **true directions reaching all 302 units — is one the mechanism
forbids**, and the harness records it as forbidden rather than omitting it.

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
A **rate check on the pilot's disjoint seeds** is registered against exactly that — the committed rate
and one decade either side — with the committed rate standing unless it is visibly off. This is the
discipline R.1c's σ calibration established after a carried-over value cost 31.5% of the arm's level.

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
paired, one-sided, **BH-FDR across the four**. Both minima: **1.0 foods** of 20, and **10% of the
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

**The two matched contrasts**, descriptive and reported in every branch:
`symmetric − random_motor` isolates the direction at matched reach; `random − random_motor` isolates
reach at a matched source.

## Outcomes — [Logbook 059](../../059-7a-shipment.md)'s three, unchanged

| verdict | test | what follows |
|---|---|---|
| `does_not_learn` | no arm beats its floor by the registered minima | the programme **stops**; 7b proceeds under PPO after the power arithmetic; the plausibility claim is given up. The rule family has now failed with two independent eligibilities |
| `learns_below_competence` | an arm beats its floor; none reaches 20% full clear | a **result**. R.1b stays blocked and 7b's gate is untouched; the readout-scale follow-up becomes the live question |
| `learns_the_cell` | an arm beats its floor **and** reaches competence | the wiring contrast becomes runnable, registered fresh in its own change; B.5, B.1, B.4 and B.4b become askable |
| `void` | stage 1's required-pass arm failed, or its required-fail arm passed | nothing here is interpretable and **no connectome run is spent** |

059's **three-active-week implementation bound** applies: the case for this programme over running 7b
under PPO is that it is cheaper. **Implementation started 2026-09-14**, so the bound falls on
**2026-10-05**.

### What this cannot be, whatever it returns

Not a result about e-prop in general (one substrate, one cell, one truncation, 16 seeds, `w_chem`
alone through a readout R.1d already showed to be part of the limit); not a result about feedback
alignment (one `B` per seed at one scale); not evidence the dynamics-derived eligibility beats
perturbation unless `scalar` separates from the floor; and not a result about the true gradient
direction unless `symmetric` separates from `random_motor`.

## Honest prior

**`does_not_learn`, with `learns_below_competence` a real possibility.** Two things pull against
e-prop here. The truncation is worst exactly where this substrate needs it most — 263 of 302 units
reach the action only through the paths it drops — and R.1d showed the readout scale already binds
around 4 foods at this operating point. What pulls for it: e-prop removes the 1/N variance that R.1
measured as decisive, and `random` is the first mechanism in this programme that can credit a far unit
with a *consistent* direction rather than a fresh draw each step.

If `scalar` is indistinguishable from the floor and `random` is not, that is the cleanest evidence yet
that the per-unit signal — not the trace — is what the rule has been missing.

## Reproduce

```bash
P=configs/scenarios/foraging/connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_eprop

# 1. stage 1 — the stop clause. VOID here means no connectome run is spent.
uv run python scripts/analysis/l4_rule_positive_control.py \
  --out docs/experiments/logbooks/supporting/063-l4-eprop/stage1-control.json \
  --csv docs/experiments/logbooks/supporting/063-l4-eprop/stage1-per-seed.csv

# 2. pilot on DISJOINT seeds, plus the rate check
uv run python scripts/run_campaign.py \
  --config ${P}_random.yml --config ${P}_frozen.yml \
  --seeds 101-104 --runs 3000 --output-dir campaigns/eprop-pilot \
  -- --theme headless --track-experiment

# 3. the arms — 80 runs. `--track-experiment` is REQUIRED for the drift and hop columns.
uv run python scripts/run_campaign.py \
  $(for R in symmetric random_motor random scalar; do printf -- "--config %s_%s.yml " "$P" "$R"; done) \
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
