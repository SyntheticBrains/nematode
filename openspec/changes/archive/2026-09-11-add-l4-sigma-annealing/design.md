# Design: annealing the perturbation scale

## The tension, stated precisely

Node perturbation needs σ to do two incompatible jobs at once.

As an *estimator*, σ is the probe. The rule learns from the covariance between the reward-prediction
error and the perturbation a unit happened to receive, so a σ too small to move behaviour produces
an eligibility indistinguishable from noise: the control's σ = 0.01 arm sits at −0.777 with an
alignment of +0.027, barely above the unrepaired rule.

As a *policy*, σ is damage. Every settling step adds `ξ ~ N(0, σ²)` to a pre-activation whose
activation is bounded in (−1, 1), and the frozen control measures what that costs a competent
policy directly: 38.7 → 8.9 with no weight ever written.

The control's dose-response shows both effects along one axis, monotone and in opposite directions:

| σ | control score | alignment | seeds above floor |
|---|---|---|---|
| 0.01 | −0.777 | +0.027 | — |
| 0.05 | −0.477 | +0.118 | 7 of 8 |
| 0.20 | −0.196 | +0.263 | 8 of 8 |

Nothing in the estimator requires one σ to serve both. Learning needs a large σ *while learning*;
running the resulting policy needs a small one *afterwards*. A schedule is the minimal intervention
that separates them, and it adds no term to the update — only a number that was already there
becomes a function of episode index.

## The schedule

One shape, fixed here:

```text
σ(e) = σ_final                                   if e ≥ E
σ(e) = σ_0 · (σ_final / σ_0) ** (e / E)          otherwise
```

with `σ_0 = plasticity_node_noise`, `σ_final = plasticity_node_noise_final` and
`E = plasticity_node_noise_anneal_episodes`, `e` the count of episodes begun.

Geometric rather than linear because σ's effect on both jobs is closer to multiplicative than
additive — the dose-response above moves by roughly equal steps in score for equal *ratios* of σ,
not equal differences — and because a geometric path spends more of its length near the small
values, where retention is decided, than a linear one does. It reaches σ_final exactly at `E` and
is constant after, so the arm's endpoint is a stated σ rather than whatever the schedule happened
to reach.

**Registered values.** σ₀ = 0.2, σ_final = 0.02, E = half the budget: 1,000 of the assay's
2,000 episodes and 10,000 of the control's 20,000 trials. σ₀ is the scale that passed the control.
σ_final sits an order of magnitude below it — under the grid's 0.05, which still cost 7 of 8 seeds
the learning bar, and above 0.01, where the estimator was inert — so the floor is a scale at which
the perturbation should be nearly harmless and the estimator nearly silent. E = half so that the
final quarter, which the plateau-tail metric, the control's score window and the trajectory
annotation all read, sits entirely at σ_final rather than partway down the decay.

**σ_final must be positive.** Both substrates decide *at forward time* whether they are
perturbing by testing the scale against zero, and when they are not, the trace falls back to the
activity: `post_factor = self.node_perturbation if self.node_noise > 0.0 else h`. A schedule that
reached zero would therefore not silence the eligibility — it would silently turn it back into the
Hebbian `pre ⊗ post` the whole sequence exists to replace. Rather than re-gate two forward passes
on the mode for a case no arm needs, a zero floor is refused at load, and the existing refusal of a
zero *initial* scale stands.

## Where it advances

The counter lives on the **topology**, not the brain, and advances through an explicit seam
method — call it `advance_schedule()` — that the brain's `prepare_episode()` calls beside
`reset_traces()`. Two facts force that placement:

- The positive control has no brain. It builds `MLPTopology` directly and drives the rule over the
  seam, calling `reset_traces()` once per trial. A counter in the brain would never advance there,
  and the annealed control arm would run at σ₀ throughout, pass, and license the assay on a false
  gate. With the counter on the topology, the harness advances it once per trial: **the control's
  trial is its schedule step**, and E = 10,000 counts trials.
- `reset_traces()` is not only the per-episode hook. The rule's load-time reset calls it too, and a
  load must *restart* the schedule, not advance it. So the counter does not ride on
  `reset_traces()`; it advances only where an episode (or trial) actually begins, and a policy load
  sets it back to zero.

The counter is a plain integer on the topology. It is not a buffer, never enters `state_dict`, and
needs no transient-buffer bookkeeping: a checkpoint written before this change loads unchanged.

Two consequences to state rather than discover:

- The counter advances on **every** `prepare_episode`, including the ones the transgenerational
  probe path issues between probes. No arm registered here uses that path; an arm that did would be
  annealing on probe episodes as well as training ones, and must not read this schedule as
  episode-indexed training time.
- On a **warm start** the counter begins at zero, so a loaded competent policy is annealed from
  σ₀. That is the intended reading for the clone assay: the arm explores at the σ that learns and
  decays from there, which is exactly the question. A checkpoint saved mid-anneal restarts the
  schedule on load rather than resuming it.

## The scale coupling, and why it is registered rather than fixed

The perturbation is drawn as `randn · σ` and the trace accumulates `pre ⊗ ξ`, so the update's
magnitude is **linear in σ**. The textbook unbiased node-perturbation estimator divides by `σ²`
(Williams 1992; Fiete & Seung 2006), which this implementation does not do — it has never needed
to, because σ was a constant folded into the rate.

Under a schedule that stops being true: decaying σ by a factor of four also decays the step by a
factor of four, so an annealed arm changes its effective learning rate as a side effect of changing
its exploration. That is a confound, and there are two honest regimes:

- **`plasticity_normalise_trace: true`** — every panel arm and both I.1 arms enable this. Each
  tensor's Hebbian term is divided by a running RMS of its own trace, and since the trace scales
  with σ, so does the RMS: most of the coupling cancels, and the arm anneals exploration at a
  roughly fixed step size. The RMS is an EMA with `scale_rate` 0.01, a time constant of about a
  hundred updates, so it lags a decaying σ by that much — negligible against a decay spread over
  a thousand episodes or ten thousand trials, and stated so the residual is known rather than
  discovered. This is the registered regime for both gates.
- **`plasticity_normalise_trace: false`** — the coupling is live and the anneal is a joint decay of
  exploration and rate. Permitted, but the arm must declare it, and a result from such an arm is
  not comparable with a normalised one.

The change deliberately does **not** add a `1/σ²` compensation. That would be a second mechanism —
a different estimator — landing in the same run as the schedule, and a pass could not be attributed
to either. If the schedule fails in the normalised regime, the compensated estimator is the next
registered variant, not a mid-flight repair.

## The gates, in order, and what each licenses

1. **The positive control, under the schedule.** The annealed arm runs the same one-step contextual
   association with the same three validity arms, over the same seeds, with the anneal compressed
   to the control's episode budget. It must clear the registered bar — 7 of 8 seeds and half the
   floor-to-optimum gap. The score window is the last 1,000 trials, which under E = 10,000 sits
   entirely at σ_final, so a pass means the policy the schedule *left behind* is competent under the
   perturbation it will actually run with. Alignment is reported separately over the decay (the
   first E trials) and the floor (the rest). At the floor the estimator is nearly silent by
   construction, so a low floor-phase alignment is expected of a good schedule and is not the
   failure signature; the signature of a schedule that anneals away its own signal is a **decay-phase
   alignment that does not rise with the constant-σ arm's and a floor-phase score below the bar**.
   *Licenses:* the schedule is still a learner, and that is a cheap and decisive place to find out.
2. **The clone assay, with a frozen control beside it.** The same registered screening arm,
   comparator, budget, metric and pass rule, plus a frozen-perturbation control running the
   *identical schedule* with `freeze_updates: true`. The control is not optional and its meaning is
   sharper than in I.1: under a schedule the frozen arm's score *recovers as σ falls*, so it traces
   the damage the schedule alone does over its whole path. Both arms' curves are binned per 250
   episodes — eight bins, the first four spanning the decay and the last four at the floor — with
   the bin's σ(e) stated beside it, and the learning arm is read against the frozen arm bin by bin,
   not against a single number. *Licenses:* nothing beyond itself — the panel stays gated on
   I.2 regardless of the outcome.

Gate 1 before gate 2, and a failure at gate 1 stops there.

## What the outcomes mean

- **Passes both.** The tension I.1 exposed is an artefact of holding σ constant, and the variant is
  the first mechanism in this sequence to both learn and retain. This does not license a panel; it
  licenses the I.4 re-read to treat the rule as repairable, and it makes I.2's statistic the thing
  standing between here and a panel arm.
- **Passes the control, fails the assay.** The schedule learns but still cannot hold a policy. The
  frozen control decides the reading: if the frozen arm recovers and the learning arm does not, the
  updates are doing the damage and consolidation-on-top-of-eligibility becomes the next mechanism
  with a real motive. If neither recovers, the damage done at high σ is not repaired by lowering σ
  later, and the schedule's *shape* — not the rule — is what failed.
- **Fails the control.** The anneal removes the signal before the policy is learned. Registered
  response: report it, and read it as a constraint on `E` relative to the task's horizon rather
  than as a refutation of annealing.

## Alternatives considered

- **Compensate the rate by `1/σ²`.** The textbook estimator, and the principled fix to the coupling.
  Rejected *here* because it changes the estimator and the schedule in one step; queued as the
  successor variant if the normalised regime fails.
- **Anneal on performance rather than episode index** — decay σ once a trailing success rate
  plateaus. Rejected: that is the oracle gate's mistake in a new place. It reads a signal the rule
  does not have, and the consolidation screen already recorded what an oracle-gated brake licenses,
  which is a diagnostic bound rather than a mechanism.
- **Per-unit adaptive σ.** More parameters, no prior result to pin them, and it confounds the
  schedule with a second mechanism. Not now.
- **Anneal the trace horizon or the action noise instead.** Both are I.3's knobs, examined there
  against the values the panels pinned, and neither addresses the perturbation that the frozen
  control shows is doing the damage.
