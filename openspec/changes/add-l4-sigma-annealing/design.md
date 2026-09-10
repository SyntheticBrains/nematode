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

`σ_final = 0` is permitted and means the arm stops exploring entirely at `E`; the eligibility is
then identically zero and the rule writes nothing further, which is a frozen policy by a different
route and is a legitimate endpoint to register. The load-time refusal of a zero σ applies to the
*initial* scale only.

## Where it advances

`prepare_episode()` is called by the runner at the start of every episode and already resets the
traces and the rule's per-episode state. The counter advances there, in the brain, and the
topologies read the current σ rather than a construction-time constant.

Two consequences to state rather than discover:

- The counter advances on **every** `prepare_episode`, including the ones the transgenerational
  probe path issues between probes. No arm registered here uses that path; an arm that did would be
  annealing on probe episodes as well as training ones, and must not read this schedule as
  episode-indexed training time.
- On a **warm start** the counter begins at zero, so a loaded competent policy is annealed from
  σ_0. That is the intended reading for the clone assay: the arm explores at the σ that learns and
  decays from there, which is exactly the question. It is *not* transparent to a checkpoint that
  was saved mid-anneal; the counter is transient state, cleared per load, never persisted.

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
  roughly fixed step size. This is the registered regime for both gates.
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
   floor-to-optimum gap — with its alignment reported at both ends of the schedule. *Licenses:* the
   schedule is still a learner. A schedule that anneals away its own signal fails here, and that is
   a cheap and decisive place to find out.
2. **The clone assay, with a frozen control beside it.** The same registered screening arm,
   comparator, budget, metric and pass rule, plus a frozen-perturbation control running the
   *identical schedule* with `freeze_updates: true`. The control is not optional and its meaning is
   sharper than in I.1: under a schedule the frozen arm's score *recovers as σ falls*, so it traces
   the damage the schedule alone does over its whole path. The learning arm is read against that
   trace, not against a single number. *Licenses:* nothing beyond itself — the panel stays gated on
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
