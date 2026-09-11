# Design: examining the knobs, and building a control that can hold the third

## Why the registered platform fails, measured rather than argued

I.3 said "each examined on the yardstick under I.1's rule". Two facts, both checked before this was
written:

**The yardstick does not learn under that rule.** A pilot on disjoint seeds 101–102 at the panels'
3000-episode budget:

| arm | full-clear % | mean foods |
|---|---|---|
| node-perturbation, seed 101 | 0.00 | 0.11 |
| node-perturbation, seed 102 | 0.00 | 0.00 |
| frozen control, seed 101 | 0.00 | **0.90** |
| frozen control, seed 102 | 0.00 | **0.35** |

The learning arm sits **below its own frozen control**, and seed 101 reached 2.70 foods by the
second sixth of its run before collapsing to 0.11 — it found something and took it apart. Two seeds
is a pilot, not a result, and it is recorded as one; it agrees with Logbook 040, which found the
same under the original rule and explained it structurally.

**The horizon is unmeasurable on the one-step control.** `trace_decay` at 0.0, 0.9 and 0.99 gives
scores and alignments identical to four decimals, because the control resets the trace every trial.
The harness says why in its own comment: each trial is its own episode, "the horizon confound this
control removes". Removing that confound was right for I.0's question and makes the control useless
for this one.

## The split that follows

Two knobs are measurable where the rule works, and one is not. So:

| knob | platform | why |
|---|---|---|
| homeostatic norm sphere | the existing control | it acts on every update, one step or many |
| exploration noise | the existing control | it sets the action distribution the reward scores |
| eligibility horizon | **a delayed control, built here** | it needs more than one step to act at all |

## The delayed control

The task gains a single parameter, the delay `D`. A trial runs: the cue is presented, the scored
action is taken, then `D` further steps elapse against a neutral observation, and only then does the
reward arrive. The eligibility for the scored action decays across those steps, so by the time the
modulator reaches it, it carries roughly `trace_decay ** D` of what it had.

**What is deliberately *not* changed:**

- **The bounds.** The cue-blind floor is `−Var[t] − σ²` and the optimum `−σ²`; both depend on the
  targets and the action noise and on nothing else. Delaying the reward does not move either, so
  the floor, the optimum, the gap and the registered pass rule carry over unchanged and a delayed
  arm is directly comparable with the committed one.
- **The scored action.** One action per trial, taken while the cue is visible, scored the same way.
  So this is a test of *credit over time*, not of memory: the network never has to hold the cue,
  which a feedforward stack could not do anyway. Choosing the memory version instead would confound
  the horizon with a capability the substrate lacks, and the result would say nothing about
  `trace_decay`.
- **`D = 0`.** It must reproduce the committed one-step control exactly. That is the regression
  anchor: an extension that changes the numbers at zero delay has changed the instrument, not
  extended it.

The intervening steps see a neutral observation rather than the cue repeated. Repeating the cue
would let the network act correctly on a fresh trace at the final step, which is not the question;
the point is that the action being credited is `D` steps in the past.

## The grids

| arm | grid | baseline |
|---|---|---|
| homeostasis | on, off | on |
| exploration noise | the pinned value and a grid around it | `exp(−1.0)` |
| horizon | `trace_decay` × `D` | `0.9`, `D = 0` |

The delays bracket the ten-step scale `trace_decay 0.9` implies — a trace at 0.9 retains 0.35 over
ten steps and 0.12 over twenty — so the grid runs through and past where the setting predicts the
signal dies. The registered pass rule is the control's own, unchanged, applied per cell.

## What the outcomes mean, fixed before the run

- **A knob passes where the baseline fails, or fails where the baseline passes.** That setting was
  holding the rule back or propping it up, and the panels that pinned it are re-read in that light
  by I.4.
- **The horizon degrades with `D` as `trace_decay` predicts.** The rule's success on a one-step task
  and failure on every multi-step one has a mechanical explanation, and the horizon becomes the next
  registered target rather than a suspicion.
- **The horizon does not degrade with `D`.** The trace is not what separates the one-step success
  from the multi-step failures, and the suspicion is closed — which is worth as much as confirming
  it, and is why the delayed arm is registered with both outcomes named.
- **Nothing moves.** The three settings are exonerated on the platform where the rule works, and
  I.4 records that the failures are not attributable to them.

## Alternatives considered

- **Fix the yardstick first** so I.3 can run as registered. That is a research programme — 040
  located the cause in local rules on dense stacks needing decorrelation — and it is not what I.3
  asked for.
- **Test the horizon on the connectome clone assay.** It is multi-step, but the rule fails there for
  reasons the assay cannot separate, and each campaign is hours. The point of a control is to ask
  one question with nothing else able to answer for it.
- **Make the delayed control a memory task**, with the cue hidden during the delay. Rejected above:
  it confounds the horizon with a capability the substrate does not have.
- **Defer the delayed control to its own change.** Tempting, since it is the larger half. Rejected
  because I.3's whole content is the three knobs, and leaving the one that matters most unexamined
  would close the item having answered the two cheap parts.
