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
reward arrive.

**What the delay measures is dilution, not decay, and the distinction decides whether the control
works at all.** The control runs the recipe's `normalise_trace`, which divides each tensor's trace
by a running RMS of its own magnitude. If the intervening steps added nothing to the trace, the
trace at reward time would be exactly `trace_decay ** D` times the credited step's, a scalar on a
fixed vector — and the normalisation would divide that scalar out. Checked against the rule's own
running scale: at `D = 20` the raw trace is 0.122 of its `D = 0` value and the normalised trace the
update uses is **1.000, identical to `D = 0`**. A zero-input delay would report the horizon
exonerated at every `D`, and the null would be the instrument's.

In a real episode the credited step is not merely decayed; it is **diluted** by the terms every
later step adds. Normalisation rescales the whole sum and leaves the credited step's *share* of it
alone, so dilution survives normalisation where decay does not. The intervening steps must
therefore drive the plastic layer. Under `plastic_layers: hidden` that layer's pre-synaptic input
is the observation itself, so the neutral observation has to be nonzero.

**The neutral observation is the uniform vector over the cue channels** — every channel at
`1 / n_cues`, identical on every trial, so it carries nothing about which cue was shown. It keeps
the observation's dimension, which is what lets `D = 0` stay bit-identical: an extra "no cue"
channel would change the actor's input width and its initialisation, and the anchor would be lost
before the delay did anything. The trace at reward is then `decay^D · E₀` plus the filler's
zero-mean terms, and the credited step's share of it falls with `D` as the real task imposes.

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

The intervening steps see the neutral observation rather than the cue repeated. Repeating the cue
would let the network act correctly on a fresh trace at the final step, which is not the question;
the point is that the action being credited is `D` steps in the past.

**Alignment at `D > 0` is measured against the scored step's gradient.** The harness compares each
update with the analytic gradient of the loss at the step the update happens on. On a delayed trial
the update lands at step `D`, where the network's output is a response to the neutral observation
and its "loss" against the target means nothing. The gradient is therefore taken at the scored
step and held until the reward step, and the update is compared with that.

## The grids

| arm | grid | baseline |
|---|---|---|
| homeostasis | on, off | on |
| exploration noise | std ∈ {0.22, 0.37, 0.61, 1.0} | 0.37 (`exp(−1.0)`) |
| horizon | `trace_decay` ∈ {0.9, 0.99, 0.999} × `D` ∈ {0, 2, 5, 10, 20} | `0.9`, `D = 0` |

The noise grid is the panels' own history: std 1.0 capped every plastic arm near its floor, 0.22
rose and collapsed, and 0.37 was selected by a probe — the grid brackets the selected value with the
two that failed. The delays bracket the ten-step scale `trace_decay 0.9` implies — a trace at 0.9
retains 0.35 over ten steps and 0.12 over twenty — so the grid runs through and past where the
setting predicts the signal dies, and the two longer decays ask whether a longer horizon recovers
it. The registered pass rule is the control's own, unchanged, applied per cell.

**Cost.** A trial at delay `D` is `D + 1` forward passes. The committed control runs 72 arms of
20,000 trials in about ten minutes; the horizon grid is 15 cells × 8 seeds with a mean delay near
seven, so roughly 1–2 hours in all. The two cheap knobs add 6 cells × 8 seeds at one step each.

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
