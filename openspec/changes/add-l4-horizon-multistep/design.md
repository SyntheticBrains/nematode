# Design: the horizon where it would matter

## The gap this closes

| | one-step control (I.3) | multi-step tasks (040–052) |
|---|---|---|
| `trace_decay` | swept {0.9, 0.99, 0.999} | **never set — all at the 0.9 default** |
| steps per episode | 1, plus a delay of 0–20 | 244–2400 |
| result at 0.9 | 89% undelayed, **−6.8% at D = 20** | every arm at or below its floor |
| result at 0.99 | **45.3% at D = 20** | never run |

The rule's success is on the row where the horizon cannot bite, and its failures are all on the row
where I.3 says it should bite hardest — at a setting nothing ever varied. That is either the
explanation of the phase or a coincidence, and one campaign separates them.

## Why the yardstick and not the connectome

The yardstick runs 3000 episodes in about eight minutes; a connectome clone run takes about ten
hours. The yardstick is also where the claim is cleanest: it is a dense feedforward stack with a
frozen readout, so there is no wiring hypothesis in play and no clone to destroy — the only question
is whether the rule learns anything at all across a long episode. Under I.1's rule at the pinned
horizon it ends **below its own frozen control**, which is an unambiguous thing to move.

A connectome arm is registered only if the yardstick moves. Spending ten-hour runs on a lever with
no multi-step evidence is the pattern this phase has been correcting.

## The metric, and why the cliff will not do

Every yardstick arm ever run sits at the full-clear floor: the committed 040 values mean **1.05%**
with **no seed competent** by the 20% threshold. A metric that is zero for both arms cannot separate
them, and the competent-fraction and level contrasts are undefined where no seed is competent.

The graded reading — plateau-tail mean foods, out of ten — is the one that can see a difference, and
this is exactly the case I.2 registered it for: *"an arm whose full-clear rate is at its floor while
its graded measure exceeds the comparator's"*. This change is its first use.

## The comparison

**Learning arm against its own frozen control, at the same horizon, paired by seed.** Not against
the committed 040 yardstick values, which ran under the original rule and would confound the
eligibility change with the horizon change. Not against a single frozen baseline, because the
perturbation's cost to a policy is not constant across horizons — a longer trace changes what the
rule writes, so each horizon needs its own no-writing control.

One-sided in the improving direction, BH-FDR across the three horizons, at the level I.2 registered.

## What each outcome means, fixed before the run

- **The rule beats its frozen control at a raised horizon and not at 0.9.** The horizon mechanism
  transfers off the control onto a real task. The seven negative results become findings about an
  instrument run at a crippling setting, I.4 says so on evidence, and a connectome arm at the raised
  horizon becomes the obvious next registration.
- **It beats its control at every horizon including 0.9.** The gain is not the horizon — something
  else changed between the I.3 pilot and this campaign, and the change is void until that is found.
  Registered as a stop, not a result.
- **It beats its control at no horizon.** The horizon story does not transfer: it is real on a
  one-step task with a delay and does not reach a 2400-step episode. I.4 records that the mechanism
  was tested and did not carry, which is a sharper statement than the silence it would otherwise
  have. **This is the expected outcome** — see below.
- **Every arm is at the floor and indistinguishable.** The yardstick is not a platform for this
  question either, and the finding is about the platform.

## The honest prior

**This probably does not work**, and the registration says so rather than discovering it.

- I.3's own recovery is partial: at twenty steps of delay even `0.999` stays **below** the
  registered bar (49.8% against 50%). A 2400-step episode is two orders further out.
- At `trace_decay 0.99` a trace retains 0.37 over a hundred steps and effectively never decays
  within a 2400-step episode. That does not remove the credit-assignment problem; it trades a trace
  too short to bridge the delay for one too long to distinguish which action earned the reward. The
  rule may fail at both ends, and this change measures one end without claiming the other is fine.
- Logbook 040 located the yardstick's failure elsewhere entirely — a local rule on a dense stack
  collapsing the representation without decorrelation. If that is the binding constraint, the
  horizon cannot fix it at any setting.

It is worth running anyway because it is an hour, because it is the only actionable lever the phase
has produced, and because a negative result here is a genuine input to I.4 rather than an absence.

## Alternatives considered

- **Write I.4 now.** Rejected: its central question is whether the negative results are about the
  instrument, and this is the one untested mechanism that would answer it.
- **Go straight to the connectome.** Ten hours a run against eight minutes, on a lever with no
  multi-step evidence.
- **Sweep the horizon more finely.** Three settings spanning two orders is enough to see a
  direction; a finer grid is a search, and the useful range was already bounded by I.3, where
  `0.999` added little over `0.99`.
