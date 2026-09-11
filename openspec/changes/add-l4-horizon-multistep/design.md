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

The graded reading — plateau-tail mean foods, out of ten — is the only metric with any range left
here, and this is the case I.2 registered it for: *"an arm whose full-clear rate is at its floor
while its graded measure exceeds the comparator's"*. This change is its first use.

**It is not a comfortable range, and the registration says so.** `foods` discriminates elsewhere —
the connectome arms span 1.95 to 9.06 — but the yardstick's committed value is **0.35 of 10**,
below even a *frozen* connectome's 3.25, over a seed range of 0.06 to 0.67 with a coefficient of
variation of **0.61**. The yardstick is floor-adjacent on both metrics, not merely on the cliff. So
the fourth outcome below — every arm at the floor and indistinguishable — is a live possibility
rather than a formality, and the claim being made for the graded reading is that it is the only
measure with room to move, not that it is a sensitive one.

## The comparison

**Learning arm against its own frozen control, at the same horizon, paired by seed.** Not against
the committed 040 yardstick values, which ran under the original rule and would confound the
eligibility change with the horizon change. Not against a single frozen baseline, because the
perturbation's cost to a policy is not constant across horizons — a longer trace changes what the
rule writes, so each horizon needs its own no-writing control.

One-sided in the improving direction, BH-FDR across the three horizons, at the level I.2
registered.

**Significance alone does not carry the verdict.** A paired rank test at eight seeds fires on the
*consistency of the sign*, not the size of the shift: eight seeds all moving one way reaches
q = 0.012 after correction whether the shift is 0.05 foods or 2.0. On a platform whose arms sit
between 0.06 and 0.67 foods, that makes a statistically clean but behaviourally meaningless result
reachable, and nothing about the test would flag it.

So the horizon counts as transferring only if the shift is **both** significant **and at least 0.5
foods**. That floor is fixed here, before the run, and its two justifications are independent of the
outcome: it is roughly the gap the I.3 pilot showed between the frozen arm and the learning arm
(0.90 against 0.11), and it is about 1.5 within-arm standard deviations of the committed yardstick
table (sd 0.21). A shift smaller than that is reported as observed and explicitly does not license
a connectome arm.

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
  within a 2400-step episode; at `0.999` it accumulates roughly a thousand steps of history before
  decaying meaningfully, and the trace is reset only between episodes. That does not remove the
  credit-assignment problem; it trades a trace too short to bridge the delay for one too long to
  distinguish which action earned the reward. **The two failures look different and the record
  should separate them**: a horizon too short leaves the policy near its frozen control, drifting
  little, because almost nothing is credited; a horizon too long moves the weights substantially in
  a direction unrelated to reward, so the arm departs from its frozen control while performing no
  better — or worse. The arms therefore report their distance from the frozen control as well as
  their score, so "did not learn" and "learned something unrelated" are distinguishable rather than
  both reading as a null.
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
