# How a phase is run

*Recorded 2026-09-12, after Phase 7a's block I. This is not a list of rules invented after a
failure. It is the pattern Phases 5 and 6 used, the one place Phase 7 departed from it, and why —
kept so that the departure is recognisable the next time it is about to happen.*

## The pattern that worked

Phase 6 ran a positive control ([Logbook 030](../experiments/logbooks/030-bit-memory-positive-control.md))
**before** the memory-axis programme it gated. Phases 5 and 6 built a ladder from oracle sensing to
biologically honest sensing and measured the gap at each rung. Phase 6 wrote a go/no-go decision at
every gate. Each of those is one of the principles below; none of them was new when Phase 7 began.

## Where Phase 7 departed, and why

Phase 7's positive control is [Logbook 048](../experiments/logbooks/048-l4-rule-positive-control.md)
— eight logbooks after its first panel, 040. Seven registered panels ran before anyone checked that
the rule could learn anything, and the control then showed it could not: its updates were nearly
orthogonal to the policy gradient.

The departure had a specific cause, and it is the failure mode most likely to recur. L4 put a **new
rule** on an **already-validated substrate**. The connectome had cleared Phase 6's gates under PPO,
so the platform felt tested, and that confidence carried over to the rule, which had never been
shown to learn anything. The positive control was skipped not from ignorance of the practice but
because the thing that needed it looked as though it had already passed.

> **A new component on a validated platform still needs its own positive control. The platform's
> validation does not transfer to it.**

This is the same trap as Phase 6's grid-versus-continuous non-commensurability lesson, one level
down.

## The principles, each with the cost of ignoring it

### Before building anything

1. **Name the question and both answers.** Register what a positive and a negative result look like
   and what each licenses, before any data exists. *Phase 7 did this consistently, and it is why
   the negatives are citable rather than merely disappointing.*

2. **Do the feasibility arithmetic.** Ask whether theory says the method can work at this scale and
   horizon, and write the estimate down. *Node perturbation's learning speed scales as roughly 1/N
   in the number of perturbed units (Werfel, Xie & Seung 2005). At 302 units, four draws per step
   and 2400-step episodes the substrate was near a worst case; nothing in the design considered it.
   The estimate would have cost an hour. Measuring it cost a day and confirmed it.*

3. **Establish that the task is solvable with the strongest available method.** Know what "learned"
   looks like before asking a constrained method to produce it. *The phase learned that PPO solves
   this task from random weights — 68.5% on the wild type, 81.2% on the rewired null — at its tenth
   item (Logbook 043), after seven panels had asked a local rule to do it. That number also bounds
   the rule: from random weights it reaches 17.8% and sits below its own unmodulated floor. An
   earlier reading of the same table, that PPO also destroys competent policies, was withdrawn in
   Logbook 056 — warm-starting hurts PPO, which is a fact about warm-starting, not about the task.*

### Instruments before substrates

4. **Positive control first.** The method learns a minimal task with closed-form bounds before it
   touches the real substrate. *Ran twelfth. Cost seven panels.*

5. **A task ladder from the control to the real task**, the method cleared at each rung before the
   next. Easy to hard: foraging, then foraging with predators, then with thermotaxis. Never jump
   orders of magnitude. *All 31 plastic configs sit on the hardest task. The only intermediate rung
   was the one-step control, and its delayed extension stops at twenty steps against episodes of
   244 to 2400. The gap between them is where every candidate explanation went to die.*

6. **Controls presuppose an effect.** Establish that the arm beats its own floors before running a
   contrast against a null. *The degree-preserving rewired null was well built on a premise nobody
   tested; panel 3 found the null's Hebbian floor sitting above the plastic arm.*

### Parameters

7. **Sweep before pin.** A setting pinned on two pilot seeds is a hypothesis, not a recipe; every
   pinned value gets a registered sensitivity check on the cheapest platform where the method works.
   *`trace_decay 0.9` sat under every result for two months. No panel config set it.*

8. **Pilots on disjoint seeds.** Registered seeds stay untouched until the protocol is fixed. *Held
   throughout Phase 7, and it is why the pilots could inform registrations without contaminating
   them.*

### Running

09. **Cheapest platform first; the expensive one only when the cheap one moves.** Estimate cost from
    a pilot configured the way the campaign will be, not a lighter one. *The yardstick-before-
    connectome rule worked every time it was applied and saved ten-hour runs twice. The one cost
    estimate scaled from a lighter pilot was off by a factor of ten.*

10. **Register a minimum effect beside significance**, and name the outcome that means stop. *A
    paired rank test at eight seeds fires on the consistency of the sign, whether the shift is
    0.05 or 2.0 of ten. On a floor-adjacent platform that is a route to reporting nothing as
    something.*

11. **A mechanism must predict, not describe.** A proposed explanation earns its place by a test
    that could have refuted it. *The horizon was a description of the one-step-success /
    multi-step-failure pattern until I.3b tested it on a multi-step task, where it did not
    transfer.*

### Closing

12. **Re-read before shipping.** State which results are about the question and which are about the
    instrument, in one record, before the shipment decision. *Phase 7's I.4.*

## What this is not

It is not a promise that following it produces a positive result. It produces a **decisive** one:
a question the evidence supports asking, answered either way in a form that can be cited. Phase 7's
path could not have produced a positive answer at all, because the thing being varied was never the
binding constraint. The principles above are what would have found that out in the first fortnight.
