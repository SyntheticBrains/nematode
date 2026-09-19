# Design: what a close has to state, and how each unresolved criterion is decided

## Every criterion gets one of five statuses, and the vocabulary is the point

Phase 5's synthesis set the precedent — three STOP verdicts recorded as substrate diagnoses rather
than failures — and Phase 6a's recorded its L3 criterion as *deferred to 6b, not an unmet MUST*. What
neither faced is a criterion that is **unreachable because an earlier result removed its
precondition**, which Phase 7 has twice. So the walkthrough uses five statuses, and each carries a
different obligation:

| status | what it asserts | obligation |
|---|---|---|
| **met** | the criterion's own test ran and passed | name the logbook |
| **unmet-with-reason** | the test ran and did not pass | name the logbook and the diagnosis |
| **deferred-with-destination** | not attempted, and it is scheduled elsewhere | name the destination and the decision that moved it |
| **superseded-by-result** | not attempted, and a committed result removed the question | name the result and why the gate no longer exists |
| **unreachable-with-reason** | not attempted, and nothing in the phase could have made it attemptable | name what was missing |

"Deferred" and "superseded" are not interchangeable. A deferred criterion is still a question; a
superseded one is not. Using the softer word for the harder case is how a phase close reads as
tidier than it is.

## The two criteria that need a decision, argued from the record

**D4, the learnable-gap-junction ablation: deferred-with-destination, narrowed.** The first draft of
this design called it *superseded*, which is the tidier word for the harder case — the error this
change's own spec requirement exists to catch. B.6 already carries a recorded status from 2026-09-13
with **two** destinations: after a rule reads the wiring, **or as a PPO arm on the block-V cells**.
Two committed results narrow it without closing it. R.2 found **every arm which writes the wiring does
worse** than the readout-only control, by 3.9 to 15.9 foods, which closes the first destination: there
is no local rule under which making a further tensor plastic is expected to help. L.5 then tested gap
junctions as **fixed features** at the per-neuron width and found removing them *helped both wirings*,
which makes a plastic-gap-junction arm less promising still. But **neither touches the second
destination**: under PPO the wiring is learning-speed-relevant (block V), gap junctions are frozen
there, and whether making them plastic changes that advantage is live and unattempted. So the status
is **deferred**, its destination the PPO arm, with R.2 and L.5 recorded as narrowing the question and
lowering its priority rather than removing it.

**Co-primary biological validation: unreachable-with-reason.** It asked for sign- or shape-level
agreement with dopamine-gated forgetting and Leifer navigation re-weighting. Both are predictions
*about a plastic wiring*: they need a rule that writes the connectome to some benefit, which R.2
established does not exist in this rule family, and which the D14 restatement removed from the
plausibility claim. Under `readout_only` the wiring never changes, so the model has no re-weighting
to compare against the data. Nothing in Phase 7 could have made this attemptable, and calling it
deferred would imply a schedule it has never had.

## Block V's confound: a standing condition, not a caveat

Logbook 065 recorded that `rewire_seed` is unset, so each seed's rewired graph derives from its run
seed and **rewiring varies with initialisation**. That was honest, and it was filed as an open caveat.
Two things change its weight. First, after L.1b the block V effect is the phase's strongest citable
result, so the confound now sits under the headline rather than beside it. Second, **Dhiman 2026
reports that the fly connectome's advantage dissolves under shared initialisation plus a
degree-preserving null** — the same control, on the same kind of claim, in another organism.

So it is promoted from caveat to **standing condition**: carried in the same sentence as the claim at
every citation site, and the control named as the successor phase's first act. The control is not
trivial to specify — "the same initialisation" has no single meaning once the mask changes, since the
init scale is `1/sqrt(chemical in-degree)` and rewiring preserves the degree sequence but not which
neuron holds which degree — so it needs its own design decision, which is precisely why it belongs at
the opening of a phase rather than the end of one.

## The unswept pins are a condition, and the arithmetic says which ones matter

L.1b's lesson was not "sweep more"; it was that an inherited pin **set the sign** of a registered
primary. Three pins remain under every `readout_only` result:

| pin | value | examined where? | why it could matter under `readout_only` |
|---|---|---|---|
| `forward_pass_depth` | 4 | **nowhere** — quoted as a fact in R.1c, never varied | it defines the reservoir's features: at depth 4 only neurons within four hops of a sensory injection reach the motor pool, so the depth partly determines *which* wiring the learner can see |
| `trace_decay` | 0.9 | on the three-factor rule's control (I.3), never here | it decays the readout's **own** eligibility trace, asserted in the topology's plastic-readout branch |
| `initial_log_std` | −1.0 | **swept by hand under the rule** (R.1d: action-noise std 0.368–1.0, null) and **named an open question by that same record** | R.1d found the anatomical readout's *scale* costs more than its direction against a fixed action noise, and left "the readout's parameterisation or the fixed `initial_log_std` beside it" as the open question, calling it cheap to ask |

Two of the three are unexamined under this learner; the third is **an open question the record already
dated and sized**, which is a stronger statement than "unswept" and is cited as R.1d's own words
rather than as a new observation. `forward_pass_depth` is the one with a mechanism argument, so it
leads Phase 8's calibration rung; `initial_log_std` follows it because R.1d already said what to ask.

## What the record rests on, stated rather than assumed

`campaigns/`, `exports/` and `experiments/` are gitignored. The committed per-seed CSVs under
`supporting/` are therefore the durable record, and they exist for every logbook from 040 onward. The
close states three things: what is re-derivable from git alone (every headline figure, from the
committed CSVs); what needs the local campaign logs (any re-derivation from raw, which still parses);
and what is gone (exports for every campaign before the readout-width era, and the tracked-experiment
records for one block V campaign, which already cost one parsed field during L.1b's pilot).

## What this close does not claim

Not that Phase 7 met its MUSTs — it did not, and the SPLIT stands. Not that the ladder is finished.
Not that the pins are calibrated. Not that block V's effect is a wiring effect independent of
initialisation, which is the open control. And not that the synthesis's status assignments are
anything but judgements from the committed record, each argued where it is made.
