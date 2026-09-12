# Design: how the 7a decision is reached

## The three branches, as the roadmap wrote them

> **GO (7a shipment) if**: 7a-i's 2×2 resolves with D2-bar results in hand and 7a-ii grounds it in the
> receptor-gated neuromodulator stack.
>
> **SPLIT-shipment if**: 7a forms a self-contained citable result before 7b work starts, **or** L4
> overshoots its software estimate.
>
> **STOP if**: Both L4 implementation and the cross-species transfer are infeasible at the substrate
> level — at which point the diagnosis itself is the Phase 7 deliverable.

Each is tested below against the record. **The verdict is ratified before the logbook is written**;
this document fixes how it is reached and what each branch would license, so the decision is not made
by whichever branch gets written up first.

## GO — unreachable, on its own clause

GO requires 7a-ii to "ground it in the receptor-gated neuromodulator stack". That stack is **B.3**,
the diffusible-signal layer with receptor-class gating. It was never built. Its tracker entry says it
is "paid for only if a rung after it can conclude", and the rung after it is **B.5**, whose gate — a
rule variant passing the clone assay — was failed by four mechanisms: the elastic anchor (13.0 against
the frozen clone's 38.7), the per-synapse protective variable (29.4, failing one clause), the oracle
gate (28.9), and the node-perturbation repair (12.0, and 20.6 with its perturbation off).

So GO is not unreachable because the work was skipped; it is unreachable because **the gate that would
have paid for it never opened**. That is a fact about the evidence, not a scheduling failure, and the
record should say so plainly.

## STOP — would overstate the record in the opposite direction

STOP requires **both** L4 implementation **and** cross-species transfer to be "infeasible at the
substrate level". Neither is established:

- **The substrate is not the limit.** It holds a competent policy — 73.7% from a full-parameter clone,
  38.7% through the chemical weights alone behind an anatomical readout — and PPO solves the task from
  random weights at 68.5%. What failed is one rule family, for a reason I.0 names.
- **Cross-species transfer was never attempted.** C.1–C.8 are unstarted. Nothing about it has been
  shown infeasible.

Taking STOP would repeat, in the opposite direction, the error
[056](../../../docs/experiments/logbooks/056-l4-ladder-reread.md) corrected: attributing to the
substrate what belongs to the instrument.

## SPLIT — the branch the evidence fits

SPLIT asks whether 7a forms a self-contained citable result before 7b work starts. It does, and it is
**two** results rather than one:

1. **A systematic negative with a diagnosed cause.** Local reward-modulated three-factor plasticity
   does not learn this task on any substrate tried, and the record says why rather than only that: the
   rule as implemented had a gradient alignment of +0.009 and sat below the cue-blind floor on a
   one-step task with closed-form bounds; the eligibility formulation that repairs it reaches +0.263
   and passes that control; and the repair still fails on every multi-step task, writing 1.28–1.31×
   its own weight norm in a worsening direction. Three brakes, a decorrelating term, structured
   instruction, σ-annealing and an eligibility-horizon account were each tested and each failed on the
   record.
2. **A positive wiring result on learning speed.** The wild-type connectome reaches competence
   **~35% sooner** than a degree-matched rewiring on a foraging cell under thermal pressure (64 paired
   seeds, replicated) and **+23.5% sooner** with temperature removed (32 seeds), with both learning
   gates passing on every seed and the untrained prior indistinguishable in three independent
   measurements. **The advantage is created by learning, not inherited from the graph.**

The second is what makes this a shipment rather than a closure: Phase 7 was asked whether the wiring
is load-bearing, and on the axis and cells where the question is measurable, it is.

**Recommendation: SPLIT**, with 7a shipped as those two results and 7b's status decided separately
below. Review is invited to challenge it — the case for STOP is that the *headline* MUST (a local rule
reading the wiring) is unmet, and a reader could hold that 7a without it is not a shipment.

## What ships, and what it may not be cited as

**Ships**: the negative with its diagnosis; the positive with its three caveats — **speed, not
endpoint performance** (the endpoint saturates and on the thermal cell the null is nominally ahead);
**two cells of one hard-foraging family**, so extension to other task families is untested; and
**rewirings drawn from run seeds 1–64**, with V.3 reusing 1–32, so the two cells are independent in
task and initialisation but not in rewiring.

**May not be cited as**: a local rule reading the wiring (no rule does); a performance advantage; a
result about any hard task; or evidence that the wiring's advantage is a property of the
thermosensory pathway (V.2 closed the shortest-path reading, and V.3 shows the projection is
unnecessary).

## 7b's gate: the letter and the rationale diverge

7b's comparative runs (C.3/C.5) are gated on "**a registered result in which the wild-type wiring
beats its rewired null under a local rule**", with the stated reason that "transferring a
wiring-indifferent learner between species measures nothing about wiring".

- **The letter is unmet.** V.1 and V.3 ran PPO. No local rule beats anything.
- **The rationale is satisfied.** The wiring is not indifferent: it is worth a quarter to a third off
  time-to-competence under an optimiser that works.

Reading the letter loosely would be the kind of move this phase has repeatedly caught itself making.
So the record states the divergence and puts **three options to ratification** rather than resolving
it silently:

1. **Amend the gate to the rationale** — 7b proceeds with PPO as the learning method for the
   comparative sweep, which D11 already lists as "secondary context", and the grounded modulated rule
   is dropped from C.5's requirements until a rule exists that learns.
2. **Keep the gate as written** — 7b stays blocked, and the phase's forward work is rule families
   (e-prop, the task ladder) rather than comparative connectomics.
3. **Split C.5** — the pipeline and single-species work proceed; the cross-species contrast waits.

This change records the options and the evidence; it does not choose.

## What this design does not do

- It does not run V.4, the fresh-rewiring panel. That is the named caveat on the claim, registered and
  unrun, and a decision should not wait on a refinement of a caveat it states honestly.
- It does not mark Phase 7 complete. The tracker forbids it and 7b is unshipped.
- It does not reopen any committed verdict.
