# 059: The 7a Shipment — What Phase 7a Established, and the Decision (7a-ii B.8 / Phase 7)

**Status**: completed — **SPLIT-shipment**, taken 2026-09-13 under the roadmap's own clause, with
**7b's gate left standing as written** and the forward programme fixed and bounded. 7a set out to ask
whether the wild-type *C. elegans* wiring is load-bearing under biologically plausible plasticity. It
ships **two results**, not one. The first is a **systematic negative with a diagnosed cause**: local
reward-modulated three-factor plasticity does not learn this task on any substrate tried, and the
record says *why* — the rule as implemented had a gradient alignment of **+0.009** and sat below the
cue-blind floor on a one-step task whose analytic reference closes 99.9% of the gap; the eligibility
that repairs it reaches **+0.263** and passes that control; and the repair still fails on every
multi-step task, writing **1.28–1.31×** its own weight norm in a worsening direction. The second is a
**positive wiring result on learning speed**: the wild-type connectome reaches competence **+35.4%
sooner** than a degree-matched rewiring on a foraging cell under thermal pressure (64 paired seeds,
replicated on an independent panel) and **+23.5% sooner** with temperature removed entirely (32
seeds), with both learning gates passing on every seed and the untrained prior indistinguishable in
three independent measurements — so **the advantage is created by learning, not inherited from the
graph**. GO was unreachable on its own clause and STOP would have overstated the record in the
opposite direction; both are recorded with why.

**Branch**: `feat/7a-shipment-decision`.

**Date**: 2026-09-13.

**OpenSpec change**: `add-7a-shipment-decision` (extends `plasticity-evaluation`: what a shipment
decision must establish, and how a gate whose letter and rationale diverge is handled).

## Objective

Record the 7a shipment decision, and the 7a-ii synthesis it rests on, before 7b or the phase closeout
proceeds. This runs nothing; every number is quoted from a committed record.

## The MUST sub-deliverables, against what was built

Deliverable 1 named six sub-deliverables that "must be designed together".

| sub-deliverable | built? | what it returned | record |
|---|---|---|---|
| Persistent activity-trace substrate on `ConnectomeTopology` | **yes** (A.2) | cross-step pre/post traces, the architectural addition every trace-based rule needed | [040](040-l4-panel.md) |
| Rate-based three-factor rule (primary arm, D1) | **yes** (A.3) | built, then shown **not to be a policy-gradient estimator** — alignment +0.009, below the cue-blind floor | [048](048-l4-rule-positive-control.md) |
| Receptor-class metadata | **partly** (B.1) | release identities grounded for 280 of 302 neurons; 3,176 of 3,709 synapses signed from the Wang 2024 atlas. **Receptor classes deferred to B.3 and never built.** | [044](044-l4-atlas-signs.md) |
| Diffusible-signal layer (serotonin, dopamine; D12) | **no** (B.3) | never built. Its own condition made it payable only if a rung after it could conclude, and that rung's gate never opened. | — |
| Modulated three-factor rule | **yes** | the third factor is the modulator scalar; ran in every panel | [040](040-l4-panel.md)–[047](047-l4-structured-instruction.md) |
| Structured (pathway-specific) instruction | **yes** (B.4b) | 18 aminergic release neurons reaching 169 of 302 and 71.1% of wild-type synapses; routing made **both** wirings worse | [047](047-l4-structured-instruction.md) |

Four of six built, one partial, one never built — and the one never built is the clause GO depends on.

## What block I established

Seven registered results had asked whether the wiring is legible to a rule whose ability to learn had
never been demonstrated. Block I tested it.

| item | result | record |
|---|---|---|
| I.0 positive control | **fail, control valid** — analytic reference 99.9% on 8/8; the rule below the cue-blind floor at every rate; alignment +0.031 mean / **+0.009 median** | [048](048-l4-rule-positive-control.md) |
| I.1 noise in the eligibility | **passes** — 89% of the gap on 8/8, alignment **+0.263** | [049](supporting/049-l4-node-perturbation/details.md) |
| I.1b σ-annealing | **fail** — 38.1% of the gap; the decay spends 70% of the run below a scale already shown not to clear the bar | [051](supporting/051-l4-sigma-annealing/details.md) |
| I.1c endpoints, perturbation off | **fail**, and **bimodal** — mean 20.6 against 38.7, two seeds improving, one to 73.4 | [052](supporting/052-l4-endpoint-evaluation/details.md) |
| I.2 a statistic matched to the shape | promotes **nothing** — 14 `no_effect`, 3 `degrades` across 17 contrasts | [053](supporting/053-l4-mixture-statistic/details.md) |
| I.3 the unexamined knobs | the eligibility horizon is limiting and recoverable on the one-step control — **−6.8% at 20 steps of delay**, 45.3% at `trace_decay 0.99` | [054](supporting/054-l4-instrument-knobs/details.md) |
| I.3b the horizon on a multi-step task | **does not transfer** — 0.393 / 0.148 / 0.144 foods against a frozen control at 2.233, drift **1.28–1.31×** own norm | [055](supporting/055-l4-horizon-multistep/details.md) |
| I.4 the re-read | of **32** registered contrasts, **16** are substrate findings block I does not reach, **10** are instrument findings, **5** are about neither, 1 is about PPO | [056](056-l4-ladder-reread.md) |

**The negative that ships is this**, and it is stronger than "our rule did not work": the rule family
was not estimating a gradient, the repair that does is destructive on every multi-step task, and six
further interventions — three consolidation brakes, a decorrelating term, structured instruction, and
σ-annealing — each failed on the record with its reason named.

## What block V established

I.4 found five contrasts uninformative because their premise — that learning finds a wild-type
advantage — had never been demonstrated by any method. Block V tested the premise itself.

| item | result | record |
|---|---|---|
| V.1 the wiring contrast on a behaviour the circuit is wired for | **`specific_wiring_efficiency`** — **+35.4%** off time-to-competence, 396 episodes against 613, 64 paired seeds, all four efficiency metrics q ≤ 0.001, replicated on an independent 32-seed panel | [057](057-wiring-premise-contrast.md) |
| V.2 why the seeds differ | **nothing predicts**, and the reason closes a hypothesis: all 39 motor neurons are within four hops of AFD in **every** graph, and the wild type's route is **longer** (3 hops against 1–2) with a longer characteristic path | [probe](supporting/057-wiring-premise-contrast/probe-v2.md) |
| V.3 difficulty or temperature | **`specific_wiring_efficiency`** — **+23.5%** with no temperature and no thermosensory projection, 32 seeds, three of four metrics significant. **Difficulty is sufficient; the projection is unnecessary.** | [058](058-wiring-premise-difficulty.md) |

The controls are what license it. Both wirings learn (gates 32/32 and 64/64 at q = 0.000), and the
**untrained prior is indistinguishable** in three independent measurements — +3.3 over 64 seeds
([041](041-l4-panel2.md)), +0.77 after grounding ([044](044-l4-atlas-signs.md)), −0.17 and −0.01 on
the block-V cells. Whatever the wild type contributes, it contributes **during learning**.

## The decision

### GO — unreachable on its own clause

GO requires 7a-ii to "ground it in the receptor-gated neuromodulator stack". That is **B.3**, never
built, and its own entry made it payable "only if a rung after it can conclude". That rung is **B.5**,
whose gate — a rule variant passing the clone assay — was failed by four mechanisms: the elastic
anchor at **13.0** against the frozen clone's **38.7**, the per-synapse protective variable at **29.4**
failing one clause, the oracle gate at **28.9**, and the node-perturbation repair at **12.0** (and
**20.6** with its perturbation off).

**GO is unreachable because a gate never opened, not because work was skipped.**

### STOP — would overstate the record in the opposite direction

STOP requires **both** L4 implementation **and** cross-species transfer infeasible *at the substrate
level*. Neither is established. The substrate holds a competent policy — **73.7%** from a
full-parameter clone, **38.7%** through the chemical weights alone behind an anatomical readout — and
PPO solves the task from random weights at **68.5%**. Cross-species transfer was never attempted.

Taking STOP would repeat, with the sign flipped, the error [056](056-l4-ladder-reread.md) corrected:
attributing to the substrate what belongs to the instrument.

### SPLIT — taken

SPLIT asks whether 7a forms a self-contained citable result before 7b work starts. It does, twice
over, and each result stands on its own evidence. **The case against is recorded**: the headline MUST —
a local rule reading the wiring — is unmet, and a reader could hold that 7a without it is not a
shipment. It is taken anyway because the clause SPLIT actually tests is the self-contained citable
result, and two of them satisfy it.

### What ships, and what it may not be cited as

**Ships**: the negative with its diagnosis; the positive with three caveats — **speed, not endpoint
performance** (the endpoint saturates and on the thermal cell the null is nominally *ahead* by 0.47
foods of 20); **two cells of one hard-foraging family**, so extension to other task families is
untested; and **rewirings drawn from run seeds 1–64** with V.3 reusing 1–32, so the two cells are
independent in task and initialisation but **not in rewiring**.

**May not be cited as**: a local rule reading the wiring (none does); a performance advantage; a result
about any hard task; or evidence that the advantage is a property of the thermosensory pathway — V.2
closed the shortest-path reading and V.3 shows the projection is unnecessary.

## 7b's gate: the letter and the rationale diverged, and the gate stands

7b's comparative runs are gated on "a registered result in which the wild-type wiring beats its
rewired null **under a local rule**", because "transferring a wiring-indifferent learner between
species measures nothing about wiring".

- **The letter is unmet.** V.1 and V.3 ran PPO.
- **The rationale is satisfied.** The wiring is not indifferent.

Reading the letter loosely is the move this phase has repeatedly caught itself making, so **the gate
stands as written** and the divergence is recorded as a decision rather than resolved by
interpretation.

### The forward programme, bounded

1. **The scale test.** The repaired rule on the MLP yardstick at widths 4 → 64 (8 → 128 perturbed
   units) on the calibrated hard-food cell — the 1/N arithmetic (Werfel, Xie & Seung 2005) that was
   never checked, and the yardstick that failed was 128 units against the connectome's 302, never a
   small-N control. ~80 runs, about **two hours**.

2. **e-prop**, D1's named fallback: eligibility from the settling dynamics, a global learning signal,
   no perturbation noise. Cleared in the protocol's order — the one-step control, the hard-food cell,
   then the wiring contrast against the registered 20% bar.

3. **Acceptance and stopping, with bars rather than judgements — and three outcomes, not two.**

   | outcome | test | what follows |
   |---|---|---|
   | **does not learn** | fails `wt_ppo − wt_frozen` on the hard-food cell, one-sided paired at 16 seeds under BH-FDR — the gate V.1 and V.3 both cleared 32/32 at q = 0.000 | the programme **stops**, 7b proceeds under PPO after the power arithmetic, biological-plausibility claim given up |
   | **learns, misses the wiring bar** | clears the floor; its `wt_ppo − rn_ppo` efficiency contrast does not reach significance **and** ≥ 20% off time-to-competence | a **result, not a failure** — a plausible rule that learns the cell but does not read the wiring where PPO does. Written up; **7b stays gated**, since the gate asks for a rule that beats the null; option 1 becomes the live question rather than an automatic consequence |
   | **learns and reads the wiring** | clears both | 7b's gate **opens by its letter**; the biological claim is available; the substrate rungs reopen |

   The ≥ 20% minimum is the registered bar V.1 and V.3 were both held to; **+35.4%** and **+23.5%** are
   what PPO achieved on those cells and are the reference, not the requirement. Separately, if e-prop's
   **implementation passes 3 active weeks** that is itself a stopping condition, since the whole case
   for this programme over running 7b under PPO is that it is cheaper.

   The middle outcome is **distinct from B.5's re-registration gate**, which asks the same two
   questions before a substrate rung is paid for: a rule in the middle outcome does not open B.5.

**Why this rather than 7b under PPO now**: block V gave the rule programme a positive control it never
had. Before it, a rule failing to find a wiring advantage was uninterpretable. Now the premise is
demonstrated with a size, and a candidate rule that learns the block-V cells can be held to whether it
*also* shows the advantage. The search has a target.

### The substrate rungs reopen conditionally

Every B-tranche rung asked its question of an instrument that could not learn. If the programme
produces a rule that learns the hard-food cell, **B.5, B.1, B.4 and B.4b** become askable and are
**re-registered fresh** on the block-V cells with the block-V bar — not re-run under their old
registrations — and **B.3** becomes payable once B.5 can conclude. B.5's gate should move from the
clone assay to that bar; that restatement is carried as a **dated note**, leaving its registered gate
visible until B.5 is itself re-registered. Block I's findings are about node perturbation specifically
and are not revisited.

## Conclusions

- **SPLIT-shipment**, taken under the roadmap's clause. Phase 7 sits at **7a complete / 7b pending**
  and is **not** COMPLETE. The registered marker is "7a complete", mirroring Phase 6's
  "6a COMPLETE / 6b pending"; it means **the shipment landed**, not that every MUST did. The one that
  did not is the headline: no biologically plausible local rule reads the wiring, and the diffusible
  layer that GO named was never built.
- **Two citable results**: a systematic negative with a diagnosed cause, and a wiring advantage on
  learning speed demonstrated across two cells with its controls.
- **GO unreachable, STOP overstating** — both recorded with why, not by elimination.
- **7b's gate stands as written**, its divergence recorded, and the forward programme is rule families
  with two stopping conditions fixed in advance.
- **No committed verdict changed.**

## Next Steps

- [ ] The scale test, then e-prop, in the protocol's order.
- [ ] V.4, the fresh-rewiring panel, as the named caveat on the positive result.
- [ ] Z.1/Z.2 only when 7b ships. S.1, the 6a preprint, is independent of all of this and unstarted.

## Data References

- No sessions: this record ran nothing. Every figure is quoted from the logbook or supporting record
  named beside it.
- Decision artefacts: `openspec/changes/add-7a-shipment-decision/` (the branch analysis, the
  ratification, the bounded programme).
