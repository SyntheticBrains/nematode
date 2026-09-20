## Overview

A.4 redistributes `plasticity-evaluation`'s 44 requirements four ways. This document records the classification rule, the full 44-row mapping, and the decisions that were not obvious.

## Design Decisions

### Decision A: The split follows the spec's own Purpose statement, not a judgement call

`plasticity-evaluation`'s Purpose has always read: *"the sequential multi-objective training evaluation protocol for testing catastrophic forgetting… trains a brain on a sequence of objectives (A → B → C → A'), measuring backward forgetting, forward transfer, and plasticity retention."*

Nothing in it covers how to register a contrast, read a bimodal outcome or carry a standing condition. The 38 methodology requirements were never in scope for the capability they were filed under. That is the classification rule: a requirement stays if the Purpose describes it, and one requirement filed as methodology turns out to qualify (the delayed-reward control, which is a property of `positive_control.py`).

### Decision B: Durable methodology goes where Phase 7 already put its durable methodology

Phase 7 registered its *wiring-contrast* rules in `architecture-comparison-protocol` (`add-wiring-premise-contrast`, `add-wiring-premise-difficulty`) and its *rule-programme* rules in `plasticity-evaluation`. The first was right and the second was expedient. The eleven rules that survive as binding are comparison methodology by the same test, so they join the first group rather than founding a third home. A new `experiment-protocol` capability was considered and rejected: it would split methodology across two specs for no gain.

### Decision C: Folding is compression, not relocation

A folded rule becomes a clause of one or two sentences under an existing principle, not a restatement. The protocol is a document meant to be read in one sitting; moving 18 requirements into it wholesale would defeat the purpose the fold exists to serve. The full registered text is preserved twice — in git history and in the archived change that added it — and the mapping below makes either findable.

The corollary is that **folding is lossy by design**, and the loss is deliberate. Where a specific clause is load-bearing for a harness that still runs (a registered minimum stated as a fraction, a verdict name meaning a failure to detect), the harness audit (task 3) promotes it to the protocol or records the harness test as its authority.

### Decision D: No new principle, no renumbering

`docs/roadmap.md`, `docs/research/literature-watch/context.md` and the Phase 8 tracker cite protocol principles by number (4, 6, 7, 10, 12, 13). A new principle inserted mid-list, or a renumbering, would silently break those citations. Principle 10 absorbs the statistic-and-metric family by widening its own scope instead. The protocol stays at thirteen principles.

## The mapping

Where each of the 44 went. Row order is the spec's.

| # | Requirement | Destination |
|---|---|---|
| 1 | Sequential Multi-Objective Training Protocol | **stays** — capability |
| 2 | Evaluation Blocks at Transition Points | **stays** — capability |
| 3 | Plasticity Metrics Computation | **stays** — capability |
| 4 | Plasticity Test Configuration | **stays** — capability |
| 5 | Results Export | **stays** — capability |
| 6 | Brain Checkpoint Persistence | **stays** — capability |
| 7 | A substrate result is read against the rule's positive control | folded → principle 4 |
| 8 | A panel's contrast reads both components of its outcome | folded → principle 10 |
| 9 | A graded metric is read beside the full-clear metric | folded → principle 10 |
| 10 | A bimodal outcome has a name that licenses nothing | folded → principle 1 |
| 11 | The re-read of committed tables cannot change a committed verdict | folded → principle 12 |
| 12 | An annealed perturbation clears the control before the assay | **retired** |
| 13 | A scheduled arm's frozen control runs the same schedule | **retired** |
| 14 | A perturbing rule's endpoint is evaluated with the perturbation off | **retired** |
| 15 | A pinned setting is examined where the rule is known to learn | folded → principle 7 |
| 16 | The control may delay the reward to make the eligibility horizon measurable | **stays** — reclassified as capability (`positive_control.py`) |
| 17 | A setting found limiting on a control is tested where it was found to matter | folded → principle 11 |
| 18 | An arm at the metric's floor is scored on the graded reading | folded → principle 10 |
| 19 | A perturbing arm is compared with a control at its own setting | **retired** |
| 20 | A re-read establishes what a body of results is evidence about | folded → principle 12 |
| 21 | A re-read carries its committed verdicts and its own corrections | folded → principle 12 |
| 22 | A shipment decision states which pre-registered branch it takes… | folded → principle 12 |
| 23 | A gate whose letter and rationale diverge is recorded… | folded → principle 12 |
| 24 | A stochastic-gradient result records the perturbation dimension… | **retired** |
| 25 | A scale-dependent mechanism is tested where the rule works… | folded → principle 11 |
| 26 | A perturbation that cannot reach the scored outcome is not exploration | **retired** |
| 27 | A diagnostic that borrows a gradient-trained component… | folded → principle 1 |
| 28 | A substituted component is tested against a same-magnitude random control | folded → principle 6 |
| 29 | An eligibility derived from a substrate's own dynamics… | **retired** |
| 30 | A rule whose update needs a per-unit signal… | **retired** |
| 31 | A mechanism whose positive control has a closed-form optimum… | folded → principle 4 |
| 32 | A wiring contrast under a learner that does not write the wiring… | → `architecture-comparison-protocol` |
| 33 | A campaign whose null carries a registered consequence… | → `architecture-comparison-protocol` |
| 34 | A control drawn from the same seeds as its runs states that coupling | → `architecture-comparison-protocol` |
| 35 | A replication uses the original instrument unmodified | → `architecture-comparison-protocol` |
| 36 | A multi-panel replication fixes the disagreement case in advance | → `architecture-comparison-protocol` |
| 37 | A capacity manipulation crossed with a structure contrast… | → `architecture-comparison-protocol` |
| 38 | A metric is chosen for the contrast it must support… | → `architecture-comparison-protocol` |
| 39 | A feature ablation on a positive structure result… | → `architecture-comparison-protocol` |
| 40 | A committed baseline is reused only under a parsed-field identity check | → `architecture-comparison-protocol` |
| 41 | A mechanism probe is registered before its correlation… | folded → principle 11 |
| 42 | A positive result carrying an inherited learner setting is re-read… | → `architecture-comparison-protocol` |
| 43 | A phase close assigns every exit criterion a status… | folded → principle 12 (already there; no-op) |
| 44 | A shipped result with an uncontrolled confound carries it as a standing condition | → `architecture-comparison-protocol` |

**Totals**: 7 stay, 11 move, 18 fold, 8 retire.

### Reading a logbook's provenance line against this table

Logbooks 048, 056 and 059–069 each carry an `**OpenSpec change**` line quoting its requirement's title in prose — for example Logbook 060: *"extends `plasticity-evaluation`: a mechanism whose predicted effect depends on a platform dimension…"*. After this change those phrases match nothing in the live specs. The table above is the index: find the title, read the destination. Rows marked **retired** resolve to the archived change named in the removal delta's Migration line.

## Open Questions (resolved during implementation, not here)

- **Which clauses survive only in a harness test.** The harness audit (task 3) answers this per module; the candidates flagged during planning are the two-thirds-of-the-established-effect minimum in the feature-ablation harness and `survives_without_it` meaning a failure to detect. Where the clause is general it goes to the protocol; where it is specific to one analysis it stays in that analysis's test with a comment saying so.
- **How far principle 10's extension goes.** The three statistic-and-metric rules share a lesson but not a sentence. Whether they compress to one clause or three is a drafting question, settled by whether the protocol still reads in one sitting.

## Risks

- **Over-compression.** Folding 18 rules into seven principles can lose a clause that a future rung needed. Mitigated by the mapping table, the archived text, and the harness audit (task 3) — and bounded by the fact that the eight retired rules are the only ones where recovery would require reading a closed programme's archive.
- **The protocol becomes the long document instead.** Verification includes a length check: past roughly 250 lines the fold has become relocation and should be tightened.
- **A future change re-adds methodology to `plasticity-evaluation`.** The spec's Purpose statement is the guard, and Decision A makes it the explicit test. A rule that the Purpose does not describe belongs elsewhere.
