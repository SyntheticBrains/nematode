## Why

Phase 8's task **A.4**, which the roadmap requires before any other 8a registration.

`openspec/specs/plasticity-evaluation/spec.md` holds **44 requirements** and is two unrelated specs stapled together. Six were created 2026-03-21 by `add-quantum-plasticity-test` and describe the sequential multi-objective forgetting protocol — which is exactly, and only, what the spec's own Purpose statement describes. The other **38 are methodology rules** registered by **eighteen Phase 7 milestone changes**, each recording the rule its rung paid for. They landed here because this was the nearest plasticity-named capability; the block-V *wiring* rules from the same phase went to `architecture-comparison-protocol` instead, which is where comparison methodology belongs.

The cost is borne by whoever designs the next rung: to find the handful of rules that still apply they must read 38, most of which bound a rule programme that closed with a diagnosed cause (Logbook 063 (`docs/experiments/logbooks/063-l4-eprop.md`)). Phase 7's own synthesis named this (Logbook 069 (`docs/experiments/logbooks/069-phase7-synthesis.md`), item 6): the rules should fold into the phase protocol (`docs/research/phase-protocol.md`) so a rung designer reads principles rather than single-use rules.

**Nothing depends on the spec mechanically.** No test, script, CI job, config or anchor link resolves against it; no other capability spec references it. The risk is traceability, and this change addresses it explicitly rather than absorbing it.

## What Changes

### 1. `plasticity-evaluation` reduced to 7 requirements

Its six original capability requirements stay. One requirement currently filed as methodology — "The control may delay the reward to make the eligibility horizon measurable" — **stays and is reclassified**: it describes `quantumnematode/plasticity/positive_control.py`, which is live code consumed by `scripts/analysis/l4_rule_positive_control.py` and three test modules. It was misfiled.

The Purpose statement is unchanged, because it already describes only what remains.

### 2. Eleven requirements moved to `architecture-comparison-protocol`

Durable comparison methodology that binds live Phase 8 work: seed coupling between control and run, capacity crossed with structure, parsed-field identity for reused baselines, standing conditions, power stated in advance, replication instrument unmodified, multi-panel disagreement, fixed-features framing, metric choice, ablation minimum effects, and the inherited-setting re-read. Moved **verbatim**. The receiving spec's Purpose widens to say it covers comparison methodology generally.

### 3. Eighteen requirements folded into the phase protocol

Generalised into clauses under existing principles 1, 4, 6, 7, 10, 11 and 12. **No new principle and no renumbering** — the roadmap and the literature-watch brief cite principles by number. Principle 10's scope extends from "register a minimum effect beside significance" to also matching the statistic and the metric to the outcome's shape, which is the lesson Phase 7's I.2 paid for. One of the eighteen (the five-status close) is already principle 12 and folds to a no-op.

### 4. Eight requirements retired

Each presupposes a rule that writes the substrate through injected per-unit perturbation — σ-annealing schedules, perturbation-dimension bookkeeping, what an e-prop eligibility drops, per-unit sign delivery. That programme is closed. Each carries a `**Reason**` and a `**Migration**` line, per the `rewrite-spiking-surrogate-gradients` and `remove-nematodebench` precedent.

### 5. Traceability measures

- An explicit `## REMOVED Requirements` delta, so the eighteen archived change deltas that added these requirements read as intentionally superseded rather than as drift.
- A **where-each-rule-went table** in `design.md`, 38 rows, so the provenance lines in Logbooks 048, 056 and 059–069 — each of which quotes its requirement's title in prose — stay resolvable after generalisation.
- An audit of the `scripts/analysis/` test docstrings: where a specific clause would survive only in a test assertion, it is either promoted to the protocol or the test is recorded as its authority.

### 6. Counts corrected

`docs/roadmap.md` and `openspec/changes/phase8-tracking/tasks.md` describe A.4 in terms of a requirement count this change changes. Logbook 069 and the roadmap's inheritance list keep their historical 42 untouched.

## Capabilities

**Modified**: `plasticity-evaluation` — 37 requirements removed; 7 remain.
**Modified**: `architecture-comparison-protocol` — 11 requirements added.

## Impact

**Docs:**

- `openspec/changes/consolidate-plasticity-methodology/` — this change and its two delta specs
- `docs/research/phase-protocol.md` — 18 folded clauses; principle 10's scope extended
- `docs/roadmap.md` — A.4's description and the technical-debt entry
- `openspec/changes/phase8-tracking/tasks.md` — A.4 ticked, count corrected

**Code:** None.

**Configs:** None.

## Breaking Changes

None. No consumer resolves against the removed requirements.

## Backward Compatibility

Documentation-only. The removed text remains in git history and in the eighteen archived changes that added it.
