# The 7a shipment decision (7a-ii B.8)

## Why

7a-ii's work is finished as far as its own gates allow, and the evidence has changed twice since B.8
was last queued.

**Block I** diagnosed the instrument: the three-factor rule seven registered results ran under was
not a policy-gradient estimator (gradient alignment **+0.009** median, below the cue-blind floor on a
task its analytic reference solves to 99.9%), the repair that passes that control fails on every
multi-step task tried, and
[Logbook 056](../../../docs/experiments/logbooks/056-l4-ladder-reread.md)'s contrast-by-contrast
re-read found only ten of 32 registered contrasts are about the instrument at all.

**Block V** then produced the phase's **first positive wiring result**, and it is neither fragile nor
narrow in the way a single panel would be:
[057](../../../docs/experiments/logbooks/057-wiring-premise-contrast.md) found the wild-type wiring
reaching competence **~35% sooner** than its degree-preserving rewired null over 64 paired seeds,
replicated on an independent 32-seed panel; [V.2](../../../docs/experiments/logbooks/supporting/057-wiring-premise-contrast/probe-v2.md)
closed the shortest-path explanation; and
[058](../../../docs/experiments/logbooks/058-wiring-premise-difficulty.md) removed temperature
entirely and the advantage survives at **+23.5%** over 32 seeds, so it needs neither the
thermosensory projection nor lethal-zone pressure — only a cell hard enough to discriminate.

Nothing else in 7a-ii can run. **B.5's gate never opened** — its clone assay was failed by four
mechanisms — and **B.3's own condition depends on B.5**, so the two items the tracker lists as queued
have no path to being paid for. B.2, B.6 and B.7 are SHOULD-level substrate and validation work that
answers questions this decision does not ask.

## What Changes

- **The 7a-ii synthesis logbook**, walking the MUST sub-deliverables against what was actually built
  and measured: the persistent trace substrate, the rate-based rule, the receptor-class metadata and
  the atlas grounding, the structured instruction, the diffusible layer that was not built — each with
  its verdict and the record that carries it.
- **The shipment verdict**, named against the roadmap's own GO / SPLIT / STOP text, with **why the
  other two branches were not available** rather than only why one was chosen. The verdict is
  ratified before the logbook is written; this change registers how it is reached, not what it is.
- **An honest status for every gate-blocked item.** B.5 recorded as closed-unmet with the four
  mechanisms that failed its gate named; B.3 as closed-unpaid with its dependency stated; B.2, B.6 and
  B.7 as deferred with reasons. The tracker currently reads as though 7a-ii has five outstanding items
  when it has one.
- **A gate divergence surfaced, not resolved.** 7b's comparative runs are gated on "a registered
  result in which the wild-type wiring beats its rewired null **under a local rule**". That letter is
  **unmet** — blocks V.1 and V.3 used PPO. Its stated rationale — "transferring a wiring-indifferent
  learner between species measures nothing about wiring" — **is now satisfied**, because the wiring is
  demonstrably not indifferent. The record states both. **Ratified 2026-09-13: the gate stands as written**, the
  forward programme is rule families with block V's result as their positive control, bounded by a
  stopping rule, and option 1 is the named fallback taken only at the bound. The substrate rungs the
  B tranche could not conclude reopen conditionally, re-registered fresh, if a rule learns the
  hard-food cell.
- **The caveats that travel with the claim**, so the shipment cannot be cited past its evidence:
  speed and not endpoint performance; two cells of one hard-foraging family; and rewirings drawn from
  run seeds 1–64 with V.3 reusing 1–32, which V.4 would close.
- **The roadmap status synced** as the tracker requires, and never to "Phase 7 COMPLETE".

Out of scope: any campaign, any rule work, 7b's own items, and the Phase 7 closeout (Z.1/Z.2).

## Capabilities

**Modified**: `plasticity-evaluation` (what a shipment decision must establish before it is recorded).

## Impact

- New: Logbook 059 and its supporting directory.
- Edited: the experiments index, `CHANGELOG.md`, the tracker (B.8 plus honest statuses for B.2, B.3,
  B.5, B.6, B.7), `docs/roadmap.md` (Phase 7 status, the GO/SPLIT/STOP outcome, the risk-table row).
- No code, no campaigns, no committed verdict altered.
