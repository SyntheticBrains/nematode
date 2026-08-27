## Overview

`phase7-tracking` is the living tracker for Phase 7 (Deepen — Plasticity & Cross-Species Transfer), authored after the plan itself was finalised in roadmap v4.2 (PR #300: pre-start review → D1–D8 ratification → three-lens adversarial review → D9–D13 + amendments → PR-review fixes). This design deliberately duplicates as little as possible: **`docs/roadmap.md` § Phase 7 is the authoritative statement** of the hypothesis, deliverables, decisions D1–D13, exit criteria, and risk table. What lives here is (a) how the tracker relates to those decisions, (b) the execution-protocol standards every Phase 7 milestone change inherits, and (c) the questions deliberately left open for per-milestone changes.

## Goals / Non-Goals

**Goals:**

- Give the committed Phase 7 work an honest home (genuine `[ ]` not-started tasks) across the three shipments (7a-i / 7a-ii / 7b) with the SHOULD/MAY items attached to their windows.
- Make D1–D13 operationally binding: milestone changes conform to them, and amendments happen here, dated, not silently in milestone changes.
- Pin the Phase 6 execution-protocol learnings where milestone changes must inherit them (they were paid for expensively — Logbooks 029/034/036).

**Non-Goals:**

- Restating or re-deciding the plan — roadmap v4.2 § Phase 7 is authoritative; on any divergence, the roadmap wins and this tracker is corrected.
- Deciding the open implementation questions below — those belong in per-milestone changes.
- Re-opening Phase 6b scheduling (decoupled per D13; `phase6b-tracking` owns it) or co-evolution (deferred, no destination).

## Design Decisions

### Decision A: The roadmap's D1–D13 are binding here, amended only here

The ratified decision table (`docs/roadmap.md` § Phase 7 § Pre-registered design decisions) is the contract this tracker enforces. One-line index (full text in the roadmap):

- **D1** rule family: rate-based three-factor primary; spiking MAY; e-prop fallback.
- **D2** success tests: within-regime — (i) plastic wild-type beats plastic rewired-null at q < 0.05; (ii) reaches/exceeds the matched-rule MLP band; PPO-arm tier placement descriptive only. Frozen-weights baseline pinned (Cook-2019 synapse-count-derived init, no learning).
- **D3** cross-species design: matched head-truncation; two homologous behaviours MUST; species-appropriate third behaviour SHOULD; dauer pathfinder first.
- **D4** learnable gap junctions: YES, bounded SHOULD ablation (C. elegans L4 substrate only).
- **D5** #254: freeze — remove dead keys, behaviour unchanged.
- **D6** env vectorisation: scoped-first; the 6b reduced-search budget contract is pinned in `phase6b-tracking` (T8.0), with the L3 evidence bar (lag-matrix or equivalent instrument) per roadmap D6.
- **D7** state-dependent action `std`: early platform change + 036 klinokinesis validation gate + post-D7 n=8 re-baseline; then substrate freeze.
- **D8** L4 package: new `learning_rules/` (never overload `quantumnematode/plasticity/`, the quantum-plasticity eval).
- **D9** head-circuit scaffold: AVA/AVB command-interneuron readout for both arms; sensor-coverage audit; head-scope evasion is distal-only.
- **D10** L4 panel arms: plastic wild-type + plastic rewired-null + frozen/vanilla floors + matched-rule MLP; PPO cells of the 2×2 are qualitative Phase 6 context, not panel rows.
- **D11** cross-species protocol: comparative cross-connectome learning MUST (under the L4 rule; PPO secondary); weight-transplant transfer SHOULD; dauer first; N=2 sensitivity run; one pre-registered transfer metric per behaviour.
- **D12** diffusible layer v1: per-modulator global scalars, brain-internal, receptor-class gating; head-truncated arms run internal-state-driven modulation only (deterministic source policy; ghost/external sources rejected).
- **D13** sequencing: 7a-i / 7a-ii split; 6a preprint SHOULD in the 7a-i window; imitation-warm-start arm SHOULD, early; 6b decoupled.

Amending any D-decision requires a dated note in this change (and a matching roadmap edit), before the affected milestone change merges — the same anti-scope-creep discipline as `phase6-tracking` Decision 4.

### Decision B: Execution-protocol standards (inherited from Phase 6, non-optional)

Every Phase 7 panel or sweep milestone conforms to these; they are credibility constraints, not refinements (provenance: Logbooks 029 § Method, 034, 036, consolidated at the roadmap v4.2 review — PR #300):

1. **n ≥ 8 paired-seed**, Wilcoxon + bootstrap CIs, **BH-FDR within-pass**, per-seed values and converged-fraction reported alongside.
2. **Uniform budget set by the slowest converger**, with a **convergence audit before ranking** (the 029 near-misses: silent last-10 fallback misread two arms; an under-budgeted pass under-ranked CfC by a tier). Use the **level-agnostic plateau metric** — never a threshold-calibrated detector on a new learning regime. L4 rules are expected to be slower than PPO; budget accordingly.
3. **Metric audit before every panel on a new behaviour regime** (the 036 curving-rate floor collapse: an unfloored smoke read was ~99% artifact).
4. **Never read a connectome result off one seed** (034's single-seed sign reversal) — bar (a) ensemble-invariance for any wiring claim.
5. **Byte-identical-when-off** for every new mechanism (traces, rules, diffusible layer, rewiring, transplant).
6. **Load-time config validation** covers all new plasticity keys from day one (the #253/#254 class of silent-drop bug).
7. **Substrate freeze**: all platform changes (D5, D7, re-baseline) land, validate, and freeze *before* the L4 panel; any comparison spanning a substrate change is qualitative (the 2026-06-14 non-commensurability precedent).

### Decision C: Shipment semantics mirror 6a/6b

The 7a GO ("7a complete / 7b pending") is a shipment decision, not phase completion — Phase 7 is marked COMPLETE only when the 7b comparative cross-connectome sweep has shipped **and the synthesis publishes** (the single completion predicate, identical in spec.md, tasks.md, and the roadmap Go/No-Go). Splits are invoked on pre-registered criteria (including by success), never on month counts; estimates are tracked in active-work weeks.

## Open Questions (resolved in per-milestone changes, not here)

- **Rule-seam shape.** `LearningRule`/`BrainTopology` Protocols exist with zero rule-side consumers; the connectome brain's PPO update is inlined. Protocol refactor vs bespoke brain is the first 7a-i milestone's headline decision (`add-l4-trace-substrate`), with the lstmppo per-step-hidden + chunked-BPTT pattern as the in-repo precedent for cross-step state.
- **Exact panel budget.** The uniform episode budget and the concrete matched-rule-MLP band values are pinned in the panel milestone change after the minimal rule's convergence behaviour is measured (per Decision B.2 the slowest arm sets it).
- **Transplant spec.** The homology-mapped weight-transplant (D11 SHOULD): per-edge mapping through the homology table, the non-homologous-edge policy, and zero-shot vs fine-tune protocol details.
- **Dauer ingest details.** Yim et al. 2024 data format and licence handling for the pathfinder loader (data is in OpenWorm's ConnectomeToolbox; exact source pinned at ingest time).
- **Predation-task env shape** (SHOULD, 7b): prey object, capture mechanics, reward design — scoped in its own change if the SHOULD is exercised.

## Risks

- **The likeliest scientific outcome is the robustness branch** (2×2 null): pre-registered as a citable closure path (roadmap risk table "Partial D2 outcome" row) — one sensitivity pass, ensemble-invariance check, warm-start-aided read, then negative-result closure. The tracker's job is to keep that branch cheap and honest, not to rescue it.
- **7b is a full phase in half-phase clothing** (feasibility review: ~8-12 active weeks — D9 scaffold, truncation re-baseline, two loaders, homology table, sweep). Mitigation: dauer-first pathfinder builds the pipeline cheaply; 7b lands beyond the window by default expectation (roadmap § timeline).
- **Solo-maintainer serialization**: the 7a-i window also carries the 6a preprint and warm-start arm (both SHOULD). If the window overloads, the preprint outranks the warm-start arm, which outranks everything MAY — the claim-stake is the item with a competitive clock (Dhiman/flyvis fast-follow risk). Exception: if the 2×2 resolves to the robustness branch, S.2 is **promoted to required-for-closure** — the roadmap's Partial-D2 pivot requires the warm-start read before negative-result closure — and this drop-priority no longer applies to it.
