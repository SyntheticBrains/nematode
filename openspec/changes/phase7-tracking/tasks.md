# Tasks: Phase 7 (Deepen — Plasticity & Cross-Species Transfer) Shipment Tracker

This is the living checklist for Phase 7. The plan is authoritative in `docs/roadmap.md`
§ Phase 7 (v4.2, PR #300): the pre-registered 2×2 central hypothesis (rule × wiring),
design decisions D1–D13, exit criteria, and risk table. Phase 7 ships in three cuts —
**7a-i** (platform freeze + minimal rule + 2×2 panel), **7a-ii** (receptor-grounded
neuromodulator stack), **7b** (cross-species comparative learning) — plus a synthesis.
Every Phase 7 milestone PR updates this checklist as part of its diff.

**Status legend**: `[ ]` not started, `[x]` closed — done, or (for unexercised SHOULD scope) dropped/deferred with a dated note, per the house dropped-scope style (`phase6-tracking` precedent), so `openspec archive` is never blocked by an honestly-unexercised SHOULD.

**Preconditions (both hold):**

1. ✅ **Gate 3 GO** — Phase 6a closed ([Logbook 037](../../../docs/experiments/logbooks/037-phase6a-synthesis.md), 2026-07-07). Phase 7 never gates on Phase 6b (`phase6b-tracking`, decoupled per D13).
2. ✅ **Plan finalised** — roadmap v4.2 with D1–D13 ratified and adversarial-review amendments merged (PR #300, merged 2026-08-27).

> **Not in Phase 7.** Co-evolution (deferred, no destination — roadmap RQ4). Phase 6b NEAT (own tracker; opportunistic pending a dated GPU/cloud decision — D13/D6). Neuropeptide layer as a rule substrate (Phase-7+ candidate, roadmap § signalling-layer scope). Full `∂C/∂t = D∇²C` PDE. *C. briggsae* (no connectome exists, re-verified 2026-08).

<!-- -->

> **Execution-protocol standards (Decision B in design.md) apply to every panel/sweep task below** — n ≥ 8 paired-seed with BH-FDR within-pass; uniform budget set by the slowest converger with a convergence audit before ranking (level-agnostic metric); metric audit before every panel; byte-identical-when-off; load-time config validation; substrate freeze before the L4 panel; no wiring claim off a single seed.

<!-- -->

> **Coarse-grained by design.** These sub-tasks are the load-bearing shape; per-milestone OpenSpec changes elaborate them (first: `add-l4-trace-substrate`).

## Shipment 7a-i — Platform Freeze + Minimal Rule + the 2×2 Panel

**OpenSpec changes**: `add-l4-trace-substrate` (first; rule-seam decision + traces — **active**, authored 2026-08-28), then a panel change (placeholders; created per milestone)
**Status**: 🟡 in progress (A.1/A.2 under `add-l4-trace-substrate`)
**Roadmap layer**: L4 (minimal)
**Approx effort**: ~3-5 active weeks (roadmap § estimate restatement)
**Roadmap reference**: `docs/roadmap.md` § Phase 7 § Required deliverables 1, D1/D2/D5/D7/D8/D10/D13

### Pre-panel platform tranche (D5/D7 — land, validate, freeze)

- [ ] P.1 **D5**: remove the #254 dead config keys (`normalize_advantages` et al.) with training behaviour unchanged; byte-identity evidence recorded.
- [ ] P.2 **D7**: state-dependent action `std` across the continuous policy heads (touches `_policy.py` consumers); byte-identical-when-off.
- [ ] P.3 **D7 validation gate**: re-run the Logbook 036 thermotaxis assay post-D7 — klinokinesis signature present before freeze (this gates the Leifer validation target's measurability).
- [ ] P.4 **Post-D7 n=8 re-baseline** of the reference arms (~24-32 runs) — becomes the panel's descriptive reference frame, superseding the Logbook 029 numbers.
- [ ] P.5 Load-time config validation extended to the new plasticity keys; substrate declared **frozen** for the L4 panel.

### L4 minimal rule + panel

- [ ] A.1 Rule-seam decision documented (`LearningRule` Protocol refactor vs bespoke brain — the Protocol currently has zero rule-side consumers; lstmppo chunked-BPTT is the cross-step-state precedent).
- [ ] A.2 Persistent pre/post activity traces on `ConnectomeTopology` (cross-step eligibility state); byte-identical-when-off.
- [ ] A.3 Minimal rate-based three-factor rule (reward-modulated Hebbian + eligibility traces) in the new `learning_rules/` package (D8).
- [ ] A.4 Baseline arms pinned per D2: frozen-weights (Cook-2019 synapse-count-derived init, no learning) + vanilla-rule sanity floors.
- [ ] A.5 Matched-rule MLP arm (D10 — the rule is substrate-generic; this is the ranking yardstick).
- [ ] A.6 Plastic degree-preserving rewired-null arm (D10 — `wiring: rewired_degree_preserving` mechanism from Logbook 034, byte-identical-when-off).
- [ ] A.7 **The 2×2 panel**: plastic wild-type vs plastic rewired-null (primary, paired n ≥ 8) + matched-rule-MLP ranking test, under the D2 success tests; PPO cells enter as qualitative Phase 6 context only.
- [ ] A.8 7a-i logbook (2×2 verdict — recovery / robustness / partial per the roadmap risk table) + roadmap Phase 7 status sync.

### SHOULD in the 7a-i window (priority order per design.md § Risks)

- [ ] S.1 **6a preprint** (arXiv/bioRxiv) drafted and submitted — platform + ranking + rewired-null + real-worm validation, with the pre-registered 2×2 in the discussion (D13).
- [ ] S.2 **Imitation-warm-start arm**: behavioural-clone connectome + rewired-null on the MLP champion's rollouts, then PPO fine-tune; n ≥ 8 (D13 — needed to interpret the L4 result).

## Shipment 7a-ii — Receptor-Grounded Neuromodulator Stack

**OpenSpec change**: placeholder; created at milestone start
**Status**: 🔲 not started
**Roadmap layer**: L4 (grounded)
**Approx effort**: ~4-6 active weeks
**Dependencies**: 7a-i panel closed
**Roadmap reference**: `docs/roadmap.md` § Phase 7 § Required deliverables 1, D4/D12

- [ ] B.1 Receptor/transmitter metadata as a **vendored-data sub-deliverable** with its own provenance doc: release identities from the Wang 2024 CRISPR neurotransmitter atlas; receptor classes from bulk-integrated CeNGEN profiles (~1-2 focused weeks of two-atlas curation).
- [ ] B.2 Internal-state sensory module (satiety/health into the observation — `BrainParams` plumbing exists; this is the minimal metabolic-state grounding).
- [ ] B.3 Diffusible-signal layer v1 per **D12**: per-modulator global scalars (serotonin, dopamine), brain-internal, receptor-class gating; head-scope source policy as pinned (internal-state-only drive on truncated arms).
- [ ] B.4 Modulated three-factor rules — third factor = modulator concentration; receptor metadata routes which synapses see which modulators.
- [ ] B.5 2×2 panel **re-run under the grounded rule** (same protocol, same frozen substrate).
- [ ] B.6 SHOULD: learnable-gap-junction ablation on the *C. elegans* L4 substrate (D4).
- [ ] B.7 SHOULD: co-primary biological validation — dopamine-gated forgetting + Leifer navigation re-weighting, **sign/shape-level** (behavioural-curve machinery; the forgetting target's full reproduction depends on the MAY slow-memory chain and is reported as such).
- [ ] B.8 7a-ii logbook + **7a shipment decision** recorded (GO → roadmap status "7a complete / 7b pending" — never Phase 7 COMPLETE).

## Shipment 7b — Cross-Species Comparative Learning (Head-Circuit Scope)

**OpenSpec change**: placeholder; created at milestone start
**Status**: 🔲 not started
**Roadmap layer**: cross-species
**Approx effort**: ~8-12 active weeks; lands beyond the window by default expectation
**Dependencies**: pipeline tasks (C.1/C.2/C.4) are rule-independent and may start after 7a-i; the comparative runs (C.3/C.5) use the **grounded modulated rule** and depend on 7a-ii — D12's head-scope third-factor source policy presupposes the diffusible layer
**Roadmap reference**: `docs/roadmap.md` § Phase 7 § Required deliverables 2, D3/D9/D11/D12

- [ ] C.1 **D9 scaffold**: AVA/AVB command-interneuron readout for head-truncated arms + scaffold-sensitivity check (second readout on one behaviour) + per-behaviour sensor-coverage audit of the truncated scope (evasion declared distal-only).
- [ ] C.2 Head-truncation of Cook 2019 (nose→RVG, chemical-only) + validation of the truncated *C. elegans* baseline.
- [ ] C.3 SHOULD — **dauer pathfinder, runs first** (D3/D11): Yim 2024 nerve-ring loader + comparative run — builds the multi-connectome pipeline with no homology tax.
- [ ] C.4 *P. pacificus* ingest: shared-core CSV loader (`stevenjcook/cook_et_al_2025_pristionchus` / *Science* SI), species-keyed classification table + validation pathways, explicit **homology mapping table** artefact, species-keyed sensor/motor projection map (replacing the hard-coded elegans tuples in `connectome_ppo.py`).
- [ ] C.5 **Comparative cross-connectome learning sweep** (D11 MUST): two homologous behaviours (klinotaxis, thermotaxis), under the grounded modulated L4 rule (D12 head-scope policy) with PPO as secondary context; one pre-registered transfer metric per behaviour; single-animal-vs-shared-core sensitivity run.
- [ ] C.6 SHOULD: homology-mapped **weight-transplant transfer** (zero-shot + fine-tune; non-homologous-edge policy stated) — the arm that earns the "transfer of trained agents" claim.
- [ ] C.7 SHOULD: species-appropriate third behaviour — elegans distal-only evasion at head scope; pacificus predatory approach/bite (**new predation-task env**, scoped in its own change).
- [ ] C.8 7b logbook + roadmap sync.

## Synthesis

**OpenSpec change**: placeholder; created at Phase 7 close
**Status**: 🔲 not started
**Dependencies**: 7b closed (Phase 7 is marked COMPLETE only when the 7b comparative sweep has shipped and this synthesis publishes)

- [ ] Z.1 Phase 7 synthesis logbook: exit-criteria walkthrough (MUST/SHOULD/MAY), the 2×2 terminal verdict against the claim-discipline bars, cross-species findings, honest negative-result documentation where applicable (Phase 5 precedent).
- [ ] Z.2 `docs/roadmap.md`: Phase 7 status → ✅ COMPLETE; Timeline Overview + exit criteria + Success Levels flipped to terminal state.

> Archiving `phase7-tracking` is an operator-side step that does NOT block task completion (same precedent as `phase5-tracking` / `phase6-tracking`: an "archive me" task would self-block). The archive happens after the synthesis PR merges, via `openspec archive phase7-tracking`.
