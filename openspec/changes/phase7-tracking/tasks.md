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

> **Execution**: the n ≥ 8 paired-seed arms below launch through `scripts/run_campaign.py` (`add-parallel-campaign-runner`), which runs the config × seed cross product as isolated subprocesses of the standard single-run entry point — each run byte-for-byte the command it replaces, so results are unchanged and only wall-clock moves.

<!-- -->

> **Coarse-grained by design.** These sub-tasks are the load-bearing shape; per-milestone OpenSpec changes elaborate them (first: `add-l4-trace-substrate`).

## Shipment 7a-i — Platform Freeze + Minimal Rule + the 2×2 Panel

**OpenSpec changes**: `add-l4-trace-substrate` (merged 2026-09-05, archived `2026-09-04-add-l4-trace-substrate` — A.1/A.2 done), `add-state-dependent-action-std` (merged 2026-09-05, archived `2026-09-05-add-state-dependent-action-std` — the P.1–P.5 platform tranche, closed under Amendment A), then a panel change (placeholder; created per milestone)
**Status**: 🟡 in progress (A.1–A.8 done 2026-09-06 — the 7a-i panel resolved to `sanity_floor_fail`, Logbook 040; P.1–P.5 closed 2026-09-05 under Amendment A — substrate FROZEN mode-off; A.9 panel 2 and A.10 panel 3 both resolved `inconclusive` 2026-09-07, Logbooks 041 and 042 written; next: S.2, then 7a-ii)
**Roadmap layer**: L4 (minimal)
**Approx effort**: ~3-5 active weeks (roadmap § estimate restatement)
**Roadmap reference**: `docs/roadmap.md` § Phase 7 § Required deliverables 1, D1/D2/D5/D7/D8/D10/D13

### Pre-panel platform tranche (D5/D7 — land, validate, freeze)

- [x] P.1 **D5** (verified already-done, 2026-09-05): the #254 dead-key removal shipped during the Phase 6 bit-memory pre-work — zero `normalize_advantages` keys and zero lstmppo renamed-key residue in `configs/` (grep-verified; no code change). Advantage normalization stays unimplemented (closed-issue constraint, 029 raw-GAE lineage).
- [x] P.2 **D7** (`add-state-dependent-action-std`, 2026-09-05): state-dependent std heads on all five continuous brains, byte-identical-when-off and step-0 bit-equal on/off as a run property (RNG-free zero-Parameter heads; no `_policy.py` changes — the helpers were already shape-generic); ships **dormant** per the Amendment A gate outcome.
- [x] P.3 **D7 validation gate — FAILED** ([Logbook 038](../../../docs/experiments/logbooks/038-state-dependent-std-gate.md), 2026-09-05): attempt 1 (ent 0.05) hit the clamp-ceiling trap (monitor caught it live); the one pre-registered entropy-only pass (ent 0.01) left klinokinesis EQUIVOCAL with weathervane regressed — capability without pressure. Dated Amendment A applied; the Leifer/Chen validation target stays unmeasurable; klinokinesis retry is recorded future work via its own pre-registered change.
- [x] P.4 **DESCOPED under Amendment A** (2026-09-05): the substrate froze mode-off, so nothing changed under Logbook 029 — **029 remains the panel's descriptive reference frame** and there is nothing to re-measure. (House dropped-scope tick.)
- [x] P.5 (2026-09-05): plasticity-key half verified already-done (the trace fields are pydantic-validated since #303); the new `continuous_std_mode` field validates via `model_fields` + a model validator. **Substrate declared FROZEN for the L4 panel in `continuous_std_mode: state_independent` (mode-off, Amendment A)** — the panel runs the unchanged 029-lineage substrate; any mode-on retry is a new pre-registered change.

### L4 minimal rule + panel

- [x] A.1 Rule-seam decision documented (`add-l4-trace-substrate`, 2026-08-28: Protocols reconciled to the seam a rule needs — mask projector + learnable parameters; the inlined PPO update extracted verbatim into `learning_rules.ConnectomePPORule` under the M1 frozen-reference byte-equivalence bar; conformance now genuinely `isinstance`-passes).
- [x] A.2 Persistent activity traces on `ConnectomeTopology` (`add-l4-trace-substrate`, 2026-08-28: conditionally-allocated eligibility buffer, v1 formula `E ← λE + M∘(h hᵀ)` under load-bearing `no_grad`, reset at `prepare_episode`; byte-identical when off AND training bit-identical with traces on until a rule consumes them).
- [x] A.3 Minimal rate-based three-factor rule (`add-l4-three-factor-rule`, 2026-09-05): `ConnectomeThreeFactorRule` in `learning_rules/` — `dw = eta * delta * E` with an EMA-baseline prediction error, per-step cadence, no gradients or critic, chemical synapses only, bounded by decay + clamp with saturation telemetry. Two ratified design choices: the eligibility trace became **temporally causal** (`h_prev (x) h`, exercising the A.2 amendment clause — the symmetric form could not distinguish reciprocal edge directions), and the frozen motor readout is set to the **anatomical** dorsal/ventral and forward/backward contrasts rather than a random orthogonal draw. Dale's law **withdrawn at spec review**: synapse signs here are arbitrary draws, not neurotransmitter identity, so the constraint would freeze noise — enforceable once 7a-ii grounds signs in the atlas. Default `ppo` path byte-identical.
- [x] A.4 Baseline arms (`add-l4-baseline-arms`, 2026-09-05): **frozen-weights** and **unmodulated-Hebbian** sanity floors, both configured under the plasticity rule rather than PPO so they share its anatomical readout — a floor decoding differently from the arm it bounds would confound decoding with learning. "Vanilla rule" was undefined in D2/D10 and is resolved here as unmodulated Hebbian (`Δw = η·E`): beating the frozen floor shows only that *something* was learned, beating this one shows something was learned **from reward**. The unmodulated arm still reports the prediction error it discards, so the ablation is visible in telemetry. **Finding (a) resolved**: D2's "Cook-2019 synapse-count-derived initial weights" corrected in wording, not code — the connectome gives edge existence, weights are `N(0, 1/√(chemical in-degree))`, and changing the initialisation would move the substrate Amendment A froze. **Finding (b) was my over-claim and is withdrawn**: the frozen arm needs no weight persistence (topology + seed + freeze flag suffice). The real persistence gap — `save_weights` is a no-op for this brain, so plastic runs discard the synapses they modified — is filed as issue #308 and blocks the imitation-warm-start arm (S.2), not this one. Also added the first connectome smoke entries: the plastic path had never run end to end through the entry point.
- [x] A.5 Matched-rule MLP arm (`add-l4-matched-rule-mlp`, 2026-09-06): D10 called the rule substrate-generic and the code disagreed — it named five connectome attributes. It now reads a `PlasticTopology` seam and `mlpppo` runs it as the ranking yardstick: same rule class, same arithmetic (proved bit-identical on the connectome against a pre-rewrite frozen reference), same hyperparameters from one shared mixin. `MLPTopology` wraps the actor by reference, so the MLP PPO path is byte-identical to a pre-refactor frozen reference too. **Ratified with Chris:** every Linear weight plastic including the output layer — the asymmetry is in the MLP's favour and makes it a *conservative* yardstick (freezing a random output layer would have matched the OLD random decoder, not the anatomical one the connectome runs). Feedforward eligibility is same-step post⊗pre (a layer already orders pre before post; the connectome needed h_prev only because h⊗h is symmetric). Review caught that the discrete path forwards the actor twice per step — only the action-selecting forward accrues eligibility. Next: A.6 rewired-null plastic arm, then the panel.
- [x] A.6 Plastic degree-preserving rewired-null arm (`add-l4-plastic-rewired-null`, 2026-09-06): one config, one key (`wiring: rewired_degree_preserving`) off the plastic wild-type arm — no mechanism, since 034's rewiring precedes topology construction so traces and updates sit on the null wiring by construction. The value is the parity claim, established by probe on the **real C3 config** (predator + thermotaxis projections on) before being specified: identical readout, all five gains, `log_std`, trace config and hyperparameters at one seed; every chemical in/out and gap degree preserved; mask, `w_chem` placement and `g_gap` differ; pairing by run seed. Probe caught a false claim — per-neuron initial weight *energy* is NOT preserved (same draws, different pairs); the spec states scale and declines energy, with a negative-space test. **Ratified with Chris:** primary arm only; rewired versions of the floors are one-key configs deferred to A.7's pre-registration. Next: A.7 the 2×2 panel.
- [x] A.7 **The 2×2 panel**: plastic wild-type vs plastic rewired-null (primary, paired n ≥ 8) + matched-rule-MLP ranking test, under the D2 success tests; PPO cells enter as qualitative Phase 6 context only. *(Complete 2026-09-06 — `add-l4-panel`, PRs #312 and #318, verdict `sanity_floor_fail`: pre-registration, rewired floors, harness and pilot runner landed. The registered pilot pinned nothing — the grid was two orders of magnitude too hot and the MLP yardstick is dead at 0.01 and frozen at ≤3e-3, its per-weight trace ~1000× smaller than the connectome's, so a matched rate is not a matched rule. **Ratified with Chris:** a substrate-invariant rule-scaling change is inserted before the panel; the grid is then re-registered and the pilot re-run. The connectome learns at 1e-4 in a diagnostic probe. Then: centred modulator (#314), rule robustness (#316: shared noise, homeostasis, tanh yardstick), frozen readout (#317); pilot 3 pinned rate 1e-3 / budget 3000; probes 5–6 showed the MLP yardstick cannot hold a policy under the rule even with a frozen readout (hidden-layer collapse without lateral decorrelation) — it enters the panel as registered and is read as a finding; a **sparse random MLP** arm is the pre-registered follow-up. Panel complete 2026-09-06 (`add-l4-panel`): **verdict `sanity_floor_fail`** — T1 +6.6 (q .50, 4/8), T2 +9.6 (q .50, 5/8), T3 −12.4 (q .73, 4/8), T4 +6.2; band vacuous. Outcomes are seed-dependent fixed points; the wild-type Hebbian floor reaches 64–78% on three seeds with no reward and beats the rewired Hebbian floor descriptively (+16.5, 5/8) while the frozen floors tie — the wiring signal is in the floors. Data under `supporting/040-l4-panel/`. Next: A.8 logbook 040; follow-ups pre-registered separately — the sparse random MLP arm, and a rule variant with anti-Hebbian/decorrelating and pathway-specific instruction (Perks et al. 2026, Nature).)*
- [x] A.8 7a-i logbook (2×2 verdict: **`sanity_floor_fail`** — the roadmap risk table's "fails to beat its baselines" branch) + roadmap Phase 7 status sync. *(2026-09-06: [Logbook 040](../../../docs/experiments/logbooks/040-l4-panel.md) published — `sanity_floor_fail`; roadmap Phase 7 row, 7a-i bullet, risk table, novelty map and literature updated; Perks et al. 2026 added; **structured (pathway-specific) instruction** added to 7a-ii's scope; **dynamic wiring** added to Future Directions. **Ratified ordering for what follows**: panel 2 (Hebbian wiring contrast + prior sweep + initialisation factor) → S.2 imitation warm start → 7a-ii with structured instruction.)*
- [x] A.9 **Panel 2 — the Hebbian wiring contrast** (`add-l4-panel2-hebbian-wiring`, 2026-09-07): verdict **`inconclusive`** — P1 wt_hebbian − rn_hebbian +14.1 (80% CI +2.7..+25.0, q .277, 10/16 seeds; +16.2 on panel 1's seeds 1–8, +11.9 on fresh seeds 9–16), the interval clear of zero but the bimodal per-seed outcome (spread 36.5) leaves the Wilcoxon at n = 16 unpowered; P4 prior sweep (64 seeds) +3.3 (CI −0.0..+6.8, q .277; competent fractions 0.22 vs 0.16), so the wild-type advantage is mostly created by the reward-free Hebbian fixed point (learning gains +25.7 vs +9.6), not present in the prior; **P3 reverses** — synapse-count-scaled initialisation (linear, random signs, unit incoming norm) halves the wild-type Hebbian fixed point (31.8 → 15.6) and erases the wiring contrast (P2 +2.0) with no detectable difference on any frozen prior. Records under `supporting/041-l4-panel2/`; [Logbook 041](../../../docs/experiments/logbooks/041-l4-panel2.md) written 2026-09-07. Follow-ups queued: a sparse random MLP arm; a rule variant with anti-Hebbian/decorrelating terms; the plastic arms under count init are not worth running as-is (count init hurts the Hebbian fixed point).
- [x] A.10 **Panel 3 — replicating the Hebbian wiring contrast on fresh seeds** (`add-l4-panel3-hebbian-replication`, 2026-09-07, ratified ahead of S.2): verdict **`inconclusive`** — R1 wt_hebbian − rn_hebbian on seeds 17–64 +8.1 (80% CI +2.3..+14.2, q .19, 24/48 positive, 22 negative); R2 competent-fraction discordance 11 vs 6 (q .19). The estimate shrank with each fresh look (+16.2 → +11.9 → +8.1); pooled over 64 seeds +9.6 (CI +4.3..+15.0, 34/64) with competent fractions 0.39 vs 0.25. The wiring's mark is in the level of the good fixed points (wild-type's best eight 56–86% against 33–56%), not their frequency; a paired rank test sees a chance sign split. Registered power at the corrected threshold was 63–75% for +12..+14 and nearer 50% for the observed +8. **The Hebbian wiring contrast stays unconfirmed after 64 seeds and is not extended further on the same hypothesis.** Records under `supporting/042-l4-panel3/`; [Logbook 042](../../../docs/experiments/logbooks/042-l4-panel3.md) written 2026-09-07.

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
- [ ] B.4b **Structured (pathway-specific) instruction** *(added 2026-09-07 after Logbook 040 and Perks et al. 2026)*: the third factor SHALL be routed per instructive pathway — which synapses receive credit is determined by the receptor/transmitter metadata *and* by the connectivity that carries the modulatory signal to them — not only by modulator concentration and receptor class. Acceptance: the 7a-ii OpenSpec states the pathway model, its provenance, and a test that credit reaches only the synapses the wiring instructs; the 7a-i panel's global-scalar result is the registered baseline.
- [ ] B.5 2×2 panel **re-run under the grounded rule** (same protocol, same frozen substrate).
- [ ] B.6 SHOULD: learnable-gap-junction ablation on the *C. elegans* L4 substrate (D4).
- [ ] B.7 SHOULD: co-primary biological validation — dopamine-gated forgetting + Leifer navigation re-weighting, **sign/shape-level** (behavioural-curve machinery; the forgetting target's full reproduction depends on the MAY slow-memory chain and is reported as such).
- [ ] B.8 *(blocked until B.4b is satisfied)* 7a-ii logbook + **7a shipment decision** recorded (GO → roadmap status "7a complete / 7b pending" — never Phase 7 COMPLETE).

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
