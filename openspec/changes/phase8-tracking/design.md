## Overview

`phase8-tracking` is the living tracker for Phase 8 (*Ground, then Embody — Measured Substrate & Body*), authored after the plan itself was finalised in roadmap v4.3 (PR #395: pre-start review → second review pass → D15–D20 ratification → the D15 premise corrected at PR review against `rewire_degree_preserving`). This design duplicates as little as possible: **`docs/roadmap.md` § Phase 8 is the authoritative statement** of the scope decision, the rungs, decisions D15–D20, exit criteria, risk table and novelty map. What lives here is (a) how the tracker relates to those decisions, (b) the execution-protocol standards every Phase 8 rung change inherits, and (c) the questions deliberately left open for per-milestone changes.

## Goals / Non-Goals

**Goals:**

- Give the committed Phase 8 work an honest home (genuine `[ ]` not-started tasks) across the two shipments (8a / 8b) with the SHOULD/MAY items attached to their windows.
- Make D15–D20 operationally binding: rung changes conform to them, and amendments happen here, dated, not silently in milestone changes.
- Pin the execution-protocol standards Phases 6 and 7 paid for — now the thirteen principles of the [phase protocol](../../../docs/research/phase-protocol.md) — where rung changes must inherit them, with the Phase 7 additions (positive control per new component, sweep before pin, re-baseline at a moved operating point, a registered minimum effect in both directions, the five-status close) stated as requirements rather than advice.

**Non-Goals:**

- Restating or re-deciding the plan — roadmap v4.3 § Phase 8 is authoritative; on any divergence, the roadmap wins and this tracker is corrected.
- Deciding the open implementation questions below — those belong in per-milestone changes.
- Re-opening anything in § Scope exclusions (cross-species, multi-agent, evolution, 3D, whole-organism fidelity, the uniform-rule programme).
- The A.4 methodology consolidation itself — it is a task here, executed as its own change against `plasticity-evaluation` and the phase protocol.

## Design Decisions

### Decision A: The roadmap's D15–D20 are binding here, amended only here

The ratified decision table (`docs/roadmap.md` § Phase 8 § Pre-registered design decisions) is the contract this tracker enforces. One-line index (full text in the roadmap):

- **D15** what "the same initialisation" means once the mask changes — *corrected at PR review*: the rewiring is a directed double-edge swap, so every neuron keeps its own in- and out-degree and the `1/sqrt(chemical in-degree)` scale is already matched neuron-for-neuron; what is unmatched is which drawn values land on which edges. Two definitions, both run as arms: (i) dense-draw-then-mask; (ii) per-neuron fan-in sharing (identical multiset of incoming weights per neuron in both graphs, differing only in value-to-partner pairing). n ≥ 16 paired seeds, both block-V cells, `rewire_seed` fixed per pair.
- **D16** operating-point discipline: no Phase 8 contrast at an unswept pin; A.2 sweeps `plasticity_rate`, readout width, `forward_pass_depth`, `initial_log_std`, `trace_decay` on the reading learner, one-factor-at-a-time first, reported as a sensitivity surface.
- **D17** the measured-weight rung: a 2×3, wiring × {random, measured, measured-shuffled}, on the reading learner and under PPO; the null's measured arm inherits values through D15(ii)'s per-neuron, pre-synaptic-index assignment; sign-only vs sign-plus-magnitude and the unit-scale sweep at pilot scale; head scope and full scope reported separately; the PPO arm runs under D15's protocol; runs after A.1 and A.2.
- **D18** muscle gain is not a learner: calibrated once on the MLP positive control, frozen across arms, sensitivity-checked; the learnable readout is a gain vector identical in size across arms.
- **D19** who generates the rhythm: default, the body — a proprioceptive wave generator at body level, the brain setting segmental muscle drive through the NMJ map; the connectome's motor circuit producing the wave itself is C.2's optional second half behind B.2 and a shorter step. Decided and recorded before C.1 registers.
- **D20** shipment shape: 8a = A + B.1 + B.2 + synthesis; 8b = C + D + B.3 + synthesis; 8b does not start until the 8a synthesis has assigned every 8a criterion a status; Phase 8 COMPLETE only when the 8b synthesis lands.

Amending any D-decision requires a dated note in this change (and a matching roadmap edit), before the affected milestone change merges — the same anti-scope-creep discipline as `phase6-tracking` Decision 4 and `phase7-tracking` Decision A.

### Decision B: Execution-protocol standards (inherited, non-optional)

Every Phase 8 rung, panel or sweep conforms to these. The first seven are Phase 6's (Logbooks 029/034/036, consolidated at PR #300); the rest are what Phase 7 paid for (Logbooks 048, 062, 067, 068, 069) and are now principles of the phase protocol.

01. **n ≥ 8 paired-seed** (n ≥ 16 where the roadmap says so), Wilcoxon + bootstrap CIs, **BH-FDR within-pass**, per-seed values and converged-fraction reported alongside.
02. **Uniform budget set by the slowest converger**, with a **convergence audit before ranking** on the level-agnostic plateau metric.
03. **Metric audit before every panel on a new behaviour regime** — and the body substrate is one.
04. **Never read a connectome result off one seed** — bar (a) ensemble-invariance for any wiring claim.
05. **Byte-identical-when-off** for every new mechanism (shared-init modes, measured-weight priors, dynamics, reversal, proprioception, the muscle readout, the body).
06. **Load-time config validation** covers every new key from day one.
07. **Substrate freeze**: every C.0 platform change lands, validates and freezes *before* C.1 registers; any comparison spanning a substrate change is qualitative.
08. **A new component on a validated platform needs its own positive control** (protocol principle 4; Logbook 048's lesson). The dynamical substrate, the muscle readout, the kinematic body and the rod chain each clear an MLP-PPO positive control before any connectome arm runs.
09. **Sweep before pin** (principle 7; D16). A setting inherited from a previous rung is a hypothesis at the new one.
10. **Re-establish the baseline at a moved operating point, in its own campaign, before ablating** (principle 6's 2026-09-19 note; Logbook 067).
11. **Register a minimum effect beside significance, in both directions of a two-sided reading** (principle 10; Logbook 068).
12. **Body-substrate results are a new reference frame** — never a controlled delta against Logbook 029 or block V; each body rung re-establishes its own floors and baselines (the 2026-06-14 non-commensurability precedent).
13. **Artefact retention registered at phase start**: which per-campaign artefacts are committed (the parsed per-seed CSVs under `supporting/`) and which are archived off-repo, so step-level exports are not lost the way Logbook 069 records.

### Decision C: Shipment semantics mirror 6a/6b and 7a/7b

The 8a synthesis writes a go/no-go decision and, on GO, records the phase state as **"8a complete / 8b pending"** — a shipment decision, not phase completion. The single completion predicate — identical in spec.md, tasks.md and the roadmap — is **the 8a synthesis + the 8b synthesis, each assigning every criterion of its shipment one of the five statuses** (met, unmet-with-reason, deferred-with-destination, superseded-by-result, unreachable-with-reason). "Well underway" never satisfies completion; splits and stops are invoked on pre-registered criteria, never on month counts; estimates are tracked in active-work weeks (≈ 19–27 for the phase; 8a ≈ 9–12).

### Decision D: External precedent is carried as convergent, with what was verified stated

The plan's novelty map was narrowed on the day the tracker was authored, by a preprint the re-aimed literature watch surfaced through a seed added that morning (Lee, bioRxiv 2026.09.06.749731, 2026-09-09 — see the roadmap's § Worm body and whole-organism models). The rule for such items, inherited from Phase 7's handling of Dhiman 2026 and FlyGM: cite as **convergent, not as support**, state what was verified and what was not, and edit the claim at its site rather than defending it. For Lee specifically: its shuffle is described as permuting the postsynaptic column and preserving out-degree; whether per-neuron in-degree survives is **not stated** (a column permutation would preserve it up to duplicate collapse), so the roadmap records the difference in *guarantees* between its null and `rewire_degree_preserving` and does not call its null weaker. Its single trained policy per condition, with Wilson intervals over evaluation episodes, is stated as the statistical difference. Its negative — no functional sensory-to-command step after atlas fitting — is registered as a **risk** against B.1, not as evidence about B.1.

## Open Questions (resolved in per-milestone changes, not here)

- **D15(i)'s code path.** The current init draws per neuron at construction; dense-draw-then-mask needs a draw that precedes the mask and is shared across the pair. Whether that is a new init mode on `ConnectomePPOBrain` or a seeded pre-draw handed to both arms is the A.1 change's first decision, byte-identical-when-off either way.
- **A.2's budget.** One-factor-at-a-time around the current point first; which pins earn a full crossing is decided from that pass, not in advance.
- **B.1's unit scale and licence.** How LDS units map onto the tanh rate model's weights (a pin, swept at pilot scale); the licence of the Creamer supplementary weights (checked before vendoring, the `PROVENANCE.md` standard); whether the Randi atlas is ingested alongside or as the fallback source.
- **B.2's integrator and time constants.** Euler vs exact-exponential leaky integration across environment steps; where per-neuron τ comes from (a single global value swept first, class-level values only if a source exists); how gap-junction coupling enters the state update.
- **C.0's step–time constant.** Derived from the validated crawl speed, undulation period and arena scale — one recorded number, cited by every kinematic target and cost estimate.
- **C.1's body.** Segment count for the curvature profile; the resistive-force formulation (Gray–Hancock-class, anisotropic drag ratio from the ElegansBot calibration); how the change of posture between steps is turned into displacement; how the proprioceptive channel is projected onto sensory neurons (with a biological argument, the way the predator projection was).
- **C.2's cost-budget number.** "A 16-seed panel in roughly a day at 16 workers" pinned to a wall-clock figure from a pilot configured the way the campaign will be (principle 9), before the rung registers.
- **D19's body-level generator.** The local propagation rule's form (a Wen-2012-class phase lag between segments driven by proprioceptive stretch) and its parameters, calibrated once on the MLP positive control.

## Risks

- **The likeliest outcome of A.1 is that the effect shrinks, not that it dissolves**, since the scale was already matched and only the value-to-edge pairing moves. Either reading is citable; the tracker's job is to keep the registered outcome honest and stop a shrunken effect being reported as the original.
- **B.1's sensory pathway.** Lee's atlas-fitted model found no functional sensory-to-command step; if measured weights leave the klinotaxis pathway unlearnable here, the sign-only pilot and the reading learner's positive control are what show it, and the rung closes as *unmet-with-reason* with the pathway named — not as a null on the wiring.
- **The body is the compute risk.** The roadmap's risk table already stops the rung at C.1 if C.2's budget fails; the tracker enforces that the budget is a registered number before C.2 registers, so the stop is a pre-registered outcome rather than a judgement call.
- **Solo-maintainer serialisation.** 8a is ~9–12 active weeks and 8b longer; if 8a overloads, A.1 + A.2 + B.1 ship 8a and B.2 carries to 8b as SHOULD (roadmap risk row "8a overshoots"). Within 8b, C.0 + C.1 outrank everything else; C.4 (the ranking through the body) is the first SHOULD to drop.
