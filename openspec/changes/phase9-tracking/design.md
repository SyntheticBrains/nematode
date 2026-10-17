## Overview

`phase9-tracking` is the living tracker for Phase 9 (*Screen, then Mechanism — the Gap Junctions, and the Ranking Under a Trigger*), authored after the plan itself was finalised in roadmap v4.5 (the Phase 9 plan review of 2026-10-14, its three-lens critique of 2026-10-15, and the maintainer's decisions D23–D32, recorded in `docs/roadmap.md` § Phase 9). This design duplicates as little as possible: **`docs/roadmap.md` § Phase 9 is the authoritative statement** of the scope decision, the deliverables, decisions D23–D32, exit criteria, risk table, novelty map and go/no-go. What lives here is (a) how the tracker relates to those decisions, (b) the execution-protocol standards every Phase 9 rung change inherits — the thirteen of the Phase 8 tracker, the five additions 8b made to the protocol, and three the Phase 9 decisions add — (c) the seed-band table, and (d) the questions deliberately left open for per-milestone changes.

## Goals / Non-Goals

**Goals:**

- Give the committed Phase 9 work an honest home (genuine `[ ]` not-started tasks) across the two shipments (9a / 9b) with the SHOULD/MAY items attached to their windows and their gates named.
- Make D23–D32 operationally binding: rung changes conform to them, and amendments happen here, dated, not silently in milestone changes.
- Pin the execution-protocol standards where rung changes must inherit them — the thirteen principles of the [phase protocol](../../../docs/research/phase-protocol.md), the five additions of [Logbook 094](../../../docs/experiments/logbooks/094-phase8-synthesis.md) § What 8b changed in the protocol, and Phase 9's own three (D24's null table, D25's collapse statistic, D32's two-arm control) — stated as requirements rather than advice.
- Reserve the seed bands per registration before any pilot runs, so registered seeds stay untouched (protocol principle 8) and no two registrations share a band.

**Non-Goals:**

- Restating or re-deciding the plan — roadmap v4.5 § Phase 9 is authoritative; on any divergence, the roadmap wins and this tracker is corrected.
- Deciding the open implementation questions below — those belong in per-milestone changes.
- Re-opening anything in § Scope exclusions (cross-species, evolution on the wiring beyond G.3's reading, a further chemical-wiring rung, the dynamical substrate as built, multi-agent, 3D, neuropeptides, spiking, the uniform-rule programme, whole-organism fidelity).
- Building anything: the gap-scale knob, the null family, the logging, the body heads and the set-point task are each their own change (roadmap § Technical Debt § Active for Phase 9, items 16–22).

## Design Decisions

### Decision A: The roadmap's D23–D32 are binding here, amended only here

The ratified decision table (`docs/roadmap.md` § Phase 9 § Pre-registered design decisions, 2026-10-15) is the contract this tracker enforces. One-line index (full text in the roadmap):

- **D23** one substrate for the phase: **Emmons 2024 everywhere**, point worm included; the gap-placement effect re-anchored on Emmons at thermal t35 in its own 64-run pilot before G.1 registers, never inside G.1's panel; G.1 carries 082's 0.0577 minimum only if the anchor's effect sits in [0.058, 0.13] and both levels read `readable`, otherwise the minimum is re-fixed from the anchor; 8a and 8b records are not re-read.
- **D24** the null a question reads: gap questions read the gap-only family, chemical questions the chemical-only and boundary-preserving nulls; the degree-preserving null is retired as a standing beside-arm; every null states what it does not preserve, publishes a matched-statistics table in the model's effective units, and carries the shortcut audit in its launch record.
- **D25** collapse is a registered statistic and remedies are platform settings: every panel over 3,000 episodes reports collapse incidence under the pre-registered per-seed definition; no 6,000-episode body panel other than M.1(c)'s confirmatory panel registers before M.1 reads out; a remedy is chosen by rescue of the known collapsers across both wirings, never tuned per arm, must hold the MLP control on fresh seeds, and is applied identically to every arm, byte-identical-when-off, frozen for the phase; at most three rounds; a collapsed seed stays in the primary reading.
- **D26** primary and secondary statistics through the body: `auc_success` stays primary; M.4 chooses one secondary from committed data before any body panel registers, reported beside, never used to re-read a verdict; best-so-far and best-block are never comparison metrics.
- **D27** the ranking's tier and trigger: R.1 runs now (per-family positive controls and body heads, each at its own calibrated entropy; a failing family is ranked last with its reason); R.2 runs once, last, only if all four conditions hold — M.1 closed, a dated body freeze, every family controlled or ranked last, a primary registered beyond the tiers — on the thermal t20 body cell with hard350-500 beside, labelled with its fidelity level; otherwise deferred-with-destination, never run at reduced n.
- **D28** cross-species stays deferred with the new reason (no gap junctions in the *P. pacificus* data, the dauer's unreleased); the within-species biological contrast is the male, read as a replication against its own shuffles behind M.2's baseline; "within the hermaphrodite's band" is not registered.
- **D29** environment vectorisation is decided by T.1, once: GO reads the hot-path prototype at ≥ 5× with the brain's measured share subtracted; GO opens a scoped port at the Phase 9/10 boundary; NO-GO closes D6 as decided.
- **D30** the publication decision is taken at G.3b, either way it reads; the literature-watch trigger stays armed and reopens it at once if the gap-junction half of the null-strength point is published with comparable controls.
- **D31** the behavioural references are repaired before they are read again: K.1 replaces the ratio band by a unit-free quintile statistic with a digitised Pierce-Shimomura 1999 figure beside it; thermal captures graded by the side of Tc; committed verdicts not re-read.
- **D32** what the placed rung claims: a state-gated placed gain at an identified synapse, not reward-driven credit assignment; its positive control names both arms (must flip at the site, must not at a degree-matched random site; both flipping is the result); the trainable-mask capacity control runs first; every result a performance claim under the Phase 7 claim discipline.

*Amendment 2026-10-16 (D23, with the matching roadmap edit; from `add-gap-operating-point`'s spec review).* D23's minimum rule gains its third case: if G.0(a)'s anchor interval includes zero, 0.0577 stands and the G.0 panel's own count-weight gap decides `no_gap_effect`. *(Clarified 2026-10-17, before the panel ran: an anchor whose interval excludes zero **below** — the null ahead, the effect reversed on Emmons — fixes no minimum, and the panel waits on a dated decision. It did not arise.)* G.0's verdict map gains `both` (the lead holds under unit weights and the scattered outliers also reproduce one), which licenses G.1–G.3b with the outlier pairs as their own G.3 class.

Amending any D-decision requires a dated note in this change (and a matching roadmap edit), before the affected milestone change merges — the same anti-scope-creep discipline as `phase6-tracking` Decision 4, `phase7-tracking` Decision A and `phase8-tracking` Decision A.

### Decision B: Execution-protocol standards (inherited, non-optional)

Every Phase 9 rung, panel or sweep conforms to these. B.1–B.13 are the Phase 8 tracker's, restated for this phase's components; B.14–B.18 are the five additions 8b made to the protocol ([Logbook 094](../../../docs/experiments/logbooks/094-phase8-synthesis.md) § What 8b changed in the protocol); B.19–B.21 are Phase 9's own, from D24, D25 and D32.

01. **n ≥ 8 paired-seed** (n ≥ 16 or more where the roadmap says so), Wilcoxon + bootstrap CIs, **BH-FDR within-pass**, per-seed values and converged-fraction reported alongside.
02. **Uniform budget set by the slowest converger**, with a **convergence audit before ranking** on the level-agnostic plateau metric.
03. **Metric audit before every panel on a new behaviour regime** — the repaired klinokinesis statistic (K.1) is audited on the existing captures before any curve is read again.
04. **Never read a connectome result off one seed** — ensemble-invariance for any wiring claim; M.3 measures how much of a null contrast is the rewiring draw.
05. **Byte-identical-when-off** for every new mechanism (the gap-scale knob and the unit-weight, log-count and top-k wirings; every gap-null mode; the per-update logging and checkpoints; each collapse remedy; the body heads; the pirouette head; the set-point task; the placed gain) — **and an arm must be shown to change only what it names, at the point the arms diverge rather than only at construction** (protocol principle 6's 2026-09-21 clause, [Logbook 070](../../../docs/experiments/logbooks/070-init-sharing-control.md)): name the state the arms share and assert the arms leave it in the same condition.
06. **Load-time config validation** covers every new key from day one.
07. **Substrate freeze**: the phase runs on Emmons 2024 everywhere (D23); the body is frozen by a dated signature before R.2 registers (D27), with no C.x change, no body-side klinokinesis rung or shorter step, and no E.1 change pending; any comparison spanning a substrate change is qualitative.
08. **A new component on a validated platform needs its own positive control** (protocol principle 4). The body heads for LSTM/GRU, CfC, Transformer and GA (R.1), any corrected gap model after an `artefact` verdict, the pirouette head (the K 2×2) and the set-point task (P) each clear a positive control before any connectome arm runs on them.
09. **Sweep before pin** (protocol principle 7; D16 carried). The gap scale is the standing instance — a pin no contrast had swept under PPO, which is why G.0 runs first. A setting inherited from a previous rung is a hypothesis at the new one.
10. **Re-establish the baseline at a moved operating point, in its own campaign, before ablating** (protocol principle 6's 2026-09-19 note; [Logbook 067](../../../docs/experiments/logbooks/067-l4-feature-ablations.md)). D23's anchor pilot is this standard applied to the substrate change.
11. **Register a minimum effect beside significance, in both directions of a two-sided reading** (protocol principle 10; [Logbook 068](../../../docs/experiments/logbooks/068-l1b-rate-calibration.md)). G.3 is read two-sided by design.
12. **Body-substrate results are a new reference frame** — never a controlled delta against Logbook 029, block V or any point-worm result; each body rung re-establishes its own floors and baselines, and R.2's claim is labelled with its fidelity level.
13. **Artefact retention registered at phase start** (H.2, A.0's wording): the parsed per-seed CSVs, analysis JSON and launch records committed under `supporting/`; raw logs, step-level exports and trained weights archived off-repo; a campaign directory removed only after its CSV is committed.
14. **Every gate a registration reads is evaluated at the registered point before launch** — floors, the saturation bar, anything that can make a level unreadable — with `gate_preflight.py` on committed runs at that point where they exist, otherwise on a pilot; a gate carried from another cell is re-checked on the new one ([Logbook 077](../../../docs/experiments/logbooks/077-thermal-null-strength.md)).
15. **A control repeated on another cell is sized and judged on that cell's own effect** ([Logbook 078](../../../docs/experiments/logbooks/078-thermal-split.md)), never on the first cell's magnitude.
16. **A difficulty pin for a follow-up is chosen by the gates alone, under a rule fixed before the pilot that decides it** (the thermal target's precedent, [Logbook 078](../../../docs/experiments/logbooks/078-thermal-split.md)).
17. **A panel sized from a small pilot sizes from an upper bound on the pilot's spread**: four of 8b's five pilot-sized readings achieved a larger spread than their pilot's, M.9 by 3.6× ([Logbook 092](../../../docs/experiments/logbooks/092-thermal-body.md)); the point estimate is not the sizing input.
18. **A positive control that fails its fixed rule is retried only as a new registration** — the failed verdict kept, the change fixed before running, new seeds, the same rule (M.9's retry is the precedent).
19. **Every gap null publishes its matched-statistics table and shortcut audit** (D24): the effective maximum and 99th percentile, effective per-neuron totals for ALA, CAN, PVD, AVA and AVB, sensor–motor gap pairs and the count–degree correlation, in the model's effective units; the shortcut audit (`sensory_motor_hops.py`, gap edges included) in the launch record; and a statement of what the null does not preserve.
20. **Every panel over 3,000 episodes reports the collapse statistic** under D25's pre-registered per-seed definition, with Wilson intervals, time-to-collapse censored at the horizon, `auc_success` to 3,000 and to the horizon separately, and IQM and performance profiles beside the mean; **no 6,000-episode body panel other than M.1(c) registers before M.1 reads**; collapsed seeds stay in the primary reading; best-so-far and best-block are never comparison metrics (D26).
21. **The two-arm positive control for any placed rung** (D32): the arm that must pass (the hand-set gain at the identified site flips the index) and the arm that must fail (the same gain at a degree-matched random first-layer site does not) are both named before the run; both flipping voids the control and is the result, and no placed arm then runs.

### Decision C: Shipment semantics mirror 6a/6b, 7a/7b and 8a/8b

Phase 9 ships in two cuts split by success (roadmap § Phase 9 § Go/No-Go, the 6a/6b, 7a/7b and 8a/8b precedent): **9a** — the screen, the instrument, the mechanism block and the ranking's controls — and **9b** — the mechanism's consequences and the ranking. The 9a synthesis writes a go/no-go decision and, on GO, records the phase state as **"9a complete / 9b pending"** — a shipment decision, not phase completion. Three things are read at the 9a synthesis and nowhere else: **G.0's verdict as the gate for 9b's gap work** (G.5, G.6, G.7, and G.8 only on `survives`), **D27's body-freeze decision** (condition 2, signed with the date or recorded as unsignable with the pending change named), and D27's conditions 1 and 3 as they stand. R.2's condition 4 — a primary registered beyond the tiers — belongs to R.2's own registration. The single completion predicate — identical in spec.md, tasks.md and the roadmap — is **the 9a synthesis + the 9b synthesis, each assigning every criterion of its shipment one of the five statuses** (met, unmet-with-reason, deferred-with-destination, superseded-by-result, unreachable-with-reason). "Well underway" never satisfies completion; splits and stops are invoked on pre-registered criteria, never on month counts; estimates are tracked in **machine-hours and engineering days**, not active weeks, because 8b's planned 13–20 active weeks ran in eleven calendar days and about 150 machine-hours (MUST ≈ 70–90 machine-hours and 18–28 engineering days; with R.2 and G.5 ≈ 120–160 machine-hours; everything taken ≈ 210–270 machine-hours and 40–60 engineering days).

### Decision D: External precedent is carried as convergent, with what was verified stated

The plan's novelty map was checked against the review's six literature sweeps of 2026-10-14 and the W40–W42 digests, every citation fetched. The rule for such items, inherited from Phase 7's handling of Dhiman 2026 and FlyGM and Phase 8's handling of Lee 2026: cite as **convergent, not as support**, state what was verified and what was not, and edit the claim at its site rather than defending it. The four related-work items added to the roadmap on 2026-10-15 are carried under that rule:

- **Churchland & Garcia-Ojalvo 2025** (arXiv:2508.09618; *iScience*) — the c302 connectome trained on foraging by derivative-free search; no nulls, no body. A second occupant of the nearest-precedent cell for closed-loop learning on the worm connectome; it occupies no Phase 9 cell.
- **"The digital sphinx"** (Brunton, Abe, Hu & Tuthill, bioRxiv 2026-03-24; eLife reviewed preprint 2026-08-17) — a worm connectome walking a fly body under deep RL, concluding "biologically meaningless", with no shuffled control (its reviewers asked for one). The nearest neighbour to the body cell; it does not occupy it.
- **An 2026** (bioRxiv 10.64898/2026.09.19.752860, v2) — in a fly LIF model, function "resides in the conjunction of wiring placement and strength"; chemical synapses only, placement and strength not separated. Directly on G.1's question without answering it.
- **`arterialist/active-inference`** (GitHub, pushed 2026-10-03; no licence, no publication) — Cook 2019 in a MuJoCo chain under active inference, no null, no validation. Recorded so the novelty map stays honest.

What was verified: each item was fetched and read, and the version notes (Lee 2026 at v2 with an identical abstract; every 2026 connectome-null negative at v1 with no journal home) are recorded in the roadmap's related-work paragraph. What was not: none of the four has been reproduced here, and none is cited as evidence for or against any Phase 9 outcome. Lee 2026's gap-junction-only shuffle remains the Phase 8 precedent — convergent with the gap focus, in a different model with different controls, its null's guarantees stated as different from `rewire_degree_preserving` rather than weaker.

## Seed-band table

**Convention**: pilots and preflights run in the **9xxx** band; registered panels in the **2xxx** band and upward as it fills; every registration takes a band disjoint from every other and from everything Phase 8 used (its highest panel band is 2301–2332, M.9; its highest pilot band is 9721–9728, M.9's retry). Registered seeds stay untouched until the protocol is fixed (protocol principle 8). The bands below are **reserved here**; a registering change may adjust its own band with a dated note on the row, and must not take another row's.

| Registration | Band | Notes |
|---|---|---|
| G.0(a) the Emmons anchor pilot (D23) | 9801–9816 | 64 runs: wild type and gap-only null, learning and frozen |
| G.0 two-seed smokes for the new levels | 9851–9852 | execution checks only, discarded — *added 2026-10-16 by `add-gap-operating-point`* |
| G.0 gate pilot for the deciding levels | 9861–9868 | 64 runs: the `unit` arms and the scattered null beside the count wild type — *added 2026-10-17 by `add-gap-operating-point`* |
| G.0(c)–(g) the operating-point arms | 2401–2464 | the 16- and 32-seed arms take the first 16 or 32 of the band |
| G.1 + G.2, one panel | 2501–2564 | 64 seeds, each arm learning and frozen |
| G.3 the class map | 2601–2664 | 48–64 seeds; frozen floors for the new wirings in an 8-seed preflight on 9821–9828 |
| G.3b the prediction, tested | 2701–2764 | 64 seeds |
| M.1(a) diagnosis re-runs | 2301–2332 (M.9's) | the 8 collapsed and 8 matched intact seeds re-run deterministically — a diagnostic on committed seeds, licensed by the identity check, not a new registration |
| M.1(b) remedies | 2301–2332 (M.9's) | the same 16 seeds under each remedy, read by discordant pairs |
| M.1(c) confirmatory | 2801–2832 | fresh seeds, ≥ 32 |
| M.2 wild type against wild type | 2901–2964 | 64 seeds; floor pilot for the chimeras on 9831–9838 |
| M.3 variance components | 3001–3064 | 8 rewirings × 8 seeds beside the wild type |
| R.1 per-family controls | 9841–9848, per family | 8 disjoint seeds; the same band for every family |
| R.2 the ranking | 3101–3132 | six families × 32 seeds, learning and frozen |
| G.5 boundary-held gap null through the body | 3201–3264 | 32 seeds if M.1 halves the spread, 64 if not |
| G.6 the male's gap layer | 3301–3364 | the cell fixed at registration |
| G.7 the reading learner on G.3b's nulls | G.3b's (2701–2764) | a probe on the same seeds |
| G.8 the contact-constrained null | 3401–3464 | only if G.0 reads `survives` |
| G.3b on hard350 (second cell) | 3701–3828 | 128 seeds, learning only — *reserved by this tracker, not named by the roadmap; the registering change confirms or adjusts it* |
| The K 2×2 | 3501–3516 | 16 seeds |
| P(a) the trainable-mask capacity control | 3601–3632 | |
| P(b), P(c) the two-arm control and the placed arm | — | bands reserved by P(b)'s change, after P(a) reads, with a dated note here |
| E.1, T.1 | — | no seeds: a cost pilot and a performance spike |

## Open Questions (resolved in per-milestone changes, not here)

- **The gap-scale knob's form, and the unit-weight arm.** Whether the multiplier applies before or after the `1/√(deg_a·deg_b)` normalisation, and whether the unit-weight arm (every count set to 1) re-normalises after setting the counts to 1 or keeps the wild type's normalisers — G.0's change decides, byte-identical-when-off either way, and the misleading comment in `connectome_ppo.py` is corrected with the knob.
- **The totals-held construction.** Whether the strength-preserving annealer (`strength_preserving_rand_sa`) or the swap-plus-Sinkhorn–Knopp rescaling is the totals-held null is decided by the annealer's registered residual energy and per-neuron tolerance on the Emmons gap matrix, stated at G.1's registration; the fallback needs float gap weights through the model, and the annealer would make `netneurotools` (or `bctpy`) a new optional dependency under the `analysis` extra, its licence checked before it is added.
- **The gap nulls' boundary sets.** On the point worm the boundary is the 17 injected sensors plus the 39 readout motor neurons; the roadmap's counts (284 pairs touching them, 809 interior) sum to Cook 2019's 1,093 gap pairs and are re-counted on Emmons 2024 under D23. On the body the motor boundary is the neurons that synapse onto the muscles the readout pools, as C.1e's boundary null drew it. Each is stated in the registering change before the null is drawn, with the distinct-null count and distance from the wild type.
- **The scatter rule's degree stratum width.** How wide a degree stratum is when a class pair's non-command endpoint is re-drawn among non-class neurons "of the same degree stratum", and whether the size-matched random scatter draws from the same strata — fixed in G.3's change before its preflight.
- **The collapse definition's parameters.** D25 fixes the shape (a trailing-200-episode peak; onset when the 100-episode mean falls below half of it and stays there ≥ 300 episodes, the first such fall the run does not recover from; censored at the horizon; amended 2026-10-17 from the first fall); M.1's change fixes the sensitivity analysis of those three numbers on the committed 085 and 092 per-seed data before the logs are read.
- **M.2's transplant rule and dataset 7's licence.** Presence-only or rescaled transplant of Witvliet dataset 8's nerve-ring chemical edges, the count rule for either, and whether dataset 7 can be vendored for the method-matched 7-against-8 arm — the licence checked before vendoring, the `PROVENANCE.md` standard.
- **The per-family entropy rule for R.1.** 084's per-dimension rule gives the number; how it is stated for a GA with no entropy term, and how the family-shaped `gate_preflight.py` reads a GA's generations against the 90% bar, is R.1's change.
- **The set-point task's cultivation phase.** How Tc becomes a per-episode cultivation variable on the thermal cell, the episode's cultivation-then-test structure, and the closed-form expectation's reading (sign(CI) = sign(C_cult − C_start)) — P(b)'s change; the salt version with a new ASE modality is biology-primary but later.
- **E.1's drag coefficients.** Registered before the pilot runs (D18's discipline): the anisotropic ratio for the overdamped chain from celegans-sim's formulation, and ElegansBot's as shipped.
- **T.1's measurement protocol.** What the hot-path prototype measures, how the brain's share (≈ 50 µs forward plus ≈ 14 µs update per step) is subtracted, and on which cell — fixed before the spike starts, since GO reads ≥ 5× off it.

## Risks

- **G.0 reads `artefact`, and the block re-aims.** Not a failure (a Park-class methods result) but a re-plan: the named substrate change lands with its own MLP positive control, 092's body contrast is re-run on it before anything else, and G.1–G.3b run on the corrected model or defer. The tracker's job is to keep that branch a named conditional task rather than a surprise, and to keep the STOP branch (no wiring effect on either cell under PPO after the remedy) stated as the roadmap states it.
- **Engineering days are the critical path.** MUST is 18–28 engineering days (M.1's logging build and remedy rounds, R.1's body heads, the null family) against 70–90 machine-hours; the point-worm campaigns fill the machine during the builds. The risk row fires when the MUST builds exceed about 28 days before R.1 runs.
- **Solo-maintainer serialisation.** When it fires, **R.1 and R.2 outrank every SHOULD**; K, P, E.1 and T.1 wait. The ranking's controls are the deliverable the North Star names, and every SHOULD item has a destination.
- **The strength-preserving annealer's failure regime.** Its authors name sparse, heavy-tailed networks as where it fails, which the gap layer is; the swap-plus-Sinkhorn–Knopp arm is the registered fallback, stated at registration, so a failed anneal changes the construction, not the question.
- **The male composite's confounds.** The male's gap layer differs from the hermaphrodite's in placement, degree sequence, edge count, count scale and outliers at once; G.6 is read only against its own shuffles on the hermaphrodite chemical graph, counts rescaled to the hermaphrodite's median, behind M.2's baseline — never as "the male against the hermaphrodite".
