# 037: Phase 6a Synthesis — Architecture Comparison on the C. elegans Connectome Substrate (Gate 3 GO)

**Status**: **Phase 6a COMPLETE — Gate 3 GO.** All four Gate-3 sub-criteria are satisfied with
evidence, no MUST architecture STOPped, and the connectome-ranking and real-worm-validation
credibility controls are in hand. Phase 6a closes; **Phase 6b (L3 NEAT topology search) is deferred**
to its own tranche, gated on GPU + environment-vectorisation readiness (tracked under
`phase6b-tracking`), and Phase 7 never gates on it. Phase 6 is marked ✅ COMPLETE only when the 6b
synthesis lands; until then it is **🟡 6a COMPLETE / 6b pending**.

**Date**: 2026-07-07.

**Scope**: the Phase-6a synthesis (Tranche 9a) — Tranches 1–7 + the connectome-structure controls
(`T7.controls`) + the real-worm validation (`T7.validation`). It rolls the tranche logbooks into the
Gate-3 decision and the Phase-6 exit-criterion walkthrough. The L3-NEAT exit criterion is satisfied at
6b (T9b), recorded here as deferred-to-6b, **not** as an unmet Phase-6a MUST.

## Objective

Phase 6 asked one load-bearing question: **on a common high-fidelity substrate, how does a network
constrained by the real *C. elegans* connectome compare to a curated set of contemporary
architectures at learning the worm's behaviours — and what does the comparison teach us about the
connectome, about architecture, and about the substrate itself?** This synthesis states the answer,
grades it against the Phase-6 exit criteria, and records the Gate-3 GO.

## Phase-6 exit-criterion walkthrough (T9.1)

The layered platform (L0–L4) and the three-behaviour scope, checked against evidence:

| Exit criterion | Status | Evidence |
|---|---|---|
| **L0 — connectome substrate operational** | ✅ | T1 — Cook-2019 302-neuron wiring imported; strict chemical mask + gap junctions. |
| **L1 — architecture-as-plugin parity** | ✅ | T2, re-verified at T5 — every family conforms to the `Brain` interface; no per-architecture branches in the sim/train loops. |
| **L2 — weight search across MUST architectures × three behaviours at the statistical bar** | ✅ | T7 / [Logbook 029](029-continuous-architecture-ranking.md) — 6 MUST integrated-C3 cells, n≥8, per-behaviour sub-metrics extracted. |
| **Rung-2 chemical gradients + log-concentration adaptation** | ✅ | T6 — static Fick-shaped fields + the adaptive/biphasic chemosensor. |
| **Corrected ASH/ADL nociception** | ✅ | T3 — two-channel predator mechano/chemo sensing (contact + distal). |
| **≥ 1 real-worm behavioural validation** | ✅ | T7 / [Logbook 035](035-realworm-chemotaxis-validation.md) (chemotaxis, both strategies) + [036](036-realworm-thermotaxis-validation.md) (thermotaxis, partial). |
| **Connectome-structure controls** | ✅ | T7.controls / [Logbook 034](034-connectome-structure-controls.md) — degree-preserving rewired-null. |
| **L3 — NEAT topology search** | ⏭️ deferred to **6b** | Relocated to `phase6b-tracking`; gated on GPU + env-vectorisation. Not a Phase-6a MUST. |
| **L4 — biologically-plausible plasticity** | ⏭️ Phase 7 | Deferred by design (Phase 6 stays on PPO-family learning rules). |

All Phase-6a exit criteria are **met with evidence**.

## The primary result — the architecture ranking (029)

On the same continuous-2D substrate (float `(speed, turn)` kinematics, Euclidean geometry, Fick fields,
adaptive sensing, one frozen continuous-native reward), the six MUST families rank (n=8, plateau-tail
full-clear success, integrated three-behaviour cell):

> **MLP 89.0 ≫ {CfC 75.8 ~ Transformer 74.0} > LSTM 60.1 > connectome 52.2 ≫ GA 15.0**

Three statistically-separated tiers (BH-FDR). The plain **MLP wins decisively and on all three
behaviours** — foraging, predator-evasion, and thermotaxis. The recurrent/attention arms (CfC,
Transformer, LSTM) form a mid-band; the **connectome ranks 5th of 6** (it *learns* the cell 8/8 — not
a STOP — but trails on predator-evasion); the gradient-free **GA collapses** to a floor (~15%),
reproducing its T4 result and confirming the floor is optimiser-fundamental, not action-space-specific.

This is a **sharper, cleaner result than the T4 grid pass** ([Logbook 025](025-weight-search-architecture-ranking.md)),
where the top cluster tied (equivariant-quantum 86 ~ CfC 84 ~ spiking 84 ~ LSTM 84) and the connectome
sat mid-pack (75.6). The sub-saturation continuous cell **discriminates where the grid cell tied**.

## The connectome, honestly (T9.2)

The headline for the focal architecture is a **negative-but-informative** one, and it must be framed
with discipline:

- **The connectome ranks 5th of 6 under PPO weight search**, beaten on all three behaviours by a plain
  MLP. Its gap is specifically on **predator-evasion**, and it is **structural, not under-training**:
  both tuning directions (more entropy, deeper forward pass) made it worse (029).
- **It is a degree-statistics result, not a wiring result.** The degree-preserving rewired-null control
  ([Logbook 034](034-connectome-structure-controls.md)) found the wild-type connectome **statistically
  indistinguishable** from its degree-matched rewirings (wild 52.78% vs rewired 56.06%; paired
  d=−3.28, CI[−8.56,+1.61] spans 0, BH-FDR q=0.770 — the rewired-null is if anything nominally
  higher), robust on both the plateau and the learning-efficiency axes. So under PPO — which
  re-optimises the chemical weights freely on whatever topology it is handed — the **specific *C.
  elegans* wiring confers no advantage over degree-matched alternatives**.
- **This is the Beiran & Litwin-Kumar (2025) degeneracy prediction realised**: a connectome
  under-constrains its trained dynamics, so the n=8 PPO weight solutions are one of many that fit the
  same wiring — we do **not** present one weight set as "the" biological solution. The positive
  precedent is Lappalainen et al. (2024): connectome-constrained networks *can* recover neural
  activity — but there the weights are biophysically constrained, not gradient-descent-free.
- **Scope as an explicit primary limitation**: this is a **wired-chemical-synapse-only** model. The
  missing peptidergic/neuromodulatory layer (Ripoll-Sánchez et al. 2023; Dag et al. 2025) and
  fixed-vs-plastic gap junctions are candidate causes of the evasion gap alongside "no cross-step
  recurrent memory". The rewired-null tested the wiring-vs-statistics question; the learnable-gap-
  junction variant (the plastic-electrical-synapse question) is **deferred** — the 6a evidence
  (degree-statistics) settles the connectome-structure question without it, so it is not raised here.

**Framing for RQ3 and Phase 7**: the 5th-place result points RQ3 toward an **optimal-primary** reading
and is the **motivating hypothesis for Phase 7 L4** — *is PPO simply the wrong learning rule for the
connectome?* — not a dead end. The connectome's value in the animal presumably lives in its evolved
fixed weights + input/output routing, not in being a better substrate for gradient-descent weight
search.

## The memory-axis finding (T9.2b)

A first-class Phase-6 result, currently scattered across four logbooks (030→031→032→033):

> **The comparison *can* cleanly resolve working memory, but the naturalistic worm behaviours do not
> demand it.**

- **Resolves** ([030](030-bit-memory-positive-control.md), [033](033-associative-memory-probe.md)):
  on an engineered delayed-match-to-cue control and a naturalistic associative-update probe, the
  memory arms separate sharply — bit-memory CfC 0.995 / Transformer 0.978 / LSTM 0.939 ≫ the
  memoryless **MLP 0.501** (chance), with the **connectome at 0.499 — indistinguishable from the MLP**
  (its recurrence is within-step settling, not cross-step working memory). The comparison has the
  resolving power.
- **Not demanded** ([032](032-ars-source-depletion.md)): the biologically-valid area-restricted-search
  probe returned a **NULL** — real *C. elegans* foraging/ARS is reactive-dominated and short-horizon,
  so the naturalistic task does not exercise the memory axis.
- **New-architecture candidates** ([031](031-minimal-rnn-candidates.md)): minGRU/minLSTM both cleared
  the memory probes (both prongs positive).

This is **what explains the memoryless MLP winning the main ranking** — the reactive multi-objective
cell under-tests working memory — and it **directly motivates the Phase-7 faithful-slow-memory
deliverable**: build a task (or a substrate feature) that genuinely demands the memory the comparison
can resolve.

## Negative findings, honestly (T9.3)

No MUST integrated-C3 cell STOPped — the connectome learns 8/8, it simply ranks last-but-GA. The
substrate-grounded diagnoses are themselves contributions: the connectome evasion-lag is structural
(029); the connectome ranking is degree-statistics (034); the GA floor is optimiser-fundamental (029);
the memory axis is resolvable-but-undemanded (030–033); and the **thermotaxis validation is a partial,
behaviour-*difference* finding** ([036](036-realworm-thermotaxis-validation.md)) — the RL worm
reproduces the thermal weathervane (weakly, sensor-driven) but not klinokinesis, because its
continuous-Gaussian action head has a **state-independent std** (it steers, it does not stochastically
random-walk) and it **migrates-and-parks** rather than isothermal-tracks. These are honest limits of
the current policy/substrate, and Phase-7 targets.

## Cross-regime comparison (T9.4)

The grid (T4/025) → continuous (T7/029) comparison is reported as a **qualitative cross-regime
observation, not a controlled single-variable delta** — the substrates are **non-commensurable** (float
kinematics, Euclidean geometry, Fick/adaptive fidelity all shift together, so carried-over parameters
do not define a matched difficulty; the `damage_radius = 0` case — valid on grid, unreachable on
continuous — is the proof). The within-T7 ranking is the clean primary result. The qualitative
observation: **the same architectures learn the same repertoire on the much higher-fidelity world, and
the ranking's *character* survives the jump** (MLP strong; GA floor; connectome trails on evasion) —
but the fidelity jump **sharpens** the separation (the grid's flat top-cluster tie becomes three clean
tiers). Non-commensurability is the stated limitation; a controlled fidelity-ablation is an explicit
out-of-Phase-6 stretch.

## Real-worm validation

The substrate's biological fidelity is anchored externally at the behaviour level (the worm's own
output, public literature, no neuron-identity mapping):

- **Chemotaxis ([035](035-realworm-chemotaxis-validation.md)) — both documented strategies reproduced,
  strongly.** Klinokinesis (turn-rate elevated down-gradient, within the Pierce-Shimomura range for
  sharp reorientations) + weathervane (curving toward the gradient), n=8 × MLP + connectome, with a
  derivative-sensing specificity control establishing the weathervane as sensor-driven (a double
  dissociation). **This satisfies Gate-3 G3.d.**
- **Thermotaxis ([036](036-realworm-thermotaxis-validation.md)) — partial (non-gating enrichment).**
  The weathervane is reproduced weakly but robustly and is sensor-driven; klinokinesis is absent — a
  behavioural-difference finding (see T9.3). Broadens the validation across modalities honestly.

## Gate-3 decision: **GO**

| Sub-criterion | Verdict | Evidence |
|---|---|---|
| **G3.a** — all 6 MUST integrated-C3 cells at the statistical bar (n≥8, per-behaviour sub-metrics) | ✅ | 029 |
| **G3.b** — connectome lands in the ranking with a clear per-behaviour wins/ties/loses verdict | ✅ | 029 (5th; evasion-lag) + 034 (degree-statistics) |
| **G3.c** — env-upgrade cross-regime comparison shipped (qualitative, non-commensurability documented) | ✅ | 025 vs 029 (T9.4) |
| **G3.d** — real-worm validation with quantitative agreement + CIs | ✅ | 035 (+ 036 enrichment) |

All four sub-criteria satisfied; no MUST cell STOPped. **Gate 3 → GO, closing Phase 6a.** Phase 6b
(L3 NEAT) proceeds under `phase6b-tracking`, gated on GPU + env-vectorisation; Phase 7 never gates on
it. This decision is recorded here (the Phase-6a synthesis logbook).

## Phase-7 trigger recommendation + 6b scheduling (T9.5)

Best-supported Phase-7 priorities, given the Phase-6 evidence:

1. **L4 biologically-plausible plasticity — the strongest-supported next step.** Three independent
   Phase-6 findings converge on it: (a) the connectome's evasion-lag under PPO ("is PPO the wrong
   learning rule?"), (b) the memory axis is resolvable-but-undemanded (a faithful-slow-memory task),
   and (c) the thermotaxis behaviour-difference (state-independent-std steering, migrate-and-park) —
   all are *learning-rule / faithful-dynamics* questions, which is exactly L4's remit.
2. ***P. pacificus* transfer / publication / collaboration** — supported by the connectome-degeneracy
   result (034) + the behaviour-level validation (035): the honest "degree-statistics, not wiring,
   under gradient descent" framing is a citable, defensible contribution (Beiran & Litwin-Kumar
   lineage) that does not over-claim.
3. **Phase 6b (NEAT) scheduling** — schedulable into Phase 7's early, CPU-heavy L4-software window;
   the **binding constraint is environment throughput (vectorisation), not the GPU**. Recommend
   settling the env-vectorisation decision before committing 6b compute.

## Limitations

- **Wired-chemical-synapse-only** connectome (no peptidergic layer; fixed gap junctions) — the primary
  scope caveat.
- **PPO-family learning only** — L4 plasticity deferred to Phase 7; the connectome result is
  specifically a *gradient-descent-weight-search* result.
- **Non-commensurable cross-regime comparison** — qualitative only.
- **Thermotaxis validation partial**; predator/mechanosensation behavioural validation not run (a
  documented future arm — a stimulus-response paradigm needing new metrics beyond the bias-curve
  machinery; more naturally Phase-7).
- **Connectome-structure controls**: the rewired-null is in hand; the learnable-gap-junction variant
  is deferred (not raised by this synthesis, since the degree-statistics result settles the
  structure-vs-statistics question).

## Conclusions

- **Phase 6a is complete; Gate 3 is GO.** The layered platform (L0–L2), the three-behaviour comparison,
  the connectome ranking with its credibility control, and the real-worm validation are all in hand
  with evidence.
- **The connectome ranks 5th of 6 under PPO, and that ranking is a degree-statistics — not a wiring —
  result.** Reported honestly as the motivating hypothesis for Phase-7 L4, not a dead end.
- **The comparison resolves working memory, but the naturalistic behaviours don't demand it** — which
  explains the memoryless MLP's win and motivates faithful-slow-memory work.
- **The substrate's chemotaxis is biologically validated** (both strategies), thermotaxis partially —
  anchoring the fidelity claim externally.
- **Phase 6b (NEAT) is deferred** to its own tranche; Phase 7 L4 is the best-supported next step.
