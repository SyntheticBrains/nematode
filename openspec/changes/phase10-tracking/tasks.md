# Tasks: Phase 10 (Fidelity on the Plate — Standard Lawn Assays, Graded Against the Worm) Shipment Tracker

This is the living checklist for Phase 10. The plan is authoritative in `docs/roadmap.md` § Phase 10
(v5.0, 2026-10-23), from the plan review of 2026-10-23 and its adversarial critique: the goal, what the
review established, the deliverables, design decisions D33–D41, exit criteria, risk table, novelty map
and go/no-go. Every Phase 10 milestone PR updates this checklist as part of its diff.

**Status legend**: `[ ]` not started; `[x]` closed (done, or given a status at the synthesis).

**Execution order** (binding where the roadmap says so):

01. H.
02. E + T.
03. B, with N in parallel.
04. B valid.
05. F.0.
06. C.1.
07. C.2.
08. C.3.
09. C.4.
10. W.
11. The synthesis.

Binding: nothing is fitted before B reads valid; nothing is fitted on real data before F.0 clears;
nothing in W registers before C.2 has read.

## H — housekeeping (MUST, first)

- [ ] **H.1 The literature watch re-aimed** at Phase 10's questions (`context.md`, `seeds.toml`; lands with
  this tracker's PR). The item closes when the first digest has run against the new brief.
- [ ] **H.2 The research question re-framed** (D33) in AGENTS.md, the README and the roadmap (lands with
  this tracker's PR; closes on merge).
- [ ] **H.3 Scenario scope recorded**: thermal, predator, multi-agent and oxygen configs frozen, kept and
  tested; thermotaxis named as the second tier. It closes when the docs say so in the scenario index.
- [ ] **H.4 The seed-band table** (design.md), cited by every registration.

## E + T — the worm, its assays and the batched evaluator (MUST)

- [ ] **E.1 One step-duration value** (D35): every step-coupled constant derived from it. These include:

  - modulator decay, body step, the classifier's window and step;
  - the STAM length, the adaptive-sensor α, `membrane_tau_steps`;
  - satiety decay, intake and regrowth;
  - `max_turn_rad`, `max_steps`, and PPO γ.

  The 5-s default stays byte-identical, and a test asserts the derivation. A 1-s point worm configured.

- [ ] **E.2 Locomotor primitives** (D36): run, reversal bout, omega/delta reorientation and pause, with
  kinematics fixed from data (Broekmans 2016; Gray 2005) and timing chosen by the controller. The
  lawn-edge question is decided in its spec.

- [ ] **E.3 Assay protocols**: Scheer's small lawn (40 min); food removal onto a bare plate; an off-food
  start; multi-hour episodes with satiety re-calibrated.

- [ ] **T.1 The batched evaluator**, about 8–12 engineering days: many animals per call, no per-step
  Python, carrying E.2's primitives and E.3's assays. A parity test against the per-step runner, and
  a cost pilot (seeds 5001–5032) that sizes W.

## B — the benchmark (MUST; the gate for everything after)

- [ ] **B.1 Reference re-extraction**: Scheer & Bargmann 2023 from the Zenodo mirror (8310289), with
  positions and cohort IDs, through the restricted unpickler. Calhoun 2014 and Roberts 2016 tracks;
  López-Cruz 2019 targets. Each with its licence, unit convention and provenance.

- [ ] **B.2 Readouts**: a trajectory-based reorientation detector, validated against Calhoun's and
  Roberts's omega flags; the classifier run on tracks downsampled to 5 s; the local-search curve; the
  repaired pirouette quintile ratio (the old K.1).

- [ ] **B.3 Bands and the grading rule** (D34): from batch-matched cohorts with the project's
  classifier; mutant effects relative to the controller's own wild type; composite rules that depend
  on closed-loop contingencies.

- [ ] **B.4 The controls**:

  - positive: cohort-held-out real tracks pass, and held-out real mutant cohorts pass the mutant
    grading;
  - matched negatives (the open-loop Scheer HMM; Poisson-timed primitives) fail;
  - trivial negatives (a random walk, a correlated random walk, the untrained policy) fail.

  **The benchmark is void unless all of these behave.**

- [ ] **B.5 Track import** (WCON or CSV), independent of the simulator, positioned against Tierpsy and
  OWMD feature sets. NeuroSparkSNT and a wider novelty search checked at registration.

## N — the neuromodulator layer from data (MUST)

- [ ] **N.1 Licences confirmed**: Bentley 2016 (CC BY, S1 Dataset) and Ripoll-Sánchez 2023 (unstated on
  the toolbox and repository pages). Vendored only if confirmed; otherwise what stands in is stated.
- [ ] **N.2 The layer built**: receptor expression and producers per neuron, replacing the hand-placed
  four-pathway circuit behind a byte-identical default, with the diffusion-range model chosen in its
  spec.
- [ ] **N.3 Its null family**: receptor-placement nulls, with a D24-style matched-statistics table.

## F.0 — the fitting methods' positive control (MUST)

- [ ] **F.0** parameter recovery at known parameters, for NPE on C.1's model and sep-CMA on C.2's
  encoder (seeds 5401–5464). It reports recovery and spread, and the spread sizes W.

## C — controllers and the fidelity table (MUST)

- [ ] **C.1 The mechanistic baseline**: a stochastic run/reversal switch, a roaming/dwelling state with
  intake and satiety dependence, pirouettes and weathervane, ≤ 30 parameters. Fitted by NPE, with
  priors set without search data; posterior-predictive checks.
- [ ] **C.2 The connectome controller**: on the primitives, with the data-derived neuromodulator layer.
  Its slow variable is stated and its feasibility for a 9-minute decay checked. It is fitted by
  sep-CMA on the benchmark's distance at ≥ 200 animals per evaluation, with the budget pre-registered
  and a plateau criterion.
- [ ] **C.3 The comparator**: the C.2 architecture trained by reward on the same primitives and lawns,
  16 seeds, γ adjusted to the 1-s step.
- [ ] **C.4 The fidelity table**: C.1–C.3 and the negatives on held-out assays and the mutant set
  (tph-1 on Scheer only, pdfr-1, tax-4), under a per-controller encoding table registered before any
  fit, and with C.2 against C.1 as a secondary contrast.

## W — the wiring question on fidelity (MUST, after C.2 reads)

- [ ] **W.1** The nulls: degree-preserving; gap-only on the corrected gap model; boundary-held;
  class-preserving left–right-symmetric; and N.3's receptor-placement nulls. Each has a fresh rewiring
  draw per fit, with parameters counted per arm.
- [ ] **W.2** The registration: the minimum is the held-out distance between two real wild-type cohorts,
  both directions. Draws per arm are sized from F.0 and the cost pilot. Every outcome is named,
  including "the nulls pass too" and "a null beats the wild type".
- [ ] **W.3** The panel, read on graded distance whatever any gate says.

## The synthesis (MUST)

- [ ] **S10** Every criterion assigned one of the five statuses. The publication decision taken (D30's
  destination: the first fidelity result).

## SHOULD

- [ ] Haley 2025's patch acceptance against encounter number.
- [ ] WCON export.

## MAY

- [ ] Thermotaxis on the new primitives (the second tier).
- [ ] Bordering against an O₂ field, with npr-1.
- [ ] A Creamer-dynamics controller graded on the benchmark.
