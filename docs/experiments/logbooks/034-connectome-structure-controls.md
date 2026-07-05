# 034: Connectome Degree-Preserving Rewired-Null Control — Wiring or Degree Statistics? (T7 / Phase 6a)

**Status**: completed — **degree-statistics**. The wild-type *C. elegans* connectome is statistically
**indistinguishable** from degree-preserving rewired nulls on the continuous integrated-C3 cell (n=8
paired seeds; paired delta wild−rewired **d=−3.28, CI[−8.56, +1.61], BH-FDR q=0.770** — if anything
the rewired-null is nominally higher). So the connectome's 5th-of-6 standing under PPO weight search
([Logbook 029](029-continuous-architecture-ranking.md)) is a property of its connectivity
**statistics** (degree / sparsity), **not** its specific wiring: a degree-matched random rewiring
performs the same. The credibility control the connectome half of the 6a synthesis needed — it
delivers a citable answer. The verdict is **robust across axes**: a follow-up efficiency re-analysis
(learning-curve AUC + episodes-to-competence on the same paired panel) also finds no wild-type
advantage (n=8, all BH-FDR **q ≥ 0.36**), so the specific wiring buys neither a higher plateau **nor
faster learning** than a degree-matched rewiring.

**Branch**: `openspec/add-connectome-structure-controls`.

**Date**: 2026-07-05.

**OpenSpec change**: `add-connectome-structure-controls` (rewiring utility + `wiring` brain option +
control harness; committed §1–§4, byte-identical when `wiring: wild_type`).

## Objective

The continuous-substrate architecture ranking ([029](029-continuous-architecture-ranking.md)) placed
the wild-type connectome **5th of 6** — beaten on all three behaviours by a plain MLP. For that result
to be credible in the Phase-6a synthesis, one question must be answered: is the connectome's standing a
property of its **specific *C. elegans* wiring**, or merely of its **degree / sparsity statistics**
(any graph with the same degree sequence would rank the same)? The standard control is a
**degree-preserving rewired null** (Dhiman 2026): compare the wild-type against rewired graphs that
keep every neuron's in/out degree exactly but scramble which neurons connect. A naive random-rewiring
null is too weak — it destroys the degree sequence and tests a less interesting hypothesis.

## Background

The roadmap already frames the interpretive stakes: Lappalainen et al. 2024 (connectome-constrained
fly-visual network recovers single-neuron responses) is the positive precedent; Beiran &
Litwin-Kumar 2025 is the counter-weight — a connectome often **under-constrains** dynamics, so many
functionally-distinct weight solutions fit the same wiring. Under PPO weight search (which re-optimises
the chemical weights freely on whatever topology it is given), that degeneracy predicts the specific
wiring may not matter beyond its degree statistics. This control tests it directly.

## Hypothesis

If the specific *C. elegans* wiring is genuinely better-than-degree-matched-random for these
behaviours, the wild-type connectome should **significantly beat** its degree-preserving rewirings
under matched initialisation + budget. If instead it is the degree statistics that set performance,
the two are indistinguishable.

## Method

### Mechanism (committed §1–§4, byte-identical when `wiring: wild_type`)

- **Rewiring utility** (`connectome/rewiring.py`) — a seeded degree-preserving **double-edge-swap**:
  directed (configuration-model) for chemical synapses, undirected for gap junctions, rejecting
  self-loops / duplicates. Preserves every neuron's in/out degree (chemical) and degree (gap) by
  construction; no graph dependency (hand-rolled on the seeded numpy RNG). Validated on the real
  connectome: **91% of chemical edges moved**, degrees exactly preserved, 0.1s, no low-acceptance
  warning at `swaps_per_edge = 10`.
- **Wiring option** on `ConnectomePPOBrain` — `wiring: wild_type | rewired_degree_preserving` +
  `rewire_seed`. The transform runs at the load→topology seam on a **dedicated** RNG (independent of
  the weight-init RNG), so the `w_chem` init draws land at the same RNG state as the wild-type run for
  the same seed — **matched initialisation**. `wild_type` is byte-identical to the pre-change brain
  (guarded by a test reconstructing the mask from the loaded Cook adjacency); only which neurons
  connect changes, so per-post fan-in (hence the strict-mask scale and gap normalisation) is intact.
- **Control harness** (`scripts/analysis/connectome_structure_controls.py`) — reuses the committed 029
  ranked metric (`t7_continuous_ranking._plateau_tail`) + the paired-seed Wilcoxon / 80% bootstrap CI
  / BH-FDR layer; pre-registered verdict: **specific-wiring** (wild-type > rewired, q\<0.05, positive
  delta) vs **degree-statistics** (CI spans 0).

### Panel

Both arms **re-run in one fresh panel** (identical code version + exact seed pairing), n=8 paired
seeds, 6000 episodes, headless, parallelised — the **same PPO recipe / budget as the 029
`connectomeppo` integrated-C3 cell** (only the `wiring` flag differs). Ranked metric = plateau-tail
(final-quarter) full-clear success %.

## Results

### Verdict: DEGREE-STATISTICS (n=8, 6000 episodes)

| arm | mean success | per-seed |
|-----|-----|-----|
| **wild_type** | **52.78%** | 57.2, 45.1, 53.7, 51.6, 51.6, 68.3, 35.3, 59.4 |
| **rewired_null** | **56.06%** | 46.5, 65.5, 56.3, 61.9, 55.8, 65.3, 50.8, 46.5 |

Paired delta (wild-type − rewired): **d = −3.28, CI[−8.56, +1.61] (spans zero), BH-FDR q = 0.770.**
The two arms are statistically indistinguishable — if anything the rewired-null is nominally *higher*.
Wild-type reproduces its 029 plateau (52.78% ≈ the ~52% 029 result — recipe match).

### The smoke → panel reversal

The single-seed smoke (seed 1) read wild-type 57.2% > rewired 46.5% (~11 pt) — superficially "specific
wiring". At n=8 the direction **reverses and vanishes**. Seed 1 was a favourable draw for wild-type
(its second-best seed) against a weak rewired-null draw; across 8 paired seeds it washes into noise —
the same connectome seed-variance 029 flagged (wild-type alone spans 35–68% per seed). A live
demonstration of why n≥8, not n=1, is the bar for this arm.

### Efficiency axis: same verdict (n=8, whole-run learning curves)

The ranked metric above is the final-quarter plateau. A natural follow-up objection: even if the
*plateau* is degree-statistics, does the specific wiring help the connectome **learn faster** — a
sample-efficiency / inductive-bias advantage the peak metric would miss? Reading the **whole**
per-episode series (not just the tail) from the same paired panel forecloses it. No efficiency metric
shows a significant wild-type advantage:

| metric (n=8 paired) | wild-type | rewired-null | delta (+ = wild better/faster) | BH-FDR q |
|---|---|---|---|---|
| learning-curve AUC, full-clear success | 0.46 | 0.44 | +0.022 (wild 6/8) | 0.365 |
| learning-curve AUC, foods/episode | 7.41 | 7.24 | +0.165 (wild 5/8) | 0.365 |
| episodes → 30% rolling success | 685 | 967 | +283 (wild 3/8) | 0.422 |
| episodes → 90% of own foods plateau | 1088 | 1538 | +450 (wild 6/8) | 0.365 |

The point estimates nominally lean wild-type on three of four metrics (the *reverse* nominal lean to
the peak metric's −3.28), but every one is non-significant at n=8 (all q ≥ 0.36) with mixed per-seed
direction — the same connectome seed-variance that reversed the single-seed smoke. **The efficiency
axis agrees with the peak axis: degree-statistics, not wiring.** Cross-arm context from
[029](029-continuous-architecture-ranking.md): the connectome is not faster-learning than the other
arms either — the MLP leads on learning-curve AUC and time-to-competence just as it leads on peak, so
there is no "structural prior → faster learning" story on any comparison run.

## Analysis

**The connectome's ranking is a degree-statistics result, not a wiring result.** Under PPO weight
search — which re-optimises the chemical weights freely on whatever topology it is handed — the
specific *C. elegans* wiring confers **no advantage** over degree-matched alternatives on this
continuous multi-objective task. This is exactly the Beiran & Litwin-Kumar 2025 degeneracy prediction:
the connectome under-constrains the trained dynamics, so a degree-matched rewiring reaches the same
performance with a different weight solution.

**What this does not say.** It does **not** mean the worm's wiring is functionless. Its value in the
animal presumably lives in the **evolved fixed weights** + the specific input/output routing that
biology tunes — not in providing a better substrate for gradient-descent weight search. The control
isolates the *topology-as-substrate-for-PPO* question, and on that question the answer is clear: degree
statistics, not wiring. This is the credible framing the 6a-synthesis connectome section needs — it
forecloses the "did the specific wiring drive the 5th-place result?" objection with a pre-registered
null.

## Conclusions

- **Degree-statistics verdict.** The wild-type connectome is indistinguishable from its
  degree-preserving rewirings (n=8, q=0.770, CI spans 0); its 029 standing reflects connectivity
  statistics, not the specific wiring.
- **The null holds on the efficiency axis too.** No learning-curve AUC or time-to-competence metric
  separates wild-type from the rewired-null (n=8, all BH-FDR q ≥ 0.36) — the wiring buys neither a
  higher plateau nor faster learning, so the credibility control forecloses the inductive-bias/
  sample-efficiency reframe as well as the peak one.
- **The single-seed smoke reversed at n=8** — a clean caution against reading connectome results off
  one seed.
- The **rewiring mechanism is committed, tested, and byte-identical when off** — a reusable
  connectome-control capability.

## Limitations

- **PPO weight search re-optimises the chemical weights**, so this tests the wiring as a *substrate for
  gradient-descent weight search*, not the worm's evolved-weight function. A fixed-weight or
  plasticity-based regime (Phase 7 L4) could give a different answer and is the faithful complement.
- **Single task** (the continuous integrated-C3 cell); the result is for this multi-objective
  foraging/predator/thermotaxis demand, not a universal statement.
- The primary null **scrambles both edge types** (chemical + gap, the tracker's "handled
  consistently"); a chemical-only sensitivity variant was not run (recorded as an option).
- The **learnable-gap-junction control** (`T7.controls.learnable_gj`, the frozen-electrical-synapse
  fairness question) is a tracked **fast-follow**, deferred from this change.
- The **efficiency metrics** (learning-curve AUC + threshold-crossing) use a fixed 6000-episode
  horizon and a 200-episode rolling window; runs that never cross a threshold are right-censored at the
  horizon. The four metrics are BH-FDR-corrected together; they are a follow-up re-analysis of the
  existing panel, not a re-run.

## Next Steps

- [ ] **Learnable-gap-junction control** (`T7.controls.learnable_gj`) — the deferred fast-follow;
  revisit if the 6a-synthesis review raises the frozen-electrical-synapse question.
- [ ] **6a synthesis (T9a)** rolls this verdict into the connectome section (T9.2 — "document the
  connectome ranking honestly": 5th of 6, and that standing is degree-statistics not wiring — on both
  the peak and learning-efficiency axes).
- [ ] **Real-worm behavioural-chemotaxis validation** (`T7.validation`) — the remaining Phase-6a gate
  (Gate 3 G3.d) before the tranche closes.

## Data References

- Mechanism + harness: the `add-connectome-structure-controls` change — `connectome/rewiring.py`, the
  `wiring` / `rewire_seed` options on `brain/arch/connectome_ppo.py`, the rewired-null config
  (`configs/scenarios/foraging_predator_thermal/connectomeppo_small_continuous2d_combined_klinotaxis_rewired_null.yml`),
  the harness (`scripts/analysis/connectome_structure_controls.py`), and their tests.
- Supporting analysis (committed): [supporting/034-connectome-structure-controls/](supporting/034-connectome-structure-controls/details.md)
  — the `controls.json` summary + the per-seed CSV + rewiring/matched-init forensics.
- Efficiency-axis follow-up: the companion harness (`scripts/analysis/connectome_structure_efficiency.py`)
  reading the **same** paired panel via `t7-connectome-controls/panel_n8/_manifest.txt`; summary committed at
  [supporting/034-connectome-structure-controls/efficiency.json](supporting/034-connectome-structure-controls/efficiency.json).
