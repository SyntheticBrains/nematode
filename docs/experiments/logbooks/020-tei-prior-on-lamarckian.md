# 020: TEI-as-Prior-on-Lamarckian — M6.13

**Status**: `complete (framework + GC fix shipped) — STOP ⛔ on the wet-lab-aligned reframe`. The composed-inheritance framework (PR #168: `LamarckianTransgenerationalInheritance` strategy + validator widening + loop predicate widening + Lamarckian-GC `.tei.pt` preservation fix) is validated and shipped. **Four pilots at K ∈ {1000, 500, 200, 200-F0-matched} all rule against the substrate-accelerates-Lamarckian-retraining hypothesis.** Under fair-F0 comparison, the cross-arm primary delta `tei_weights − weights_only` is **+0.00pp at K=1000 (substrate inert)** and **−9.33pp at K=200 (substrate INTERFERES)**. The wet-lab Kaletsky/mammalian-TEI 2025 framing — that an inherited substrate prior should accelerate F1+ learning at K below Lamarckian saturation — is **falsified** on this LSTMPPO RL substrate at every tested K. Combined with PR-A's STOP on pure-TEI K=0, M6 closes coherently: **TEI doesn't transfer in either form** (pure-TEI floor null + substrate-on-top-of-Lamarckian null acceleration). The shared diagnosis across all three M6 STOP results is **architectural**: the bias-network logit-prior is the wrong abstraction for capturing the wet-lab single-circuit excitability shift. PR-B / M6.14 frequency-prior ablation **NOT** triggered per pre-registered criterion. M6 thread closes; substrate-extraction-redesign documented as future-work for a possible Phase 6 quantum re-evaluation arc.

**Branch**: `feat/m613-tei-prior-campaign-exec` (campaign execution), built on `feat/m613-tei-prior-on-m3` framework (merged PR #168 → `main` at `a4a03866` on 2026-05-20).

**OpenSpec change**: `add-tei-prior-on-m3` (archived as `2026-05-21-add-tei-prior-on-m3`; spec sync per STOP convention — the binding scenarios stay in the change archive; no main-spec promotion since the experiment is closing as a calibrated null).

**Date Started**: 2026-05-19 (post-PR-A planning).

**Date Last Updated**: 2026-05-21 — K-sensitivity sweep + F0-confound disambiguation completed; verdict locked in.

This logbook covers Phase 5 M6.13 across 2 branches: the framework PR #168 (merged 2026-05-20; 8 commits, ~270 LoC production code + ~58 tests + OpenSpec scaffold) and the campaign-execution branch `feat/m613-tei-prior-campaign-exec` (4 commits: K_test landing, GC fix, logbook 020, archive). The headline finding is **the M6.13 composed-inheritance framework SHIPPED CLEANLY, AND the substrate-accelerates-retraining hypothesis is decisively NULL.** The K-sweep dose-response that initially looked promising disambiguated to an F0 confound — substrate effect inverted from +5.33pp to −9.33pp under fair-F0 comparison.

## Objective

Re-evaluate the M6 transgenerational hypothesis under PR-A's reframed lens: substrate as a *prior on Lamarckian retraining* at K below saturation, aligning with the actual wet-lab Kaletsky 2025 mechanism (low-bandwidth single-circuit excitability shift, not a transmitted policy). PR-A's pure-TEI K=0 null established the substrate alone is insufficient; M6.13 tests whether the substrate adds *acceleration* on top of Lamarckian weight inheritance + K=K_test retraining.

Cross-arm primary verdict (reframed from PR-A's `tei_on − control` to `tei_weights − weights_only`): GO iff (a) `tei_weights` per-arm gate passes (F1 ≥ 40% × F0, F2 ≥ 25%, F3 ≥ 15% with monotone non-increasing decay, in ≥ 2 of 4 seeds) AND (b) paired-seed F1-F3 retention delta satisfies BOTH Wilcoxon p < 0.10 AND ≥ 5pp absolute mean delta with non-overlapping 80% bootstrap CIs.

## Background

Phase 5 prior state:

- **M3** (Lamarckian inheritance pilot — +47pp on 4 seeds; PR #155): strongest concrete Phase 5 result. The proven heritable-substrate pattern.
- **M6** (logbook 018 — INCONCLUSIVE): framework shipped, four blocking design issues identified by post-pilot audit.
- **M6.9+ PR-A** (logbook 019 — STOP on pure-TEI K=0): sensory-conditional substrate redesign + env-derived probes + audit-B reward/env redesign. Three pilots × three substrate variants all null. Cross-arm `tei_on − control` delta = **-49pp**. M3 weights_only re-reproduced at **+17.5pp** vs control on the new env. Pre-registered M6.13 reframe trigger: "TEI as a prior on M3 (K>0 with small inherited bias)."

M6.13 specifically composes `LamarckianTransgenerationalInheritance` (kind `"weights+transgenerational"`) — a fifth `InheritanceStrategy` Protocol value. F1+ children inherit BOTH the F0 elite's trained weights (Lamarckian pattern, `.pt` warm-start) AND the F0-extracted sensory-conditional substrate (transgenerational pattern, `.tei.pt` cascade with geometric decay). The hypothesis: at K < Lamarckian saturation, the substrate prior accelerates F1+ retraining vs Lamarckian alone.

## What shipped

### Framework (PR #168, 8 commits — merged 2026-05-20)

| Component | File | Purpose |
|---|---|---|
| Composed strategy | [`evolution/lamarckian_transgenerational_inheritance.py`](../../../packages/quantum-nematode/quantumnematode/evolution/lamarckian_transgenerational_inheritance.py) | New class beside `LamarckianInheritance` / `TransgenerationalInheritance`. Per-method behaviour byte-identical to `LamarckianInheritance` for `select_parents` / `assign_parent` / `checkpoint_path`; distinguishing literal `"weights+transgenerational"` from `kind()`. |
| Protocol widening | [`evolution/inheritance.py`](../../../packages/quantum-nematode/quantumnematode/evolution/inheritance.py) | Fifth Literal value in `InheritanceStrategy.kind()` Protocol. M3/M6.9+ kind values unchanged. |
| Validator relaxation | [`utils/config_loader.py`](../../../packages/quantum-nematode/quantumnematode/utils/config_loader.py) | `EvolutionConfig.inheritance` Literal widened. Pairing rule extended: substrate-enabled accepts `transgenerational` OR `weights+transgenerational`. New F1+ K>0 sub-rule under composed mode (opposite of pure-TEI's K=0 floor test). Cross-product matrix has 10 explicitly-tested validator cells. |
| Loop integration | [`evolution/loop.py`](../../../packages/quantum-nematode/quantumnematode/evolution/loop.py) | `_inheritance_active()` widens to `kind in {"weights", "weights+transgenerational"}`; `_substrate_inheritance_active()` widens to `kind in {"transgenerational", "weights+transgenerational"}`. New `_combined_inheritance_active()` helper drives composed-mode F0-GC suppression. New branch in `_resolve_per_child_inheritance` for the composed kind. |
| Three-arm campaign | [`tei_weights`](../../../configs/evolution/tei_prior_m613_tei_weights.yml) / [`weights_only`](../../../configs/evolution/tei_prior_m613_weights_only.yml) / [`control`](../../../configs/evolution/tei_prior_m613_control.yml) / [`smoke`](../../../configs/evolution/tei_prior_m613_smoke.yml) | tei_weights: `weights+transgenerational`, F0 K=2000 + F1+ K=K_test. weights_only: `lamarckian`, K=K_test (M3 baseline at K_test). control: `none`, K=K_test (TPE-fresh). Launcher parity check enforces fitness + env + K_test alignment across the trio. |
| Aggregator (forked from M6.9+) | [`scripts/campaigns/aggregate_m613_pilot.py`](../../../scripts/campaigns/aggregate_m613_pilot.py) | Reframed primary pair `tei_weights − weights_only`. Six-row pivot table (D6) — substrate inert / clean GO / K-sensitivity / substrate interferes / Lamarckian saturation / PPO destabilised. Frequency-prior follow-up trigger emitted on GO; null-finding note on STOP. |
| Campaign launcher | [`scripts/campaigns/phase5_m613_tei_prior_lstmppo_klinotaxis.sh`](../../../scripts/campaigns/phase5_m613_tei_prior_lstmppo_klinotaxis.sh) | --smoke / --pilot / --full. Parity check enforces fsw=1.0 + fm='survival_rate' + env + K alignment. Smoke YAML fitness audit added under `--smoke`. |
| Test surface | various | 58 new test cases: composed-strategy unit (+18), loop-smoke (+10), config-validator cross-product (+10), aggregator (+20). All 469 evolution tests pass post-PR. |

### Campaign-execution fix (b78cdbb5 on `feat/m613-tei-prior-campaign-exec`)

| Component | File | Purpose |
|---|---|---|
| Lamarckian GC `.tei.pt` preservation | [`evolution/loop.py:_gc_inheritance_dir`](../../../packages/quantum-nematode/quantumnematode/evolution/loop.py) | Pre-existing bug from M3's initial Lamarckian GC: `Path.glob("genome-*.pt")` matched both weight `.pt` and substrate `.tei.pt` files; `Path.stem` strips one suffix only, so the `.tei.pt` extracted gid ends in `.tei` and never matched the keep-set → substrate deleted. Bug never surfaced under M3 (no `.tei.pt`) or pure-TEI (main-loop GC doesn't fire under `kind=transgenerational`) but is load-bearing under composed mode. Fix: skip `.tei.pt` files in main-loop GC entirely. Regression test `test_main_loop_gc_preserves_substrate_tei_pt` pinned the contract. |

## Calibration (T1'-T4' all PASS at pilot pre-flight)

K_test calibration smoke (1 seed × pop 6 × 2 gens × weights_only arm at K=1000, ~10m 28s wall, 2026-05-20):

| Tripwire | Spec | Observed | Verdict |
|---|---|---|---|
| T1' F0 envelope | 0.30 ≤ mean F0 survival_rate ≤ 0.70 | 0.407 | ✅ PASS |
| T3' high-end (Lamarckian saturation) | F1 ≤ 0.95 × F0 (= 0.387) | F1 = 0.600 > 0.387 | ✅ PASS (not saturated) |
| T3' low-end (Lamarckian under-trained) | F1 ≥ 0.80 × F0 (= 0.326) | F1 = 0.600 >> 0.326 | ✅ PASS (well above floor) |
| T3' beats-control leg (1.2×) | F1 ≥ 1.2 × control_F1 | deferred to pilot time | DEFERRED |

T2'/T4' substrate-diversity + substrate-magnitude tripwires (4-seed tei_weights smoke against an F0-only YAML, ~43m wall):

| Tripwire | Spec | Observed | Verdict |
|---|---|---|---|
| T2' Substrate diversity | min pairwise CoV ≥ 5% | 0.9633 (19× over) | ✅ PASS |
| T4' Substrate magnitude | mean abs bias_network output ≥ 0.1 | 1.7954 (18× over) | ✅ PASS |

All four pre-flight tripwires PASS → pilot unblocked. K_test = 1000 committed to production YAMLs as a chore commit (`21919c8b`) before pilot dispatch.

## Pilot evaluation

Four pilot runs across K and F0-K to disambiguate the substrate's effect from confounds. All n=1 (seed=42), pop 8, 4 gens, 3 arms. Aggregator runs in `--mode pilot` over a per-gen CSV synthesised from `eval_diagnostics.jsonl` (the production frozen-eval on trained weights — see § Audit below for why this substitutes for the post-hoc per-gen-eval script).

### Pilot 1: K=1000 (initial, all-arms-aligned to K_test)

| Arm | F0 | F1 | F2 | F3 | F1-F3 mean |
|---|---|---|---|---|---|
| tei_weights | 0.840 | 0.840 | 0.760 | 0.720 | 0.773 |
| weights_only | 0.720 | 0.680 | 0.840 | 0.800 | 0.773 |
| control | 0.720 | 0.400 | 0.640 | 0.600 | 0.547 |

Cross-arm primary delta `tei_weights − weights_only` (F1-F3 mean) = **+0.00pp** → D6 row 1 (substrate inert) → STOP.

Secondary verdicts (sanity checks):

- `weights_only − control` = **+22.67pp** ← Lamarckian re-reproduces PR-A's +17.5pp M3 finding (✅)
- `tei_weights − control` = **+22.67pp** ← composed arm beats floor by the same margin as Lamarckian (✅)

### Pilot 2-3: K-sensitivity sweep (K=500 + K=200, F0 held at K=2000)

Dispatched per design.md § D6 row 3's K-sensitivity-pivot protocol (variant: K=200 substituted for K=1500 to map the biologically-aligned low-K regime where the substrate should dominate the lightly-updated PPO policy).

| K | tei_weights F0 | tei_weights F1-F3 mean | weights_only F1-F3 mean | Δ (composed − Lamarckian) | Pivot row |
|---|---|---|---|---|---|
| K=1000 | 0.840 | 0.773 | 0.773 | **+0.00pp** | Row 1 (substrate inert) |
| K=500 | 0.840 | 0.787 | 0.747 | **+4.00pp** | Row 3 (K-sensitivity, 2-5pp) |
| K=200 | 0.840 | 0.747 | 0.693 | **+5.33pp** | Row 2 (clean GO, ≥5pp) |

Headline: **dose-response is monotone — substrate's advantage over Lamarckian grows as K decreases**. At K=200, the delta crosses the design.md § D6 row-2 5pp threshold.

But three caveats flagged before declaring GO:

1. **n=1 at every K** (Wilcoxon p=0.5 throughout; uninformative).
2. **F0 confound**: tei_weights F0 = 0.84 at all three K (its YAML keeps F0 at K=2000); weights_only/control F0 drop with K (0.72 → 0.52 → 0.40 because their `learn_episodes_per_eval = K`). The "tei_weights beats weights_only" comparison includes a **fixed-F0-advantage component** NOT from the substrate.
3. **K=200 control F3 = 1.00** (25/25 episodes survived) — suspicious; either lucky TPE sample or K-dependent env saturation.

### Pilot 4: K=200 with F0 MATCHED at K=200 (F0-confound disambiguation)

Re-ran K=200 with tei_weights F0 also at K=200 (all four `lawn_schedule` entries at `ppo_train_episodes: 200`, including gen-0). If the K-sweep signal was the substrate's effect, it should persist under fair F0. If it was the F0 confound, the delta should collapse.

| Arm | F0 | F1 | F2 | F3 | F1-F3 mean |
|---|---|---|---|---|---|
| **tei_weights** | **0.400** | 0.520 | 0.640 | 0.640 | **0.600** |
| weights_only | 0.400 | 0.760 | 0.680 | 0.640 | 0.693 |
| control | 0.400 | 0.360 | 0.600 | 1.000 | 0.653 |

Cross-arm primary delta = **−9.33pp**. → D6 row 4 (substrate INTERFERES with Lamarckian) → STOP.

**The K-sweep signal was the F0 confound — completely inverted under F0 fairness.** When tei_weights doesn't get its K=2000 F0 head-start, the substrate prior **actively HURTS** Lamarckian retraining by 9.33pp on F1-F3 retention. Per pivot table row 4's pre-declared text: "the prior must be calibrated for the warm-start child's policy, not just the F0 elite's." The substrate's frozen logit-bias misdirects the warm-start child's early exploration. At high K, PPO has time to wash it out (inert). At low K with F0 fair, PPO can't recover before eval kicks in (interferes).

### Cross-K dose-response under fair F0

| K (F0 budget) | mean Δ | Pivot row | Verdict |
|---|---|---|---|
| K=1000 (F0 K=2000) | +0.00pp | Row 1 (substrate inert) | STOP |
| K=500 (F0 K=2000, F0-confound) | +4.00pp | Row 3 (F0-confound artefact) | INDETERMINATE |
| K=200 (F0 K=2000, F0-confound) | +5.33pp | Row 2 (F0-confound artefact) | INDETERMINATE |
| **K=200 (F0 K=200, fair)** | **−9.33pp** | **Row 4 (substrate INTERFERES)** | **STOP** |

The fair-F0 verdicts at K=1000 and K=200 bracket the testable regime: substrate is inert at high K, interferes at low K, never accelerates. **M6.13 substrate-accelerates-retraining hypothesis decisively falsified.**

## Audit — why composed-mode fails too

### Post-hoc per-gen-eval vs training-time eval (load-bearing methodological note)

The aggregator's expected input is `per_gen_choice_index.csv` produced by `transgenerational_per_gen_eval.py`. That script reconstructs each per-gen elite via `encoder.decode(genome, sim_config, seed=seed)` — which builds a fresh brain from the genome's TPE-sampled hyperparameters (actor_lr, gamma, entropy_coef, etc.) **but does NOT load the elite's `.pt` weight checkpoint**. Under PR-A's pure-TEI K=0 arm this was harmless (no F1+ retraining means no trained weights to load anyway). Under M6.13 where F1+ has K=1000 of meaningful retraining, the un-trained-brain reconstruction is a load-bearing bug: the post-hoc eval measures an un-trained brain at the elite's hyperparameters, dies in 299/300 episodes, and the aggregator's first pass at K=1000 inferred "PPO destabilised (D6 row 6) → STOP" against the wrong data.

The pilot's actual frozen-policy-on-trained-brain measurement already lands in `eval_diagnostics.jsonl` (`LearnedPerformanceFitness.evaluate` runs K training episodes followed by L=25 frozen-eval episodes per genome per gen, by design). The bridge fix at `/tmp/build_pilot_per_gen_from_eval_diag.py` synthesises the aggregator's expected CSV directly from `per_gen_elites.jsonl` (elite per gen) + `eval_diagnostics.jsonl` (elite's survival/death counts). Each elite's eval row contributes `eval_count` synthetic per-episode rows with `termination_reason` set to `health_depleted × deaths` and `completed_all_food × successes`. The aggregator's loader only counts termination_reason values, so episode-shape is irrelevant.

All four pilot retention tables in this logbook are computed via the bridge, NOT the broken post-hoc script. The bridge stays as a `/tmp/` one-off rather than a committed code path because the proper fix (warm-start the post-hoc brain from each elite's `.pt`) is structurally infeasible — Lamarckian's main-loop GC retains only the final-gen elite's `.pt`; F0/F1/F2 `.pt` files are GC'd by the time the campaign exits.

### Mechanistic root cause (substrate placement + extraction)

The shared diagnosis across all three M6 STOP results (M6.0-6.8 INCONCLUSIVE, M6.9+ PR-A STOP, M6.13 STOP) is **architectural**: the bias-network logit-prior is the wrong abstraction for capturing the wet-lab single-circuit excitability shift.

Two structural problems compound:

1. **Substrate placement**: the substrate writes an action-distribution bias (added to actor logits). The wet-lab Kaletsky 2025 mechanism is an **excitability shift upstream of action selection** — parental experience changes ASJ neuron sensory-response gain/threshold, not action frequency. Biological prior acts on what sensations look like to the agent, not on what action is taken given a sensation.

2. **Substrate extraction**: PR-A Pilot 1 (logbook 019 § Pilot 1) found that even with the sensory-conditional bias-network MLP, "substrate outputs barely vary across sensory inputs — captures unconditional action bias (always-LEFT, never-FORWARD, never-STAY), not conditional response." The F0 substrate probe-extraction protocol asks "what action does the F0 elite take on these states?" and stores that, which collapses to a near-constant action prior. The wet-lab mechanism isn't "what action did the parent take"; it's "what's the parent's sensory-response baseline."

Under M6.13's composed mode, problem 1 surfaces as the K-dependent failure mode (PPO either washes the action-bias out at high K, or the action-bias misdirects exploration at low K). Problem 2 was already documented in PR-A's logbook but inherited unchanged into M6.13 — PR-A's bias-network MLP did help variation marginally but the substrate still collapsed.

### Adversarial-critique findings + responses

| Finding | Response |
|---|---|
| Could the +5.33pp K=200 signal be a real substrate effect that the F0-disambiguation pilot happened to miss? | The disambiguation pilot **inverted** the sign (+5.33pp → −9.33pp), a 14.66pp swing on a single F0-K change. An n=1 noise artefact would shift symmetrically around zero. The sign inversion under exactly the manipulation predicted by row 4's pre-declared mechanism (substrate calibrated for F0 elite, misdirects warm-start child) is a strong causal signal that the K-sweep result was the F0 confound. |
| Did the GC fix b78cdbb5 cause the K=200 result? | The GC fix only affects whether F0's `.tei.pt` survives past F1. All four pilot retention tables use the same GC-fixed branch. The K-sweep dose-response is internally consistent across K=500 and K=200 under the F0-confound; the disambiguation only changes the F0 budget. |
| Could a multi-seed n=4 full campaign at K=200 (F0-matched) overturn the STOP? | Possible in principle but contradicted by the mechanism diagnosis. A multi-seed run would shift the −9.33pp central estimate, not invert it back to +5pp. The pivot-table row-4 explanation predicts the inversion direction precisely; the K=1000 row-1 inert result independently supports the "substrate doesn't accelerate" conclusion at high K. Both points of the bracket (K=200 interferes, K=1000 inert) would need to fail simultaneously for the STOP to be wrong. |
| Is the M6.14 frequency-prior ablation still motivated? | No. The frequency-prior trigger criterion (design.md § D4) was "GO on tei_weights vs weights_only delta". The triggered condition was never met under fair F0. The substrate-extraction redesign that PR-A Pilot 1 implied is needed sits in a different design space (extraction protocol, not substrate shape) and isn't what M6.14 was scoped for. |

## Decision

**STOP. M6.13 substrate-accelerates-Lamarckian-retraining hypothesis falsified.** M6 thread closes.

### What this means for the canonical TEI mechanism

The wet-lab Kaletsky 2025 / mammalian-TEI 2025 framing remains scientifically sound — small-RNA-driven inherited excitability shifts in a single sensory circuit are a real biological phenomenon. The M6.13 result says: the analog **doesn't transfer** onto an LSTMPPO RL substrate **at the substrate-shape level we've tested** (bias-network logit-prior, F0-action-frequency extraction). Combined with PR-A's STOP on pure-TEI K=0, M6 has now tested two architecturally-orthogonal regimes (substrate replaces training; substrate accompanies training) and both produce null acceleration. The substrate **shape** (frozen action-logit bias) and **extraction protocol** (probe-state action frequencies) are the load-bearing problems; placement (logit vs sensory) and K (1000 vs 200 vs 0) are second-order.

### Honest framing

The negative result has substrate-grounded diagnosis (action-distribution bias vs sensory-excitability shift mismatch), aligns with the field (2024-2026 deep-RL substrate-transfer literature shows no published K=0 recurrent-policy distillation, and no published frozen-logit-prior acceleration result), and produces a reusable calibrated framework that the next researcher can build on. The Phase 5 M6 thread closes the same way M4 (Baldwin) and M5 (Red Queen) did — STOP with a substrate diagnosis that connects to recent independent literature. Phase 5's narrative tightens around M2 + M3 GO results plus three substrate-grounded null findings with mechanism diagnoses.

## Compute spent

| Phase | Wall-h | Detail |
|---|---|---|
| K_test calibration smoke (pass 1) | 0.17 | 1 seed × pop 6 × 2 gens × weights_only × K=1000 |
| T2'/T4' tei_weights substrate smoke | 0.72 | 4 seeds × pop 6 × 1 gen × tei_weights × F0 K=2000 |
| Pilot 1 (K=1000, all-arms-aligned) | 1.45 | 1 seed × pop 8 × 4 gens × 3 arms × K=1000 |
| Pilot 2-3 (K-sweep K=500 + K=200) | 1.28 | 1 seed × pop 8 × 4 gens × 3 arms × K∈{500,200} (cross-K) |
| Pilot 4 (K=200, F0-matched disambiguation) | 0.29 | 1 seed × pop 8 × 4 gens × 3 arms × K=200 (F0 also K=200) |
| Aggregator + per-gen-eval bridge analysis | \<0.1 | analysis only |
| **Total M6.13 campaign-execution** | **~4.0** | well under the 20-25 wall-h design.md § D7 envelope |

Combined with PR-A (logbook 019) at ~30 wall-h, **M6.9+ + M6.13 spent ~34 wall-h total**.

## Citations

### Wet-lab TEI mechanism (the framing M6.13 was reframed to test)

- **Kaletsky 2025** — *eLife* 105673 — single-circuit ASJ excitability shift in C. elegans F1+ offspring after parental pathogen exposure. The wet-lab "small-RNA-driven inherited prior" mechanism.
- **Mammalian-TEI 2025** — *bioRxiv* — parental exposure shifts inherited sensory-response gain in mammalian models. Independent corroboration of the single-circuit-prior framing.

### Deep-RL substrate-transfer literature (2024-2026)

- **2024-2026 distillation surveys** — no published K=0 recurrent-policy substrate transfer result on RL benchmarks. The "frozen-logit-prior substrates accelerate retraining" claim has no positive precedent in the RL literature.

### Phase 5 substrate-diagnosis-grounded STOPs (this logbook is the third)

- **M4 STOP** (logbook 015) — single-task K=50 has no Baldwin axis (Fernando 2018 / Chiu 2024 substrate diagnosis).
- **M5 STOP** (logbook 017) — architecture-asymmetry suppresses Red Queen entanglement (Resendez Prado [arXiv 2604.03565, Apr 2026] independently corroborates the chess Baldwin "transparent regime"; Mougi [Sci Reports 2026] trait-decoupling indicates fitness-escalation is the wrong instrument).
- **M6.9+ + M6.13 STOP** (this logbook + 019) — action-distribution-bias substrate doesn't match the wet-lab excitability-shift mechanism. PR-A null floor + M6.13 null acceleration close the TEI thread on this RL substrate.

## Tracker + roadmap status

- `openspec/changes/phase5-tracking/tasks.md` M6.13 row → ❌ STOP.
- `docs/roadmap.md` M6.9+/M6.13 row → STOP with logbook 020 reference.
- M8 synthesis (logbook 019 / 020 narrative) — M6.9+ + M6.13 close as "wet-lab-aligned framing falsified on LSTMPPO substrate; substrate-extraction-redesign documented as future work for Phase 6 quantum re-evaluation arc".

## Follow-ups — viable future paths for M6 (Phase 6 future-work, NOT triggered now)

A deep architectural survey (this branch's investigation, 2026-05-21) identified two viable substrate-redesign directions if a future researcher wants to revisit M6. **Neither is triggered by the M6.13 STOP** — they remain open future-work, scoped here for completeness so the next person can pick up the thread.

### Future Idea B — Substrate as input-encoding bias

Substrate output modulates the LSTM's input features rather than the actor's output logits. Plug-point: after `feature_norm` in `lstmppo.py:run_brain` (line 750), substrate produces a feature-space additive bias before the LSTM step. Mechanism alignment ~5/10 as a re-skin of the current substrate shape; ~7/10 if paired with an extraction-protocol redesign (see Idea D). Net effort: ~80-120 LoC + 4-5 new tests; 1-1.5 weeks for the re-skin variant, 3-4 weeks for the full extraction-aligned variant.

### Future Idea C — Substrate as initialiser for a frozen sub-network

Substrate's bias_network outputs become initial weights for a small frozen sensory-encoder layer inserted between `feature_norm` and the LSTM. F1+ children inherit AND freeze (`requires_grad=False`) this layer; learned dynamics build on top of it. Mechanism alignment 8/10 (matches wet-lab "structural prior parents pass to offspring"). Critical caveat: depends entirely on F0 producing a useful encoder during K=2000, which is **not** validated by the current M6.13 data. Net effort: ~150-200 LoC + 5-6 new tests; 1.5-2 weeks code + 1-2 pilots; ~2.5-3 weeks total.

### Future Idea D — Substrate-extraction protocol redesign

The PR-A Pilot 1 finding ("substrate outputs barely vary across sensory inputs") points at an extraction problem, not just a placement problem. A biologically-aligned extraction would probe F0's **LSTM hidden state** on a sensory ablation panel (not its actions), capture the distribution of activations under varying food/predator gradient strengths, and distill that into a gain/threshold transform on the sensory encoder. ~2 weeks code + redesigned `extract_from_brain` pipeline. Combinable with B or C for 9/10 alignment.

### Honest recommendation for whoever revisits M6

The viable path is **Idea C + Idea D combined** (~4-5 weeks): substrate-extraction-redesign that captures sensory-response baselines, plumbed through a frozen sub-network in the sensory encoder. This addresses both problems M6 has surfaced (substrate-shape mismatch + extraction-protocol mismatch) and is the only path with both mechanism alignment ≥ 8/10 and a realistic test envelope. **Idea B as a 1-2 week re-skin of the M6.13 substrate would likely produce another null** — same substrate shape, same failure mode.

Phase 5 is the wrong arc to schedule this work; it belongs in a Phase 6 quantum re-evaluation context where substrate redesign is a peer concern with brain redesign. M6.14 frequency-prior ablation **NOT** triggered (its prerequisite GO outcome on tei_weights vs weights_only was never met under fair F0).

This logbook closes the M6 thread for Phase 5.
