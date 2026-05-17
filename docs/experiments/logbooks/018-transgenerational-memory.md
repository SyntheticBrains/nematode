# 018: Transgenerational Memory — M6 (TEI substrate + cascade + paired-arm ablation)

**Status**: `complete (framework shipped) — INCONCLUSIVE ⚠️ on the science`. The framework changes (TransgenerationalInheritance strategy + TransgenerationalMemory dataclass + LSTMPPO `tei_prior` actor-logit hook + per-gen `lawn_schedule` loop consumer + F0 substrate-extraction telemetry pipeline + paired-arm aggregator with F0-baseline override + per-gen survival/choice-index evaluator) are validated and shipped. The paired-arm pilot and full campaign ran cleanly across 4 seeds × 4 generations × pop 16 × paired TEI-on/TEI-off arms (~16 wall-h on the choice-index pipeline; substrate cascade applied correctly across 12 F1+ generations × 2 arms). **However, a post-pilot deep-dive audit found four design issues that prevent the data from cleanly answering "does transgenerational memory work on this RL substrate?".** The literal aggregator output under the choice-index gate is STOP (0/4 seeds pass either arm — geometry-dominated metric); under the survival-rate gate with F0 training-time override it is PIVOT for TEI-on (1/4 passes) and STOP for TEI-off (0/4). Neither verdict is load-bearing because the gates were calibrated against a substrate design that cannot, by construction, encode pathogen-conditional avoidance. See § Audit and § Decision below. M6.9+ (the proper TEI re-evaluation) is recorded as the follow-up.

**Branch**: `feat/m6-transgenerational-memory`.

**OpenSpec change**: `add-transgenerational-memory` (archived as `2026-05-17-add-transgenerational-memory`; no spec sync per INCONCLUSIVE convention — the audit-flagged decisions D1/D5/D6 stay in the change archive for reference rather than promoted to main specs).

**Date Started**: 2026-05-10 (substrate work began under commits 1-7; pilot screening began 2026-05-14).

**Date Last Updated**: 2026-05-17 — M6 closed INCONCLUSIVE after post-pilot audit. Verdict text below preserves the literal aggregator outputs for traceability + adds the audit findings + the M6.9+ scoping. M3 (Lamarckian inheritance pilot, +47pp on 4 seeds) remains the strongest concrete Phase 5 result.

This logbook covers Phase 5 M6 in eight commits across one PR: framework, configs, evaluator, aggregator, pilot+full campaigns, and the post-pilot deep-dive audit. The headline finding is **the framework works correctly (12/12 aggregator tests pass; cascade arithmetic round-trips; gate logic discriminates synthetic pass/fail cases), but the M6 design has four substantive issues — substrate shape, training-reward + env geometry, F0 extraction context, and asymmetric F1+ compute — that prevent a clean answer about whether TEI manifests on this codebase. M6.9+ (a follow-up plan to be detailed in a new OpenSpec change) will redesign the substrate and revisit the four issues.**

## Objective

Demonstrate transgenerational epigenetic inheritance (TEI) of pathogen avoidance on the M3-validated LSTMPPO+klinotaxis substrate. The biology (Posner 2023 + Akinosho/Vidal-Gadea 2025): F0 generation pathogen exposure produces small-RNA signals that persist into F1/F2/F3 offspring and bias behaviour toward avoidance **without** direct re-exposure in offspring. Decision gate (tasks.md M6): F1 ≥ 40% × F0, F2 ≥ 25%, F3 ≥ 15%, monotone non-increasing decay, in ≥ 2 of 4 seeds for GO.

A successful M6 would have been the third headline Phase 5 mechanism deliverable (after M3 Lamarckian's strong positive and M5 Co-Evolution's STOP). No prior computational replication of the canonical (Posner/Murphy-defended) TEI mechanism exists on an RL substrate as of May 2026.

## Background

Phase 5 had already shipped:

- **M0** (evolution framework — `EvolutionLoop`, `GenomeEncoder` Protocol, `LearnedPerformanceFitness`, inheritance Protocol)
- **M1** (predator-brain refactor — `PredatorBrain` seam)
- **M2** (hyperparameter evolution — TPE wins for hyperparam, CMA-ES wins for weights)
- **M3** (Lamarckian inheritance pilot — +47pp on 4 seeds; the strongest concrete Phase 5 result)
- **M4** (Baldwin inheritance — INCONCLUSIVE then M4.5 STOP; framework shipped, science unsettled)
- **M5** (Co-evolution arms race — STOP at fixed-architecture substrate; methodology contributions ship intact)

M6's substrate readiness was high: M3 shipped a proven heritable-substrate pattern (`WeightPersistence` + `LamarckianInheritance`, bit-exact across 18 LSTMPPO tensors); Phase 1's predator + nociception sensory machinery was shipped; and the `PredatorType.STATIONARY` toxic-zone primitive already existed at [`env/env.py:91-95`](../../../packages/quantum-nematode/quantumnematode/env/env.py#L91-L95) as a natural pathogen-lawn substrate.

## What shipped

### Substrate (commits 1-7)

| Component | File | Purpose |
|---|---|---|
| Substrate dataclass | [`agent/transgenerational_memory.py`](../../../packages/quantum-nematode/quantumnematode/agent/transgenerational_memory.py) | Frozen dataclass: `logit_bias: Tensor[num_actions]` + `lineage_depth: int` + `source_genome_id: str`; clamp magnitude ≤ 2.0 post-init; `apply_to_logits`, `inherit_from`, `save/load`, `extract_from_brain` telemetry |
| Inheritance strategy | [`evolution/transgenerational_inheritance.py`](../../../packages/quantum-nematode/quantumnematode/evolution/transgenerational_inheritance.py) | `kind() == "transgenerational"`; `select_parents` lex-tie-broken on genome_id; `checkpoint_path` returns `inheritance/gen-NNN/genome-<gid>.tei.pt` |
| LSTMPPO actor-logit hook | [`brain/arch/lstmppo.py:898-925`](../../../packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py#L898-L925) | Reads `self.tei_prior: Tensor[num_actions] \| None`, adds to actor logits before softmax at every step; PPO consistency snapshot ensures rollout-time prior matches update-time prior |
| Per-gen lawn_schedule | [`evolution/loop.py:836-934`](../../../packages/quantum-nematode/quantumnematode/evolution/loop.py#L836-L934) | `_build_per_gen_sim_config(gen)` consumes the schedule entry to toggle `predators.enabled` + `learn_episodes_per_eval` per generation |
| F0 substrate extraction | [`evolution/loop.py:570-734`](../../../packages/quantum-nematode/quantumnematode/evolution/loop.py#L570-L734) | Telemetry pass after F0 `optimizer.tell` + `select_parents`: decode elite genome, load F0 weights, run brain over deterministic probe `BrainParams`, write `.tei.pt`; per-gen elite snapshot writer |
| Config schema | [`utils/config_loader.py`](../../../packages/quantum-nematode/quantumnematode/utils/config_loader.py) | `TransgenerationalConfig` + `LawnScheduleEntry`; validator pairs `transgenerational.enabled` ↔ `inheritance` value; coverage validator ensures every gen has exactly one schedule entry |
| Configs | [`configs/evolution/transgenerational_pathogen_avoidance_lstmppo_klinotaxis{,_tei_off,_smoke}.yml`](../../../configs/evolution) | TEI-on + TEI-off paired-arm pilot + F0 calibration smoke |
| Per-gen evaluator | [`scripts/campaigns/transgenerational_per_gen_eval.py`](../../../scripts/campaigns/transgenerational_per_gen_eval.py) | Offline post-hoc per-(arm, seed, gen) choice-index + survival-rate + termination-reason CSV |
| Paired-arm aggregator | [`scripts/campaigns/aggregate_m6_pilot.py`](../../../scripts/campaigns/aggregate_m6_pilot.py) | Retention table, decision gate (ratio + monotone), cross-seed verdict (GO/PIVOT/STOP); `--campaign-root` flag loads F0 training-time fitness from `per_gen_elites.jsonl` as gate override |
| Campaign shell | [`scripts/campaigns/phase5_m6_transgenerational_lstmppo_klinotaxis.sh`](../../../scripts/campaigns/phase5_m6_transgenerational_lstmppo_klinotaxis.sh) | Paired-arm launcher, F0 calibration smoke target, pilot + full modes |

Test coverage: ~37 new cases across substrate (round-trip, clamp, decay, apply_to_logits, extract_from_brain), inheritance strategy (Protocol conformance, lex-tie-break, checkpoint_path), loop smoke (per-gen schedule, 2-gen paired arms diverge in F1 choice index), LSTMPPO TEI prior (persistence across 100 rollout steps, None default no-op), and aggregator (12 cases including F0-baseline override and JSONL loader). `uv run pytest -m "not nightly"` clean. Ruff + pyright clean.

### Pilot trajectory — five iterations chasing a learnable F0

The original M6 plan called for a single 1-seed pilot + 4-seed full campaign. The pilot turned into a five-stage diagnostic chain when each prior stage's F0 substrate failed to encode a useful avoidance bias. The progression is recorded in detail in [`tmp/evaluations/transgenerational/transgenerational_scratchpad.md`](../../../tmp/evaluations/transgenerational/transgenerational_scratchpad.md) lines 700-1330; the headline arc:

1. **F0 calibration smoke** (damage_radius=3, pop=6 × 1 gen × ~50 ep) — passed at choice_index 0.93 (ceiling-saturated, NOT inside the envelope 0.45 ≤ F0 ≤ 0.85). Diagnostic: damage_radius too small relative to grid size.
2. **Path A retune** (damage_radius 3 → 5, pop=8 × K=1000) — F0 elite fitness ~0.46 (success_rate=0.68 × survival_rate=0.68 under composite fitness), substrate `logit_bias = [-2.0, -2.0, +1.39, -2.0]` for `[FORWARD, LEFT, RIGHT, STAY]` (i.e. "always turn RIGHT"). Trained on real pathogen-conditional dynamics.
3. **Path AA hyperparameter sweep** (K ∈ {500, 1000}, pop=8 with TPE) — confirmed survival_rate as a parallel-tracked metric with real dynamic range (vs choice_index which is geometry-dominated and clamps near 0.93 even for an untrained agent).
4. **Path AAA composite fitness** (success_rate × (1 − fitness_survival_weight × death_rate), weight=1.0) — addressed the M3-era "food-grabber dominance" failure mode where success_rate-only selected F0 elites with 68% success and 8% survival (i.e. 24 of 25 eval episodes died). Path AAA F0 elite selected at composite 0.46 with 68% survival.
5. **Full campaign** (4 seeds × pop 16 × 4 gens × paired TEI-on / TEI-off, K=1000 at F0, K=0 at F1+ for TEI-on; TEI-off short-circuits the schedule and runs K=1000 every gen — see Audit D below for why this is a confound).

## Headline results

### Choice-index gate ([summary/retention_table.csv](../../../artifacts/logbooks/018-transgenerational-memory/full/retention_table.csv); [summary/decision_gate.csv](../../../artifacts/logbooks/018-transgenerational-memory/full/decision_gate.csv))

| arm | seed | F0 | F1 | F2 | F3 | F1/F0 pass | monotone | overall |
|---|--:|--:|--:|--:|--:|:--:|:--:|:--:|
| tei_on | 42 | 0.93 | 0.95 | 0.96 | 0.93 | yes | NO | FAIL |
| tei_on | 43 | 0.94 | 0.91 | 0.92 | 0.94 | yes | NO | FAIL |
| tei_on | 44 | 0.93 | 0.95 | 0.98 | 0.96 | yes | NO | FAIL |
| tei_on | 45 | 0.90 | 0.97 | 0.95 | 0.94 | yes | NO | FAIL |
| tei_off | 42 | 0.93 | 0.93 | 0.91 | 0.93 | yes | NO | FAIL |
| tei_off | 43 | 0.94 | 0.93 | 0.94 | 0.93 | yes | NO | FAIL |
| tei_off | 44 | 0.93 | 0.91 | 0.90 | 0.91 | yes | NO | FAIL |
| tei_off | 45 | 0.90 | 0.93 | 0.95 | 0.95 | yes | NO | FAIL |

**Cross-seed verdict: STOP (0 of 4 seeds pass either arm).** Every seed has F1 ≥ F0 — monotonicity fails in every cell because the choice_index metric (fraction of episode steps spent outside damage radius) is geometry-dominated. An agent that wanders randomly already spends >90% of steps outside the lawn (the lawn is a small disk inside a large grid). The metric doesn't discriminate "learned avoidance" from "random walk that happens to miss the lawn." This is **expected by metric design** — see Audit A below.

### Survival-rate gate with F0 training-time override ([summary/survival_retention_table.csv](../../../artifacts/logbooks/018-transgenerational-memory/full/survival_retention_table.csv); [summary/survival_decision_gate.csv](../../../artifacts/logbooks/018-transgenerational-memory/full/survival_decision_gate.csv))

| arm | seed | F0 (trained) | F1 | F2 | F3 | F1/F0 pass | monotone | overall |
|---|--:|--:|--:|--:|--:|:--:|:--:|:--:|
| tei_on | 42 | 0.46 | 0.12 | 0.08 | 0.00 | NO | yes | FAIL |
| tei_on | 43 | 0.36 | 0.00 | 0.08 | 0.04 | NO | NO | FAIL |
| tei_on | 44 | 0.29 | **0.24** | **0.24** | **0.12** | yes | yes | **PASS** |
| tei_on | 45 | 0.17 | 0.16 | 0.24 | 0.04 | yes | NO | FAIL |
| tei_off | 42 | 0.46 | 0.12 | 0.04 | 0.00 | NO | yes | FAIL |
| tei_off | 43 | 0.36 | 0.04 | 0.08 | 0.00 | NO | NO | FAIL |
| tei_off | 44 | 0.29 | 0.04 | 0.00 | 0.00 | NO | yes | FAIL |
| tei_off | 45 | 0.17 | 0.04 | 0.08 | 0.04 | NO | yes | FAIL |

**Cross-seed verdict: PIVOT for TEI-on (1 of 4 seeds pass); STOP for TEI-off (0 of 4).** The F0 column under override is the training-time `LearnedPerformanceFitness` composite (`success_rate × (1 − death_rate)`) read from each arm/seed's `per_gen_elites.jsonl`; F1+ entries are the post-hoc survival_rate (1 − HEALTH_DEPLETED rate). The override exists because the post-hoc evaluator decodes an UNTRAINED brain at F0 (since the trained F0 weights are GC'd by the substrate-extraction pipeline) — without the override, F0 reads ~0.08 (the untrained-brain baseline) and the monotone check trivially fails at F1 > F0 across the board.

**The TEI-on > TEI-off paired delta at F1+** (averaged across seeds):

| Gen | TEI-on mean survival | TEI-off mean survival | TEI-on − TEI-off |
|--:|--:|--:|--:|
| F1 | 0.13 | 0.06 | **+0.07** |
| F2 | 0.16 | 0.05 | **+0.11** |
| F3 | 0.05 | 0.01 | **+0.04** |

TEI-on outperforms TEI-off at every F1+ generation (cross-seed averages, ~6-12 percentage points absolute). One seed (TEI-on seed 44) clears the monotone + ratio gate cleanly. This is the closest M6 came to an empirical positive signal — but see Audit D below for why the asymmetric F1+ compute envelope makes this comparison answer a different question than the spec posed.

### Verdict (per the M6 decision-gate spec + post-pilot audit)

**INCONCLUSIVE ⚠️.** The framework is shipped. The literal aggregator output is STOP under the choice-index gate and PIVOT (TEI-on, 1/4) / STOP (TEI-off, 0/4) under the survival-rate gate with F0 override. Neither verdict is load-bearing because the post-pilot audit identified four blocking design issues that mean the gates compared a substrate that cannot, by construction, encode pathogen-conditional avoidance against a non-symmetric control. We can't claim "TEI doesn't work on this codebase" from this data — only "the M6 design as built doesn't show clean signal, and four diagnoses point at *why*."

## Audit — four design issues that prevent a clean verdict

After the full campaign completed and the literal verdict was drafted, a fresh-eyes deep-dive against the live code found four issues that any one of which would invalidate the verdict. All four together mean the M6 pilot did not measure what the gates claimed to measure. Verdict downgraded to INCONCLUSIVE as a result.

### A. Substrate shape is gradient-unconditional

The substrate (design D1) is a per-action additive logit bias of shape `(num_actions,)` — a single vector added to the actor logits at every step of every episode. Biology is **gradient-conditional**: in C. elegans, F1 offspring of pathogen-exposed F0 worms avoid pathogen *when they sense the chemoattractant gradient*, and forage normally otherwise. The shipped substrate cannot express that. It can only express "globally prefer/avoid action X" — a constant motion bias that fires whether the agent is near a pathogen, near food, or in empty space. See [`agent/transgenerational_memory.py:169-193`](../../../packages/quantum-nematode/quantumnematode/agent/transgenerational_memory.py#L169-L193) (`apply_to_logits`) — it broadcasts the bias over the entire trajectory unconditionally.

**Empirical fingerprint**: the F0 substrate extracted from the Path AAA F0 elite (seed 42) was `[-2.0, -2.0, +1.39, -2.0]` for `[FORWARD, LEFT, RIGHT, STAY]` — i.e. "always turn RIGHT." 3 of 4 seeds (42, 44, 45) produced **bit-identical** substrates of this form. The substrate captures a motion-bias attractor that PPO finds reliably under our env, not a pathogen-conditional avoidance heuristic. See Audit B / G below for why PPO finds this attractor.

### B. Training reward + env geometry — not fitness shape

A natural follow-up suspicion is that the composite fitness `success_rate × (1 − fitness_survival_weight × death_rate)` rewards "starvation-by-inaction" as a survival strategy. **This is mathematically false**: a pure starver has `success_rate = 0`, so the composite is `0 × (anything) = 0`. The composite faithfully ranks "circle right" above "starve in place" because the former occasionally completes a food run.

The real upstream failure is the **PPO step reward** in [`agent/reward_calculator.py:104-133`](../../../packages/quantum-nematode/quantumnematode/agent/reward_calculator.py#L104-L133), which has both a food-approach term AND a distance-scaled predator-evasion term + a contact penalty. Under our env's geometry (small toxic-zone disk + a strong evasion penalty term), "circle right always" is a low-curvature local optimum: high evasion reward (tangent motion away from the lawn), low pathogen contact, occasional food pickup, no need for any pathogen-conditional logic. PPO finds this attractor in K=1000 episodes; composite fitness then faithfully ranks the resulting policy as the F0 elite.

**Remedy is upstream of fitness**: re-shape the env or the training reward so "circle right always" isn't the easiest policy to find. Examples: multi-lawn layouts that punish unconditional motion bias; gradient-only reward (no contact penalty); per-step reward that requires positive food progress.

### C. F0 extraction probes have no pathogen context

The F0 substrate-extraction telemetry pass at [`evolution/loop.py:732-834`](../../../packages/quantum-nematode/quantumnematode/evolution/loop.py#L732-L834) (`_build_f0_probe_params`) constructs three synthetic `BrainParams` and runs the trained F0 brain over them to extract the empirical action distribution. **All three probes have `predator_gradient_strength=0.0` and `predator_gradient_direction=0.0`** — they vary only `food_gradient_strength` and `food_gradient_direction`. The substrate captures the F0 policy's deviation from uniform action distribution **when seeing food gradients at three fixed strengths**, NOT when near a pathogen.

The function's docstring (lines 735-740) explicitly flags this as a "minimal default implementation" with follow-up needed: *"a follow-up commit may replace this with a ring-of-probe-positions generator that uses the env's STATIONARY-predator coordinates."* The follow-up commit never landed. Commit 7 ("F0 probe fix") added the explicit `predator_gradient_strength=0.0` lines while addressing the STAM-dim alignment issue (a different bug) — confirming this is the current state, not a stub leftover. See `git show 2add25af -- packages/quantum-nematode/quantumnematode/evolution/loop.py`.

Compounds Issue A: even if the substrate shape could express gradient-conditional avoidance, the extraction contexts don't include any pathogen-gradient signal to capture.

### D. F1+ asymmetric compute — `transgenerational+weights` never tested

Two implementation facts that together make the TEI-on vs TEI-off comparison non-symmetric in compute:

1. **TEI-on F1+ evaluates a fresh-random brain + logit_bias** — at [`evolution/fitness.py:459`](../../../packages/quantum-nematode/quantumnematode/evolution/fitness.py#L459), each F1+ candidate is built via `encoder.decode(genome, sim_config, seed=seed)` which produces a freshly-random LSTMPPO from the TPE-sampled genome params. The F0 elite's trained weights are GC'd at the generation boundary; only the `logit_bias` carries over via the substrate-load pipeline at [`fitness.py:469-508`](../../../packages/quantum-nematode/quantumnematode/evolution/fitness.py#L469-L508). At F1+ with `ppo_train_episodes: 0`, the eval phase runs 25 episodes on this fresh-random + biased brain. **Training-time elite fitness at TEI-on F1+ is 0.0** in every gen, every seed — the substrate alone cannot rescue an untrained policy.
2. **TEI-off F1+ retrains from scratch with K=1000 every gen** — at [`evolution/loop.py:869-870`](../../../packages/quantum-nematode/quantumnematode/evolution/loop.py#L869-L870), `if cfg.transgenerational is None or not cfg.transgenerational.enabled: return self.sim_config`. With `enabled=false` on the control arm, the lawn_schedule's `ppo_train_episodes: 0` override is never applied; the base `evolution.learn_episodes_per_eval: 1000` is used at every gen. The TEI-off arm runs **K=1000 fresh training every generation, no inheritance** — not the "no train, no inheritance" pure-control the spec D6 implied.

The post-hoc evaluator's TEI-on F1+ > TEI-off F1+ delta (+0.04 to +0.11 absolute) therefore answers a **different question** than the spec posed:

- Spec wanted: "does the substrate carry any avoidance signal at all?"
- Implementation measured: "does the substrate (applied to a fresh-random brain) carry **at least as much** avoidance signal as a freshly K=1000 trained brain (no inheritance)?"

The original M6 plan flagged exactly this scenario as an escalation criterion: *"if pure-TEI F1 floor collapses to chance, schedule a second pilot with a new `kind() == "transgenerational+weights"`."* The pure-TEI floor did collapse (TEI-on F1+ training-time fitness ≡ 0.0); the `transgenerational+weights` configuration was never run.

### Secondary tuning concerns (real but contingent on A/C/D being fixed)

| # | Where | Note |
|---|---|---|
| E | Elite tie-break | TEI-on F1+ fitnesses all tie at 0 → lex-smallest UUID picks the "elite". Genome UUIDs are structural and seed-independent (see [`evolution/genome.py:91`](../../../packages/quantum-nematode/quantumnematode/evolution/genome.py#L91)). The same `idx` is selected as elite across all 4 seeds. Lineage CSV records artefactual elites at F1+. |
| F | Decay factor 0.6 | Biology-anchored (F0=1.0 / F1=0.6 / F2=0.36 / F3=0.216 cascade), never tuned for our env. Moot until A/C/D are fixed. |
| G | F0 substrate determinism | 3 of 4 seeds produce bit-identical `[-2.0, -2.0, +1.39, -2.0]`. Compounds A/B/C: with motion-bias-only substrates and zero-pathogen-context probes, different trained F0 brains can look identical to the extractor. |

These are footnoted as tuning knobs that only matter after A/C/D are fixed.

## Why no TEI signal — root cause synthesis

The four issues map onto a single root cause: **the substrate carries policy-level motion bias, not avoidance behaviour**. Specifically:

1. The brain's *trained avoidance behaviour* is encoded in its LSTM hidden state + GRU + feature weights — distributed, recurrent, gradient-conditional state. PPO learns it; CMA-ES could in principle inherit it (M3 confirmed); but a 4-vector `logit_bias` (Issue A) is a strictly weaker substrate.
2. The *training reward + env layout* (Issue B) produces a low-curvature attractor at "circle right always", which is what the substrate happens to capture in 3/4 seeds (Issue G).
3. The *extraction contexts* (Issue C) don't include any pathogen-gradient signal, so even a gradient-conditional brain would have its avoidance response invisible to the extractor.
4. The *F1+ comparison* (Issue D) is non-symmetric and the `transgenerational+weights` follow-up that the plan flagged was never run, so we cannot answer "does the substrate add value on top of trained weights" — only "does the substrate alone match K=1000 retraining" (it doesn't).

**My read**: cause #1 (substrate shape) is dominant, but causes #2-#4 each compound the problem in different ways. A serious TEI attempt on this RL substrate needs all four addressed: a richer substrate (probably a small parametric bias-network operating on the brain's state vector, not a constant logit), env+reward redesign to remove the motion-bias local optimum, env-derived pathogen-context probes, and a `transgenerational+weights` configuration as the symmetric control.

## Methodology contributions

Even with INCONCLUSIVE M6, the framework + post-pilot tooling produces reusable methodology for future TEI attempts:

1. **`TransgenerationalInheritance` strategy + `TransgenerationalMemory` dataclass** as a reference for any future heritable-substrate-without-retraining inheritance kind. The Protocol conformance pattern (`kind() == "transgenerational"`, `select_parents` lex-tie-break, `checkpoint_path`, `inherit_from` decay) generalises to other substrates (small-RNA arrays, methylation tags, etc.).
2. **LSTMPPO `tei_prior` actor-logit hook** — minimal-blast-radius brain integration: an optional attribute on the brain instance, default `None`, no Brain Protocol change. The runner sets it before `prepare_episode()`; the brain reads it inside `run_brain` at the actor head. Pattern carries forward to any future substrate that biases the policy at evaluation time.
3. **Per-gen `lawn_schedule` loop consumer** + paired-arm aggregator with F0 training-time override — diagnoses the "post-hoc evaluator's F0 is untrained" trap that any future inheritance-without-retraining design will hit. The override flag + JSONL loader at [`scripts/campaigns/aggregate_m6_pilot.py`](../../../scripts/campaigns/aggregate_m6_pilot.py) ship as the right primitive.
4. **Survival-rate as a parallel-tracked metric to choice-index** — Path AAA forensics established that choice_index is geometry-dominated and clamps near 0.93 even for an untrained agent in this env; survival_rate (1 − HEALTH_DEPLETED rate) has real dynamic range. Future avoidance-task pilots should default to survival_rate as the primary metric, with choice-index as a sanity-check secondary.
5. **F0 calibration smoke as a hard pre-flight gate** — the original `0.45 ≤ F0 ≤ 0.85` envelope check (M6.5 pre-flight) caught the damage_radius=3 ceiling-saturation issue in 30 minutes, before committing to the full ~16 wall-hour campaign. Pattern carries forward to any future biological-fidelity milestone with a calibratable substrate.

These instruments are the M6 contribution that survives the INCONCLUSIVE verdict.

## Compute envelope

- Substrate dev (commits 1-7): zero pilot compute; all unit tests + smoke runs.
- F0 calibration smoke (Path A retune): 1 seed × 1 gen × pop 6 × ~50 ep ≈ 30 min.
- Path AA hyperparameter sweep: 2 seeds × pop 8 × 1 gen × K ∈ {500, 1000} ≈ 1 wall-h.
- Path AAA pilot: 1 seed × pop 8 × 4 gens × paired arms ≈ 4 wall-h.
- Full campaign: 4 seeds × pop 16 × 4 gens × paired arms ≈ 14 wall-h actual (lower than the planned 16 because TEI-on F1+ runs K=0 — only the eval phase consumes compute at F1+).
- Post-hoc evaluator runs (per-gen choice-index + survival-rate): ~10 min per (arm, seed) pair × 8 pairs = ~1.3 wall-h.

Total: **~21 wall-h** across the entire diagnostic chain + full campaign. The five-iteration pilot trajectory cost roughly the same as a single planned-from-the-start campaign would have; the iterations bought the diagnostic chain that surfaced the four audit issues.

## Citations and biology grounding

- **Posner 2023** (Cell): primary TEI mechanism in C. elegans — sRNA-mediated F0→F3 avoidance inheritance.
- **Hunter critique** (2022): replication-failure critique, documented in tasks.md line 232.
- **Murphy rebuttal** (2023): defence of the original Posner mechanism.
- **Akinosho/Vidal-Gadea 2025** (eLife): independent validation; Kaletsky F2 ≈ 0.5–0.6 on PA14 — our intended F0 envelope anchor. The biological "choice index" definition (`1 - steps_inside_damage_radius/total_steps`) matches our [`scripts/campaigns/transgenerational_per_gen_eval.py`](../../../scripts/campaigns/transgenerational_per_gen_eval.py) implementation.

The biological grounding remains sound — the failure mode is in the RL substrate's expressiveness (Issue A), not in the biology citation.

## What's next: M6.9+ TEI re-evaluation

The four audit issues map onto a four-axis redesign agenda, lightly scoped here pending a dedicated next-PR OpenSpec change (per the precedent set by M4 → M4.5: framework PR ships first; redesign plan lives in its own change). Details deliberately left thin in this logbook; the next OpenSpec will surface additional issues after a deeper dive and propose specific substrates.

- **M6.9 — Substrate redesign** (addresses Audit A + G). A richer substrate shape that can express gradient-conditional avoidance — most likely a small parametric bias-network whose input is the brain's sensory state and whose output is a per-step additive logit bias, not a constant 4-vector. Forward-compatible with the existing `TransgenerationalInheritance.kind()` Protocol and `.tei.pt` serialisation.
- **M6.10 — Training reward + env redesign** (addresses Audit B). Multi-lawn layouts + reward-shaping to remove the "circle right always" local optimum. Likely re-uses the M5 fair-test cell-grid methodology to verify the redesign is harder for unconditional motion bias.
- **M6.11 — Env-derived extraction contexts** (addresses Audit C). Replace the three synthetic `_build_f0_probe_params` entries with a ring of probe positions sampled from the actual env's STATIONARY-predator coordinates, per the deferred follow-up that commit 7's docstring explicitly flagged. Probe contexts include real `predator_gradient_strength > 0` values.
- **M6.12 — `transgenerational+weights` configuration** (addresses Audit D). A symmetric-compute variant: F1+ inherits both the F0 elite's trained weights (via the M3 Lamarckian pattern) AND the decayed substrate, with paired-arm ablation against "weights-only" (M3) and "substrate-only" (current M6) controls. This is the configuration the original M6 plan named as the escalation path; M6.12 is when it actually runs.

The substrate work in M6.9 is *not* planned in detail in this logbook. The next PR's OpenSpec change will draft proposal + design + spec + tasks for M6.9+ in artifact-driven order, surface any further design issues that a deeper review finds, and produce the implementation plan. M6.9+ stays optional in the Phase 5 plan; the recommendation if it's scheduled is to use M6's survival-rate metric + F0-training-fitness override as the verdict primitives (they're discriminative where choice-index isn't).

## Verdict line

**M6: INCONCLUSIVE ⚠️.** Framework shipped (TransgenerationalInheritance + TransgenerationalMemory + LSTMPPO `tei_prior` hook + per-gen lawn_schedule + paired-arm aggregator + per-gen evaluator + 12/12 aggregator tests + ~37 substrate/loop tests). Literal aggregator output STOP (choice-index gate, 0/4 seeds either arm) / PIVOT TEI-on, STOP TEI-off (survival-rate gate with F0 override, 1/4 vs 0/4) — neither verdict is load-bearing because the post-pilot audit identified four blocking design issues (substrate shape, training-reward + env geometry, F0 extraction context, asymmetric F1+ compute + missing `transgenerational+weights` control) that mean the gates compared a substrate that cannot, by construction, encode pathogen-conditional avoidance against a non-symmetric control. M3 (Lamarckian inheritance pilot, +47pp / +79pp) remains the strongest concrete Phase 5 result. M6.9+ (substrate redesign + env-derived probes + training-reward redesign + `transgenerational+weights`) is the proper TEI re-evaluation, to be planned in a dedicated follow-up OpenSpec change.
