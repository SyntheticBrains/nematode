# Design: Transgenerational Memory Re-evaluation (M6.9+ PR-A)

## Context

The M6 milestone closed INCONCLUSIVE after a five-iteration calibration chain (Path A → AA → AAA) revealed four blocking design issues — see [logbook 018](../../../docs/experiments/logbooks/018-transgenerational-memory.md) § Audit and the archived [`design.md` addendum](../archive/2026-05-17-add-transgenerational-memory/design.md). The framework is mechanically sound (`TransgenerationalInheritance` strategy + `TransgenerationalMemory` dataclass + `tei_prior` actor-logit hook + per-gen `lawn_schedule` consumer + F0 extraction telemetry + paired-arm aggregator with F0-override + per-gen evaluator with `FrozenEvalRunner`) — what's missing is biological fidelity in four places (substrate shape, training reward + env geometry, F0 extraction context, asymmetric F1+ compute).

PR-A addresses the first three (substrate / env+reward / probes) on a three-arm `{tei_on, weights_only, control}` campaign with explicit mid-flight tripwires + a pre-declared pivot table. Audit D (symmetric-compute `transgenerational+weights` control) is gated to PR-B and scaffolded as a separate OpenSpec change only if PR-A's pure-TEI floor signal is non-zero.

**Pre-mortem-driven design**: the original M6 plan committed early to substrate shape, decay shape, and probe topology, then locked the campaign in. M6.9+ instead promotes these to YAML knobs so mid-flight pivots are cheap, AND adds four explicit tripwires at the calibration smoke layer that catch the M6 failure mode (3-of-4 bit-identical substrate convergence) BEFORE pilot compute is spent.

**Constraints (rationale)**:

1. *Biological framing forces sensory-conditional substrate.* Kaletsky/Murphy P11 is *"necessary AND sufficient"* (switch-like, context-dependent). Logbook 018 § Audit A: *"Biology is gradient-conditional: in C. elegans, F1 offspring of pathogen-exposed F0 worms avoid pathogen when they sense the chemoattractant gradient, and forage normally otherwise."* A constant logit-bias vector cannot express this; a small parametric bias-network (sensory_input → per-action logit offsets) is the minimal biologically-defensible shape.
2. *Audit-flagged env redesign forces removing both feasibility and incentive of the motion-bias attractor.* Geometry alone (density+grid_size YAML) removes feasibility of an unobstructed circular path; reward shape change (`gradient_only` mode) removes the incentive. Either alone may leave a softer attractor.
3. *Audit-flagged probe redesign forces env-derived sampling.* Pathogen contexts must be present in the extraction probes; M6's all-zero `predator_gradient_strength` is the structural source of the substrate being unable to capture pathogen response.
4. *n=4 cross-arm verdict forces noise-aware statistics.* A bare 5pp threshold at n=4 is noise-bounded (M5 ran n=12 for credible signal). Wilcoxon + bootstrap CI = the noise-aware primary verdict.
5. *Tripwires + pivot table force mid-flight gating.* M6's failure mode was running the full campaign then auditing post-hoc. Four tripwires at calibration smoke + a binding pivot table at pilot review eliminate that path.

**Stakeholders**: Phase 5 tracker M6 section ([phase5-tracking/tasks.md](../phase5-tracking/tasks.md) lines 222–246); Phase 5 synthesis logbook (M8); roadmap M6 row ([docs/roadmap.md](../../../docs/roadmap.md)).

## Goals / Non-Goals

**Goals**

- A sensory-conditional substrate (`TransgenerationalMemory.bias_network`) capable of expressing context-dependent action biases independent of trained weights. Architecture configurable via YAML.
- An audit-corrected env + reward shape that removes the M6 "circle right always" motion-bias local optimum. New reward_mode is YAML-gated; default preserves M3/M4/M5/M6 byte-equivalence.
- Env-derived F0 extraction probes with real `predator_gradient_strength` values, computed from actual stationary-predator positions. Probe ring topology configurable via YAML.
- A three-arm paired-cohort campaign `{tei_on, weights_only, control}` that disentangles pure-TEI signal from the M3 weight-flow baseline. Four mid-flight tripwires + pre-declared pivot table + n=4-noise-aware cross-arm verdict.
- A binding go/no-go between PR-A (audit A/B/C corrected) and PR-B (TEI+weights audit-D-corrected control). PR-B is scaffolded as a separate OpenSpec change only at PR-A's verdict.

**Non-Goals**

- Audit D fix (`transgenerational+weights` symmetric-compute control). Gated to PR-B; scaffolded at PR-A close iff `tei_on > control` cross-arm delta is statistically non-zero.
- Brain architecture evolution (NEAT — separate milestone M7).
- Multi-pathogen substrate (single sRNA-token-equivalent only).
- Reward shapes beyond `default` and `gradient_only` (e.g. sparse-reward; deferred to follow-up if pilot pivot table flags it).
- Wet-lab biology replication beyond Kaletsky envelope anchoring. M6.9+'s survival_rate envelope is project-internal calibration, not wet-lab analogue (honest gap — see Risks R8).
- Spec-syncing the modified-capability deltas to main specs at archive time. Per the M4 / M6 INCONCLUSIVE precedent: delta requirements stay in the change archive until either PR-A produces a GO (then sync) or M6.13+ supersedes the design.

## Decisions

### D1. Substrate shape — sensory-conditional parametric bias-network (YAML-configurable)

`TransgenerationalMemory` is extended with an optional `bias_network: torch.nn.Sequential | None`. When `None`, behaviour is byte-equivalent to M6 (legacy `logit_bias` path); existing M6 tests stay green. When set, `apply_to_logits(logits, sensory_input)` returns `logits + bias_network(sensory_input)`. Output clamped at `|x| ≤ LOGIT_BIAS_CLAMP = 2.0` (current M6 cap).

**Configurable defaults** under `TransgenerationalConfig.bias_network`:

| Field | Default | Range | Why configurable (post-mortem R1) |
|---|---|---|---|
| `hidden_dim` | 8 | int ≥ 0 (0 = linear projection) | Capacity dial — if pilot shows underfit, bump to 16/32 via YAML. |
| `activation` | `"tanh"` | `tanh / relu / gelu` | Ablation knob. |
| `input_features` | `["predator_gradient_strength", "predator_gradient_direction_sin", "food_gradient_strength"]` | list of `BrainParams` field names + `_sin`/`_cos` suffix support for angles | If 3-input sensory slice misses signal the brain uses, mid-flight add `["stam_state_mean", "agent_direction_sin"]`. |
| `decay_shape` | `"geometric"` | `geometric / linear / sigmoid` | Biology silent on decay shape (lit harvest). Geometric matches "diluted across generations" framing; linear/sigmoid for sensitivity analysis. |
| `decay_factor` | 0.6 | float ∈ [0, 1] | Unchanged from M6. Cascade F0=1.0 → F1=0.6 → F2=0.36 → F3=0.216 under geometric. |

**Decay (per `decay_shape`)**:

- `geometric`: each weight tensor scaled by `decay_factor` per generation. Multiplicative — same as M6.
- `linear`: scaled by `max(0, 1 − lineage_depth × (1 − decay_factor))`. Reaches zero at lineage_depth = 1/(1−decay_factor) ≈ 2.5 generations under decay_factor=0.6.
- `sigmoid`: cumulative scale at depth `d` is `sigmoid(K × (M − d))` with **fixed** `K = 2.0`, `M = 1.0` — explicitly **independent of `decay_factor`**. Slow-then-fast shape: `cum(0) ≈ 0.881`, `cum(1) = 0.500`, `cum(2) ≈ 0.119`, `cum(3) ≈ 0.018`. Used as a sensitivity-analysis option for pilot pivots ("decay shape too aggressive under geometric collapse" → switch to sigmoid for slower start). Calibration-by-decay-factor is reserved for `geometric` and `linear`.

**Output clamp** is applied identically across decay shapes — substrate magnitude bounded by `LOGIT_BIAS_CLAMP`.

**Backwards compatibility**: legacy `logit_bias: Tensor[num_actions]` field retained on `TransgenerationalMemory`; when `bias_network is None`, `apply_to_logits` falls back to `logits + self.logit_bias`. M6 round-trip tests stay green; M6.9+ tests cover the new `bias_network` path.

**Biological framing**: Kaletsky/Murphy P11 *"necessary AND sufficient"* — context-dependent switch. The bias-network is the **decoder** (sensory-state → action-bias); the F0 elite learns the decoder during training; cascade preserves it across generations.

**Alternatives considered**:

- *Linear projection only (W @ sensory + b, ~12 params)*: rejected — minimum capacity may collapse to constant-bias case when projection is near-zero (regression to M6).
- *Single attention-head gate (bias = sigmoid(gate) × base_bias)*: rejected — adds complexity without clear scientific gain; the MLP subsumes the gate.
- *Per-step recurrent substrate (substrate has hidden state)*: rejected — out of scope; conflates substrate with brain architecture.

**Implementation notes**:

- *Frozen dataclass + `nn.Sequential` field.* `TransgenerationalMemory` is `@dataclass(frozen=True)` (M6 invariant — prevents cross-generation aliasing mutation). PyTorch `nn.Module` instances are valid frozen-dataclass field values, but Python's default dataclass `__eq__` falls back to identity comparison on modules — fine for our use case. The `__post_init__` pattern mirrors the existing `logit_bias` clamp-clone: when a caller passes a module, `__post_init__` SHALL `copy.deepcopy` it then `object.__setattr__(self, "bias_network", deepcopy_result)` so caller-side mutations cannot bleed across generations. Identical to the M6 `logit_bias.detach().clone()` pattern, applied to module state.
- *Sensory feature transforms (sin/cos).* The default `input_features` list contains `predator_gradient_direction_sin`. This is **not** an existing `BrainParams` field — it's a derived transform computed on the fly inside `apply_to_logits` / `extract_from_brain`: when an input feature name ends in `_sin` or `_cos`, the substrate strips the suffix, reads the base radian field (`predator_gradient_direction`), and applies `math.sin` / `math.cos`. The validator at YAML load time accepts both the raw field names AND the `_sin` / `_cos` derived names; unknown stems are rejected with a message listing the supported `BrainParams` fields.

### D2. F0 substrate extraction — env-derived probe ring (YAML-configurable)

`EvolutionLoop._build_f0_probe_params` rewritten:

1. Read `env.predators` from the F0-eval env (available via `sim_config.environment.predators`).
2. For each predator, generate `probe_ring.count` (default 8) ring positions at distance `probe_ring.radius_offset + damage_radius` from the predator center, evenly distributed by angle.
3. For each probe position, compute `predator_gradient_strength = 1.0 / (1.0 + manhattan_distance(probe_pos, predator_pos))` and `predator_gradient_direction = atan2(predator_y − probe_y, predator_x − probe_x)`. Both pure functions, unit-testable.
4. Optionally vary food_gradient per probe-ring iteration if `probe_ring.include_food_gradient_variants: true` (default `false` — keeps probes pathogen-isolated).
5. Return `num_lawns × probe_ring.count` total probes (default: 5 lawns × 8 = 40 probes on the M6.10 env).

**Configurable fields** under `TransgenerationalConfig.probe_ring`:

| Field | Default | Why configurable |
|---|---|---|
| `count` | 8 | If predators on the new 15×15 grid overlap, halve to 4. |
| `radius_offset` | 1 | Offset from `damage_radius` of the closest probe. |
| `include_food_gradient_variants` | `false` | Defaults to pathogen-isolated; sensitivity analysis enables paired food contexts. |

**Bias-network fit** in `extract_from_brain`: for each probe, run the F0 brain `n=10` action-selection samples (deterministic on `extraction_seed`); compute empirical action distribution; fit the MLP by 50 epochs of Adam over `(sensory_input → logit_offset_from_uniform)`. Closed-form least-squares when `hidden_dim == 0` (linear projection).

**Alternatives considered**:

- *3 synthetic probes (M6 default)*: rejected — Audit C explicitly faulted this for having no pathogen context.
- *Random sampling of env states*: rejected — non-deterministic; would not give a reproducible substrate across runs at the same seed.
- *Probe at every grid cell*: rejected — `grid_size² = 225` probes is 5.6× the ring default for no scientific gain.

### D3. Env redesign — density + reward shape (YAML for env, code for reward_mode)

**Env (YAML only)**:

- `environment.grid_size: 15` (was 20). Removes the geometric feasibility of an unobstructed circular path through 3 lawns.
- `environment.predators.count: 5` (was 3). Increased lawn density forces gradient-conditional navigation.
- `environment.predators.damage_radius: 3` (unchanged). Preserves the M6 single-lawn coverage geometry.
- `environment.foods.count: 8` (was 6). Keeps density proportional to grid area (15²/20² × 6 ≈ 3.4 + buffer for redundancy).

**Reward (code in `agent/reward_calculator.py`)**: new `reward_mode: Literal["default", "gradient_only"] = "default"` field on `RewardConfig`. Branch in `RewardCalculator.calculate_reward`:

- `reward_mode == "default"` (legacy M3/M4/M5/M6): unchanged. Distance-scaled `evasion_reward = penalty_predator_proximity * (curr_pred_dist − prev_pred_dist)` term + contact penalty + food-approach + per-step cost.
- `reward_mode == "gradient_only"`: drop the distance-scaled evasion term; keep contact penalty (`if curr_pred_dist <= 1: reward -= penalty_predator_proximity`) + `HEALTH_DEPLETED` termination + food-approach + per-step cost.

**Why both axes together (per pre-mortem)**: density alone removes geometric feasibility of the circular attractor; reward shape change removes the incentive. Together they guarantee "circle right always" is neither feasible nor incentivised. Either alone may leave a softer attractor (lower-amplitude but still present).

**Backwards compatibility**: `reward_mode: "default"` is the schema default; M3/M4/M5/M6 configs that don't set the field are byte-equivalent to today. Test asserts via existing M3 reproduction.

### D4. Three-arm campaign + n=4-noise-aware verdict

| Arm | `inheritance` | F1+ train | F1+ inheritance | What it tests |
|---|---|--:|---|---|
| `tei_on` | `transgenerational` | K=0 | substrate only (decoded sensory MLP, decayed) | Pure-TEI floor — does the audit-A/B/C-corrected substrate carry signal? |
| `weights_only` | `lamarckian` | K_full | F0 elite weights (M3 pattern) | M3 baseline at new env — sanity-check that M3 still works under audit-corrected env. |
| `control` | `none` | K_full | no inheritance | TPE fresh-from-scratch every gen — environmental ceiling at this compute budget. |

**Cross-arm primary verdict** (per pre-mortem R4): GO iff **`tei_on` passes its per-arm gate AND `tei_on − control` paired-seed delta is statistically distinguishable from zero**, via:

- one-sided Wilcoxon signed-rank on 4 paired (F1+ retention) seed deltas with `p < 0.10`, AND
- ≥ 5pp absolute delta AND non-overlapping 80% bootstrap CIs (1000 resamples per seed).

Both must agree on direction. A bare 5pp delta at n=4 is noise-bounded (M5 needed n=12 for credible signal); this is the load-bearing scientific check.

**Cross-arm secondary verdicts**: `weights_only` vs `control` (M3 reproduction on new env); `tei_on` vs `weights_only` (substrate-vs-weights — interpretation deferred to PR-B if PR-A signal exists).

**Three-arm YAML structure (validator-pairing constraint)**: the existing `EvolutionConfig._validate_inheritance` method in `config_loader.py` enforces a one-bit pairing contract — `transgenerational.enabled=True` requires `inheritance: transgenerational`; `transgenerational.enabled=False` requires `inheritance: none`. PR-A is **not** relaxing that validator. Instead the three arm YAMLs differ structurally:

- `tei_on.yml`: `inheritance: transgenerational` + full `transgenerational:` block (`enabled: true`, bias_network, decay_shape, probe_ring, lawn_schedule with F1+ K=0).
- `weights_only.yml`: `inheritance: lamarckian`. **MUST omit the `transgenerational:` block entirely** (set to `None` via absence — the validator at line 1289 only fires when the block is present). Same env + reward as `tei_on.yml`. K=K_full at every gen.
- `control.yml`: `inheritance: none`. **MUST also omit the `transgenerational:` block entirely.** Same env + reward as the other two. K=K_full at every gen.

This is why the three arms cannot share a single base YAML with arm-specific overlays — the validator rejects the unsupported pairings. They share env + brain + reward subtrees as plain YAML duplication; the campaign shell sanity-checks the three files at launch.

**`fitness_survival_weight` parity across arms**: all three arms (including `weights_only` and `control`) MUST set `evolution.fitness_survival_weight: 1.0` to match `tei_on`'s F0 elite-selection rule. Without this, `weights_only` would use raw `success_rate` (M3-byte-equivalent at weight=0.0) and produce food-grabber-dominant elites — confounding the M3 reproduction check on the new env. The fitness_survival_weight is on `EvolutionConfig` (not `TransgenerationalConfig`), so all three YAMLs set it directly. The composite fitness `success_rate × (1 − w × death_rate)` aligns elite selection with the avoidance measurement target across all three arms.

**PR-B trigger**: if `tei_on − control` is statistically non-zero by the criterion above, PR-B is scaffolded as a separate OpenSpec change `add-transgenerational-memory-weights`. If null, PR-B is NOT run; logbook 019 documents the honest pivot to M6.13.

**Alternatives considered**:

- *Four-arm bundled in PR-A* (TEI-on + weights-only + TEI+weights + control): rejected — runs the TEI+weights arm whose interpretation depends on an unvalidated pure-TEI floor (structural twin of M6's audit D). Pre-mortem R5.
- *Two-arm minimal (TEI+weights vs weights-only)*: rejected — punts pure-TEI floor question to M6.13 indefinitely; less rigorous baseline.
- *Bare 5pp threshold without statistical check*: rejected — n=4 noise-bounded.

### D5. Decision gate + metrics

**Primary metric**: `survival_rate` = 1 − `HEALTH_DEPLETED` rate. Choice_index tracked as sanity-check secondary (M6 PR #166 aggregator already supports both).

**F0-baseline override**: aggregator's `--campaign-root` flag (shipped in PR #166) threads training-time F0 fitness from each `per_gen_elites.jsonl` as a per-(arm, seed) gate override. Without override, the post-hoc evaluator decodes an UNTRAINED brain at F0 and the monotone check trivially fails.

**Calibration smoke envelope** (constructed, no wet-lab analogue — honest gap):

- `0.30 ≤ mean F0 survival_rate ≤ 0.70` across the 4 calibration seeds.
- See § D6 tripwires below for the multi-condition gate.

**Gate per (arm, seed)**:

1. F1 survival ≥ 40% × F0 (training-time override).
2. F2 survival ≥ 25% × F0.
3. F3 survival ≥ 15% × F0.
4. Monotone non-increasing: F0 ≥ F1 ≥ F2 ≥ F3.

**Cross-seed verdict per arm**: GO ≥ 2 of 4 / PIVOT = 1 / STOP = 0.

**Cross-arm primary verdict**: see § D4 above.

### D6. Mid-flight tripwires + pre-declared pivot table

**Four tripwires at the calibration smoke layer** (T1–T4 — all must pass before pilot is unblocked):

| # | Tripwire | Pass criterion | Failure pivot |
|--:|---|---|---|
| T1 | F0 survival envelope | `0.30 ≤ mean F0 survival_rate ≤ 0.70` | Retune env (count, damage_radius, K) per the M6 Path A/AA/AAA chain. |
| T2 | Substrate diversity | Pairwise coefficient-of-variation across 4 seeds' `bias_network.state_dict()` > 5% | STOP — M6 attractor signature. Widen `bias_network.hidden_dim` or `input_features`. |
| T3 | M6-floor-to-beat | F0 survival_rate > M6 "circle right always" baseline survival on the new env (recomputable from M6 artefacts at K=0) | STOP — env redesign insufficient. Increase lawn density or strengthen `gradient_only` reward. |
| T4 | Substrate magnitude | mean absolute bias-network output over probes > 0.1 | Substrate degenerate. Increase MLP fit epochs or check probe distribution. |

**Pre-declared pilot pivot table** (binding at pilot review, BEFORE full campaign):

| Pilot observation | Pre-declared pivot |
|---|---|
| All 3 arms collapse to survival_rate ≈ chance at F0 | Reward mode is the culprit; retune `penalty_predator_contact` upward OR widen lawn distribution. Re-run pilot. |
| `tei_on F1 ≈ control F1` (substrate inert) | Widen `bias_network.hidden_dim` 8→16 OR add features (e.g. `stam_state_mean`). Re-run pilot. |
| `tei_on F0` diverse across seeds BUT F1 retention near-uniform across F1/F2/F3 | Decay shape too aggressive; try `decay_shape: "linear"` or `decay_factor: 0.8`. |
| F1 > F0 (monotone-decay violated) at pilot | Substrate unstable / MLP too sensitive to F0 elite idiosyncrasies; reduce `bias_network.hidden_dim` 8→4 OR limit fit epochs (50→20). |
| `tei_on > control` by ≥ 5pp at pilot, but `weights_only ≈ tei_on` | Substrate signal real but matched by weight-flow; PR-B becomes the load-bearing question. |
| All three arms differentiate cleanly, monotone-decay holds | Full campaign proceeds with no pivot. |

This table is binding: if the pilot shows pattern X, the next action is action X. Eliminates "ran the full campaign anyway and audited post-hoc."

**Per-seed gen-2 monotonicity warning** (during full campaign): after each seed completes gen 2, launcher checks `F0 ≥ F1` and emits a non-fatal warning to campaign log. Aggregator surfaces in `summary.md`. n=1 partial is too noisy to abort; warning surfaces pattern early without fatal abort.

### D7. Compute envelope

| Layer | Compute | Wall-h |
|---|---|--:|
| Smoke unit tests | local | \<1 |
| F0 calibration smoke | 4 seeds × pop 6 × 1 gen × `tei_on` only | ~2 |
| Pilot | 1 seed × pop 8 × 4 gens × 3 arms | ~3 |
| Full campaign | 4 seeds × pop 16 × 4 gens × 3 arms | ~22-28 |
| Post-hoc per-gen evaluator | ~30 min × 12 (arm, seed) pairs | ~6 |
| **Total PR-A** | | **~33** |

PR-B (if triggered) adds ~12-15 wall-h for the `tei_weights` arm.

## Risks / Trade-offs

| # | Risk | Mitigation |
|---|---|---|
| R1 | Bias-network overflow under multiplicative decay across generations | All decay shapes shrink magnitude monotonically; output clamped at `LOGIT_BIAS_CLAMP = 2.0`. Unit test asserts decay preserves direction + monotonically shrinks norm. |
| R2 | Sensory input shape drift between F0 elite training env and F1+ inheritance | `bias_network.input_features` persisted in `.tei.pt` payload; runtime validator at substrate load. |
| R3 | F0 survival_rate envelope uncalibratable on new env | T1 tripwire is a hard pre-flight gate. |
| R4 | Substrate converges to same attractor across seeds (M6 failure mode) | T2 substrate-diversity tripwire catches at calibration smoke before pilot. |
| R5 | LSTMPPO + bias-network interaction — MLP drowns recurrent state | `tei_prior` applies at every step before softmax (same hook M6 used). T4 magnitude tripwire catches degenerate near-zero substrate. |
| R6 | Reward-shape regression breaks legacy configs | `reward_mode: "default"` byte-equivalent to M3/M4/M5/M6. Test asserts via existing M3 reproduction. |
| R7 | n=4 cross-arm verdict noise-dominated | Wilcoxon p\<0.10 AND non-overlapping 80% bootstrap CIs — both must agree. Bare 5pp threshold explicitly rejected. |
| R8 | Survival_rate envelope not biology-anchored (honest gap) | Logbook 019 documents the envelope as project-internal calibration. T3 M6-floor-to-beat is the empirical anchor. |
| R9 | Pilot pivot table doesn't cover an unforeseen failure mode | User-review pause at Layer 4 is the human-in-the-loop catch-all. Pre-committed pivots cover *expected* failure modes; user-review covers novel ones. |
| R10 | PR-B trigger criterion (`tei_on > control`) too lenient | Wilcoxon + bootstrap noise check forces a real signal. If PR-A null, PR-B is honest-pivoted (NOT run). |
| R11 | Decay-shape choice (geometric/linear/sigmoid) biases the cascade | All three shapes shipped + configurable via YAML; pilot pivot table includes `decay_shape: "linear"` as an explicit pivot if monotone-decay near-uniform across F1/F2/F3 under geometric. |

## Migration Plan

The change is **additive** — no breaking changes to existing inheritance modes (`none` / `lamarckian` / `baldwin` / `transgenerational`-with-legacy-`logit_bias`) or to non-LSTMPPO brains. `RewardConfig.reward_mode = "default"` and `TransgenerationalConfig.bias_network = None` preserve byte-equivalence on the default path.

**Rollout order (matches commit grouping)**:

1. Scaffold M6.9 sensory-conditional substrate (additive `bias_network` field; default `None` → byte-equivalent).
2. Add M6.10 `gradient_only` reward mode (additive Literal field; default `"default"` → byte-equivalent).
3. Rewrite M6.11 F0 probe-ring (replaces M6's synthetic probes — but only the `transgenerational` inheritance path consumes them, so non-TEI runs unaffected).
4. Wire bias-network application + cascade in LSTMPPO (extends `tei_prior` call site; default-None branch preserves byte-equivalence).
5. Ship three-arm YAML configs + launcher shell + calibration smoke tripwires.
6. Ship three-arm aggregator + n=4-noise-aware verdict logic + pilot pivot table emit.
7. Ship substrate-diversity tripwire script.
8. (Post-experiment) publish logbook 019 + tracker tick + roadmap tick + PR-B trigger decision.

**Rollback**: each commit is self-contained. If a downstream commit fails, the preceding state is byte-equivalent to pre-change `main` for all non-M6.9+ inheritance modes + non-`gradient_only` reward configs.

## Open Questions

None blocking implementation. Three honest gaps remain:

1. **Decay shape choice not biology-anchored.** Geometric is the literature framing default (M6 inherited it from Posner/Kaletsky framing). Linear/sigmoid are sensitivity-analysis options. Pilot pivot table includes `decay_shape: "linear"` as a pre-declared pivot. PR-A documents the choice as project-internal calibration in logbook 019.
2. **Survival_rate envelope `[0.30, 0.70]` constructed not anchored.** No wet-lab analogue to survival_rate exists (Kaletsky 0.55 anchors choice_index only). T1 + T3 tripwires use the constructed envelope + the M6 empirical floor as joint anchor. PR-A documents this gap in logbook 019.
3. **PR-B's substrate fork (`transgenerational+weights` kind) not designed in PR-A.** Deliberate — scaffolded as a separate OpenSpec change at PR-A close iff PR-A's pure-TEI floor signal is non-zero. PR-A does not lock in PR-B's design.
