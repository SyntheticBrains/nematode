# Proposal: Transgenerational Memory Re-evaluation (M6.9+ PR-A)

## Why

M6 (the original transgenerational-memory milestone) closed **INCONCLUSIVE ⚠️** in [PR #166](https://github.com/SyntheticBrains/nematode/pull/166). The framework deliverable shipped solid (`TransgenerationalInheritance` + `TransgenerationalMemory` + LSTMPPO `tei_prior` hook + per-gen `lawn_schedule` + paired-arm aggregator + per-gen evaluator + ~37 substrate/loop tests + 12 aggregator tests pass) but the post-pilot deep-dive audit ([logbook 018](../../../docs/experiments/logbooks/018-transgenerational-memory.md) § Audit) identified **four blocking design issues** that mean the M6 gates compared a substrate which cannot, by construction, encode pathogen-conditional avoidance against a non-symmetric control:

- **A** — Substrate shape is gradient-unconditional. The per-action constant `logit_bias: Tensor[num_actions]` cannot express "AVOID when near pathogen, FORAGE otherwise." 3 of 4 F0 elite seeds produced bit-identical `[-2.0, -2.0, +1.39, -2.0]` ("always turn RIGHT") substrates — motion bias, not pathogen-conditional avoidance.
- **B** — PPO training reward + env geometry produces a "circle right always" motion-bias local optimum. The distance-scaled evasion term in [`reward_calculator.py:104-133`](../../../packages/quantum-nematode/quantumnematode/agent/reward_calculator.py#L104-L133) rewards tangential motion at constant radius; under our env geometry (small toxic-zone disk + 3 lawns + 20×20 grid) that local optimum is geometrically feasible AND incentivised.
- **C** — F0 extraction probes have no pathogen context. All three probes in [`evolution/loop.py:732-834`](../../../packages/quantum-nematode/quantumnematode/evolution/loop.py#L732-L834) have `predator_gradient_strength=0.0` — they only vary `food_gradient_*`. The substrate captures policy deviation under food-only contexts, not under pathogen contexts.
- **D** — F1+ comparison non-symmetric. TEI-on F1+ runs fresh-random brain + logit_bias (25 eval episodes only); TEI-off F1+ short-circuits the schedule and runs K=1000 fresh training every gen. The original M6 plan flagged a `transgenerational+weights` follow-up as the symmetric-compute control — never run.

M6.9+ is the proper TEI re-evaluation that addresses A/B/C in this PR (PR-A) with a three-arm campaign `{tei_on, weights_only, control}` and explicit mid-flight tripwires + a pre-declared pivot table to avoid the M6 failure mode of *"ran the full campaign then audited post-hoc."* Audit D is gated to a follow-up OpenSpec change (PR-B) — only triggered if PR-A produces a non-zero pure-TEI floor signal. The two-PR split prevents running the TEI+weights arm whose interpretation depends on an unvalidated pure-TEI claim — the structural twin of M6's asymmetric-compute confound.

**Intended outcome**: a defensible verdict on the M6 hypothesis under audit-corrected substrate + env + probes. A **GO** at PR-A would credibly claim *"pure-TEI substrate carries a measurable pathogen-conditional avoidance signal on this RL substrate under audit-corrected env."* PR-B would then upgrade the claim to *"substrate adds value on top of trained weights"* (the strongest scientific claim available).

## What Changes

**Substrate redesign (M6.9 — addresses Audit A)**. `TransgenerationalMemory` extended with an optional `bias_network: nn.Sequential | None`. When `None`, behaviour is byte-equivalent to M6 (legacy `logit_bias` path). When set, `apply_to_logits(logits, sensory_input)` returns `logits + bias_network(sensory_input)`. Architecture is YAML-configurable (defaults: 3-input → 8-hidden → 4-output tanh MLP, ~60 params, geometric decay at 0.6). The pre-mortem-driven configurability (`bias_network.hidden_dim`, `activation`, `input_features`, `decay_shape`, `probe_ring.count`, `probe_ring.radius_offset`) lets us pivot mid-flight via YAML rather than a refactor.

**Env + training reward redesign (M6.10 — addresses Audit B)**. Two changes:

- Env density change (YAML only): grid_size 20→15, predators count 3→5, food count 6→8. Removes the geometric feasibility of an unobstructed circular path.
- New `reward_mode: Literal["default", "gradient_only"]` field on `RewardConfig` (code). Under `gradient_only`, drop the distance-scaled `evasion_reward = penalty_predator_proximity * (curr − prev)` term; keep contact penalty + `HEALTH_DEPLETED` termination + food-approach term. Default `"default"` preserves byte-equivalence with M3/M4/M5/M6.

**F0 extraction redesign (M6.11 — addresses Audit C)**. `EvolutionLoop._build_f0_probe_params` rewritten to use an env-derived probe ring. For each stationary predator, generate `probe_ring.count` (default 8) ring positions at distance `probe_ring.radius_offset + damage_radius` from the predator center. Each probe sets `predator_gradient_strength` + `predator_gradient_direction` from `(probe_pos, predator_pos)` geometry. The synthetic-probes path is fully replaced.

**Three-arm campaign + n=4-noise-aware verdict**:

| Arm | `inheritance` | F1+ train | F1+ inheritance | What it tests |
|---|---|--:|---|---|
| `tei_on` | `transgenerational` | K=0 | substrate only | **Pure-TEI floor** — does the audit-A/B/C-corrected substrate carry signal? |
| `weights_only` | `lamarckian` | K_full | F0 elite weights (M3 pattern) | M3 baseline at new env. |
| `control` | `none` | K_full | no inheritance | TPE fresh-from-scratch every gen — environmental ceiling. |

Cross-arm primary verdict: GO iff `tei_on` passes per-arm gate (F1≥40%, F2≥25%, F3≥15% × F0 training-time fitness; monotone non-increasing) AND `tei_on − control` paired-seed delta is statistically distinguishable from zero (one-sided Wilcoxon signed-rank p < 0.10 AND ≥ 5pp delta with non-overlapping 80% bootstrap CIs — both must agree). Bare 5pp threshold rejected as noise-bounded at n=4.

**Mid-flight tripwires** (catches at calibration smoke, BEFORE pilot):

1. F0 survival envelope `0.30 ≤ mean ≤ 0.70` (constructed, no wet-lab anchor).
2. Substrate diversity (pairwise CoV across 4 seeds' `bias_network.state_dict()` > 5%) — catches the M6 "3-of-4 bit-identical" failure mode.
3. M6-floor-to-beat (F0 survival exceeds M6's "circle right always" baseline on the new env).
4. Substrate magnitude (mean `|bias_network output|` over probes > 0.1) — substrate non-degenerate.

**Pre-declared pivot table** (binding at pilot review, BEFORE full campaign): six observed-pattern → pre-committed-pivot rows in design.md § D6.

**Out of scope** (deferred to PR-B / M6.13+):

- M6.12 (TEI+weights symmetric-compute control) — gated on PR-A's pure-TEI floor signal. Scaffolded as a separate OpenSpec change `add-transgenerational-memory-weights` at PR-A close. NOT bundled here.
- Brain architecture evolution (M7 NEAT).
- Multi-pathogen substrate.
- Reward shapes beyond `default` and `gradient_only`.

## Capabilities

### New Capabilities

None. PR-A extends existing capabilities; no new spec files.

### Modified Capabilities

- `evolution-framework`: extend `TransgenerationalMemory` with optional `bias_network` field; new `apply_to_logits` signature accepting `sensory_input`; new decay_shape branching in `inherit_from`; env-derived F0 probe ring in `EvolutionLoop._build_f0_probe_params`; `TransgenerationalConfig` Pydantic schema gains `bias_network.*`, `decay_shape`, `probe_ring.*` fields. M6 legacy path remains byte-equivalent when new fields are absent.
- `lstm-ppo-brain`: extend the `tei_prior` actor-logit hook to accept a callable bias-network or 1-D tensor (current). Pass current step's sensory_input through to the substrate. Default-None branch preserves byte-equivalence.
- `configuration-system`: extend `RewardConfig` with `reward_mode: Literal["default", "gradient_only"]`; extend `TransgenerationalConfig` Pydantic schema per evolution-framework changes; validators enforce shape constraints on new fields.

## Impact

- **New code**: ~880 LoC across 4 new files + 5 modified files (per [plan v2](../../../tmp/evaluations/transgenerational/transgenerational_scratchpad.md) compute envelope).
- **Modified code**: 5 existing files — `agent/transgenerational_memory.py` (+~180), `agent/reward_calculator.py` (+~30), `evolution/loop.py` (+~60), `utils/config_loader.py` (+~80), `brain/arch/lstmppo.py` (+~20). All preserve M6 byte-equivalence on the default path.
- **Brain Protocol**: unchanged (`tei_prior` attribute remains additive; non-LSTMPPO brains unaffected).
- **Existing inheritance modes** (`none`, `lamarckian`, `baldwin`): no behaviour change.
- **Env/reward path**: `RewardConfig.reward_mode = "default"` is the schema default; M3/M4/M5/M6 configs that don't set it are byte-equivalent.
- **Compute envelope**: F0 calibration smoke ~2 wall-h; pilot ~3 wall-h; full ~22-28 wall-h; post-hoc ~6 wall-h. Total PR-A ≈ **~33 wall-h**.
- **Decision-gate evaluation**: per-(arm, seed) survival_rate retention table + per-arm cross-seed verdict + cross-arm noise-aware primary verdict + PR-B trigger decision. User-reviewed before logbook 019 publication (per `feedback_logbook_review_before_verdict.md`).
- **Tests**: +55 cases across substrate (+18), reward (+6), probe ring (+10), loop smoke (+6), aggregator (+12), substrate-diversity tripwire (+4).
- **Risk**: low for framework code (additive, byte-equivalent defaults); calibration-tripwire-gated for compute spend.
