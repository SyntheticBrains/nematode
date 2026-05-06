# 016: Predator-Brain Refactor — M1 (PredatorBrain Protocol + heuristic adapter + per-predator metrics)

**Status**: `complete` — M1 GO. Refactor proven byte-equivalent at both the predator-trajectory level (23 unit tests across `PredatorType × speed` parameter grid + RNG-state snapshots) and the campaign-metric level (20 (config, seed) cells × 4 metrics; all 80 deltas exactly 0.0). M5 (co-evolution arms race) prerequisite shipped; predators can now host pluggable brains.

**Branch**: `feat/m1-predator-brain-refactor` (PR pending).

**OpenSpec change**: archived as `2026-05-06-add-learning-predators` (deltas synced to main specs in this PR).

**Date Started**: 2026-05-05.

**Date Last Updated**: 2026-05-06 — M1 closed. Refactor's behavioural impact on existing predator-bearing scenarios is identically zero; per-predator metrics + brain seam shipped without regression.

This logbook covers M1 in a single PR: a `PredatorBrain` Protocol with `HeuristicPredatorBrain` adapter, predator delegation refactor, config plumbing, and per-predator metrics in `MultiAgentEpisodeResult`. The headline finding is **identical pre/post numbers across all measurable surfaces** — predator-brain pluggability lands as pure substrate for M5 with zero behavioural cost.

## Objective

Introduce a `PredatorBrain` Protocol seam in [`packages/quantum-nematode/quantumnematode/env/predator_brain.py`](../../../packages/quantum-nematode/quantumnematode/env/predator_brain.py) so M5 (co-evolution arms race) can plug evolvable predator brains into existing scenarios without further refactoring of `Predator` or the env step loop. Ship only the substrate (Protocol + heuristic adapter + per-predator metrics + factory + dispatcher); learnable predator brains and the co-evolution loop are explicitly M5 scope.

The refactor must produce **byte-equivalent behaviour** for every existing predator-bearing scenario:

- predator trajectories step-for-step identical to pre-M1 across `PredatorType ∈ {STATIONARY, PURSUIT}` × `speed ∈ {0.5, 1.0, 2.0}`
- env RNG state advancement identical step-for-step (so downstream consumers like food spawning + agent decisions stay synchronised)
- agent-survival rate, total food collected, episode length, and predator-engagement counters per (config, seed) cell unchanged within ±0.0 (the byte-equivalence test makes this provable)

## Background

Phase 5 M2/M3/M4 (logbooks 012/013/015) all evolved agent-side behaviour while predators remained heuristic, single-class entities. M5 (co-evolution) needs predators to host evolvable brains — but `Predator.update_position` had hardcoded movement (`STATIONARY` early-return; in-range `_update_pursuit`; out-of-range `_update_random`), all coupled inline to the accumulator + grid clamp.

M1's job is to break that coupling cleanly. The substrate must accommodate learnable brains in M5 without re-touching `Predator` or the env step loop.

## Hypothesis

1. **Movement-mechanics decoupling**: a brain that returns one of `PredatorAction.{STAY, UP, DOWN, LEFT, RIGHT}` per accumulator-step, with the harness owning the kinematics (accumulator + clamp), can replicate the existing heuristic byte-for-byte.
2. **Frozen-branch invariant**: legacy `update_position` resolves the in-range / out-of-range decision **once per call** (not per accumulator-step). For multi-step movement at `speed > 1.0`, the brain abstraction must preserve this — the harness pre-resolves `chase_target` + `is_pursuing` once and passes them frozen to every brain call within the accumulator loop.
3. **Per-predator metrics keyed by synthesised IDs** can be threaded cleanly into `MultiAgentEpisodeResult` without disrupting the existing per-agent metric pattern.
4. **Pluggable brain configuration** via optional YAML `predators.brain_config` block leaves every existing scenario YAML working unchanged (defaults to `HeuristicPredatorBrain`).
5. **Regression baseline** across 20 (config, seed) cells × 4 metrics (single-agent + multi-agent pursuit scenarios) shows all deltas at floor (0.0) when comparing pre-M1.1 commit `73684213` against M1 head.

Hypothesis 1–4 → confirmed mechanically; 23 byte-equivalence unit tests across the parameter grid pass cleanly.
Hypothesis 5 → confirmed empirically; **80/80 metric-cells show exactly 0.0 delta** between pre-refactor and post-refactor baselines.

## Method

### Architecture changes

Five layers shipped under [`openspec/changes/archive/2026-05-06-add-learning-predators/`](../../../openspec/changes/archive/2026-05-06-add-learning-predators/):

1. **[`env/predator_brain.py`](../../../packages/quantum-nematode/quantumnematode/env/predator_brain.py)** (NEW): `PredatorAction` enum (cardinal-direction action), `PredatorBrainParams` frozen dataclass (predator state + frozen branch context + env RNG), `PredatorBrain` Protocol (`@runtime_checkable`), `PredatorBrainConfig` runtime dataclass, `HeuristicPredatorBrain` adapter encapsulating the legacy heuristic byte-for-byte.
2. **[`env/env.py`](../../../packages/quantum-nematode/quantumnematode/env/env.py)**: `Predator.__init__` extended with `predator_id: str` + `brain: PredatorBrain | None = None` + per-instance counters (`kills`, `prey_proximity_steps`, `distance_traveled`). `Predator.update_position` rewritten to (a) STATIONARY early-return, (b) resolve `chase_target` ONCE, (c) resolve `is_pursuing` ONCE, (d) advance accumulator loop calling `brain.run_brain` per step. Legacy `_update_random` and `_update_pursuit` deleted. New `_apply_action(action, grid_size)` helper owns the position-delta + grid clamp + distance-counter increment. `DynamicForagingEnvironment` gains `_build_predator_brain()` dispatcher (heuristic-only in M1; `NotImplementedError` for unknown kinds) and `_make_predator(...)` factory centralising all three Predator construction sites (safe-spawn, fallback-spawn, env-copy). `PredatorParams.brain_config` field added. `_initialize_predators` synthesises `predator_id = f"predator_{i}"`. `copy()` preserves source predator_id, brain (via `brain.copy()`), AND per-predator field values (predator_type/speed/detection_radius/damage_radius — needed for the existing `test_damage_radius_copied_in_env_copy` test to keep passing).
3. **[`utils/config_loader.py`](../../../packages/quantum-nematode/quantumnematode/utils/config_loader.py)**: new Pydantic `PredatorBrainConfigSchema(BaseModel)` with `kind: Literal["heuristic"]`. `PredatorConfig.brain_config: PredatorBrainConfigSchema | None = None` field. `to_params()` translates the YAML-validated schema to the runtime dataclass.
4. **[`agent/multi_agent.py`](../../../packages/quantum-nematode/quantumnematode/agent/multi_agent.py)**: `MultiAgentEpisodeResult` extended with `per_predator_kills`, `per_predator_prey_proximity_steps`, `per_predator_distance_traveled` dict fields. `MultiAgentSimulation.run_episode` step loop adds prey-proximity counter (per predator × alive agent loop after `update_predators`) and kill attribution helper `_attribute_kill_to_predator(agent_position)` (closest-by-Manhattan with lex tie-break on `predator_id`; defensive global-closest fallback for residual-HP edge cases).
5. **Example YAML** [`configs/scenarios/multi_agent_pursuit/mlpppo_small_5agents_pursuit_oracle_explicit_brain.yml`](../../../configs/scenarios/multi_agent_pursuit/mlpppo_small_5agents_pursuit_oracle_explicit_brain.yml) — identical to default-brain pursuit config but with explicit `predators.brain_config: {kind: heuristic}` block; smoke-tested via `run_simulation.py --runs 1`.

### Test coverage

| File | Tests | Coverage |
|---|---|---|
| [`test_predator_brain.py`](../../../packages/quantum-nematode/tests/quantumnematode_tests/env/test_predator_brain.py) | 21 | Protocol conformance, copy independence, stationary-STAY, pursuit greedy axis selection (all 4 directions + tie-break + at-target), random-branch RNG-draw count + direction mapping (parametrised 0/1/2/3), no-target → random, lifecycle hooks no-op |
| [`test_predator_brain_byte_equivalence.py`](../../../packages/quantum-nematode/tests/quantumnematode_tests/env/test_predator_brain_byte_equivalence.py) | 23 | **The byte-equivalence gate.** `TestByteEquivalence × {STATIONARY, PURSUIT} × {0.5, 1.0, 2.0} × {static-agent, moving-agent, multi-agent-target}` = 18 parametrised scenarios over 1000-step horizons asserting step-by-step position equality. `TestRngStateAdvancement × {0.5, 1.0, 2.0} + random-only` = 4 tests asserting `env.rng.bit_generator.state` snapshots match after every step. `TestUpdatePredatorsOrderingInvariant` (1 test) verifies env passes agent_positions in `agents.values()` insertion order. |
| [`test_predator_brain_config.py`](../../../packages/quantum-nematode/tests/quantumnematode_tests/env/test_predator_brain_config.py) | 15 | Default brain → heuristic; explicit `kind: heuristic` matches default; predator_id format + lex ordering; ID stability across `update_predators()` calls; ID reproducibility across env instances; unknown-kind rejection (runtime + Pydantic Literal); YAML dispatch (omitted block, explicit block, unknown kind); `env.copy()` preserves IDs + positions + brain copy independence. |
| [`test_multi_agent.py::TestPerPredatorMetrics`](../../../packages/quantum-nematode/tests/quantumnematode_tests/agent/test_multi_agent.py) | 6 | Distance accumulation bounds, proximity-step semantics, dict-keys-match-predator-ids, kill attribution at distinct distances (closest wins), kill attribution at equal distances (lex tie-break), defensive fallback when no predator covers. |

**Total**: 65 new tests across the four files. Full env + multi_agent regression: **369 tests pass** post-M1 (was 305 pre-M1, +64 net new — one redundant test in `test_predator_brain.py` was dropped during the M1 lint cleanup, so net is 65 new − 1 dropped = +64). Full non-nightly suite: 2425 tests pass. Smoke suite: 22 tests pass.

### Regression baseline

Two-arm campaign covering both simulation orchestration paths:

- **Multi-agent arm**: 200 episodes × 3 multi-agent pursuit scenarios × 4 seeds (42, 43, 44, 45). Configs:
  - [`mlpppo_small_5agents_pursuit_oracle.yml`](../../../configs/scenarios/multi_agent_pursuit/mlpppo_small_5agents_pursuit_oracle.yml)
  - [`mlpppo_small_5agents_pursuit_no_alarm_oracle.yml`](../../../configs/scenarios/multi_agent_pursuit/mlpppo_small_5agents_pursuit_no_alarm_oracle.yml)
  - [`lstmppo_small_5agents_pursuit_alarm_klinotaxis.yml`](../../../configs/scenarios/multi_agent_pursuit/lstmppo_small_5agents_pursuit_alarm_klinotaxis.yml)
- **Single-agent arm**: 100 episodes × 2 single-agent pursuit scenarios × 4 seeds. Configs:
  - [`pursuit/mlpppo_small_oracle.yml`](../../../configs/scenarios/pursuit/mlpppo_small_oracle.yml)
  - [`pursuit/lstmppo_small_klinotaxis.yml`](../../../configs/scenarios/pursuit/lstmppo_small_klinotaxis.yml) (this one includes STAM)

Total: 20 cells. Pre-refactor measurement on commit `73684213` (pre-M1.1); post-refactor on M1 HEAD (`3d45e75c`). Each cell extracts four metrics (rationale: `mean_alive_rate` saturates near 0.0 on the multi-agent arm under heuristic predator pressure; richer metrics discriminate behavioural drift):

- `mean_success`: per-run success indicator (multi-agent: `mean_success` from `multi_agent_summary.csv`; single-agent: success-bool aggregated from `simulation_results.csv`)
- `mean_total_food`: foods collected per episode
- `mean_steps`: episode length (single-agent only; the multi-agent CSV has no per-run steps column)
- `mean_predator_engagement`: arm-specific composite (multi-agent: `proximity_events / num_episodes`; single-agent: `(predator_encounters + successful_evasions) / num_episodes`)

Both runs use `--theme headless` (no rendering). Wall-time ~30 min per arm (~60 min total per pre/post round, ~2 hours for the full pre+post regression budget).

## Results

### Byte-equivalence gate (PRIMARY)

23/23 byte-equivalence tests pass. Position trajectories step-by-step identical to legacy across all parametrisations; env RNG state matches one-for-one after every step.

```text
TestByteEquivalence::test_static_agent_byte_equivalent[predator_type0-speed0]            PASSED  (STATIONARY, 0.5)
TestByteEquivalence::test_static_agent_byte_equivalent[predator_type0-speed1]            PASSED  (STATIONARY, 1.0)
TestByteEquivalence::test_static_agent_byte_equivalent[predator_type0-speed2]            PASSED  (STATIONARY, 2.0)
TestByteEquivalence::test_static_agent_byte_equivalent[predator_type1-speed0]            PASSED  (PURSUIT,    0.5)
TestByteEquivalence::test_static_agent_byte_equivalent[predator_type1-speed1]            PASSED  (PURSUIT,    1.0)
TestByteEquivalence::test_static_agent_byte_equivalent[predator_type1-speed2]            PASSED  (PURSUIT,    2.0)
TestByteEquivalence::test_moving_agent_byte_equivalent          [6 combinations]         PASSED  (all)
TestByteEquivalence::test_multi_agent_targeting_byte_equivalent [6 combinations]         PASSED  (all)
TestRngStateAdvancement::test_rng_state_identical_across_steps_pursuit[0.5/1.0/2.0]      PASSED
TestRngStateAdvancement::test_rng_state_identical_across_steps_random                    PASSED
TestUpdatePredatorsOrderingInvariant::test_predator_targets_lex_first_equidistant_agent  PASSED
```

### Regression-baseline gate (SECONDARY)

20 (arm, config, seed) cells × 4 metrics = 80 metric-cells. Pre-refactor measured on commit `73684213` (pre-M1.1); post-refactor on M1 HEAD (`3d45e75c`).

**Result: every single delta is exactly 0.0.** Per-metric stats:

| Metric | max|Δ| | mean|Δ| | any non-zero |
|---|---|---|---|
| `mean_success` | 0.000000 | 0.000000 | False |
| `mean_total_food` | 0.000000 | 0.000000 | False |
| `mean_steps` | 0.000000 | 0.000000 | False |
| `mean_predator_engagement` | 0.000000 | 0.000000 | False |

Per-cell pre/post values (showing the full table for completeness — every row matches):

| Arm | Config | Seed | mean_success | mean_total_food | mean_steps | mean_predator_engagement | n_episodes |
|---|---|---|---|---|---|---|---|
| multi | mlpppo_5agents_oracle | 42 | 0.616 | 37.600 | 0.000 | 626.950 | 200 |
| multi | mlpppo_5agents_oracle | 43 | 0.569 | 36.350 | 0.000 | 614.770 | 200 |
| multi | mlpppo_5agents_oracle | 44 | 0.578 | 36.635 | 0.000 | 622.915 | 200 |
| multi | mlpppo_5agents_oracle | 45 | 0.564 | 35.880 | 0.000 | 614.410 | 200 |
| multi | mlpppo_5agents_no_alarm | 42 | 0.597 | 37.100 | 0.000 | 607.285 | 200 |
| multi | mlpppo_5agents_no_alarm | 43 | 0.575 | 37.315 | 0.000 | 631.685 | 200 |
| multi | mlpppo_5agents_no_alarm | 44 | 0.578 | 35.760 | 0.000 | 589.765 | 200 |
| multi | mlpppo_5agents_no_alarm | 45 | 0.573 | 36.830 | 0.000 | 621.435 | 200 |
| multi | lstmppo_5agents_klinotaxis | 42 | 0.813 | 42.415 | 0.000 | 789.350 | 200 |
| multi | lstmppo_5agents_klinotaxis | 43 | 0.818 | 42.395 | 0.000 | 762.315 | 200 |
| multi | lstmppo_5agents_klinotaxis | 44 | 0.828 | 43.340 | 0.000 | 840.900 | 200 |
| multi | lstmppo_5agents_klinotaxis | 45 | 0.810 | 42.255 | 0.000 | 773.715 | 200 |
| single | mlpppo_small_oracle | 42 | 0.380 | 6.420 | 157.970 | 5.510 | 100 |
| single | mlpppo_small_oracle | 43 | 0.310 | 5.820 | 160.480 | 4.910 | 100 |
| single | mlpppo_small_oracle | 44 | 0.300 | 5.050 | 122.970 | 3.400 | 100 |
| single | mlpppo_small_oracle | 45 | 0.450 | 6.580 | 168.200 | 6.750 | 100 |
| single | lstmppo_small_klinotaxis | 42 | 0.150 | 3.040 | 171.720 | 4.430 | 100 |
| single | lstmppo_small_klinotaxis | 43 | 0.160 | 3.440 | 175.500 | 5.060 | 100 |
| single | lstmppo_small_klinotaxis | 44 | 0.150 | 3.120 | 168.470 | 3.650 | 100 |
| single | lstmppo_small_klinotaxis | 45 | 0.220 | 3.950 | 205.480 | 4.770 | 100 |

The `mean_steps` column for the multi-agent arm is 0.0 because `multi_agent_summary.csv` doesn't have a per-run aggregate steps column. This is a known data-availability constraint, not a regression — both pre and post values are exactly 0.0 for that column on the multi-agent arm by construction.

### Wall-time

| Phase | Wall |
|---|---|
| Pre-refactor baseline (20 cells, headless) | ~30 min |
| Post-refactor baseline (20 cells, headless) | ~40 min (small per-cell overhead from extra brain dispatch + per-predator counter increments; not significant) |
| Unit-test suite (full env + multi_agent) | ~5 sec |
| Smoke suite | ~80 sec |

## Analysis

### Decision: GO ✅

The refactor produces zero observable behavioural change. Both the predator-trajectory level (byte-equivalence unit tests) and the campaign-metric level (4 metrics × 20 cells) confirm identity. M5 can build on this substrate without worrying about silent baseline drift.

### Why both gates ended up at floor

The multi-agent arm's `mean_success` saturating at 0.575-0.825 (not 0.0) was a noteworthy finding from the in-flight baseline — earlier session notes had identified `mean_alive_rate` as floor-saturated, but the richer four-metric extraction shows agents do collect food (~36-43 per episode at 5 agents) and partially survive predator pressure. This means the campaign-baseline gate is genuinely discriminating; the 0.0 deltas reflect actual behavioural identity rather than vacuous floor-pass.

The byte-equivalence unit tests are the stronger guarantee — they prove that `Predator.update_position` produces step-for-step identical positions AND advances `env.rng.bit_generator.state` identically to the legacy implementation across the full parameter grid. Given those properties, the campaign-level metric identity follows mathematically (the agent's policy, food spawning, etc., depend on env's RNG state + agent positions, neither of which can drift if predator behaviour is byte-identical).

### Frozen-branch invariant — the subtle bit that almost broke

Mid-implementation, while reading the legacy `_update_pursuit` carefully, I discovered the original design's "brain decides per accumulator-step" pattern would have silently diverged from legacy on multi-step movement at `speed > 1.0`. Legacy `update_position` resolves the in-range / out-of-range decision **once at the top of the call**; if a predator at `speed=2.0` moves out of detection radius mid-call, it still completes its remaining accumulator steps in greedy mode (not random).

Fix: extend `PredatorBrainParams` with `chase_target` (frozen target) and `is_pursuing` (frozen branch flag) — both pre-resolved by `update_position` once per call and passed unchanged across every brain invocation in the accumulator loop. Only `predator_position` varies between iterations. The byte-equivalence test parametrisation on `speed ∈ {0.5, 1.0, 2.0}` is exactly what catches this: `speed=2.0` exercises the multi-step regime, and the random-direction branch (which IS RNG-consuming) would diverge if branching happened per accumulator-step.

Documented as Decision 7 in [`design.md`](../../../openspec/changes/archive/2026-05-06-add-learning-predators/design.md) and as scenario "PredatorBrainParams Surface" in the modified `environment-simulation/spec.md`.

### `_make_predator` factory — caught a regression mid-implementation

Initial M1.3 wiring used `Predator(...)` constructor calls directly at three sites (two in `_initialize_predators`, one in `copy()`). The existing `test_damage_radius_copied_in_env_copy` test (env.py legacy suite) failed because my `copy()` was reading `damage_radius` from `self.predator` (the env-level `PredatorParams`), not from the source predator's actual field value. Tests like that one pre-date M1 and mutate `env.predators[0].damage_radius` directly to test runtime mutability.

Fix: extended the `_make_predator` factory to accept per-predator field overrides (`predator_type`, `speed`, `detection_radius`, `damage_radius`) — each defaulting to `None` and falling back to `self.predator.*` when not provided. `_initialize_predators` uses defaults (None values → env-level config); `copy()` passes the source predator's actual field values explicitly. End state preserves both the legacy mutation test and the M1 invariant that `_initialize_predators` synthesises predators from config.

Documented as Decision 8 in `design.md`.

### Per-predator metric fidelity

Per-predator metrics are populated by the simulation harness, not the env. `Predator._apply_action` increments `self.distance_traveled` only when post-clamp position differs from pre-action position (so STAY actions and wall-blocked moves contribute 0). The simulation step loop iterates `env.predators × alive_agents` post-`update_predators()` to bump `prey_proximity_steps` by 1 per predator per step that has any prey in detection radius. Kill attribution is sim-side post-call (the env's `apply_predator_damage_for(aid)` doesn't know which predator caused the damage); the new `_attribute_kill_to_predator(agent_position)` helper finds the closest covering predator with lex tie-break on `predator_id`.

The metric values flow into `MultiAgentEpisodeResult` via `_build_result`. Tests verify all three properties: distance bounds, proximity counters, kill attribution under distinct + equal distances.

### Carry-forward to M5

M5 (co-evolution arms race) starts with this substrate fully in place:

1. `PredatorBrain` Protocol surface is final — M5 adds new `Brain` implementations (e.g. `MLPPPOPredatorBrain`) without touching `Predator` or the env step loop.
2. `_build_predator_brain` dispatcher is the M5 extension point — add new `kind` branches (`"mlpppo"`, `"lstmppo"`, etc.) alongside the existing `"heuristic"` case. Pydantic `Literal` extends similarly.
3. `env.copy()` already preserves brain state via `pred.brain.copy()` so future stateful (learnable) brains keep their per-instance state across env-snapshot / evolution-loop replay.
4. Per-predator metrics are ready as M5 fitness signals — `per_predator_kills` is a natural predator-fitness target; `per_predator_prey_proximity_steps` measures hunting effort independently of success.
5. The `_make_predator` factory centralises Predator construction so M5 only modifies one site.

## Conclusions

1. **Predator-brain seam shipped with zero behavioural cost.** Byte-equivalence proven at two levels (predator-trajectory unit tests + campaign metric deltas), both at floor 0.0 across the full parameter grid and all 20 (config, seed) cells.
2. **Frozen-branch invariant** is the load-bearing design decision. `chase_target` + `is_pursuing` pre-resolved per-call and passed frozen across the accumulator loop preserves legacy semantics on multi-step movement at `speed > 1.0`. Without it, the random-branch RNG draws would have desynchronised silently.
3. **`_make_predator` factory is the right abstraction** — caught a real regression mid-implementation (`test_damage_radius_copied_in_env_copy`) and centralises the three current Predator construction sites for M5.
4. **Per-predator metrics fit the existing per-agent dict pattern cleanly.** `MultiAgentEpisodeResult.{per_predator_kills, per_predator_prey_proximity_steps, per_predator_distance_traveled}` mirror `per_agent_food` / `per_agent_reward` exactly. Kill attribution sim-side (closest-by-Manhattan, lex tie-break) is principled and testable.
5. **Pluggable YAML config + Pydantic Literal validation** rejects unknown brain kinds at YAML load (M5 forward-compat). No existing scenario YAML touched.
6. **Test count went from 305 to 369** — 64 net new tests (65 added across 4 new test files − 1 redundant test dropped during lint cleanup). Coverage: Protocol conformance, byte-equivalence, config plumbing, per-predator metrics. Full env + multi_agent suite green; full non-nightly suite (2425) green; smoke clean.
7. **The campaign-baseline approach was vindicated** — earlier session notes worried about `mean_alive_rate` saturation, but the four-metric extraction (success, food, steps, predator_engagement) discriminates real behavioural change. Both pre and post baselines show meaningful per-config variation; the 0.0 delta isn't vacuous.
8. **Wall-time impact**: post-refactor baseline ran ~10 min slower than pre-refactor (~40 min vs ~30 min) due to extra per-predator counter increments in the sim step loop. Negligible in absolute terms; not a concern for M5 evolution loops where sim-time is tiny vs PPO training overhead.
9. **M5 unblocked.** Predator-as-brain refactor is complete; co-evolution can proceed on this substrate.

## Next Steps

- [x] M1.1-M1.8 ticked in [`openspec/changes/2026-04-26-phase5-tracking/tasks.md`](../../../openspec/changes/2026-04-26-phase5-tracking/tasks.md); roadmap M1 row flipped to ✅ complete
- [x] OpenSpec change `add-learning-predators` validated `--strict`; ready for archival on PR merge
- [ ] M5 (co-evolution arms race) starts on this substrate. Predator brains plug in via `_build_predator_brain` dispatcher; per-predator metrics provide fitness signal candidates; brain-state copy semantics already in place for env-copy boundaries
- [ ] Future PR (post-M5): widen `PredatorBrainConfigSchema.kind` Literal to include learnable kinds (e.g. `"mlpppo"`); add corresponding dispatcher branches in `_build_predator_brain`

## Data References

### Regression-baseline campaign

- **Pre-refactor CSV** ([commit `73684213`](https://github.com/SyntheticBrains/nematode/commit/73684213)): [`artifacts/logbooks/016-predator-brain-refactor/baseline_pre.csv`](../../../artifacts/logbooks/016-predator-brain-refactor/baseline_pre.csv)
- **Post-refactor CSV** ([commit `3d45e75c`](https://github.com/SyntheticBrains/nematode/commit/3d45e75c)): [`artifacts/logbooks/016-predator-brain-refactor/baseline_post.csv`](../../../artifacts/logbooks/016-predator-brain-refactor/baseline_post.csv)
- **Runner script archived**: [`artifacts/logbooks/016-predator-brain-refactor/run_baseline.sh`](../../../artifacts/logbooks/016-predator-brain-refactor/run_baseline.sh)
- **Forensic notes** (saturation finding, raw per-cell numbers, intermediate decisions): `tmp/evaluations/evolution/evolution_scratchpad.md` § 2026-05-05 — M1. This file is gitignored under the repo's `tmp/` policy; the section persists in the local working tree of the contributor who wrote it. If you need the raw forensic trace and don't have a local copy, the relevant numbers are reproducible from the committed CSVs at `artifacts/logbooks/016-predator-brain-refactor/baseline_{pre,post}.csv`.

### Framework artefacts

- **OpenSpec change** (archived in this PR with deltas synced to main specs): [`openspec/changes/archive/2026-05-06-add-learning-predators/`](../../../openspec/changes/archive/2026-05-06-add-learning-predators/)
- **Spec deltas** synced into:
  - [`openspec/specs/environment-simulation/spec.md`](../../../openspec/specs/environment-simulation/spec.md) — adds "Predator Brain Abstraction" + "Predator ID Synthesis" requirements; modifies "Predator Entities in Dynamic Environments" with brain seam + ID synthesis scenarios
  - [`openspec/specs/multi-agent/spec.md`](../../../openspec/specs/multi-agent/spec.md) — adds "Per-Predator Metrics in MultiAgentEpisodeResult" + "Multi-Predator Kill Attribution Rule" requirements; modifies "Multi-Agent Metrics and Results" with per-predator scenario
- **Predator brain module**: [`packages/quantum-nematode/quantumnematode/env/predator_brain.py`](../../../packages/quantum-nematode/quantumnematode/env/predator_brain.py)
- **Tests** (65 new across 4 files; full env+multi_agent suite 369 passing post-M1, was 305 pre-M1; full non-nightly suite 2425):
  - [`test_predator_brain.py`](../../../packages/quantum-nematode/tests/quantumnematode_tests/env/test_predator_brain.py) — 21 tests
  - [`test_predator_brain_byte_equivalence.py`](../../../packages/quantum-nematode/tests/quantumnematode_tests/env/test_predator_brain_byte_equivalence.py) — 23 tests (the gate)
  - [`test_predator_brain_config.py`](../../../packages/quantum-nematode/tests/quantumnematode_tests/env/test_predator_brain_config.py) — 15 tests
  - [`test_multi_agent.py::TestPerPredatorMetrics`](../../../packages/quantum-nematode/tests/quantumnematode_tests/agent/test_multi_agent.py) — 6 tests
  - [`_legacy_predator_reference.py`](../../../packages/quantum-nematode/tests/quantumnematode_tests/env/_legacy_predator_reference.py) — frozen pre-M1 implementation, used by the byte-equivalence test as the reference fixture
- **Example YAML config**: [`configs/scenarios/multi_agent_pursuit/mlpppo_small_5agents_pursuit_oracle_explicit_brain.yml`](../../../configs/scenarios/multi_agent_pursuit/mlpppo_small_5agents_pursuit_oracle_explicit_brain.yml)
