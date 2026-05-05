## 1. PredatorBrain Protocol module

- [x] 1.1 Create `packages/quantum-nematode/quantumnematode/env/predator_brain.py` with `PredatorAction` enum (`STAY`, `UP`, `DOWN`, `LEFT`, `RIGHT`)
- [x] 1.2 Add `PredatorBrainParams` frozen dataclass: `predator_id`, `predator_position`, `predator_type`, `detection_radius`, `damage_radius`, `agent_positions`, `chase_target`, `is_pursuing`, `grid_size`, `rng`, `step_index`. `chase_target` and `is_pursuing` are pre-resolved by `update_position` ONCE per call and passed unchanged across all accumulator-loop iterations within that call (frozen branch invariant — preserves byte-equivalence to legacy `update_position` which decides branch once per call rather than per accumulator-step)
- [x] 1.3 Add `PredatorBrain` Protocol decorated `@runtime_checkable` with `run_brain`, `prepare_episode`, `post_process_episode`, `copy`
- [x] 1.4 Add `HeuristicPredatorBrain` class encapsulating the existing `_update_pursuit` / `_update_random` logic byte-for-byte (single `rng.integers(4)` draw per call on the random branch; greedy axis selection on the pursuit branch with `if abs(dx) >= abs(dy)` precedence)
- [x] 1.5 Append `PredatorBrain`, `HeuristicPredatorBrain`, `PredatorBrainParams`, `PredatorAction` to the existing `__all__` list in `env/__init__.py` (existing exports `PredatorParams`, `PredatorType` at lines 20-21, 40-41 — extend in place rather than creating a new list). Also added `PredatorBrainConfig` for M1.3's config plumbing
- [x] 1.6 Tests: Protocol conformance via `@runtime_checkable` `isinstance`; `HeuristicPredatorBrain.copy()` returns independent instance; stationary returns STAY; in-range pursuit greedy axis selection; out-of-range pursuit falls back to random with seeded RNG → expected direction; random branch consumes exactly one RNG draw per call. **21 tests pass; full env test suite (307 tests) clean.**

## 2. Predator delegation + legacy removal

- [x] 2.0 Add `DynamicForagingEnvironment._make_predator(self, predator_id: str, position: tuple[int, int], *, movement_accumulator: float = 0.0, brain: PredatorBrain | None = None, predator_type=None, speed=None, detection_radius=None, damage_radius=None) -> Predator` factory. Centralises construction so all three call sites (env.py:1505, env.py:1526, env.py:3643 — the `copy()` method) call through the factory. Per-predator field overrides (predator_type/speed/detection_radius/damage_radius) accept `None` and fall back to env-level `self.predator.*` defaults; `copy()` passes the source predator's actual field values to preserve post-init mutations across env-copy boundaries (this preserves the `test_damage_radius_copied_in_env_copy` legacy test invariant). Implemented alongside `_build_predator_brain` dispatcher in M1.3
- [x] 2.1 Extend `Predator.__init__` (env.py:495) with `predator_id: str` and `brain: PredatorBrain | None = None`; init `kills`/`prey_proximity_steps`/`distance_traveled` counters to 0
- [x] 2.2 Add `Predator._apply_action(action, grid_size, rng)` containing the accumulator advance, multi-step-per-update loop (capped at 10), and `max(0, ...)` / `min(grid_size-1, ...)` grid clamp extracted from the legacy `_update_random` / `_update_pursuit` helpers — implemented as `_apply_action_loop` (the loop) + `_apply_action` (the per-step kinematics with grid clamp); split for clarity
- [x] 2.3 Refactor `Predator.update_position` (env.py:529) to: (a) early-return on STATIONARY, (b) compute `chase_target` ONCE from `agent_positions` (or fall back to `agent_pos` for backward-compat single-agent path) using `min(... key=Manhattan)`, (c) compute `is_pursuing = (predator_type == PURSUIT and chase_target is not None and Manhattan(self.position, chase_target) <= detection_radius)` ONCE, (d) advance the accumulator loop in `_apply_action_loop`, calling `self.brain.run_brain(params)` once per accumulator-step with `PredatorBrainParams` carrying the FROZEN `chase_target` + `is_pursuing` fields plus the CURRENT `predator_position`. The frozen-branch invariant is the byte-equivalence requirement: legacy `_update_pursuit` branches once per call (not per accumulator-step)
- [x] 2.4 Delete `Predator._update_random` (env.py:570-617) and `Predator._update_pursuit` (env.py:619-681) — `HeuristicPredatorBrain` is now the single source of truth
- [x] 2.5 Update `DynamicForagingEnvironment.copy()` (env.py:3582-3653 — actual method name is `copy()`, NOT `copy_environment()`; tasks doc was incorrect) to call the factory with explicit per-predator field threading (predator_id, position, movement_accumulator, brain via `p.brain.copy()`, predator_type, speed, detection_radius, damage_radius). Without per-field threading, post-init mutations to individual predator fields would be lost across env-copy boundaries (caught by the existing `test_damage_radius_copied_in_env_copy` test in the legacy suite)
- [x] 2.6 Byte-equivalence test (the gate): `test_legacy_path_byte_equivalent_to_new_path` parametrised across `PredatorType ∈ {STATIONARY, PURSUIT}` × `speed ∈ {0.5, 1.0, 2.0}`. Implemented in `test_predator_brain_byte_equivalence.py` with **3 test classes covering 18 parametrised combinations of position trajectory equality + RNG state equality**. All pass. Includes static-agent, moving-agent, and multi-agent-targeting test scenarios over 1000-step horizons; compares `env.rng.bit_generator.state` snapshots after every step
- [x] 2.7 Add direct ordering-invariant test: `test_update_predators_passes_agents_in_insertion_order` — verifies `update_predators()` builds `agent_positions` matching `tuple((a.position[0], a.position[1]) for a in self.agents.values() if a.alive)` so target tie-breaking is deterministic across env reconstructions. Implemented as `TestUpdatePredatorsOrderingInvariant` in the byte-equivalence test file
- [x] 2.8 Run `uv run pytest -m "not nightly" packages/quantum-nematode/tests/quantumnematode_tests/env/test_predator_brain*.py -v` — green: 21 (test_predator_brain.py) + 23 (test_predator_brain_byte_equivalence.py) = **44 tests pass**. Full env suite (330 tests) regresses cleanly. `pytest -m smoke -v` clean (22 tests).

## 3. Config plumbing

- [x] 3.1 Added `PredatorBrainConfig` dataclass to `env/predator_brain.py` (already shipped in M1.1 — see `env/__init__.py` exports). `kind: Literal["heuristic"]`; optional `extra: dict[str, Any] | None`. Runtime dataclass; YAML schema is `PredatorBrainConfigSchema` in config_loader.py (3.5). Two-type pattern matches existing `PredatorParams` ↔ `PredatorConfig` split
- [x] 3.2 Extend `PredatorParams` (env.py:154) with `brain_config: PredatorBrainConfig | None = None`
- [x] 3.3 Add `_build_predator_brain(self)` dispatcher in `DynamicForagingEnvironment`: returns `HeuristicPredatorBrain()` for `brain_config is None` or `kind == "heuristic"`; raises `NotImplementedError` for other kinds (M5 extends)
- [x] 3.4 Modify `_initialize_predators` to call `self._make_predator(predator_id=f"predator_{i}", position=candidate)` for both the safe-spawn branch and the fallback branch. Loop index `i` (0-based) determines the synthesised ID. The factory handles brain construction internally via `_build_predator_brain`
- [x] 3.5 Extended `PredatorConfig` (config_loader.py:311) with optional `brain_config: PredatorBrainConfigSchema | None = None`. New `PredatorBrainConfigSchema(BaseModel)` defined alongside; Pydantic Literal["heuristic"] rejects unknown kinds at YAML load time
- [x] 3.6 Extended `PredatorConfig.to_params()` to translate `brain_config` to runtime `PredatorBrainConfig` (no-op when `None`)
- [x] 3.7 Tests: 15 tests in `test_predator_brain_config.py` covering default brain → heuristic; explicit kind: heuristic; predator_id synthesis matches `f"predator_{i}"`; lex ordering; ID stability across `update_predators()` calls within the same env instance (env has no top-level `reset()`; predators persist as instances); ID reproducibility + same spawn positions across two env instances with the same config + seed; unknown kind rejection (runtime + Pydantic-level); `env.copy()` preserves IDs + per-predator field values + brain copy independence. **All 15 pass; full env suite (345) regresses cleanly**
- [x] 3.8 Added example YAML config `configs/scenarios/multi_agent_pursuit/mlpppo_small_5agents_pursuit_oracle_explicit_brain.yml` — identical to the default-brain pursuit config but with an explicit `predators.brain_config: {kind: heuristic}` block. Smoke-tested via `run_simulation.py --runs 1`

## 4. Per-predator metrics

- [x] 4.1 Extended `MultiAgentEpisodeResult` (multi_agent.py:119) with three `dict[str, int]` fields using `field(default_factory=dict)`: `per_predator_kills`, `per_predator_prey_proximity_steps`, `per_predator_distance_traveled`. Docstring updated
- [x] 4.2 `Predator._apply_action` increments `self.distance_traveled` by 1 only when the post-clamp position differs from the pre-action position (already shipped in M1.2; this task confirms the behaviour and is verified by M1.4 metric tests)
- [x] 4.3 In `MultiAgentSimulation.run_episode` step loop, after `self.env.update_predators()`: predator × alive-agent loop increments each `pred.prey_proximity_steps` by 1 if any alive agent is within `pred.detection_radius` (Manhattan). Uses `break` after first match so stationary predators with multiple agents in range count once per step
- [x] 4.4 Implemented sim-side kill attribution via new `MultiAgentSimulation._attribute_kill_to_predator(agent_position)` helper. Phase 1: covering predators (Manhattan ≤ `damage_radius`) → closest by Manhattan, lex tie-break on `predator_id`. Phase 2 (defensive fallback): if no covering predator, global-closest with same tie-break + debug log. Increments both sim-local `_kills_by_predator[id]` and per-Predator-instance `kills` counter
- [x] 4.5 In `MultiAgentSimulation._build_result`, copy per-predator counters from `self.env.predators` (distance + proximity) and the sim-local kill dict into the new `MultiAgentEpisodeResult` fields
- [x] 4.6 Tests in `tests/quantumnematode_tests/agent/test_multi_agent.py::TestPerPredatorMetrics`: 6 tests covering distance accumulation bounds, proximity-step semantics, all-three-dicts-keyed-by-predator-id invariant, kill attribution with distinct distances, lex tie-break on equal distances, and defensive fallback when no predator covers. **All 6 pass; full multi_agent + env regression: 370 tests pass (was 345); no regressions**

## 5. Regression baseline + logbook

The byte-equivalence parametrised test (task 2.6) is the **primary** regression gate (catches RNG-ordering and trajectory-drift bugs at the predator level). The campaign runs below are the **secondary** sanity-check layer that exercise the predator code through both `MultiAgentSimulation` and the single-agent `Simulation` call paths under real training dynamics.

- [x] 5.1 Pre-refactor baseline campaign: 4 seeds (42, 43, 44, 45) on the M1 base commit (before delegation refactor lands). Run with `--theme headless`. Estimated wall-time: ~25-30 min total at headless rates. Two arms covering both simulation-orchestration paths:

  - **Multi-agent arm**: 200 episodes × 3 multi-agent pursuit scenarios (`mlpppo_small_5agents_pursuit_oracle.yml`, `mlpppo_small_5agents_pursuit_no_alarm_oracle.yml`, `lstmppo_small_5agents_pursuit_alarm_klinotaxis.yml`). Higher episode count vs. single-agent arm because multi-agent variance is higher: 5 agents per episode produce 5 independent food-seeking trajectories, so per-episode metric stds are larger and need more episodes to converge per-cell
  - **Single-agent arm**: 100 episodes × 2 single-agent pursuit scenarios (`pursuit/mlpppo_small_oracle.yml`, `pursuit/lstmppo_small_klinotaxis.yml`) — closes the gap that the multi-agent arm leaves: single-agent `Simulation` invokes `env.update_predators()` from a different orchestration path. Note: `pursuit/lstmppo_small_klinotaxis.yml` includes STAM (short-term associative memory); if regression drift surfaces here but not on `pursuit/mlpppo_small_oracle.yml`, suspect STAM-adjacent code rather than the predator refactor itself

  The two arms write different CSVs with different schemas: multi-agent path emits `multi_agent_summary.csv` (cols include `agents_alive_at_end`, `total_food`, `proximity_events`, `mean_success`); single-agent path emits `simulation_results.csv` (cols include `success`, `foods_collected`, `steps`, `predator_encounters`, `died_to_health_depletion`, `successful_evasions`). Metrics extracted in 5.3 below are reconciled across both schemas

  Output: `tmp/m1_regression_baseline/baseline_pre.csv` with columns `arm, config, seed, mean_success, mean_total_food, mean_steps, mean_predator_engagement, n_episodes, session_id`. `mean_predator_engagement` aliases differently per arm — multi-agent reads `proximity_events / num_episodes` (already in summary CSV); single-agent reads `(predator_encounters + successful_evasions) / num_episodes` (composite signal of predator-trajectory effects). `mean_success` will saturate at 0.0 on the multi-agent pursuit arm (heuristic predators kill all 5 agents reliably under default config); single-agent arm is non-saturated and provides discriminating signal

- [x] 5.2 Post-refactor baseline campaign: identical protocol on the M1 head commit (after all M1.1-M1.4 tasks land). Output: `tmp/m1_regression_baseline/baseline_post.csv`

- [x] 5.3 Compute deltas across all four metrics, per-cell. **Result: every single delta is exactly 0.0 across all 4 metrics × 20 (config, seed) cells = 80 metric-cells.** No tolerance adjustment needed — the byte-equivalence gate (task 2.6) holds at the full-simulation level. Initial pre-registered tolerances:

  - `mean_success`: per-cell `|post − pre| ≤ 0.02` (vacuous-pass on saturated cells; flagged in logbook 016 narrative)
  - `mean_total_food`: per-cell `|post − pre| ≤ 0.5 food units` (discriminates predator-induced foraging-path changes; non-saturated on both arms because agents collect food before dying)
  - `mean_steps`: per-cell `|post − pre| ≤ 5 steps` (single-agent arm only — `multi_agent_summary.csv` does not have a per-run `steps` column; the script writes 0.0 for multi-agent cells, so multi-agent `mean_steps` deltas are noise-free by construction. Single-agent arm carries the timing-discrimination work)
  - `mean_predator_engagement`: per-cell tolerance arm-specific — multi-agent `|post − pre| ≤ 5 events` (proximity-events scale, from `proximity_events / num_episodes`); single-agent `|post − pre| ≤ 0.5 encounters/episode` (composite of `predator_encounters + successful_evasions`, both per-episode counts)

  Per-scenario means use the same per-cell tolerances. Any per-cell metric exceeding tolerance triggers a forensic step-by-step trajectory diff against `git stash`-able legacy reference for the failing (config, seed) cell. The byte-equivalence test (2.6) is expected to flag any drift before this stage; multi-metric campaign serves as cross-check.

  **The byte-equivalence test (2.6) is the PRIMARY regression gate**; the multi-metric campaign here is a secondary cross-check that exercises both `MultiAgentSimulation` and single-agent `Simulation` code paths under real training dynamics

- [x] 5.4 Wrote `docs/experiments/logbooks/016-predator-brain-refactor.md` documenting framework changes, both gates' results (byte-equivalence + multi-metric campaign), and carry-forward to M5

- [x] 5.5 Baseline CSVs staged under `artifacts/logbooks/016-predator-brain-refactor/{baseline_pre,baseline_post}.csv`; runner script archived at `artifacts/logbooks/016-predator-brain-refactor/run_baseline.sh`. Forensic notes in `tmp/evaluations/evolution/evolution_scratchpad.md` (gitignored)

## 6. Smoke + final checks

- [x] 6.1 `uv run pytest -m smoke -v` clean (22 tests pass, 73s)
- [x] 6.2 `uv run pytest -m "not nightly"` clean (2425 tests pass, 2m22s — includes the byte-equivalence and per-predator-metric tests)
- [x] 6.3 `uv run pre-commit run -a` clean (mdformat + markdownlint + ruff check + ruff format + pyright + tests all green)
- [x] 6.4 `openspec validate add-learning-predators --strict` clean

## 7. Tracker + roadmap update

- [x] 7.1 Ticked M1.1-M1.8 sub-tasks in `openspec/changes/2026-04-26-phase5-tracking/tasks.md` (M1 section)
- [x] 7.2 Flipped M1 row in `docs/roadmap.md` Phase 5 milestone tracker table from 🟡 next to ✅ complete
- [x] 7.3 Added detailed result summary to the M1 row referencing logbook 016, byte-equivalence gate, and 80/80 zero-delta regression-baseline outcome
