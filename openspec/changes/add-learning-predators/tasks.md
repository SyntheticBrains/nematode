## 1. PredatorBrain Protocol module

- [ ] 1.1 Create `packages/quantum-nematode/quantumnematode/env/predator_brain.py` with `PredatorAction` enum (`STAY`, `UP`, `DOWN`, `LEFT`, `RIGHT`)
- [ ] 1.2 Add `PredatorBrainParams` frozen dataclass: `predator_id`, `predator_position`, `predator_type`, `detection_radius`, `damage_radius`, `agent_positions`, `grid_size`, `rng`, `step_index`
- [ ] 1.3 Add `PredatorBrain` Protocol decorated `@runtime_checkable` with `run_brain`, `prepare_episode`, `post_process_episode`, `copy`
- [ ] 1.4 Add `HeuristicPredatorBrain` class encapsulating the existing `_update_pursuit` / `_update_random` logic byte-for-byte (single `rng.integers(4)` draw per call on the random branch; greedy axis selection on the pursuit branch with `if abs(dx) >= abs(dy)` precedence)
- [ ] 1.5 Re-export `PredatorBrain`, `HeuristicPredatorBrain`, `PredatorBrainParams`, `PredatorAction` from `env/__init__.py`
- [ ] 1.6 Tests: Protocol conformance via `@runtime_checkable` `isinstance`; `HeuristicPredatorBrain.copy()` returns independent instance; stationary returns STAY; in-range pursuit greedy axis selection; out-of-range pursuit falls back to random with seeded RNG → expected direction; random branch consumes exactly one RNG draw per call

## 2. Predator delegation + legacy removal

- [ ] 2.1 Extend `Predator.__init__` (env.py:495) with `predator_id: str` and `brain: PredatorBrain | None = None`; init `kills`/`prey_proximity_steps`/`distance_traveled` counters to 0
- [ ] 2.2 Add `Predator._apply_action(action, grid_size, rng)` containing the accumulator advance, multi-step-per-update loop (capped at 10), and `max(0, ...)` / `min(grid_size-1, ...)` grid clamp extracted from the legacy `_update_random` / `_update_pursuit` helpers
- [ ] 2.3 Refactor `Predator.update_position` (env.py:529) to build `PredatorBrainParams` from current state and call `self.brain.run_brain(params)` → `self._apply_action(action, ...)`. Pass `agent_positions` as an ordered tuple from `self.agents.values()` insertion order to preserve target tie-breaking semantics
- [ ] 2.4 Delete `Predator._update_random` (env.py:570-617) and `Predator._update_pursuit` (env.py:619-681) — `HeuristicPredatorBrain` is now the single source of truth
- [ ] 2.5 Byte-equivalence test (the gate): `test_legacy_path_byte_equivalent_to_new_path` parametrised across `PredatorType ∈ {STATIONARY, PURSUIT}` × `speed ∈ {0.5, 1.0, 2.0}`. Run two predators with identical seeds for 1000 steps — one constructed from a `git stash`-able legacy reference, one with `HeuristicPredatorBrain` — assert step-by-step position equality AND env RNG state advancement equality
- [ ] 2.6 Run `uv run pytest -m "not nightly" packages/quantum-nematode/tests/quantumnematode_tests/env/test_predator_brain.py -v` — must be green before committing

## 3. Config plumbing

- [ ] 3.1 Add `PredatorBrainConfig` dataclass to `env/predator_brain.py` (or `env/env.py` next to `PredatorParams`): `kind: Literal["heuristic"]` (M5 will extend); optional `extra: dict[str, Any] | None`
- [ ] 3.2 Extend `PredatorParams` (env.py:154) with `brain_config: PredatorBrainConfig | None = None`
- [ ] 3.3 Add `_build_predator_brain(predator_params)` dispatcher in env.py: returns `HeuristicPredatorBrain(predator_type, detection_radius)` for `brain_config is None` or `kind == "heuristic"`; raises `NotImplementedError` for other kinds (M5 extends)
- [ ] 3.4 Modify `_initialize_predators` (env.py:1481) to: synthesise `predator_id = f"predator_{i}"`, build brain via `_build_predator_brain`, pass both to the `Predator(...)` constructor
- [ ] 3.5 Extend `PredatorConfig` (config_loader.py:311) with optional `brain_config: PredatorBrainConfigSchema | None = None` (Pydantic; `kind: Literal["heuristic"]` only in M1)
- [ ] 3.6 Extend `PredatorConfig.to_params()` (config_loader.py:347) to translate `brain_config` to `PredatorBrainConfig` (no-op when `None`)
- [ ] 3.7 Tests: default `brain_config=None` → `HeuristicPredatorBrain` instance; explicit `kind: heuristic` produces same brain type; `predator_id` synthesis matches `f"predator_{i}"` for spawn loop index; ID stability across env resets

## 4. Per-predator metrics

- [ ] 4.1 Extend `MultiAgentEpisodeResult` (multi_agent.py:119) with three `dict[str, int]` fields using `field(default_factory=dict)`: `per_predator_kills`, `per_predator_prey_proximity_steps`, `per_predator_distance_traveled`
- [ ] 4.2 In `Predator._apply_action`, increment `self.distance_traveled` by 1 whenever the action moves the predator to a different position (not blocked by clamp)
- [ ] 4.3 In `MultiAgentSimulation.run_episode` step loop (multi_agent.py:612-648), after `self.env.update_predators()`: iterate `self.env.predators` × alive agents and increment each `pred.prey_proximity_steps` by 1 if any alive agent is within `pred.detection_radius` (Manhattan)
- [ ] 4.4 Implement kill attribution helper in `MultiAgentSimulation`: when `apply_predator_damage_for(aid)` brings HP to 0, find the closest predator (by Manhattan) whose `damage_radius` covers the agent; tie-break on lexicographic `predator_id`; increment a sim-local `_kills_by_predator: dict[str, int]`
- [ ] 4.5 In `MultiAgentSimulation._build_result` (multi_agent.py:1021), copy per-predator counters from `self.env.predators` (distance + proximity) and the sim-local kill dict into the `MultiAgentEpisodeResult` fields
- [ ] 4.6 Tests (in `tests/quantumnematode_tests/agent/test_multi_agent.py`): `test_per_predator_distance_traveled_records_movements` (synthetic 10-step, sum ≤ 10×N predators); `test_per_predator_prey_proximity_steps_increments_when_agent_in_range`; `test_per_predator_kills_attribution_distinct_distances` (predator_0 at dist 1, predator_1 at dist 2 → predator_0 gets the kill); `test_per_predator_kills_attribution_lex_tiebreak` (both at dist 1 → `predator_0` wins lex tie-break)

## 5. Regression baseline + logbook

- [ ] 5.1 Pre-refactor baseline campaign: 4 seeds (42, 43, 44, 45) × 200 episodes × 3 multi-agent pursuit scenarios (`mlpppo_small_5agents_pursuit_oracle.yml`, `mlpppo_small_5agents_pursuit_no_alarm_oracle.yml`, `lstmppo_small_5agents_pursuit_alarm_klinotaxis.yml`) on the M1 base commit (before delegation refactor lands). Run with `--theme headless`. Output: `tmp/m1_regression_baseline/baseline_pre.csv` with columns `config, seed, mean_alive_rate, std_alive_rate, n_episodes, session_id`
- [ ] 5.2 Post-refactor baseline campaign: identical protocol on the M1 head commit (after all M1.1-M1.4 tasks land). Output: `tmp/m1_regression_baseline/baseline_post.csv`
- [ ] 5.3 Compute deltas: per-cell `|post − pre| ≤ 0.02`; per-scenario mean delta ≤ 0.02. Strict-equivalence cross-check (50 episodes seed 42, single config) byte-identical predator trajectories — write as `test_predator_byte_equivalence_against_legacy_baseline` if needed
- [ ] 5.4 Write `docs/experiments/logbooks/016-predator-brain-refactor.md` documenting: framework changes, baseline pre/post numbers, regression-gate verdict, logbook 016 conventions match logbooks 012-015
- [ ] 5.5 Stash baseline CSVs under `artifacts/logbooks/016-predator-brain-refactor/baseline_pre.csv` and `baseline_post.csv`; copy `tmp/m1_regression_baseline/run_baseline.sh` for archival

## 6. Smoke + final checks

- [ ] 6.1 `uv run pytest -m smoke -v` clean
- [ ] 6.2 `uv run pytest -m "not nightly"` clean (full unit test suite, including the byte-equivalence and per-predator-metric tests)
- [ ] 6.3 `uv run pre-commit run -a` clean
- [ ] 6.4 `openspec validate add-learning-predators --strict` clean

## 7. Tracker + roadmap update

- [ ] 7.1 Tick M1.1-M1.8 sub-tasks in `openspec/changes/2026-04-26-phase5-tracking/tasks.md` (M1 section, lines 47-65)
- [ ] 7.2 Flip M1 row in `docs/roadmap.md` Phase 5 milestone tracker table from 🟡 next to ✅ complete
- [ ] 7.3 Add brief result summary to the M1 row referencing logbook 016 and the regression-gate outcome
