## 1. PredatorBrain Protocol module

- [ ] 1.1 Create `packages/quantum-nematode/quantumnematode/env/predator_brain.py` with `PredatorAction` enum (`STAY`, `UP`, `DOWN`, `LEFT`, `RIGHT`)
- [ ] 1.2 Add `PredatorBrainParams` frozen dataclass: `predator_id`, `predator_position`, `predator_type`, `detection_radius`, `damage_radius`, `agent_positions`, `grid_size`, `rng`, `step_index`
- [ ] 1.3 Add `PredatorBrain` Protocol decorated `@runtime_checkable` with `run_brain`, `prepare_episode`, `post_process_episode`, `copy`
- [ ] 1.4 Add `HeuristicPredatorBrain` class encapsulating the existing `_update_pursuit` / `_update_random` logic byte-for-byte (single `rng.integers(4)` draw per call on the random branch; greedy axis selection on the pursuit branch with `if abs(dx) >= abs(dy)` precedence)
- [ ] 1.5 Append `PredatorBrain`, `HeuristicPredatorBrain`, `PredatorBrainParams`, `PredatorAction` to the existing `__all__` list in `env/__init__.py` (existing exports `PredatorParams`, `PredatorType` at lines 20-21, 40-41 — extend in place rather than creating a new list)
- [ ] 1.6 Tests: Protocol conformance via `@runtime_checkable` `isinstance`; `HeuristicPredatorBrain.copy()` returns independent instance; stationary returns STAY; in-range pursuit greedy axis selection; out-of-range pursuit falls back to random with seeded RNG → expected direction; random branch consumes exactly one RNG draw per call

## 2. Predator delegation + legacy removal

- [ ] 2.0 Add `DynamicForagingEnvironment._make_predator(self, predator_id: str, position: tuple[int, int], *, movement_accumulator: float = 0.0) -> Predator` factory that reads defaults from `self.predator`, builds the brain via `_build_predator_brain` (added in 3.3), and returns a fully-constructed `Predator`. Centralises construction so the three current sites (env.py:1505, env.py:1526, env.py:3564) collapse to single-line factory calls. Per design Decision 8
- [ ] 2.1 Extend `Predator.__init__` (env.py:495) with `predator_id: str` and `brain: PredatorBrain | None = None`; init `kills`/`prey_proximity_steps`/`distance_traveled` counters to 0
- [ ] 2.2 Add `Predator._apply_action(action, grid_size, rng)` containing the accumulator advance, multi-step-per-update loop (capped at 10), and `max(0, ...)` / `min(grid_size-1, ...)` grid clamp extracted from the legacy `_update_random` / `_update_pursuit` helpers
- [ ] 2.3 Refactor `Predator.update_position` (env.py:529) to build `PredatorBrainParams` from `self` state plus the `agent_positions` argument (already passed in by `update_predators` — the env owns the `self.agents.values()` ordering at env.py:1974, NOT the predator). Pass the existing `agent_positions: list[tuple[int, int]] | None` through unchanged into `PredatorBrainParams.agent_positions` as an ordered tuple. Then call `self.brain.run_brain(params)` → `self._apply_action(action, ...)`
- [ ] 2.4 Delete `Predator._update_random` (env.py:570-617) and `Predator._update_pursuit` (env.py:619-681) — `HeuristicPredatorBrain` is now the single source of truth
- [ ] 2.5 Update `DynamicForagingEnvironment.copy_environment` (env.py:3563-3573) to preserve `predator_id` exactly (NOT re-synthesise) and copy `brain` via `pred.brain.copy()`. Implementation: collapse the existing list comprehension to call the factory: `new_env.predators = [new_env._make_predator(predator_id=p.predator_id, position=p.position, movement_accumulator=p.movement_accumulator) for p in self.predators]`, then for each new predator overwrite its brain with `p.brain.copy()` (the factory builds a fresh brain from config; the copy semantics require the *source* brain's state). Without this fix, env-copy boundaries (env-snapshot, evolution-loop replay) would create predators with re-synthesised IDs (per-predator metric mismatch) and brains constructed from config rather than copied from source (breaks any future learnable predator brain that has mutable internal state)
- [ ] 2.6 Byte-equivalence test (the gate): `test_legacy_path_byte_equivalent_to_new_path` parametrised across `PredatorType ∈ {STATIONARY, PURSUIT}` × `speed ∈ {0.5, 1.0, 2.0}`. Run two predators with identical seeds for 1000 steps — one constructed from a `git stash`-able legacy reference, one with `HeuristicPredatorBrain` — assert step-by-step position equality AND env RNG state advancement equality (compare `env.rng.bit_generator.state` snapshots before/after each step pair)
- [ ] 2.7 Add direct ordering-invariant test: `test_update_predators_passes_agents_in_insertion_order` — verifies `update_predators()` builds `agent_positions` matching `tuple((a.position[0], a.position[1]) for a in self.agents.values() if a.alive)` so target tie-breaking is deterministic across env reconstructions
- [ ] 2.8 Run `uv run pytest -m "not nightly" packages/quantum-nematode/tests/quantumnematode_tests/env/test_predator_brain.py -v` — must be green before committing

## 3. Config plumbing

- [ ] 3.1 Add `PredatorBrainConfig` dataclass to `env/predator_brain.py` (preferred — keeps brain-config alongside the Protocol; avoids further bloating env.py which is already 3500+ lines): `kind: Literal["heuristic"]` (M5 will extend); optional `extra: dict[str, Any] | None`. This is the runtime dataclass; the YAML-loading Pydantic schema is added separately at 3.5 (intentional two-type pattern matching the existing `PredatorParams` dataclass + `PredatorConfig` Pydantic model split at config_loader.py)
- [ ] 3.2 Extend `PredatorParams` (env.py:154) with `brain_config: PredatorBrainConfig | None = None`
- [ ] 3.3 Add `_build_predator_brain(predator_params)` dispatcher in env.py: returns `HeuristicPredatorBrain(predator_type, detection_radius)` for `brain_config is None` or `kind == "heuristic"`; raises `NotImplementedError` for other kinds (M5 extends)
- [ ] 3.4 Modify `_initialize_predators` (env.py:1481) to call `self._make_predator(predator_id=f"predator_{i}", position=candidate)` for both the safe-spawn branch (line 1505) and the fallback branch (line 1526). Loop index `i` (0-based) determines the synthesised ID. The factory (2.0) handles brain construction internally via `_build_predator_brain`
- [ ] 3.5 Extend `PredatorConfig` (config_loader.py:311) with optional `brain_config: PredatorBrainConfigSchema | None = None` (Pydantic BaseModel for YAML validation; `kind: Literal["heuristic"]` only in M1). Two-type pattern is intentional: dataclass for runtime use (3.1), Pydantic model for YAML loading + validation, mirroring the existing `PredatorParams` ↔ `PredatorConfig` split
- [ ] 3.6 Extend `PredatorConfig.to_params()` (config_loader.py:347) to translate `brain_config` to `PredatorBrainConfig` (no-op when `None`)
- [ ] 3.7 Tests: default `brain_config=None` → `HeuristicPredatorBrain` instance; explicit `kind: heuristic` produces same brain type; `predator_id` synthesis matches `f"predator_{i}"` for spawn loop index; ID stability across env `reset()` calls within the same env instance; ID reproducibility across two env instances with the same config + seed
- [ ] 3.8 (Optional but recommended) Add an example YAML config exercising the explicit `brain_config` block: `configs/scenarios/multi_agent_pursuit/mlpppo_small_5agents_pursuit_oracle_explicit_brain.yml` — identical to the default-brain pursuit config but with an explicit `predators.brain_config: {kind: heuristic}` block. Helps reviewers see the contract and gives M5 a clear extension point to grep for. Not required for the regression gate (the implicit-default scenarios are sufficient)

## 4. Per-predator metrics

- [ ] 4.1 Extend `MultiAgentEpisodeResult` (multi_agent.py:119) with three `dict[str, int]` fields using `field(default_factory=dict)`: `per_predator_kills`, `per_predator_prey_proximity_steps`, `per_predator_distance_traveled`
- [ ] 4.2 In `Predator._apply_action`, increment `self.distance_traveled` by 1 only when the post-clamp position differs from the pre-action position. STAY actions and wall-blocked moves (where `max(0, ...)` / `min(grid_size-1, ...)` clamped the new position back to the current one) SHALL contribute 0 to the counter
- [ ] 4.3 In `MultiAgentSimulation.run_episode` step loop (multi_agent.py:612-648), after `self.env.update_predators()`: iterate `self.env.predators` × alive agents and increment each `pred.prey_proximity_steps` by 1 if any alive agent is within `pred.detection_radius` (Manhattan)
- [ ] 4.4 Implement sim-side kill attribution. Note that `env.apply_predator_damage_for(aid)` (env.py:2166) applies a fixed `predator_damage` per call without knowing which predator caused it — the env-side `is_agent_in_damage_radius_for(aid)` (env.py:2017) only returns boolean coverage. Therefore attribution must be done by the simulation: after each `apply_predator_damage_for(aid)` call where `self.env.agents[aid].hp <= 0`, the sim iterates `self.env.predators` to find covering predators (Manhattan ≤ `pred.damage_radius`), selects the closest by Manhattan distance (using the agent's pre-damage position), tie-breaks on lexicographic `predator_id`, and increments a sim-local `_kills_by_predator: dict[str, int]`. Defensive case: if no covering predator is found (e.g. damage tick on residual HP from prior step where the predator has since moved), credit the global-closest predator and emit a debug-level log warning
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
