## 1. PredatorBrain Protocol module

- [x] 1.1 Create `packages/quantum-nematode/quantumnematode/env/predator_brain.py` with `PredatorAction` enum (`STAY`, `UP`, `DOWN`, `LEFT`, `RIGHT`)
- [x] 1.2 Add `PredatorBrainParams` frozen dataclass: `predator_id`, `predator_position`, `predator_type`, `detection_radius`, `damage_radius`, `agent_positions`, `chase_target`, `is_pursuing`, `grid_size`, `rng`, `step_index`. `chase_target` and `is_pursuing` are pre-resolved by `update_position` ONCE per call and passed unchanged across all accumulator-loop iterations within that call (frozen branch invariant — preserves byte-equivalence to legacy `update_position` which decides branch once per call rather than per accumulator-step)
- [x] 1.3 Add `PredatorBrain` Protocol decorated `@runtime_checkable` with `run_brain`, `prepare_episode`, `post_process_episode`, `copy`
- [x] 1.4 Add `HeuristicPredatorBrain` class encapsulating the existing `_update_pursuit` / `_update_random` logic byte-for-byte (single `rng.integers(4)` draw per call on the random branch; greedy axis selection on the pursuit branch with `if abs(dx) >= abs(dy)` precedence)
- [x] 1.5 Append `PredatorBrain`, `HeuristicPredatorBrain`, `PredatorBrainParams`, `PredatorAction` to the existing `__all__` list in `env/__init__.py` (existing exports `PredatorParams`, `PredatorType` at lines 20-21, 40-41 — extend in place rather than creating a new list). Also added `PredatorBrainConfig` for M1.3's config plumbing
- [x] 1.6 Tests: Protocol conformance via `@runtime_checkable` `isinstance`; `HeuristicPredatorBrain.copy()` returns independent instance; stationary returns STAY; in-range pursuit greedy axis selection; out-of-range pursuit falls back to random with seeded RNG → expected direction; random branch consumes exactly one RNG draw per call. **21 tests pass; full env test suite (307 tests) clean.**

## 2. Predator delegation + legacy removal

- [ ] 2.0 Add `DynamicForagingEnvironment._make_predator(self, predator_id: str, position: tuple[int, int], *, movement_accumulator: float = 0.0, brain: PredatorBrain | None = None) -> Predator` factory that reads defaults from `self.predator` and returns a fully-constructed `Predator`. When `brain` is `None`, the factory builds one from config (initially inlined heuristic-only build; task 3.3 later extracts `_build_predator_brain` and the factory delegates to it). When `brain` is provided (e.g. `copy_environment` passing `pred.brain.copy()`), the factory uses the provided brain unchanged. Centralises construction so the three current sites (env.py:1505, env.py:1526, env.py:3564) collapse to single-line factory calls. Per design Decision 8
- [ ] 2.1 Extend `Predator.__init__` (env.py:495) with `predator_id: str` and `brain: PredatorBrain | None = None`; init `kills`/`prey_proximity_steps`/`distance_traveled` counters to 0
- [ ] 2.2 Add `Predator._apply_action(action, grid_size, rng)` containing the accumulator advance, multi-step-per-update loop (capped at 10), and `max(0, ...)` / `min(grid_size-1, ...)` grid clamp extracted from the legacy `_update_random` / `_update_pursuit` helpers
- [ ] 2.3 Refactor `Predator.update_position` (env.py:529) to: (a) early-return on STATIONARY, (b) compute `chase_target` ONCE from `agent_positions` (or fall back to `agent_pos` for backward-compat single-agent path) using `min(... key=Manhattan)`, (c) compute `is_pursuing = (predator_type == PURSUIT and chase_target is not None and Manhattan(self.position, chase_target) <= detection_radius)` ONCE, (d) advance the accumulator loop in `_apply_action`, calling `self.brain.run_brain(params)` once per accumulator-step with `PredatorBrainParams` carrying the FROZEN `chase_target` + `is_pursuing` fields plus the CURRENT `predator_position`. The frozen-branch invariant is the byte-equivalence requirement: legacy `_update_pursuit` branches once per call (not per accumulator-step)
- [ ] 2.4 Delete `Predator._update_random` (env.py:570-617) and `Predator._update_pursuit` (env.py:619-681) — `HeuristicPredatorBrain` is now the single source of truth
- [ ] 2.5 Update `DynamicForagingEnvironment.copy_environment` (env.py:3563-3573) to preserve `predator_id` exactly (NOT re-synthesise) and copy `brain` via `pred.brain.copy()`. Single-pass factory call (the factory now accepts an optional `brain` arg per 2.0):

  ```python
  new_env.predators = [
      new_env._make_predator(
          predator_id=p.predator_id,
          position=p.position,
          movement_accumulator=p.movement_accumulator,
          brain=p.brain.copy(),
      )
      for p in self.predators
  ]
  ```

  Without this fix, env-copy boundaries (env-snapshot, evolution-loop replay) would create predators with re-synthesised IDs (per-predator metric mismatch) and brains constructed from config rather than copied from source (breaks any future learnable predator brain that has mutable internal state)
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
- [ ] 4.4 Implement sim-side kill attribution. Note that `env.apply_predator_damage_for(aid)` (env.py:2166) applies a fixed `predator_damage` per call without knowing which predator caused it — the env-side `is_agent_in_damage_radius_for(aid)` (env.py:2017) only returns boolean coverage. Therefore attribution must be done by the simulation: after each `apply_predator_damage_for(aid)` call where `self.env.agents[aid].hp <= 0`, the sim iterates `self.env.predators` to find covering predators (Manhattan ≤ `pred.damage_radius`), selects the closest by Manhattan distance (using the agent's pre-damage position), tie-breaks on lexicographic `predator_id`, and increments a sim-local `_kills_by_predator: dict[str, int]`. Defensive case: if no covering predator is found (e.g. damage tick on residual HP from prior step where the predator has since moved), credit the predator with smallest Manhattan distance to the agent's pre-damage position (no `damage_radius` constraint), with lex tie-break on `predator_id` as in the primary case. Emit a debug-level log warning so the case is visible in forensic logs
- [ ] 4.5 In `MultiAgentSimulation._build_result` (multi_agent.py:1021), copy per-predator counters from `self.env.predators` (distance + proximity) and the sim-local kill dict into the `MultiAgentEpisodeResult` fields
- [ ] 4.6 Tests (in `tests/quantumnematode_tests/agent/test_multi_agent.py`): `test_per_predator_distance_traveled_records_movements` (synthetic 10-step, sum ≤ 10×N predators); `test_per_predator_prey_proximity_steps_increments_when_agent_in_range`; `test_per_predator_kills_attribution_distinct_distances` (predator_0 at dist 1, predator_1 at dist 2 → predator_0 gets the kill); `test_per_predator_kills_attribution_lex_tiebreak` (both at dist 1 → `predator_0` wins lex tie-break)

## 5. Regression baseline + logbook

The byte-equivalence parametrised test (task 2.6) is the **primary** regression gate (catches RNG-ordering and trajectory-drift bugs at the predator level). The campaign runs below are the **secondary** sanity-check layer that exercise the predator code through both `MultiAgentSimulation` and the single-agent `Simulation` call paths under real training dynamics.

- [ ] 5.1 Pre-refactor baseline campaign: 4 seeds (42, 43, 44, 45) on the M1 base commit (before delegation refactor lands). Run with `--theme headless`. Estimated wall-time: ~25-30 min total at headless rates. Two arms covering both simulation-orchestration paths:
  - **Multi-agent arm**: 200 episodes × 3 multi-agent pursuit scenarios (`mlpppo_small_5agents_pursuit_oracle.yml`, `mlpppo_small_5agents_pursuit_no_alarm_oracle.yml`, `lstmppo_small_5agents_pursuit_alarm_klinotaxis.yml`). Higher episode count vs. single-agent arm because multi-agent variance is higher: 5 agents per episode produce 5 independent food-seeking trajectories, so per-episode metric stds are larger and need more episodes to converge per-cell
  - **Single-agent arm**: 100 episodes × 2 single-agent pursuit scenarios (`pursuit/mlpppo_small_oracle.yml`, `pursuit/lstmppo_small_klinotaxis.yml`) — closes the gap that the multi-agent arm leaves: single-agent `Simulation` invokes `env.update_predators()` from a different orchestration path. Note: `pursuit/lstmppo_small_klinotaxis.yml` includes STAM (short-term associative memory); if regression drift surfaces here but not on `pursuit/mlpppo_small_oracle.yml`, suspect STAM-adjacent code rather than the predator refactor itself

  The two arms write different CSVs with different schemas: multi-agent path emits `multi_agent_summary.csv` (cols include `agents_alive_at_end`, `total_food`, `proximity_events`, `mean_success`); single-agent path emits `simulation_results.csv` (cols include `success`, `foods_collected`, `steps`, `predator_encounters`, `died_to_health_depletion`, `successful_evasions`). Metrics extracted in 5.3 below are reconciled across both schemas

  Output: `tmp/m1_regression_baseline/baseline_pre.csv` with columns `arm, config, seed, mean_success, mean_total_food, mean_steps, mean_predator_engagement, n_episodes, session_id`. `mean_predator_engagement` aliases differently per arm — multi-agent reads `proximity_events / num_episodes` (already in summary CSV); single-agent reads `(predator_encounters + successful_evasions) / num_episodes` (composite signal of predator-trajectory effects). `mean_success` will saturate at 0.0 on the multi-agent pursuit arm (heuristic predators kill all 5 agents reliably under default config); single-agent arm is non-saturated and provides discriminating signal
- [ ] 5.2 Post-refactor baseline campaign: identical protocol on the M1 head commit (after all M1.1-M1.4 tasks land). Output: `tmp/m1_regression_baseline/baseline_post.csv`
- [ ] 5.3 Compute deltas across all four metrics, per-cell. **Tolerances are pre-registered estimates and may be tightened or loosened to ±2σ of pre-refactor std after the pre-refactor measurement completes; rationale documented in logbook 016.** Initial estimates:
  - `mean_success`: per-cell `|post − pre| ≤ 0.02` (vacuous-pass on saturated cells; flagged in logbook 016 narrative)
  - `mean_total_food`: per-cell `|post − pre| ≤ 0.5 food units` (discriminates predator-induced foraging-path changes; non-saturated on both arms because agents collect food before dying)
  - `mean_steps`: per-cell `|post − pre| ≤ 5 steps` (single-agent arm only — `multi_agent_summary.csv` does not have a per-run `steps` column; the script writes 0.0 for multi-agent cells, so multi-agent `mean_steps` deltas are noise-free by construction. Single-agent arm carries the timing-discrimination work)
  - `mean_predator_engagement`: per-cell tolerance arm-specific — multi-agent `|post − pre| ≤ 5 events` (proximity-events scale, from `proximity_events / num_episodes`); single-agent `|post − pre| ≤ 0.5 encounters/episode` (composite of `predator_encounters + successful_evasions`, both per-episode counts)

  Per-scenario means use the same per-cell tolerances. Any per-cell metric exceeding tolerance triggers a forensic step-by-step trajectory diff against `git stash`-able legacy reference for the failing (config, seed) cell. The byte-equivalence test (2.6) is expected to flag any drift before this stage; multi-metric campaign serves as cross-check.

  **The byte-equivalence test (2.6) is the PRIMARY regression gate**; the multi-metric campaign here is a secondary cross-check that exercises both `MultiAgentSimulation` and single-agent `Simulation` code paths under real training dynamics
- [ ] 5.4 Write `docs/experiments/logbooks/016-predator-brain-refactor.md` documenting: framework changes, baseline pre/post numbers across both arms, four-metric regression-gate verdict (including the saturated-metric finding for the multi-agent pursuit arm), logbook 016 conventions match logbooks 012-015
- [ ] 5.5 Stash baseline CSVs under `artifacts/logbooks/016-predator-brain-refactor/baseline_pre.csv` and `baseline_post.csv`; copy `tmp/m1_regression_baseline/run_baseline.sh` to `artifacts/logbooks/016-predator-brain-refactor/run_baseline.sh` for archival (the `tmp/` original is gitignored; only the archived copy at `artifacts/logbooks/016-predator-brain-refactor/` is committed). Forensic notes (saturation finding, raw per-cell numbers, intermediate decisions) live in `tmp/evaluations/evolution/evolution_scratchpad.md` for cross-session reference (gitignored)

## 6. Smoke + final checks

- [ ] 6.1 `uv run pytest -m smoke -v` clean
- [ ] 6.2 `uv run pytest -m "not nightly"` clean (full unit test suite, including the byte-equivalence and per-predator-metric tests)
- [ ] 6.3 `uv run pre-commit run -a` clean
- [ ] 6.4 `openspec validate add-learning-predators --strict` clean

## 7. Tracker + roadmap update

- [ ] 7.1 Tick M1.1-M1.8 sub-tasks in `openspec/changes/2026-04-26-phase5-tracking/tasks.md` (M1 section, lines 47-65)
- [ ] 7.2 Flip M1 row in `docs/roadmap.md` Phase 5 milestone tracker table from 🟡 next to ✅ complete
- [ ] 7.3 Add brief result summary to the M1 row referencing logbook 016 and the regression-gate outcome
