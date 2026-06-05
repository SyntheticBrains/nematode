## 1. Shared action-policy module + action contract (discrete parity first)

- [ ] 1.1 Create `brain/arch/_policy.py`: a shared helper/mixin exposing build-head, sample, `log_prob`, `entropy`, and PPO surrogate terms, with a discrete (Categorical) mode that reproduces the current per-brain logic. Follow the leaf-module precedent (`_ppo_buffer.py`, `_brain.py`): underscore-prefixed, imported directly by brain modules, NOT re-exported through `brain/arch/__init__.py` (which imports the brain modules and would invert the dependency / risk a cycle).
- [ ] 1.2 Extend the action contract: relax `ActionData.action` from required `Action` to `Action | None` (the runner already null-guards it at `agent/agent.py:855`, `runners.py:983/985`) and add an optional `continuous: tuple[float, float] | None` field; define how the carrier conveys `(speed, turn)`. Grid substrate keeps emitting `action: Action`.
- [ ] 1.3 Route the agent/runner action-application to dispatch on the environment's action mode (grid → discrete `Action`; continuous-2D → continuous vector), with NO branch on the concrete brain class. Cover BOTH action-consumption surfaces: the single-action path (`runners.py:720`) AND the many-worlds runner-up path (`runners.py:983-1010` + `.action.value` logging at `:802`) — route both through the mode dispatch or guard many-worlds as discrete-only on the continuous substrate. Generalise the `move_agent(action: Action)` signature to carry the continuous vector. Explicitly guard the `top_action.action.value` log line (`runners.py:802`) so it does not `AttributeError` when `action` is `None` in continuous mode.
- [ ] 1.4 Unit tests for the shared module's discrete mode (sample/log_prob/entropy) against the pre-refactor reference outputs.

## 2. Migrate MUST brains to the shared module + regression bar (GATE before continuous work)

- [ ] 2.1 Migrate MLP-PPO discrete path onto `_policy.py`.
- [ ] 2.2 Migrate LSTM-PPO discrete path onto `_policy.py` (handle the TEI-bias-shape coupling).
- [ ] 2.3 Migrate CfC-PPO discrete path onto `_policy.py` (handle the AutoNCP motor-count coupling).
- [ ] 2.4 Migrate connectome-PPO discrete path onto `_policy.py`.
- [ ] 2.5 Migration-regression bar: byte-equivalence for MLP-PPO + LSTM-PPO on one fixed-seed discrete smoke config each (pre/post); CfC + connectome within a declared, documented seeded-RNG tolerance. Record results; do not proceed to §4 until green.

## 3. Continuous-2D environment + config discriminator

- [ ] 3.1 Add `EnvironmentConfig.env_type` discriminator (`"grid"` default | `"continuous_2d"`) + continuous-env config fields (world bounds, body-length scale, max step displacement, capture radius, sweep amplitude) with documented defaults.
- [ ] 3.2 Add env-type dispatch in `create_env_from_config` ([`config_loader.py:2702`](../../../packages/quantum-nematode/quantumnematode/utils/config_loader.py#L2702)); generalise the `EnvironmentType` alias to the base/union; route the direct `DynamicForagingEnvironment(...)` construction sites that bypass the factory through it — `agent/agent.py` (×2: `:402`, `:1328`), `env/env.py` `copy()` (`:3995`), `scripts/export_screenshot.py` (×2: `:41`, `:173`). (`scripts/run_simulation.py` already uses the factory — no change there; the evolution `loop/fitness/predator_fitness` paths inherit the discriminator via the factory automatically — confirm.)
- [ ] 3.3 Implement `Continuous2DEnvironment(BaseEnvironment)`: float positions, kinematic `(speed, turn)` movement with world-bound clamping. **Override** `BaseEnvironment._apply_movement` / `move_agent` (and `DynamicForagingEnvironment`'s `move_agent` override at `:2457`) — these are grid-coupled (`DIRECTION_MAP`, discrete `Action` signature) and cannot be inherited as-is.
- [ ] 3.4 Capture-radius food consumption (replace cell-equality); continuous source placement (replace integer sampling); Euclidean distances for pheromone / predator-mechano / nearest-food.
- [ ] 3.5 Physically-scaled klinotaxis lateral sampling on continuous coordinates; drop the integer-cell offset + `int(...)` position cast; feed STAM continuous readings.
- [ ] 3.6 Unit tests: kinematics + world-bound clamping; capture-radius consumption (in/out of radius); continuous placement validity; Euclidean fields; klinotaxis continuous dC + STAM dC/dt.
- [ ] 3.7 Verify the grid environment is unchanged (T4 grid regression on an existing scenario).
- [ ] 3.8 Factory-dispatch tests (`environment-simulation` spec): `env_type: continuous_2d` → `Continuous2DEnvironment`; omitted/`grid` → `DynamicForagingEnvironment`; callers route through `create_env_from_config`, not a concrete-class alias.
- [ ] 3.9 Config-loader tests (`configuration-system` spec, via `tests/.../utils/test_config_loader_yaml_compat.py`): continuous-2D env fields + continuous-action fields parse/validate; a legacy grid YAML with no `env_type` loads byte-identically (discrete default preserved).

## 4. Continuous-action heads on MUST brains + connectome adapter

- [ ] 4.1 Implement the tanh-squashed Gaussian continuous mode in `_policy.py` (2-D action; tanh + affine rescale to speed/turn bounds; log-det-Jacobian-corrected log-prob; clamped log-std; finite-output guarantees).
- [ ] 4.2 Add continuous heads to MLP-PPO, LSTM-PPO, CfC-PPO with the action mode + continuous dim (2) carried in `BrainConfig` (NOT a new `_build_infra_kwargs` per-arch branch); the existing `num_actions=4` infra-kwarg is interpreted as the discrete action-set size and ignored in continuous mode.
- [ ] 4.3 Connectome continuous-output adapter: `_pool_motor` → `(mean, log_std)` head via `_policy.py`; preserve strict-mask + fixed gap junctions (forward + post-step projection unchanged).
- [ ] 4.4 Unit tests: tanh-Gaussian log-prob/entropy/Jacobian correctness; bounded sampled actions; connectome strict-mask invariant across output modes.
- [ ] 4.5 Continuous-substrate smoke training: connectome + MLP-PPO on continuous-2D klinotaxis train without NaN/Inf with a learning signal (last-quarter > first-quarter return).

## 5. Transformer architecture + plugin-parity verification (Gate 2 G2.b/G2.c)

- [ ] 5.1 Implement the transformer/attention PPO brain (`brain/arch/transformer*.py`) registered via `@register_brain`; conform to the `Brain` Protocol; discrete head sufficient for the parity test (continuous head opportunistic).
- [ ] 5.2 Wire it through the ≤6-file budget: enum member, package import/export, config union, `_build_infra_kwargs` branch only if non-default, test file.
- [ ] 5.3 Refactor the `scripts/run_simulation.py` CLI-default `match brain_type` block ([`:286-298`](../../../scripts/run_simulation.py#L286)) into a registry-driven default so no per-arch branch remains at the entrypoint. **Ship this as a SEPARATE prior commit** (one-time platform fix) so it is NOT counted in the transformer-addition diff measured by G2.b.
- [ ] 5.4 Plugin-parity test: assert files-touched ≤ 6 on the transformer-addition commit(s) only (via `git diff --name-only`), measured per the [plugin-developer-guide](../../../docs/architecture/plugin-developer-guide.md) enumeration — new module, `dtypes.py` enum, `brain/arch/__init__.py`, `config_loader.py` union, `brain_factory.py` (only if non-default), test (YAML example excluded); plus a code-review verdict of no per-architecture branches in the simulation/training loops. Record engineer-hours (G2.a, documented, not load-bearing). Land the transformer runtime wiring in its OWN commit, isolated from the §5.6 doc-enumeration updates and any YAML config example, so the `git diff --name-only` measurement is uncontaminated.
- [ ] 5.5 Transformer trains on a klinotaxis smoke without collapse (learning signal).
- [ ] 5.6 Update the brain-arch enumerations that live OUTSIDE the ≤6 parity budget (and are not counted in G2.b): `AGENTS.md:33`, `README.md:36`, `CONTRIBUTING.md:140`, `openspec/config.yaml:18` — bump 24→25 and add the transformer entry. These are required for merge but are explicitly not per-architecture parity cost.

## 6. Gate 2 floor check + analysis

- [ ] 6.1 G2.d floor check: connectome + MLP-PPO on continuous-2D klinotaxis (single seed, T4 episode budget) reach mean episode return ≥ 50% of their T4 grid-substrate baseline for the same architecture. Record the numbers.
- [ ] 6.2 Platform-refactor delta analysis: continuous-2D vs discrete-grid movement kinematics; continuous-action vs discrete-action training stability (T5.analysis).

## 7. Logbook, Gate 2 decision, tracker + roadmap updates

- [ ] 7.1 Author continuous-2D scenario YAML(s) under `configs/scenarios/`; existing grid scenarios untouched.
- [ ] 7.1b Register a continuous-2D scenario in the smoke list (`SIMULATION_CONFIGS`, `tests/quantumnematode_tests/test_smoke.py:25`) so CI has a standing guard that the new substrate keeps parsing + running.
- [ ] 7.2 Publish the T5 logbook (`docs/experiments/logbooks/0XX-platform-refactor.md`) with the parity verification + floor-check evidence.
- [ ] 7.3 Record the Gate 2 GO/PIVOT/STOP decision in the T5 logbook against G2.a–G2.d; PIVOT trigger if G2.b OR G2.c fails; STOP trigger if post-pivot the interface is still incompatible with a MUST family.
- [ ] 7.4 Update `openspec/changes/phase6-tracking/tasks.md` T5 rows + the Gate 2 decision link; flip the `docs/roadmap.md` Phase 6 Tranche Tracker T5 row + Gate 2 outcome.
- [ ] 7.5 `openspec validate add-continuous-2d-and-action-heads --strict` clean; targeted `pre-commit run --files <changed>` clean; full `pre-commit run -a` before push.

## 8. SHOULD/MAY continuous heads (opportunistic, NOT gating Gate 2)

- [ ] 8.1 Continuous-output adaptation for SHOULD/MAY families (quantum, spiking, reservoir, hybrid) as opportunity allows; document any that don't adapt cleanly and defer.
