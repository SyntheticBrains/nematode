## Context

Phase 5 M1 introduces the predator-brain abstraction seam ahead of M5 (co-evolution). Today the `Predator` class at [packages/quantum-nematode/quantumnematode/env/env.py:471-682](../../../packages/quantum-nematode/quantumnematode/env/env.py) carries movement state (`position`, `speed`, `movement_accumulator`, `detection_radius`, `damage_radius`) and a `PredatorType` enum that the `update_position` method branches on (`STATIONARY` early-returns; `PURSUIT` in-range calls `_update_pursuit`; out-of-range falls through to `_update_random`). Both private helpers couple movement decisions to grid mechanics: each one runs the accumulator-advance loop and the `max(0, ...)` / `min(grid_size-1, ...)` clamp in-line.

The codebase's agent-side `Brain` Protocol at [packages/quantum-nematode/quantumnematode/brain/arch/\_brain.py:345-395](../../../packages/quantum-nematode/quantumnematode/brain/arch/_brain.py) uses `typing.Protocol` + `@runtime_checkable` (not ABC). Per-agent metrics in `MultiAgentEpisodeResult` ([agent/multi_agent.py:119-181](../../../packages/quantum-nematode/quantumnematode/agent/multi_agent.py)) follow a dict-keyed pattern (`per_agent_food`, `per_agent_reward`, `per_agent_satiety`); per-predator metrics added by this change mirror that idiom exactly.

The change ships under feature branch `feat/m1-predator-brain-refactor`. The regression gate is ±2pp agent-survival rate across 4 seeds × 200 episodes × 3 multi-agent pursuit scenarios, captured pre-refactor on the M1 base commit and post-refactor on M1 head; results documented in logbook 016.

## Goals / Non-Goals

**Goals:**

- Predators expose a pluggable policy seam via `PredatorBrain` Protocol so M5 can introduce learnable predator brains without further refactoring of `Predator` or the env step loop.
- `HeuristicPredatorBrain` becomes the single source of truth for the existing heuristic behaviour; the legacy `_update_pursuit` / `_update_random` helpers are deleted in the same commit that introduces the delegation.
- Brain-induced behaviour is byte-equivalent to legacy behaviour for every existing scenario (same RNG draws, same target tie-breaking, same accumulator timing).
- Per-predator metrics (`kills`, `prey_proximity_steps`, `distance_traveled`) are exposed in `MultiAgentEpisodeResult` for future predator-fitness functions.
- `Predator` instances gain a synthesised `predator_id` so per-predator metrics can be keyed deterministically.

**Non-Goals:**

- No learnable predator brain (e.g. `MLPPPOPredatorBrain`) ships in this change. The Protocol must accommodate them; M5 adds the first one.
- No `PredatorBrainEncoder` for the evolution-framework capability; no `evolution-framework/spec.md` delta.
- No predator-specific reward / fitness function — predators have no reward signal in M1.
- No multi-predator coordination (predator-side pheromones, shared observations, prey-tagging).
- No heuristic improvements (smarter pursuit, A\* pathfinding, prey-trajectory memory) — ship the *current* heuristic, byte-equivalent.
- No per-predator visualisation in the renderer.
- No per-predator metrics in the single-agent `EpisodeResult` ([runners.py:70-87](../../../packages/quantum-nematode/quantumnematode/agent/runners.py)). The M1 sub-task wording mentions "EpisodeResult" loosely; the existing per-agent metric pattern lives entirely in `MultiAgentEpisodeResult` and that's the right surface here.

## Decisions

### Decision 1: Movement-mechanics decoupling — brain returns intent, harness owns kinematics

The brain's `run_brain` returns one of `PredatorAction.{STAY, UP, DOWN, LEFT, RIGHT}`. The harness layer is split into two helpers:

- `Predator._apply_action_loop(...)` owns: (a) advancing `movement_accumulator` by `self.speed`, (b) the multi-step-per-update loop (capped at 10 steps, matching the legacy code), (c) building `PredatorBrainParams` per accumulator-step with FROZEN `chase_target` + `is_pursuing` fields and the CURRENT `predator_position`, (d) calling `self.brain.run_brain(params)` once per step.
- `Predator._apply_action(action, grid_size)` owns: per-step kinematics — translates the brain's `PredatorAction` into a position delta, applies the `max(0, ...)` / `min(grid_size-1, ...)` grid clamp, and increments `self.distance_traveled` only when post-clamp position differs from pre-action position.

**Why:** Brain logic stays testable without env state. Future learnable brains don't need to re-derive the accumulator dance. The action enum is small, exhaustive for cardinal-grid movement, and trivially serialisable for evolution-framework genome encoders later. The split between the loop and the per-step kinematics keeps each helper's responsibility focused (loop owns control flow + params construction; per-step owns position + distance counter).

**Alternative considered:** brain returns `(dx, dy)` deltas. Rejected — couples the brain to grid mechanics (must know `grid_size` to avoid out-of-bounds) and re-introduces the validation logic the harness already does.

### Decision 2: Protocol over ABC

`PredatorBrain` is a `typing.Protocol` decorated `@runtime_checkable`, mirroring `Brain` at [\_brain.py:345-395](../../../packages/quantum-nematode/quantumnematode/brain/arch/_brain.py). Required methods: `run_brain(params) -> PredatorAction`, `prepare_episode() -> None`, `post_process_episode(*, episode_success=None) -> None`, `copy() -> PredatorBrain`.

**Why:** Matches codebase convention. `prepare_episode` / `post_process_episode` are no-ops for `HeuristicPredatorBrain` but are part of the Protocol so M5's RL brains have guaranteed lifecycle hooks.

**Alternative considered:** Inheriting from agent `Brain` Protocol directly. Rejected — agent `Brain` carries `history_data` / `latest_data` attrs and an `update_memory` method that don't fit the predator surface.

### Decision 3: `PredatorBrainParams` is minimal

Frozen dataclass with 11 fields: `predator_id`, `predator_position`, `predator_type`, `detection_radius`, `damage_radius`, `agent_positions: tuple[tuple[int, int], ...]`, `chase_target: tuple[int, int] | None`, `is_pursuing: bool`, `grid_size`, `rng`, `step_index`. NOT a Pydantic clone of agent `BrainParams`'s ~90-field surface.

The two non-obvious fields — `chase_target` and `is_pursuing` — encode the **frozen-branch invariant** (Decision 7 below): pre-resolved by `Predator.update_position` ONCE per call and passed unchanged across every accumulator-loop iteration in that call. Only `predator_position` varies per accumulator-step within a single `update_position` call.

**Why:** Predators don't sense food gradients, oxygen, temperature, pheromones, or any of the agent-specific channels. Future M5 learnable predator brains can derive whatever sensory features they want from these primitives without committing M1 to a heavy schema. The frozen-branch fields are necessary for byte-equivalence (see Decision 7).

### Decision 4: `HeuristicPredatorBrain` is the single source of truth from day one

The M1.2 commit deletes `Predator._update_pursuit` and `Predator._update_random`. Their logic moves entirely into `HeuristicPredatorBrain.run_brain`. Existing scenarios continue to work because `_initialize_predators` defaults `brain_config=None` → `HeuristicPredatorBrain` for every spawned predator.

**Why:** Cleanest end state. Two implementations of the same logic would invite drift and reviewer confusion.

**Safety net:** The byte-equivalence parametrised test (`test_legacy_path_byte_equivalent_to_new_path`) is written *before* M1.2 lands. It runs two predators with identical seeds — one constructed from a `git stash`-able legacy reference, one with `HeuristicPredatorBrain` — for 1000 steps across `{STATIONARY, PURSUIT}` × `{0.5, 1.0, 2.0}` and asserts step-by-step position equality. If it passes, the heuristic moved cleanly.

**Alternative considered:** keep a `_update_legacy` fallback for `brain is None` through M1, drop in early-M5 cleanup commit. Rejected per user's explicit preference for the cleaner end state.

### Decision 5: Predator ID synthesis at spawn

`_initialize_predators` synthesises `predator_id = f"predator_{i}"` keyed on the spawn loop index. Stored as `Predator.predator_id` instance attribute.

**Why:** Stable across episodes (same env config + same seed → same IDs); deterministic; lexicographically ordered (so the kill-attribution tie-break is unambiguous). Mirrors `AgentState.agent_id` which is constructed similarly.

### Decision 6: Kill attribution rule — closest-by-Manhattan, lex tie-break

When `apply_predator_damage_for(aid)` brings an agent's HP to 0, the simulation credits the kill to the predator with the smallest Manhattan distance to the agent whose `damage_radius` covers the agent. Ties broken by lexicographic `predator_id` (so `predator_0` beats `predator_1`).

**Why:** Deterministic; intuitive (closest predator "got" the prey); cheap to compute; preserves integer-counter idiom of existing `per_agent_food` etc. Documented prominently in this design doc and in the modified `multi-agent` spec scenario.

**Alternatives considered:** (a) split kill 1/N across all covering predators — rejected, breaks integer-counter idiom and complicates downstream fitness functions. (b) first-in-list — rejected, attribution becomes spawn-order-dependent rather than principled.

### Decision 7: RNG ordering preservation strategy

The legacy random branch calls `rng.integers(4)` once per accumulator-step. The new path's accumulator loop calls `brain.run_brain(params)` once per accumulator-step; on the random branch (out-of-detection-range pursuit) `HeuristicPredatorBrain.run_brain` makes the same single `rng.integers(4)` draw and maps it to `UP/DOWN/LEFT/RIGHT` exactly as legacy code did. The pursuit branch is RNG-free in both legacy and new code.

**Why:** Byte-equivalence requires not just the same final position but the same RNG state advancement, since the env's RNG is shared with food spawning, agent decisions, and other downstream consumers. A subtle off-by-one in RNG draws would silently desynchronise downstream consumers and break the regression baseline. Verified by the `test_random_branch_consumes_one_rng_draw_per_call` unit test before M1.2 lands.

## Risks / Trade-offs

**[R1] Heuristic non-equivalence (RNG ordering)** → Biggest risk; would tank the regression gate. **Mitigation**: brain calls `rng.integers(4)` inside `run_brain` (not in the harness); accumulator loop in `Predator._apply_action_loop` calls `brain.run_brain` once per accumulated step (the loop is split from `_apply_action` per Decision 1). The `test_predator_brain_byte_equivalence.py` parametrised tests are the gate — must pass before delegation lands. If the regression baseline shows mismatches, re-derive byte-equivalence from `_legacy_predator_reference._LegacyPredatorReference` (the frozen pre-M1 reference fixture committed alongside the byte-equivalence tests).

**[R2] Multi-agent target tie-breaking** → Python `min(iterable, key=...)` returns the first equal-key element. Legacy `min(agent_positions, key=...)` ([env.py:558](../../../packages/quantum-nematode/quantumnematode/env/env.py)) iterates `self.agents.values()` (insertion-stable since Python 3.7). **Mitigation**: pass `agent_positions` as an ordered tuple through `PredatorBrainParams` — do not sort or rebuild. Add a unit test for stable ordering.

**[R3] Accumulator timing across pursuit/random branches** → Legacy code's pursuit branch out-of-range falls through to `_update_random`, which advances the accumulator. **Mitigation**: brain returns the random-direction action, `_apply_action` advances accumulator once per call. Verified by the byte-equivalence test parametrised on `speed ∈ {0.5, 1.0, 2.0}` (the `0.5` case forces accumulator-fractional behaviour; `2.0` forces multi-step).

**[R4] Spec creep into M5** → Temptation to add `MLPPPOPredatorBrain` skeletons "while we're here". **Mitigation**: explicit non-goal in this design doc; reviewer checklist line in the PR description; the `PredatorBrain` Protocol surface accommodates learnable brains without M1 needing to ship one.

**[R5] Kill attribution challenges in logbook review** → Reviewers may ask why `predator_0` got a contested kill. **Mitigation**: closest-then-lex rule documented in this design doc + in the modified `multi-agent` spec scenario; a unit test covers the contested-damage case explicitly.

### Decision 8: Extract `_make_predator` factory to centralise Predator construction

There are three `Predator(...)` construction sites in env.py: the safe-spawn branch of `_initialize_predators`, its fallback branch (when Poisson sampling fails to find a valid spawn), and `DynamicForagingEnvironment.copy()` (the env-copy path). Adding `predator_id` + `brain` to the constructor signature means all three sites must thread the new fields consistently. To prevent drift, M1 extracts a private `_make_predator` factory.

**Final signature** (extended mid-implementation; see "Per-predator field overrides" below):

```python
def _make_predator(
    self,
    predator_id: str,
    position: tuple[int, int],
    *,
    movement_accumulator: float = 0.0,
    brain: PredatorBrain | None = None,
    predator_type: PredatorType | None = None,
    speed: float | None = None,
    detection_radius: int | None = None,
    damage_radius: int | None = None,
) -> Predator
```

Behaviour:

- When `brain is None`, the factory builds a fresh brain via `self._build_predator_brain()`. When `brain` is provided (e.g. `pred.brain.copy()` from `env.copy()`), the factory uses it unchanged — preserving stateful brains across env-copy boundaries (M5 forward-compat).
- Each per-predator field override (`predator_type`/`speed`/`detection_radius`/`damage_radius`) defaults to `None` and falls back to the corresponding `self.predator.*` value from the env-level `PredatorParams`. `_initialize_predators` uses the `None` defaults; `env.copy()` passes the source predator's actual field values explicitly.

**Why the per-predator overrides matter:** the existing legacy test `test_damage_radius_copied_in_env_copy` mutates `env.predators[0].damage_radius = 7` post-init, then asserts `env.copy()` preserves the mutated value. Without per-field threading through the factory, the copy would read `damage_radius` from `self.predator` (the env-level config, unchanged at 5), losing the mutation. The override params solve this without breaking the centralisation invariant. (Surfaced as a real regression mid-implementation; documented in the M1.3 commit message.)

**Why not a public factory:** `_make_predator` is private to `DynamicForagingEnvironment` because it depends on `self.predator`. Making it public would force callers to pass `PredatorParams` redundantly. M5 will revisit this if external construction surfaces become necessary.

**Note on `env.copy()`:** the copy path preserves the *source* env's `predator_id`s exactly (not re-synthesising) and the source brain via `pred.brain.copy()`. Method name is `copy()`, not `copy_environment()` (an earlier draft of the tasks doc had the wrong name; corrected during implementation).
