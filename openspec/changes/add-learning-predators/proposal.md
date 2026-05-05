## Why

Predators in this codebase are currently a single `Predator` class with hardcoded movement (stationary or pursuit, by enum + branch in `update_position`). To run M5 (co-evolution arms race) — where predators evolve hunting strategies under Red Queen dynamics with co-evolving prey — predators need a pluggable brain seam. M1 introduces that seam without changing observable behaviour for any existing predator-bearing scenario, and adds the per-predator metrics that future predator-fitness functions will read.

This change is the prerequisite for M5; it ships only the substrate (Protocol + heuristic adapter + per-predator metrics). Learnable predator brains (e.g. MLPPPO predators) and predator-side evolution loops are explicitly M5 scope.

## What Changes

- New module `quantumnematode/env/predator_brain.py` containing `PredatorBrain` Protocol, `PredatorBrainParams` dataclass, `PredatorAction` enum, and `HeuristicPredatorBrain` adapter that encapsulates the existing `_update_pursuit` / `_update_random` logic byte-for-byte.
- `Predator.update_position` (env.py) now delegates entirely to `self.brain.run_brain(params)` plus a new `_apply_action` helper that owns the accumulator + grid-clamp logic. The legacy `_update_pursuit` / `_update_random` helpers are deleted; `HeuristicPredatorBrain` becomes the single source of truth for heuristic behaviour.
- `Predator` gains a synthesised `predator_id: str` (assigned `f"predator_{i}"` at spawn) and three per-instance counters (`kills`, `prey_proximity_steps`, `distance_traveled`).
- `PredatorParams` gains an optional `brain_config: PredatorBrainConfig | None = None` field; `_initialize_predators` defaults to `HeuristicPredatorBrain` when `brain_config` is `None`, so every existing scenario YAML continues to work unchanged.
- `PredatorConfig` (YAML schema, `config_loader.py`) gains an optional `brain_config` block. Only `kind: "heuristic"` is honoured in this change; M5 will extend the dispatcher with learnable brain kinds.
- `MultiAgentEpisodeResult` gains three per-predator metric dicts: `per_predator_kills`, `per_predator_prey_proximity_steps`, `per_predator_distance_traveled`, mirroring the existing `per_agent_food` / `per_agent_reward` pattern. Kill attribution under multi-predator damage uses the closest-by-Manhattan predator; ties broken by lexicographic `predator_id`.

Not breaking. Every existing scenario YAML produces identical agent-survival rates within ±2pp on the M1 regression-baseline campaign (4 seeds × 200 episodes × 3 multi-agent pursuit scenarios).

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `environment-simulation`: extends the "Predator Entities in Dynamic Environments" requirement with a pluggable brain seam. Predators now expose an extensible policy abstraction via `PredatorBrain`; the default `HeuristicPredatorBrain` SHALL preserve current heuristic behaviour byte-for-byte when no `brain_config` is supplied. Adds per-predator ID synthesis, the `_apply_action` accumulator-ownership invariant, and the action-space contract (`PredatorAction.STAY/UP/DOWN/LEFT/RIGHT`).
- `multi-agent`: extends the "Multi-Agent Metrics and Results" requirement with per-predator metric dicts (`per_predator_kills`, `per_predator_prey_proximity_steps`, `per_predator_distance_traveled`) keyed by synthesised predator IDs. Defines kill-attribution rule for multi-predator damage and proximity-step counting semantics.

## Impact

**Code:**

- New: `packages/quantum-nematode/quantumnematode/env/predator_brain.py`
- New: `packages/quantum-nematode/tests/quantumnematode_tests/env/test_predator_brain.py`
- Modified: `packages/quantum-nematode/quantumnematode/env/env.py` (Predator class, PredatorParams, `_initialize_predators`)
- Modified: `packages/quantum-nematode/quantumnematode/utils/config_loader.py` (`PredatorConfig.brain_config` field, `to_params` translation)
- Modified: `packages/quantum-nematode/quantumnematode/agent/multi_agent.py` (`MultiAgentEpisodeResult` dict fields, step-loop accumulation, kill attribution, `_build_result`)

**Configs:** No existing scenario YAML touched. Existing predator configs continue to instantiate `HeuristicPredatorBrain` by default.

**Tests:** Adds `tests/env/test_predator_brain.py` with Protocol conformance, heuristic byte-equivalence parametrised across `PredatorType ∈ {STATIONARY, PURSUIT}` × `speed ∈ {0.5, 1.0, 2.0}`, YAML dispatch tests, and per-predator-metric tests in the multi-agent test module.

**Regression gate:** 4 seeds × 200 episodes × 3 multi-agent pursuit scenarios; per-cell `|post − pre| ≤ 0.02` agent-survival rate, per-scenario mean delta ≤ 2pp. Recorded in logbook 016.

**Out of scope (M5+):** `MLPPPOPredatorBrain` / any learnable predator brain, `PredatorBrainEncoder` for the evolution loop, predator-specific reward/fitness functions, multi-predator coordination (predator-side pheromones), heuristic improvements (smarter pursuit, A\* pathfinding), per-predator visualisation badges in the renderer, and per-predator metrics in the single-agent `EpisodeResult` (existing per-agent metric pattern lives entirely in `MultiAgentEpisodeResult`).

**Dependencies:** None added. The Protocol uses only `typing.Protocol` + `dataclasses` from stdlib.
