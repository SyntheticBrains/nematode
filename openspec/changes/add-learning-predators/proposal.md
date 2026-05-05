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

Not breaking. Verified at two regression-gate levels:

- **Byte-equivalence unit tests** (the primary gate): 23 tests across `{STATIONARY, PURSUIT} × speed ∈ {0.5, 1.0, 2.0} × {static-agent, moving-agent, multi-agent-target}` parametrisations show step-by-step position equality + RNG-state equality vs. the frozen pre-M1 reference over 1000-step horizons.
- **Multi-metric campaign baseline** (secondary cross-check): 4 seeds × 200 episodes × 3 multi-agent pursuit scenarios + 4 seeds × 100 episodes × 2 single-agent pursuit scenarios = 20 (config, seed) cells × 4 metrics (mean_success, mean_total_food, mean_steps, mean_predator_engagement) = **80 metric-cells with all deltas exactly 0.0** between the pre-M1 base commit and the M1 head commit.

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

**Tests:** 65 net new tests across four files:

- `tests/env/test_predator_brain.py` (21 tests) — Protocol conformance, copy independence, heuristic semantics (stationary STAY, pursuit greedy axis selection with horizontal-first tie-break, RNG-draw count, direction mapping).
- `tests/env/test_predator_brain_byte_equivalence.py` (23 tests) — **the primary regression gate**. Position trajectory + env RNG-state equality vs. `_legacy_predator_reference._LegacyPredatorReference` (frozen pre-M1 implementation) parametrised across `{STATIONARY, PURSUIT} × speed ∈ {0.5, 1.0, 2.0} × {static-agent, moving-agent, multi-agent-target}` over 1000-step horizons.
- `tests/env/test_predator_brain_config.py` (15 tests) — default brain dispatch, explicit `kind: heuristic`, `predator_id` synthesis + lex ordering, ID stability across `update_predators()` calls within an env, ID reproducibility across env instances, unknown-kind rejection (Pydantic + runtime), `env.copy()` preserves IDs + per-predator field values + brain copy independence, YAML schema dispatch.
- `tests/agent/test_multi_agent.py::TestPerPredatorMetrics` (6 tests) — distance/proximity/kill counter semantics, kill attribution at distinct + equal distances, defensive fallback when no predator covers.

Full suite: 2425 non-nightly + 22 smoke = 2447 tests passing post-M1 (was ~2380 pre-M1).

**Regression gate:** Multi-metric two-arm campaign on commits `73684213` (pre-M1.1) and `3d45e75c` (M1 head). 20 (arm, config, seed) cells × 4 metrics = 80 metric-cells; all deltas exactly 0.0. Recorded in logbook 016 + artefacts at `artifacts/logbooks/016-predator-brain-refactor/`.

**Out of scope (M5+):** `MLPPPOPredatorBrain` / any learnable predator brain, `PredatorBrainEncoder` for the evolution loop, predator-specific reward/fitness functions, multi-predator coordination (predator-side pheromones), heuristic improvements (smarter pursuit, A\* pathfinding), per-predator visualisation badges in the renderer, and per-predator metrics in the single-agent `EpisodeResult` (existing per-agent metric pattern lives entirely in `MultiAgentEpisodeResult`).

**Dependencies:** None added. The Protocol uses only `typing.Protocol` + `dataclasses` from stdlib.
