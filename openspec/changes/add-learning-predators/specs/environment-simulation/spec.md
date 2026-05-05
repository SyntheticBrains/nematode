## ADDED Requirements

### Requirement: Predator Brain Abstraction

The system SHALL expose a pluggable policy seam for predators via a `PredatorBrain` Protocol, allowing future co-evolutionary work to introduce learnable predator brains without changing the predator step loop or env state. The default `HeuristicPredatorBrain` SHALL preserve the heuristic predator behaviour byte-for-byte (same RNG draws, same target tie-breaking, same accumulator timing) when no `brain_config` is supplied in the YAML.

#### Scenario: PredatorBrain Protocol Surface

- **GIVEN** a class implementing `PredatorBrain`
- **THEN** it SHALL implement `run_brain(params: PredatorBrainParams) -> PredatorAction`
- **AND** it SHALL implement `prepare_episode() -> None` (may be a no-op)
- **AND** it SHALL implement `post_process_episode(*, episode_success: bool | None = None) -> None` (may be a no-op)
- **AND** it SHALL implement `copy() -> PredatorBrain`
- **AND** the Protocol SHALL be `@runtime_checkable` so `isinstance` works for runtime dispatch

#### Scenario: Action Space Contract

- **GIVEN** a `PredatorBrain.run_brain` invocation
- **WHEN** the brain returns its decision
- **THEN** the return value SHALL be one of `PredatorAction.{STAY, UP, DOWN, LEFT, RIGHT}`
- **AND** the harness (`Predator._apply_action`) SHALL own the `movement_accumulator` advance, the multi-step-per-update loop (capped at 10), and the `max(0, ...)` / `min(grid_size-1, ...)` grid clamp
- **AND** the brain SHALL NOT mutate the predator's `position` directly

#### Scenario: PredatorBrainParams Surface

- **GIVEN** a `PredatorBrainParams` instance passed to `run_brain`
- **THEN** it SHALL be a frozen dataclass with fields: `predator_id`, `predator_position`, `predator_type`, `detection_radius`, `damage_radius`, `agent_positions`, `grid_size`, `rng`, `step_index`
- **AND** `agent_positions` SHALL be a tuple ordered by env's `agents.values()` insertion order so target tie-breaking is deterministic
- **AND** `rng` SHALL be the env's RNG so RNG-state advancement is shared with downstream consumers (food spawning, agent decisions)

#### Scenario: HeuristicPredatorBrain Byte-Equivalence

- **GIVEN** two predators with identical `position`, `predator_type`, `speed`, `detection_radius`, and `damage_radius`
- **AND** identical seeded RNG states
- **WHEN** one predator runs the legacy `_update_pursuit` / `_update_random` logic and the other runs `HeuristicPredatorBrain.run_brain` followed by `_apply_action` for 1000 steps
- **THEN** their `position` trajectories SHALL be step-by-step equal across `PredatorType ∈ {STATIONARY, PURSUIT}` × `speed ∈ {0.5, 1.0, 2.0}`
- **AND** the env RNG state SHALL advance identically in both cases

#### Scenario: Default Brain Construction at Spawn

- **GIVEN** a YAML config with `predators.enabled: true` and no `predators.brain_config:` block
- **WHEN** the environment initialises
- **THEN** `PredatorParams.brain_config` SHALL be `None`
- **AND** `_initialize_predators` SHALL construct a `HeuristicPredatorBrain` for each spawned predator
- **AND** the predator's heuristic behaviour SHALL be unchanged from pre-refactor

#### Scenario: Explicit Heuristic Brain Configuration

- **GIVEN** a YAML config with `predators.brain_config: {kind: "heuristic"}`
- **WHEN** the environment initialises
- **THEN** the resulting predator SHALL behave identically to the default-brain case (same `HeuristicPredatorBrain` instance type)

### Requirement: Predator ID Synthesis

The system SHALL assign a stable, deterministic `predator_id` to each spawned predator so per-predator metrics can be keyed unambiguously and the kill-attribution tie-break rule has a defined ordering.

#### Scenario: ID Format and Assignment

- **GIVEN** an environment with `predators.count: N`
- **WHEN** `_initialize_predators` runs
- **THEN** the i-th spawned predator SHALL have `predator_id == f"predator_{i}"`
- **AND** IDs SHALL be lexicographically ordered (so `predator_0 < predator_1 < ... < predator_{N-1}`)
- **AND** ID assignment SHALL be deterministic given the same env config and seed

#### Scenario: ID Stability Across Episodes

- **GIVEN** the same env config and same seed
- **WHEN** the env is reset between episodes
- **THEN** each predator's `predator_id` SHALL remain unchanged across resets

## MODIFIED Requirements

### Requirement: Predator Entities in Dynamic Environments

The system SHALL support configurable predator entities in dynamic foraging environments that move independently and pose a threat to the agent. Each predator SHALL have a stable `predator_id` (assigned by spawn order, see "Predator ID Synthesis") and SHALL delegate its per-step movement decision to a `PredatorBrain` (see "Predator Brain Abstraction"). The default `HeuristicPredatorBrain` preserves the original `STATIONARY` / `PURSUIT` movement semantics.

#### Scenario: Predator Initialization

- **GIVEN** a dynamic environment configured with `predators.enabled: true` and `predators.count: 3`
- **WHEN** the environment is initialized
- **THEN** exactly 3 predator entities SHALL be spawned at random valid positions
- **AND** each predator SHALL have independent position and movement state
- **AND** each predator SHALL be assigned a synthesised `predator_id` (`"predator_0"`, `"predator_1"`, `"predator_2"`)
- **AND** each predator SHALL be constructed with a `HeuristicPredatorBrain` by default (when no `brain_config` supplied)
- **AND** predators SHALL be tracked separately from food sources

#### Scenario: Predator Random Movement

- **GIVEN** a predator at position (10, 10) with `movement_pattern: "random"`
- **WHEN** the predator updates its position
- **THEN** the predator SHALL move one cell in a random direction (up, down, left, right)
- **AND** the predator SHALL stay within bounds of the environment
- **AND** movement SHALL occur at rate specified by `speed` parameter (default 1.0)
- **AND** the random direction SHALL be drawn from a single `rng.integers(4)` call inside the brain's `run_brain` method

#### Scenario: Predator Speed Configuration

- **GIVEN** a predator configured with `speed: 0.5`
- **WHEN** the simulation advances
- **THEN** the predator SHALL move at half the rate of the agent
- **AND** this SHALL be implemented via accumulator-based movement (moves every 2 steps)
- **AND** the accumulator advancement SHALL happen in `Predator._apply_action`, not in the brain

#### Scenario: Multiple Predators Independence

- **GIVEN** 3 predators in the environment
- **WHEN** each predator updates
- **THEN** each predator SHALL move independently
- **AND** predators MAY occupy the same cell
- **AND** predators SHALL not interact with each other
- **AND** each predator's brain SHALL be invoked once per accumulator-step
