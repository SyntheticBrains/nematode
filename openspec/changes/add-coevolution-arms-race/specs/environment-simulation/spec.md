## ADDED Requirements

### Requirement: Learnable Predator Brain Dispatch

The system SHALL register `MLPPPOPredatorBrain` as a learnable implementation of the existing `PredatorBrain` Protocol (see "Predator Brain Abstraction") and dispatch it via the `PredatorBrainConfig.kind` Literal extension. When `kind: "mlpppo_predator"` is specified in YAML, the runtime SHALL construct a `MLPPPOPredatorBrain` with predator-specific I/O wrapping; when `kind: "heuristic"` (or `brain_config` omitted), the runtime SHALL continue to construct a `HeuristicPredatorBrain` byte-for-byte equivalent to the legacy heuristic behaviour.

#### Scenario: PredatorBrainConfig kind extension

- **GIVEN** the runtime `PredatorBrainConfig` dataclass and the YAML `PredatorBrainConfigSchema` Pydantic model
- **THEN** the `kind` field SHALL be a `Literal["heuristic", "mlpppo_predator"]`
- **AND** `kind: "heuristic"` SHALL remain the default (preserves existing scenario YAML behaviour)
- **AND** any value not in the literal SHALL be rejected at YAML validation time

#### Scenario: Heuristic Default Preserved

- **GIVEN** a YAML config with no `brain_config:` block (or with `kind: "heuristic"`)
- **WHEN** the environment initialises
- **THEN** the predator SHALL be constructed with `HeuristicPredatorBrain`
- **AND** behaviour SHALL be byte-equivalent to the M1 baseline (no learnable code path enters)

#### Scenario: Learnable Dispatch on mlpppo_predator

- **GIVEN** a YAML config with `predators.brain_config: {kind: "mlpppo_predator", ...}`
- **WHEN** the environment initialises via `_build_predator_brain`
- **THEN** each spawned predator SHALL be constructed with a `MLPPPOPredatorBrain` instance
- **AND** the brain SHALL satisfy the `PredatorBrain` Protocol (`run_brain`, `prepare_episode`, `post_process_episode`, `copy`)
- **AND** the brain SHALL be `isinstance(brain, PredatorBrain)` via the `@runtime_checkable` Protocol from M1

#### Scenario: Action Space Compatibility

- **GIVEN** a `MLPPPOPredatorBrain.run_brain(params)` invocation
- **THEN** the return value SHALL be one of `PredatorAction.{STAY, UP, DOWN, LEFT, RIGHT}`
- **AND** the harness `Predator._apply_action_loop` SHALL own the accumulator + grid clamp (unchanged from M1)
- **AND** the brain SHALL NOT mutate `Predator.position` directly

### Requirement: MLPPPO Predator I/O Encoding Contract

The system SHALL define a fixed-dimensional, normalised input encoding for the `MLPPPOPredatorBrain` that derives observations from the existing `PredatorBrainParams` surface (see "Predator Brain Abstraction") without requiring the brain to access env internals.

#### Scenario: Input Encoding Components

- **GIVEN** a `MLPPPOPredatorBrain` instance and a `PredatorBrainParams params` for one accumulator-step
- **WHEN** the brain encodes the observation
- **THEN** the input SHALL be an ordered, fixed-length float vector composed of:
  - `params.predator_position[0] / params.grid_size` (1 float)
  - `params.predator_position[1] / params.grid_size` (1 float)
  - For each of `params.agent_positions[:k_nearest=2]`: `(x / grid_size, y / grid_size, present_flag ∈ {0, 1})` (3 floats × 2 = 6 floats)
  - `params.detection_radius / params.grid_size` (1 float)
  - `params.damage_radius / params.grid_size` (1 float)
  - `params.step_index / max_steps` (1 float)
- **AND** the total input dimension SHALL be 11 floats
- **AND** when fewer than `k_nearest` agents are alive, missing slots SHALL be filled with zeros and `present_flag=0`

#### Scenario: Output Action Mapping

- **GIVEN** the brain's policy head emits a 5-way categorical
- **WHEN** an action is sampled
- **THEN** the index→action mapping SHALL be `0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT`
- **AND** the returned `PredatorAction` SHALL be the corresponding enum member

#### Scenario: Determinism Under Fixed Seed

- **GIVEN** two `MLPPPOPredatorBrain` instances with identical weights and identical `PredatorBrainParams` (including identical RNG state)
- **WHEN** both invoke `run_brain` on the same params
- **THEN** they SHALL return the same `PredatorAction`
- **AND** their RNG state SHALL advance identically (matches the env-RNG-shared invariant from M1)

### Requirement: Predator Brain Pretraining

The system SHALL provide a behavioural-cloning pretrain helper that trains a `MLPPPOPredatorBrain` to imitate `HeuristicPredatorBrain` decisions, used to bootstrap gen-0 predator weights for the M5 co-evolution loop.

#### Scenario: Imitation Loss Decreases

- **GIVEN** a fresh `MLPPPOPredatorBrain` with random-init weights and a `HeuristicPredatorBrain` teacher
- **WHEN** the pretrain helper runs for 50 episodes against a representative env config
- **THEN** the cross-entropy imitation loss SHALL decrease monotonically in a windowed sense (final-window mean < initial-window mean)
- **AND** the trained brain SHALL match the teacher's action on more than 70% of held-out test states

#### Scenario: Pretrained Weights Round-Trip Through Encoder

- **GIVEN** a pretrained `MLPPPOPredatorBrain` instance
- **WHEN** the brain's weights are extracted via `WeightPersistence`, encoded as a genome, and decoded back into a fresh brain
- **THEN** the decoded brain SHALL produce the same action as the original on a fixed test set of `PredatorBrainParams`
