## ADDED Requirements

### Requirement: Learnable Predator Brain Dispatch

The system SHALL register `MLPPPOPredatorBrain` as a learnable implementation of the existing `PredatorBrain` Protocol (see "Predator Brain Abstraction") and dispatch it via the `PredatorBrainConfig.kind` Literal extension. When `kind: "mlpppo_predator"` is specified in YAML, the runtime SHALL construct a `MLPPPOPredatorBrain` with predator-specific I/O wrapping; when `kind: "heuristic"` (or `brain_config` omitted), the runtime SHALL continue to construct a `HeuristicPredatorBrain` byte-for-byte equivalent to the legacy heuristic behaviour.

**Import boundary (no env→evolution circular dep):** `_build_predator_brain` in `quantumnematode/env/env.py` SHALL import `MLPPPOPredatorBrain` directly from `quantumnematode/env/mlpppo_predator_brain.py`. The dispatcher SHALL NOT consult the `PREDATOR_ENCODER_REGISTRY` (which lives in `quantumnematode/evolution/predator_encoders.py` and is reserved for `CoevolutionLoop` and other evolution-time consumers). This separation prevents an `env` → `evolution` import dependency.

**Seed propagation:** `_build_predator_brain` is a method on `DynamicForagingEnvironment` and takes no explicit seed argument. Seed flows through the `PredatorBrainConfig.extra` dict via the optional `seed` key (a `dict[str, Any]` carried on the `PredatorBrainConfig` dataclass and the `PredatorBrainConfigSchema` Pydantic model). When `extra["seed"]` is set, the dispatcher passes it as the `seed` constructor argument to `MLPPPOPredatorBrain.__init__`, which then calls `torch.manual_seed(seed)` to make orthogonal-init reproducible. (Note: `torch.manual_seed` rather than the agent-side helper `set_global_seed`, because the predator brain has no `BrainConfig`-shaped seed plumbing — it's a leaner Protocol-based brain with direct `torch` setup. Effect is equivalent for orthogonal-init reproducibility.)

**Optional `extra` config keys honoured by the `mlpppo_predator` dispatcher:**

- `actor_hidden_dim: int` — overrides `DEFAULT_ACTOR_HIDDEN_DIM` (default 64).
- `critic_hidden_dim: int` — overrides `DEFAULT_CRITIC_HIDDEN_DIM` (default 64).
- `num_hidden_layers: int` — overrides `DEFAULT_NUM_HIDDEN_LAYERS` (default 2; must be >= 1).
- `seed: int` — torch RNG seed for orthogonal-init reproducibility.
- `sample: bool` — when True, `run_brain` samples from the action distribution via `np.searchsorted` on cumulative softmax (consuming one `params.rng.random()` draw per call); when False (default), `run_brain` returns the deterministic argmax action without consuming any RNG state.

A `weights_path` load-from-disk hook is intentionally NOT included in PR 1 — M5 co-evolution loads pre-trained weights via the genome encoder (PR 2 / PR 3), not via this dispatcher. The hook may be added to `extra` in a future PR if standalone scenarios need to spawn pre-trained predators outside the co-evolution loop.

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
  - `params.step_index / max_steps_normalizer` (1 float, where `max_steps_normalizer` is currently a hardcoded module constant of 1000 — matching the M3 lamarckian / pilot scenario default `max_steps`)
- **AND** the total input dimension SHALL be 11 floats
- **AND** when fewer than `k_nearest` agents are alive, missing slots SHALL be filled with zeros and `present_flag=0`
- **AND** `max_steps` is NOT carried on `PredatorBrainParams` — the normaliser lives as a module-level literal in `mlpppo_predator_brain.py`. Future scenarios with `max_steps != 1000` would need to pass the value via `extra` config or extend `PredatorBrainParams`; deferred until a non-1000 scenario is needed

#### Scenario: Output Action Mapping

- **GIVEN** the brain's policy head emits a 5-way categorical
- **WHEN** an action is sampled
- **THEN** the index→action mapping SHALL be `0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT`
- **AND** the returned `PredatorAction` SHALL be the corresponding enum member

#### Scenario: Determinism Under Fixed Seed (argmax mode, default)

- **GIVEN** two `MLPPPOPredatorBrain` instances with identical weights, both constructed with `sample=False` (the default)
- **AND** identical `PredatorBrainParams` (including identical `rng` state)
- **WHEN** both invoke `run_brain` on the same params
- **THEN** they SHALL return the same `PredatorAction`
- **AND** the env's RNG state SHALL NOT advance (argmax mode is RNG-free; the brain reads logits and picks `argmax` deterministically)
- **AND** this differs from `HeuristicPredatorBrain`'s out-of-pursuit branch which DOES consume one `rng.integers(4)` draw — argmax-mode `MLPPPOPredatorBrain` has no equivalent RNG-coupling

#### Scenario: Determinism Under Fixed Seed (sample mode)

- **GIVEN** two `MLPPPOPredatorBrain` instances with identical weights, both constructed with `sample=True`
- **AND** identical `PredatorBrainParams` (including identical seeded `rng` state — both rng instances at byte-identical state)
- **WHEN** both invoke `run_brain` on the same params
- **THEN** they SHALL return the same `PredatorAction`
- **AND** the env's RNG state SHALL advance by exactly one `rng.random()` draw per `run_brain` call (consumed by `np.searchsorted` on the cumulative softmax for the categorical sample). This matches the env-RNG-shared invariant from M1 — the brain consumes the env's RNG rather than carrying its own
- **AND** sample mode is NOT used in M5 co-evolution by default (PR 2 / PR 3 fitness functions construct brains with `sample=False`); the mode is exposed for ablation experiments and possible future exploration-noise injection during pretrain

### Requirement: Predator Brain Pretraining

The system SHALL provide a behavioural-cloning pretrain helper at `quantumnematode/env/_predator_brain_pretrain.py` (`pretrain_against_heuristic`) that trains a `MLPPPOPredatorBrain` to imitate `HeuristicPredatorBrain` decisions. The helper bootstraps gen-0 predator weights for the M5 co-evolution loop (D7 arm A); CMA-ES outer-loop evolution then refines from this starting point.

**Module name (`_predator_brain_pretrain.py`):** the leading underscore signals "module-private" but `pretrain_against_heuristic` is exported and called by `CoevolutionLoop.__init__` (PR 3) for arm-A bootstrap. Naming reflects "internal helper to the M5 stack", not actual access restriction.

**Training scope:**

- **Actor-only optimization.** The Adam optimizer is constructed over `brain.actor.parameters()` only — the critic head receives no gradient signal because the synthesis pipeline produces only action labels (from the heuristic teacher), not value targets. Critic weights remain at their orthogonal-init values until CMA-ES outer-loop evolution updates them via `WeightPersistence`.
- **In-pursuit-only filtering.** Synthesised `PredatorBrainParams` are filtered to in-pursuit states (`params.is_pursuing == True`) before being added to a training batch. Out-of-pursuit teacher actions are uniform-random (`rng.integers(4)` draw mapped to UP/DOWN/LEFT/RIGHT) and provide no learnable signal — including them would cap held-out accuracy at ~40% even with a perfect classifier on the in-pursuit branch. Pretraining teaches "chase the nearest agent on the larger-delta axis" only; out-of-pursuit policy emerges from CMA-ES outer-loop evolution.
- **Synthesised training data.** Training pairs `(PredatorBrainParams, heuristic_action)` are synthesised from random env-shaped states via `_synthesize_params` — NOT rolled out from a real `DynamicForagingEnvironment`. The heuristic teacher is deterministic given params, so synthesis is faithful (the brain doesn't care whether params came from a real env rollout or were sampled).

#### Scenario: Imitation Loss Decreases

- **GIVEN** a fresh `MLPPPOPredatorBrain` with random-init weights and a `HeuristicPredatorBrain` teacher
- **WHEN** `pretrain_against_heuristic(brain, teacher, num_batches=50, seed=...)` runs (defaults: 50 batches × batch_size=64 in-pursuit synthesised samples × Adam lr=1e-3 on actor parameters only)
- **THEN** the final-window mean cross-entropy imitation loss (last 10 batches) SHALL be strictly less than the initial-window mean (first 10 batches) by a non-trivial margin (≥ 0.05 absolute reduction in mean cross-entropy, indicating real gradient signal beyond stochastic noise)
- **AND** monotonicity is NOT required (SGD on noisy synthesised batches naturally non-monotonic; the windowed-mean comparison is the falsifiable claim)
- **AND** action-match accuracy on held-out in-pursuit states is NOT a spec invariant — the 50-batch budget primarily teaches the brain to break the orthogonal-init symmetry and bias the actor toward agent-direction-correlated outputs; the heuristic teacher's chase logic (axis-greedy with horizontal-first tie-break) requires learning an `argmax(|dx|, |dy|)` operator that converges slowly from raw normalised position inputs, and the residual policy is left for CMA-ES outer-loop evolution to refine. Pretraining is bootstrapping (avoid zero-fitness-gradient gen-0), not a replacement for evolution
- **AND** the critic weights SHALL be unchanged across the pretrain (no gradient flow); only actor weights move

#### Scenario: Pretrained Weights Round-Trip Through Encoder

- **GIVEN** a pretrained `MLPPPOPredatorBrain` instance
- **WHEN** the brain's weights are extracted via `WeightPersistence` (yielding `policy` and `value` components), encoded as a genome, and decoded back into a fresh brain (using a different init seed to verify the load actually overwrites the orthogonal init)
- **THEN** the decoded brain SHALL produce the same action as the original on a fixed test set of `PredatorBrainParams`
- **AND** both `policy` (actor) and `value` (critic) component weights SHALL round-trip — even though pretrain only updates the actor, both heads are part of the genome surface (CMA-ES evolves both during co-evolution)
