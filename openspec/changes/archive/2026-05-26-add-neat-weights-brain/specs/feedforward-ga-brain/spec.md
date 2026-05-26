## ADDED Requirements

### Requirement: Feedforward GA Brain Architecture

The system SHALL provide a feed-forward brain architecture whose weights are evolved by the existing [`GeneticAlgorithmOptimizer`](../../../../packages/quantum-nematode/quantumnematode/optimizers/evolutionary.py) rather than updated by gradient descent. The brain's topology SHALL be a fixed 2-hidden-layer feed-forward network with hidden width matching `MLPPPOBrain` small (`hidden_dim=64`), so the cross-architecture comparison isolates the optimiser-family change (PPO vs GA) from a network-capacity confound. The brain SHALL produce action logits over the 4-action `DEFAULT_ACTIONS` set via a softmax head.

#### Scenario: Brain construction builds the matched-capacity topology

- **WHEN** a brain config specifies `name: feedforwardga`
- **THEN** the brain SHALL construct a feed-forward network with the structure `input_dim → 64 → 64 → 4` using ReLU activations between hidden layers
- **AND** the brain SHALL NOT construct a critic / value head (GA fitness uses episode return directly, not bootstrapped value estimates)
- **AND** the input dimensionality SHALL be derived from the active sensory-module configuration at construction time (mirroring `MLPPPOBrain`'s sensory-module consumption)

#### Scenario: Forward pass produces finite logits over the 4-action set

- **WHEN** `run_brain()` is called with any well-formed `BrainParams`
- **THEN** the returned action logits SHALL be a 4-vector matching the `DEFAULT_ACTIONS` order ([FORWARD, LEFT, RIGHT, STAY])
- **AND** the logits SHALL be finite (no NaN, no Inf)
- **AND** the brain SHALL sample the action by applying softmax to the logits and drawing from the resulting categorical distribution

#### Scenario: Forward pass produces non-degenerate variance across an episode

- **WHEN** the brain is run for at least one episode on the smoke config with non-zero environmental gradients
- **THEN** the variance across the 4-action logits over a sample of ≥ 100 forward passes SHALL be strictly greater than zero
- **AND** the logits SHALL not collapse to a constant action across that sample

### Requirement: WeightPersistence Protocol Conformance

The `FeedforwardGABrain` SHALL implement the `WeightPersistence` protocol from `packages/quantum-nematode/quantumnematode/brain/weights.py:55` so its weights can be serialised + restored by the evolution framework. This is the integration surface that the evolution `Genome` ↔ brain weights round-trip depends on.

#### Scenario: Brain exposes its weights as named WeightComponents

- **GIVEN** an instance of `FeedforwardGABrain`
- **WHEN** `brain.get_weight_components()` is called
- **THEN** the returned dict SHALL contain at minimum one entry per learnable network layer (e.g. `"policy"` covering all hidden + output layer weights), keyed by component name
- **AND** each entry SHALL be a `WeightComponent` carrying a `state` dict of named `torch.Tensor` parameters

#### Scenario: Brain restores weights from a WeightComponents dict

- **GIVEN** an instance of `FeedforwardGABrain` and a `components` dict captured from another instance with the same topology
- **WHEN** `brain.load_weight_components(components)` is called
- **THEN** the brain's network parameters SHALL match the source brain's parameters element-for-element within float32 ulp tolerance after the call
- **AND** the brain's `run_brain()` SHALL subsequently produce identical pre-sampling action logits to the source brain for the same `BrainParams` input

#### Scenario: Brain supports encoder restoration via PPO-attribute shims

- **WHEN** an encoder's `decode()` method assigns `brain._episode_count = 0` and then calls `brain._update_learning_rate()` (per the `_ClassicalPPOEncoder.decode()` contract at `packages/quantum-nematode/quantumnematode/evolution/encoders.py:305-306`)
- **THEN** the `FeedforwardGABrain` instance SHALL accept assignment to `_episode_count` (typed `int`, written but not read by GA logic)
- **AND** the `FeedforwardGABrain` SHALL expose a `_update_learning_rate()` method that is a no-op (GA brain has no LR scheduler)

### Requirement: GA-Based Weight Evolution Integration

The `FeedforwardGABrain` SHALL integrate with the existing evolution-framework infrastructure (`evolution/loop.py`, `evolution/genome.py`, `evolution/encoders.py`, `evolution/fitness.py`) such that running the brain through `scripts/run_evolution.py` evolves its weights across generations without modifying any evolution-framework code beyond the additive encoder registration. The brain SHALL be passive in the fitness pipeline: `update_memory()` and `post_process_episode()` are no-op-safe because `FrozenEvalRunner` monkey-patches them to no-ops at `packages/quantum-nematode/quantumnematode/evolution/fitness.py:122-123` during evaluation.

#### Scenario: Brain's update_memory and post_process_episode do not mutate weights

- **GIVEN** an instance of `FeedforwardGABrain`
- **WHEN** the brain's `update_memory()` method is called with any per-step reward signal
- **THEN** the method SHALL NOT perform any gradient computation, parameter mutation, or LR update
- **WHEN** the brain's `post_process_episode()` method is called with any `episode_success` value
- **THEN** the method SHALL NOT perform any gradient computation, parameter mutation, or LR update
- **AND** neither method SHALL raise an exception when called (so `FrozenEvalRunner`'s monkey-patch boundary remains a no-op safety net rather than a strict contract)

#### Scenario: Encoder is registered in ENCODER_REGISTRY at import

- **WHEN** `quantumnematode.evolution.encoders` is imported
- **THEN** `ENCODER_REGISTRY` SHALL contain a `"feedforwardga"` key
- **AND** the registered encoder SHALL be the `FeedforwardGAEncoder` class

#### Scenario: Brain is consumed by the existing evolution loop

- **WHEN** `scripts/run_evolution.py` is invoked with a config naming `feedforwardga` as the brain
- **THEN** the launcher SHALL instantiate the brain through the L1 plugin registry (no per-architecture branch in the launcher)
- **AND** the existing `EvolutionLoop` SHALL run the brain across generations, applying the `GeneticAlgorithmOptimizer`'s `ask` / `tell` cycle to the brain's weight tensors via the registered `FeedforwardGAEncoder`
- **AND** the launcher SHALL produce the same standard evolution-framework artefacts (per-generation fitness CSV, best-genome export) as it does for other GA-evolved configurations

#### Scenario: Encoder round-trip preserves weights

- **GIVEN** a `FeedforwardGABrain` instance A with arbitrary weights
- **WHEN** the encoder serialises A into a `Genome` and then decodes the same `Genome` into a fresh brain B
- **THEN** B's weight tensors SHALL match A's weight tensors element-for-element within float32 ulp tolerance
- **AND** B's `run_brain()` SHALL produce identical pre-sampling action logits to A's for the same `BrainParams` input

> **Note on logits vs sampled actions.** The forward pass is deterministic given identical weights and identical inputs; the post-softmax categorical sampling step is stochastic and intentionally NOT part of this scenario's comparison surface. Tests SHALL compare logits, not sampled action indices.

### Requirement: FeedforwardGABrainConfig

The brain SHALL be configured via a Pydantic `FeedforwardGABrainConfig` model exposing the topology dimensions and sensory-module configuration. The config SHALL NOT expose GA optimiser hyperparameters — those live in the YAML `evolution:` block per the existing `configs/evolution/mlpppo_foraging_small.yml` precedent and are consumed by the evolution loop. The config SHALL default to MLPPPO-small-matched capacity so the cross-architecture comparison framing holds without per-config overrides.

#### Scenario: Required fields

- **WHEN** `FeedforwardGABrainConfig` is constructed
- **THEN** the config SHALL accept at minimum the following fields:
  - `hidden_dim: int` (default `64`, matches `MLPPPOBrainConfig.actor_hidden_dim` small default)
  - `num_hidden_layers: int` (default `2`, matches `MLPPPOBrainConfig.num_hidden_layers` small default)
  - `sensory_modules: list[ModuleName]` (mirrors `MLPPPOBrainConfig.sensory_modules` at [`packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py:116`](../../../../packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py#L116); drives the input-dim inference at construction time via the existing `get_classical_feature_dimension()` helper which consumes `ModuleName` enum values)
- **AND** the config SHALL be a `BrainConfig` subclass so it is consumed by the existing config-loading machinery

#### Scenario: GA hyperparameters live in the evolution block, not the brain config

- **WHEN** a smoke evolution config naming `feedforwardga` as the brain is constructed
- **THEN** the YAML's `evolution:` block SHALL carry the GA-specific hyperparameters consumed by `GeneticAlgorithmOptimizer.__init__` (`population_size`, `sigma0`, `elite_fraction`, `mutation_rate`, `crossover_rate`, `seed`) — matching the existing `configs/evolution/mlpppo_foraging_small.yml` shape
- **AND** the `brain:` block's `config:` SHALL NOT carry GA hyperparameters (no `population_size`, no `mutation_rate`, etc.)

### Requirement: Feedforward GA Brain Registration

The `FeedforwardGABrain` SHALL be registered through the brain plugin registry (`brain-architecture` capability) and SHALL be instantiable through the same `instantiate_brain(...)` code path as every other brain. Registration SHALL use the `@register_brain` decorator pattern established by `add-architecture-plugin-interface`.

#### Scenario: FeedforwardGABrain self-registers at import

- **WHEN** `quantumnematode.brain.arch.feedforward_ga` is imported
- **THEN** the `@register_brain` decorator applied to `FeedforwardGABrain` SHALL store a `Registration` model in the registry with `name="feedforwardga"`, `config_cls=FeedforwardGABrainConfig`, `brain_type=BrainType.FEEDFORWARDGA`, `families=("classical",)`
- **AND** `list_registered_brains()` SHALL include `"feedforwardga"`

#### Scenario: BrainType enum includes FEEDFORWARDGA

- **WHEN** `quantumnematode.brain.arch.dtypes.BrainType` is inspected
- **THEN** the enum SHALL contain a member `FEEDFORWARDGA = "feedforwardga"`
- **AND** the `BRAIN_TYPES` `Literal` in the same module SHALL include `BrainType.FEEDFORWARDGA`

#### Scenario: Feedforward GA brain instantiates through the same code path as MLPPPO

- **WHEN** `instantiate_brain("mlpppo", mlpppo_cfg)` and `instantiate_brain("feedforwardga", ga_cfg)` are called from the same caller
- **THEN** both calls SHALL return Brain Protocol-conforming instances
- **AND** neither call SHALL require a brain-specific branch in the caller's code (the dispatch happens inside `instantiate_brain` via the registry only)

### Requirement: Smoke Evolution Config and End-to-End Verification

The capability SHALL ship a smoke evolution-config that demonstrates end-to-end weight evolution on the existing klinotaxis foraging substrate, runnable headlessly through `scripts/run_evolution.py`. The config SHALL live under `configs/evolution/` (not `configs/scenarios/`) because evolution configs are a distinct family with a required `evolution:` block.

#### Scenario: Smoke config runs end-to-end without errors

- **WHEN** `uv run python scripts/run_evolution.py --config configs/evolution/feedforwardga_foraging_small.yml --seed 2026` is invoked
- **THEN** the run SHALL complete the generation budget specified in the config without raising any exception
- **AND** the run SHALL produce the standard evolution-framework per-generation fitness CSV
- **AND** the best-fitness-per-generation series SHALL show non-degenerate variance across generations (i.e. the GA is actually exploring weight space, not stuck at the initial population's fitness)
- **AND** the launcher SHALL NOT accept a `--theme` flag (evolution always runs headless; reference: `scripts/run_evolution.py` argparser)
