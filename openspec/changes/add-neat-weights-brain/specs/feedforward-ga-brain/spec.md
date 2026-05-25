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

### Requirement: GA-Based Weight Evolution Integration

The `FeedforwardGABrain` SHALL integrate with the existing evolution-framework infrastructure (`evolution/loop.py`, `evolution/genome.py`, `evolution/encoders.py`, `evolution/fitness.py`) such that running the brain through `scripts/run_evolution.py` evolves its weights across generations without modifying any evolution-framework code. The brain SHALL accumulate episode-level fitness signal through the standard Brain Protocol hooks.

#### Scenario: Brain weights are GA-evolved, not gradient-updated

- **GIVEN** an instance of `FeedforwardGABrain`
- **WHEN** the brain's `update_memory()` method is called with per-step reward signal
- **THEN** the method SHALL accumulate the reward into the brain's per-episode fitness accumulator
- **AND** the method SHALL NOT perform any gradient computation or parameter update
- **WHEN** the brain's `post_process_episode()` method is called
- **THEN** the method SHALL finalise the episode's fitness score (e.g. cumulative reward) and expose it for GA selection
- **AND** the method SHALL NOT perform any gradient computation or parameter update

#### Scenario: Brain is consumed by the existing evolution loop

- **WHEN** `scripts/run_evolution.py` is invoked with a config naming `feedforwardga` as the brain
- **THEN** the launcher SHALL instantiate the brain through the L1 plugin registry (no per-architecture branch in the launcher)
- **AND** the existing `EvolutionLoop` SHALL run the brain across generations, applying the `GeneticAlgorithmOptimizer`'s selection / mutation / crossover steps to the brain's weight tensors
- **AND** the launcher SHALL produce the same standard evolution artefacts (per-generation fitness CSV, best-genome export) as it does for other GA-evolved configurations

#### Scenario: GA optimisation produces deterministic results with a seeded RNG

- **GIVEN** two `FeedforwardGABrain` evolution runs invoked with identical configs and identical RNG seeds
- **WHEN** both runs complete the same number of generations on the same fitness task
- **THEN** the per-generation best-fitness trajectory SHALL be byte-identical across the two runs
- **AND** the final-generation best-genome weight tensor SHALL be byte-identical across the two runs

### Requirement: FeedforwardGABrainConfig

The brain SHALL be configured via a Pydantic `FeedforwardGABrainConfig` model exposing the topology dimensions, GA hyperparameters, and sensory-module configuration. The config SHALL default to MLPPPO-small-matched capacity so the cross-architecture comparison framing holds without per-config overrides.

#### Scenario: Required fields

- **WHEN** `FeedforwardGABrainConfig` is constructed
- **THEN** the config SHALL accept the following fields:
  - `hidden_dim: int` (default `64`, matches `MLPPPOBrainConfig.actor_hidden_dim` small default)
  - `num_hidden_layers: int` (default `2`, matches `MLPPPOBrainConfig.num_hidden_layers` small default)
  - `population_size: int` (default `32`, exposed to `GeneticAlgorithmOptimizer`)
  - `mutation_sigma: float` (default `0.1`, Gaussian-mutation standard deviation passed to the optimiser)
  - `tournament_size: int` (default `3`, selection-pressure parameter passed to the optimiser)
  - `crossover_rate: float` (default `0.5`, two-point-crossover probability passed to the optimiser)
  - `elite_fraction: float` (default `0.1`, mu+lambda elite-preservation fraction passed to the optimiser)
  - `sensory_modules: list[str]` (mirrors `MLPPPOBrainConfig.sensory_modules`; drives the input-dim inference at construction time)
- **AND** the config SHALL be a `BaseBrainConfig` subclass so it is consumed by the existing `BRAIN_CONFIG_MAP` registry

### Requirement: Feedforward GA Brain Registration

The `FeedforwardGABrain` SHALL be registered through the brain plugin registry (`brain-architecture` capability) and SHALL be instantiable through the same `instantiate_brain(...)` code path as every other brain.

#### Scenario: FeedforwardGABrain self-registers at import

- **WHEN** `quantumnematode.brain.arch.feedforward_ga` is imported
- **THEN** the registry SHALL contain a registration tuple for `("feedforwardga", FeedforwardGABrainConfig, FeedforwardGABrain, BrainType.FEEDFORWARDGA, ("classical",))`
- **AND** `list_registered_brains()` SHALL include `"feedforwardga"`

#### Scenario: Feedforward GA brain instantiates through the same code path as MLPPPO

- **WHEN** `instantiate_brain("mlpppo", mlpppo_cfg)` and `instantiate_brain("feedforwardga", ga_cfg)` are called from the same caller
- **THEN** both calls SHALL return Brain Protocol-conforming instances
- **AND** neither call SHALL require a brain-specific branch in the caller's code (the dispatch happens inside `instantiate_brain` via the registry only)

### Requirement: Smoke Config and End-to-End Verification

The capability SHALL ship a smoke config that demonstrates end-to-end weight evolution on the existing klinotaxis foraging substrate, runnable headlessly through `scripts/run_evolution.py`.

#### Scenario: Smoke config runs end-to-end without errors

- **WHEN** `uv run python scripts/run_evolution.py --config configs/scenarios/foraging/feedforwardga_small_klinotaxis.yml --theme headless --generations 10 --seed 2026` is invoked
- **THEN** the run SHALL complete 10 generations without raising any exception
- **AND** the run SHALL produce the standard evolution-framework per-generation fitness CSV
- **AND** the best-fitness-per-generation series SHALL show non-degenerate variance across generations (i.e. the GA is actually exploring weight space, not stuck at the initial population's fitness)
