# weight-persistence Specification

## Purpose

Define the unified weight persistence system for brain architectures. Provides a component-based save/load protocol (`WeightPersistence`) that brains opt into, with free functions `save_weights()` / `load_weights()` handling serialization, metadata, and component filtering. Brains that do not implement the protocol are silently skipped on save and raise `TypeError` on load. Implementations are provided for MLPPPOBrain and HybridClassicalBrain.

## Requirements

### Requirement: WeightPersistence Protocol

The system SHALL provide a `WeightPersistence` protocol in `quantumnematode.brain.weights` that brains MAY implement for component-based weight save/load.

#### Scenario: Protocol Definition

- **WHEN** a brain class wants to support structured weight persistence
- **THEN** it SHALL implement the `WeightPersistence` protocol with two methods:
  - `get_weight_components(*, components: set[str] | None = None) -> dict[str, WeightComponent]`
  - `load_weight_components(components: dict[str, WeightComponent]) -> None`
- **AND** the protocol SHALL be `@runtime_checkable` for isinstance checks

#### Scenario: WeightComponent Structure

- **WHEN** a brain returns weight components
- **THEN** each `WeightComponent` SHALL be a dataclass with fields:
  - `name` (str): component identifier
  - `state` (dict[str, Any]): serializable state (state_dict or parameter dict)
  - `metadata` (dict[str, Any]): shape and type information for diagnostics

#### Scenario: Component Filtering on Get

- **WHEN** `get_weight_components(components={"policy", "value"})` is called with a filter set
- **THEN** only the requested components SHALL be returned
- **AND** unknown component names SHALL raise a `ValueError` listing valid component names

#### Scenario: Component Filtering Returns All When None

- **WHEN** `get_weight_components(components=None)` is called without a filter
- **THEN** all components the brain supports SHALL be returned

### Requirement: Save Weights Function

The system SHALL provide a `save_weights()` free function that saves brain weights to a single `.pt` file.

#### Scenario: Save With WeightPersistence Brain

- **WHEN** `save_weights(brain, path)` is called on a brain implementing `WeightPersistence`
- **THEN** the function SHALL call `brain.get_weight_components()` to collect components
- **AND** SHALL write all component states to a single `.pt` file keyed by component name
- **AND** SHALL include a `_metadata` key in the saved dict
- **AND** SHALL use `torch.save()` for serialization

#### Scenario: Save With Non-Implementing Brain

- **WHEN** `save_weights(brain, path)` is called on a brain that does NOT implement `WeightPersistence`
- **THEN** the function SHALL skip saving and log a debug message
- **AND** SHALL NOT raise an error
- **AND** callers requiring strict behavior (e.g. CLI `--save-weights`) SHALL validate `isinstance(brain, WeightPersistence)` before calling

#### Scenario: Save Creates Parent Directories

- **WHEN** `save_weights(brain, path)` is called with a path whose parent directories do not exist
- **THEN** the function SHALL create all necessary parent directories before saving
- **AND** SHALL NOT raise a FileNotFoundError

#### Scenario: Save With Component Filter

- **WHEN** `save_weights(brain, path, components={"policy"})` is called with a component filter
- **THEN** only the specified components SHALL be saved to the file
- **AND** `_metadata.components` SHALL list only the saved components

#### Scenario: Metadata Contents

- **WHEN** a weight file is saved
- **THEN** the `_metadata` key SHALL contain:
  - `brain_type` (str): the brain's class name (e.g. `"MLPPPOBrain"`)
  - `saved_at` (str): ISO 8601 UTC timestamp
  - `components` (list[str]): names of saved components
  - `shapes` (dict\[str, list[int]\]): mapping of `component.param_name` to tensor shape
  - `episode_count` (int | None): extracted from `training_state` component if present, else from `getattr(brain, '_episode_count', None)`

### Requirement: Load Weights Function

The system SHALL provide a `load_weights()` free function that loads brain weights from a `.pt` file.

#### Scenario: Load With WeightPersistence Brain

- **WHEN** `load_weights(brain, path)` is called on a brain implementing `WeightPersistence`
- **THEN** the function SHALL load the file via `torch.load(path, weights_only=True)`
- **AND** SHALL construct `WeightComponent` objects from the loaded data
- **AND** SHALL call `brain.load_weight_components(components)` to apply the weights

#### Scenario: Load With Non-Implementing Brain

- **WHEN** `load_weights(brain, path)` is called on a brain that does NOT implement `WeightPersistence`
- **THEN** the function SHALL raise a `TypeError` with a message indicating the brain does not support weight persistence
- **AND** SHALL name the brain's class in the error message

#### Scenario: Load With Component Filter

- **WHEN** `load_weights(brain, path, components={"policy"})` is called with a filter
- **THEN** only the specified components SHALL be loaded from the file
- **AND** other components in the file SHALL be ignored
- **AND** brain state for non-loaded components SHALL remain unchanged

#### Scenario: Load File Not Found

- **WHEN** `load_weights(brain, path)` is called with a path that does not exist
- **THEN** the function SHALL raise `FileNotFoundError` with the path in the message

#### Scenario: Brain Type Mismatch Warning

- **WHEN** the `_metadata.brain_type` in the file does not match the brain's class name
- **THEN** the function SHALL log a warning with both the expected and actual brain types
- **AND** SHALL proceed with loading (not raise an error)

#### Scenario: Load With Missing Components Warning

- **WHEN** `load_weights(brain, path, components={"policy", "nonexistent"})` is called with a filter that includes names not present in the file
- **THEN** the function SHALL log a warning listing the missing component names and the available components in the file
- **AND** SHALL proceed with loading only the components that were found

#### Scenario: Security — weights_only=True

- **WHEN** loading any weight file
- **THEN** `torch.load()` SHALL always be called with `weights_only=True`
- **AND** SHALL NOT use `weights_only=False` to prevent arbitrary code execution

### Requirement: MLP PPO Weight Persistence

`MLPPPOBrain` SHALL implement the `WeightPersistence` protocol for full save/load support.

#### Scenario: MLP PPO Weight Components

- **WHEN** `get_weight_components()` is called on an MLPPPOBrain
- **THEN** the returned dict SHALL contain these components:
  - `"policy"`: actor network state_dict
  - `"value"`: critic network state_dict
  - `"optimizer"`: joint Adam optimizer state_dict (MLPPPOBrain uses a single optimizer for both actor and critic)
  - `"training_state"`: dict containing `episode_count` (int)

#### Scenario: MLP PPO Load Weight Components

- **WHEN** `load_weight_components()` is called with valid components
- **THEN** the actor state_dict SHALL be loaded via `self.actor.load_state_dict()`
- **AND** the critic state_dict SHALL be loaded via `self.critic.load_state_dict()`
- **AND** optimizer states SHALL be restored
- **AND** `_episode_count` SHALL be restored from `training_state`
- **AND** the learning rate SHALL recalculate from the restored episode count (no scheduler object — computed on-the-fly from `_episode_count` and config)
- **AND** the PPO rollout buffer SHALL be reset to prevent stale experience from corrupting the first update

#### Scenario: MLP PPO Weight Loading Is Caller-Controlled

- **WHEN** `MLPPPOBrain` is instantiated with `config.weights_path` set
- **THEN** the brain SHALL NOT load weights during `__init__`
- **AND** the training loop (`run_simulation.py`) SHALL be responsible for calling `load_weights()` after brain construction, resolving CLI vs config paths

#### Scenario: MLP PPO Round-Trip Consistency

- **WHEN** weights are saved from one MLPPPOBrain instance and loaded into another with identical architecture
- **THEN** both instances SHALL produce identical action probability distributions for the same input
- **AND** actor and critic state_dicts SHALL be equal

#### Scenario: MLP PPO Architecture Mismatch

- **WHEN** weights saved from an MLPPPOBrain with `input_dim=12` are loaded into one with `input_dim=8`
- **THEN** `load_state_dict()` SHALL raise an error indicating shape mismatch
- **AND** the error message SHALL identify which parameter has the wrong shape

#### Scenario: MLP PPO Optimizer State Mismatch

- **WHEN** optimizer state from a different architecture is loaded
- **THEN** the network state_dict load SHALL fail first (catching the mismatch before optimizer state is attempted)
- **AND** optimizer state SHALL only be loaded if network state loads succeed

### Requirement: HybridClassical Weight Persistence

`HybridClassicalBrain` SHALL implement the `WeightPersistence` protocol as a thin wrapper around its existing private save/load methods.

#### Scenario: HybridClassical Weight Components

- **WHEN** `get_weight_components()` is called on a HybridClassicalBrain
- **THEN** the returned dict SHALL contain these components:
  - `"reflex"`: reflex MLP state_dict (`self.reflex_mlp.state_dict()`)
  - `"cortex.policy"`: cortex actor state_dict
  - `"cortex.value"`: cortex critic state_dict

#### Scenario: HybridClassical Partial Load

- **WHEN** `load_weight_components()` is called with only `{"reflex": component}`
- **THEN** only the reflex MLP weights SHALL be loaded
- **AND** cortex actor and critic weights SHALL remain unchanged

#### Scenario: HybridClassical Existing Config Fields Continue Working

- **WHEN** HybridClassicalBrain is instantiated with `reflex_weights_path` or `cortex_weights_path` set
- **THEN** existing load methods SHALL be called as before
- **AND** behavior SHALL be identical to before this change
