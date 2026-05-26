## MODIFIED Requirements

### Requirement: First-Class Encoder Coverage for Classical Brains

The encoder registry SHALL include `MLPPPOEncoder`, `LSTMPPOEncoder`, and `FeedforwardGAEncoder` at minimum, so that downstream evolutionary work — hyperparameter sweeps, inheritance-strategy pilots, co-evolution, transgenerational memory experiments, and the GA-vs-PPO weight-search cross-architecture comparison — can target classical brains without further framework changes.

#### Scenario: MLPPPO encoder is registered

- **GIVEN** the `quantumnematode.evolution` module is imported
- **WHEN** `ENCODER_REGISTRY["mlpppo"]` is accessed
- **THEN** the value SHALL be the `MLPPPOEncoder` class

#### Scenario: LSTMPPO encoder is registered

- **GIVEN** the `quantumnematode.evolution` module is imported
- **WHEN** `ENCODER_REGISTRY["lstmppo"]` is accessed
- **THEN** the value SHALL be the `LSTMPPOEncoder` class

#### Scenario: LSTMPPO encoder includes all recurrent and feed-forward components

- **GIVEN** an `LSTMPPOBrain` instance (whose `get_weight_components()` returns `{"lstm", "layer_norm", "policy", "value", "actor_optimizer", "critic_optimizer", "training_state"}`)
- **WHEN** the encoder serializes the brain
- **THEN** the serialized weight components SHALL include `"lstm"`, `"layer_norm"`, `"policy"`, and `"value"` (all four learned-weight components)
- **AND** SHALL NOT include `"actor_optimizer"`, `"critic_optimizer"`, or `"training_state"` (denylist)
- **AND** the per-episode hidden state (`_pending_h_state`, `_pending_c_state`) SHALL NOT be part of the genome (it is reset at `prepare_episode()` per existing brain code)

#### Scenario: FeedforwardGA encoder is registered

- **GIVEN** the `quantumnematode.evolution` module is imported
- **WHEN** `ENCODER_REGISTRY["feedforwardga"]` is accessed
- **THEN** the value SHALL be the `FeedforwardGAEncoder` class

#### Scenario: FeedforwardGA encoder serialises the single policy weight component

- **GIVEN** a `FeedforwardGABrain` instance (whose `get_weight_components()` returns `{"policy"}` — no critic, no optimizer state)
- **WHEN** the encoder serialises the brain
- **THEN** the serialised weight components SHALL include `"policy"` (the only learned-weight component)
- **AND** SHALL NOT introduce any `"value"` / `"critic"` / `"optimizer"` / `"training_state"` slot (the GA brain exposes no such components, so the encoder's flatten step has nothing else to serialise)

#### Scenario: FeedforwardGA encoder inherits decode contract via subclassing \_ClassicalPPOEncoder

- **GIVEN** the `FeedforwardGAEncoder` subclasses `_ClassicalPPOEncoder` (per the implementation's encoder hierarchy)
- **WHEN** `encoder.decode(genome, sim_config, seed=...)` runs on a `FeedforwardGABrain`
- **THEN** the inherited decode flow SHALL assign `brain._episode_count = 0` and call `brain._update_learning_rate()` (the existing `_ClassicalPPOEncoder.decode()` contract)
- **AND** the brain SHALL satisfy this contract via no-op shims — `_episode_count` accepts assignment but is not read by GA logic; `_update_learning_rate()` is a no-op because GA has no LR scheduler — so the inherited base class works without modification
