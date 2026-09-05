# Spec: connectome-ppo-brain

## ADDED Requirements

### Requirement: Sanity-floor arms share the substrate they bound

The panel's floors are only interpretable if they differ from the arm they bound in the dimension under test and in nothing else. Selecting a floor SHALL therefore change the learning behaviour without changing the substrate the learning acts on.

Concretely: a frozen-weights floor and an unmodulated-Hebbian floor SHALL construct the same topology parameters, including the **same motor readout**, as the plasticity arm they are compared against at the same seed. Because the readout is set to the anatomical contrast under the plasticity rule and left at its random initialisation under the gradient rule, a floor configured under the gradient rule would carry a different decoder, and any gap between the arms would confound decoding with learning.

`ConnectomePPOBrainConfig` SHALL expose the unmodulated mode through the same rule-selection field as the other rules, so an arm is a one-key change. The unmodulated selection SHALL carry the same activity-trace requirement as the modulated one, validated at load time, since it reads the same trace.

#### Scenario: The floors and the plastic arm share a substrate

- **WHEN** a plastic arm, a frozen-weights floor, and an unmodulated-Hebbian floor are constructed at the same seed
- **THEN** all three SHALL have byte-identical topology parameters at initialisation, the motor readout included

#### Scenario: The frozen floor does not learn

- **GIVEN** a frozen-weights floor
- **WHEN** an episode is run
- **THEN** the chemical weights SHALL be bit-identical to their initial values

#### Scenario: The unmodulated floor learns without reward

- **GIVEN** an unmodulated-Hebbian floor
- **WHEN** an episode is run
- **THEN** the chemical weights SHALL have changed from their initial values
- **AND** running the same episode under a different reward stream SHALL produce the same weights

#### Scenario: Unmodulated selection requires a trace

- **WHEN** a configuration selects the unmodulated mode without enabling activity traces
- **THEN** loading SHALL fail with an error naming both fields

### Requirement: Sanity-floor configurations

Configurations for both floors SHALL ship with this change, each differing from the plastic wild-type arm by the minimal key set that selects it, so the floor comparison holds every other condition constant.

#### Scenario: Each floor config is a minimal delta

- **WHEN** a floor configuration is compared with the plastic wild-type configuration
- **THEN** the only differences SHALL be the keys that select that floor
- **AND** no environment, reward, satiety, or sensing key SHALL differ

#### Scenario: Floor configs load and select their arms

- **WHEN** each floor configuration is loaded through the standard loader
- **THEN** it SHALL validate
- **AND** the resulting brain SHALL be configured as that floor

## MODIFIED Requirements

### Requirement: Learning-rule selection on the connectome brain

`ConnectomePPOBrainConfig` SHALL expose `learning_rule` selecting between the PPO update and the **plasticity rules** — the reward-modulated three-factor rule and its unmodulated ablation — defaulting to PPO. The selection SHALL be a configuration option on the existing brain rather than a separate brain class, so that panel arms differ in exactly the dimension under test and share every other code path.

Selecting **any** plasticity rule SHALL require activity traces to be enabled; the pairing SHALL be validated at load time, because a plasticity rule with no eligibility trace to read would train silently and produce no weight change at all. The requirement SHALL be expressed over the set of plasticity rules rather than by naming one, so a rule added later inherits it rather than silently escaping it.

Under **any** plasticity rule the brain SHALL invoke the rule **once per environment step** as rewards arrive, rather than accumulating a rollout and updating when full. The rollout buffer, advantage estimation, and value head SHALL NOT be exercised.

"Not exercised" requires an explicit mechanism, because the action path computes a state value on **every** step and reaches it through a property that delegates to the PPO rule. Under a plasticity rule that call SHALL be skipped rather than satisfied by a substitute value head: a rule that owns no critic must not be given one merely to keep an unused call site alive. The per-step value and bootstrap value SHALL remain unset, and the experience buffer SHALL NOT be appended to.

The back-compatibility accessors exposing the value head and optimiser SHALL, under any plasticity rule, raise an error naming **the configured rule** and stating that it owns neither, rather than failing with an attribute error from inside the delegation. Naming the configured rule rather than a fixed one matters once more than one plasticity rule exists: an arm told it is a rule it is not would send a reader looking for the wrong behaviour.

The default selection SHALL be byte-identical to the pre-change brain: identical seeds SHALL produce identical parameters after the same number of updates, verified against a frozen reference of the pre-change behaviour.

#### Scenario: Default selection is byte-identical

- **WHEN** a brain is constructed with `learning_rule` unset
- **THEN** every constructed tensor SHALL be byte-identical to a pre-change construction at the same seed
- **AND** training SHALL produce bit-identical parameters to the frozen pre-change reference

#### Scenario: Three-factor without traces fails at load

- **WHEN** a configuration selects any plasticity rule without enabling activity traces
- **THEN** loading SHALL fail with an error naming both fields and stating that the rule requires a trace

#### Scenario: Three-factor updates once per step

- **GIVEN** a brain configured with a plasticity rule
- **WHEN** a sequence of environment steps with rewards is run
- **THEN** one rule update SHALL occur per step
- **AND** the update SHALL NOT wait for a rollout buffer to fill

#### Scenario: PPO machinery is dormant under the three-factor rule

- **GIVEN** a brain configured with a plasticity rule that is not frozen
- **WHEN** an episode runs to completion
- **THEN** no value estimate, advantage, or clipped-surrogate loss SHALL be computed
- **AND** the experience buffer SHALL remain empty
- **AND** the chemical weights SHALL have changed from their initial values

#### Scenario: Action selection does not require a value head

- **GIVEN** a brain configured with a plasticity rule
- **WHEN** actions are selected across many steps
- **THEN** every step SHALL succeed without a value head existing
- **AND** the per-step and bootstrap value state SHALL remain unset

#### Scenario: PPO-only accessors fail informatively

- **GIVEN** a brain configured with a plasticity rule
- **WHEN** the value-head or optimiser accessor is used
- **THEN** it SHALL raise an error naming the configured rule and stating that it owns neither
- **AND** it SHALL NOT name a different plasticity rule

#### Scenario: Selecting a rule does not change construction

- **WHEN** two brains are constructed at the same seed with different rule selections
- **THEN** every topology parameter other than the motor readout SHALL be byte-identical at initialisation
- **AND** the readout SHALL differ only as the anatomical-readout requirement prescribes
