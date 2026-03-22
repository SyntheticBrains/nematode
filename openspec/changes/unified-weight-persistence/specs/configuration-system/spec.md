## ADDED Requirements

### Requirement: Weight Path Configuration Field

The base `BrainConfig` class SHALL support an optional `weights_path` field for config-based weight loading.

#### Scenario: Weights Path Field on BrainConfig

- **WHEN** any brain configuration is parsed
- **THEN** `BrainConfig` SHALL accept an optional `weights_path` field (str | None, default None)
- **AND** the field SHALL be inherited by all brain-specific config classes

#### Scenario: Weights Path in YAML

- **WHEN** a YAML config includes `brain.config.weights_path: "artifacts/models/stage1.pt"`
- **THEN** the configuration system SHALL parse it as a string path
- **AND** SHALL pass it through to the brain config instance

#### Scenario: Weights Path Default

- **WHEN** a YAML config does not include `weights_path` under brain config
- **THEN** the value SHALL default to `None`
- **AND** no weight loading SHALL occur from config

#### Scenario: Backward Compatibility

- **WHEN** existing YAML configs that do not include `weights_path` are loaded
- **THEN** parsing SHALL succeed without errors
- **AND** all existing brain config fields SHALL continue to work unchanged
