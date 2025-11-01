# Spec Delta: Satiety Management

## ADDED Requirements

### Requirement: Satiety manager encapsulates hunger mechanics
The system SHALL provide a `SatietyManager` class that manages the agent's satiety level, including decay, restoration, and starvation detection.

#### Scenario: Initialize satiety manager
**GIVEN** a SatietyConfig with initial_satiety=1.0, decay_rate=0.01, starvation_threshold=0.0
**WHEN** a SatietyManager is created with this config
**THEN** the manager SHALL initialize with current_satiety=1.0
**AND** SHALL store the configuration parameters

#### Scenario: Decay satiety over time
**GIVEN** a SatietyManager at satiety level 0.8 with decay_rate=0.01
**WHEN** decay_satiety() is called
**THEN** the current satiety SHALL decrease by 0.01 to 0.79
**AND** the method SHALL return the new satiety value

#### Scenario: Satiety cannot decay below zero
**GIVEN** a SatietyManager at satiety level 0.005 with decay_rate=0.01
**WHEN** decay_satiety() is called
**THEN** the current satiety SHALL be clamped to 0.0
**AND** SHALL NOT become negative

#### Scenario: Restore satiety to full
**GIVEN** a SatietyManager at satiety level 0.3
**WHEN** restore_satiety() is called with amount=1.0
**THEN** the current satiety SHALL be set to 1.0
**AND** the method SHALL return the new satiety value

#### Scenario: Detect starvation
**GIVEN** a SatietyManager with starvation_threshold=0.0
**WHEN** current satiety reaches 0.0
**THEN** is_starved() SHALL return True

#### Scenario: Not starved above threshold
**GIVEN** a SatietyManager with starvation_threshold=0.0
**WHEN** current satiety is 0.01
**THEN** is_starved() SHALL return False

### Requirement: Satiety manager provides read-only access to current satiety
The SatietyManager SHALL provide a property to read the current satiety level without allowing direct modification.

#### Scenario: Read current satiety
**GIVEN** a SatietyManager with current satiety at 0.75
**WHEN** the current_satiety property is accessed
**THEN** it SHALL return 0.75
**AND** SHALL NOT allow direct assignment to the property

### Requirement: Satiety manager supports custom restoration amounts
The SatietyManager SHALL allow restoring satiety by arbitrary amounts, not just to full satiety.

#### Scenario: Partial satiety restoration
**GIVEN** a SatietyManager at satiety level 0.5
**WHEN** restore_satiety(amount=0.3) is called
**THEN** the current satiety SHALL increase to 0.8
**AND** SHALL be clamped at 1.0 if the sum exceeds the maximum
