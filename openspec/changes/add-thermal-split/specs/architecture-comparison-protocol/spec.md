## ADDED Requirements

### Requirement: A difficulty pin for a follow-up is chosen by the gates alone, under a rule fixed before the pilot that decides it

When a follow-up panel must move a difficulty setting to make its gates readable, the setting SHALL be
chosen from a pilot on seeds disjoint from the registered band, by the registered gates alone, under a
selection rule written down before the pilot that decides it runs. The pilot's gate evaluation for every
candidate and the rule SHALL be committed with the registration.

#### Scenario: A pilot shows the arms' plateaus

- **GIVEN** a pilot whose output shows each arm's plateau at several settings
- **WHEN** the setting for the registered panel is chosen
- **THEN** it SHALL be the one the pre-written rule selects from the gate statuses
- **AND** no wiring gap SHALL inform the choice

#### Scenario: A further candidate is piloted after earlier ones were seen

- **GIVEN** candidates already piloted and seen
- **WHEN** another candidate is piloted
- **THEN** the selection rule SHALL be committed in writing before that pilot runs
