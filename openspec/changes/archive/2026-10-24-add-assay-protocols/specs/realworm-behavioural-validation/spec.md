## ADDED Requirements

### Requirement: Capture records the assay

Under an assay, each captured step SHALL record its phase. The capture file SHALL carry a top-level
`assay`: the protocol's name, each phase's name, start step and duration, the assay phase, and the
analysis window in steps. On a bare-plate phase each step SHALL still record satiety, and `on_lawn: false`. Without an assay the capture SHALL be byte-identical to before.

#### Scenario: A food-removal capture is labelled

- **WHEN** the López-Cruz protocol runs with capture on at a 1-second step
- **THEN** steps 0–2699 SHALL carry `phase: fed` and steps from 2700 `phase: assay`
- **AND** the file's `assay` SHALL give the assay phase's start step as 2700
