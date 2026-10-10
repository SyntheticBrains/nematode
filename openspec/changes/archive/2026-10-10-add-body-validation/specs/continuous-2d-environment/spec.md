## ADDED Requirements

### Requirement: The posture record's frame angle

Each sub-step's posture record SHALL carry the body's frame angle beside its time, curvature and head
position, so a posture can be placed in the world frame.

#### Scenario: The frame angle places the posture

- **WHEN** a posture is reconstructed from its record
- **THEN** its head-to-midline direction SHALL match the body's heading at that sub-step
