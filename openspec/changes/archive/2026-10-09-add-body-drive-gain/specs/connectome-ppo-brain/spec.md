## MODIFIED Requirements

### Requirement: The anatomical neuromuscular readout

Under `action_space: body_drive` the connectome brain's policy mean SHALL be its settled rates through
the fixed signed neuromuscular map: every cell with a neuromuscular junction, each entry its EM count
signed +1 for acetylcholine, −1 for GABA and 0 otherwise, pooled to quadrant and segment and normalised
per column. The direction channel SHALL be the forward-minus-backward motor-class contrast. The map
SHALL have no learnable parameters. Each of the 25 drive outputs SHALL be scaled by a learnable gain,
starting at 1, the same 25 gains for every wiring.

#### Scenario: Signs follow the transmitter

- **WHEN** the map is built
- **THEN** every acetylcholine cell's entries SHALL be non-negative, every GABA cell's non-positive, and
  every other cell's zero

#### Scenario: The readout is not learned

- **WHEN** a `body_drive` connectome brain lists its learnable parameters
- **THEN** none of them SHALL be a motor readout

#### Scenario: A gain vector scales the anatomy

- **WHEN** a `body_drive` connectome brain is built
- **THEN** its learnable parameters SHALL include 25 drive gains, each starting at 1
- **AND** scaling a gain SHALL scale that output of the anatomical drive by the same factor

#### Scenario: The trainable count does not depend on the wiring

- **WHEN** the wild type and a rewired null are built under the body drive
- **THEN** they SHALL have the same number of learnable parameters
