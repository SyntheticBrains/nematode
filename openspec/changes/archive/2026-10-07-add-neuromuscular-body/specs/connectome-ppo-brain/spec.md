## ADDED Requirements

### Requirement: The anatomical neuromuscular readout

Under `action_space: body_drive` the connectome brain's policy mean SHALL be its settled rates through
the fixed signed neuromuscular map: every cell with a neuromuscular junction, each entry its EM count
signed +1 for acetylcholine, −1 for GABA and 0 otherwise, pooled to quadrant and segment and normalised
per column. The direction channel SHALL be the forward-minus-backward motor-class contrast. The readout
SHALL have no learnable parameters.

#### Scenario: Signs follow the transmitter

- **WHEN** the map is built
- **THEN** every acetylcholine cell's entries SHALL be non-negative, every GABA cell's non-positive, and
  every other cell's zero

#### Scenario: The readout is not learned

- **WHEN** a `body_drive` connectome brain lists its learnable parameters
- **THEN** none of them SHALL be a motor readout
