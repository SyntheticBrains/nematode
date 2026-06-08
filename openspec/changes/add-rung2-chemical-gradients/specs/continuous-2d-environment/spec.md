## MODIFIED Requirements

### Requirement: Source placement and Euclidean fields

The continuous-2D environment SHALL place food and predator sources at **real-valued (float) coordinates** within the world bounds and SHALL compute food capture and nearest-food distance using **true Euclidean distance** (not Manhattan, and **not** rounded to integer). The float source coordinates are confined to the continuous-2D environment; the discrete grid environment retains integer source coordinates and remains byte-stable.

#### Scenario: Sources within bounds

- **WHEN** sources are initialised in the continuous-2D environment
- **THEN** their coordinates are real-valued, lie within the world bounds, and respect the existing minimum-separation validity checks

#### Scenario: Euclidean distances

- **WHEN** food capture or nearest-food distance is computed in the continuous-2D environment
- **THEN** it uses real-valued Euclidean distance (replacing the grid's Manhattan distance and without integer rounding of the result)

#### Scenario: Grid environment unchanged

- **WHEN** the discrete grid environment places or measures sources
- **THEN** it retains integer source coordinates and Manhattan distance, byte-stable against the pre-change behaviour

### Requirement: Heading-aware klinotaxis sensing on the continuous substrate

Klinotaxis (head-sweep) sensing in the continuous-2D environment SHALL sample lateral concentrations perpendicular to the agent's continuous heading (`heading_rad`), scaled by the configured sweep amplitude (honoured as configured, including sub-cell amplitudes), rather than at the grid's fixed cardinal ±1-cell offsets, so the directional gradient rotates with the continuous heading, and SHALL feed STAM the resulting readings. Lateral samples SHALL be evaluated against the **continuous concentration field at real-valued sample positions** (no integer-cell snapping of the sample points).

#### Scenario: Heading-aware lateral sampling

- **WHEN** klinotaxis sensing runs in the continuous-2D environment
- **THEN** left/right samples are taken perpendicular to `heading_rad` (matching the grid offsets at cardinal headings, rotating smoothly otherwise) at real-valued sample positions, and `dC` is the right-minus-left difference

#### Scenario: Continuous-field sampling

- **WHEN** a lateral sample position is evaluated
- **THEN** the concentration is read from the continuous field at the real-valued position, without snapping the sample point to an integer cell

#### Scenario: STAM receives the readings

- **WHEN** lateral readings are produced
- **THEN** STAM records them and computes the dC/dt temporal derivative from its history
