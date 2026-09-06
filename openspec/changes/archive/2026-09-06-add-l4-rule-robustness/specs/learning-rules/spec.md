## ADDED Requirements

### Requirement: Homeostatic incoming-norm scaling

The three-factor rule SHALL offer a default-off homeostatic mode, shared by every plastic brain
through the plasticity configuration mixin. When on, the rule SHALL capture at construction, for
every unit of every plastic tensor, the norm of that unit's incoming plastic weights over the
tensor's masked entries, and after each plastic update SHALL rescale each unit's incoming weights over the masked
entries only — off-edge entries SHALL never be written — so that their norm returns to that
target, dividing by the unit's current norm; a unit with a zero target, or whose incoming weights are all zero (no direction to scale along), SHALL be left untouched; the magnitude clamp
SHALL be applied after the rescale. Nothing SHALL be rescaled under a freeze. The rule SHALL obtain each
tensor's fan-in axis from the plastic-topology seam and SHALL NOT name any substrate's layout.
The decay term SHALL remain in place; under homeostasis its uniform shrink is undone by the
rescale. The rule SHALL report the mean relative norm deviation
before the rescale (`plasticity_norm_drift`), NaN when the mode is off. With the mode off, the
rule SHALL be bit-identical to the rule without this requirement.

#### Scenario: Incoming norms return to their targets on both substrates

- **GIVEN** homeostasis on and a rule stepped over the connectome and over the MLP with
  non-trivial traces and modulators
- **WHEN** an update has been applied
- **THEN** every unit with a non-zero target SHALL have incoming norm equal to its target within
  floating-point tolerance, on the connectome over the chemical mask and on the MLP over each
  layer's rows
- **AND** under the connectome's soft-prior mask the off-edge entries SHALL be unchanged by the
  rescale

#### Scenario: Units without inputs and the bound are respected

- **GIVEN** homeostasis on
- **WHEN** a unit has no incoming edges
- **THEN** its (zero) incoming weights SHALL be unchanged
- **AND** no weight SHALL exceed the magnitude bound after the rescale
- **AND** a unit whose norm is positive but tiny SHALL be restored to its target by its actual norm
- **AND** a unit whose incoming weights are all zero SHALL be left at zero

#### Scenario: Homeostasis off is bit-identical and a freeze rescales nothing

- **GIVEN** the default configuration
- **WHEN** the rule steps in any mode
- **THEN** the weights and the existing telemetry SHALL be bit-identical to the frozen reference
- **AND** with homeostasis on under a freeze, no weight SHALL change while the norm drift is still
  reported

#### Scenario: The connectome's targets follow its initialisation scale

- **GIVEN** the connectome with homeostasis on
- **WHEN** the rule captures its targets
- **THEN** every neuron with chemical inputs SHALL have a target near 1, the incoming norm its
  `1/√k` initialisation gives by construction

## MODIFIED Requirements

### Requirement: Plastic-topology seam

The project SHALL define a `PlasticTopology` Protocol carrying exactly what a local plasticity rule touches, so the same rule can drive different substrates without naming any of them: an ordered list of plastic weight tensors, an aligned list of eligibility traces of the same shapes, an aligned list of boolean edge masks, an aligned list of fan-in axes (`plastic_fan_in_axes`, for each tensor the axis to reduce over to obtain one unit's incoming weights), and a flag stating whether traces are enabled. Masking is expressed through the aligned masks: the rule multiplies each tensor's update by its own mask, which on a 0/1 mask is bitwise-identical to the connectome's projector and needs no projector member on the seam.

The seam SHALL be a list from the outset, so that a substrate with one plastic tensor and a substrate with one per layer are handled by the same code path. A dense substrate SHALL expose an all-true mask rather than omitting one, so mask-dependent telemetry has the same meaning on every substrate.

The connectome topology SHALL expose the seam as views over its existing chemical-weight, edge-mask, and trace tensors, adding no state and leaving its trace update unchanged. Its chemical matrix is indexed `[pre, post]`, so its fan-in axis SHALL be `0`; a `Linear` weight is `[out, in]`, so the MLP topology's fan-in axis SHALL be `1` for every layer.

#### Scenario: Both substrates satisfy the seam

- **WHEN** the connectome topology and the MLP topology are inspected
- **THEN** each SHALL satisfy the `PlasticTopology` Protocol at runtime
- **AND** each SHALL expose aligned lists whose traces and masks match their weights' shapes entry for entry
- **AND** each SHALL expose one fan-in axis per plastic tensor: `0` for the chemical matrix, `1` for each MLP layer

#### Scenario: The connectome seam is a view, not a copy

- **WHEN** the connectome's seam is read
- **THEN** its plastic weight SHALL be the same tensor object as `w_chem`
- **AND** its trace SHALL be the same tensor object as the topology's eligibility buffer

#### Scenario: A dense substrate exposes a full mask

- **WHEN** the MLP topology's masks are read
- **THEN** every mask SHALL be all-true with the shape of its weight
