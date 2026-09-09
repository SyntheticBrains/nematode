## MODIFIED Requirements

### Requirement: Plastic-topology seam

The project SHALL define a `PlasticTopology` Protocol carrying exactly what a local plasticity rule touches, so the same rule can drive different substrates without naming any of them: an ordered list of plastic weight tensors, an aligned list of eligibility traces of the same shapes, an aligned list of boolean edge masks, an aligned list of fan-in axes (`plastic_fan_in_axes`, for each tensor the axis to reduce over to obtain one unit's incoming weights), an aligned list of post-synaptic activities (`plastic_post_activities`, for each tensor the activity vector of its post-synaptic units from the step the trace was last accumulated on, indexed along the axis complementary to the fan-in axis), and a flag stating whether traces are enabled. Masking is expressed through the aligned masks: the rule multiplies each tensor's update by its own mask, which on a 0/1 mask is bitwise-identical to the connectome's projector and needs no projector member on the seam.

The seam SHALL be a list from the outset, so that a substrate with one plastic tensor and a substrate with one per layer are handled by the same code path. A dense substrate SHALL expose an all-true mask rather than omitting one, so mask-dependent telemetry has the same meaning on every substrate.

The connectome topology SHALL expose the seam as views over its existing chemical-weight, edge-mask, trace and activity tensors, adding no state and leaving its trace update unchanged. Its chemical matrix is indexed `[pre, post]`, so its fan-in axis SHALL be `0` and its post-synaptic activity is indexed along axis `1`; a `Linear` weight is `[out, in]`, so the MLP topology's fan-in axis SHALL be `1` for every layer and its post-synaptic activity is indexed along axis `0`. The post-synaptic activity SHALL be the same vector the trace's post-synaptic factor was built from, so that a term reading it and the Hebbian term agree on what the unit did.

#### Scenario: Both substrates satisfy the seam

- **WHEN** the connectome topology and the MLP topology are inspected
- **THEN** each SHALL satisfy the `PlasticTopology` Protocol at runtime
- **AND** each SHALL expose aligned lists whose traces and masks match their weights' shapes entry for entry
- **AND** each SHALL expose one fan-in axis per plastic tensor: `0` for the chemical matrix, `1` for each MLP layer
- **AND** each SHALL expose one post-synaptic activity vector per plastic tensor whose length equals the weight's extent along the axis complementary to its fan-in axis

#### Scenario: The connectome seam is a view, not a copy

- **WHEN** the connectome's seam is read
- **THEN** its plastic weight SHALL be the same tensor object as `w_chem`
- **AND** its trace SHALL be the same tensor object as the topology's eligibility buffer

#### Scenario: A dense substrate exposes a full mask

- **WHEN** the MLP topology's masks are read
- **THEN** every mask SHALL be all-true with the shape of its weight

#### Scenario: The post-synaptic activity is the trace's own

- **WHEN** a trace has just been accumulated from pre-synaptic activity `x` and post-synaptic activity `y`
- **THEN** the seam's post-synaptic activity for that tensor SHALL equal `y`

## ADDED Requirements

### Requirement: Decorrelating terms for the three-factor update

The three-factor rule SHALL offer a **decorrelating term** selected by a single configuration
value shared by every plastic brain through the plasticity configuration mixin, one of `none`,
`anti_hebbian_inhibitory` or `oja`, defaulting to `none`. With `none` selected the rule SHALL be
bit-identical to the rule without this requirement and SHALL add no operation to the update.

With `anti_hebbian_inhibitory` selected, the Hebbian term SHALL be negated for every synapse whose
grounded sign is inhibitory, and left unchanged for grounded excitatory synapses and for synapses
the substrate grounds no sign for. The term's magnitude SHALL be unchanged, so the variant
redirects the update rather than resizing it. Selecting it SHALL be rejected at load on a
substrate whose signs are not grounded, since the negation would otherwise key on an arbitrary
draw.

With `oja` selected, the update SHALL gain a term `− η · γ · y² · w` inside the same masked update
as the weight decay, where `y` is the post-synaptic activity the eligibility trace was built from,
broadcast along each weight's post-synaptic axis, and `γ` is a configured coefficient. A
coefficient of zero SHALL be rejected with the mechanism selected, since it names a term that
would not act.

The decorrelating term SHALL compose with every existing term: the Hebbian term, the weight decay,
consolidation, the mask, Dale's-law projection, the homeostatic rescale and the magnitude clamp
SHALL keep their order, with the clamp last.

#### Scenario: The default path is unchanged

- **WHEN** the decorrelation selector is `none`
- **THEN** the weight trajectory SHALL be bit-identical to the rule without decorrelation

#### Scenario: Inhibitory synapses learn with the opposite sign

- **GIVEN** `anti_hebbian_inhibitory` selected on a substrate with grounded signs
- **WHEN** a step is applied
- **THEN** the **Hebbian term** at each grounded inhibitory synapse SHALL equal the negation of the
  Hebbian term the rule would have applied there with the selector off, leaving the weight decay,
  the Oja term and any consolidation term as they were
- **AND** the Hebbian term at grounded excitatory and ungrounded synapses SHALL be unchanged

#### Scenario: The variant redirects the update without resizing it

- **WHEN** `anti_hebbian_inhibitory` is selected
- **THEN** the total absolute magnitude of the Hebbian term over the edge set SHALL equal the
  magnitude the same step would have produced with the selector off

#### Scenario: Flipping an arbitrary sign is refused

- **GIVEN** a configuration selecting `anti_hebbian_inhibitory` on a substrate whose synapse signs
  are not grounded in the atlas
- **WHEN** the configuration is loaded
- **THEN** loading SHALL fail with a message saying the variant keys on grounded signs

#### Scenario: The Oja term opposes growth in proportion to post-synaptic activity

- **GIVEN** `oja` selected with a positive coefficient
- **WHEN** a step is applied with a post-synaptic unit active and its incoming weight non-zero
- **THEN** that weight's update SHALL include a term proportional to the negative of the weight
  times the square of that unit's activity
- **AND** a unit with zero activity SHALL receive no such term

#### Scenario: A named term with no coefficient is rejected

- **GIVEN** a configuration selecting `oja` with a coefficient of zero
- **WHEN** the configuration is loaded
- **THEN** loading SHALL fail with a message naming the coefficient

### Requirement: Decorrelation telemetry

The rule SHALL report the share of the update's total absolute magnitude carried by the
decorrelating term — the negated subset's share under `anti_hebbian_inhibitory`, the Oja term's
share under `oja`, and zero under `none` — beside its existing plasticity telemetry, recorded by
the shared plasticity report. An arm that improved by decorrelating and an arm that improved by
updating less are different results, and this quantity separates them.

#### Scenario: A recovery attributable to something other than the term is visible

- **WHEN** a screen reads a run's telemetry
- **THEN** the decorrelation share SHALL be available per step
- **AND** a run whose share is near zero SHALL be distinguishable from one whose share is large
