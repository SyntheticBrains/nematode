## ADDED Requirements

### Requirement: The eligibility may carry the perturbation instead of the activity

The three-factor rule SHALL offer an **eligibility mode** selected by a configuration value shared
through the plasticity configuration mixin, one of `hebbian` or `node_perturbation`, defaulting to
`hebbian`. With `hebbian` selected the rule SHALL be bit-identical to the rule without this
requirement.

With `node_perturbation` selected, the eligibility SHALL be accumulated as pre-synaptic activity
times the post-synaptic unit's **perturbation** rather than its activity, so that the update
correlates the reward surprise with the perturbation that produced it. Every other term SHALL keep
its order and meaning: the modulator, both scaling switches, the weight decay, the mask, any
decorrelating or consolidation term, Dale's-law projection, third-factor routing, the homeostatic
rescale, and the magnitude clamp last.

Selecting `node_perturbation` SHALL be rejected at load where the perturbation scale is zero, since the eligibility would be identically zero and the arm would appear to be a rule that learns nothing rather than one that was given nothing to learn from; that refusal SHALL be repeated at brain construction, since a copied configuration skips validators. The rule itself SHALL refuse construction over a topology that exposes no perturbation, as it refuses a sign-keyed variant without signs and a routed one without a pathway.

#### Scenario: The default path is unchanged

- **WHEN** the eligibility mode is `hebbian`
- **THEN** the weight trajectory SHALL be bit-identical to the rule without this requirement

#### Scenario: The trace carries the perturbation

- **GIVEN** `node_perturbation` selected on a topology that perturbs its units
- **WHEN** a trace-accumulating forward pass and a step are applied
- **THEN** the eligibility SHALL equal pre-synaptic activity times the exposed perturbation, and
  SHALL NOT equal pre-synaptic activity times post-synaptic activity

#### Scenario: A mode with nothing to learn from is refused

- **GIVEN** a configuration selecting `node_perturbation` with a perturbation scale of zero
- **WHEN** the configuration is loaded, or a brain is built from a copy of it
- **THEN** it SHALL fail with a message naming the zero perturbation

#### Scenario: The rule refuses a topology that cannot perturb

- **GIVEN** `node_perturbation` selected and a topology exposing no perturbation
- **WHEN** the rule is constructed over it
- **THEN** construction SHALL fail with a message naming the missing seam member

### Requirement: A new eligibility clears the rule's positive control before any substrate arm

A variant of the three-factor rule that changes what its eligibility carries SHALL be run through
the rule's positive control, at that control's registered pass rule, **before** any arm of it is
built on a connectome substrate. The variant's **gradient alignment** SHALL be reported from the
same runs beside the pass or fail.

A variant that does not pass SHALL NOT have a connectome arm built for it, and the record SHALL
state that it is not an instrument. A variant that passes with an alignment near zero SHALL be
recorded as having learned by some route other than the one it was built for, rather than as
confirming the estimator it claims to be.

#### Scenario: A failing variant gets no substrate arm

- **GIVEN** an eligibility variant that does not pass the positive control
- **WHEN** the result is recorded
- **THEN** no connectome arm SHALL be built for it
- **AND** the record SHALL state that it is not an instrument

#### Scenario: Passing without alignment is not a vindication

- **GIVEN** a variant that passes the control with a gradient alignment near zero
- **WHEN** the result is written up
- **THEN** the record SHALL state that it learned by some route other than the one it was built for
