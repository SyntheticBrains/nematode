## ADDED Requirements

### Requirement: Consolidation mechanisms for the three-factor update

The three-factor rule SHALL offer a **consolidation mechanism** selected by a single
configuration value shared by every plastic brain through the plasticity configuration mixin,
one of `none`, `anchor`, `rigidity` or `oracle`, defaulting to `none`. With `none` selected the
rule SHALL be bit-identical to the rule without this requirement, SHALL allocate no consolidation
state, and SHALL add no operation to the update. A selector other than `none` whose own
parameters are all zero SHALL be rejected at load, since it names a mechanism that would not act.

With `anchor` selected, each plastic tensor SHALL carry an anchor of its own shape, initialised
to the weights the rule was constructed over. The update SHALL gain a restoring term
`− η · κ_a · (w − a)` applied inside the same masked update as the weight decay, and after the
write the anchor SHALL advance as `a ← a + ρ_a · (w − a)` from the weights as they then are. An
anchor rate of zero SHALL hold the anchor fixed at the weights the rule started from.

With `rigidity` selected, each plastic tensor SHALL carry a non-negative protective variable of
its own shape, zero at construction. The Hebbian term's rate SHALL be divided by `1 + κ_c · c`
using the protective variable's value from **before** this step's growth, and after the update the
protective variable SHALL advance as `c ← (1 − λ_c) · c + γ_c · max(m, 0) · |E|`, where `m` is the
effective modulator and `E` the eligibility trace. Under the unmodulated mode `m` is `1.0` and the
growth term SHALL follow the trace alone, so the two arms differ in the modulator and nowhere
else.

With `oracle` selected, the rule SHALL maintain a trailing episode-success rate as an exponential
moving average of an episode-success flag supplied by the brain at the end of each episode, and
SHALL scale the plasticity rate by `clamp(1 − s / s_ref, 0, 1)` for a configured reference rate
`s_ref`. This mechanism consumes a quantity that is a property of the task's scoring rather than
of the reward stream the rule observes; its implementation SHALL say so, and it SHALL NOT be
offered as a default or presented as a biologically plausible mechanism.

Consolidation SHALL compose with, and never replace, the existing terms: the Hebbian term, the
weight decay, the mask, Dale's-law sign projection, the homeostatic rescale and the magnitude
clamp SHALL keep their order, with the clamp last, and consolidation state SHALL be updated after
the write.

#### Scenario: The default path is unchanged

- **WHEN** the consolidation selector is `none`
- **THEN** the weight trajectory SHALL be bit-identical to the rule without consolidation
- **AND** no anchor and no protective variable SHALL be allocated

#### Scenario: A named mechanism with no effect is rejected

- **GIVEN** a configuration selecting `anchor` with a stiffness of zero, or `rigidity` with a
  growth of zero
- **WHEN** the configuration is loaded
- **THEN** loading SHALL fail with a message naming the parameter that leaves the mechanism inert

#### Scenario: The anchor opposes departure and follows slowly

- **GIVEN** `anchor` selected with a positive stiffness and a positive anchor rate
- **WHEN** a weight has moved away from its anchor and a step is applied
- **THEN** the update SHALL include a term proportional to the negative departure
- **AND** the anchor SHALL then move toward the weight by the anchor rate

#### Scenario: A fixed anchor holds the starting policy

- **GIVEN** `anchor` selected with an anchor rate of zero
- **WHEN** any number of steps are applied
- **THEN** the anchor SHALL remain the weights the rule was constructed over

#### Scenario: Rigidity grows where reward gated a large trace and slows later writes

- **GIVEN** `rigidity` selected
- **WHEN** steps with a positive modulator and a large trace are applied to the same synapses
- **THEN** those synapses' protective variable SHALL increase
- **AND** a subsequent step's Hebbian term at those synapses SHALL be smaller than the same step
  applied with the mechanism off
- **AND** the divisor SHALL use the pre-growth value, so a step is not charged for the rigidity it
  creates

#### Scenario: Rigidity decays so nothing is frozen permanently

- **GIVEN** a synapse whose protective variable is positive
- **WHEN** steps are applied that neither reinforce it nor drive its trace
- **THEN** its protective variable SHALL decrease toward zero at the configured decay

#### Scenario: The oracle stops updating at its reference success rate

- **GIVEN** `oracle` selected with a reference rate `s_ref`
- **WHEN** the trailing episode-success rate reaches or exceeds `s_ref`
- **THEN** the effective plasticity rate SHALL be zero and no weight SHALL change
- **AND** when the trailing rate is far below `s_ref` the rate SHALL be the configured rate

#### Scenario: Consolidation is applied before the bound and after the sign

- **WHEN** a consolidated update is applied on a substrate with grounded signs and homeostasis on
- **THEN** the sign projection, the homeostatic rescale and the magnitude clamp SHALL apply in
  their existing order, with the clamp last

### Requirement: Consolidation state follows a loaded policy

When the rule's running state is reset — the path taken when weights are loaded into a brain —
the elastic anchor SHALL be re-anchored to the weights the rule now starts from and the protective
variable SHALL be returned to zero, alongside the existing baseline, scale, centre, trace and
homeostatic-target resets. An anchor left at the values of a previous substrate would pull a
loaded policy toward weights it no longer has, with a force proportional to the distance between
them.

Consolidation state SHALL NOT be persisted with weights: a checkpoint carries the policy, and a
rule constructed over a loaded policy SHALL anchor to that policy.

#### Scenario: A loaded clone is anchored to itself

- **GIVEN** a rule with `anchor` selected, constructed over a random initialisation
- **WHEN** a cloned competent policy is loaded and the rule's state is reset
- **THEN** the anchor SHALL equal the loaded weights
- **AND** the first update's restoring term SHALL be zero

#### Scenario: The protective variable does not survive a load

- **GIVEN** a rule with `rigidity` selected whose protective variable has grown
- **WHEN** weights are loaded and the rule's state is reset
- **THEN** the protective variable SHALL be zero

### Requirement: Consolidation telemetry

The rule SHALL report, beside its existing plasticity telemetry, the effective rate multiplier it
actually applied, the mean absolute departure of the weights from their anchor over the edge set,
and the mean protective variable over the edge set. Under `none` the multiplier SHALL be `1` and
the two means SHALL be reported as not-a-number, in the same way the scaling telemetry reports an
inactive estimator. Every key SHALL be recorded by the shared plasticity report so the arms of a
screen remain comparable step for step.

#### Scenario: A variant that held a policy by not moving is distinguishable

- **WHEN** a screen reads a run's telemetry
- **THEN** the effective rate multiplier and the anchor departure SHALL be available per step
- **AND** they SHALL make a run that consolidated distinguishable from one whose updates were
  suppressed outright
