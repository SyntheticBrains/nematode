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
- **THEN** the update at each grounded inhibitory synapse SHALL equal the negation of the update
  the rule would have applied there with the selector off
- **AND** the update at grounded excitatory and ungrounded synapses SHALL be unchanged

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
