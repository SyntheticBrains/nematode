## MODIFIED Requirements

### Requirement: A measured prior for the chemical weights

The connectome brain configuration SHALL accept `weight_prior` — `random` (default), `measured`,
`measured_signs` or `measured_shuffled` — and `measured_weight_scale`, a positive multiplier
defaulting to 1.0. Under `random` construction SHALL be bit-identical to the brain without this
requirement. Under any other prior the edge loop SHALL take the same draws in the same order from the
same generator as under `random`, and only chemical edges covered by the measured table SHALL change;
every uncovered edge SHALL keep its draw, except where the fan-in placement below reassigns it on a
rewired wiring, and every other parameter SHALL be unchanged.

A covered edge's measured value SHALL be multiplied by the per-post-neuron scale the random draw
uses, by `measured_weight_scale`, and by one constant, the same for every edge and keeping every sign,
chosen so that at the default multiplier the root mean square of the placed values over the wild
type's covered edges equals the random draw's expected root mean square on those same edges.
`measured` SHALL place that value on its edge; `measured_signs` SHALL keep the draw's magnitude and
take the measured sign; `measured_shuffled` SHALL permute the values among the wild type's covered
edges using a generator of its own, derived from the run seed and independent of every generator the
weight draw uses.

Every quantity above that names the wild type SHALL be computed from the table and the connectome
**before** any rewiring is applied, so a rewired brain uses the same wild-type values as the wild-type
brain at that seed. On a rewired wiring each post-synaptic neuron SHALL receive its wild-type incoming
values: those values, taken in order of their own wild-type pre-synaptic index, SHALL be placed on the
neuron's first *k* incoming edges in pre-synaptic-index order, where *k* is its wild-type covered
count. Under `measured` the values placed are the normalised measured values; under
`measured_shuffled` they are the values the permutation assigned to that neuron's wild-type edges.

What the neuron's remaining incoming edges receive, and what `measured_signs` places, depend on the
weight draw:

- Under the default edge-order draw, the remaining edges SHALL keep their draw, and `measured_signs`
  SHALL place each wild-type sign on the magnitude of the receiving edge's own draw.
- Under the per-neuron fan-in draw, the remaining edges SHALL receive the wild type's **uncovered**
  draws for that neuron, in order of their wild-type pre-synaptic index, and `measured_signs` SHALL
  place on the first *k* edges the wild type's covered values, each being the magnitude of the wild
  type's draw at that edge times its measured sign. Every neuron then carries the wild type's exact
  multiset of incoming values under every prior.

`measured_weight_scale` SHALL be refused at any value other than its default unless the prior is
`measured` or `measured_shuffled`, the only priors that read it. A non-`random` prior SHALL be refused
together with atlas-grounded signs, with the dense-mask weight draw, or with count-scaled
initialisation. Every refusal SHALL apply both when the configuration is validated and when the brain
is constructed. The run's training state SHALL record the prior and the multiplier.

#### Scenario: The default is bit-identical

- **WHEN** a connectome brain is built with the default `weight_prior`
- **THEN** its parameters SHALL be bit-identical to the brain without this requirement at the same seed

#### Scenario: A prior changes covered chemical edges and nothing else

- **GIVEN** two wild-type brains from the same configuration and seed, one `random` and one with
  another prior, under either permitted weight draw
- **WHEN** their parameters are compared
- **THEN** every parameter other than the chemical weights SHALL be identical
- **AND** every chemical edge the table does not cover SHALL be identical

#### Scenario: At the default multiplier the covered edges carry the draw's magnitude

- **WHEN** a brain is built under `measured` at the default multiplier
- **THEN** the root mean square of its covered chemical weights SHALL equal the random draw's
  expected root mean square on those edges
- **AND** each covered weight divided by its per-neuron scale SHALL be its fitted value times one
  constant common to all of them

#### Scenario: The shared generator is left where it was

- **WHEN** brains are built under each prior and each permitted weight draw at one seed
- **THEN** the generator the rollout buffer shares SHALL yield the same next values after
  construction under every combination

#### Scenario: The rewired null receives each neuron's wild-type values

- **GIVEN** a rewired wiring under `measured`, `measured_shuffled` or `measured_signs`, with the
  edge-order draw
- **WHEN** a post-synaptic neuron's incoming chemical weights are read
- **THEN** its first *k* incoming edges in pre-synaptic-index order SHALL carry the values (or, under
  `measured_signs`, the signs) its wild-type edges carried under the same prior at the same seed, in
  the order of their wild-type pre-synaptic index
- **AND** its remaining incoming edges SHALL be identical to the `random` build

#### Scenario: Under the fan-in draw every neuron keeps its wild-type multiset

- **GIVEN** a wild-type and a rewired brain at one seed under the per-neuron fan-in draw and any
  measured prior
- **WHEN** a post-synaptic neuron's incoming chemical weights are read on each wiring
- **THEN** the rewired neuron's first *k* incoming edges SHALL carry the wild type's covered values,
  and its remaining edges the wild type's uncovered values, each in wild-type pre-synaptic order
- **AND** the neuron's multiset of incoming values SHALL be identical on the two wirings

#### Scenario: The shuffle does not depend on the draw

- **WHEN** a brain is built under `measured_shuffled` at one seed under each permitted weight draw
- **THEN** the permutation of covered values SHALL be identical under every draw

#### Scenario: A multiplier no prior reads is refused

- **WHEN** `measured_weight_scale` is set away from its default under `random` or `measured_signs`
- **THEN** validation SHALL raise, and construction from a configuration that skipped validation
  SHALL raise

#### Scenario: An untested pairing is refused

- **WHEN** a non-`random` prior is combined with atlas-grounded signs, the dense-mask weight draw, or
  count-scaled initialisation
- **THEN** validation SHALL raise, and construction from a configuration that skipped validation
  SHALL raise
