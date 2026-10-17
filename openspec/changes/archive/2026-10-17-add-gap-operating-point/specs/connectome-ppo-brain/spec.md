## ADDED Requirements

### Requirement: Gap-junction weight options

The brain SHALL offer `gap_weight_mode` (`count`, the default; `unit`; `log_count`) and
`gap_weight_scale` (a positive float, default 1.0). Under `unit` every existing gap pair's weight SHALL
be 1 and under `log_count` it SHALL be `log1p(count)`, each applied before the fan-in normalisation;
`gap_weight_scale` SHALL multiply the normalised gap matrix. Under a rewired wiring the counts SHALL
travel with their edges before the mode is applied. Either option away from its default SHALL be
refused when gap junctions are off. At their defaults every output, parameter and random draw SHALL be
identical to the brain before these options existed.

#### Scenario: Unit weights remove the counts and keep the normalisers

- **WHEN** a brain is built with `gap_weight_mode: unit`
- **THEN** each existing pair's coupling SHALL equal `1/sqrt(d_i d_j)`, `d` the binary gap degree, and
  every other entry SHALL be zero

#### Scenario: The scale multiplies the normalised matrix

- **WHEN** a brain is built with `gap_weight_scale: s`
- **THEN** its gap matrix SHALL equal `s` times the matrix built at scale 1 under the same mode

#### Scenario: Defaults are byte-identical

- **WHEN** both options are at their defaults
- **THEN** the gap matrix, the learnable parameters, every output and the next value of every random
  generator SHALL equal the brain's before these options existed

### Requirement: Gap-outlier wirings

The brain SHALL offer `wiring: rewired_gap_outliers_held` and `wiring: rewired_gap_outliers_scattered`,
with `gap_outlier_min_count` (a positive integer, default 20) naming the outlier pairs (count at or
above it). Both SHALL hold the chemical graph, autapses included, exactly. **Held** SHALL rewire the
gap junctions by the degree-preserving undirected swap over the non-outlier pairs only, every outlier
keeping its pair and count. **Scattered** SHALL draw the first edge of every swap from the outliers and
the second from the non-outliers, so each outlier's count moves to a new pair. Both SHALL preserve
every neuron's gap degree. `gap_outlier_min_count` SHALL be refused away from its default under any
other wiring. The existing wirings SHALL draw the graphs they drew before these wirings existed.

#### Scenario: Held outliers keep their place

- **WHEN** the held wiring is drawn at any seed
- **THEN** every pair with count at or above the threshold SHALL be present with its count, the other
  gap pairs SHALL differ from the wild type's, and every neuron's gap degree SHALL equal the wild type's

#### Scenario: Scattered outliers land elsewhere

- **WHEN** the scattered wiring is drawn at any seed
- **THEN** each outlier count SHALL sit on a pair that differs from its wild-type pair wherever a valid
  swap existed, the number of accepted swaps SHALL be reported, and every neuron's gap degree SHALL
  equal the wild type's

#### Scenario: Existing nulls are unchanged

- **WHEN** any existing wiring is drawn at a pinned seed
- **THEN** its graph SHALL equal the graph drawn before these wirings existed
