# Reading H0: does the true wiring beat degree-preserving rewirings on measured responses?

Registered 2026-11-14, before anything below is computed. This is the independent check's proposed
test (INDEPENDENT_CHECK.md, point 7), adopted by the maintainer. It decides the final
recommendation: the outcomes are mapped to recommendations below. The reading is made once.

## What it asks

The unfitted network on the true connectome correlates with measured responses to single-neuron
stimulation above a relabelling null (MA_C2_RESULT.md). Is that the specific wiring, or only the
degree statistics that a degree-preserving rewiring keeps?

## The comparison

- **The networks:** C.2's encoded brain, built exactly as in the M-a reading (MA_C2_READING.md and
  its amendment 1). It uses the probe module's `PROCEDURE_SETTINGS`, with the encoder at the box
  centre, `[0.5] × 137`. The box centre carries no fitting, so the comparison is between wirings
  alone.
  - **True wiring:** `wiring: wild_type`.
  - **19 null wirings:** `wiring: rewired_degree_preserving`, with `rewire_seed` 9741 to 9759. This
    is the package's existing null: every neuron's chemical in- and out-degree is preserved, the gap
    junctions are rewired with their counts, and the rest of the brain is built on the rewired
    graph by the package's own path.
  - With 19 draws, the true wiring is the highest of 20 by chance with probability 0.05.
- **The stimulation, the response, the 154 names, the metric** (Creamer's `nan_corr`, diagonal
  masked) and **the measured test matrix** are all exactly as in the M-a reading's primary
  definition, amendment 1 included.

## The checks before any null is read

1. Every null brain builds, and its wiring differs from the true one.
2. **A positive control:** the true wiring at the box centre reproduces the M-a reading's 0.0653
   against the test matrix, to within 10⁻⁴. If it does not, stop and report without reading the
   nulls.

## The minimum effect

m0 = 0.25 × (r_true − 0.0216), where 0.0216 is the 97.5th percentile of the relabelling null
built from the box centre's own matrix in the M-a reading. In words, the specific wiring must carry
at least a quarter of what the true wiring scores above chance.

## The readings, and what each means for the recommendation

- **PASS:** r_true exceeds all 19 null draws, and exceeds their median by at least m0.
  - Recommendation: the slimmed hybrid.
  - H1 is the standing readout, with this degree-preserving null as its floor.
  - H2 runs only with a causal bar: the held-out-neuron fit must also predict the measured
    stimulation responses above a fit on a degree-preserving null.
  - H3 is gated as before.
- **SMALL:** r_true exceeds all 19 draws, but by less than m0 over their median.
  - Recommendation: stay, plus H1 with the degree-preserving floor, as description only.
  - No neural fit.
- **FAIL:** r_true does not exceed all 19 draws.
  - Recommendation: stay, with no neural track.
  - The readout's signal is degree statistics.

## Reported beside the reading, as description

- C.2's fitted answer (`hinge4.json`'s `z`) written onto each null wiring, and its rank among them.
  Its fit was made on the true wiring, so this is only description.
- The same readings against the training matrix.
- The null's median and range.
- Each draw's fraction of edges moved.

## Limits, stated before reading

- These are the M-a reading's limits: a static network, and arrays from the fuller, unlicensed
  record.
- The package's null rewires the chemical and gap graphs separately. It does not preserve the
  overlap of the two graphs, or reciprocity.
