## Overview

A.6t repeats A.6 and its split on block V's thermal cell, as one panel. The decisions below are the ones
that differ from hard350.

## Decisions

### Decision A: One panel, two interactions, one family

On hard350 the split followed A.6 as a second campaign that reused A.6's runs. Here both narrower nulls
run together on fresh seeds, so the wild type and the current null serve three levels, and the two
interactions are corrected as one family. Nothing is reused from A.1's thermal runs, so the panel has
one seed band and no identity check against older runs.

### Decision B: The minimum is the thermal cell's own

Block V's magnitude on this cell has drifted: A.1 found it at 35% of its committed size on the episode
metric and 51% on `auc_success`. The minimum is 2/3 of A.1's observed thermal effect (+0.0829), not
V.4's committed one and not hard350's, and both interactions are read against it. Taking the split's
minimum from this campaign's combined move, as the hard350 split took A.6's, would make the minimum a
fraction of the campaign's own data, which the protocol forbids. This is the lesson the spec delta
records.

### Decision C: 128 seeds, chosen knowing the likely outcome

Thermal's per-seed spread is about four times hard350's. 128 seeds bring the detectable effect to the
minimum under the proxy discount A.6 observed. If thermal behaves as hard350 did, the combined move is
about 0.04, below the minimum, so the registered verdicts will most likely read `below` or
`unresolved`. The panel is still worth its 16 hours for the gaps it resolves against each narrower
null, which the combined paper's 8a half cites; the maintainer chose 128 seeds with this stated.

## Risks

- **Saturation.** If both learning arms of a level reach the 90% bar, that level is unreadable. Block
  V's target-20 remedy was adopted for exactly this on this cell, and A.1 read it at 32 seeds.
- **The proxy.** The sensitivity is taken from A.1's interaction spread, a different quantity. The
  achieved spread is reported beside the registered one and never used to re-read a verdict.
