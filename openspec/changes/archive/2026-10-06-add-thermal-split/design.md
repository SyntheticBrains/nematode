## Overview

The split's question, at a readable thermal point. The decisions below are where this differs from
A.6t.

## Decisions

### Decision A: The point is chosen by the gates, under a rule fixed in advance

The pilot showed each arm's plateau, so choosing a target by eye could have been choosing by the wiring
gap. After targets 25, 30 and 40, the rule was written down before target 35 ran: the lowest readable
target, which keeps the panel closest to block V's point. It picked 35.

### Decision B: The minimum is scaled, and says so

No committed data measures block V's effect at target 35. The minimum takes A.1's effect at target 20 and
scales it by the wild type's own `auc_success` ratio between the targets, a quantity that carries no
wiring information. A minimum from this campaign's own data is forbidden, and an unscaled one would judge
a 0.31-auc regime against a 0.72-auc effect.

### Decision C: The split's arms only, and two readings

The chemical-only null is left out: the split separates the gap junctions from the autapses on an exact
pairing, and the hard350 figure the combined paper cites is the lead against the gap-held null. So both
the split and that lead are registered readings, corrected together.

## Risks

- **The spread at target 35 is unknown.** Sized from A.6t's achieved spread at target 20, the readings are
  resolvable at the minimum if the spread scales with the wild type's `auc_success`, and 1.4–1.7× short if
  it does not shrink. Stated in the launch record; the achieved spread is reported beside.
- **Target 35 is a different operating point from block V's.** The panel says what holds at 35, and block
  V's target-20 claim keeps its 077 condition; neither is read as the other.
