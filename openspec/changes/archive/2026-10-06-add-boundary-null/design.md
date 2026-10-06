## Overview

Two readings that share a mechanism: the registered hop predictor asks whether a null's shortcuts
predict, seed by seed, how much the wild type leads it; the boundary-null panel removes the shortcuts
and asks what lead remains.

## Decisions

### Decision A: The boundary is the brain's own injection and readout sets

The sensory side is every neuron any projection injects into (food ASE/AWC/AWA, thermal AFD, predator
ASH/ASI/ALM/AVM/PLM: 17), so the null is the same on every cell; the motor side is the 39 neurons of the
four classes the readout pools. That matches the mechanism A.2 measured — the injected signal's route
to the readout — rather than the anatomical sensory and motor classes, most of which no signal enters
here. Holding every edge out of the sensory set and into the motor set keeps every chemical route of one
or two hops between them; with the gap junctions held as in the chemical-only null, the propagating
graph's one-hop count and reach within two hops equal the wild type's (0 and 26).

### Decision B: Built on the chemical-only null, read against it

D21 makes the chemical-only null 8b's primary. The boundary null is that null with the boundary held
too, so the panel's interaction, gap(boundary) − gap(chemical), isolates the boundary. Its base is A.6's
committed lead over the chemical-only null, +0.0215 `auc_success`; the minimum is 2/3 of it, 0.0143,
since A.6's minimum (0.041) exceeds the whole base.

### Decision C: 128 fresh seeds

A.6's committed spread puts the detectable effect at 0.027 at 32 seeds and 0.014 at 128. The maintainer
chose 128 fresh seeds over reusing A.6's 32, for power and one seed band.

### Decision D: The predictor is registered on committed data

The one-hop count varies from 4 to 14 across the 48 committed seeds (sd 2.4), so it can carry a
correlation; reach within two hops saturates (37–39 of 39) and cannot. The predictor's direction,
minimum and test are fixed before it is computed. At n = 48 a true rho of −0.3 is detected about 55% of
the time, which the registration states.

## Risks

- **The boundary null may learn differently for reasons beyond shortcuts.** Holding 544 edges leaves
  3,165 interior edges to rewire; the panel says what holding the boundary does to the lead, and the
  predictor separately says whether shortcuts track the gap.
- **The predictor's two sources differ in seeds and campaigns.** Both are hard350 PPO at block V's point
  under the current null; pooling them is stated in the registration.
