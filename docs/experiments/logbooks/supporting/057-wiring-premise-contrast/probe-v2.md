# V.2 — does a graph property of the rewiring predict how slowly that seed learns?

**Probe, registered before the analysis ran. Exploratory: it generates hypotheses and settles
nothing.** Whatever it finds needs its own registered test before it is claimed.

## The question

[Logbook 057](../../057-wiring-premise-contrast.md) found the wild-type wiring reaching competence
~35% sooner than its degree-preserving rewired null, and the effect is **heterogeneous**: 44 of 64
seeds favour the wild type, and the rewired arm's times range from 32 to 2275 episodes against the
wild type's 46 to 1287. Each seed's rewiring is a **different graph** — `rewire_seed` derives from
the run seed — while the wild type is the same graph on every seed with only its weight
initialisation varying. So the rewired arm's spread carries a source the wild type's does not.

**Does a structural property of each seed's rewiring predict how slowly that seed learned?**

There is a concrete reason to expect one. The substrate settles for `forward_pass_depth: 4` steps
per environment step, so a signal needs a directed path of **at most four hops** from its sensory
neuron to a motor-readout neuron to reach the action within one forward pass. A rewiring that pushes
the thermosensory route beyond four hops cannot deliver temperature to the motor output at all.

## What is measured, fixed before looking

The 64 rewired graphs are regenerated offline — `rewire_degree_preserving` is deterministic given
its seed — and each is scored on four properties, correlated (Spearman) against that seed's rewired
`episodes_to_30pct_success` from the committed panel data, **BH-FDR across the four**:

| id | property | hypothesis |
|---|---|---|
| **P1** | shortest directed path, AFDL/AFDR → any motor-readout neuron (chemical edges) | longer ⇒ slower |
| **P2** | motor-readout neurons reachable from AFD **within 4 hops** | fewer ⇒ slower |
| **P3** | the same within 4 hops for the **food** sensors (ASE/AWC/AWA) | **specificity control** — if the effect is thermosensory, P3 should not predict |
| **P4** | characteristic path length of the whole chemical graph | **global control** — carries no pathway information; if it predicts as well as P1/P2, the effect is not about the sensory route |

`AFDL/AFDR` are the thermotaxis projection targets and the motor readout is the VB/DB/VA/DA classes,
both as the brain defines them. The wild type's own values are reported beside the distribution.

## How it will be read

- **P1 or P2 predicts, P3 and P4 do not.** The strongest available outcome: the thermosensory route's
  length is a candidate mechanism, and the registered follow-up is a wiring arm that rewires only
  non-sensory edges.
- **P4 predicts too.** The effect is about global graph structure rather than the sensory pathway, and
  the follow-up is a different one.
- **Nothing predicts.** The heterogeneity is not explained by these four properties. Recorded as such;
  the result in 057 stands on its statistics either way.

**This probe cannot strengthen 057's claim.** It can only propose a mechanism for it, and a
correlation over 64 seeds with four properties tested is a hypothesis, not a finding.

______________________________________________________________________

## Outcome, 2026-09-12 — nothing predicts, and the reason is the finding

`scripts/analysis/wiring_premise_graph_probe.py`, all 64 rewirings regenerated from their seeds.

| property | wild type | rewirings | ρ | q |
|---|---|---|---|---|
| **P1** shortest AFD → motor path | **3 hops** | **1–2 hops** (mean 1.22) | +0.053 | 1.000 |
| **P2** motor neurons reached within 4 hops | 39 of 39 | **constant at 39 of 39** | — | — |
| **P3** the same for the food sensors | 39 of 39 | **constant at 39 of 39** | — | — |
| **P4** characteristic path length | 3.086 | 2.576–2.598 | −0.104 | 1.000 |

**None of the four predicts how slowly a seed learned**, and three of them cannot in principle,
because they barely vary across degree-preserving rewirings. That is worth stating plainly rather
than reporting as four failed correlations:

1. **The settling-depth hypothesis is false.** It supposed a rewiring could push the thermosensory
   route beyond the four hops the substrate settles for. **No rewiring does** — every one of the 39
   motor-readout neurons is reachable from AFD within four hops in **every** graph, the wild type
   included. In a 302-neuron graph with 3,709 chemical edges, four hops reaches everything;
   reachability is saturated and carries no signal.
2. **The wild type's advantage is not a shortcut.** Its shortest AFD → motor path is **3 hops**; every
   rewiring's is **1 or 2**, and its characteristic path length is **longer** (3.086 against
   2.576–2.598). **The rewirings are uniformly better connected by these measures and learn slower.**
   So "the evolved pathway delivers the signal sooner" is not the mechanism — the opposite is true of
   the geometry.
3. **The seed-to-seed heterogeneity is unexplained by structure at this level.** The rewirings differ
   from each other almost not at all on these measures (P4 spans 0.02 across 64 graphs) while their
   learning times span 32 to 2275 episodes. Whatever drives that spread is not gross connectivity.

**What this rules out, and what it leaves.** It closes the shortest-path reading of the pathway
hypothesis, which was the version this probe was built to test. It leaves the hypotheses these four
measures cannot see: **which** neurons sit on the route rather than how many hops it takes, the
pairing of degrees across edges, the sign and weight structure the rewiring redistributes, and
whatever motif organisation the wild type has that a degree-preserving swap destroys. None of those
is testable by regenerating graphs and counting hops.

**It does not touch Logbook 057's result**, which stands on its statistics. A probe that proposes no
mechanism removes one candidate and changes nothing else.

**For V.3**: the registered follow-up — rewiring only non-sensory edges — is still the right next
test, and now for a sharper reason. If the wild type's advantage survives scrambling everything
except the sensory projections, the mechanism is in those projections despite their being *longer*
than the alternatives; if it does not, the advantage is distributed through the graph and the
sensory route is not special.
