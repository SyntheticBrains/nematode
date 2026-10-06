# 079: Against a Null That Keeps the Sensory-Motor Boundary, Block V's Remaining Lead Disappears (Phase 8b A.3)

**Status**: completed — **`boundary`; `no_lead`. The hop predictor: `unresolved`.**

On hard350 under PPO at block V's point, the wild type leads the chemical-only null by +0.024
`auc_success`. That null already holds the wild type's gap junctions and autapses. Holding the
**sensory-motor boundary** as well removes the rest. A null that keeps every chemical edge out of an
injected sensor and into a readout motor neuron, and rewires only the interior, learns as fast as the
wild type, and on episodes faster.

| registered reading | mean | 80% CI | q | verdict |
|---|---|---|---|---|
| base effect: wild type vs chemical-only null | +0.0239 `auc_success` | [+0.017, +0.031] | — | present; gate passes |
| **interaction**: gap(boundary) − gap(chemical) | **−0.0271** | [−0.033, −0.021] | < 0.001 | **boundary** |
| **lead**: wild type vs boundary null | **−0.0032** | [−0.010, +0.003] | 0.94 | **no_lead** |
| hop predictor: Spearman rho, 48 seeds | −0.154 | [−0.336, +0.042] | p 0.29 | **unresolved** |

Beside them, on episodes to competence: the wild type is +108 against the chemical-only null and
**−108 [−175, −41] against the boundary null**.

**What block V's advantage on hard350 is made of, after A.6, its split and this panel:** about half
came from the degree-preserving null's rewired gap junctions, and the rest from how the null rewires the
edges out of the sensors and into the motor neurons. Against a null that keeps both, **no advantage of
the wild type's interior wiring is detectable**, within ±0.014 `auc_success`.

**Date**: 2026-10-06.

**OpenSpec change**: `add-boundary-null`, which adds the `rewired_boundary_held` wiring to
`connectome-ppo-brain`.

**Pre-registration**: [supporting/079-boundary-null/launch.md](supporting/079-boundary-null/launch.md),
committed before any scored run and before the predictor was computed, after the gate preflight and a
spec review.

## Objective

A degree-preserving rewiring can route an injected sensor straight onto a readout motor neuron. The wild
type has no motor neuron one hop from a food sensor; the current null has about nine and the
chemical-only null about eight ([Logbook 071](071-operating-point-surface.md),
[074](074-null-strength-control.md)). A.2 found block V's advantage depth-critical, and tied that to these
shortcuts after the fact. Park (arXiv:2609.39248) found that nulls keeping the sensory-motor boundary
erase a fly connectome's apparent difference. This logbook registers the shortcut statistic as a
predictor, and builds the null that removes the shortcuts.

## Method

### The boundary-preserving null

`wiring: rewired_boundary_held` is the chemical-only null — chemical graph rewired by the
degree-preserving swap, gap junctions and the 38 autapses held — with every chemical edge leaving one of
the 17 injected sensors (ASE, AWC, AWA, AFD, ASH, ASI, ALM, AVM, PLM) or entering one of the 39 readout
motor neurons (VB, DB, VA, DA) held at the wild type's with its count. 544 edges are held and the 3,165
interior edges are rewired. Tests confirm the boundary is held exactly, every degree is exact, and every
one- and two-hop chemical route from those sensors to those motor neurons equals the wild type's; over
the propagating graph its one-hop count and reach within two hops are the wild type's (0 and 26 of 39).

### The hop predictor

Over the 48 committed hard350 PPO seeds at block V's point (A.2's centre, seeds 161–176; A.6's `full`
level, seeds 305–336): Spearman's rho between each seed's current-null one-hop count, which runs 4–14,
and its wild-type minus null `auc_success` gap. Predicted negative, minimum |rho| 0.3, a two-sided
permutation p and an 80% bootstrap interval. Computed once, after the registration was committed.

### The panel

The wild type, the chemical-only null and the boundary null, each learning and frozen, on hard350 under
PPO at block V's point (edge-order draw, pooled readout, depth 4). **Seeds 641–768 (128)**, fresh,
3,000 episodes, 768 runs, all succeeded in 11.8 hours. Two readings corrected together, against a
minimum of **0.0143**, two-thirds of A.6's committed +0.0215 lead over the chemical-only null. The
interaction reads only where that lead exists on this panel's seeds, a base-effect gate added at spec
review.

## Results

### The hop predictor: unresolved

rho = −0.154, two-sided p = 0.29, 80% interval [−0.336, +0.042]. The sign is the predicted one; the
interval reaches past the −0.3 minimum, so neither a prediction nor its absence is shown. The
registration stated the test's power at about 55% for a true rho of −0.3.

### Gates and the base effect

| level | wild type plateau / floor | null plateau / floor | beats floor | saturated |
|---|---|---|---|---|
| chemical | 76.5% / 0.1% | 74.6% / 0.0% | yes | no |
| boundary | 76.5% / 0.1% | 75.5% / 0.0% | yes | no |

The wild type's lead over the chemical-only null is +0.0239 `auc_success` [+0.017, +0.031], ahead on 80
of 128 seeds. That replicates A.6's +0.0215 on fresh seeds, and passes the base-effect gate.

### The registered readings

**Interaction → `boundary`.** Holding the boundary moves the wild type's lead by −0.0271 \[−0.033,
−0.021\] at q < 0.001, about 1.9 times the minimum, and by −216 episodes [−287, −145]. The whole
chemical-only lead goes.

**Lead → `no_lead`.** Against the boundary null the wild type is at −0.0032 [−0.010, +0.003], inside
the ±0.0143 band, ahead on 65 of 128 seeds. Beside it, on episodes, the boundary null reaches competence
108 episodes sooner [−175, −41].

### Achieved sensitivity

| reading | achieved sd | achieved MDE | registered MDE | minimum |
|---|---|---|---|---|
| interaction | 0.056 | 0.012 | 0.0135 | 0.0143 |
| lead | 0.056 | 0.012 | 0.0140 | 0.0143 |

Drift reads as PPO's positive control. Censoring is comparable: every arm crosses 30% on every seed.

## Analysis

**The shortcuts did not help the chemical-only null at depth 4; the boundary arrangement helped the wild
type.** The interaction is negative: the null that keeps the wild type's boundary does better than the
null that rewires it. A.2's depth surface found the null winning at depth 2, where only shortcuts reach
the motor layer. At depth 4, where the wild type's routes also arrive, rewiring the boundary costs the
null. The boundary null holds two things at once — the absence of shortcuts, and which interneurons each
sensor feeds and each motor neuron hears — and this panel does not separate them.

**Park's finding, reproduced under learning.** In an embodied fly connectome under evolution, Park found
the connectome's difference from degree-preserving nulls vanishing against boundary-preserving ones. Here,
under gradient learning on the worm connectome, the wild type's remaining lead vanishes the same way. Both
say a degree-preserving null scrambles the sensory-motor interface, and comparisons against it measure
that interface.

## What this establishes, and what it does not

- **Established**: on hard350 under PPO at depth 4, against a null with the wild type's gap junctions,
  autapses and sensory-motor boundary, the wild type's interior chemical wiring confers no learning-speed
  advantage detectable at ±0.014 `auc_success`; on episodes the null is faster.
- **Established**: the lead the wild type holds over the chemical-only null lies in the boundary edges.
- **Not established**: which part of the boundary — the absence of shortcuts or the specific
  sensor-to-interneuron and interneuron-to-motor edges.
- **Not established**: anything on the thermal cell, at other depths, or under the reading learner. Block
  V's thermal half keeps Logbook 078's condition.
- **The predictor**: unresolved; nothing is claimed about shortcuts predicting the gap seed by seed.

## Block V, restated

On hard350 under PPO at depth 4 the wild type learns faster than a degree-preserving null: by +0.049
`auc_success` against the current null, about half of it from that null's rewired gap junctions
([Logbook 075](075-gap-only-split.md)), and **the rest from its rewired sensory-motor boundary**. Against
a null keeping the wild type's gap junctions, autapses and boundary, **no advantage remains**. On thermal
at target 35 about 84% came from the gap junctions ([Logbook 078](078-thermal-split.md)); the boundary
was not tested there.

## Registered consequences

D21 named the boundary-preserving null as the family's third null, reported beside the chemical-only
primary. This result makes it the null against which any claim about the connectome's **interior**
wiring has to be read. Whether it should become 8b's primary null for C.1e is a decision for the D21
amendment, not for this logbook.

## Artefacts

- [launch.md](supporting/079-boundary-null/launch.md), [preflight.json](supporting/079-boundary-null/preflight.json):
  the registration and its pilot gate evidence.
- [hop-predictor.json](supporting/079-boundary-null/hop-predictor.json): the predictor, per seed.
- [control.json](supporting/079-boundary-null/control.json): gates, gaps, the interaction, censoring,
  drift and the reading.
- [per-seed.csv](supporting/079-boundary-null/per-seed.csv): every arm's plateau and floor, both gaps and
  the interaction, per seed.
- Reproduce: `scripts/analysis/hop_predictor.py`; `scripts/analysis/boundary_null.py --campaign <campaign> --out-dir <dir> --out control.json --csv per-seed.csv`. Campaign directories archived
  off-repo.
