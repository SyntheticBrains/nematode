# 070: The Wiring Advantage Survives a Shared Initialisation (Phase 8 A.1)

**Status**: completed — **no dissolution on any arm**, survival on five of eight readings, three
inconclusive with a stated cause. Block V's standing condition is **partially discharged**: the
value-to-edge pairing is not what the advantage rests on. A second finding is ranked above the
first below — **the thermal cell's effect came in at a third of its committed size**, so block V's
magnitude is less stable across seed sets than its direction.

**Date**: 2026-09-21.

**OpenSpec change**: `add-init-sharing-control` (extends `architecture-comparison-protocol`: a claim
that two arms share an initialisation is verified by test rather than argued from how the random
stream is consumed).

**Pre-registration**: [`supporting/070-init-sharing-control/launch.md`](supporting/070-init-sharing-control/launch.md),
written before any panel seed ran.

## Objective

Block V's advantage — the wild-type connectome reaching competence 23–55% sooner than its
degree-preserving rewired null under PPO — ships with a standing condition: rewiring and
initialisation vary together. [Dhiman 2026](https://arxiv.org/abs/2604.04033) reports exactly that
advantage dissolving in the fly under shared initialisation plus a degree-preserving null.

Reading the initialisation path first narrowed the question, and narrowed the decision that fed it.
The draw is per-edge over the `(pre, post)`-sorted edge list; both graphs carry 3,709 edges, so they
**already consume the same standard-normal stream** and the *n*-th value matches. Everything
peripheral — readout, gains, critic — is **already byte-identical** across wirings at one seed,
asserted by a committed test. What differs is only **which edge the *n*-th value lands on**.

So D15's premise, inherited from [069](069-phase7-synthesis.md), was broader than the code: it had
rewiring preserving the degree *sequence* but not which neuron holds which degree, where the
implementation is a directed double-edge swap that preserves every labelled neuron's in- and
out-degree. Corrected at PR review before this panel was designed.

## Method

Wiring {wild type, rewired null} × draw {`edge_order`, `dense_mask`, `per_neuron_fanin`}, both cells,
**32 paired seeds (129–160)**, four arms per cell per mode. **768 runs, 768 succeeded**, 11.75 h at
16×.

Neither sharing definition is uniquely "the same initialisation" once the edge set changes, so both
ran: `dense_mask` gives every edge present in both graphs the identical value; `per_neuron_fanin`
gives every neuron the identical multiset of incoming weights and moves only the pairing. Both
properties are **asserted against constructed brains**, never argued — the requirement this change
adds, and the test that the default does *not* share them is what shows the confound was real.

The primary is the **interaction** of draw mode with wiring, paired by seed, read through the
**unmodified** `wiring_premise` and `connectome_structure_efficiency`. The `edge_order` level was
re-run rather than reused: reuse forbids partial reuse and the dependency set had moved.

**The panel is 32 seeds, not D15's floor of 16, because the pilot said so.** Its registered job was
to confirm the modes run; it also measured the across-mode correlation the panel's power turns on,
which V.4 cannot supply because it has no shared-initialisation arm. That came back unusable
(−0.66 to +0.34 at n = 4), so sensitivity was computed at ρ = 0 from V.4's committed spread — and at
16 seeds the minimum detectable interaction was **1.25 and 1.14 times the effect it would have to
cancel**. A control that exists to answer a published critique should not be unable to see the
answer.

## Results

### The registered gate passed

The `edge_order` baseline reproduced block V's **direction** on both cells and both metrics, so the
interaction was readable rather than void.

### The primary: no dissolution anywhere

Branches assigned by the registered rule — dissolution is a significant interaction removing ≥ 2/3
of the baseline effect; survival is an interval excluding a 2/3 reduction; both directions carry the
same bar.

| cell | mode | interaction (primary) | 2/3 bar | p | branch |
|---|---|---|---|---|---|
| hard350 | `dense_mask` | −190.59 (−32.8%) | −386.96 | 0.915 | **survival** |
| hard350 | `per_neuron_fanin` | −188.03 (−32.4%) | −386.96 | 0.884 | inconclusive |
| thermal | `dense_mask` | +109.75 (+81.5%) | −89.73 | 0.200 | inconclusive |
| thermal | `per_neuron_fanin` | +37.62 (+28.0%) | −89.73 | 0.327 | inconclusive |

On `auc_success`, reported beside as registered: survival on hard350 under both modes and on thermal
under `dense_mask`; inconclusive on thermal under `per_neuron_fanin`. **Five of eight readings clear
survival. None is dissolution, and none is even a significant shrinkage.**

The point estimates disagree in sign between cells — hard350 leans toward a third of the effect
removed, thermal toward the effect growing — and neither approaches significance. **Reported as a
split, not pooled**, per the multi-panel rule.

### The finding ranked above that one: block V's magnitude is not stable

| cell | metric | observed here | V.4 committed | ratio |
|---|---|---|---|---|
| thermal | episodes-to-competence | 134.6 | 382.3 | **35%** |
| thermal | `auc_success` | 0.083 | 0.163 | **51%** |
| hard350 | episodes-to-competence | 580.4 | 553.1 | 105% |
| hard350 | `auc_success` | 0.061 | 0.079 | 77% |

Direction replicated on both cells; **magnitude did not on thermal**. Across four seed sets the
thermal effect has now run +35.4% (V.1), +46.4% and +32.6% (V.1's panels), +55.3% (V.4) and roughly
a third of the committed size here. The effect is real and directionally stable across every panel
that has measured it; **its size moves substantially between seed sets**, which is a property of the
result that no previous record states.

### Why three readings are inconclusive, and what that is a statement about

Power was registered against **V.4's** effect sizes. The thermal effect arrived at a third of that,
so the 2/3 bar is small in absolute terms (−89.7) while the interaction's own spread is ~1,049. The
achieved minimum detectable interaction on thermal is **3.4× its observed effect**.

**Thermal's inconclusive readings are therefore a statement about the panel, not about the wiring.**
Resolving them is not a seed-count problem: closing a 3.4× gap needs roughly 800 seeds, against 32
here. The thermal effect on these seeds is too small relative to its variance for an interaction
test at any affordable size.

## What this establishes, and what it does not

**Establishes.** The wiring advantage does not depend on the value-to-edge pairing. Under both
defensible definitions of a shared initialisation, on both cells, no interaction approaches
significance and none approaches the registered dissolution bar. Dhiman's mechanism — the advantage
being an initialisation artefact — is **not** what is happening here, on the axis this panel tests.

**Does not establish.** That block V is unconditional. Three of eight readings could not resolve, the
strongest single cell is hard350, and the across-seed coupling between a null's graph and its weights
is **untested** — a different experiment, which [065](065-wiring-fresh-rewiring.md) already called one,
and which pinning `rewire_seed` across seeds would address at the cost of reintroducing the
shared-nulls caveat V.4 closed.

## Registered consequences

- **The standing condition is partially discharged.** The pairing half is answered; the across-seed
  half is not. Block V is restated wherever it is cited as: *learning-speed-relevant under PPO,
  robust to initialisation pairing, with a magnitude that varies substantially across seed sets*.
- **The roadmap's risk row does not fire.** Its trigger was "A.1 reads no wiring effect under either
  D15 definition"; the effect is there under both.
- **B.1 proceeds unchanged.** It was never conditional on this outcome — measured weights are a
  different question from random-weight legibility.
- **A follow-up is registered, not scheduled**: pin `rewire_seed` across seeds to isolate the graph
  from the weights.

## Artefacts

- [`supporting/070-init-sharing-control/launch.md`](supporting/070-init-sharing-control/launch.md) —
  the pre-registration, written before any panel seed ran
- [`supporting/070-init-sharing-control/per-seed-interactions.csv`](supporting/070-init-sharing-control/per-seed-interactions.csv)
  — 256 rows, the per-seed unit the primary test consumes; every figure above is re-derivable from it
- [`supporting/070-init-sharing-control/init_sharing_control.json`](supporting/070-init-sharing-control/init_sharing_control.json)
  — the full harness output
- Raw campaign logs: archived off-repo per A.0. 768 runs, 369 MB.
