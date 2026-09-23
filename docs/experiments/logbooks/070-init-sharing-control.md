# 070: The Wiring Advantage Survives a Shared Initialisation (Phase 8 A.1)

**Status**: completed — **no dissolution detected on any arm**; survival **established on five of
eight readings**, the other three **unresolved at this panel's sensitivity**. Block V's standing
condition is **partially discharged**: no evidence that the advantage rests on the value-to-edge
pairing, on the scopes that resolved. A second finding is ranked above the
first below — **the thermal cell's effect came in at a third of its committed size**, so block V's
magnitude is less stable across seed sets than its direction.

*(**Conditioned 2026-09-23**, [Logbook 071](071-operating-point-surface.md): A.2 found the wiring advantage this control protects is depth-critical — present at settling depths 4 and 6, abolished at 3, reversed at 2 — so this logbook's result holds at the committed depth of 4. Re-read at depth 6 on 32 fresh seeds, the control is consistent: no dissolution, two of four hard350 readings surviving, though that panel cannot rule out a total dissolution because the depth-6 effect is smaller. This record's verdict stands as read at its own setting.)*

**Date**: 2026-09-21. *(The `dense_mask` arms were re-run the same day after review found a defect in them; see § The defect this panel shipped with, and the re-run. Every figure below is the corrected one.)*

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
| hard350 | `dense_mask` | −38.91 (−6.7%) | −386.96 | 0.681 | **survival** |
| hard350 | `per_neuron_fanin` | −188.03 (−32.4%) | −386.96 | 0.884 | inconclusive |
| thermal | `dense_mask` | +142.84 (+106.1%) | −89.73 | 0.116 | **survival** |
| thermal | `per_neuron_fanin` | +37.62 (+28.0%) | −89.73 | 0.327 | inconclusive |

On `auc_success`, reported beside as registered: survival on hard350 under **both** modes and on
thermal under `dense_mask`; inconclusive on thermal under `per_neuron_fanin`. **Five of eight
readings establish survival, three are unresolved, and none is dissolution or even a significant
shrinkage.**

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

**All three are `per_neuron_fanin`**, and they fail to resolve for two different reasons.

**Two are thermal, and they are a statement about the panel, not about the wiring.** Power was
registered against **V.4's** effect sizes; the thermal effect arrived at a third of that, so the 2/3
bar is small in absolute terms (−89.7) while the interaction's spread is ~1,044. Achieved minimum
detectable interaction on thermal runs **1.6× to 3.4× the observed effect** depending on metric and
mode. Resolving that is not a seed-count problem at any affordable size: clearing the registered 2/3
bar at the observed thermal effect needs roughly **800 seeds**, against 32 here.

**One is hard350's primary, and it is a near miss rather than a power failure.** That reading is
well powered — the detectable interaction is 0.68× the baseline effect. Its point estimate removes
**a third of the baseline effect**: −188.03 against a baseline gap of +580.44, or 32.4%. The
registered bar asks for two thirds removed, which on that baseline is **−386.96**, so the estimate
itself is nowhere near dissolution — but the interval runs out to **−395.9** and so fails to exclude
the bar by about **nine episodes** on a 580-episode effect. It is reported as unresolved because the
registered rule says so, not because the evidence is balanced.

| cell | mode | metric | detectable interaction ÷ baseline effect |
|---|---|---|---|
| hard350 | `dense_mask` | both | 0.50 |
| hard350 | `per_neuron_fanin` | `auc_success` | 0.52 |
| hard350 | `per_neuron_fanin` | primary | 0.68 |
| thermal | `dense_mask` | `auc_success` | 1.42 |
| thermal | `dense_mask` | primary | 2.52 |
| thermal | `per_neuron_fanin` | `auc_success` | 1.60 |
| thermal | `per_neuron_fanin` | primary | 3.41 |

## The defect this panel shipped with, and the re-run

**Found in review, after the 768-run panel had read out.** `ConnectomePPOBrain` passes one generator
to both the chemical weight initialisation and the rollout buffer, whose `get_minibatches` consumes
it for the minibatch permutation. `dense_mask` draws a dense 302×302 matrix — **91,204 values against
the baseline's 3,709** — so it left that generator in a different state and moved **PPO's minibatch
order as well as the weights**. Two manipulations under one name.

**Every check in place passed.** The suite asserted bitwise identity of every parameter the mode did
not claim to touch, across both axes, and it was correct: the initial parameters *were* identical.
The divergence only exists once training starts. This is the failure the requirement this very
change adds — that a sharing claim be verified by test rather than argued from how the stream is
consumed — exists to prevent, and it was argued rather than tested.

**The blast radius was measured, not assumed.** Replaying each mode's consumption showed
`per_neuron_fanin` leaves the shared generator in a state **identical** to the baseline (it draws the
same 3,709 values, merely grouped), and only `dense_mask` diverges. So eight arms were affected and
sixteen were not.

**The fix** keeps `edge_order` bit-identical: the loop always takes one value per edge from the
shared generator whatever the mode, and a sharing mode overwrites it with one from a dedicated
generator seeded off the run seed. Both sharing modes therefore produce the *same weights as before*;
only the shared stream's position changes.

**The re-run** covered the eight `dense_mask` arms, 256 runs, 256/256 succeeded. The 512 unaffected
logs were carried over and their reuse licensed the way the rule requires rather than by the argument
above: **one seed per reused arm re-run under the fixed code, all sixteen compared field by field
against the carried-over logs, every episode of every success and foods series identical.**

**What moved.** The four unchanged arms came back **bit-identical**, which is the cleanest available
confirmation the fix touched only what it claimed. One reading changed branch: thermal under
`dense_mask` on the primary went from inconclusive to **survival**. And hard350's apparent reduction
under `dense_mask` shrank from −190.6 to −38.9 — **the contamination had been making the effect look
more eroded than it is.** The confound worked against block V, not for it.

**A second error, found while checking the first.** The committed version of this logbook reported
**"five of eight readings clear survival"**. Re-deriving every branch from the committed artefact
shows the contaminated panel supported **four**, not five: one on the primary (hard350 under
`dense_mask`) and three on `auc_success`. The tally sentence was an over-count by one; the tables
beside it were correct, and nobody reading the tables could have reached five. That wrong count was
propagated to five citation sites before it was caught.

**The correction and the re-run happen to land on the same number, for different reasons.** The
re-run genuinely moves thermal's primary from unresolved to survival, taking the true count from
four to five. So the figure now standing at those five sites — five survival, three unresolved — is
right, but it was not right when it was written. Anyone reconciling the two versions by their
summary counts alone would conclude the re-run changed nothing, and that is wrong twice over.

The lesson is recorded as a clause of [phase protocol](../../research/phase-protocol.md) principle 6:
an arm must be shown to change only what it names, **at the point the arms diverge**, not only at
construction.

## What this establishes, and what it does not

**Establishes.** No dissolution was detected anywhere: across both definitions and both cells, no
interaction approaches significance and none approaches the registered dissolution bar. Survival is
**established** on five of eight readings — **both** metrics on hard350 under `dense_mask`, **both**
on thermal under `dense_mask`, and `auc_success` on hard350 under `per_neuron_fanin`. On those
scopes Dhiman's mechanism, the advantage being an initialisation artefact, is **not** what is
happening.

**Leaves unresolved**, and these are not evidence of survival: thermal under `per_neuron_fanin` on
both metrics, and hard350 under `per_neuron_fanin` on the primary. Three of eight readings could not
place the effect either way, and **all three are `per_neuron_fanin`** — the definition that shares a
neuron's multiset rather than each edge's value. Whether that definition is genuinely weaker or
merely landed on the noisier arms is not something this panel can separate.

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
- **A follow-up is tracked as M.5**, a MAY in the Phase 8 tracker: the across-seed half of the
  standing condition. Its design is **not** the obvious one — pinning `rewire_seed` to a single
  constant makes the result about one rewiring and reintroduces the shared-nulls caveat V.4
  closed. It needs a variance-components shape: several pinned graphs, several weight seeds
  within each. This panel's magnitude finding is what makes it worth doing.

## Artefacts

- [`supporting/070-init-sharing-control/launch.md`](supporting/070-init-sharing-control/launch.md) —
  the pre-registration, written before any panel seed ran
- [`supporting/070-init-sharing-control/per-seed-interactions.csv`](supporting/070-init-sharing-control/per-seed-interactions.csv)
  — 256 rows, the per-seed unit the primary test consumes; every figure above is re-derivable from it
- [`supporting/070-init-sharing-control/init_sharing_control.json`](supporting/070-init-sharing-control/init_sharing_control.json)
  — the full harness output
- Raw campaign logs: archived off-repo per A.0. The merged panel is 768 runs — 512 from the
  original campaign and 256 from the `dense_mask` re-run — plus the 16-run identity check that
  licensed the reuse.
