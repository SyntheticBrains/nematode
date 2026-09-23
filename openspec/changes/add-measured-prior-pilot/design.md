## Context

B.1a (`openspec/changes/archive/2026-09-23-add-measured-weight-prior/`) built `weight_prior` and the
`measured_weight_scale` multiplier, both byte-identical when off, and refused every pairing it did
not test. B.1c's 2×3 needs the multiplier swept on both learners, the Lee 2026 risk read, and a PPO
arm under D15's shared initialisation. This change is the pilot that supplies all three.

These decisions were taken before this change: PPO runs under `per_neuron_fanin`; the multiplier grid
is 0.25, 0.5, 1, 2 and 4; both wirings run at every level; each learner gets 8 seeds.

## Goals / Non-Goals

**Goals:**

- Define and allow a measured prior under the fan-in draw.
- Sweep the multiplier on both learners.
- Read sign-only against sign-plus-magnitude.
- Select each learner's multiplier by a rule fixed in advance.

**Non-Goals:**

- `measured_shuffled` arms. That prior has `measured`'s value distribution at every multiplier, so
  it inherits the chosen level, and it is B.1c's control.
- Thermal.
- A wiring verdict. At 8 seeds the gap is descriptive.
- Readout width under PPO. Tracker B.1c already records it as unresolved, and it stays with B.1c.

## Decisions

### Decision A: The fan-in pairing keeps the wild type's multiset on the null

`per_neuron_fanin` draws each post-synaptic neuron's block of in-degree values and lays it on the
neuron's incoming edges in pre-synaptic-index order. A degree-preserving rewiring keeps in-degree,
and the blocks are drawn post by post in index order, so the block for neuron *j* holds **the same
values** on both wirings. That identity is the definition A.1 ran as D15(ii).

A measured prior covers some of each neuron's incoming edges. Overwriting after the draw, as under
`edge_order`, would place:

- on the wild type, its covered values plus the block entries at its **uncovered** positions;
- on the null, the covered values on its first *k* edges plus the block entries at positions *k*
  onward.

Those are different subsets of one block wherever a covered position lies past the first *k*, so
the per-neuron multiset, which is the property the draw exists to share, would be lost silently.

**The definition, per post-synaptic neuron.** Let *c* be the positions, in the wild type's
pre-sorted incoming list, of the edges the prior covers, and *k* = |*c*|.

| wiring | edges | value |
|---|---|---|
| wild type | covered | the prior's value, as B.1a defines it for each prior |
| wild type | uncovered | its own block entry, unchanged |
| null | first *k*, in pre-synaptic order | the wild type's covered values, in wild-type pre-synaptic order |
| null | remaining, in pre-synaptic order | the wild type's uncovered block entries, in wild-type pre-synaptic order |

Under `measured_signs` a covered value is the magnitude of the wild type's block entry at that
position times the measured sign, so on the null the sign-only prior also places the wild type's
magnitudes, not the receiving edge's own draw.

**Two properties hold, and both are asserted by test:**

1. Every neuron carries the wild type's exact multiset on both wirings, under every prior.
2. The wild type's uncovered edges are bit-identical to its random-prior fan-in build.

The null's remaining edges are **not** its random build, and they should not be: under this draw
the random null's own values are a placement of the same block.

**Mechanism.** The rewired assignment already carries each neuron's wild-type covered values. Under
the fan-in draw it also carries each neuron's covered positions *c* in the wild type's list. The
topology holds the block for every neuron, so it can place the wild type's uncovered entries on the
null without a second draw. Under `edge_order` nothing changes: B.1a's rule stands, and every
remaining edge keeps its draw.

**`dense_mask` stays refused.** It shares values by edge identity, which a per-neuron placement does
not respect. No definition of it composes with the null's per-neuron assignment, and nothing
registered needs one.

### Decision B: The shuffle gets its own generator

`measured_shuffled` permutes with `get_rng(seed)`, and the fan-in draw also uses `get_rng(seed)`.
They are two generator objects on one stream: the permutation's first integers and the draw's first
normals come from the same bits. Under `edge_order` that never mattered, because the draw came from
the shared generator. Under the fan-in draw it would correlate the shuffle with the uncovered values
it sits beside.

The shuffle therefore takes a generator seeded from the run seed and a fixed tag, which gives a
different stream at every seed and stays deterministic. This changes `measured_shuffled`'s values,
but no run has used that prior (B.1a registered none), so no committed result moves. A test asserts
the permutation is unchanged when the draw mode changes.

### Decision C: Parents and levels

| learner | parents (wild type learn, null learn, wild type frozen, null frozen) |
|---|---|
| PPO | A.1's committed `…_hard350_fanin`, `…_hard350_rewired_null_fanin`, `…_hard350_frozen_fanin`, `…_hard350_rewired_null_frozen_fanin` |
| reading | A.2's reading centre: `…_eprop_readout_only`, `…_eprop_readout_only_rewired_null`, `…_eprop_frozen`, `…_eprop_frozen_rewired_null` |

**Levels:**

- `random`: the parents, unchanged;
- `sign`: `weight_prior: measured_signs`;
- `m025`, `m05`, `m1`, `m2`, `m4`: `weight_prior: measured` at those multipliers.

The multiplier and the prior change the substrate before any learning, so **every level gets its
own frozen floor**. That is 24 new configs per learner. The level `m1` sets only `weight_prior`,
because 1.0 is the default and writing it would be a no-op key.

**The reading half's draw.** The reading half stays on `edge_order`. That is the draw A.2 swept it
at, and D15 is a PPO protocol: A.1 exists because PPO trains the weights it was initialised with. The
reading learner freezes the chemical matrix, so its measured prior is the substrate it reads, not an
initialisation, and A.2's reading surface was measured under `edge_order`. Moving it would leave the
reading learner's operating point.

**The PPO half's draw.** The PPO half moves to `per_neuron_fanin`, which A.2 did not sweep. Its
committed centre is A.1's: `auc_success` survival **established** on hard350, and the primary
inconclusive at −32% with a wide interval. The pilot's `random` level re-reads that centre on fresh
seeds, and the logbook states the draw beside every PPO figure.

### Decision D: Seeds

- **PPO:** 113–120.
- **Reading:** 121–128.

Both bands are untouched: the burnt bands are 1–96, 101–108 and 129–160; A.2's pilot took 109–112;
A.2's panels 161–192; the A.1 re-read 193–224. B.1c's panel starts at 225, so the pilot never shares
a seed with the contrast it calibrates. A test asserts the bands are fresh and disjoint.

### Decision E: What is registered, and the branches

**Per learner and level:**

- The **learning gate**: each learning arm against its own level's frozen floor, under A.2's paired
  test, with its saturation flag.
- The **wiring gap** on the metric A.2's censoring rule chooses, with its interval. The gap is
  descriptive only.

**The branches, in the order they are read:**

1. **Broken arm.** A learning arm that does not beat its floor is reported as a gate failure at that
   level, never as a wiring reading.
2. **Pathway unlearnable (Lee).** The random level's wild type passes, and **no** measured level and
   not `sign` passes. B.1 closes *unmet-with-reason* for that learner, with the pathway named, and
   B.1c does not run that learner. If the random level itself fails, the pilot is uninformative for
   that learner, and the record says so rather than reading branch 2.
3. **Magnitude is the obstacle.** `sign` passes and every `measured` level fails. B.1c runs `measured`
   at 1.0 anyway, as registered, carrying the pilot's gate failure beside it, with `sign` added as
   a reported arm.
4. **Multiplier selection.**
   - Choose **1.0** if the wild type passes there. That is the point where the covered edges carry
     the random draw's magnitude, so B.1c compares structure rather than size.
   - Otherwise choose the passing level nearest 1.0 on the log scale, with ties going to the smaller
     multiplier.
   - **The wiring gap never enters the selection.**
5. **Sign across the multiplier.** This is recorded only where a level's gap interval excludes zero
   on the side opposite to the random level's. At 8 seeds it becomes a **registered condition B.1c
   carries**, not a trigger for a crossing and not a finding.

### Decision F: The analysis reuses A.2's helpers, factored rather than copied

`operating_point_surface.learning_gates` decides the floor level through A.2's own `arms_at` table.
It gains a `floor_level` argument (A.2's call sites pass what they compute today), so the pilot calls
the same function. `wiring_gap`, `censoring_rates`, `choose_metric`, `two_sided` and the two
instruments are imported unchanged.

A second copy of any of them is how two scripts come to disagree about a gate. A.2's own tests pin
its behaviour through the refactor.

## Risks / Trade-offs

- **Four floors per level is most of the cost.** Half the runs are frozen arms, and without them a
  level that fails its gate cannot be told apart from a broken substrate. Kept.
- **Eight seeds is thin.** The pilot selects a level and reads branches; it does not estimate
  effects, and the logbook says so where it reports a gap.
- **PPO moves to a draw A.2 did not sweep.** This is stated beside every PPO figure. The pilot's own
  `random` level re-reads A.1's committed fan-in point.
- **The Lee reading can be mimicked by the multiplier.** A level can fail because its magnitude is
  wrong rather than its structure, which is why branch 2 needs every level and `sign` to fail, not
  one.
