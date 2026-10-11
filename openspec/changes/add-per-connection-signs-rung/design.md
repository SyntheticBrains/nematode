## Overview

A third sign model for the connectome brain, built on H.2's vendored per-connection table. With it
comes a registered rung asking whether the wild type's lead over a rewired null depends on its
signs. The decisions below were taken with the user on 2026-10-12.

## Decisions

### Decision A: The per-connection sign model

`synapse_signs: per_connection` signs each chemical synapse from `per_connection_signs()`. The table
takes the first source that gives a sign:

1. a cited physiology measurement;
2. Fenyves et al. 2020's expression-based polarity, where it rests on a transmitter the cell releases;
3. the per-neuron transmitter rule;
4. no sign.

**As under the atlas model:**

- **Magnitudes are untouched.** Every synapse takes exactly one draw first, and a grounded synapse
  keeps the draw's magnitude with its sign, `|drawn| × sign`. The RNG stream and every magnitude are
  identical to the random-sign build.
- **Unsigned synapses keep their draw.** A synapse without a sign keeps the drawn value and a zero in
  the `chem_sign` buffer, so enforcement leaves it free.
- **The buffer is checked on load.** The sign buffer is a wiring buffer, so a checkpoint saved under
  another sign model is refused.

The table and the atlas ground the same 3,176 of 3,709 synapses, and they differ in 342 signs. On
the wild type, the per-connection and atlas models therefore differ only in those 342 synapses.

The table is defined on the wild type's edges. Cook 2019 and Emmons 2024 have the same chemical
edges, so either source can use it.

### Decision B: A null's signs travel with its edges

**How a null is built.** Each accepted swap takes `(a→b), (c→d)` to `(a→d), (c→b)`. So every null
edge keeps a wild-type edge's presynaptic neuron and takes a new target.

**What already travels.** Synapse counts move with the edge: `(a→d)` carries `(a→b)`'s count.

**The rule.** Signs travel the same way: **each null edge takes the sign of the wild-type edge whose
slot it took.**

- **What it preserves.** Every neuron keeps its exact set of outgoing signs, and only their placement
  differs. The wild type and the null then differ in placement alone, not in where their signs came
  from.
- **It matches today's atlas signs.** An atlas sign depends only on the presynaptic neuron, so under
  the atlas model this rule gives exactly what the code does now.

**How the origin is known.** The swap edits an edge list in place by index. So the rewiring can
return, beside the null, a map from each null edge to the wild-type edge in its slot:

- edges outside the swap (autapses and boundary-held edges) map to themselves;
- recording the map draws nothing, so the rewired graph is unchanged.

**The alternative, not taken.** Signing a null's edges by the per-neuron rule would make the two
wirings differ in sign source as well as placement.

### Decision C: Enforcement and the paths that read signs

Every place that accepts `atlas` accepts `per_connection`:

- **PPO's sign hold:** the clamp after each optimiser step.
- **The plastic rules:** their sign projection and the decorrelating variant's sign reads.
- **The validators:** enforcement requires grounded signs, and a measured weight prior is refused with
  grounded signs because one synapse would have two sign sources.
- **Checkpoint metadata.**

Each of these checks becomes "signs are grounded", meaning not random, instead of "signs are the
atlas's".

**When off, nothing changes.** With `synapse_signs` at `random` or `atlas`, the brain is
byte-identical to before.

### Decision D: The rung

**The cell** is block V's thermal cell at target 35 under PPO, the cell of Logbooks 078 and 082. It
is chosen because it does not saturate and its gates are known, **not** because its wiring effect is
large.

- **That large effect is gap-junction placement** (+0.0865 in Logbook 078's split; +0.096 in M.8).
  Signs touch only chemical synapses, so they cannot touch it.
- **The relevant baseline is small.** That is the wild type's lead over a null that differs in
  chemical placement alone. At random signs it is **+0.016** [+0.006, +0.027] here, against Logbook
  078's gap-held null, read as `lead_below_minimum`. On hard350 it is +0.025.
- **So the rung asks whether grounding signs makes the chemical wiring matter** where, at random
  signs, it barely does. That is the phase's question at this rung, read where its answer is
  currently near zero.

**The wiring** is the wild type against **the chemical-only null**. It is D21's primary null, and it
differs from the wild type in chemical placement alone, which is where signs live.

**The arms:** signs {random, atlas, per-connection} × wiring {wild type, chemical-only null}, each
learning and frozen. That is 12 arms.

- **Signs are enforced at both grounded levels.** Under PPO a grounded sign is a constraint, not
  only an initialisation, and B.3 found this matters.
- **Random signs** have nothing to enforce.

**The metric** is `auc_success`, the thermal cell's primary metric.

**Gates, per sign level, in the gate preflight's shape:**

- each learner beats its frozen floor;
- the two learners do not both saturate.

A level failing either leaves the rung unreadable.

**The readings** are three, on the wild type's lead `gap = wild type − null`, paired by seed:

- **the lead:** `gap(per_connection)`. Does the wild type's chemical wiring matter once signs are
  real? This is the phase's question at this rung, read as M.8 read its lead.
- **grounding:** `gap(per_connection) − gap(random)`;
- **receptor correction:** `gap(per_connection) − gap(atlas)`.

Each is classified by `mc.classify`. The q values come from two-sided Wilcoxon tests,
**Holm-adjusted** across the three. Holm is implemented directly, as in `lawn_states.holm`, since the
house family correction is BH-FDR.

**The minimum** is 2/3 of 0.0865, which is **0.058**: **a change the size of the cell's largest
committed wiring effect.** 0.0865 is how far a change to the null's gap placement moved the lead in
Logbook 078's split. M.8 used the same reference.

- **What a pass would mean.** The readings ask whether signs move the chemical lead by a cell-scale
  amount. That would take it from +0.016 to beyond +0.07.
- **What is likely.** `no_move` or `below` are the likely outcomes unless grounding transforms the
  chemical wiring's role. The minimum is stated that way rather than anchored on the small chemical
  lead itself: 2/3 of +0.016 is 0.011, below what an affordable panel resolves here.
- **Who decided.** The user chose this framing at the spec review.

**The verdicts,** per reading (for the lead, `move_wt` means the wild type's chemical wiring
matters by at least the minimum):

| state | verdict |
|---|---|
| `move_wt` | the lead grows by at least the minimum |
| `move_null` | the lead shrinks by at least the minimum |
| `below` | a significant change smaller than the minimum |
| `no_move` | no change at the minimum |
| `unresolved` | unresolved at this sensitivity |

**Reported beside, never read as verdicts:**

- the lead at the random and atlas levels;
- `gap(atlas) − gap(random)`;
- `episodes_to_30pct_success`;
- each level's learning against its floor;
- the frozen floors' own wiring gap.

**Seeds and sizing:**

- **The pilot** runs on 9501–9504, configured as the panel. It sets the readings' spreads and the
  cost.
- **The panel** starts at seed 2201 and is sized so the receptor-correction reading's MDE reaches the
  minimum. That reading is the narrower question, and its spread is the one the pilot cannot
  predict.
- **The size** is between 16 and 64 seeds. A pilot predicting more than 64 goes back to the user
  before registration.

### Decision E: B.3's knockouts on per-connection signs, beside

The rung adds two modulated arms: B.3's intact and frozen arms (Logbook 090) with
`synapse_signs: per_connection` in place of `atlas`. They run on B.3's seeds, 2001–2032.

**What is read:** B.3's two acute knockouts, on on-lawn speed and dwelling share, against B.3's own
intact arm on the same seeds.

**Its standing:** it is **reported beside the rung and never read as a verdict.** It asks whether
correcting AWC → AIY, where both modulators converge, changes what the learner makes of serotonin
and PDF.

It is not registered as a second rung, because B.3's readout already showed that knockouts on
single-state seeds say little. Each substrate's knockouts are read against its own intact arm, so
comparing knockout effects across the two substrates is description only.

## Risks

- **The receptor correction touches 342 of 3,709 synapses,** and most of them may sit away from the
  routes the thermal task uses. A `no_move` there is a finding about where the task's computation
  lives, not a failed rung.
- **Measured weights did not move the wiring effect** (B.1c, Logbook 073). Signs may not either. The
  rung is registered so that either outcome reads.
- **Enforcement under PPO is new** (B.3, #444), and grounded and random signs are learned under
  different constraints. The grounding contrast reads that difference together with the signs
  themselves; the logbook names this confound. The beside readings report each level's learning so
  the constraint's cost is visible. The receptor-correction contrast has no such confound, since
  both of its levels are enforced.

### Decision F: Each sign level at random signs' operating point (added after the pilot, 2026-10-12)

**What the pilot found** (seeds 9501–9504). Grounded signs pushed the settling network off its
operating point.

- **Saturation at initialisation.** Over a fixed set of inputs, 30–34% of units saturated (|h| > 0.99)
  under the atlas model and 18–22% under per-connection signs, against **6%** under random signs.
- **The cause.** The chemical weights are drawn for random signs, whose excitation and inhibition
  largely cancel. Grounded signs are mostly excitatory and add coherently.
- **What it did to learning.** 5 of 16 grounded learning runs never learned in 3,000 episodes, and
  the others took off late. With enforcement off, 2 of 6 still never learned, so it is the starting
  signs, not the sign hold.
- **A confound.** Per-connection signs, with their extra inhibition, sat closer to the operating point
  than the atlas. Their contrast would have read operating point as well as placement.

**The change, decided with the user.** `chemical_weight_scale`, a multiplier on every chemical
weight once drawn, signed and placed (default 1.0, byte-identical). It is set per sign level and
shared by both wirings at that level, so the wiring contrast is untouched.

**The calibration rule,** fixed before it runs:

- **Seeds** 9601–9616, disjoint from every pilot and the panel.
- **Brains:** the wild type and the chemical-only null at each grounded level, at initialisation.
  Saturation is an initialisation property, so no training is needed.
- **Inputs:** 64 food and thermal input vectors drawn uniformly on [−1, 1] from a fixed generator.
- **Statistic:** the mean fraction of units with |h| > 0.99 after settling, over inputs, seeds and
  both wirings.
- **Target:** the random-sign level's statistic at scale 1.0, on the same seeds and inputs.
- **Choice:** for each grounded level, the scale on the grid {1.0, 0.85, 0.7, 0.6, 0.5, 0.42, 0.35,
  0.3, 0.25, 0.2} whose statistic is closest to the target, ties to the larger scale.
- **Reported beside:** each scale's action-mean spread across inputs.

**Then:** a second pilot on seeds 9511–9514 with the calibrated scales, configured as the panel,
before registration.
