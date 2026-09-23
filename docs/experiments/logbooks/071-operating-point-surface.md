# 071: The Wiring Advantage Holds at a Point, Not Across the Region (Phase 8 A.2)

**Status**: completed — **the committed operating point sits on an edge, not a plateau.** Under PPO
the wild type's advantage replicates at the committed point (+536 episodes, 92% of A.1's size) and
survives at a settling depth of 6 and at every initial-noise level tried, but it is **abolished one
hop down, at depth 3, and reversed at depth 2**. Under the reading learner the sign is the other way
at the same point — the rewired null is ahead, as Logbooks 064 and 066 already found — and only
readout width moves it, by an interaction within 5% of Logbook 066's. A graph measurement accounts
for the depth result: the real connectome has **no** motor neuron one hop from a food sensor, and a
degree-preserving rewiring manufactures about nine. A.1's shared-initialisation control, re-read at
depth 6 on 32 seeds, is **consistent with A.1** — no dissolution, two of four readings surviving —
but cannot rule out a total dissolution, because the effect it protects is smaller there.

**Date**: 2026-09-21 to 2026-09-23.

**OpenSpec change**: `add-operating-point-surface` (extends `architecture-comparison-protocol`: a swept
level is shown to reach the learner it is set on).

**Pre-registration**: [supporting/071-operating-point-surface/launch.md](supporting/071-operating-point-surface/launch.md),
written before any panel seed ran, with one dated amendment made after the pilot and before the
panel read out.

## Objective

Roadmap decision D16: **no Phase 8 contrast runs at a pin unswept for the learner it uses.** [Logbook
068](068-l1b-rate-calibration.md) had found one inherited pin setting the sign of a registered
primary, and Churchland et al. 2026 report the same sensitivity in a connectome reservoir. A.2 asks
whether block V's wiring effect holds across the region of each learner's settings or is a fact
about one point in it, and fixes the point every later rung cites.

## Method

One factor at a time around the committed point, on **hard350**, wiring crossed with each pin level.
Full design, sensitivity and branch map are in the launch record; in brief:

| | PPO half | reading half |
|---|---|---|
| learner | PPO | `readout_only`, three-factor rule with e-prop eligibility |
| pins | readout width, `forward_pass_depth`, `initial_log_std` | those three plus `plasticity_rate`, `trace_decay` |
| arms / seeds / runs | 32 / 161–176 / 512 | 40 / 177–192 / 640 |
| wall clock | 7.7 h | 16.9 h |

**One cell, chosen on measured sensitivity.** A.1 put thermal's detectable interaction at 1.6–3.4×
its observed effect even at 32 seeds, so a sign question cannot be answered there at any affordable
size. **Frozen floors** run at every level of the three construction pins and are shared across the
learning-only pins, which cannot reach a frozen arm — asserted by test, not argued. **Every level's
reach into its learner is asserted by test**, including the negative cases: `plasticity_rate` and
`trace_decay` are accepted by a PPO arm's configuration and read by nothing, so they are confined to
the reading half. The primary at each level is the **interaction** with the campaign's own centre;
the trigger for a full crossing is the **sign** of the wiring gap. BH-FDR across each half's family.

Both instruments ran **unmodified**, the PPO half through `wiring_premise` and the reading half
through the efficiency module directly with its own arm labels, as L.1b did.

## Results

### The PPO half: depth-critical

The registered gate passes: the centre reproduces block V on both metrics, and this time the
magnitude replicates too — **+536.4** episodes and **+0.058** `auc_success`, 92% and 95% of A.1's
figures. Every arm at every level clears its own frozen floor.

| level | wiring gap | interaction with centre | q | reading |
|---|---|---|---|---|
| depth 2 | −0.236 auc [−0.259, −0.213] | −0.294 | 0.003 | **reversed** |
| depth 3 | −166 ep [−362, +47] | −702 | 0.031 | **abolished** |
| depth 6 | +413 ep [+227, +598] | −124 | 0.823 | survives |
| log-std −1.0 | +313 ep [+85, +544] | −223 | 0.521 | survives, interaction unresolved |
| log-std −0.5 | +481 ep [+233, +725] | −55 | 0.900 | survives, interaction unresolved |
| log-std +0.5 | +153 ep [+121, +188] | −383 | 0.187 | survives, reduced |
| per-neuron readout | +1.25 ep [−27, +29] | −535 | 0.029 | **saturated — not a reading** |

At depth 2 the rewired null reaches **62.3%** plateau full-clear against the wild type's **10.1%**,
and both clear their floors, so it is a genuine reversal and not a broken arm. At depth 3 the wild
type learns fine (77.0%) but no faster than its null. **The per-neuron level is not a wiring
finding**: both arms sit at 96.5% and 97.7%, above the instrument's 90% saturation bar, and a +1.25
episode gap with a tight interval is two arms tied at the ceiling. Readout width is therefore
**unresolved under PPO on this cell**.

### The reading half: the other sign, and only width moves it

The drift obligation is discharged cleanly: `w_chem` drift is **0.00** on every level, every seed,
both wirings, with complete evidence — so the half is readable rather than void. That zero carries
weight because the same check, run on the PPO panel where PPO writes the matrix, reads a relative
drift of **0.77 to 1.44** at every level and wiring, recorded in `ppo-surface.json`.

At the centre the **rewired null is ahead**: −0.210 `auc_success` [−0.287, −0.136], −706 episodes,
plateau 57.4% against the wild type's 36.7%. This is not a failed replication of block V, which is a
PPO result; it **reproduces this learner's own committed prior**, [064](064-l4-frozen-features.md)
and [066](066-l4-readout-width.md), which found the null ahead at the pooled width. The registered
gate was written as "reproduces block V's direction", which was PPO-shaped; the comparator for this
half is L.0 and L.1, and it matches them.

Across the surface the negative gap barely moves. After correction across 22 tests **no interaction
is significant**; two carry the largest raw signals:

| level | interaction | 80% CI | raw two-sided p | q |
|---|---|---|---|---|
| per-neuron readout | **+0.268** | [+0.151, +0.390] | 0.021 | 0.213 |
| depth 3 | −0.174 | [−0.259, −0.091] | 0.021 | 0.213 |
| depth 6 | +0.179 | [+0.052, +0.306] | 0.144 | 0.396 |

**The width interaction replicates Logbook 066 to within 5%** — +0.268 here against +0.2818 there, on
a fresh seed band at a sixth of the seed count, each interval containing the other's point estimate.
The family correction demotes it; the data do not. Depth leans the same way as under PPO — shallower
widens the null's lead, deeper narrows it — but none of that survives correction and it is recorded
as a lean. Several levels leave the wild type barely learning: at depth 2 it reaches **0.1%**
plateau and beats its floor on only 3 of 16 seeds, so its gate passes on the interval alone and it is
reported as a marginal pass, not a learning arm.

### Why depth matters: a graph measurement

[`scripts/analysis/sensory_motor_hops.py`](../../../scripts/analysis/sensory_motor_hops.py) walks the
graph the simulation propagates through — the brain's own masked chemical matrix plus gap junctions,
3,709 chemical edges in both — from the six food sensors to the 39 motor neurons.

| motor neurons at | 1 hop | 2 hops | 3 hops |
|---|---|---|---|
| wild type | **0** | 26 | 13 |
| rewired null, mean of 8 | **9.2** | 29.5 | 0.2 |

The real connectome has no motor neuron one hop from a food sensor, as a sensory → interneuron →
command → motor pathway implies; the rewiring manufactures about nine shortcuts. At a settling budget
of 2 the wild type can drive 26 of its motor neurons and the null essentially all 39 — the reversal.
At 3 the wild type reaches all 39, but its most distant 13 arrive with **no iteration to spare**
while the null's have at least one — the abolition. The advantage appears once the wild type has
slack on its most distant motor neurons.

**This is a description of the graphs, not a mechanism confirmed.** It was computed after the
surface was read, and the structural-predictor rule wants a statistic, a direction and a minimum
named before any correlation. It is handed to A.3 to register properly. Its value is that it
predicts **the operating point at which the effect exists**, where [Logbook 069](069-phase7-synthesis.md)
found no graph property predicting learning time.

### A.1 re-read at depth 6

Registered by the dated amendment: the re-read is owed where the sign moves **at or adjacent to** the
committed setting, and depth 3 is adjacent. Depth 6 is the only depth it can use, since at 2 and 3
there is no advantage to dissolve. A.1's design unchanged — same arms, both sharing definitions,
same instruments — on fresh seeds 193–224.

| mode | metric | interaction | share of baseline | reading |
|---|---|---|---|---|
| `dense_mask` | episodes (primary) | −11.0 | −3.8% | unresolved |
| `dense_mask` | `auc_success` | −0.0075 | −18.2% | unresolved |
| `per_neuron_fanin` | episodes (primary) | +122.8 | +42.8% | **survival** |
| `per_neuron_fanin` | `auc_success` | +0.0198 | +48.0% | **survival** |

**No dissolution, two of four surviving** — against three of four at depth 4 on this cell in A.1.
Consistent with A.1, and not a stronger confirmation of it: the detectable interaction is still
1.15–1.35 times the baseline, because the depth-6 wiring effect on these seeds is **+286.9** episodes
against A.1's +580.4 at depth 4, and below the registered 20% efficiency minimum. The re-read adds
that the effect A.1 protects is weaker at depth 6.

## Corrections made in the open

Recorded here rather than folded silently into the figures above.

- **The re-read was launched under-powered.** At 16 seeds its detectable interaction on the primary
  metric was 2.00 and 1.72 times the baseline — it could not have detected a total dissolution.
  A.1's own launch record had rejected exactly this and moved to 32. It was extended to 32 in the
  same campaign directory under a verified-unchanged execution path. **It mattered**: at 16 seeds
  `dense_mask` on `auc_success` read **−76%** of the baseline with an 80% interval excluding zero; at
  32 it reads **−18%**. Written up at 16 seeds, noise would have entered the record as an erosion.
- **BH-FDR was registered and not implemented** in the driver. Caught while reading the PPO surface
  and added — with a two-sided statistic, since the instrument's test is one-sided — before any
  branch was assigned.
- **The learning gate was first coded stricter than registered**, as "beats its floor on every
  seed", which reported PPO depth 2 as a gate failure on one seed of sixteen. It is the instrument's
  paired test, and every level passes it.
- **The metric rule was re-scoped per level** after the pilot and before the panel. Pooled, one
  badly-censored level (depth 2, spread 1.00) would have voided the censored primary everywhere. The
  launch record says the change was made with the pilot's rates in view.
- **A latent ordering bug** left the drift check undefined at run time; the PPO path never reached it,
  so only the reading half could expose it.
- **Cost.** The PPO half took 7.7 h against the registered 6.9; the reading half landed on its 16.9.

## What this establishes, and what it does not

**Establishes.** Under PPO the wiring advantage is real at the committed point and **depth-critical**:
present at depths 4 and 6, abolished at 3, reversed at 2, with every arm learning. It is robust to
initial action noise. Under the reading learner the null is ahead at the committed point across every
setting of four pins, and only readout width moves that — reproducing L.1's interaction on fresh
seeds. The two learners **disagree about the sign** of the wiring effect at the same operating point.

**Leaves unresolved.** Readout width under PPO, because that level saturates. Whether A.1's survival
holds at depth 6 against a total dissolution, because the panel cannot detect one there. The
reading-half interactions after correction, including the width replication.

**Does not establish.** That path length **causes** the depth dependence — the hop measurement is
post-hoc and registers nothing. That any setting is better than another: A.2 declares no winner and
may not ablate against the point it establishes.

**Block V's claim, restated with its conditions.** Under gradient descent, at a settling depth of at
least four hops, with a pooled motor readout, on hard350, the wild-type connectome reaches competence
sooner than its degree-preserving rewired null, and survives a shared initialisation at depths 4 and
6\. Every clause is load-bearing.

## Registered consequences

- **A.1 re-read**: owed and done (above), consistent with A.1 at depth 6.
- **Readout width under PPO** is unresolved on this cell. B.1's PPO arm cites A.2 for it, so it runs at
  the pooled width and says so, or buys the reading on a non-saturating cell first — recorded on
  tracker B.1c.
- **Thermal confirmation at depths 2 and 3**, branch 2 of the registration, and **the full crossing
  for `forward_pass_depth`**, which D16 gives any pin that moves the sign — both
  *deferred-with-destination* to tracker M.6. The crossing's priority fell once the hop measurement
  explained depth on its own; that is recorded rather than used to drop it.
- **The hop metric** is handed to A.3 to be registered as a structural predictor.
- **Later rungs** cite this surface for the point they run at, per learner. The reading learner's
  surface is flat except in width; PPO's is flat except in depth, below 4, and in width, unread.

## Artefacts

All under [supporting/071-operating-point-surface/](supporting/071-operating-point-surface/):

- `launch.md` — the registration, with its dated amendment.
- `ppo-surface.json`, `ppo-per-seed.csv` — the PPO half: per-level gaps, interactions, gates,
  saturation, metric choice. 224 per-seed rows.
- `reading-surface.json`, `reading-per-seed.csv` — the reading half, including the drift evidence.
  352 per-seed rows.
- `a1-reread-d6.json` — the 32-seed re-read.
- `sensory-motor-hops.json` — the hop measurement, wild type and eight rewirings.

Drivers: `scripts/analysis/operating_point_surface.py`, `scripts/analysis/init_sharing_reread.py`,
`scripts/analysis/sensory_motor_hops.py`; configs generated by
`scripts/campaigns/generate_operating_point_configs.py`. Raw campaign logs — 1,600 runs across four
campaign directories — are archived off-repo per A.0.
