# 091: Grounding the Synapse Signs Does Not Make the Wild Type's Chemical Wiring Matter, Receptor Corrections Included (Phase 8b M.10)

**Status**: completed. The rung is readable, and its three registered readings all come back **no
move at the minimum**.

On the thermal cell at target 35 under PPO, the wild type's chemical wiring leads the chemical-only
null by no more under per-connection signs than under random or per-neuron atlas signs.

| registered reading, seeds 2201–2250 | mean `auc_success` | 80% CI | Holm q | verdict |
|---|---|---|---|---|
| **lead**: wild type − null, per-connection signs | −0.012 | [−0.036, +0.011] | 1.0 | `no_move` → **no lead at the minimum** |
| **grounding**: lead(per-connection) − lead(random) | −0.018 | [−0.054, +0.018] | 1.0 | `no_move` → **no change at the minimum** |
| **receptor**: lead(per-connection) − lead(atlas) | −0.003 | [−0.032, +0.025] | 1.0 | `no_move` → **no change at the minimum** |

Every interval lies inside ±0.058, the registered minimum: a change the size of the cell's largest
committed wiring effect.

**Beside them, never read as verdicts:**

- **Grounded signs often fail to learn at all,** even at a calibrated operating point. Runs with no
  learning: about a fifth of runs under grounded signs, against 3 of 100 runs under random signs.
- **Under per-connection signs the wild type fails more often than the null:** 10 of 50 seeds against
  3 (Fisher p = 0.07).
- **The first pilot's per-connection lead was an operating-point artefact.** +0.10 to +0.14
  uncalibrated; it is gone once every sign level runs at the same operating point.
- **B.3's knockouts on per-connection signs repeat B.3:** no effect on on-lawn speed at the minimum,
  and the same costs to intake.

**Date**: 2026-10-12.

**OpenSpec change**: `add-per-connection-signs-rung`.

**Pre-registration**: [supporting/091-per-connection-signs/launch.md](supporting/091-per-connection-signs/launch.md),
committed at b596ab68 before any panel run. It followed two pilots, a scale calibration and two spec
reviews.

## Objective

Phase 8 asks, at each substrate rung, whether the wild-type wiring starts to matter once that piece
is real. Synapse signs were the connectome brain's nearest unrun fidelity step.

**Two sign models existed:**

- **Random:** each sign is drawn.
- **The per-neuron atlas rule:** a synapse takes its presynaptic cell's transmitter sign. It makes
  every glutamatergic synapse excitatory.

**H.2 vendored a per-connection table:**

- **Its sources,** in order: physiology for 51 connections, receptor-expression polarity (Fenyves et
  al. 2020) for 1,676, the per-neuron rule for 1,449, and no sign for 533.
- **Against the atlas:** it grounds the same 3,176 synapses and **corrects 342 of the atlas's
  signs**, AWC → AIY among them.

**The rung's question:** does grounding the signs, and correcting them, make the wild type's
**chemical** wiring matter?

**Why it matters on this cell.** At random signs it barely does: +0.016 against Logbook 078's
gap-held null. The cell's large wiring effect, +0.0865, is gap-junction placement, which signs do not
touch.

## Method

**The cell** is block V's thermal cell at target 35, the cell of Logbooks 078 and 082:

- the connectome PPO brain on Cook 2019;
- settling to depth 4;
- 3,000 episodes per run.

**The arms:** sign model {random, atlas, per-connection} × wiring {wild type, chemical-only null},
each learning and frozen. That is 12 arms on seeds 2201–2250, 600 runs, and all of them succeeded.

**The sign model** (`synapse_signs: per_connection`):

- **Signs only.** Every synapse keeps its drawn magnitude and takes its table sign.
- **Null edges.** In a rewiring each null edge keeps its presynaptic neuron, so it carries the sign
  of the wild-type edge whose place it took, as it already carries that edge's synapse count. Each
  neuron keeps its set of outgoing signs; only their placement differs.
- **The sign hold.** Grounded signs are held under PPO.

**The operating point.** This decision came after the first pilot (design Decision F).

- **The problem.** At the draw's scale, grounded signs saturate the settled network. The committed
  calibration at scale 1.0 has 35.0% of units saturated under the atlas and 24.4% under per-connection
  signs, against 8.5% under random signs. A first check on the pilot's seeds, with a different input
  set and not committed, read 30–34%, 18–22% and 6%. Mostly
  excitatory synapses add coherently.
- **The knob.** `chemical_weight_scale` multiplies every chemical weight. It is shared by both
  wirings at a level.
- **The calibration.** On seeds 9601–9616, at initialisation, under a rule fixed before it ran, the
  saturated fraction was matched to random signs' (8.5%). That chose **0.35 for the atlas and 0.42
  for per-connection signs**.

**The readings** are three, on `gap = wild type − null` (`auc_success`), paired by seed:

- **the lead** under per-connection signs;
- **grounding**, per-connection − random;
- **receptor correction**, per-connection − atlas.

Each is classified by `mc.classify` at 0.058, with two-sided Wilcoxon q values Holm-adjusted across
the three.

- **The grounding contrast** reads three things at once: the signs, the sign hold and the scale.
- **The receptor contrast** compares two held, calibrated levels, so it reads placement alone.

**Gates:** per sign level, each learner beats its floor, and the two do not both saturate.

### How the registration was arrived at

**The first pilot was uncalibrated** (seeds 9501–9504).

- 4 of 16 grounded learning runs never learned (plateau under 5%), and a fifth barely did (5.3%).
- A 6-run check with the sign hold off still left 2 that never learned, so the starting signs were
  the cause, not the hold.
- On the seeds that learned, the per-connection lead was **+0.10 to +0.14**. Per-connection signs
  sat closer to the operating point than the atlas, so that contrast would have read operating point
  as well as placement.

**The second pilot was calibrated** (seeds 9511–9514).

- 1 of 16 grounded learners never learned.
- The per-connection lead became **−0.070** [−0.121, −0.019].
- The receptor reading's spread set the panel at **50 seeds**.

## Results

### Gates

Every level is readable, with both learners well above their floors (success plateau, %):

| sign level | wild type | null | floors |
|---|---|---|---|
| random | 66.3 | 66.7 | 0.0 |
| atlas (scale 0.35) | 44.1 | 46.8 | 0.0 |
| per-connection (scale 0.42) | 51.1 | 55.8 | 0.0 |

### The registered readings

All three come back `no_move` (table above). The achieved spreads are reported beside the minimum
and never used to re-read it:

| reading | achieved sd | achieved MDE | pilot sd |
|---|---|---|---|
| lead | 0.132 | 0.046 | 0.088 |
| grounding | 0.205 | 0.072 | 0.146 |
| receptor | 0.156 | 0.055 | 0.164 |

The grounding reading's achieved MDE is above the minimum, but its interval still lies inside
±0.058, which is what `no_move` asks.

### Described beside

**The lead at each level** (`auc_success`):

| level | lead [80% CI] | among seeds where both learners learn | `episodes_to_30pct_success` lead |
|---|---|---|---|
| random | +0.006 [−0.019, +0.030] | +0.014 (47 seeds) | +27 |
| atlas | −0.008 [−0.034, +0.016] | −0.014 (34 seeds) | −75 |
| per-connection | −0.012 [−0.036, +0.011] | +0.018 (37 seeds) | −86 |

atlas − random: −0.014 [−0.048, +0.015].

**Runs that never learn** (plateau under 5% after 3,000 episodes):

| level | wild type | null |
|---|---|---|
| random | 2 / 50 | 1 / 50 |
| atlas | 11 / 50 | 11 / 50 |
| per-connection | **10 / 50** | **3 / 50** (Fisher p = 0.07) |

- **It is partly a property of the seed.** 10 seeds fail at both grounded levels.
- **Calibration matched saturation but not learnability.** About a fifth of grounded runs still never
  start, against one in sixteen in the second pilot.

**B.3's knockouts on per-connection signs** ([b3-perconn.json](supporting/091-per-connection-signs/b3-perconn.json)).
These are B.3's modulated arms with per-connection signs in place of the atlas's, on B.3's seeds
2001–2032 and at scale 1.0, as B.3 ran.

- **Gates:** intake passes (+2.46), and dwelling passes (+0.435 over the floor). Every seed learned.
- **On speed,** with B.3's rule: acute *tph-1* +0.002 [−0.004, +0.009] and acute *pdfr-1* +0.002
  [−0.004, +0.008]. Both read **no effect at the minimum.** In B.3 these were *tph-1* `no_move` and
  *pdfr-1* `unresolved`.
- **On dwelling share,** oriented: *tph-1* **+0.115** [+0.051, +0.183], in the mutant's direction,
  as in B.3's +0.109. *pdfr-1* −0.083 [−0.158, −0.012], against the mutant.
- **Intake costs:** −0.53 and −0.50, against B.3's −0.59 and −0.42.
- **PDF's inhibition of NSM** ends weaker than it starts on 29 of 32 seeds, at a median of 0.33. In B.3
  it was 29 of 32, at 0.39.

Correcting AWC → AIY, where both modulators converge, changes none of B.3's readings
qualitatively.

## What this establishes, and what it does not

**Established (registered):**

- **The wild type's chemical wiring does not lead the chemical-only null by a cell-scale amount under
  per-connection signs** on this cell, under PPO.
- **Grounding the signs does not move that lead by a cell-scale amount, and neither does correcting
  342 of them.** The receptor reading compares two held, calibrated levels, so the corrections'
  placement is what it tests.

**Described, with that standing:**

- **An apparent sign effect can be an operating-point effect.** The uncalibrated pilot's
  per-connection lead of +0.10 to +0.14 disappeared once every level ran at the same saturation.
  This rung's design depends on that calibration, and so would any other comparison of sign models
  on this substrate.
- **Grounded signs make the settling substrate hard to start learning.** A fifth of runs never take
  off in 3,000 episodes, even at matched saturation.
- **Under per-connection signs the wild type fails more often than the null** (10 against 3, p =
  0.07). If real, the true wiring with true signs is harder to start learning than a scrambled one.
  Not tested.

**Not established:**

- **That signs never matter.** Signs here are held under PPO, magnitudes are drawn, and the cell
  rewards a learned chemotaxis policy. A substrate with measured weights *and* signs, or a learner
  that does not rebuild the weights, could differ.
- **Why grounded runs fail to start,** beyond the saturation the calibration removed.

## Biological fidelity

- **Signs are per connection, not per receptor.** The table gives one sign per connection; real
  synapses can carry receptors of both signs, and the 533 synapses with no sign stay at their draw.
- **Mostly excitatory is the biology.** Under both grounded models most chemical synapses are
  excitatory, as cholinergic and glutamatergic transmission dominate the worm. The saturation that
  needed calibrating is a property of graded tanh units and a drawn scale set for random signs. Real
  neurons have adaptation and inhibition that this substrate does not.
- **The calibration is an operating-point choice, not biology.** It sets the weight scale so the
  substrate is equally saturated under every sign model. The real network's scale is unknown here;
  B.1's measured weights did not move the wiring effect either.
- **The thermal cell's large wiring effect is gap junctions,** which signs do not touch. This rung
  asked only about the chemical wiring.

## Next

- **M.10 is met.** The sign model and its calibration exist, and the rung has its registered readings.
- **The M.9 decision**, the thermal body cell, comes next, as the user set before this rung.
- **The realistic replay renderer** follows this readout, as decided with the user.
- **Candidates, none scheduled:**
  - **Learnability under grounded signs.** Explain why a fifth of grounded runs never start, for
    example with per-cell gain or excitation/inhibition balance.
  - **The wild type's higher failure rate** under per-connection signs, against more seeds.

## Artefacts

**The registration and the panel:**

- [launch.md](supporting/091-per-connection-signs/launch.md) and [preflight.txt](supporting/091-per-connection-signs/preflight.txt): the registration and its gate preflight on the calibrated pilot.
- [panel.json](supporting/091-per-connection-signs/panel.json): the gates, the three readings and their verdicts, every gap and change, and both metrics.
- [per-seed.csv](supporting/091-per-connection-signs/per-seed.csv): one row per seed, with every arm's plateau and floor.

**The calibration and the pilots:**

- [scale-calibration.json](supporting/091-per-connection-signs/scale-calibration.json): the calibration.
- [pilot.json](supporting/091-per-connection-signs/pilot.json) and [pilot-per-seed.csv](supporting/091-per-connection-signs/pilot-per-seed.csv): the uncalibrated pilot.
- [pilot2.json](supporting/091-per-connection-signs/pilot2.json) and [pilot2-per-seed.csv](supporting/091-per-connection-signs/pilot2-per-seed.csv): the calibrated pilot.

**B.3 beside:** [b3-perconn.json](supporting/091-per-connection-signs/b3-perconn.json) and [b3-perconn-per-run.csv](supporting/091-per-connection-signs/b3-perconn-per-run.csv).

**To reproduce:**

```bash
uv run python scripts/analysis/sign_scale_calibration.py --out scale-calibration.json
uv run python scripts/analysis/sign_rung.py score --logs campaigns/m10-panel/logs \
  --seeds $(seq 2201 2250) --out-dir /tmp/m10-score --out panel.json --csv per-seed.csv
```

The campaign directories and session records are archived off-repo.
