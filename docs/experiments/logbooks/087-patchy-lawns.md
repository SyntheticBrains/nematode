# 087: On Patchy Lawns, a Learner That Pays to Move Dwells Where Its Untrained Policy Never Does (Phase 8b D.1)

**Status**: completed — **`dwells`.** On lawns where moving costs energy and food depletes slowly, MLP-PPO
reading its own satiety spends far more of its on-lawn time in the dwelling state than its untrained
policy does, by the registered measure.

| registered reading, seeds 2001–2016 | mean | 80% CI | q | state |
|---|---|---|---|---|
| learner − untrained floor, on-lawn dwelling share | **+0.603** | [+0.509, +0.698] | 6 × 10⁻⁵ | `move_wt` → **dwells** |

The interval lies above the registered minimum of 0.394, and 15 of 16 seeds are positive.

**Beside it, never read as verdicts:**

- **Reading satiety does not change how much the worm dwells,** but it changes the bouts. The arm
  without `internal_state` dwells as much, +0.652 over the floor (learner minus it: −0.05
  [−0.19, +0.09]). With satiety, 9 of 16 seeds take complete bouts of both states, against 3. Its
  median bouts last 354 s dwelling and 170 s roaming, against 74 s and 87 s without, on the scale of
  the real reference (about 8 and 1.6 minutes).
- **Depletion runs the wrong way, and the measure is confounded.** Worms roam *less* where the cell
  under them is grazed (−0.18; positive on 2 of 16 seeds). Dwelling is what grazes a cell, so the
  state sets the density as much as the density sets the state.
- **Learners roam more than real worms:** 39% of on-lawn time, against 17% for real wild type read by
  the same classifier.

**Date**: 2026-10-11.

**OpenSpec change**: `add-patchy-lawns`.

**Pre-registration**: [supporting/087-patchy-lawns/launch.md](supporting/087-patchy-lawns/launch.md),
committed at fda00013 before any panel run, after four pilots, the gate preflight and the spec review.

## Objective

D.1 is the positive control for a roaming/dwelling readout, which B.3's serotonin/PDF field will read
next.

**The question:** can a learner that reads its own satiety take the dwelling state on patchy lawns,
where its untrained policy does not?

**The constraint:** the readout has to read simulated worms the way real ones are read. It runs on the
point worm, since C.3 (Logbook 086) found the kinematic body's speed pinned and its turns pivots.

## Method

### The instrument

Roaming and dwelling are read from speed and angular speed over 10-second windows, as in Flavell et
al. 2013 and Scheer & Bargmann 2023.

- **Calibration data.** The model is calibrated on Scheer & Bargmann's 1,586 wild-type animals
  (Dryad, CC0), measured at the point worm's 5-second step exactly as simulated worms are, against
  the authors' own labels.
- **The deposit was read safely.** The 4.9 GB source pickle was read only after an opcode check,
  with an allow-listed unpickler.
- **The authors' model was read correctly.** Their deposited two-state model, decoded here, reproduces
  their labels on 99.2% of 372,860 bins from their own measures.
- **Their line did not transfer to our step.** Recalibrated to the 5 s step, it reached held-out
  κ = 0.49 against a gate of 0.6 fixed in advance, and was not used. At that step a dwelling worm's
  turn is close to random.
- **One retry was registered before it was computed** (f015a7ea). It is a two-state Gaussian-emission
  model on log speed and angular speed, fitted with the labels. It reached held-out **κ = 0.632**,
  passing close to the bar.
- **It calls more roaming than the authors** (17.4% against 10.6% on lawns), so absolute fractions
  are described, never matched.
- **States are read on lawns only.** Off food, worms search rather than roam or dwell.

### The cell, as four pilots shaped it

All four pilots ran on seeds disjoint from the panel, and each change was decided with the user and
recorded before the next ran:

| pilot | cell | what it showed | what changed |
|---|---|---|---|
| 1 | intake 10% of a cell per step, movement free | learners tripled intake but roamed 98–99% of on-lawn time | moving costs energy; intake 2% per step |
| 2 | movement costed, start off food | the learner stood still and starved | the worm starts on a lawn |
| 3 | start on a lawn | the gate passed, but speed noise sat near the states' boundary and turns were random | `entropy_coef` 0.05 → 0.005 |
| 4 | as registered | the gate passed; dwelling appeared | none: the registration |

The first pilot's lesson is that **dwelling has to pay**. With movement free and food quick to
deplete, an optimal forager roams continuously.

The registered cell:

- **Lawns:** 4 disc lawns of 2.5 mm radius in a 20 mm arena, 1 mm cells, and the worm starting on
  one.
- **Intake:** 2% of the cell's density per step, independent of speed.
- **Energy**, in intake value: eating pays 10 reward per unit and restores 0.3 of maximum satiety per
  unit. Moving costs 0.013 per mm, from reward and satiety alike. Basal decay is 0.8 per step, from
  300\.
- **Episodes:** 720 steps, an hour of worm time.
- **Actions:** signed speed, and turns up to π per step.
- **No shaping:** no reward term favours either state.

### The arms

| arm | role |
|---|---|
| MLP-PPO with `internal_state` | the learner |
| its untrained policy | the floor |
| MLP-PPO without `internal_state` | beside |

- **Panel:** seeds 2001–2016, 3,000 episodes per run, 48 runs, all of which succeeded.
- **Evaluation:** each run's final weights, frozen, over 30 held-out episodes.

## Results

### Gate

| arm | intake per episode (final quarter) | against the floor (1.19) | gate |
|---|---|---|---|
| learner | 3.44 | +2.26 [+1.82, +2.69] | passes |
| without `internal_state` | 2.62 | +1.43 [+1.14, +1.69] | passes |

### The registered reading

**+0.603 [+0.509, +0.698]**, q = 6 × 10⁻⁵. The learner's on-lawn dwelling share is 0.614, against its
floor's 0.011.

| seed | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 |
|---|---|---|---|---|---|---|---|---|
| learner − floor | +0.82 | +0.68 | +0.88 | +0.69 | +0.54 | +0.91 | +0.22 | +0.98 |

| seed | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 |
|---|---|---|---|---|---|---|---|---|
| learner − floor | +0.37 | −0.01 | +0.39 | +0.92 | +0.28 | +0.67 | +0.57 | +0.77 |

The verdict is **dwells**. Achieved spread: sd 0.286 and MDE 0.178, against the pilot's 0.377 and
0.234. Reported beside the registered minimum, never used to re-read it.

### Described beside

| reading | learner | without `internal_state` | floor |
|---|---|---|---|
| on-lawn dwelling share, mean (median) | 0.614 (0.680) | 0.663 (0.784) | 0.011 |
| seeds with complete bouts of both states | 9 / 16 | 3 / 16 | 0 |
| median dwelling bout | 354 s | 74 s | — |
| median roaming bout | 170 s | 87 s | — |
| roaming where grazed − where fresh | −0.18 (2 / 16 positive) | −0.12 (3 / 16 positive) | — |
| intake per held-out episode | 3.48 | 2.91 | 1.17 |

- **Against each other.** The learner minus the arm without `internal_state` dwells −0.05
  [−0.19, +0.09]. Reading satiety does not change how much of the time the worm dwells.
- **Bouts.** With satiety, bouts are longer and both states recur. The reference model's mean bouts
  are about 470 s dwelling and 98 s roaming, so the learner dwells somewhat shorter and roams
  somewhat longer than a real worm.
- **Against real worms.** The learner roams 39% of its on-lawn time. Real wild type, read by the same
  classifier, roams 17.4%.

**Why the depletion reading is confounded.** It compares roaming on grazed cells with roaming on fresh
ones. Dwelling grazes the cell the worm sits on, and roaming carries it onto fresh cells, so the
state shapes the density under the worm as much as the density shapes the state. The prediction,
that a worm leaves a spot as it runs out, needs a different statistic: the chance that a dwelling
bout ends, as a function of its spot's density. That would be a new registration, not a re-read.

## What this establishes, and what it does not

**Established:**

- **The learner dwells.** On patchy lawns where moving costs energy and food depletes over minutes,
  MLP-PPO takes a slow, turning state on food that its untrained policy never takes, and spends most
  of its on-lawn time in it. The readout can see a learned dwelling state, so B.3 may read it.
- **Dwelling has to pay.** With movement free and food quick to deplete, learners roam continuously.
  With movement costed and food slow to deplete, they dwell. The conditions are a property of the
  task, as the marginal-value framing expects.

**Not established:**

- **That the states are worm-like in their dependence on depletion.** The reading that was meant to
  test this is confounded, and it ran the wrong way.
- **That reading satiety makes the worm dwell.** It does not change the dwelling share. It is
  associated with longer, recurring bouts and more intake, which is described here, not tested.
- **Patch-leaving.** One lawn holds about three times an episode's food, so leaving is never needed.
  It is deferred to D.1b.
- **Anything about the connectome.** This control is the MLP's. B.3 puts the serotonin/PDF field on the
  connectome.

## Biological fidelity

- **The cell's economics were chosen to make dwelling possible, and that is said plainly.**
  - Locomotion costs energy and lawns deplete over minutes, both true of real worms in kind. Their
    magnitudes are calibrations.
  - The worm starts on food, as assays place worms.
  - Intake does not depend on speed, so slowness is never paid for itself.
- **The instrument passes its gate, close to the bar,** and calls more roaming than the authors. The
  learner's 39% roaming should be read against the classifier's 17.4% on real worms, not the authors'
  10.6%.
- **A stopped worm counts as dwelling,** as real dwelling worms often pause. Here "dwells" means slow
  or stopped while on food.
- **The learning settings are not biology.** A lower entropy bonus was needed for the policy to hold
  still; at the base's 0.05 its noise alone looked like roaming. Real dwelling is a modulated state
  (serotonin through MOD-1; Flavell et al. 2013), which is what B.3 will model.

## Next

- **D.1's tracker item is met.** The roaming/dwelling readout passes its positive control.
- **D.1b, patch-leaving.**
  - **Cell:** smaller lawns, so leaving is needed.
  - **Readings:** the leaving rate, and whether leaving comes from roaming, against Scheer & Bargmann's
    lawn exits and entries.
  - **Also carried there:** the quality direction (Shtonda & Avery 2006, to be verified) and a clean
    depletion statistic, the chance that a dwelling bout ends as its spot's density falls.
- **B.3.**
  - **What it adds:** serotonin and PDF as global concentrations on the connectome, with release from
    NSM and the PDF-1 neurons, and gating at the cited receptor cells (MOD-1 in AIY, RIF and ASI;
    PDFR-1 as cited).
  - **Mutants:** raised mutants graded on the published directions. *tph-1* should roam more and
    *pdfr-1* less, as in Ji et al. 2021's patch assay.
  - **Cell:** this readout and this cell, with the connectome in place of the MLP.

## Artefacts

**The registration and the panel:**

- [launch.md](supporting/087-patchy-lawns/launch.md) and [preflight.json](supporting/087-patchy-lawns/preflight.json): the registration and its gate preflight.
- [panel.json](supporting/087-patchy-lawns/panel.json): the gates (with each seed's plateau), the reading, its verdict, and the readings beside it.
- [per-run.csv](supporting/087-patchy-lawns/per-run.csv): one row per evaluated run.

**The pilots:**

- [pilot1-states.json](supporting/087-patchy-lawns/pilot1-states.json), [pilot2-states.json](supporting/087-patchy-lawns/pilot2-states.json), [pilot3-states.json](supporting/087-patchy-lawns/pilot3-states.json), [pilot4-states.json](supporting/087-patchy-lawns/pilot4-states.json).

**The instrument's data** (`data/roaming_dwelling/`), with [PROVENANCE.md](../../../data/roaming_dwelling/PROVENANCE.md):

- `reference_hmm.json`: the authors' model.
- `scheer2023_windows.npz`: the real worms' windows.
- `calibration.json`: the failed line.
- `calibration_retry.json`: the passing model.
- `ji2021_readings.json`: Ji et al.'s directions.

**To reproduce:**

```bash
uv run python scripts/analysis/lawn_states.py --logs campaigns/d1-panel/logs \
  --seeds $(seq 2001 2016) --episodes 30 --out panel.json --csv per-run.csv
```

The campaign directories, the worktree's session records and the source pickle are archived off-repo.
