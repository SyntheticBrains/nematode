## Overview

D.1b tests whether a learner's patch-leaving comes out of its roaming state, as Scheer & Bargmann
2023 found in real worms. It reuses D.1's lawn model, instrument and learner. It changes the lawn
geometry, so leaving pays, and adds leaving readings. The user decided its primary reading and its
scope on 2026-10-11.

## Decisions

### Decision A: The cell

D.1's cell is kept: the point worm's MLP-PPO reading `internal_state`, a movement cost, intake 2% per
step, a start on a lawn, `entropy_coef` 0.005, and no state-shaping reward. Only the lawns change.

- **Size and count.** Lawns are smaller and more numerous, so one lawn holds less than an episode's
  food and leaving to another pays.
- **The arithmetic.** At radius 1.0 mm a lawn has 5 cells, holding about 3.3 units grazed to the
  point where staying stops paying, against about 3.1 needed to survive an episode. At 1.5 mm it has
  9 cells and about 5.9 units.
- **The pilot decides,** on disjoint seeds, choosing the cell where the learner both learns and
  leaves.

Quality is fixed at 1.0. The quality direction (Shtonda & Avery 2006) is deferred to a later cell
and verified against its source first.

### Decision B: The readings

From the evaluation capture, with D.1's windows and calibrated classifier:

- **Lawn exits.** A step off a lawn that stays off for at least one full 10-second window. A worm
  on a small lawn's edge flickers on and off, and a flicker is not an exit. Real tracks use the same
  rule: an off-lawn window after an in-lawn run.
- **Pre-exit roaming, over minutes.** The roaming share of the on-lawn windows in the band **30 to
  180 s before each exit**, which leaves out the last 30 s. Leaving means moving, and the classifier
  reads movement as roaming, so the final windows are mostly roaming for any worm that walks off.
  Scheer & Bargmann's claim is about minutes: real roaming stays well above baseline far from the
  exit (55% over 60 s, 44% over 3 min, against 10.7% overall).
- **Enrichment.** A run's pre-exit roaming share in that band, minus its overall on-lawn roaming
  share.
- **The last-window enrichment** is reported beside it, labelled as partly mechanical.
- **Leaving rate.** Exits per worm-hour on lawns.
- **The dwelling-end statistic.** For each on-lawn dwelling window, whether the next window is
  roaming, split by the density of the cell under the worm (below or at least half). It is
  reported per run as the difference between grazed and fresh. This replaces D.1's
  grazed-against-fresh roaming share, which was confounded because dwelling is what grazes a cell.

### Decision C: The real reference

Scheer & Bargmann's 1,586 wild-type animals, from the deposit already read for D.1:

- **Method.** Lawn exits are read from the in-lawn runs with the same exit rule, aligned to the
  same 10-second windows. States are read
  two ways: with D.1's calibrated classifier on the windows measured at the 5 s step (the matched
  reference), and with the authors' own labels (beside).
- **Vendored.** `scheer2023_leaving.json`, with provenance:
  - the 30–180 s band enrichment, the registered statistic;
  - the last-window enrichment;
  - the leaving rate;
  - the pre-exit roaming share at 10, 30, 60 and 180 s.

### Decision D: The registered reading

**Primary.** The learner's 30–180 s band pre-exit roaming enrichment per seed, a one-sample reading (mean, 80%
bootstrap interval, two-sided Wilcoxon q against zero). It is classified by `mc.classify` at a
minimum fixed from the pilot: 2/3 of the pilot's mean, floored at **0.10**. The floor is judged: a
learner whose roaming in the minutes before leaving runs 10 points above its own baseline has a
coupling about half the size real worms show. Their 30–180 s band runs at 32.3% roaming against
10.7% overall, about 22 points above baseline, by the authors' labels. It is never re-read after
the data.

- **Its direction is the real worms'.** Leaving comes out of roaming, so the enrichment is positive.
- **Every state is named,** including the opposite direction.

**Gates, read in order:**

1. **The learner learns.** Its intake plateau beats its untrained floor's, paired by seed.
2. **The learner leaves.** Every seed has at least a minimum number of exits over its evaluation
   episodes, set in the registration from the pilot. A seed below it is reported, and too few
   readable seeds leave the panel unreadable.

**Beside, never read as verdicts:**

- the leaving rate against the real worms' 0.63 per hour;
- the enrichment against the real matched reference;
- the dwelling-end statistic;
- the untrained floor's leaving rate. The floor roams constantly, so its enrichment is not a
  comparator.

## Risks

- **The learner may not leave.** It may stay and starve, or graze one lawn out. The pilot exists to
  find a cell where it leaves. If no cell in a small sweep works, D.1b closes with that reason.
- **Leaving is partly coupled to roaming by mechanics.** To leave, a worm has to move, and a moving
  worm reads as roaming. That is why the registered band leaves out the last 30 s before each exit,
  why the real reference is read the same way, and why the last-window enrichment is reported only
  as partly mechanical.
