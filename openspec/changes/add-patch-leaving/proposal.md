## Why

D.1 (Logbook 087) showed that a learner on patchy lawns dwells where its untrained policy never does.
It could not test patch-leaving, because one of its lawns held about three times an episode's food.
Scheer & Bargmann 2023's central result is that worms leave food out of the roaming state. In their
1,586 wild-type animals:

| reading | value |
|---|---|
| roaming in the 10 s before a lawn exit | 65% |
| roaming overall on lawns | 10.7% |
| lawn exits | 0.63 per animal-hour |

D.1b asks whether a learner's leaving is coupled to its roaming the same way. It also carries the
depletion statistic D.1 could not read cleanly: whether a dwelling bout ends sooner as its spot runs
out. D.1b runs before B.3, on the MLP.

## What Changes

- **A leaving cell**: D.1's cell with smaller lawns, so that one lawn cannot feed an episode and
  leaving pays. The pilot sets their size and count.
- **A matched real reference**: the pre-exit roaming enrichment on Scheer & Bargmann's real tracks,
  read with D.1's calibrated classifier at the point worm's step, beside the authors' own labels,
  vendored with provenance.
- **Leaving readings** in the lawn analysis:
  - lawn exits from the behaviour capture;
  - the roaming share before each exit, against the run's overall on-lawn roaming share;
  - the leaving rate;
  - the chance that a dwelling window is followed by roaming, by the density of the spot.
- **A registered positive control**: the learner's pre-exit roaming enrichment, graded against the
  real worms' direction, behind gates that it learns and that it leaves at all.

## Capabilities

**Modified**:

- `realworm-behavioural-validation`: lawn exits, pre-exit roaming enrichment, and the dwelling-end
  depletion statistic.

## Impact

- `scripts/analysis/lawn_states.py`: the leaving readings and the registered reading.
- `scripts/analysis/patch_leaving_reference.py`: new; the real reference from the deposit.
- `data/roaming_dwelling/scheer2023_leaving.json`: new, with provenance.
- `scripts/campaigns/generate_lawn_configs.py`: the leaving cell's configs.
- Tests, Logbook 088, tracker D.1b, roadmap.
