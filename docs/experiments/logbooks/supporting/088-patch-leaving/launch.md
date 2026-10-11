# D.1b — patch-leaving: registration and launch

**Registered 2026-10-11, before any panel run.** Change: `add-patch-leaving`. Three pilot cells, all on
seeds disjoint from the panel's, are committed beside this record.

## The question

**Does the learner's patch-leaving come out of its roaming state, as a worm's does?** Scheer &
Bargmann 2023 found that worms leave food from the roaming state. Read with the same classifier at
the point worm's step, their wild type's roaming in the 30 to 180 s before an exit runs **+0.288**
above its overall on-lawn roaming (46.3% against 17.5%; `data/roaming_dwelling/scheer2023_leaving.json`).

## The cell

D.1's cell, with only the lawns changed:

- **Learner:** MLP-PPO reading `internal_state`.
- **Energy:** a movement cost of 0.013 per mm, from reward and satiety alike; intake 2% of a cell's
  density per step.
- **Start:** on a lawn.
- **Learning settings:** `entropy_coef` 0.005, and no state-shaping reward.
- **Episodes:** 720 steps, 3,000 per run.
- **Lawns: 4 lawns of 1.5 mm radius** (9 cells each), at least 2 mm apart.

| arm | config stem | role |
|---|---|---|
| learner | `mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_leaving15x4_internal` | |
| floor | `..._lawns_leaving15x4_internal_frozen` | its untrained policy |

**Seeds 2101–2116 (16), 32 runs.** Each run's final weights are evaluated frozen for **30 held-out
episodes**, read with D.1's instrument.

## The reading

**Each read seed's band enrichment:** its roaming share in on-lawn windows 30 to 180 s before each
exit, minus its overall on-lawn roaming share.

- **The exit rule:** an exit is the first fully off-lawn window after an on-lawn run, so a flicker
  at the edge does not count.
- **Why the band:** it leaves out the last 30 s, since leaving means moving and moving reads as
  roaming.
- **The test:** one-sample (mean, 80% bootstrap interval, two-sided Wilcoxon q against zero),
  classified at the **minimum of 0.10** by `mc.classify`.

| state | verdict | what it licenses |
|---|---|---|
| `move_wt` | **leaves from roaming** | the learner's roaming runs at least 0.10 above its own baseline in the minutes before it leaves, as a worm's does |
| `move_null` | **leaves from dwelling** | its roaming runs at least 0.10 below baseline before leaving |
| `below` | **coupling below the minimum** | a significant coupling smaller than 0.10 |
| `no_move` | **no coupling at the minimum** | the interval sits inside ±0.10 and includes zero: the learner's leaving does not come out of a roaming state at the size real worms show |
| `unresolved` | **unresolved at this sensitivity** | none; the achieved interval is reported as the bound |

**Gates, in order:**

1. **The learner learns.** Its intake plateau beats its floor's, paired by seed, with the 80% interval
   above zero. If not, the panel is unreadable.
2. **A seed is read only if it leaves at least 10 times** over its 30 evaluation episodes.
3. **A seed is read only if it takes both states**, with on-lawn roaming between 5% and 95%.

Seeds failing gate 2 or 3 are reported and left out. **Fewer than 12 read seeds leave the panel
unreadable.**

**The minimum.** Two-thirds of the pilot's mean (+0.020) is 0.013, below the floor of **0.10**, so
the minimum is the floor. The floor is judged: a coupling of 10 points is about a third of what real
worms show by the same classifier (+0.288), and about half by the authors' labels (+0.227).

**What the pilot predicts.** The pilot's learners show no coupling: band enrichments of +0.07, +0.01,
−0.03 and +0.03. The likeliest verdict is therefore **no coupling at the minimum**. It is stated now,
before the panel runs, so that it cannot be mistaken for a finding read into the data afterwards.

**Reported beside, never read as verdicts:**

- the leaving rate, against real worms' 0.21 exits per on-lawn hour;
- the last-window enrichment, which is partly mechanical;
- the dwelling-end statistic: roaming after dwelling where the spot is grazed, against fresh;
- the floor's leaving rate and states;
- intake, and complete bouts;
- the six-lawn cell's pilot, as description;
- the achieved spread and MDE, beside the minimum, never used to re-read it.

## Sizing

From the second pilot's four-lawn arm ([pilot2-4lawns-states.json](pilot2-4lawns-states.json)):

- the mean is +0.020, with sd 0.042;
- the MDE at 16 seeds is **0.026**, well below the minimum, so the panel runs the 16-seed floor.

## The pilots that shaped the cell

| pilot | seeds | cell | what it showed | decision |
|---|---|---|---|---|
| 1 | 9201–9204 | 8 lawns of 1.0 mm | the learner learned and left, but 3 of 4 seeds never roamed: it crept between lawns | the both-states gate; 1.5 mm lawns |
| 2 | 9205–9208 | 4 lawns of 1.5 mm | every gate passed on every seed; enrichment near 0 | **registered** |
| 2 | 9205–9208 | 6 lawns of 1.5 mm | the gates passed except one seed's states; enrichment near 0 | described beside |

## Before launch

- **Gate preflight** on the four-lawn pilot, configured as here ([preflight.json](preflight.json)):
  - learning: +1.21 [+0.98, +1.44], passes;
  - exits: 52–72 per seed, at least 10 on all four;
  - roaming: 42–93%, within the bounds on all four.
- **Cost:** 16 runs took about 9 minutes on 16 workers, so 32 runs take about **20 minutes**,
  plus a few minutes of evaluation.
- **Seeds:** the panel's (2101–2116) are disjoint from every pilot's (9101–9208) and from D.1's
  panel (2001–2016).

## Launch

From a worktree at this commit:

```bash
git -C ../nematode-d1 checkout --detach <this commit>
cd ../nematode-d1
L=configs/scenarios/foraging/mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_leaving15x4
uv run python scripts/run_campaign.py --config ${L}_internal.yml --config ${L}_internal_frozen.yml \
  --seeds 2101-2116 --runs 3000 --workers 16 --output-dir ../nematode/campaigns/d1b-panel \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# Then, from the main checkout:
rsync -a --ignore-existing ../nematode-d1/experiments/ experiments/
rsync -a --ignore-existing ../nematode-d1/exports/ exports/
uv run python scripts/analysis/lawn_states.py --logs campaigns/d1b-panel/logs --cell leaving15x4 \
  --seeds $(seq 2101 2116) --episodes 30 --leaving-minimum 0.10 --min-exits 10 --min-seeds 12 \
  --out panel.json --csv per-run.csv
```

## Retention (A.0)

**Committed:** this record, `preflight.json`, the pilots' readings, and the panel's `panel.json` and
`per-run.csv`. **Archived off-repo:** the campaign directories and the worktree's session records.
