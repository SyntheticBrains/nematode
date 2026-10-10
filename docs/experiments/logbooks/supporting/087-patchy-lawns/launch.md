# D.1 — roaming and dwelling on patchy lawns: registration and launch

**Registered 2026-10-11, before any panel run.** Change: `add-patchy-lawns`. Four pilots, all on seeds
disjoint from the panel's, shaped the cell. Their readings are committed beside this record.

## The question

**Does a learner that reads its own satiety dwell on lawns, where its untrained policy does not?** This
is the positive control for D.1's roaming/dwelling readout, which B.3's serotonin/PDF field will read
next. B.3 starts only on a readable result.

## The cell

The point worm's MLP-PPO, width 64, with the hard-food cell's sensing: klinotaxis, the adaptive
fold-change sensor, the Fick field. Point food is replaced by lawns:

- **Lawns:** 4 disc lawns of 2.5 mm radius in a 20 mm arena, 1 mm cells, at least 2 mm apart.
- **Start:** the worm starts on the first lawn.
- **Intake:** 2% of a cell's density per step, independent of speed, and no regrowth.
- **Energy**, in intake value:
  - eating pays 10 reward per unit, and restores 0.3 of maximum satiety per unit;
  - each millimetre moved costs 0.013 of the same, from reward and satiety alike;
  - basal satiety decay is 0.8 per step, from 300.
  - the base config's other terms are carried over unchanged: a step penalty of 0.005, a starvation
    penalty of 10, and a wall-collision penalty of 0.02.
- **Episodes:** 720 steps (an hour of worm time), 3,000 episodes per run.
- **Actions:** signed speed and turns up to π per step.
- **Learning settings:** `entropy_coef` 0.005. The stuck-position, anti-dithering, exploration and
  distance rewards are zero.

| arm | config stem | role |
|---|---|---|
| learner | `mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal` | reads `internal_state` |
| floor | `..._lawns_internal_frozen` | its untrained policy |
| without internal state | `..._lawns_blind` | beside |

**Seeds 2001–2016 (16), 48 runs.**

## The instrument

Each run's final weights (the floor's untrained policy) are evaluated frozen for **30 held-out
episodes** with behaviour capture. Positions are taken once per 5 s step and cut into 10-second
windows. A window is on a lawn only when all three of its positions are.

States come from the model calibrated on Scheer & Bargmann 2023's 1,586 wild-type animals
(`data/roaming_dwelling/calibration_retry.json`): a two-state Gaussian-emission HMM on log speed and
angular speed, decoded within each on-lawn run. On held-out real animals it agrees with the authors'
labels at **κ = 0.632**, close to its registered gate of 0.6. It calls more roaming than the authors
do (17.4% of on-lawn windows against 10.6%), so absolute fractions are described and never matched.

## The reading

**The learner's on-lawn dwelling share minus its floor's**, paired by seed. It is classified at the
**minimum of 0.394** by `mc.classify`, from the mean, the 80% bootstrap interval and the two-sided
Wilcoxon q:

| state | verdict | what it licenses |
|---|---|---|
| `move_wt` | **dwells** | the learner takes the dwelling state on lawns, at least 0.394 more of its on-lawn time than its untrained policy; B.3 may read this readout |
| `move_null` | **dwells less than its floor** | the learner dwells less than its untrained policy |
| `below` | **dwelling below the minimum** | a significant difference smaller than 0.394 |
| `no_move` | **no dwelling at the minimum** | the interval sits inside ±0.394 and includes zero |
| `unresolved` | **unresolved at this sensitivity** | none; the achieved interval is reported as the bound |

**Gate first.** The learner's intake plateau, the final quarter of its training episodes, must beat
its floor's, paired by seed, with the 80% interval above zero. If it does not, the panel is
**unreadable**.

**Reported beside, never read as a verdict:**

- the same dwelling contrast for the arm without `internal_state`, and the learner against it;
- each arm's seeds with complete bouts of both states, and their durations;
- roaming where the cell under the worm is grazed (below half density) against fresh, per seed;
- the dwelling share against real worms' (the classifier's 17.4% roaming on held-out real animals);
- each run's learned action noise;
- the achieved spread and MDE, beside the registered minimum, never used to re-read it.

**A worm that stops is read as dwelling.** A window with no movement has angular speed 0, and the
calibrated model places it in dwelling, as real dwelling worms often pause. "Dwells" therefore means
slow or stopped while on food.

**A positive does not mean more than it says.** "Dwells" says the learner takes a slow, turning state
on food that its untrained policy never takes. It does not say the states are worm-like in their
timing or their dependence on depletion. Those are the readings beside it. In the fourth pilot, bouts
of both states appeared on one seed of four, and the depletion direction was mixed.

## Sizing

The fourth pilot (seeds 9113–9116, configured as here, [pilot4-states.json](pilot4-states.json)):

- **Reference:** the learner's dwelling share minus its floor's per seed is +0.254, +0.874, +0.958
  and +0.277, a mean of **+0.591** with sd 0.377.
- **Minimum:** 2/3 of the reference, **0.394**.
- **Panel size:** the smallest n whose MDE (`2.487 × sd / √n`) reaches the minimum is below the
  16-seed floor, so the panel runs **16**, where the MDE is **0.234**.

## The pilots that shaped the cell

Each pilot's change was decided with the user and recorded in the design:

| pilot | seeds | cell | what it showed | what changed |
|---|---|---|---|---|
| 1 | 9101–9104 | intake 10% per step, movement free | learners tripled intake but roamed 98–99% on lawns | movement costs energy; intake 2% per step |
| 2 | 9105–9108 | movement costed, worm starting off food | the learner stood still and starved | the worm starts on a lawn |
| 3 | 9109–9112 | starting on a lawn | the gate passed, but speed noise near the states' boundary and random turns | `entropy_coef` 0.05 → 0.005 |
| 4 | 9113–9116 | as registered | the gate passed (+1.04 [+0.46, +1.62]); dwelling appeared | none: this registration |

A failed first placement attempt (lawns that would not fit) is kept as
`campaigns/d1-pilot-failed-placement`.

## Before launch

- **Gate preflight**, read on the fourth pilot's runs, configured as here
  ([preflight.json](preflight.json)): the learner passes, +1.04 intake [+0.46, +1.62]. The arm without
  internal state does not (+0.22 [−0.26, +0.70]); it is beside, not gated.
- **Cost:**
  - training, from the fourth pilot: 12 runs in about 8 minutes on 12 workers, so 48 runs on 16
    workers take about **30 minutes**;
  - evaluation: a few minutes.
- **Readiness.** The panel's seeds (2001–2016) are disjoint from every pilot's (9101–9116).

## Launch

From a worktree at this commit:

```bash
git -C ../nematode-d1 checkout --detach <this commit>
cd ../nematode-d1
L=configs/scenarios/foraging/mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns
uv run python scripts/run_campaign.py --config ${L}_internal.yml --config ${L}_blind.yml \
  --config ${L}_internal_frozen.yml --seeds 2001-2016 --runs 3000 --workers 16 \
  --output-dir ../nematode/campaigns/d1-panel \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# Then, from the main checkout:
rsync -a --ignore-existing ../nematode-d1/experiments/ experiments/
rsync -a --ignore-existing ../nematode-d1/exports/ exports/
uv run python scripts/analysis/lawn_states.py --logs campaigns/d1-panel/logs \
  --seeds $(seq 2001 2016) --episodes 30 --out panel.json
```

## Retention (A.0)

**Committed:** this record, `preflight.json`, the pilots' readings, and the panel's `panel.json`.
**Archived off-repo:** the campaign directories, the worktree's session records, and Scheer &
Bargmann's source pickle.
