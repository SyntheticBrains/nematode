# C.3 — body-level validation: registration and launch

**Registered 2026-10-10, before any scored evaluation and before the control's training.** Change:
`add-body-validation`. The runs graded are C.1e's panel ([Logbook 085](../../085-body-wiring.md)); one
control is trained.

## The question

**Does the trained worm move like *C. elegans* through the body, in its posture and in its behaviour?**
Body checks confirm the generator and that the trained drive does not distort it. Behaviour readings
are emergent: speed, forward bouts, and Logbook 035's klinokinesis and weathervane curves. Nothing here
is a contrast between arms, apart from the one control.

## The arms

Each run's final weights are evaluated frozen for **30 held-out episodes** (run indices from 1,000,000,
never trained on), with posture capture at every sub-step (4 Hz) and behaviour capture at every step.
The three graded arms are evaluated again at 40 sub-steps for the half-step check.

| arm | config stem | seeds | role |
|---|---|---|---|
| wild type, PPO | `connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_body500_wt_ppo` | 1801–1864 | graded |
| chemical-only null, PPO | `..._body500_chemnull_ppo` | 1801–1864 | graded |
| MLP-PPO | `mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64_body500` | 1801–1864 | graded |
| wild type, frozen | `..._body500_wt_frozen` | 1801–1864 | the bias curves' floor |
| chemical-only null, frozen | `..._body500_chemnull_frozen` | 1801–1864 | the bias curves' floor |
| MLP, untrained | `mlpppo_..._body500`, each seed's initial weights | 1801–1864 | the MLP's floor: C.1e trained no frozen MLP |
| MLP, derivative sensing | `mlpppo_..._body500_derivative` | 1801–1816 | the weathervane's control (new) |
| MLP, derivative sensing, frozen | `mlpppo_..._body500_derivative_frozen` | 1801–1816 | the control's floor (new) |

416 evaluated runs. The control is C.1e's MLP config with `chemotaxis_mode: derivative` and nothing
else changed: no synthetic lateral sample, so any head-sweep is the body's own.

## Body checks

| check | pass | partial | otherwise |
|---|---|---|---|
| frequency (band crossings, ±0.31 κL) | 0.20–0.45 Hz | 0.10–0.60 Hz | fail |
| wavelength (adjacent-segment delays) | 0.50–0.80 body lengths | 0.40–1.00 | fail |
| variance the first four eigenworms capture, pooled over postures | ≥ 85% | ≥ 70% | fail |
| half-step agreement, 20 against 40 sub-steps | every instrument within 10% (reversal fraction 10% or 0.01) | | fails |

Bands from Wormlight's checkpoint 1; the eigenworm reference is Stephens et al. 2008, on the WormPose
basis vendored in `data/posture/` (real postures: 96.49%). A body rocking in place passes the eigenworm
clause; it grades the shapes, not the crawl.

## Behaviour readings

| reading | pass | partial | otherwise |
|---|---|---|---|
| speed, steps away from the 1 mm wall margin | 0.12–0.30 body lengths/s | 0.06–0.50 | fail |
| share of episodes with a forward run of ≥ 20 s | ≥ 80% | ≥ 50% | fail |
| klinokinesis, thresholded and threshold-free | REPRODUCED / PARTIAL / ABSENT, sign-only (035) | | |
| weathervane, thresholded and threshold-free | REPRODUCED / PARTIAL / ABSENT, sign-only (035) | | |

*(Note added 2026-10-10, after the evaluation: "sign-only" was wrong for one statistic. 035's harness
grades the thresholded klinokinesis ratio against a literature range, 1.5–3.0, and the other three on
sign alone. The harness's own verdicts are what Logbook 086 reports; none is re-read.)*

- **Edge rule (C.1d).** A banded reading whose grade differs between 20 and 40 sub-steps is reported
  as on the edge, with both grades. Speed sits at the band's floor (Logbook 083), so this is expected.
- **The bias curves** run Logbook 035's harness over each arm's 30 episodes per seed, the 1 mm wall
  margin, `θ_sharp` 0.45, its seeded bootstrap and its combined verdicts. A step through the body is 5
  worm-seconds, so per-step heading changes run larger than the point worm's; the threshold-free
  companions are why both are graded.
- **What learning added: each arm against its floor, paired by seed.** For each of the four bias
  statistics, the arm's per-seed value minus its floor's, oriented to the reference's direction, with
  its 80% bootstrap interval and one-sided Wilcoxon p. **A bias is attributed to learning only where
  that interval lies above zero.** The pairs are the wild type and its frozen runs, the null and its
  frozen runs, the MLP and its untrained policy, and the control and its frozen runs. Each floor's
  own verdicts are reported beside. An arm whose curve is REPRODUCED but whose difference from its
  floor is not above zero is reported as showing the bias without learning having added it.
- **Why paired, fixed after the cost pilot.** The change's design read each arm "beside" its floor,
  which should show no bias. On the pilot's two seeds the wild type's frozen floor already graded its
  weathervane PRESENT_PARTIAL, on the 15% of its transitions that survive the wall margin (the
  trained arms keep 77%). A floor that leans by chance or geometry would have voided the arm's reading
  outright under a binary rule, and it would not have measured what learning added. The pilot's
  floor reading is a calibration, not a C.3 reading.
- **The forward-bout reading is nearly met by construction**: an episode is 2,500 worm-seconds and
  trained reversal fractions are about 1%. It is graded, and that is said with it.

**Reported, not graded:** omega turns (the head line, 0.2 body lengths to the head, turning more than
135° net between consecutive zero-crossings of the head segment's curvature; their rate per worm-minute
and heading change), **the share of those turns whose posture reaches an omega's** (the third
eigenworm's peak magnitude across the swing above the real postures' 99th percentile of it, 10.6),
reversal fraction, and amplitude (radius in the first two eigenworms' plane,
against the real postures' median 5.07, 5th–95th percentile 2.30–7.87). Long reversals and pirouettes
cannot occur in this body.

**The omega-posture share was added after the cost pilot, and before any scored evaluation.** The
change's design expected omega turns to be rare. On the pilot's runs (seeds 1701–1702, disjoint from the
band), the trained wild type and null turned the head line past 135° within one swing 3.8 and 2.1 times
per worm-minute, with a median heading change near 150°, while their postures stayed ordinary: on one
wild-type run, the third eigenworm's peak across a turn had median 3.3 and 95th percentile 5.5, and no
turn reached the real postures' 99th percentile (10.6). The heading criterion, as Wormlight defined it,
counts steering pivots as well as omega turns. The companion separates the two. Nothing graded changed.

## The control

**Gate first.** The control's learning arm must beat its frozen floor, paired by seed on plateau
success: all 16 seeds present and the paired 80% interval above zero (C.1d's control gate; no seed
need be competent). If it does not, or a seed is missing, the control is **unreadable** and the weathervane's specificity is recorded as **untested**,
never inferred.

**The reading, if readable.** The MLP's weathervane slope minus the control's, paired by seed over
1801–1816, both the thresholded and the all-step slope, with its 80% bootstrap interval and one-sided
Wilcoxon p, reported as an effect size, as 035 did.

"Shows a weathervane" below means the harness's combined weathervane verdict is PRESENT or
PRESENT_PARTIAL: at least one of its two slopes significant in the right direction.

| outcome | what it licenses |
|---|---|
| interval above zero, and the control shows no weathervane | consistent with the MLP's weathervane needing the synthetic head-sweep; not separable from the control's weaker foraging (see Before launch) |
| interval includes zero, and the control shows a weathervane | the weathervane survives without the sweep: the body's own head swing carries it |
| interval includes zero, and the control shows none | neither arm shows a robust weathervane, or the two cannot be told apart; nothing about specificity |
| interval above zero, and the control shows a weathervane | the body carries a weathervane and the sweep strengthens it |
| interval below zero | the control curves more than the MLP; reported, no specificity claim |

The thresholded slope is the deciding one, as in 035; the all-step slope is reported beside it. If the
MLP arm itself shows no weathervane, the comparison has nothing to explain and is reported as such.

The connectome arms run klinotaxis sensing too; the control is the MLP's, so its reading speaks to the
MLP arm and, by the shared sensor, bounds what the connectome arms' weathervane can be attributed to.
It is not a test of the connectome.

## Grading, and what is not done

Every arm gets a grade per check and per reading. Differences between arms are described, not tested.
There is no `unreadable` state for the graded arms: an instrument that cannot be read, for want of
undulating steps or forward bouts, is recorded as unread. Logbook 036's thermotaxis curves are out of
scope (M.9).

## Before launch

- **Evaluation cost**, from a pilot on C.1e's pilot runs (seeds 1701–1702, the four connectome arms,
  configured as here, [evaluation-pilot.json](evaluation-pilot.json)): 8 runs in 106 s on 8 workers
  beside a training campaign, about 100 s for a graded run's two passes and 50 s for a floor's one.
  The 416 runs are about **35 minutes** on 16 workers. A later smoke run on the same seeds added the
  MLP's untrained floor and the paired floor comparison, and both read. Its readings are a calibration of the instruments,
  not readings of C.3: every instrument and the bias-curve harness read, the half-step check agreed on
  both graded arms, and speed read 0.11–0.12 body lengths/s, at the pass band's floor, as C.1d found.
- **The control's gate**, from a pilot of the control on seeds 1701–1704, learning and frozen, 3,000
  episodes as here ([control-preflight.json](control-preflight.json),
  `gate_preflight.py --panel body_validation`): **`readable`**, `launch: true`. Its plateaus are 0.0,
  24.7, 25.5 and 22.1% against 0% floors: C.1d's control gate reads `fallback`, above its floor but
  with no seed competent (≥ 30%). The registered gate is the floor, so this passes, and the pilot
  says the 16-seed control will very likely be readable.
- **What the pilot says about the control, before it runs.** Through the body, the MLP without the
  head-sweep forages far worse than with it (about 18% against C.1e's MLP at 70.8%), where on the point
  worm Logbook 035's derivative control reached 98.7–100%. So the reading is asymmetric, and that is
  fixed now: **a weathervane that survives in the control is clean evidence the body carries it; a
  weathervane that collapses cannot be told apart from the control foraging worse**, and is reported
  as consistent with the sweep carrying it, not as establishing it. The control's own foraging gap is
  reported beside, as a description of how much the sweep does through the body.
- **Control training cost**, from the pilot: 8 runs took 41 minutes on 8 workers (35–40 minutes per
  run). 32 runs on 16 workers are about **1.5 hours**.
- **Readiness.** The graded runs exist; nothing about them is chosen after reading them. The pilots'
  seeds (1701–1704) are disjoint from the registered band.

## Launch

From a worktree at this commit:

```bash
git -C ../nematode-c3 checkout --detach <this commit>
cd ../nematode-c3
M=configs/scenarios/foraging/mlpppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_ppo_w64_body500_derivative
uv run python scripts/run_campaign.py --config ${M}.yml --config ${M}_frozen.yml \
  --seeds 1801-1816 --runs 3000 --workers 16 --output-dir ../nematode/campaigns/c3-control \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log

# Then, from the main checkout:
rsync -a --ignore-existing ../nematode-c3/experiments/ experiments/
rsync -a --ignore-existing ../nematode-c3/exports/ exports/
uv run python scripts/analysis/body_validation.py --logs campaigns/c1e-panel/logs \
  --logs campaigns/c3-control/logs --out-dir build/c3 --out validation.json --workers 16
```

## Retention (A.0)

Committed: this record, the pilots' summaries, and `validation.json`. Archived off-repo: the control
campaign's raw logs, the evaluation captures, and the worktree's session records.
