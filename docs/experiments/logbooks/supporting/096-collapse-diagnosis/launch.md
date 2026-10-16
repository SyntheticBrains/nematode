# M.1(a) — the collapse instrument and the diagnosis: registration and launch

**Registered 2026-10-16, before any log of these runs is read.** Change: `add-collapse-instrument`.
Tracker `phase9-tracking` M.1(a); roadmap v4.5 D25 and D26.

## The question

Logbook 092's thermal body panel lost eight learners late in training: the wild type on 2 of 32 seeds
and the gap-only null on 6 of 32, every one after episode 4,000. Late collapse of small PPO learners is
documented at this scale, and its candidate mechanisms leave different traces before the success rate
falls. **What happens inside a body learner in the 1,500 episodes before it collapses, and does it
happen in its intact twin?** This is a diagnosis. It chooses no remedy: M.1(b) takes its remedy order
from what is indicated here.

## The runs

Each run is a re-run of a committed run with read-only telemetry. Logbook 092's configs, Cook 2019, the
body at target 20, 6,000 episodes, nothing changed but the two flags.

| seed | collapsed on 092's panel | intact twin on the same seed |
|---|---|---|
| 2313, 2328 | wild type (`..._body_wt_ppo_t20`) | gap-only null (`..._body_gaponly_ppo_t20`) |
| 2304, 2305, 2307, 2308, 2319, 2330 | gap-only null | wild type |

A pair shares a seed, a task draw and every initial chemical weight, so each collapse is read against
its own intact twin. That is **16 runs**. The chemical-only null's five collapses are left out: no Phase 9
block reads that null on the gap question, and its seeds bring no intact twin on the same wiring pair.

**The MLP control's 8 retry seeds** (9721–9728, `mlpppo_small_continuous2d_thermal_klinotaxis_body_ppo_t20`)
are re-run the same way for the MLP's collapse incidence at 6,000 episodes, which D25's remedy rule needs.
They have no intact twin and never join the cause stacks. **24 runs** in all.

Every run carries `--update-telemetry --checkpoint-every 250` beside the output controls.

## Identity

A re-run licenses the diagnosis only if its final-quarter plateau equals the committed value to the
precision it was printed at: 092's `per-seed.csv` (`gap_only_wt_plateau`, `gap_only_rn_plateau`, six
decimals) and `retry/per_seed.csv` (`mlp_plateau_pct`, three decimals). The step-level logs are archived
off-repo, so the plateau is the field compared; every other field is **uncompared**. Telemetry and
checkpoints were shown not to change a run: on and off, the outputs, every weight and optimiser tensor,
the results CSV and the next draw of every generator are identical in the change's tests, and again on
these runs' own configs before launch ([probe.json](probe.json): 12 episodes each of the gap-only null on 2304, the wild type on 2313 and the MLP on 9721, with the flags and without — every `Run:` line, the results CSV and every tensor in `final.pt` identical, the checkpoints on schedule).

If a re-run fails the identity check, the diagnosis runs on the re-runs it has and says so, and the
committed incidence is not re-read.

## The statistic

**Collapse** (D25, `scripts/analysis/collapse.py`): the peak at episode *t* is the highest trailing-200
success rate seen by *t*; onset is the first episode where the trailing-100 rate falls below half the
peak and stays below for at least 300 episodes; censored at the horizon. Each collapsed re-run's onset
is its own, under this definition. A pair leaves the stacks, named, when its collapsed run does not
collapse under it, when its intact twin does, or when either run's telemetry stops before the window
ends. Both wild-type collapses came in 092's last 500-episode block, so the 300-episode persistence may
censor them; with six pairs the rule below reads at least 5 collapsed and none intact.

## The causes, fixed before any log is read

Every telemetry statistic is taken per episode (the mean of that episode's updates, carried forward
between), smoothed over 100 episodes, and aligned on the collapse's onset.

**The window** is the 1,500 episodes ending 100 before onset: the success rate crosses half its peak
some way into the fall and the statistics are smoothed over 100 episodes, so the last hundred episodes
already carry the fall.

**Each run is read as its change from its own baseline**, its smoothed mean over the 1,500 episodes
before the window. The wirings differ from the first episode (on the re-runs' own configs, the
saturated fraction is about 0.05–0.07 on the wild type and 0.13–0.20 on the gap-only null), and six
collapsed runs are gap-only while six intact twins are wild type, so a level would read the wiring as
a cause.

**The envelope.** Each intact twin is aligned on its collapsed twin's onset; the eight give, for each
statistic and each window episode, a mean and a sample standard deviation (`ddof = 1`); the envelope is
mean ± 2 sd, its half-width never below 10% of the intact runs' mean absolute baseline nor below the
statistic's absolute floor (0.01 for the fractions, 1e-4 for KL, 0.05 for explained variance, 1 for
`srank`). A run **departs** on a statistic when its change sits outside the envelope in the predicted
direction for at least 200 consecutive window episodes; the stretch's start, as a lead before onset, is
reported. Each intact twin is tested against the envelope of the other seven.

Each cause is a conjunction of departures (`collapse_diagnosis.CAUSES`):

| cause | the pattern that identifies it |
|---|---|
| 1. weight growth → saturation → rank loss | weight norm (all but the critic) up, saturated fraction up, `srank` down |
| 2. a fixed step against a vanishing signal | raw advantage std down, gradient norm departing in neither direction |
| 3. stale optimiser second moments | pre-clip gradient norm up and approximate KL up, the gradient's departure first (the sustained form) |
| 4. critic failure | explained variance down and value loss up; the critic-to-actor gradient ratio beside |
| 5. variance collapse | `log_std` down |
| 6. mean-head saturation | the `\|u\| > 2` fraction up, `log_std` departing in neither direction |

Entropy, clip fraction, KL and the critic-to-actor gradient ratio around onset are reported beside as
context, never as departures. **The transient form of cause 3**, a gradient-norm jump followed within
tens of updates by a KL spike, cannot survive the smoothing and the 200-episode rule, so it is counted
beside and never read as the cause: per update, unsmoothed, in the window, a gradient-norm spike
followed within 20 updates by a KL spike, a spike being 5 scaled MADs above the run's own baseline
median. `srank` is computed on a fixed 64 rows of each update; a shorter batch's is not read.

**The rule.** A cause is **indicated** when it departs in at least 6 of the 8 collapsed runs and in at
most 1 of the 8 intact twins; **not indicated** when it departs in at most 2 collapsed runs; otherwise
**mixed**. If fewer than 8 pairs remain, the same fractions apply (three quarters, one eighth, one
quarter). More than one cause may be indicated. A `mixed` reading is a result: M.1(b) then tries D25's
remedy list in its own order.

**The MLP's incidence** is reported with its Wilson interval beside 092's committed panel incidence
(wild type 2/32, gap-only null 6/32, under 092's rule). The 16 connectome re-runs were chosen half on
collapse, so their incidence is reported only as a check of the definition against 092's rule.

**Deferred, with a destination: incidence on the committed 085 and 092 panels** with its sensitivity to
the definition, which the tracker asks of M.1(a). No committed file holds those panels' per-episode
series; their logs are archived off-repo. It runs once they are restored, and is reported in Logbook
096 or its dated addendum:

```bash
uv run python scripts/analysis/collapse.py --logs <archive>/m9-panel/logs \
  --stem connectomeppo_small_continuous2d_thermal_klinotaxis_body_wt_ppo_t20 \
  --stem connectomeppo_small_continuous2d_thermal_klinotaxis_body_gaponly_ppo_t20 \
  --stem connectomeppo_small_continuous2d_thermal_klinotaxis_body_chemnull_ppo_t20 \
  --out docs/experiments/logbooks/supporting/096-collapse-diagnosis/incidence-092.json
```

and the same over Logbook 085's panel logs, with that panel's arms.

*(Computed 2026-10-17, from the archive, before any re-run: [incidence-092.json](incidence-092.json)
and [incidence-085.json](incidence-085.json); the archive's location is the maintainer's and is not
recorded here.)* Under the registered definition, against 092's block rule:

| panel | arm | D25 collapsed | 95% Wilson | 092's rule | grid range |
|---|---|---|---|---|---|
| 092 (6,000 episodes) | wild type | 7 / 32 | [0.11, 0.39] | 2 | 6–9 |
| 092 | gap-only null | 9 / 32 | [0.16, 0.45] | 6 | 7–20 |
| 092 | chemical-only null | 11 / 32 | [0.20, 0.52] | 5 | 8–17 |
| 085 (3,000 episodes) | wild type | 35 / 64 | [0.43, 0.66] | 6 | 14–50 |
| 085 | chemical-only null | 32 / 64 | [0.38, 0.62] | 3 | 16–45 |
| 085 | MLP | 13 / 64 | [0.12, 0.32] | 3 | 3–31 |

**The definition counts transient dips.** Most runs it flags on 085 fall early (23 of the wild type's
35 before episode 1,500) and recover (22 of 35); many flagged 092 runs end high (final rate 0.77–0.89).
Seven of the eight known collapsers' onsets fall on their late collapse (episodes 3,858–5,554); the
gap-only null on 2305 is flagged at an early dip (episode 1,721) it recovered from before its late fall,
so the diagnosis would align that pair on the wrong event. This is reported for the maintainer's
decision on D25 before this campaign launches; nothing here is amended by it.

## Before launch

- **No gate preflight.** This is a diagnosis licensed by an identity check, not a registered contrast:
  it reads no floor or saturation gate. The configs and seeds are 092's own.
- **Cost.** 092 measured the body thermal cell at 6,000 episodes at about 84 minutes a run at 16
  workers. The 16 connectome runs are one wave and the 8 MLP runs a second: **about 3 hours**; the probe's telemetry overhead was within run-to-run noise (−5% to +1% on 12-episode runs).
  The campaign runs with the machine to itself: 16 more workers beside another campaign on 18 cores
  would double both campaigns' wall-clock time.
- **Order.** The tracker runs G.0's arms before these re-runs, so this campaign launches when G.0's
  panel has finished.
- **Disk.** Telemetry is one line per update, which fires every 256 steps and at every episode end
  (one or two a body episode, a few tens of MB a run); checkpoints every 250 episodes are 24 files a
  run. Both sit in each run's export, archived off-repo.

## Readiness review

`/nematode-review-spec add-collapse-instrument`, 2026-10-16, before this record was committed. Every
finding was fixed before launch:

- **Blocking**: the envelope mixed wirings that differ from the first episode (fixed by reading each
  run as its change from its own baseline); `srank` tracked batch size, which tracks episode length
  (fixed by a fixed 64-row computation, logged, with shorter batches not read).
- **Should fix**: the window ended at onset, inside the fall (now 100 episodes before it); the
  transient form of cause 3 was undetectable (counted beside, per update); zero-width envelopes (a
  floor and a sample sd); intact twins that collapse, and truncated telemetry, are now excluded and
  named; the stacks and the context around onset are written out; the launch commands are
  zsh-safe and create the worktree; the 085/092 incidence is deferred with a destination; the
  selected re-runs' incidence is labelled as a check, with 092's panel incidence beside the MLP's.
- **Minor**: the MLP reads its head features before the optimiser step; the ReLU saturation rule is
  named in the spec; the on/off identity is checked on these runs' own configs.

## Launch

From a fresh worktree at this commit, so the main checkout's work cannot touch the running code or
configs. The commands are bash; the flags are written out, not held in a variable, so they work
pasted into zsh too.

```bash
git worktree add --detach ../nematode-m1 <this commit>
cd ../nematode-m1
uv sync --extra cpu --extra torch --extra pixel --extra analysis --extra qpu
P=configs/scenarios/thermal_foraging/connectomeppo_small_continuous2d_thermal_klinotaxis_body
uv run python scripts/run_campaign.py --config ${P}_wt_ppo_t20.yml --config ${P}_gaponly_ppo_t20.yml \
  --seeds 2304,2305,2307,2308,2313,2319,2328,2330 --runs 6000 --workers 16 \
  --output-dir ../nematode/campaigns/m1-diagnosis \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log \
  --update-telemetry --checkpoint-every 250
uv run python scripts/run_campaign.py \
  --config configs/scenarios/thermal_foraging/mlpppo_small_continuous2d_thermal_klinotaxis_body_ppo_t20.yml \
  --seeds 9721-9728 --runs 6000 --workers 16 --output-dir ../nematode/campaigns/m1-mlp \
  -- --theme headless --track-experiment --no-detailed-export --no-file-log \
  --update-telemetry --checkpoint-every 250

# Then, from the main checkout: copy the worktree's records in, and read.
rsync -a --ignore-existing ../nematode-m1/experiments/ experiments/
rsync -a --ignore-existing ../nematode-m1/exports/ exports/
D=docs/experiments/logbooks/supporting/096-collapse-diagnosis
uv run python scripts/analysis/collapse_diagnosis.py identity --logs campaigns/m1-diagnosis/logs \
  --logs campaigns/m1-mlp/logs --out $D/identity.json
uv run python scripts/analysis/collapse_diagnosis.py diagnose --logs campaigns/m1-diagnosis/logs \
  --out $D/diagnosis.json --csv $D/stacks.csv
uv run python scripts/analysis/collapse_diagnosis.py mlp --logs campaigns/m1-mlp/logs \
  --logs campaigns/m1-diagnosis/logs --out $D/mlp.json
```

## Retention (A.0)

Committed: this record, `identity.json`, `diagnosis.json`, `stacks.csv` (the aligned, smoothed series
every 25 episodes, so the figures can be rebuilt once the telemetry is archived) and `mlp.json`. Archived off-repo: the two
campaigns' raw logs, the worktree's session records, the telemetry files and the checkpoints.
