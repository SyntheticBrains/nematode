# M.1(b) — collapse remedies, read against a placebo: registration and launch

**Registered 2026-10-17, before any round runs.** Change: `add-collapse-remedies`. Tracker
`phase9-tracking` M.1(b); roadmap v4.5 D25, as amended by the maintainer on 2026-10-17.

## The question

M.1(a) found no registered cause behind the body learners' late collapses: collapsed and intact runs
drift alike until the fall ([Logbook 096](../../096-collapse-diagnosis.md)). D25 lists the remedies in
order of comparison risk. **Does any of them lower the late drawdown, without slowing learning, more than
re-rolling the run would?**

## The rounds

Logbook 092's body configs, Cook 2019, target 20, 6,000 episodes, with update telemetry. Every round runs
M.9's 16 seeds — the eight collapsers (wild type 2313, 2328; gap-only null 2304, 2305, 2307, 2308, 2319,
2330\) and their eight intact twins on the other wiring — and each remedy round also runs the MLP control's
retry seeds 9721–9728 as a preflight for M.1(c).

| round | brain keys | runs |
|---|---|---|
| placebo | `rollout_shuffle_stream: true` (the minibatch permutations from their own seeded stream) | 16 |
| 1, KL early stopping | the placebo's stream; `kl_stop_factor: 3` (a minibatch stops the update when its KL exceeds 3 × the median, over the learner's last 100 updates, of each update's largest minibatch KL) | 16 + 8 MLP |
| 2, weight clipping | the placebo's stream; `weight_clip: 2` (±2 × each tensor's largest magnitude at construction) | 16 + 8 MLP |
| 3, Adam's second moment | the placebo's stream; `adam_eps: 1e-4`, `adam_beta2: 0.99` | 16 + 8 MLP |

Every remedy carries the placebo's stream, so a remedy and the placebo differ in the remedy alone. Each
value is fixed here by convention or by D25's list and is not tuned on these seeds. Running advantage
normalisation, D25's fourth remedy, is not built; a learning-rate schedule is not in this list.

**Predicted reach, stated before the rounds**: the KL stop acts on each learner's outsized updates; the
window holds the same statistic each minibatch is tested on (an update's largest minibatch KL), so an
ordinary update's late epochs do not trip it. Its firing rate, measured before launch, is below. Weight
clipping at
κ = 2 acts chiefly on the critic and the thermal gains (grown 1.3–6.1× and 1.6–5.3× by the end of
M.1(a)'s runs), while `log_std` and the body drive's log-gains, zero at construction, are exempt; Adam's
settings act on every step, through the second moment. Round 2 therefore cannot touch exploration or the
body drive's gains, which grew most; a round-2 failure says nothing about them.

## The reading, fixed before any round runs

**The statistic**: each run's **drawdown**, its highest trailing-200-episode success rate minus its mean
success over its last 500 episodes (on M.1(a)'s runs, 0.62–0.92 for the collapsers and 0.06–0.27 for the
intact twins).

**A remedy passes** when, over the 16 runs paired with the placebo by seed and wiring
(`paired_seed_wilcoxon_bootstrap`, 80% interval):

- the drawdown falls: `placebo − remedy` has a mean of at least **0.05** and an interval above zero;
- learning is not slowed: success to episode 3,000, `remedy − placebo`, has an interval whose lower end
  is above **−0.05** (every M.1(a) onset is after episode 3,858).

The rounds run in order and stop at the first pass. After three rounds without one, **the stop rule
fires**: the collapse statistic ships alone, and R.2 and G.5 register at 3,000 episodes with it. If a
passing remedy then fails M.1(c), the next remedy in order is tried, within the three rounds.

**Beside, never deciding**: collapse counts under D25 (as amended), rescues and harms against M.1(a)'s
original re-runs, the KL stop's firing rate, and the MLP preflight (its plateau and collapses under the
remedy against M.1(a)'s MLP re-runs). Collapse counts against the original runs are not evidence: those
runs were chosen because they collapsed.

**Failed runs**: a run that exits non-zero is re-run once on the same seed; if it fails again it counts
against the remedy (its drawdown taken as its peak, its early success as zero). A placebo run that fails
twice leaves the round unreadable.

## Before launch

- **Identity** ([probe.json](probe.json)): 092's configs with every stabiliser key written out at its
  default reproduce M.1(a)'s runs, episode for episode over the first 20, on the gap-only null (2304) and
  the MLP (9721). Off is byte-identical in the change's tests (weights, optimiser state, saved keys, and
  the next draw of the brain's and the buffer's generators).
- **The KL stop's firing rate**, measured on the round-1 configs before launch (150 episodes, seeds
  9893–9894, discarded), over the updates after each run's 100-update window: the connectome wild type's
  stop ended **14 of 73 updates (19%)**, on average after 21 minibatch steps; the MLP's **5 of 191
  (2.6%)**, after 22. So it acts selectively on both learners, late in the updates it stops. These are
  early-training rates; the round itself reports the rate over the full 6,000 episodes.
- **No gate preflight**: no round reads a floor or saturation gate, and the configs are 092's.
- **Cost**: M.1(a) measured 7,775 seconds for the same 16 connectome runs at 16 workers. The commands are
  sequential campaigns: the placebo (about 2.2 hours), then each round's connectome runs (about 2.2) and
  MLP runs (about 1.1). The placebo and round 1 take about 5.5 hours; each further round about 3.3; at
  most about 12 hours, with the machine to itself, after G.0-A's selection pilot.
- **Readiness review**: two spec reviews of the change (2026-10-17), whose findings and the maintainer's
  two decisions are recorded in the design and as dated amendments to D25.

## Launch

From a worktree at this commit, after G.0-A's pilot has finished:

```bash
git worktree add --detach ../nematode-m1b <this commit>
cd ../nematode-m1b
uv sync --extra cpu --extra torch --extra pixel --extra analysis --extra qpu
T=configs/scenarios/thermal_foraging
B=${T}/connectomeppo_small_continuous2d_thermal_klinotaxis_body
M=${T}/mlpppo_small_continuous2d_thermal_klinotaxis_body
OUT=(-- --theme headless --track-experiment --no-detailed-export --no-file-log --update-telemetry)
round() {
  local name=$1
  uv run python scripts/run_campaign.py --config ${B}_wt_ppo_m1b_${name}_t20.yml \
    --config ${B}_gaponly_ppo_m1b_${name}_t20.yml \
    --seeds 2304,2305,2307,2308,2313,2319,2328,2330 --runs 6000 --workers 16 \
    --output-dir ../nematode/campaigns/m1b-${name} "${OUT[@]}"
  if [ "$name" != placebo ]; then
    uv run python scripts/run_campaign.py --config ${M}_ppo_m1b_${name}_t20.yml \
      --seeds 9721-9728 --runs 6000 --workers 16 --output-dir ../nematode/campaigns/m1b-${name} \
      "${OUT[@]}"
  fi
}
round placebo; round klstop

D=../nematode/docs/experiments/logbooks/supporting/098-collapse-remedies
uv run python scripts/analysis/collapse_remedies.py read --round klstop \
  --logs ../nematode/campaigns/m1b-placebo/logs --logs ../nematode/campaigns/m1b-klstop/logs \
  --original ../nematode/campaigns/m1-diagnosis/logs --original ../nematode/campaigns/m1-mlp/logs \
  --out $D/klstop.json
```

Read round 1. If it does not pass, run `round wclip` and read it with `--round wclip`; if that does not
pass, run `round adam` and read it with `--round adam`. A missing log is refused, never read as a failed
run; every original M.1(a) run must be present.

## Retention (A.0)

Committed: this record, `probe.json`, and each round's reading (`klstop.json`, `wclip.json`,
`adam.json`, each with every run's drawdown and early success). Archived off-repo: the rounds' raw logs,
telemetry and session records.
