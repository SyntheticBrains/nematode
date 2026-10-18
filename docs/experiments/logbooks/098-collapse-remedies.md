# 098: No Standard PPO Remedy Beats Re-Rolling the Run; Late Collapse Is Not a Property of the Seed (Phase 9 M.1(b))

**Status**: completed. **No remedy passes, and the stop rule fires.** Three standard PPO remedies were
each compared with a placebo that only re-rolls the run's minibatch-shuffle stream. None lowered the late
drawdown by the registered margin.

| round | drawdown reduction, placebo − remedy (80% CI) | early success to episode 3,000, remedy − placebo (80% CI) | passes |
|---|---|---|---|
| 1, KL early stopping | −0.031 [−0.140, +0.074] | +0.024 [−0.025, +0.075] | no: no reduction |
| 2, weight clipping | −0.076 [−0.148, −0.005] | −0.041 [−0.108, +0.027] | no: drawdown worse, guard fails |
| 3, Adam's second moment | +0.026 [−0.075, +0.119] | −0.041 [−0.102, +0.020] | no: under the margin, guard fails |

A remedy needed a drawdown reduction of at least 0.05, with an interval above zero, and an early-success
interval whose lower end stays above −0.05. **Per the registration, the collapse statistic ships alone,
and R.2 and G.5 register at 3,000 episodes with it.** M.1(c), the confirmatory test of a passing remedy,
does not run.

**The placebo is the finding beside the verdict.** M.1(b) re-ran the eight runs that collapsed in
Logbook 092 and their eight intact twins. Only the minibatch-shuffle stream changed, and **the placebo
collapsed on 2 of the 16**: one of the original eight collapsers, and one twin that had not collapsed.
Each re-rolled arm collapses on a different subset. Across the four re-rolled 16-run arms, 9 of 64 runs
collapse (14%, 95% Wilson [8%, 25%]), against 092's committed base rate of 8 of 64 (12.5%). **Which run
collapses is a matter of the training stream, not the seed, and the rate does not move under any remedy
tried.**

**Date**: 2026-10-18.

**OpenSpec change**: `add-collapse-remedies`.

**Pre-registration**: [supporting/098-collapse-remedies/launch.md](supporting/098-collapse-remedies/launch.md),
committed at ddda93cc before any round ran. The placebo-paired selection and the relative KL stop were
D25 amendments by the maintainer on 2026-10-17, also before any round.

## The question

M.1(a) found no registered cause behind the body learners' late collapses
([Logbook 096](096-collapse-diagnosis.md)): collapsed and intact runs drift alike until the fall. D25
lists standard remedies for late PPO collapse in order of comparison risk. **Does any of them lower the
late drawdown, without slowing learning, more than re-rolling the run would?**

The rounds compare each remedy with a placebo rather than with the original runs, as amended. Those runs
were chosen because they collapsed, so against them a remedy that does nothing would still "rescue" most
of them.

## Method

**The runs.** Logbook 092's body configs on Cook 2019, at target 20 and 6,000 episodes, with update
telemetry. The 16 runs are the eight collapsers (wild type 2313 and 2328; gap-only null 2304, 2305,
2307, 2308, 2319 and 2330) and the other wiring's intact run on each of those seeds.

**The placebo** (`rollout_shuffle_stream: true`) draws the rollout buffer's minibatch permutations from a
separate seeded stream. Every remedy carries the same stream, so a remedy and the placebo differ by the
remedy alone, paired by seed and wiring.

**The remedies**, at values fixed by convention and never tuned on these seeds:

1. A KL stop, relative to each learner. A minibatch stops the update when its KL exceeds 3 × the median,
   over the last 100 updates, of each update's largest minibatch KL.
2. Weight clipping, at ± 2 × each tensor's largest magnitude at construction.
3. Adam with ε = 1e-4 and β₂ = 0.99.

Each remedy round also ran the MLP control on seeds 9721–9728, as M.1(c)'s preflight.

**The statistic.** Each run's drawdown is its highest trailing-200-episode success rate minus its mean
success over its last 500 episodes. Early success is the mean success to episode 3,000. Both were read
with `paired_seed_wilcoxon_bootstrap` over the 16 pairs, by the registered harness
`collapse_remedies.py read` at ddda93cc. Every one of the 88 runs (16 placebo, then 24 per round) exited
0\.

## Results

**Per round, against the placebo** (means over the 16 runs):

| arm | drawdown | early success | collapses (D25) | runs collapsed |
|---|---|---|---|---|
| placebo | 0.212 | 0.311 | 2 | null 2308; wild type 2307 |
| KL stop | 0.242 | 0.335 | 3 | null 2308, 2330; wild type 2313 |
| weight clip | 0.288 | 0.270 | 3 | null 2305; wild type 2305, 2307 |
| Adam | 0.186 | 0.270 | 1 | wild type 2313 |

No remedy's drawdown interval clears the 0.05 bar.

- **Weight clipping is worse than doing nothing.** Its interval lies wholly below zero.
- **Adam's settings come closest,** at +0.026, but their interval spans zero, and early learning falls by
  0.041 with an interval reaching −0.102.
- **The KL stop** fired on a median of 13.5% of updates (5.5–17.5% across runs), close to the 19% probed
  before launch. It neither lowered the drawdown nor slowed learning.

**Beside, never deciding:**

- **Against the original runs.** Remedy rescues and harms of M.1(a)'s collapsers were 5 and 0 (KL stop),
  7 and 2 (weight clip), and 7 and 0 (Adam). The placebo alone takes the original collapsers' drawdowns
  from 0.62–0.92 to 0.02–0.47. These counts are not evidence for any remedy, which is why the rule never
  read them.

- **Collapse does not follow the seed.** Of the placebo's two collapses, one is an original collapser
  (null 2308) and one an original intact twin (wild type 2307). Across the four re-rolled arms, the runs
  that collapse change from arm to arm, and the pooled rate (9 of 64) matches 092's (8 of 64).

- **The MLP preflight.** The MLP's plateau fell under every remedy, against M.1(a)'s MLP re-runs on the
  same seeds:

  | remedy | plateau change | 80% CI |
  |---|---|---|
  | KL stop | −15.0 points | [−27.7, −3.7] |
  | weight clip | −20.8 points | [−37.6, −3.9] |
  | Adam | −19.7 points | [−37.9, −1.4] |

  The shared cause is the placebo's stream, which every remedy carries. Under it, **seed 9724 never
  succeeds once in any round** (0 of 6,000 episodes, three times), where its original run learned to 84%.
  Seed 9726 falls to 12% and 0% in two rounds. No MLP placebo arm was registered, so the drop cannot be
  split between the stream and the remedies. None of the MLP runs collapsed.

- **Run times.** Round 2 took 5.1 hours of wall time against round 1's 3.1, but the runner's own clock,
  which does not count system sleep, logged 2.4 hours for its connectome runs. Mean episode length was
  395–407 steps in every round. The difference is the host sleeping, not the remedy.

## Interpretation

Three standard remedies for late PPO collapse do not prevent it on this body cell. The cleaner result is
the placebo's. Re-rolling only the minibatch order turns most of 092's collapsers into intact runs, and
some intact runs into collapsers, at an unchanged rate. Late collapse here is a stochastic event with a
stable incidence of about one run in eight by 6,000 episodes. It is not a property of a seed, a wiring, or
any trait M.1(a) measured before the fall.

That is what the collapse statistic (D25) is for. Incidence is reported per arm with its interval, and
every collapsed seed stays in the primary reading. The registered consequence follows: R.2 and G.5
register at 3,000 episodes, before the hazard begins (every M.1(a) onset falls after episode 3,858), with
the statistic reported beside. The MLP's collapse-free record in M.1(a) (0 of 8) and its dead seed under a
re-rolled stream are both description. Neither changes that consequence.

What this does not show: that no remedy works. Running advantage normalisation, D25's fourth remedy, was
not built. A learning-rate schedule, and remedies acting on `log_std` or the body drive's gains (which
weight clipping exempts), were not tried. The stop rule bounds the search at three rounds, as registered.

## What follows

- **The stop rule fires** (tracker M.1(b), roadmap risk row "M.1 finds no remedy"). The collapse statistic
  ships, M.1(c) does not run, and **R.2 and G.5 register at 3,000 episodes** with collapse incidence
  reported beside.
- **092's body contrast on the corrected gap model** was sequenced after M.1 reads out, and M.1 has now
  read. Its horizon is for its own change to fix. The risk row puts R.2 and G.5 at 3,000 episodes, and
  this reading gives no remedy that would make 6,000 episodes cleaner.
- **The stabiliser keys stay in the brains**, off by default and byte-identical when off (`docs/usage.md`).

## Data

- [supporting/098-collapse-remedies/launch.md](supporting/098-collapse-remedies/launch.md): the
  registration.
- [supporting/098-collapse-remedies/probe.json](supporting/098-collapse-remedies/probe.json): the
  identity probe and the KL stop's pre-launch firing rates.
- [klstop.json](supporting/098-collapse-remedies/klstop.json),
  [wclip.json](supporting/098-collapse-remedies/wclip.json),
  [adam.json](supporting/098-collapse-remedies/adam.json): each round's reading, with every run's
  drawdown, early success, collapse and onset, and the MLP preflight per seed.
- Raw logs and telemetry of the four campaigns: archived off-repo.
