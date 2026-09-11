# The eligibility horizon on a multi-step task (7a-ii I.3b)

## Why

I.3 found the one setting that matters and it is untested where it would matter. On the positive
control, the node-perturbation rule closes 89% of the floor-to-optimum gap undelayed, 44.8% at ten
steps of delay, and **−6.8% at twenty — below the cue-blind floor**. Raising `trace_decay` from the
pinned 0.9 to 0.99 takes that twenty-step cell to 45.3%.

**Every result this phase has recorded ran at the pinned 0.9.** Not one panel config sets
`trace_decay`; they all take the default. Their episodes run 244 to 2400 steps — two orders beyond
the twenty at which the pinned decay already falls below the floor on a task with one scored action.

So the phase's central pattern — the rule learns a one-step association and fails every multi-step
task — now has a candidate mechanism that has never been tested on a multi-step task. I.4 is meant
to state which of the seven negative results are findings about the wiring and which about the
instrument. Writing it now would record "these may be artefacts of a crippled horizon, untested".
One cheap experiment replaces that with an answer.

The platform is the MLP yardstick: 3000 episodes in about eight minutes, and under I.1's rule at the
pinned horizon it currently ends **below its own frozen control**. That is the thing to move.

## What Changes

- **Six arms**: the yardstick under the node-perturbation rule at `trace_decay` ∈ {0.9, 0.99,
  0.999}, each with its own frozen control at the same setting. The frozen arm is what the learning
  arm must beat, and it is required at each horizon because the perturbation's cost to a policy is
  not a constant across settings.
- **Scored on the graded metric, with I.2's family.** The full-clear rate is at its floor for every
  yardstick arm ever run — 1.05% mean, no seed competent — so it cannot see a difference. Plateau
  tail mean foods can. This is the first registered use of the graded reading I.2 built, and the
  case it was built for.
- **The registered comparison is the learning arm against its own frozen control**, paired by seed,
  one-sided, BH-FDR across the three horizons. The committed 040 yardstick values are carried as a
  descriptive reference, not a comparator: they ran under the original rule.
- Records under `supporting/055-l4-horizon-multistep/`, tests, docs.

Out of scope: **the connectome.** Clone-assay runs cost about ten hours each, and spending them on
an untested lever is what this change exists to avoid. A connectome arm is registered only if the
yardstick moves.

## Capabilities

**Modified**: `plasticity-evaluation` (testing a setting on the platform whose failure motivated it).

## Impact

- New: six arm configs, the supporting directory, an analysis entry point. Edited: `CHANGELOG.md`,
  docs.
- No package code, no substrate or rule change. `trace_decay` is an existing config key and 0.9 is
  its default, so the baseline arms are the yardstick as it has always run.
