# I.3b — the eligibility horizon does not transfer to a multi-step task

**Run 2026-09-11. Verdict: `does_not_transfer`. 48/48 runs succeeded. Licenses nothing.**

## The result

MLP yardstick, node-perturbation rule at σ = 0.2, 8 seeds, 3000 episodes. Each horizon against
**its own** frozen control, on plateau-tail mean foods.

| `trace_decay` | learning | frozen | shift | q | drift | beats control |
|---|---|---|---|---|---|---|
| **0.9** (the pinned default) | 0.393 | 2.233 | **−1.841** | 1.000 | 1.29 | no |
| 0.99 | 0.148 | 2.233 | **−2.086** | 1.000 | 1.31 | no |
| 0.999 | 0.144 | 2.233 | **−2.090** | 1.000 | 1.28 | no |

**The learning arm is far worse than doing nothing at every horizon, and raising the horizon makes
it worse.** 0.393 → 0.148 → 0.144 foods as the trace lengthens, against a frozen control at 2.233.
The learning arm is below its control on 6, 8 and 7 of 8 seeds respectively.

The registered one-sided test is in the improving direction, so q = 1.000 throughout: there is no
improvement to detect. The 0.5-foods minimum never comes into play.

## The horizon is not what was wrong

I.3 predicted a specific shape: at the pinned 0.9 the rule should be starved of credit, and a longer
trace should recover it. **The opposite happened.** The recovery that was large and monotone on the
one-step control is absent here and reversed.

The drift measurement — registered because a horizon too short and one too long would otherwise both
read as a null — says which failure this is. Relative weight distance from the frozen control is
**1.28 to 1.31 at every horizon**: the weights move further than their own norm. This is not a rule
starved of credit and sitting still. It is a rule writing a great deal, in a direction that makes
the policy worse, and the horizon barely changes how much it writes.

That was the case the design named as "a horizon too long — substantial drift, no gain". It is
present at **every** setting including the pinned one, which means the horizon is not the variable
that controls it.

## An internal check that passed

The three frozen controls return **identical** values (2.233 mean, the same eight per-seed numbers).
They must: `trace_decay` is only used to accumulate an eligibility trace, and a frozen arm writes
nothing, so the setting cannot reach behaviour. Three independently launched sets of eight runs
agreeing exactly is a check that the arms differ in what they are supposed to differ in.

## What this establishes

- **The horizon mechanism does not carry off the one-step control.** It is real there — I.3's
  numbers stand — and it does not explain the phase's multi-step failures. The suspicion is closed,
  which is what the registration said this outcome was worth.
- **The yardstick's failure is destructive, not starved.** Its learning arm is worse than its own
  frozen control by roughly two foods, having moved its weights by more than their own norm. Logbook
  040 reached this conclusion under the original rule and located the cause in a local rule
  collapsing a dense stack without decorrelation; the node-perturbation eligibility does not change
  it, and neither does the horizon.
- **A frozen perturbed policy reaches 2.233 foods where 040's committed learning arm reached 0.35.**
  Both rules — the original and the repaired one — leave the yardstick well below simply not
  learning. That is a statement about the platform, and it is the fourth registered outcome's
  neighbourhood: the yardstick is a poor place to ask this question, and both rules destroy it.

## What it does not establish

- Not that the horizon is irrelevant on the connectome. This tested a dense feedforward stack, which
  040 showed fails for a reason of its own. The registration excluded the connectome deliberately —
  ten hours a run against eight minutes — and this result is exactly the evidence that says not to
  spend them.
- Not that `trace_decay 0.9` is correct. It is not *limiting* here; whether it is right for a
  2400-step episode is untested and now untestable on this platform.
- Nothing about the rule's behaviour on a task it can learn. Every arm here is far below its floor.

## Cost, recorded

48 runs, 71 minutes wall, 15.5 hours summed. The estimate given before the run was ~40 minutes,
scaled from a pilot that used four workers and no experiment tracking; this used sixteen workers and
`--track-experiment`, and ran roughly ten times slower per run. Scale future estimates from a run
configured the way the campaign will be.
