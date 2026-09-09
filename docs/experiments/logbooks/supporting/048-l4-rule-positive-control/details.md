# Rule positive-control details

Run by `scripts/analysis/l4_rule_positive_control.py` at the registered budget: 8 seeds × 20,000
trials × 5 arm/rate combinations, scored on mean reward over each run's final 1,000 trials.
`control.json` is the full output and `per-seed.csv` the table.

## Outcome: `fail`, and the control is valid

| arm | mean | seeds above floor | result |
|---|---|---|---|
| `analytic` (reference) | **−0.1361** | 8/8 | **passes** |
| `hebbian` (floor) | −0.8753 | 3/8 | does not pass |
| `three_factor` @ 1e-4 | −0.7895 | 1/8 | does not pass |
| `three_factor` @ 1e-3 (pinned) | −0.7482 | 1/8 | does not pass |
| `three_factor` @ 1e-2 | −0.7504 | 2/8 | does not pass |

Cue-blind floor **−0.6909**; optimum **−0.1353**; the gap a learner can win is **0.5556**.

**The control is not void.** The analytic reference reaches −0.1361 against an optimum of −0.1353
— it closes 99.9% of the gap on every one of eight seeds — so the task is learnable, the topology
can express the answer, and the optimiser works. The unmodulated arm does not pass, so the task
does not leak its answer without reward. Both validity conditions hold, which means the
three-factor arm's failure is a fact about the rule.

**The rule does not merely fail to learn; it ends below the cue-blind floor.** At every rate the
mean sits around −0.75 against a floor of −0.69: a policy that ignored the cue entirely and
emitted the mean target would score better than what the rule produces after 20,000 trials. Only
one or two seeds of eight finish above the floor at any rate, and the rate grid spans two orders
of magnitude, so this is not a rate artefact.

## The diagnosis

| quantity | three-factor arm |
|---|---|
| mean modulator | +0.0020 (min 0.0015, max 0.0022) |
| mean absolute weight change per step | 2.62 × 10⁻⁴ |
| **gradient alignment** | **+0.031 mean, +0.009 median** (min −0.035, max +0.260) |
| alignment by rate (mean / median) | 1e-4: +0.077 / +0.074 · 1e-3: +0.020 / +0.025 · 1e-2: −0.002 / +0.001 |

The rule is not inert and it is not starved of reward information: the trace is live, the weights
move every step, and the modulator is a well-behaved centred prediction error. What the update is
not is *aimed*. Its cosine against the gradient-descent direction of the same trials is **+0.031 mean and +0.009 median** — indistinguishable from orthogonal — while the reference arm's is 1.0 by construction. Both statistics are recorded because the mean of a long-tailed per-run quantity overstates the typical run: one seed reaches +0.26 while the median sits at +0.009. The per-rate breakdown says the same thing at every rate, falling from +0.077 at 1e-4 to −0.002 at 1e-2.
The unmodulated arm's alignment is −0.015, statistically the same thing.

That is the measurement the reframing predicted. With a Hebbian eligibility (`pre × post`) and
exploration noise applied only at the action, an internal synapse's update carries no information
about which way to move to make the sampled action more likely; the rule reinforces correlation
structure that happens to be present, which on any substrate is a drift with no particular
relationship to reward. **The three-factor rule as implemented is not a policy-gradient estimator,
and this is the direct evidence.**

## What this licenses

Under the registered map, a `fail` means:

- **The seven negative results (Logbooks 040–046 and the routed third factor) are reframed as
  characterising a non-learner.** They remain valid as records of what this rule does on those
  substrates; they cannot support the inference that the wild-type wiring carries no learnable
  signal, because the instrument that produced them does not learn where learning is easiest.
- **I.1 is the critical path** — an eligibility with the noise inside it (node perturbation, or
  the exploration propagated into the trace), cleared on the MLP yardstick before anything else.
- **No substrate rung runs until it lands.** B.3's receptor layer and B.5's panel stay queued.

## Limitations

- The control tests one task. A rule could fail here and work on a task with different structure —
  though a rule that cannot learn a one-step association with dense reward is not a promising
  candidate for a 2400-step foraging task with sparse reward.
- The action is unsquashed and the brain's head is not under test; if the head were the problem,
  this control would not see it. The head is shared by the PPO arms, which do learn, so it is not
  a likely culprit.
- The topology is the panels' favourable arrangement (hidden plastic, frozen readout) at their
  pinned recipe. An all-plastic variant was not run and is the named follow-up.
- The alignment is measured against the gradient-**descent** direction of the immediate loss, so a rule reducing the loss scores positively. A rule
  estimating a longer-horizon return would legitimately differ; on a one-step task there is no
  such difference to hide behind.

## Facts

- 40 runs (8 seeds × 5 arm/rate combinations), all completed, single process, no environment.
- The launch record fixing the task, the pins, the pass rule, the void conditions and the reading
  of each outcome was committed before this run.
