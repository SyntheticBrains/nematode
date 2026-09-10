# Design: a positive control for the three-factor rule

## What is under test, and what is deliberately not

The instrument is the committed `ThreeFactorRule`: its eligibility trace, its modulator, its
scaling switches and its bound. Everything else that could explain a null is removed rather than
controlled for — no environment, no runner, no reward shaping, no plateau metric, no 2400-step
horizon, no connectome. The rule drives the committed `MLPTopology` seam with synthetic
observations and rewards, in process.

That isolation is the point. Seven results are consistent with two readings — the wiring is not
legible, or the rule does not learn — and only a task with a known answer separates them. It also
means a failure here is unambiguous in a way a foraging null is not: there is nothing left to
blame.

## The task

A one-step continuous contextual association, with the smallest structure that reward-modulated
Hebbian learning is supposed to solve:

- A cue `c` is drawn uniformly from `K` one-hot alternatives (`K = 4`).
- The policy maps the cue to a scalar action `a = μ(c) + ε`, `ε ~ N(0, σ²)`: the topology's actor gives the mean and Gaussian noise is added at the action. The action is **unsquashed**. The brain's tanh-squashed Gaussian head lives in the brain, not in the topology, and is deliberately not under test — leaving it out is what keeps the floor and the optimum in closed form.
- Each cue has a target `t(c)`, fixed at construction and spread across the action range.
- Reward is `r = −(a − t(c))²`, so the optimum is `a = t(c)` and reward is bounded above by 0.

Two floors are computable in closed form rather than measured, which is what makes this a control:

- **Cue-blind floor.** A policy emitting a constant mean `a₀` regardless of cue earns at best `−Var[t] − σ²`, achieved at `a₀ = E[t]`; the exploration noise costs every policy `σ²`, so it appears on both sides. No cue-insensitive policy beats it.
- **Optimum.** A cue-sensitive policy with `μ(c) = t(c)` earns `−σ²`. The gap between floor and optimum is therefore exactly `Var[t]`, and the pass bar is anchored on it.

The gap between them is the whole signal, and it exists only if the learner uses the cue — which
it can only discover from reward, because the targets are not in the observation.

**Why this task and not a bandit.** A discrete bandit would need a different action head from the one the panels run. Keeping a continuous action keeps the rule's code path — the seam, the trace accumulation in the topology's forward, the modulator, the scaling switches — identical to the connectome arms'. What is *not* shared is the brain's action head, which the topology does not contain; the harness adds Gaussian noise to the raw mean instead, at the plastic arms' `σ` (the `log_std −1.0` every panel arm is frozen at).

## Each trial is an episode

Trials are independent, but `MLPTopology` decays its trace across forward calls, so without a
reset the eligibility gating trial *n*'s reward would carry trial *n−1*'s cue — the horizon
confound this control exists to remove, reintroduced by accident. The harness calls
`reset_traces()` before every trial, exactly as the panels reset traces per episode. With a
one-step trial the eligibility is then that step's `pre × post` and nothing else, which is also
what makes the gradient-alignment measurement below well defined.

## The three arms

| arm | what it is | what it establishes |
|---|---|---|
| `three_factor` | the committed rule, modulated | the instrument under test |
| `hebbian` | the committed rule, unmodulated | the floor: it never sees reward, so it must **not** solve a task whose answer only reward reveals |
| `analytic` | plain gradient descent on `−(a − t(c))²` through the same topology | the ceiling: the task is learnable here and the topology can express the answer |

The `analytic` arm is not a baseline to beat; it is a **validity check on the control itself**. If
it fails, the control is **void** — the task, the topology or the optimiser is at fault, and the
three-factor arm's result carries no information. That outcome is registered here so a void control
cannot be reported as a negative.

The `hebbian` arm is the mirror check: unmodulated learning solving a reward-only task would mean the task leaks its answer through the observation statistics, which would also void the control. The bar — seven of eight seeds and halfway to the optimum — is not reachable by chance, so a void on this condition is a real leak and not noise.

## The instrument, pinned

The rule under test is the panels' rule at the panels' pins, because that is the instrument every
registered result used: `plasticity_rate 1e-3`, both scaling switches on (modulator and trace
normalisation), homeostasis on, `weight_decay` and `weight_bound` at the panels' values, and
exploration at `log_std −1.0`. A control run at some other setting would say nothing about the
rule the panels ran.

One declared sensitivity grid, on the rate alone: `{1e-4, 1e-3, 1e-2}`, run as three sub-arms of
the three-factor arm. **Any rate passing counts as a pass**, because the claim under test is that
the rule can learn at all, and a fail must not be a rate artefact. The pinned rate is reported as
the primary and the others beside it.

The topology is `Linear(4, 8) → tanh → Linear(8, 1)` with the **hidden layer plastic and the
readout frozen** — the panels' arrangement (`plastic_layers: hidden`), and the one most
favourable to the rule, since Logbook 040 found a plastic output layer collapses on its own
output. With one-hot cues the hidden layer alone can express `μ(c) = t(c)` for all four cues (32
free weights against four constraints), which the analytic arm demonstrates. The all-plastic
variant is a named follow-up, not an arm of the control.

## The pass rule, fixed before any run

Over `n = 8` seeds, at a budget of 20,000 trials — chosen so the analytic arm converges with margin at a conservative step size, verified by its own test rather than a pilot:

- **Pass** — the three-factor arm's mean final reward beats the cue-blind floor on at least 7 of 8
  seeds, and its mean is at least halfway from that floor to the optimum.
- **Fail** — it does not.
- **Void** — the analytic arm does not itself pass, or the unmodulated arm does; the record reports
  `void` and draws no conclusion about the rule.

"Halfway to the optimum" is a deliberately weak bar. The claim under test is not that this rule is
efficient; it is that it moves policies toward reward at all.

## Telemetry, kept whatever the outcome

Three quantities per run, because a failure needs a diagnosis and this is the cheapest place to
get one:

- the **modulator** and its scale, which say whether the third factor carries reward information;
- the **eligibility magnitude**, which says whether there is a trace to gate;
- the **gradient alignment** — the cosine between the rule's update **accumulated over a block of 100 trials** and the analytic gradient summed over the same block. A single-step cosine of a stochastic estimator is noisy by nature and would read near zero even for a correct rule; over a block a correct estimator's alignment is clearly positive. This is the quantity that separates "learns slowly" from "moves in a
  direction unrelated to reward", and it is the direct measurement of the theoretical concern the
  reframing named: with a Hebbian eligibility and noise only at the action, an internal synapse's
  update need not correlate with the policy gradient at all.

An alignment near zero with a healthy modulator and a healthy trace **is** the diagnosis, and it
points straight at I.1.

## What each outcome licenses

- **Pass** — the rule learns when the task is trivial. The seven negative results stand as findings
  about the substrate *or* about the gap between this task and foraging, and block I's remaining
  items (horizon, homeostasis, exploration, metric) are what close that gap. I.1's eligibility
  variant becomes optional rather than required.
- **Fail** — the rule does not learn where learning is easiest. The seven results are reframed as
  characterising a non-learner, I.1 becomes the critical path, and no substrate rung runs until it
  lands. This is the outcome the reframing expects, and naming it in advance is what stops it being
  rationalised afterwards.
- **Void** — nothing is concluded and the control is rebuilt.

## Alternatives considered

- **Testing through the environment and runner.** More faithful and much less diagnostic: a null
  would be attributable to the horizon, the reward shaping or the metric, which is the ambiguity
  this change exists to remove. If the control passes, running it through the full path is the
  natural follow-up and would isolate the integration.
- **A discrete bandit.** Cleaner theory, different action head from every panel arm.
- **Using the connectome as the substrate.** Confounds the rule with the wiring, which is the
  confound under investigation.
- **Skipping the analytic arm.** It is the only thing standing between a failed control and a
  false conclusion about the rule.
