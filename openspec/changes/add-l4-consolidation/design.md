# Design: consolidation mechanisms for the local rule

## The measurement this design is built on

Before choosing a mechanism, the premise behind the diagnostic's first candidate was checked
against the committed run exports. The candidate was an update magnitude monotone in `|δ|`, on
the reasoning that a competent policy produces small prediction errors.

Per-step prediction error, three runs per arm, sampled from the detailed telemetry:

| arm | plateau-tail success | median `|δ|` | q75 | q95 |
|---|---|---|---|---|
| `plastic_frozen_clone` | ≈39% | 0.266 | 0.451 | 0.80 |
| `plastic_frozen` | ≈11% | 0.245 | 0.475 | 0.88 |
| `plastic_clone` | (learning) | 0.229 | 0.402 | 0.74 |

The running modulator scale — the rule's own bias-corrected RMS of `δ` — agrees: pooled over
every plastic arm on disk it sits between 0.40 and 0.81 without regard to arm, wiring,
initialisation or whether the run started from a competent clone. The per-step reward prediction
error on this task is dominated by environment stochasticity, and carries no usable information
about policy quality.

Two consequences:

1. **A step monotone in `|δ|` does not consolidate here.** With `|δ|` the same size for good and
   bad policies, fixing or annealing `σ` rescales every step by a constant, which is a change of
   rate; the diagnostic already ran rates from `1e-3` to `1e-5` and found the drift accumulates
   in proportion to rate × time whatever the rate. This mechanism was dropped for that reason,
   and the measurement is recorded here so the drop is reviewable rather than asserted.
2. **No online scalar substitutes for it.** The running reward baseline over the second half of
   each run gives −0.237 for the competent clone arm, −0.384 for the random-start frozen arm and
   −0.263 for the rewired null, which is closer to the clone than to the arm it resembles in
   behaviour; median episode return ranks the rewired null (−16.3) above the competent clone
   (−17.2). A rule that must recognise its own competence from reward alone cannot do so on this
   task at this budget.

The design therefore takes the constraint as given: **the two shippable mechanisms do not need a
quality signal**, and the one that uses a quality signal is registered as an oracle rather than
as a candidate.

## Mechanism 1: elastic anchor

Each plastic tensor carries an anchor `a` of the same shape. The anchor starts at the weights the
rule was built over, follows them slowly, and the update gains a restoring term toward it:

```text
u   = η · m · E / ρ_E            (today's Hebbian term)
u  -= η · λ_w · w                (today's decay)
u  -= η · κ_a · (w − a)          (new: restoring force toward the anchor)
a  ← a + ρ_a · (w − a)           (after the write, from the weights as they now are)
```

`ρ_a` is the anchor rate and `κ_a` the stiffness, both configured, both zero by default. This is
decay toward a moving anchor rather than toward zero, and it opposes drift directly: a
constant-speed departure builds a restoring force proportional to how far it has gone, while the
anchor's own motion keeps the mechanism from freezing the substrate outright. `ρ_a = 0` is the
limiting case — a fixed anchor at the policy the rule started from — and is worth running because
it bounds what the family can do.

The restoring term is applied inside the same masked update as the decay, so it never writes off
the edge set. The anchor is a slow variable and not a target the rule optimises; nothing about it
depends on knowing whether the policy is good.

## Mechanism 2: reinforced rigidity

Each plastic tensor carries a non-negative protective variable `c`, zero at construction. After
the update is computed and before it is applied, `c` scales the rate down and then absorbs what
the step reinforced:

```text
u   = η / (1 + κ_c · c) · m · E / ρ_E − η · λ_w · w
c  ← (1 − λ_c) · c + γ_c · max(m, 0) · |E| / ρ_E
```

`γ_c` is the growth rate, `λ_c` the decay and `κ_c` the strength, all configured, growth zero by
default. The trace enters growth as the update sees it — divided by the running trace scale
`ρ_E` when trace normalisation is on — so a pinned `γ_c` means the same root-mean-square
growth on the sparse connectome and on the dense yardstick, which is the invariance the rest
of the rule already keeps. Rigidity accumulates where a positive modulator has repeatedly gated a large trace — the
synapses reward has been writing — and decays slowly everywhere, so nothing is frozen forever.
Under the unmodulated Hebbian floor `m` is `1.0` and rigidity accumulates wherever the trace is
large, which is the same mechanism with the reward gate removed, matching how every other switch
in this rule behaves across the two arms.

The rate divisor uses the value of `c` from before this step's growth, the same pre-update
convention the modulator scale and centre already follow, so a step is never charged for the
rigidity it is about to create.

## Mechanism 3: the oracle gate

The rule keeps a trailing success rate `s`, an exponential moving average over episodes of the
environment's own episode-success flag, and scales the rate:

```text
η_eff = η · clamp(1 − s / s_ref, 0, 1)
```

`s_ref` is a configured reference success rate. At or above it the rule stops writing; well below
it the rule is unchanged. The flag arrives through the brain's existing
`post_process_episode(episode_success=...)` hook, which is a no-op on this brain today, so no new
plumbing crosses the runner boundary.

Two structural facts about this arm are recorded so its result is read correctly. With `s_ref`
pinned just above the comparator's mean, the multiplier is about 0.03 once the trailing estimate
reaches the clone's own level: the arm is a thermostat at the frozen clone's performance, and
**"improves" is unavailable to it by construction** — it bounds *holding*, nothing else. And the
trailing estimate starts at zero, so the first hundred or so episodes run at close to the full
rate before the gate closes; if that early window is enough to take the clone apart, the arm
will show it, and that is a result about the rule's speed rather than about gating. Pinning
`s_ref` above the comparator so that improvement were possible was considered and not taken:
it would turn a bound into a tuned candidate.

This is an oracle. Episode success is a property of the task's scoring, not of anything the
animal could compute from its own reward stream, and the docstring and the spec say so. It exists
because a negative screen on the two shippable mechanisms is ambiguous on its own: if the oracle
also fails, consolidation is not the missing piece; if the oracle passes and the others do not,
the missing piece is the quality signal, and that is a different research question with different
candidates (a critic, a slow behavioural statistic, an interoceptive proxy).

## Where the terms sit in the update

The order is fixed and each position is load-bearing:

1. Hebbian term, trace-normalised if enabled, with the rate divided by the protective variable.
2. Weight decay toward zero.
3. Elastic restoring term toward the anchor.
4. Mask, then write.
5. Sign projection (Dale's law), unchanged.
6. Homeostatic rescale, unchanged.
7. Clamp to the magnitude bound, last, so the bound always holds.
8. Anchor and protective-variable updates, from the weights as they now are.

Consolidation state is updated after the write for the same reason the modulator scale is
estimated before it: each quantity is measured against the step it actually saw.

**Why homeostasis does not undo these brakes as it undoes decay.** The diagnostic's reason that
decay is not a brake on this substrate is exact: decay shrinks every incoming weight of a unit
by the same factor, and the incoming-norm rescale multiplies them back by its inverse. The
restoring term is not uniform across a unit's incoming weights — it points from each weight
toward its own anchor — so the rescale, a single positive scalar per unit, cancels only its
radial component and leaves the change of direction it made. The protective variable is a
per-synapse rate divisor applied before the write, which a per-unit scalar after the write
cannot touch. Both act in the space the rescale leaves free, which is the space drift lives in.

## State lifecycle

`reset_state()` re-anchors the elastic anchor to the weights the rule now starts from and returns
the protective variable to zero. This is not incidental. The warm-start panel lost four arms to
exactly this class of defect — homeostatic norm targets left at the values of a random
initialisation while the weights were a loaded clone, so the first plastic step pulled the clone
back toward noise. An anchor left at construction values behind a loaded policy would do the same
thing, with more force. The screen loads a clone on every seed, so this path is the one it runs.

`reset_episode()` gains the oracle's episode bookkeeping and stays a no-op otherwise.

Consolidation state is transient rule state and is not persisted with weights; a checkpoint
carries the policy, and a rule rebuilt over it starts its anchor from that policy.

## Configuration

One selector plus each mechanism's own parameters on the shared plasticity mixin, defaults off:

| field | default | meaning |
|---|---|---|
| `plasticity_consolidation` | `none` | one of `none`, `anchor`, `rigidity`, `oracle` |
| `plasticity_anchor_rate` | `0.0` | `ρ_a`, the anchor's EMA rate, `[0, 1)` |
| `plasticity_anchor_stiffness` | `0.0` | `κ_a`, the restoring coefficient, `≥ 0` |
| `plasticity_rigidity_growth` | `0.0` | `γ_c`, `≥ 0` |
| `plasticity_rigidity_decay` | `0.0` | `λ_c`, `[0, 1)` |
| `plasticity_rigidity_strength` | `0.0` | `κ_c`, `≥ 0` |
| `plasticity_oracle_reference` | `1.0` | `s_ref`, the success rate at which updating stops, `(0, 1]`; pinned per screen |
| `plasticity_oracle_rate` | `0.01` | the EMA rate of the trailing success estimate, `(0, 1]` |

A selector rather than independent booleans: the mechanisms are alternatives being screened, and
a config that silently ran two at once would produce a result attributable to neither. Selecting a
mechanism that its own parameters leave inert is rejected at load, since that is a config that
looks like it consolidates and does not: `anchor` with a stiffness of zero, `rigidity` with a
growth or a strength of zero. The oracle's reference defaults to `1.0`, which never closes the
gate, so a run that selects it pins the reference in its launch record rather than inheriting a
number from one panel.

## Telemetry

Three keys beside the existing plasticity series: the effective rate multiplier actually applied
(`1` under `none`), the mean absolute anchor departure over the edge set, and the mean protective
variable over the edge set. A variant can hold the clone by consolidating or by barely moving,
and these separate the two — as does the endpoint cosine the screen already reports.

## The screen

The clone assay, exactly as registered with the diagnostic, once per variant: the wild-type
plastic clone arm with the variant's rule keys and nothing else changed, the S.2 plastic-set
wild-type clone for each seed, seeds 1–8 paired, 2000 episodes, no extension, plateau-tail
full-clear success, compared against the committed `wt_clone_frozen` per-seed values (39.3, 44.0,
40.0, 21.3, 47.1, 33.3, 61.3, 23.3; mean 38.7). **Holds** is a mean within 5 points of the frozen
clone's and at least 6 of 8 seeds no more than 10 points below their own; **improves** is a mean
above it with at least 6 of 8 seeds above their own; **pass** is either.

It is a screen, not a confirmatory test: it reuses seeds Logbook 043 already reported, so a pass
licenses running the registered panel and nothing more. No multiple-comparisons family is
declared and no verdict map applies, and the record says so.

**Pilot.** Each mechanism has two hyperparameters with no prior value to inherit, so a
pre-declared pilot pins them before the screen: seeds 1–2 of the same clone arm, 2000 episodes,
over the declared grid — anchor `κ_a ∈ {0.01, 0.1, 1.0}` × `ρ_a ∈ {0.0, 0.001}`, rigidity
`κ_c ∈ {1, 10}` × `γ_c ∈ {0.01, 0.1}` at `λ_c = 0.001` — pinning the combination with the highest
mean plateau tail across the two seeds, ties to the smaller `κ`. The oracle takes `s_ref = 0.4`,
just above the frozen clone's 38.7% mean, and `0.01` as its EMA rate; it has no pilot because
both values are pinned by the comparator rather than chosen. The pilot's grid, criterion and
results are written into the launch record before the screen runs, and the pilot's two seeds are
reported with the screen so a pin that only worked on its pilot seeds is visible.

## What a result means

A variant that passes licenses the 2×2 panel re-run under that rule; it does not on its own say
the wiring is legible, which is what the panel exists to ask. A variant that passes with a high
endpoint cosine held the policy; one that passes with a low cosine found a different policy of
similar quality, which is a different and weaker claim. If both shippable mechanisms fail and the
oracle passes, the record states that consolidation works on this substrate but the rule cannot
see when to apply it, and the next candidate is a quality signal rather than another brake. If
the oracle fails too, consolidation is not the missing piece and the queue moves to structured
instruction with that written down.

## Alternatives considered

- **Annealed or fixed modulator scale** — dropped on the measurement above: `|δ|` does not shrink
  when the policy is good, so this is a rate change in disguise.
- **Elastic weight consolidation with a Fisher diagonal** — needs a gradient the rule does not
  have, and a task boundary this continuous setting does not provide.
- **Freezing a fraction of synapses outright** — equivalent to shrinking the plastic set, which
  the panel's frozen arms already bound, and it answers a different question.
- **Consolidating on the eligibility trace's own consistency** (rigidity where the trace has been
  stable) — attractive, quality-signal-free and untested; deferred rather than rejected, since it
  is a third arm on the same harness once these two are read.
