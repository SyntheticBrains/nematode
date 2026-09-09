# Consolidation mechanisms for the local rule (7a-ii B.4)

## Why

The clone-destruction diagnostic found the minimal rule's failure mode precisely: it is a
biased, near-constant-speed drift on the norm sphere that passes through good policies and
never stops. Rates from `1e-3` to `1e-5` all drift; no modulator form anchors the policy;
homeostasis removes the decay term as a brake. The named consequence was that a consolidation
mechanism — something that slows or stops updating once a policy is good — comes before
structured instruction, because even perfect credit assignment drifts away from what it found.
Logbook 044 then closed the substrate explanation: grounding synapse signs left the prior
unchanged and made reward-free learning worse, so the next result has to come from the rule.

The diagnostic's first candidate was an update magnitude that scales monotonically with the
prediction error's magnitude, on the reasoning that a good policy produces a small `|δ|`.
**Recon against the committed run telemetry contradicts that premise.** Per-step prediction
errors from the competent frozen clone (≈39% success) and the mostly-dead random-start frozen
arm (≈11%) have the same distribution — median `|δ|` 0.266 against 0.245, q95 0.80 against 0.88 —
and the running modulator scale sits at 0.4–0.8 in every arm on disk, competent and dead alike.
On this task `|δ|` is dominated by environment stochasticity, not by policy quality, so a step
monotone in `|δ|` is not smaller when the policy is good; fixing or annealing the scale only
changes the step's units, which is a rate change, and rate changes were already shown not to
anchor the policy.

The same recon rules out the other signals the rule can compute online. The running reward
baseline separates weakly and inconsistently (competent clone −0.237, random-start frozen −0.384,
but the rewired null −0.263), and median episode return ranks the rewired null above the
competent clone. **Nothing the rule can see reliably tells it that its policy is good.** That is
the constraint this change is designed around: the mechanisms it delivers do not need to know.

Ratified with Chris 2026-09-09: two quality-signal-free consolidation mechanisms as screened
arms, plus one oracle arm that gates on the environment's episode success — biologically
implausible by construction and run only as a diagnostic upper bound, so that a negative screen
can distinguish "consolidation is the wrong idea" from "the rule cannot see when to consolidate".

## What Changes

- **Elastic anchor** (`consolidation: anchor`): each plastic tensor carries a slow exponential
  moving average of its own weights, and the update gains a restoring term toward that anchor.
  Drift is opposed at the anchor's timescale without any quality signal; with the anchor rate at
  zero the anchor is the weights the rule started from, which is the informative limiting case.
- **Reinforced rigidity** (`consolidation: rigidity`): a per-synapse protective variable that
  grows where updates have been positively reinforced and decays slowly, dividing the effective
  rate by `1 + κ·c`. Synapses that reward has repeatedly written become progressively harder to
  write — dopamine-gated consolidation rather than dopamine-gated change.
- **Oracle gate** (`consolidation: oracle`): the effective rate is scaled by how far a trailing
  episode-success rate sits below a pinned reference. It consumes a metric the animal has no
  access to, is documented as an upper bound and never as a shippable mechanism, and reaches the
  rule through the brain's existing `post_process_episode(episode_success=...)` hook.
- **State that follows a loaded policy**: `reset_state()` re-anchors the elastic anchor to the
  weights the rule now starts from and clears the protective variable, the same re-anchoring the
  homeostatic targets needed after the warm-start defect.
- **Telemetry**: the effective rate multiplier, the mean anchor departure and the mean protective
  variable, beside the existing plasticity keys, so a screen can say whether a variant held the
  clone by consolidating or by not moving.
- **The registered screen**: the clone assay as already defined — wild-type plastic clone arm,
  seeds 1–8, 2000 episodes, comparator Logbook 043's committed frozen-clone values, pass = holds
  or improves — run once per variant after a pre-declared pilot pins each mechanism's two
  hyperparameters, with the endpoint cosine to the clone reported beside the metric.
- Configs, a screen harness, a launch record, the runs, records under
  `supporting/045-l4-consolidation/`, tests, docs.

Out of scope: the anti-Hebbian/decorrelating term (its own change, with its own screen — the
clone assay cannot see what it is for); structured pathway-specific instruction; the 2×2 panel
re-run, which this screen gates.

## Capabilities

**Modified**: `learning-rules` (the consolidation mechanisms, their state lifecycle and
telemetry), `l4-plasticity-panel` (the registered screen and its pass rule).

## Impact

- New: three consolidation code paths in `learning_rules/three_factor.py`, config fields on the
  plasticity mixin, clone-arm configs per variant, a screen harness, the supporting directory.
  Edited: `brain/arch/_plasticity_config.py`, `brain/arch/connectome_ppo.py` (the episode-success
  seam), `docs/architectures.md`, `configs/README.md`, `CHANGELOG.md`.
- Defaults are byte-identical: `consolidation: none` is today's code path, allocates no extra
  state and adds no operation to the update.
