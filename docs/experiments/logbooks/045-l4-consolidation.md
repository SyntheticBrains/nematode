# 045: Consolidation — Three Brakes on a Rule That Never Stops (7a-ii B.4 / Phase 7)

**Status**: completed — **no mechanism passes the clone assay**, so the 2×2 panel stays gated. The
clone-destruction diagnostic had found the minimal rule's failure exactly: a biased,
near-constant-speed drift that passes through good policies and never consolidates, at any rate.
This screens three brakes on it. The diagnostic's own first candidate — an update magnitude
monotone in the prediction error — was **dropped before it was built**, on the committed
telemetry: per-step errors are the same size for a competent clone and a mostly-dead random start
(median 0.266 against 0.245), so `|δ|` carries no information about policy quality and a step
monotone in it is a rate change in disguise. Neither the reward baseline nor episode return does
better, so the two shippable mechanisms were designed to need no quality signal, and a third
arm that uses one was registered as a **bound rather than a candidate**. The results: an elastic
anchor does not hold the policy at all (13.0 against the frozen clone's 38.7, none of eight seeds
at or above); a per-synapse protective variable comes closest and fails one clause (29.4, six of
eight seeds inside the per-seed band, endpoint cosines **0.67–0.79** against the unbraked rules'
0.2–0.45, at a uniform **91% rate cut**); the oracle gate fails in the way its own launch record
predicted, its trailing estimate arriving a hundred episodes after a policy this rule destroys in
tens. The headline is the rigidity arm's: **a 91% per-synapse rate cut, applied exactly where
reward had been writing, does not stop the drift.**

**Branch**: `feat/l4-consolidation` (PR #331).

**Date**: 2026-09-09.

**OpenSpec change**: `add-l4-consolidation` (archived; the three mechanisms, their state
lifecycle and telemetry, and the registered screen; extends capabilities `learning-rules` and
`l4-plasticity-panel`).

## Objective

Give the local rule a mechanism that slows or stops updating once a policy is good, and put it
through the clone assay — the one protocol defined with the diagnostic — before any expensive
panel is re-run under it.

## Background

Four registered panels ([040](040-l4-panel.md)–[043](043-l4-warm-start.md)) and the
[clone-destruction diagnostic](supporting/043-l4-warm-start/destruction-diagnostic.md) converged
on one reading: the rule's failure is not a credit-assignment sign error but the **absence of
consolidation**. Rates from `1e-3` to `1e-5` all drift, no modulator form anchors the policy, and
homeostasis removes the decay term as a brake. [Logbook 044](044-l4-atlas-signs.md) then closed
the substrate explanation — grounding synapse signs left the prior unchanged and made reward-free
learning worse — so the next result had to come from the rule. Ratified with Chris 2026-09-09:
two quality-signal-free mechanisms as screened arms, plus one oracle arm.

## The measurement that changed the design

The diagnostic proposed an update magnitude scaling monotonically with `|δ|`, reasoning that a
competent policy produces small prediction errors. Checked against the committed run exports
before anything was built, that premise fails.

| arm | plateau-tail success | median |δ| | q75 | q95 |
|---|---|---|---|---|
| `plastic_frozen_clone` | ≈39% | 0.266 | 0.451 | 0.80 |
| `plastic_frozen` | ≈11% | 0.245 | 0.475 | 0.88 |

The rule's own running RMS of `δ` agrees: pooled over every plastic arm on disk it sits between
0.40 and 0.81 without regard to arm, wiring, initialisation, or whether the run started from a
competent clone. On this task the per-step reward prediction error is dominated by environment
stochasticity. Fixing or annealing the scale rescales every step by a constant — a rate change,
and rates were already shown not to anchor the policy.

The other signals the rule could compute online fail too. The running reward baseline over each
run's second half gives −0.237 for the competent clone arm, −0.384 for the random-start frozen
arm and −0.263 for the rewired null, which is closer to the clone than to the arm it behaviourally
resembles; median episode return ranks the rewired null (−16.3) **above** the competent clone
(−17.2). **Nothing the rule can see tells it that its policy is good.** That constraint is what
the mechanisms below are designed around.

## Method

Three arms, each the wild-type plastic clone arm with the mechanism's rule keys and nothing else
changed.

- **Elastic anchor.** Each plastic tensor keeps a slow moving average of its own weights and the
  update gains a restoring term `−η·κ_a·(w−a)`. Unlike decay, this is not a uniform per-unit
  shrink, so the homeostatic rescale cancels only its radial component and leaves the change of
  direction it made — which is why decay is not a brake on this substrate and this could be.
- **Reinforced rigidity.** A per-synapse protective variable grows where a positive modulator met
  a large trace — on the trace as the update sees it, so a pinned growth rate is
  substrate-invariant — and divides the Hebbian rate by `1 + κ_c·c` at its pre-growth value.
  Synapses reward has repeatedly written become progressively harder to write.
- **Oracle gate.** The rate is scaled by how far a trailing episode-success rate sits below a
  reference. It reads the environment's success flag, not the reward stream, so it is **not a
  mechanism a nervous system could host**; it was registered to separate "consolidation is the
  wrong idea" from "the rule cannot see when to consolidate".

**The assay** is the diagnostic's registered protocol, unchanged: seeds 1–8 paired, 2000
episodes, no extension, plateau-tail full-clear success, started from each seed's plastic-set
clone, against the warm-start panel's committed `wt_clone_frozen` values (mean 38.7). *Holds* =
mean within 5 points and ≥ 6 of 8 seeds no more than 10 points below their own; *improves* = mean
above and ≥ 6 of 8 above their own; *pass* = either. It is a **screen, not a confirmatory test**:
it reuses seeds already reported, declares no family and assigns no verdict, and a pass would
license the panel and nothing more.

**Pins** came from a pre-declared pilot: seeds 1–2 over a written grid, highest mean plateau tail,
ties to the weaker constraint. 20 runs. Anchor took `κ_a = 1.0, ρ_a = 0.001` (mean 25.70);
rigidity took `κ_c = 10, γ_c = 0.01, λ_c = 0.001` (37.00). The oracle had no pilot — both its
values are fixed by the comparator. The launch record noted before the screen that no anchor
combination came within 16 points of the comparator and that the anchor family's spread is
seed-dominated, an informal prediction that the anchor arm would fail.

## Results

### The screen

| arm | mean | Δ vs frozen clone | within hold | at or above | cosine to clone | rate multiplier |
|---|---|---|---|---|---|---|
| anchor | 13.0 | −25.7 | 2/8 | 0/8 | 0.49 | 1.00 |
| rigidity | 29.4 | −9.4 | **6/8** | 2/8 | **0.74** | 0.09 |
| oracle | 28.9 | −9.8 | 4/8 | 3/8 | 0.55 | 0.49 |

**None passes.** Rigidity fails on the mean clause alone, meeting the per-seed clause at exactly
the required 6 of 8 and missing the 5-point band by 4.4 points.

### Per-seed

| seed | frozen clone | anchor | rigidity | oracle |
|---|---|---|---|---|
| 1 | 39.3 | 13.8 | 37.4 | 43.0 |
| 2 | 44.0 | 37.6 | 36.6 | 42.6 |
| 3 | 40.0 | 1.8 | 46.4 | 18.0 |
| 4 | 21.3 | 10.2 | **53.4** | **55.2** |
| 5 | 47.1 | 1.0 | 2.4 | 0.0 |
| 6 | 33.3 | 25.8 | 28.0 | **58.6** |
| 7 | 61.3 | 13.6 | 9.2 | 13.2 |
| 8 | 23.3 | 0.2 | 21.4 | 0.4 |

### The runs still walk through good policies

100-episode blocks from the start of the run, against each seed's clone:

| arm, seed | clone | blocks 1–5 | final |
|---|---|---|---|
| anchor, 5 | 47.1 | 27, 14, **76, 75**, 14 | 1.0 |
| rigidity, 5 | 47.1 | 30, **56, 59, 56**, 29 | 2.4 |
| rigidity, 7 | 61.3 | 25, 28, 17, 22, 29 | 9.2 |
| oracle, 8 | 23.3 | 2, 0, 2, 1, — | 0.4 |

## Analysis

- **Rigidity preserves the policy better than anything tried on this substrate and it is still
  not enough.** Endpoint cosines of 0.67–0.79 on every seed, against the 0.2–0.45 the unbraked
  rules produced in the diagnostic, at a uniform rate multiplier of 0.09. The brake engaged hard,
  exactly where reward had been writing, and held the weights near where they started — and the
  arm still loses 9.4 points, at two seeds it destroys outright rather than by degrading
  everywhere. **A targeted 91% rate cut does not stop the drift**, which is the diagnostic's
  rate-insensitivity again with the cut aimed per synapse instead of applied globally.
- **The anchor separates two things that turn out to be different.** It opposes *departure* and
  does not preserve *behaviour*: at the pinned stiffness the weights stay closer to their start
  than the unbraked rule leaves them (cosine 0.49) and the policy is gone anyway (13.0, 0 of 8 at
  or above). The rule can walk a long way inside the region the anchor tolerates and arrive
  somewhere useless.
- **The oracle's failure is about lag, not about gating, and the launch record said so first.**
  Its early curves show seeds 1, 2, 3 and 8 at 2–4% within the first hundred episodes, from clones
  of 23–44%: the policy is gone before a trailing estimate at an EMA rate of 0.01 can respond, and
  the gate then stays open because success is low — a reactive brake inside a positive-feedback
  loop. Where the gate did close it worked: seeds 4 and 6 gate at multipliers 0.06 and 0.09 and
  finish at 55.2 and 58.6 from clones of 21.3 and 33.3. **This arm does not settle the question it
  was registered to settle.** A quality-gated brake fast enough to act is untested, not refuted.
- **The runs still pass through good policies and leave them.** Under the anchor, seed 5 reaches
  **76%** in its third block — well above its 47.1% clone and above anything the panels produced —
  and ends at 1.0%. Under rigidity the same seed holds 56–59% for three blocks and ends at 2.4%.
  Braking changed how fast the drift moves and not that it keeps moving through and past what it
  finds. That is the diagnostic's central claim, now surviving three attempts to stop it.
- **The better the starting policy, the more it loses.** Rank correlation between a seed's
  frozen-clone level and its delta: −0.79 (rigidity), −0.74 (oracle), −0.57 (anchor). The two
  strongest clones (61.3, 47.1) are destroyed in all three arms; the weakest (21.3) improves by
  more than 30 points under two of them. Part of this is regression to the mean on a noisy metric,
  and it is descriptive, not a registered contrast — but it is consistent with the drift walking
  to fixed points of its own, from which a strong start has further to fall.

## Conclusions

- **Consolidation as a brake on the update does not hold a competent policy on this substrate.**
  Three mechanisms, one of them cheating with a signal the animal does not have, and none passes.
- The nearest miss is informative: rigidity fails one clause while meeting the other, with the
  best policy preservation this substrate has shown. If a variant is going to pass this assay, it
  is likely to be in that family.
- **Slowing the update is not the same as consolidating a policy.** Every arm reduced how fast the
  weights move, two of them substantially, and the drift still walked past good policies. What is
  missing is not a smaller step but something that makes a good policy a place the dynamics
  *stay*.
- Registering the oracle as a bound was worth its eight runs even though it failed: its per-seed
  behaviour (gate closes → 55.2 and 58.6; gate stays open → 0.0 and 0.4) is the clearest evidence
  in the panel that gating *can* work, and it converts a flat negative into a specific open
  question.
- The recon that killed the first candidate before it was built is the cheapest result here: two
  hours of reading committed telemetry replaced a build-and-screen cycle that would have failed
  for a reason already visible in the data.

## Limitations

- One pinned setting per mechanism, chosen on two seeds. A different corner of each grid might
  behave differently, and the anchor's pilot spread was wide enough that its pin is weakly
  determined.
- The oracle's lag is a property of the pinned EMA rate, not of gating. The arm bounds nothing
  about a faster signal, which is exactly why the record carries it as untested.
- n = 8, and two seeds dominate the rigidity arm's failure. The assay's per-seed clause exists so
  that a mean cannot hide this, and it did its job — but it also means the arm's verdict rests on
  two runs.
- The clone assay asks whether a mechanism *holds a cloned policy*. A mechanism that helps
  learning from random weights and does not hold a clone would fail this screen; nothing here
  speaks to that case.
- Regression to the mean is not excluded from the clone-level correlation.

## Next Steps

The panel stays gated. The queue moves to the **anti-Hebbian/decorrelating term** — promoted by
Logbook 044's finding that a purely potentiating rule on a mostly-excitatory network has no
inhibitory brake — and then to **B.4b structured, pathway-specific instruction**. Two items are
carried forward explicitly: a **fast quality-gated brake** (an episode's outcome applied at once,
or a within-episode proxy) is untested rather than refuted, and the **rigidity family** is the
nearest miss and the first place to look if consolidation is revisited.

## Data References

- Registration and design: `openspec/changes/archive/2026-09-09-add-l4-consolidation/`;
  capabilities `openspec/specs/learning-rules/spec.md`,
  `openspec/specs/l4-plasticity-panel/spec.md`.
- Everything the screen produced:
  [supporting/045-l4-consolidation/](supporting/045-l4-consolidation/details.md) — `launch.md`
  (the grid, the criterion, the pins and the oracle's declaration, all written before the runs),
  `screen.json`, `per-seed.csv`, `_manifest.txt`, `details.md`.
- The protocol: [the clone assay](supporting/043-l4-warm-start/destruction-diagnostic.md#the-clone-assay).
- Tooling: `learning_rules/three_factor.py`, `brain/arch/_plasticity_config.py`,
  `scripts/analysis/l4_consolidation_screen.py`; arm configs under
  `configs/scenarios/foraging_predator_thermal/`.
