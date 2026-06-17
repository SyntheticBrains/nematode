# Design — continuous distance-reward metric coherence

## Context

The shared reward calculator computes potential-based distance shaping
`scale · (prev_dist − curr_dist)`. On the continuous-2D substrate `curr_dist` comes from the
Euclidean-overridden `get_nearest_food_distance_for`, but `prev_dist` was computed inline as Manhattan.
Because Manhattan ≥ Euclidean, the mismatch yields a systematically positive per-step term (~+1.1/step
measured) that never telescopes — paying the agent to loiter near food. Forensics
(`tmp/evaluations/t7-continuous-prep/`, "ROOT CAUSE" entries): reward ≈ 0.75·steps regardless of
foraging; surviving-without-completing out-scores completing. This is the root driver of the T7 C1
recurrent collapse (it inflates returns → value explosion, and makes survival optimal → policy drift).

## Decision 1 — Fix the metric, not the formula (RQ5-safe)

The shaping *formula* and coefficients are correct and frozen (RQ5). The bug is a **metric
inconsistency**: `prev` and `curr` must be measured in the same metric for the potential term to
telescope. The fix makes `prev` use the environment's native metric (Euclidean on continuous, Manhattan
on grid) — completing the Euclidean-coherence the `continuous-2d-environment` spec already requires for
`curr`. This realizes the *intended* telescoping behaviour rather than changing the reward design.

## Decision 2 — Env-native `*_distance_from(pos)` queries (not an inline metric switch)

Rather than branch on substrate type inside the reward calculator, add
`get_nearest_food_distance_from(pos)` / `get_nearest_predator_distance_from(pos)` to the environment:
Manhattan in the base (identical to the existing `_for` queries and the previous inline loop → grid
byte-stable), Euclidean in the continuous override (identical metric to its `_for` queries). The reward
calculator stays substrate-agnostic; the metric lives where the geometry lives. Symmetric with the
existing `get_nearest_*_distance_for` pair.

## Decision 3 — Residual discretisation asymmetry is acceptable

`curr_dist` uses the agent's true float position (`pos_continuous`); `prev_dist` uses `path[-2]`, the
discretised position view. After the metric fix both are Euclidean, leaving only a ≤~0.5 mm,
**zero-mean** discretisation difference between the true and discretised previous position. This is
non-systematic (rounds both ways) and cannot create a survival incentive, unlike the ~1.1/step
systematic Manhattan−Euclidean bias it replaces. Tracking the true float previous position would
require changing `path`'s representation (consumed by anti-dithering exact-equality and the
visited-cells exploration bonus) — out of scope for this bugfix.

## Decision 4 — Predator-evasion term too

The `default`-mode predator-evasion term has the identical mismatch (`prev_pred_dist` inline-Manhattan
vs Euclidean `get_nearest_predator_distance_for`). It is fixed the same way
(`get_nearest_predator_distance_from(prev_pos)`), with a `None` guard preserving the existing
no-previous-step fallback. The other predator reward modes (`gradient_*`, `distal_*`) do not use a
previous-distance delta and are unaffected.

## Decision 5 — Validation

- **Unit / byte-stability:** grid reward assertions unchanged (base Manhattan-from-pos reproduces the
  old inline loop); a continuous case shows the foraging distance term telescopes (wandering with no net
  approach → ≈0 cumulative distance reward, not a growing positive sum).
- **Behavioural smoke:** on the continuous substrate the per-step reward drops from ~0.75 to ~0.07 and
  completed-all-target episodes out-score survive-without-completing episodes (the incentive inverts to
  favour completion/efficiency).
- **Convergence (downstream, gates the T7 foraging lock):** C1 foraging re-run n≥4 across the MUST arms
  with the corrected reward — the recurrent arms must sustain foraging through training (no late
  collapse); MLP/Transformer hold.

## Risks / alternatives considered

- **Recompute via cached previous `curr_dist`** — caching last step's Euclidean `curr_dist` as this
  step's `prev_dist` avoids any position-representation issue, but breaks when a food is eaten (the
  nearest-food set changes; the recompute-from-prev-position approach handles that case). Rejected.
- **Change `path` to store float positions** — fully removes the discretisation residual but alters a
  representation consumed by anti-dithering and the exploration bonus (exact-cell semantics). Rejected
  for this bugfix; the residual is zero-mean and immaterial.
- **Relationship to `add-ppo-value-normalization`** — that change bounds the critic against value
  explosion; this change removes the *cause* of the explosion (spurious unbounded reward). They are
  complementary: value normalization remains a standard stabilization (and protects the recurrent
  connectome), but the C1 re-validation determines whether it is still load-bearing once the reward is
  correct.
