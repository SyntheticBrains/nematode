## Why

The continuous-2D substrate reuses the shared reward calculator. Its potential-based distance
shaping is `reward_distance_scale · (prev_dist − curr_dist)` — designed to telescope (sum over an
episode ≈ scale · net approach) so it *guides* toward food without being the objective. But the two
distances are computed with **different metrics on the continuous substrate**:

- `curr_dist` = `env.get_nearest_food_distance_for(agent_id)` → **Euclidean** (continuous override).
- `prev_dist` was computed **inline as Manhattan** (`abs(Δx) + abs(Δy)`) in the reward calculator.

Since Manhattan ≥ Euclidean for every offset, `prev_Manhattan − curr_Euclidean` is **systematically
positive even when the agent does not approach food** — the term stops telescoping and pays a
**spurious per-step reward just for existing near food**. Measured: a worm wandering near food (no
net approach) earns **~+1.1 reward/step** under the mismatch vs **~0.000/step** with a consistent
metric. Episode forensics confirm the consequence: total episode reward ≈ `0.75 · steps` regardless
of foraging quality, and **surviving-without-completing out-rewards completing the target**
(max-steps episodes eating 6.8/10 scored reward 593 > completed-all-10 episodes' 465). The objective
was rewarding *survival*, not *task completion*.

This is the root cause of the T7 C1 recurrent-arm collapse (`tmp/evaluations/t7-continuous-prep/`):
the spurious per-step reward (a) inflates returns so the recurrent critic's value estimate explodes
(the symptom the parallel `add-ppo-value-normalization` change treats), and (b) makes the optimal
policy "survive/loiter near food," so recurrent policies drift to non-foraging. The longer the
episode, the worse the skew — which is exactly why the long-episode foraging task triggered it. The
**same mismatch is present in the predator-evasion term** (`prev_pred_dist` inline-Manhattan vs the
Euclidean `get_nearest_predator_distance_for`), skewing the Stage-1 predator calibration.

The `continuous-2d-environment` spec already requires `get_nearest_*_distance_for` to return Euclidean
"so the predator distance the reward calculator consumes is coherent with the continuous geometry" —
this change completes that coherence for the *previous*-step distance the calculator computes itself.

## What Changes

- **Env-native "distance-from-position" queries.** Add `get_nearest_food_distance_from(pos)` and
  `get_nearest_predator_distance_from(pos)` to the base environment (Manhattan, matching the existing
  `_for` queries), overridden on the continuous-2D environment to return **Euclidean** distance from
  an arbitrary position. These give the reward calculator a way to compute the *previous*-step
  distance in the **same metric** the environment uses for the current-step distance.
- **Reward calculator uses the env-native metric for the previous-step distance.** The foraging
  distance term and the predator-evasion term compute `prev_dist` via the new `*_distance_from(prev_pos)`
  queries instead of an inline Manhattan loop. On the continuous substrate both `prev` and `curr` are
  now Euclidean, so the potential-based term telescopes (no spurious survival reward); on the grid both
  remain Manhattan — **byte-stable** (the base method reproduces the previous inline computation
  exactly).
- **Reward formula unchanged (RQ5-safe).** This is a metric-coherence **bugfix**: the shaping *formula*
  (`scale · (prev − curr)`) and all coefficients are untouched. It realizes the telescoping behaviour
  the formula was always meant to have on the continuous substrate, rather than redesigning the reward.
- **Re-validation (downstream).** C1 foraging convergence is re-run n≥4 across the MUST arms with the
  corrected reward — completion becomes the optimal behaviour and returns are bounded, so the recurrent
  collapse is expected to resolve at the root.

## Capabilities

### Modified Capabilities

- `continuous-2d-environment`: extends the Euclidean-coherence guarantee — the environment SHALL expose
  a native-metric nearest-distance-from-arbitrary-position query (Euclidean on the continuous substrate),
  and the foraging/predator distance-reward terms SHALL compute the previous-step distance with it, so
  the potential-based shaping telescopes coherently with the continuous geometry instead of mixing
  Manhattan (previous) with Euclidean (current). Grid behaviour is Manhattan-both and byte-stable.

## Impact

- **Code:**
  - `packages/quantum-nematode/quantumnematode/env/env.py` — add base `get_nearest_food_distance_from(pos)`
    and `get_nearest_predator_distance_from(pos)` (Manhattan, matching the existing `_for` queries).
  - `packages/quantum-nematode/quantumnematode/env/continuous_2d.py` — override both with Euclidean
    (`math.hypot` from the given position, consistent with the existing `_for` overrides).
  - `packages/quantum-nematode/quantumnematode/agent/reward_calculator.py` — compute `prev_dist`
    (foraging) and `prev_pred_dist` (predator-evasion, `default` reward mode) via the new env queries;
    guard `None`.
- **Tests:** reward-calculator mocks gain the two `*_distance_from` methods (Manhattan-from-pos, matching
  the grid behaviour) so existing grid reward assertions stay byte-stable; a continuous-substrate test
  asserts the foraging distance term telescopes (no net approach → ~0 reward) instead of paying a
  spurious per-step reward.
- **Downstream:** T7 C1 foraging convergence re-validated n≥4 with the corrected reward; the recurrent
  value-explosion (`add-ppo-value-normalization`) is re-assessed against the corrected reward (the
  metric fix removes its root driver; value normalization remains a standard stabilization). Stage-1
  predator calibration inherits the corrected evasion term. No new dependencies.
