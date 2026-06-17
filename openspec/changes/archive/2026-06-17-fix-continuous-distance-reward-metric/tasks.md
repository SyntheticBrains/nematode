# Tasks — continuous distance-reward metric coherence

## 1. Env-native distance-from-position queries

- [x] 1.1 Add `get_nearest_food_distance_from(pos)` and `get_nearest_predator_distance_from(pos)` to the
  base environment (`env.py`) — Manhattan, identical to the existing `_for` queries; `None` when no
  foods / no enabled predators.
- [x] 1.2 Override both on the continuous-2D environment (`continuous_2d.py`) with Euclidean
  (`math.hypot` from the given position), consistent with its `_for` overrides.

## 2. Reward calculator uses the native metric for the previous-step distance

- [x] 2.1 Foraging: compute `prev_dist` via `env.get_nearest_food_distance_from(prev_pos)` (was inline
  Manhattan).
- [x] 2.2 Predator (`default` reward mode): compute `prev_pred_dist` via
  `env.get_nearest_predator_distance_from(prev_pos)`, guarding `None` (preserve the
  no-previous-step fallback).

## 3. Tests

- [x] 3.1 Update reward-calculator mocks to provide the two `*_distance_from` methods (Manhattan-from-pos
  over the mock's foods/predators) so existing **grid** reward assertions stay byte-stable.
- [x] 3.2 Add a continuous-substrate test asserting the foraging distance term telescopes (tangential
  wander with no net approach → ≈0 cumulative distance reward), guarding against re-introduction of
  the metric mismatch.

## 4. Validation — C1 foraging convergence re-run (gates the T7 foraging lock)

- [x] 4.1 Re-run C1 foraging convergence n≥4 across MLP / Transformer / LSTM / CfC with the corrected
  reward. Acceptance: the recurrent arms (LSTM, CfC) sustain foraging through training (no late
  collapse) and reach a genuine foraging level; MLP / Transformer hold their converged levels.
- [x] 4.2 Record per-arch converged foraging (final-quarter completed-target success + efficiency) in the
  T7 forensics scratchpad → T7 logbook; note the corrected metric and the spurious-reward removal.
- [x] 4.3 Re-assess `add-ppo-value-normalization`: with the corrected reward, determine whether value
  normalization is still load-bearing for recurrent convergence or reduces to standard hygiene
  (keep for the recurrent connectome / numerical robustness either way).

## 5. Gates + tracking

- [x] 5.1 `openspec validate fix-continuous-distance-reward-metric --strict`.
- [x] 5.2 Targeted `pre-commit` (ruff / pyright / markdownlint) on changed files; reward-calculator +
  continuous-env test suites pass (incl. grid byte-stability).
- [x] 5.3 Note in `phase6-tracking` that the T7 C1 recurrent collapse root cause was the continuous
  distance-reward metric mismatch (and that Stage-1 predator calibration used the pre-fix evasion
  term).
