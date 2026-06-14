## Why

`add-continuous-predator-kinematics` made the continuous-2D predator **detection**, **damage**,
and **contact-zone** checks Euclidean against the worm's `pos_continuous`, but left the
**nearest-predator-distance query** (`get_nearest_predator_distance_for` / `get_nearest_predator_distance`,
`env.py`) on the inherited **integer-Manhattan** computation (`abs(Δx) + abs(Δy)` against the
discretised integer `position`). The predator-evasion **reward** queries that method —
`reward_calculator.py` reads `curr_pred_dist = env.get_nearest_predator_distance_for(...)` and the
`distal_chemo_contact_trigger` reward (used by every T7 predator config) fires its contact penalty
on `curr_pred_dist <= 1`. So on the continuous substrate the reward measures a **Manhattan diamond
geometry the worm no longer lives in**, the last predator-geometry coherence gap `predator_kinematics`
documented and deferred.

This is the `T7.prep.reward_coherence` (Stage 0.5) item: a **coherence/fidelity fix**, sequenced before
the Stage-1 difficulty calibration (a coherent predator distance changes predator-evasion learnability,
so calibrating against the incoherent metric would force a redo).

## What Changes

- **Euclidean nearest-predator-distance on the continuous substrate.** `Continuous2DEnvironment`
  overrides `get_nearest_predator_distance_for` (and the single-agent convenience
  `get_nearest_predator_distance`) to return the **true Euclidean distance** between the agent's
  `pos_continuous` and each predator's real-valued position, mirroring the existing Euclidean
  detection/damage/contact-zone overrides. This completes the continuous predator-geometry set.
- **Reward formula unchanged (RQ5 coherence-only).** The reward calculator is **not edited** — it calls
  the same env method, which now returns a coherent (Euclidean) distance on continuous. No reward
  weights, structure, or thresholds change; this is a substrate-geometry fix, not reward tuning.
- **Grid environment unchanged and byte-stable.** The override lives only on the continuous subclass;
  the discrete grid keeps integer-Manhattan nearest-predator-distance.
- **Out of scope (documented):** the `reward_mode == "default"` inline `prev_pred_dist` Manhattan
  computation in `reward_calculator.py` is **not** changed — no T7 config uses `default` mode (all use
  `distal_chemo_contact_trigger`, which skips that term), and the agent `path` is integer-typed
  (`list[tuple[int, ...]]`), so a coherent Euclidean fix there needs a separate continuous
  path-history change. Left as a deferred follow-up.

## Capabilities

### New Capabilities

<!-- None — this extends the existing continuous-2D environment capability. -->

### Modified Capabilities

- `continuous-2d-environment`: the "Euclidean predator detection, damage, and contact-zone geometry"
  requirement currently scopes Euclidean predator distance to detection/damage/contact-zone. This
  change extends it to the **nearest-predator-distance query** (`get_nearest_predator_distance_for` /
  `get_nearest_predator_distance`), so the predator distance the reward consumes is Euclidean against
  `pos_continuous` on the continuous substrate. Grid byte-stability is preserved.

## Impact

- **Code:**
  - `packages/quantum-nematode/quantumnematode/env/continuous_2d.py` — add Euclidean overrides of
    `get_nearest_predator_distance_for` and `get_nearest_predator_distance` (reuse the existing
    `_agent_xy` / `_predator_xy` float helpers and `math.hypot`).
  - No change to `reward_calculator.py` (the reward formula is unchanged).
- **Tests:** continuous nearest-predator-distance is Euclidean (not Manhattan) against `pos_continuous`;
  grid nearest-predator-distance remains integer-Manhattan, byte-stable.
- **Downstream:** unblocks `T7.prep.reward_coherence` and the Stage-1 predator-difficulty calibration
  (now done against the coherent reward). No new dependencies.
