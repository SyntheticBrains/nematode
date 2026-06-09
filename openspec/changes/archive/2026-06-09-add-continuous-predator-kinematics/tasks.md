## 1. Predator continuous state (additive, grid byte-stable)

- [x] 1.1 Add optional `pos_continuous: tuple[float, float] | None = None` and continuous `heading_rad: float = 0.0` fields to `Predator` (env.py), documented as continuous-substrate-only (None/unused on the grid).
- [x] 1.2 Preserve the new `pos_continuous`/`heading_rad` fields across env `copy()` (which happens mid-episode for some brains/parallel paths) — otherwise continuous predators reset/teleport on copy. *As-built: set post-construction in `copy()` alongside the runtime metric counters (`kills`/`distance_traveled`), and in the subclass spawn helper `_spawn_continuous_predator`, rather than adding `_make_predator` kwargs — matching the existing state-vs-config split in `copy()`.*
- [x] 1.3 Keep the integer `position` as the `round()`-and-`[0, grid_size-1]`-clamped view of `pos_continuous` after every continuous spawn/move. *As-built: reuse the existing `Continuous2DEnvironment._discretise` (already used for the agent) rather than adding a new helper.*
- [x] 1.4 Confirm the grid path never reads the new fields; run the predator byte-equivalence suite to prove the grid `Predator`/movement/`copy()` are unchanged.

## 2. Float predator placement (continuous override)

- [x] 2.1 Override `_initialize_predators` in `Continuous2DEnvironment` to sample candidate predator coordinates as real-valued floats in `[0, world_size_mm)` via `self.rng`, retaining the existing Euclidean min-separation-from-agent spawn check (reuse the `MAX_POISSON_ATTEMPTS` retry loop); set `pos_continuous` + initial `heading_rad`, and the synced integer `position` view.
- [x] 2.2 Construct via `_make_predator` (carry detection/damage radii + speed; stationary vs pursuit typing preserved). Pass the **rounded int** as `position` to satisfy its `tuple[int, int]` contract, then set the float `pos_continuous` + a random initial `heading_rad` post-construction (the `_spawn_continuous_predator` helper).

## 3. Continuous predator kinematics (continuous override)

- [x] 3.1 Override predator movement on the continuous substrate (override `update_predators`, or a continuous `update_position` path) so it bypasses the cardinal `PredatorBrain` and applies the analytic `(speed, heading)` rule from design D3. Gather agent **float** positions (`pos_continuous`, fallback integer `position`) for nearest-target selection — the base `update_predators` builds integer `alive_positions`, which the continuous path must not use. Covers the multi-agent path (same method).
- [x] 3.2 Pursuit: when an agent is within the **per-predator** Euclidean `self.detection_radius`, orient `heading_rad` toward the bearing to the nearest agent's `pos_continuous`, then advance by `speed · max_step_mm` along the new heading.
- [x] 3.3 Wander: when no agent is in range (and not stationary), perturb `heading_rad` by a bounded random angle from `self.rng`, then advance by `speed · max_step_mm`.
- [x] 3.4 Stationary predators do not move (early return preserved). Drop the integer `movement_accumulator` multi-step logic on the continuous path (speed scales displacement directly).
- [x] 3.5 Clamp the new position per-axis to `[0, world_size_mm]` (partial move, no rejection/error), then sync the integer `position` view (task 1.2).

## 4. Euclidean detection / damage / contact zones (continuous override)

- [x] 4.1 Override `is_agent_in_danger_for` in `Continuous2DEnvironment` to use `math.hypot` between predator `pos_continuous` and agent `pos_continuous` vs the **env-level** `self.predator.detection_radius` (Euclidean-mm) — preserving the base method's env-level source (note the deliberate asymmetry vs the per-predator radius used by pursuit steering in 3.2).
- [x] 4.2 Override `is_agent_in_damage_radius_for` to use the same Euclidean distance vs each predator's `damage_radius`.
- [x] 4.3 Override `get_agent_predator_contact_zone_for` to (a) select the nearest predator within its damage radius by Euclidean distance and (b) classify the approach cone using the worm's continuous forward unit vector `(cos heading_rad, sin heading_rad)` in world coords (same convention as `_kinematic_move`), retaining the ±45° anterior/lateral/posterior cones.
- [x] 4.4 In all three overrides, fall back to the agent's integer `position` when `pos_continuous` is `None` (defensive — should not occur on the continuous substrate).
- [x] 4.5 Confirm the predator-evasion reward formula is untouched (only the distance metric changes — RQ5 guardrail).

## 5. Renderer follow-through

- [x] 5.1 Make the continuous renderer reflect the smooth float motion: `Continuous2DRenderer._render_entities` (predator sprites) and the predator detection/damage-ring drawing both currently read the integer `pred.position` via `_world_to_pixel(float(pred.position[0]), ...)` — switch them to read `pred.pos_continuous` (fallback `position`). Without this the physics is smooth but predators still render quantised (the motivating visible jank persists). No "jank workaround" exists to remove; this is the substantive renderer edit. *As-built: also updated the third `Continuous2DRenderer` predator-position site, `_render_toxic_zones` (stationary-predator damage discs), which the branch review caught still reading the integer `position`.*

## 6. Tests

- [x] 6.1 Continuous predator movement: pursuit predator steers toward the agent and advances by a sub-cell real-valued displacement (not an integer-cell jump); position stays within world bounds; integer `position` equals the rounded float.
- [x] 6.2 Wandering predator advances continuously and stays in bounds; stationary predator never moves.
- [x] 6.3 Euclidean detection/damage: an agent at a known float offset is in danger/damage iff Euclidean distance ≤ radius (verify a diagonal case that Manhattan and Euclidean would classify differently).
- [x] 6.4 Contact-zone geometry: predator dead ahead → ANTERIOR, behind → POSTERIOR, abeam → LATERAL, using `heading_rad`.
- [x] 6.5 Float spawn: predators spawn at non-integer coordinates within bounds and outside the min-separation radius.
- [x] 6.6 Grid byte-stability: the existing predator byte-equivalence suite still passes (grid path unchanged).
- [x] 6.7 *(branch-review additions)* Multi-agent pursuit targets the nearest **alive** agent (dead agents excluded); contact-zone overlap (`rel_len == 0`) → ANTERIOR; per-axis y-bound clamping.

## 7. Validation + gates

- [x] 7.1 `openspec validate add-continuous-predator-kinematics --strict` passes.
- [x] 7.2 Targeted `pre-commit run --files <changed>` (ruff/pyright/markdownlint) green during iteration; full `pre-commit run -a` before push. *(Targeted runs green; full `-a` pending the pre-push gate.)*
- [x] 7.3 Full `uv run pytest -m "not nightly"` green. *(3869 passed, 1 skipped, 2 xfailed.)*
- [x] 7.4 Tick `phase6-tracking` `T7.prep.predator_kinematics` and record the Manhattan→Euclidean metric-shift as a validate-don't-retune item for the T7 C2 predator smoke.
