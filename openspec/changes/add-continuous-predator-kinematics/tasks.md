## 1. Predator continuous state (additive, grid byte-stable)

- [ ] 1.1 Add optional `pos_continuous: tuple[float, float] | None = None` and continuous `heading_rad: float = 0.0` fields to `Predator` (env.py), documented as continuous-substrate-only (None/unused on the grid).
- [ ] 1.2 Add a small helper on `Predator` (or `Continuous2DEnvironment`) that sets the integer `position` as the `round()`-and-`[0, grid_size-1]`-clamped view of `pos_continuous`, mirroring the agent `_discretise` pattern. Used after every continuous spawn/move.
- [ ] 1.3 Confirm the grid path never reads the new fields; run the predator byte-equivalence suite to prove the grid `Predator`/movement is unchanged.

## 2. Float predator placement (continuous override)

- [ ] 2.1 Override `_initialize_predators` in `Continuous2DEnvironment` to sample candidate predator coordinates as real-valued floats in `[0, world_size_mm)` via `self.rng`, retaining the existing Euclidean min-separation-from-agent spawn check; set `pos_continuous` + synced integer `position` + initial `heading_rad`.
- [ ] 2.2 Reuse `_make_predator` for construction (carry detection/damage radii + speed); ensure stationary vs pursuit typing is preserved.

## 3. Continuous predator kinematics (continuous override)

- [ ] 3.1 Override predator movement on the continuous substrate (override `update_predators`, or a continuous `update_position` path) so it bypasses the cardinal `PredatorBrain` and applies the analytic `(speed, heading)` rule from design D3.
- [ ] 3.2 Pursuit: when an agent is within the Euclidean `detection_radius`, orient `heading_rad` toward the bearing to the nearest agent's `pos_continuous`, then advance by `speed · max_step_mm` along the new heading.
- [ ] 3.3 Wander: when no agent is in range (and not stationary), perturb `heading_rad` by a bounded random angle from `self.rng`, then advance by `speed · max_step_mm`.
- [ ] 3.4 Stationary predators do not move (early return preserved). Drop the integer `movement_accumulator` multi-step logic on the continuous path (speed scales displacement directly).
- [ ] 3.5 Clamp the new position per-axis to `[0, world_size_mm]` (partial move, no rejection/error), then sync the integer `position` view (task 1.2).

## 4. Euclidean detection / damage / contact zones (continuous override)

- [ ] 4.1 Override `is_agent_in_danger_for` in `Continuous2DEnvironment` to use `math.hypot` between predator `pos_continuous` and agent `pos_continuous` vs `detection_radius` (Euclidean-mm).
- [ ] 4.2 Override `is_agent_in_damage_radius_for` to use the same Euclidean distance vs each predator's `damage_radius`.
- [ ] 4.3 Override `get_agent_predator_contact_zone_for` to (a) select the nearest predator within its damage radius by Euclidean distance and (b) classify the approach cone using the worm's continuous forward unit vector from `heading_rad` (same convention as `_kinematic_move`/the renderer), retaining the ±45° anterior/lateral/posterior cones.
- [ ] 4.4 Confirm the predator-evasion reward formula is untouched (only the distance metric changes — RQ5 guardrail).

## 5. Renderer follow-through

- [ ] 5.1 Remove/relax any continuous-renderer visual-jank workaround for quantised predators now that predator motion is smooth; confirm predator sprites + detection/damage rings read the float-capable position correctly.

## 6. Tests

- [ ] 6.1 Continuous predator movement: pursuit predator steers toward the agent and advances by a sub-cell real-valued displacement (not an integer-cell jump); position stays within world bounds; integer `position` equals the rounded float.
- [ ] 6.2 Wandering predator advances continuously and stays in bounds; stationary predator never moves.
- [ ] 6.3 Euclidean detection/damage: an agent at a known float offset is in danger/damage iff Euclidean distance ≤ radius (verify a diagonal case that Manhattan and Euclidean would classify differently).
- [ ] 6.4 Contact-zone geometry: predator dead ahead → ANTERIOR, behind → POSTERIOR, abeam → LATERAL, using `heading_rad`.
- [ ] 6.5 Float spawn: predators spawn at non-integer coordinates within bounds and outside the min-separation radius.
- [ ] 6.6 Grid byte-stability: the existing predator byte-equivalence suite still passes (grid path unchanged).

## 7. Validation + gates

- [ ] 7.1 `openspec validate add-continuous-predator-kinematics --strict` passes.
- [ ] 7.2 Targeted `pre-commit run --files <changed>` (ruff/pyright/markdownlint) green during iteration; full `pre-commit run -a` before push.
- [ ] 7.3 Full `uv run pytest -m "not nightly"` green.
- [ ] 7.4 Tick `phase6-tracking` `T7.prep.predator_kinematics` and record the Manhattan→Euclidean metric-shift as a validate-don't-retune item for the T7 C2 predator smoke.
