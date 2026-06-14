# Tasks

## 1. Implementation

- [x] 1.1 Add `max_turn_rad: float` to `Continuous2DParams` (`env/continuous_2d.py`), documented as the
  max per-step angular velocity (rad), the rotational analogue of `max_step_mm`. Default = the
  re-validated realistic value (task 3.3).
- [x] 1.2 In `move_agent_normalized`, rescale `turn = turn_norm * self.continuous.max_turn_rad`
  (was `turn = turn_norm * math.pi`); update the method docstring (turn maps to `[-max_turn_rad, +max_turn_rad]`).
- [x] 1.3 Add `Continuous2DConfig.max_turn_rad` (`Field(default=…, gt=0.0)`) in `utils/config_loader.py`
  and thread it through `create_env_from_config` into `Continuous2DParams`.

## 2. Tests

- [x] 2.1 `move_agent_normalized` rescales turn by `max_turn_rad`: with a configured `max_turn_rad`,
  `turn_norm = 1.0` rotates the heading by exactly `max_turn_rad` (and `-1.0` by `-max_turn_rad`).
- [x] 2.2 The default `max_turn_rad` is a realistic bound: `0 < default < math.pi`.
- [x] 2.3 `Continuous2DConfig.max_turn_rad` parses from YAML and the factory wires it into
  `Continuous2DParams`.
- [x] 2.4 Grid/discrete-action path is unaffected (does not use `move_agent_normalized`).

## 3. Validate + gate

- [x] 3.1 `openspec validate fix-continuous-turn-rate-bound --strict`.
- [x] 3.2 Targeted `pre-commit` on changed files; full `pre-commit run -a` before push;
  `uv run pytest -m "not nightly"` for the env suite.
- [x] 3.3 **Re-validation (downstream, recorded in the T7 scratchpad):** sweep `max_turn_rad ∈ {0.5, 0.79, 1.05}` on the C1 MLP foraging task; pick the tightest value that still converges to the
  10-food target; confirm the helicopter spin is gone on `--theme pixel_continuous`; set that as the
  `max_turn_rad` default.

**Re-validation result (2026-06-14):** swept {0.5, 0.79, 1.05}; foraging 79/88/100%, but 0.5 (29°) firms to ~98% with entropy 0.05 / ~3000 ep (it was under-trained), and predator evasion at 0.5 vs 1.05 is comparable (survival 60% vs 62%). **Locked the biologically realistic `max_turn_rad = 0.5` rad (~29°/step).** Carry-forward: the C1 foraging recipe gains `entropy_coef 0.05` (or ~3000 ep) at this bound.
