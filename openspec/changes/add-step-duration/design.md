## Context

The point worm's step is 5 worm-seconds. Nothing states that; it follows from three things that are
unlinked:

- `env/worm_time.py` infers it as `max_step_mm / CRAWL_SPEED_MM_PER_S` (0.2 mm/s), and the body reads
  the result;
- `ConnectomePPOBrainConfig.modulator_step_seconds` (5.0) sets the modulator decay `exp(−Δt/60 s)`;
- `validation/roaming_dwelling.py` hard-codes `STEP_SECONDS = 5.0` and `WINDOW_STEPS = 2`.

Every other step-coupled constant is written in steps, in configs that set it explicitly:

- 520 configs set `max_step_mm: 1.0`;
- 647 set the STAM buffer and decay;
- 791 set `gamma: 0.99`.

No test pins a default-config run's outputs, so byte identity at the default has to hold by
construction.

D35 (roadmap v5.0) fixes the Phase 10 point worm at 1 worm-second. The maintainer's decisions of
2026-10-23:

- **E.1 lands alone.** The locomotor primitives (E.2) are the next change.
- **Configs are authored at the reference step and derived.**
- **A non-default step is refused** outside lawn foraging with `connectomeppo` or `mlpppo`.

## Goals / Non-Goals

**Goals**

- One value, `step_seconds`, from which every step-coupled constant the lawn point worm reads is
  derived.
- Default runs byte-identical, by construction and by test.
- Physical quantities invariant under the derivation, each asserted by a test:
  - distance per second;
  - turn per second;
  - time to starve;
  - STAM half-life in seconds;
  - the adaptive sensor's time constant in seconds;
  - γ's horizon in seconds;
  - modulator decay per second;
  - lawn depletion per second.
- A 1-s connectome lawn worm and a 1-s MLP lawn worm, configured and smoked.

**Non-goals**

- Locomotor primitives, assay protocols and the batched evaluator (E.2, E.3, T.1).

- Re-calibrating any instrument at 1 s. The classifier keeps its 5-s calibration by downsampling, and
  the bias-curve statistics are repaired in B.2.

- Derivations for the frozen scenarios, the body, plasticity or the other 24 brains.

- Training-budget settings:

  - `rollout_buffer_size`;
  - `bptt_chunk_length`;
  - `num_epochs`;
  - learning rate;
  - entropy coefficient;
  - episodes per run.

  These are hyperparameters, not physical constants, and C.3's registration chooses them.

- Any claim that a 1-s worm learns or behaves like a 5-s one. The derivation preserves the
  environment's physics and the learner's discount horizon in seconds. Learning itself is not
  preserved, because a 1-s worm makes five times as many decisions per worm-second.

## Decisions

### Decision A: One top-level key, the reference step inferred

`SimulationConfig.step_seconds: float | None = Field(default=None, gt=0)`. It is top-level because
the derived constants live in the environment, the agent's satiety and reward, the brain and
`max_steps`.

The **reference step** `s_ref` is the duration the config's per-step values are written for. It is
`step_worm_seconds(max_step_mm)` on the config *as written*: 5.0 at 1 mm. With `s = step_seconds`, the
**ratio** is `r = s_ref / s`.

Load-time validation:

- `r` must be a whole number within 1e-9, from 1 to 100. Counts then multiply exactly, and a track
  downsamples by an integer stride. A 1-s step on the 5-s reference gives `r = 5`. A longer step than
  the reference is refused: no Phase 10 use needs it, and counts would have to divide.
- `max_step_mm` must be positive when `step_seconds` is set, or the reference is undefined.
- `r = 1` derives nothing: the config is returned unchanged.

Rejected: a separate `reference_step_seconds` key. It would be a second time value the crawl-speed
relation already fixes, and the two could disagree.

### Decision B: The derivation is one function at load, idempotent

`derive_step_coupled(config) -> SimulationConfig` lives in `config_loader.py` and is called at the end
of `load_simulation_config`. Every entry point that reads a YAML therefore sees derived values:
`run_simulation`, `run_evolution`, `run_campaign`'s children, and the analysis scripts that load
configs. It returns a new config via `model_copy(update=...)` on each sub-model. It never mutates.

**Idempotence.** The derived config's `max_step_mm` is `s × CRAWL_SPEED_MM_PER_S`, so its own reference
step is `s` and its ratio is 1. A second application derives nothing. A test applies the function twice
and compares the results.

**The record.** At load, the loader logs one INFO line per derived constant, with its authored and
derived values. The run log therefore records what ran, beside the existing `Config:` line.

### Decision C: The derivation table

Write `k = 1/r = s/s_ref`. Each step-coupled constant on the lawn point worm has one rule:

| Kind | Rule | Constants |
|---|---|---|
| amount per step | `x · k` | `continuous.max_step_mm`, `continuous.max_turn_rad`, `satiety.satiety_decay_rate`, `lawns.regrowth_per_step`, `reward.penalty_step`, `reward.penalty_boundary_collision`, `brain.config.modulator_step_seconds` (connectome) |
| fraction removed per step | `1 − (1 − p)^k` | `lawns.intake_fraction`, `sensing.adaptive_chemosensor_alpha` |
| discount per step | `d^k` | `brain.config.gamma`, `brain.config.gae_lambda` |
| exponent rate per step | `λ · k` | `sensing.stam_decay_rate` (weights `exp(−λ i)`) |
| length in steps | `n · r` (exact integer) | `max_steps`, `sensing.stam_buffer_size`, `reward.stuck_position_threshold` |
| time constant in steps | `τ · r` | `brain.config.membrane_tau_steps` (connectome, leaky) |
| per-step difference feature | `scale · r` | `sensing.derivative_scale` |

Why each rule:

- **Linear amounts** keep the rate per second. A worm at full speed covers 0.2 mm/s at either step, and
  turns at most `max_turn_rad / s_ref` rad/s. On D.1's lawn cell (`max_turn_rad: π`) a 1-s worm turns
  at most π/5 per step, so a 180° change of heading takes 5 s. Backing up through signed speed is
  still a single step. E.2's reorientation primitive supersedes this.
- **Compounded fractions** keep the remaining amount after any whole number of reference steps exact.
  `(1 − p')^r = 1 − p`, so lawn depletion and the adaptive baseline's relaxation per second match.
- **Discounts** keep the discount over t worm-seconds, `γ^(t/s)`, equal, so the return's effective horizon, `1/(1 − γ)`, is
  100 reference steps, 500 s, at either step. GAE λ is treated the same way, so the advantage's bias–variance trade is
  made over the same span of worm time.
- **Lengths in steps** keep their span in seconds: the STAM buffer stays 150 s, an episode of 720
  reference steps stays one hour, and the stuck threshold keeps its window.
- **The STAM rate** keeps each sample's weight as a function of its age in seconds, so the half-life
  stays about 35 s (6.9 reference steps).
- **The membrane time constant** stays in seconds, `τ_steps · s`. `forward_pass_depth` is not
  derived: it is propagation hops per decision, not a time constant. The leaky sub-step is therefore
  `s / depth`, five times finer at 1 s. That is a smaller Euler step on the same ODE, not a different
  model.
- **The derivative feature.** STAM's dC/dt and the klinotaxis derivative are per-step differences, and
  no code divides them by Δt. At 1 s they shrink about five-fold, so the scale grows by `r` and the
  feature stays a change per reference step. The fold-change readout divides that same difference by
  `C + ε`, and is covered by the same scale.

**Not step-coupled, and left alone:** event rewards (`reward_goal`, `penalty_starvation`,
`reward_per_intake`), `satiety_gain_per_food`, `satiety_per_intake` (per unit eaten),
`movement_cost_per_mm` (per mm), `initial_satiety`, every spatial setting, `forward_pass_depth`, the
modulators' 60-s timescale (already in seconds), and the training-budget settings (Non-goals).
`max_food_ingested` is the largest intake one step can yield, so the ingestion drive is a fraction of a
full bite at either step, with no rule of its own.

**Reward terms the lawn cell must already set to zero** (`penalty_stuck_position`,
`penalty_anti_dithering`, `reward_exploration`, `reward_distance_scale`) are refused non-zero at a
non-default step. Each is either a shaping term the lawn validator already forbids, or one whose
meaning depends on the step ("where it was two steps before"). Their rules are never needed.

### Decision D: Refusals (D41's scope)

A config with `step_seconds` set and `r ≠ 1` is refused at load, the message naming the offending key
and the reason, when any of the following holds:

- `environment.env_type` is not `continuous_2d`;
- `foraging.food_model` is not `lawns`;
- thermotaxis, aerotaxis, predators, pheromones or social feeding are enabled;
- `multi_agent` is set;
- a memory task is set (`bit_memory`, `associative_memory`);
- `continuous.body_model` is `kinematic` (the body opens Phase 11; its reversal and refractory counts
  would need their own rules);
- `brain.name` is not `connectomeppo` or `mlpppo`;
- within-lifetime plasticity is on (`learning_rule ≠ ppo`, or `enable_activity_traces`). Its trace
  decay and rate are per-step and would need their own validation.

Each refusal has a test. A frozen scenario keeps working at its default.

### Decision E: The environment's step duration

`Continuous2DParams` gains `step_seconds: float | None = None`, mapped from the top-level key by
`create_env_from_config`. `ContinuousEnvironment` sets `BodyParams.step_seconds` from it when set, and
from `step_worm_seconds(max_step_mm)` otherwise, which is today's path. The kinematic body is refused at
a non-default step (Decision D), so the body only ever sees today's value or an equal one. The replay
already records `env.body_params.step_seconds`.

### Decision F: Captures and the classifier

- **Capture.** `write_behaviour_capture` adds a top-level `"step_seconds"` to `behaviour_capture.json`
  only when the run's step differs from 5.0. A default capture is byte-identical. A reader treats an
  absent key as 5.0.
- **Classifier.** A new `downsample_to_reference(x, y, step_seconds)` keeps every `r`-th position for a
  track sampled at a step that divides 5 s. It refuses any other step. `lawn_states.py` and
  `patch_leaving_reference.py` take the step from the capture or the config, never from
  `rd.STEP_SECONDS`, and downsample before `window_measures`. The HMM and its 10-s windows are applied
  unchanged. This is the roadmap's registered fallback. B.2 checks it on real tracks re-sampled to 1 s
  and 5 s.
  - **Per-step measures at 1 s.** Satiety, intake and on-lawn status are read at the downsampled
    positions' steps. Intake is summed over each 5-s block, so on-lawn hours and intake totals are
    unchanged in meaning.
- **Bias-curve harness.** `behavioural_chemotaxis_validation.py` refuses a capture with
  `step_seconds ≠ 5`, naming B.2. Its turn-rate ratio and `theta_sharp` are per-transition statistics
  calibrated at 5 s.

### Decision G: The 1-s configs

`generate_lawn_configs.py` gains two arms. Each is its base config plus `step_seconds: 1.0`:

- `connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_modulated_s1.yml`;
- `mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1.yml`.

`test_lawn_configs.py` loads both and asserts their derived values. A smoke run of three episodes each,
on seeds 5991–5993, checks that they execute and that a captured episode classifies. It also records
wall-clock per worm-hour beside the 5-s arms', as a note for T.1, not a pilot.

## Positive control (protocol principle 4)

The new component is the derivation. Its positive control is a test that a known physical behaviour is
reproduced at both steps:

- **A scripted worm.** It is fixed at full speed on a straight heading, well clear of every lawn,
  with no learner. After 30 worm-seconds its position, path length (6 mm) and satiety match between
  `s = 5` (6 steps) and `s = 1` (30 steps), to within floating-point tolerance. Turning in place at full
  turn, its heading after 30 s matches too. Turning while moving is left out: a polygon's corners
  depend on the step, so the endpoints differ by discretisation, not by a fault.
- **A grazing worm.** It is held still on one cell. That cell's density after 60 s matches at both
  steps.
- **The STAM weights** at ages 0–150 s, and the adaptive baseline's step response at 5, 10, … 60 s,
  match at both steps.
- **The must-fail arm.** The same scripted worm with the derivation bypassed (per-step values at 1 s
  unconverted) is shown to move five times as far, so the test can fail.

## Risks / Trade-offs

- **Under-coverage.** A step-coupled constant missed by the table would silently keep its per-step
  value. Mitigation:
  - Decision D refuses everything outside the covered surface;
  - a test walks every field of the lawn point worm's config models, each assigned in a table to
    either a rule or "not step-coupled" with a reason, and fails on any unassigned field. A new field
    must therefore be classified.
- **Float drift at the default.** Avoided by not deriving at all when the key is unset or `r = 1`.
- **More decisions per worm-second.** PPO at 1 s gets five times as many updates' worth of steps per
  episode, against the same rollout buffer. That is a training-budget question for C.3's registration,
  stated there.
- **`max_turn_rad: π` at 1 s.** It becomes π/5 rad/s. Whether real reorientation rates exceed that is
  E.2's question, because E.2 replaces free turning with primitives.

## Migration

None. Every committed config leaves `step_seconds` unset. The modified spec requirement on the step
duration keeps its 5-s scenario.
