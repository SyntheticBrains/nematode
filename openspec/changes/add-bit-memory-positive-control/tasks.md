# Tasks: Bit-Memory Working-Memory Positive Control

## 1. Observation channels (cue + go-signal sensory modules)

- [x] 1.1 Add `CUE` and `GO_SIGNAL` members to the `ModuleName` enum and register them in `SENSORY_MODULES` (`brain/modules.py`), each with `classical_dim = 1`, an extraction function reading the new `BrainParams` fields, and a description marking them as bit-memory task channels.
- [x] 1.2 Add `cue_signal: float | None` and `go_signal: float | None` fields to `BrainParams` (`brain/arch/_brain.py`); default `None` (treated as `0.0` by the extractors).
- [x] 1.3 Unit test: registering both modules with `classical_dim = 1` is load-bearing (the unknown-module fallback in `extract_classical_features` emits 2 zeros, which would make input_dim = 4). Assert `get_classical_feature_dimension([cue, go_signal]) == 2`; with `cue_signal`/`go_signal` set, `extract_classical_features([cue, go_signal], params)` yields the expected 2-dim observation; with them unset it yields zeros.

## 2. Configuration schema

- [x] 2.1 Add a `BitMemoryTaskConfig` block to `utils/config_loader.py` and wire it into the environment config; off by default. Give the fields sensible within-window **starter defaults** (refined by the calibration in §6.2, kept under the Transformer's `window_size = 16`): `enabled: bool = False`, `trials_per_episode = 20`, `cue_steps = 2`, `delay_steps = 8`, `response_steps = 1`, `reward_correct = 1.0`, `penalty_wrong = 0.0` — so the schema is usable before calibration.
- [x] 2.2 Validation: assert `cue_steps + delay_steps + response_steps >= 1` and (with a warning) flag when the per-trial span exceeds a referenced attention `window_size` (the Transformer-confound guard from design Decision 3).
- [x] 2.3 Config-load test: a config with the task enabled parses; a config without it is unchanged. (NB the `#253` unknown-key warning is brain-config-only; the env-side `BitMemoryTaskConfig` is guarded by pydantic field validation instead — assert an unknown key under `bit_memory_task` raises rather than silently dropping.)
- [x] 2.4 **No-external-memory-aid invariant (hard assertion, the validity canary — design Risks / spec "No external memory aids").** When `bit_memory_task.enabled`, assert at config-resolve time that the sensing mode is oracle/none, that `stam` is **absent** from the resolved `sensory_modules` (guarding the `apply_sensing_mode` / `validate_sensing_config` STAM auto-injection path), and that the assembled observation dimension is exactly 2 (cue + go). Fail loudly if violated — a leaked cue invalidates the whole control.

## 3. Task mechanics (env exposes → agent injects → runner scores)

> Placement verified against the agent/runner architecture: there is no scoring `env.step()`. The env owns the phase machine and *exposes* cue/go state; the agent's param builder *reads* it into `BrainParams`; the runner scores the post-action response and drives termination/deactivation (design Decision 6).

- [x] 3.1 **Env — phase machine + cue state.** Add a config-gated state machine to the environment (`env/env.py`, inherited by `continuous_2d.py`): per-episode trial counter, per-trial cue sampled uniformly from the env RNG, `(phase, step_in_phase)` advancing cue → delay → response → next trial, and a getter (e.g. `get_bit_memory_signals(agent_id)`) returning the current cue value (cue phase only) and go flag (response phase only).
- [x] 3.2 **Agent — cue/go injection into `BrainParams`.** In `agent._create_brain_params` (the method that already pulls per-agent env state such as `predator_contact_zone`), read the env's `get_bit_memory_signals(...)` and populate the new `cue_signal` / `go_signal` fields (cue value during the cue phase, `0` during delay/response; go = `1` during response, else `0`). The env does **not** push into `BrainParams` (spec: cue/go channels).
- [x] 3.3 **Runner — post-action response scoring + termination.** In the `StandardEpisodeRunner.run` loop, add a `bit_memory_task.enabled` post-action branch: on response steps read the binary response from the just-emitted `top_action` (`sign(top_action.continuous[1])` for continuous arms; `LEFT`/`RIGHT` for discrete), compare to the trial cue, grant `reward_correct` on a match / the wrong-response outcome otherwise (parameters from `BitMemoryTaskConfig`), record per-trial correctness, and drive episode-done off the **trial counter** rather than `env.reached_goal()` (spec: cue-conditioned reward, binary readout).
- [x] 3.4 **Runner/config — foraging deactivation (enumerate the touch-points).** When the task is enabled, bypass the runner's independent handler calls — food collection, `_handle_predator_phase`, `_handle_temperature_effects`, `_handle_oxygen_effects`, `_handle_starvation_check`, and satiety decay — and make agent movement inert (the action is consumed only as the response, never as locomotion), via a runner-level early-branch or env-config flags that no-op each handler. Disabling them "in the environment" alone is insufficient — these are runner calls (design Decision 6 / review S4).
- [x] 3.5 Expose per-episode cue-match success for experiment tracking (the metric the analysis harness consumes).
- [x] 3.6 Tests: phase transitions land on the configured boundaries; the cue channel is exactly `0` on every delay/response step **and the assembled observation is exactly 2-dim with no `stam` channel** (the validity invariant, §2.4); `go_signal` is `1` only during the response phase; `sign(turn)` correct vs wrong scoring; no cue reward in cue/delay phases; episode-done fires off the trial counter; **disabled = byte-identical existing behaviour**.

## 4. Per-arm configs

- [x] 4.1 Author `configs/scenarios/bit_memory/{mlpppo,lstmppo,cfcppo,transformerppo,connectomeppo}_small_bit_memory.yml` (the filename carries **no `{sensing}` suffix** — a deliberate, documented deviation from the `{brain}_{size}[_{variant}]_{sensing}` convention in AGENTS.md, since bit-memory is a task variant whose input is the cue/go channels, not a sensing mode): `sensory_modules: [cue, go_signal]`, the `bit_memory_task` block (span kept `< transformerppo.window_size` per design Decision 3), each arm's existing action head, and a fixed seed block for paired-seed runs.
- [x] 4.2 Smoke: each config loads and runs a short headless episode (`--theme headless`) without error and produces the per-episode cue-match success metric.

## 5. Separation analysis harness

- [x] 5.1 `scripts/analysis/bit_memory_separation.py`: read each run's per-episode cue-match success, compute per-arm mean success over the converged tail, and the pairwise paired-seed deltas (one-sided Wilcoxon + 80% bootstrap CI + BH-FDR) by importing the helpers from `weight_search_architecture_ranking` (same methodology as the T7 ranking). Print + write a JSON summary with per-arm success, the pairwise table, and the separation verdict.
- [x] 5.2 Test the metric extraction + verdict logic on a tiny synthetic fixture (memory arms high, MLP at chance → "separation"; all at chance → "null").

## 6. Learnability pre-check + calibration

- [x] 6.1 Pre-check (guards the null-vs-mis-built risk, design Risks): confirm at least one memory arm (LSTM) reaches the success threshold on the easiest setting (short delay, generous budget) **before** running the full panel; if it cannot, debug delay/budget/reward, not the gate verdict.
- [x] 6.2 Calibrate `cue_steps` / `delay_steps` / `response_steps` / `trials_per_episode` / training budget so the task is learnable by the memory arms while the span stays within the Transformer window; record the calibrated values and the pre-registered success threshold (design Open Questions).

## 7. Evaluation + verdict

- [x] 7.1 Run the 5 MUST arms at `n ≥ 8` paired seeds, headless, parallelised (`OMP_NUM_THREADS=1`, `xargs -P`).
- [x] 7.2 Compute the separation (harness from §5); record per-arm cue-match success, the pairwise BH-FDR table, and whether the recurrent/attention arms clear both chance and the MLP (spec: separation evaluation).
- [x] 7.3 Interpret the connectome result against design Decision 7 (cross-step state vs within-step settling).

## 8. Tracker + supporting analysis

- [x] 8.1 Tick `T7.separation.bit_memory_control` in `openspec/changes/phase6-tracking/tasks.md`, record the verdict (separation vs null), and update the conditional status of `T7.separation.ars_depletion` and `T7.separation.new_arch_candidates` accordingly (proceed on separation; defer on a null, recording the null as the finding).
- [x] 8.2 Write the supporting analysis (per-arm success, pairwise table, verdict, calibration) under the relevant logbook supporting folder — committed artefacts only, **no `tmp/` references**.

## 9. Pre-merge gates

- [x] 9.1 `openspec validate add-bit-memory-positive-control --strict` passes.
- [x] 9.2 Targeted `pre-commit` (ruff / pyright / markdownlint) on changed files during iteration; full `uv run pytest -m "not nightly"` before push.
- [ ] 9.3 After merge, archive the OpenSpec change (`openspec archive add-bit-memory-positive-control`) so the `bit-memory-positive-control` delta applies into `openspec/specs/`.
