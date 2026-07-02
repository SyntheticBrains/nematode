# Tasks: Chemosensory Associative-Memory Probe

Extends the bit-memory task capability (archived change `add-bit-memory-positive-control`) — reuse its
cue/go channels, phase/trial machine, no-aid contract, binary readout, reward-in-runner pattern, and
separation harness; the new surface is the **outcome (valence) channel** and the **two-cue
conditioning** phase.

## 1. Observation channels (add the outcome/valence channel)

- [ ] 1.1 Add an `OUTCOME` member to the `ModuleName` enum + register it in `SENSORY_MODULES`
  (`brain/modules.py`) with `classical_dim = 1`, an extractor reading a new `BrainParams.outcome_signal`
  field, and a description marking it an associative-memory task channel. Reuse the existing `CUE` and
  `GO_SIGNAL` modules for cue-identity + go.
- [ ] 1.2 Add `outcome_signal: float | None` to `BrainParams` (`brain/arch/_brain.py`), default `None`
  (treated as `0.0`). `cue_signal` / `go_signal` already exist (bit-memory).
- [ ] 1.3 Unit test: `get_classical_feature_dimension([cue, outcome, go_signal]) == 3`; with the three
  fields set, `extract_classical_features` yields the expected 3-dim observation; unset → zeros.

## 2. Configuration schema

- [ ] 2.1 Add an `AssociativeMemoryTaskConfig` block to `utils/config_loader.py`, off by default, with
  within-window starter defaults (kept under the Transformer `window_size = 16`): `enabled = False`,
  `trials_per_episode = 20`, `cond_steps_per_cue = 1` (two cues → 2 conditioning steps; the reversal
  block reuses it → 2 more on reversal trials), `reversal_prob = 0.5`, `delay_steps = 8`,
  `response_steps = 1`, `reward_correct = 1.0`, `penalty_wrong = 0.0` (worst-case span with reversal =
  `2·cond + 2·cond + delay + response = 13 < 16`).
- [ ] 2.2 Validation (mirror `BitMemoryTaskConfig`, config_loader.py:1062): assert
  `2·cond_steps_per_cue + delay_steps + response_steps >= 1` and `0 <= reversal_prob <= 1`; warn when
  the **worst-case (reversal) per-trial span** (`4·cond_steps_per_cue + delay_steps + response_steps`)
  exceeds a referenced attention `window_size` (Transformer-confound guard, D1); and **pin
  `reward_correct = 1.0` / `penalty_wrong = 0.0`** via a validator, since the harness metric
  (accuracy = reward / num_responses) holds only at those values.
- [ ] 2.3 Config-load test: a config with the task enabled parses; a config without it is unchanged;
  an unknown key under `associative_memory_task` raises (pydantic field validation), not silently
  dropped.
- [ ] 2.4 **No-external-memory-aid invariant (validity canary — D3, spec "No external memory aids").**
  When enabled, assert at config-resolve that the sensing mode is oracle/none, `stam` is **absent** from
  the resolved `sensory_modules` (guards the STAM auto-injection path), and the assembled observation
  dimension is exactly 3 (cue + outcome + go). Fail loudly if violated.

## 3. Task mechanics (env exposes → agent injects → runner scores)

- [ ] 3.1 **Dedicated task class (mirror `env/bit_memory.py`).** Create `env/associative_memory.py`
  with an `AssociativeMemoryTask` class paralleling `BitMemoryTask`: an `AssociativeMemoryPhase` enum
  (conditioning / **reversal** / delay / response), a per-trial sample of the initial rewarded cue ∈
  {A, B}, presentation order, and a reversal draw (`rng.random() < reversal_prob`) from an injected RNG;
  the trial's **current rewarded cue** = initial ⊕ reversed. `advance()` steps conditioning →
  (reversal if drawn) → delay → response → next trial. `signals()` returns `(cue_identity, outcome,
  go)` — the current presentation step's cue `+1`/`-1` + outcome (`+1` rewarded / `-1` else, **flipped**
  during the reversal block), or the go flag in the response phase, all zero in delay/response.
  `record_response()` scores against the **current** rewarded cue; `take_reward()`, `reset()`,
  `rebind_rng()`, `done()`, `num_responses()`, accuracy (overall + reversal / non-reversal split). The
  env holds `self.associative_memory: AssociativeMemoryTask | None` + a thin `get_associative_signals()`
  getter (mirroring `self.bit_memory` / `get_bit_memory_signals`, env.py:1085).
- [ ] 3.2 **Agent — cue/outcome/go injection into `BrainParams`.** In `agent._create_brain_params`,
  read `get_associative_signals(...)` and populate `cue_signal` / `outcome_signal` / `go_signal` (cue +
  outcome during the conditioning **and reversal** blocks; go = `1` during response, else `0`). The env
  does not push into `BrainParams`.
- [ ] 3.3 **Runner — dedicated step + scoring + termination (mirror the bit-memory path).** Add
  `_run_associative_memory_step` / `_terminate_associative_memory` / `_associative_memory_turn` paralleling
  `_run_bit_memory_step` / `_terminate_bit_memory` / `_bit_memory_turn` (`runners.py:696/651/624`),
  dispatched when `associative_memory_task.enabled`: on response steps read the binary response
  (`sign(top_action.continuous[1])` continuous; `LEFT`/`RIGHT` discrete) → `am.record_response`, compare
  to the trial's **current** rewarded cue, grant the reward via `am.take_reward`, and drive episode-done
  off `am.done()`.
- [ ] 3.4 **Foraging deactivation (via the dedicated step).** As bit-memory, `_run_associative_memory_step`
  is the *entire* step when enabled — it does **not** invoke the food / `_handle_predator_phase` /
  `_handle_temperature_effects` / `_handle_oxygen_effects` / `_handle_starvation_check` / satiety-decay
  handlers, and the action is consumed only as the response (movement inert). Verify no handler path leaks.
- [ ] 3.5 Surface the per-episode response accuracy for the harness (with `reward_correct = 1`,
  `penalty_wrong = 0` the episode reward equals the correct count → accuracy = `reward / num_responses`);
  log `AssocMemory: accuracy=… (reversal=… non_reversal=…)` at episode end so the update demand is
  readable directly.
- [ ] 3.6 Tests: phase transitions land on the configured boundaries (incl. the reversal block only on
  reversal trials); cue-identity + outcome are exactly `0` on every delay/response step **and the
  assembled observation is exactly 3-dim with no `stam`** (validity invariant, §2.4); go = `1` only in
  response; per-trial rewarded-cue + reversal randomisation; the reversal block presents **flipped**
  outcomes and sets the current rewarded cue to the post-reversal one; scoring is against the current
  rewarded cue (`sign(turn)`); a hold-only response (always the initial cue) is correct on non-reversal
  trials and wrong on reversal trials; no reward in conditioning/reversal/delay; episode-done off the
  trial counter; **disabled = byte-identical**.

## 4. Per-arm configs

- [ ] 4.1 Author `configs/scenarios/associative_memory/{mlpppo,lstmppo,cfcppo,transformerppo,mingruppo,minlstmppo}_small_associative_memory.yml`
  (no `{sensing}` suffix — a task variant, per the bit-memory precedent + the AGENTS.md note):
  `sensory_modules: [cue, outcome, go_signal]`, the `associative_memory_task` block (span `<
  transformerppo.window_size`), **matched `entropy_coef` across arms** (D7), each arm's action head, and
  a fixed seed block for paired-seed runs. **Skip connectome** (at-chance on bit-memory, D7).
- [ ] 4.2 Smoke: each config loads + runs a short headless episode (`--theme headless`) without error and
  emits the per-episode accuracy metric.

## 5. Separation analysis harness

- [ ] 5.1 `scripts/analysis/associative_memory_separation.py`: read each run's per-episode reward from the
  `.out` → accuracy (= `reward / num_responses`, `num_responses = trials_per_episode × response_steps`),
  per-arm plateau-tail (final-quarter) mean, and pairwise paired-seed deltas (one-sided Wilcoxon + 80%
  bootstrap CI + BH-FDR) by importing `weight_search_architecture_ranking` helpers; also parse the logged
  **reversal / non-reversal accuracy split** so per-arm update deficits are visible; print + write a JSON
  summary with per-arm accuracy (overall + split), the pairwise table, and the separation verdict.
- [ ] 5.2 Test the metric extraction + verdict logic on a tiny synthetic fixture (memory arms high, MLP
  at chance → "separation"; all at chance → "null").

## 6. Learnability pre-check + calibration

- [ ] 6.1 Pre-check (guards null-vs-mis-built): confirm at least one memory arm (LSTM or Transformer)
  clears the accuracy threshold on the easiest setting (short delay, generous budget) **before** the full
  panel; if not, debug delay/budget/reward, not the verdict.
- [ ] 6.2 Calibrate `cond_steps_per_cue` / `delay_steps` / `response_steps` / `reversal_prob` /
  `trials_per_episode` / budget / matched `entropy_coef` so the task is learnable by the memory arms with
  the worst-case (reversal) span within the Transformer window; record the calibrated values + the
  pre-registered success threshold.

## 7. Evaluation + verdict

- [ ] 7.1 Run the arm panel (MLP + LSTM/CfC/Transformer/minGRU/minLSTM) at `n ≥ 8` paired seeds, headless,
  parallelised (`OMP_NUM_THREADS=1`, `xargs -P`).
- [ ] 7.2 Compute the separation (harness §5); record per-arm accuracy, the pairwise BH-FDR table, and
  whether the memory arms clear both chance and the MLP.
- [ ] 7.3 **PAUSE for user review of the evaluation + verdict before writing the logbook** (project
  convention).

## 8. Logbook + tracker

- [ ] 8.1 Write the logbook (objective / method / results / analysis / limitations) + committed supporting
  artefacts (no `tmp/` references).
- [ ] 8.2 Add the logbook row to `docs/experiments/README.md`.
- [ ] 8.3 Tick `T7.separation.associative_memory` in `openspec/changes/phase6-tracking/tasks.md` with the
  verdict (separation = a second naturalistic memory data point; null = reported as such).
- [ ] 8.4 Document the `_associative_memory` config variant / the `associative_memory/` scenario family in
  `AGENTS.md`.

## 9. Pre-merge gates

- [ ] 9.1 Targeted `pre-commit` during iteration; full `pre-commit run -a` before push.
- [ ] 9.2 `openspec validate add-associative-memory-probe --strict`.
- [ ] 9.3 Full `uv run pytest -m "not nightly"` green (disabled = byte-identical holds).
- [ ] 9.4 Archive the change in-PR (`openspec archive add-associative-memory-probe -y`).
