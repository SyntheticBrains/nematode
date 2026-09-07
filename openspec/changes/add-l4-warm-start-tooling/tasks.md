# Tasks: warm-start tooling

## 1. Connectome weight persistence

- [ ] 1.1 `get_weight_components` / `load_weight_components` on `ConnectomePPOBrain`: `topology`
  (full topology state incl. `m_chem`, `g_gap`), `value` + `optimizer` under PPO,
  `training_state`; load validates std mode and wiring before mutating, resets the rollout
  buffer and the plastic rule's running state.
- [ ] 1.2 Tests: bit-identical round trip under PPO and under the plastic rule on the C3 configs;
  wiring and std-mode mismatches refused before mutation; PPO components present only under
  PPO and ignored by a plastic brain; resets; frozen-reference and wiring-arm tests unchanged.

## 2. The reported action mean

- [ ] 2.1 `ActionData.continuous_mean`; set on the continuous path by the MLP-PPO and connectome
  brains from the same forward as the sample.
- [ ] 2.2 Tests: equals the squashed, rescaled mean on both brains; `None` on a discrete brain.

## 3. Rollout recording

- [ ] 3.1 `RolloutRecorder` (JSON lines: episode, step, params, action, action_mean, probability;
  per-episode flush); attached by `--record-rollouts PATH`; runners call it after `run_brain`
  and make no call when absent.
- [ ] 3.2 Tests: a short headless run writes one line per step with the recorded `BrainParams`
  matching the brain's history; no flag → no file, no call.

## 4. The cloning trainer

- [ ] 4.1 `scripts/campaigns/l4_behavioural_clone.py`: build the student at the seed through the
  entry point's factory; preprocess through the student's feature path; batched forward →
  readout → squashed mean; action-space MSE; Adam over `plastic` (masked `w_chem`) or `full`;
  seeded holdout; losses and norm change reported; refuse to save without improvement; save
  via `save_weights` with `clone.json` beside it.
- [ ] 4.2 Tests: self-cloning on a small foraging config (held-out loss down an order of
  magnitude; `plastic` leaves gains/readout/noise bit-identical, `full` changes them; the saved
  file loads and reproduces the student's mean on a recorded observation).

## 5. Close-out

- [ ] 5.1 `docs/architectures.md`, `docs/usage.md`, `CHANGELOG.md`; issue #308 referenced in the
  PR as closed.
- [ ] 5.2 Pre-commit gate on all files exit 0; full suite green.
- [ ] 5.3 No implementation code or docstring references a planning document.
- [ ] 5.4 Re-review for drift, archive, review the branch, open the PR.
