## 1. Enrich Critic Input with Hidden Spike Rates

- [ ] 1.1 Add `episode_hidden_spikes: list[np.ndarray]` to episode buffer in `_init_episode_state()`
- [ ] 1.2 Store detached hidden spike rates (averaged across integration steps) in `run_brain()` when `use_critic` is True
- [ ] 1.3 Clear `episode_hidden_spikes` alongside other buffers in all reset/clear sites (`_reinforce_update`, `learn`, `_reset_episode`)
- [ ] 1.4 Update critic `input_dim` from `self.input_dim` to `self.input_dim + self.num_hidden` in `__init__`
- [ ] 1.5 Update `_update_critic()` to concatenate `episode_features` with `episode_hidden_spikes` for critic input batch
- [ ] 1.6 Update inline critic `V(s)` calls in `run_brain()` and deferred bootstrap to use concatenated features + hidden spikes

## 2. Improve CriticMLP Initialization

- [ ] 2.1 Add orthogonal weight initialization (gain=sqrt(2)) and zero bias initialization to `CriticMLP.__init__()`

## 3. Enhance Critic Diagnostics Logging

- [ ] 3.1 Add `explained_variance` computation to `_update_critic()`: `1 - Var(returns - predicted) / Var(returns)`
- [ ] 3.2 Upgrade `_update_critic()` logging from debug-level to info-level with fields: `value_loss`, `critic_pred_mean`, `critic_pred_std`, `target_return_mean`, `explained_variance`

## 4. Update copy() for New Buffer Field

- [ ] 4.1 Ensure `copy()` method handles the new `episode_hidden_spikes` buffer (deep copy or re-init)

## 5. Tests

- [ ] 5.1 Add test for critic instantiation with enriched input dim (`num_sensory + num_hidden`)
- [ ] 5.2 Add test for CriticMLP orthogonal initialization (check weight orthogonality and zero biases)
- [ ] 5.3 Add test for hidden spike rate storage in episode buffer during `run_brain()`
- [ ] 5.4 Add test for critic input construction (features + hidden spikes concatenation) in `_update_critic()`
- [ ] 5.5 Add test for explained variance computation in critic update logging
- [ ] 5.6 Verify all existing QSNNReinforce tests still pass with `use_critic: false` (no regression)

## 6. Experiment Config

- [ ] 6.1 Create `qsnnreinforce_a2c_pursuit_predators_small.yml` config based on `qsnnreinforce_pursuit_predators_small.yml` with `use_critic: true` and critic parameters

## 7. Validation

- [ ] 7.1 Run `uv run pre-commit run -a` and fix any ruff/pyright violations
- [ ] 7.2 Run `uv run pytest tests/quantumnematode_tests/brain/arch/test_qsnnreinforce.py -v` — all tests pass
