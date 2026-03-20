## 1. Dependencies & Configuration

- [ ] 1.1 Add `scipy` to project dependencies in `pyproject.toml` (needed for `scipy.stats.ttest_ind`)
- [ ] 1.2 Define `PlasticityConfig` Pydantic model with fields: `training_episodes_per_phase`, `eval_episodes`, `seeds`, and `phases` (list of phase entries each with `name`, `environment`, and `reward` config sections)
- [ ] 1.3 Create `configs/studies/plasticity/qrh_plasticity.yml` — QRH architecture with 4-phase protocol (foraging → pursuit → thermotaxis+pursuit → foraging return) on 100×100 grid, 200 training eps, 50 eval eps, 8 seeds
- [ ] 1.4 Create `configs/studies/plasticity/crh_plasticity.yml` — CRH (classical ablation control), matching QRH config structure
- [ ] 1.5 Create `configs/studies/plasticity/hybridquantum_plasticity.yml` — HybridQuantum in Stage 3 mode with pre-trained weights, 100×100 grid, matching phase structure
- [ ] 1.6 Create `configs/studies/plasticity/hybridclassical_plasticity.yml` — HybridClassical (classical ablation control), matching HybridQuantum config
- [ ] 1.7 Create `configs/studies/plasticity/mlpppo_plasticity.yml` — MLP PPO pure classical baseline, matching phase structure

## 2. Core Protocol Script

- [ ] 2.1 Create `scripts/run_plasticity_test.py` with CLI argument parsing (`--config` path to plasticity YAML, single architecture per invocation)
- [ ] 2.2 Implement config loading: parse the plasticity YAML, validate via PlasticityConfig, construct brain config and per-phase environment/reward configs
- [ ] 2.3 Implement brain construction: instantiate brain once from config, reuse across all phases and seeds
- [ ] 2.4 Implement seed loop: for each seed, re-initialise brain weights, run the full 4-phase protocol, collect results
- [ ] 2.5 Implement phase execution loop: for each phase, construct a fresh `QuantumNematodeAgent` with the existing brain + phase-specific environment/reward config, run `training_episodes_per_phase` episodes
- [ ] 2.6 Implement environment switching: between phases, construct new agent with new environment config while passing the same brain instance (weights preserved in memory)

## 3. Evaluation Blocks

- [ ] 3.1 Implement eval runner: run `eval_episodes` episodes on a specified objective without learning — call `run_brain()` for action selection but skip `learn()` and `post_process_episode()` calls. Save/restore optimizer `state_dict()` and clear PPO buffer via `buffer.reset()` before and after eval block.
- [ ] 3.2 Implement full evaluation matrix at each transition point: eval on A at all 5 points (pre-training, post-A, post-B, post-C, post-A'); eval on B at pre-training, post-A, post-B; eval on C at post-C only. Each eval records mean success rate, mean reward, mean steps.

## 4. Metrics Computation

- [ ] 4.1 Implement backward forgetting (BF): `post_A_score - post_C_score_on_A` per seed, with mean ± std aggregation
- [ ] 4.2 Implement forward transfer (FT): `post_A_eval_on_B - random_baseline_on_B` per seed
- [ ] 4.3 Implement plasticity retention (PR): compare convergence rate during phase A' vs original phase A (episodes to reach threshold)
- [ ] 4.4 Implement per-architecture aggregate: compute mean ± std across seeds for BF, FT, PR

## 5. Checkpointing

- [ ] 5.1 Implement generic brain checkpoint save: extract `state_dict()` from all PyTorch modules in the brain (actor, critic, optimizer) and save via `torch.save` to `exports/{session_id}/plasticity/seed_{seed}/checkpoint_post_{phase_name}.pt`
- [ ] 5.2 Handle architecture-specific save for HybridQuantum/HybridClassical (reflex + cortex + critic components via existing `save_*_weights()` helpers)

## 6. Results Export

- [ ] 6.1 Implement per-seed phase results CSV: write per-episode training metrics and per-transition eval metrics to `exports/{session_id}/plasticity/seed_{seed}/phase_results.csv`
- [ ] 6.2 Implement aggregate metrics CSV: after all seeds complete, write mean ± std for BF, FT, PR, and per-phase eval scores to `exports/{session_id}/plasticity/aggregate_metrics.csv`
- [ ] 6.3 Print per-architecture summary table to console (Rich) showing BF, FT, PR metrics across seeds

## 7. Cross-Architecture Comparison (Post-Hoc)

- [ ] 7.1 Create `scripts/compare_plasticity_results.py`: reads aggregate CSVs from multiple architecture runs, computes forgetting ratios (FR = mean_BF_quantum / mean_BF_classical) and two-sample t-test p-values (`scipy.stats.ttest_ind`) for quantum vs classical pairs (QRH vs CRH, HybridQuantum vs HybridClassical)
- [ ] 7.2 Print cross-architecture comparison table (Rich) with FR, p-values, and hypothesis verdict (FR ≤ 0.5 with p < 0.05 = confirmed)

## 8. Experiment Tracking Extension

- [ ] 8.1 Extend experiment metadata to include `plasticity` section with protocol parameters, per-phase results, and aggregate plasticity metrics when run type is `plasticity_evaluation`
- [ ] 8.2 Tag plasticity runs with experiment type `"plasticity_evaluation"` in metadata JSON

## 9. Testing

- [ ] 9.1 Add unit test for PlasticityConfig validation (valid config loads, missing fields rejected)
- [ ] 9.2 Add unit test for metrics computation (BF, FT, PR with known inputs)
- [ ] 9.3 Add smoke test: run plasticity protocol with MLP PPO on tiny grid (15×15), 5 training eps, 2 eval eps, 1 seed — verify CSV output, checkpoint files, and metrics computation complete without error
- [ ] 9.4 Run `uv run pre-commit run -a` and fix any lint/type issues
