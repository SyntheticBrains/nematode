## 1. Configuration

- [ ] 1.1 Define `PlasticityConfig` Pydantic model with fields: `training_episodes_per_phase`, `eval_episodes`, `seeds`, and `phases` (list of phase entries each with `name`, `environment`, and `reward` config sections)
- [ ] 1.2 Create `configs/studies/plasticity/qrh_plasticity.yml` — QRH architecture with 4-phase protocol (foraging → pursuit → thermotaxis+pursuit → foraging return), 200 training eps, 50 eval eps, 8 seeds
- [ ] 1.3 Create `configs/studies/plasticity/crh_plasticity.yml` — CRH (classical ablation control), matching QRH config structure
- [ ] 1.4 Create `configs/studies/plasticity/hybridquantum_plasticity.yml` — HybridQuantum in Stage 3 mode with pre-trained weights, matching phase structure
- [ ] 1.5 Create `configs/studies/plasticity/hybridclassical_plasticity.yml` — HybridClassical (classical ablation control), matching HybridQuantum config
- [ ] 1.6 Create `configs/studies/plasticity/mlpppo_plasticity.yml` — MLP PPO pure classical baseline, matching phase structure

## 2. Core Protocol Script

- [ ] 2.1 Create `scripts/run_plasticity_test.py` with CLI argument parsing (`--config` path to plasticity YAML)
- [ ] 2.2 Implement config loading: parse the plasticity YAML, validate all required fields, construct brain config and per-phase environment/reward configs
- [ ] 2.3 Implement brain construction: instantiate brain once from config, reuse across all phases
- [ ] 2.4 Implement phase execution loop: for each phase, construct a fresh `QuantumNematodeAgent` with the existing brain + phase-specific environment/reward config, run `training_episodes_per_phase` episodes
- [ ] 2.5 Implement environment switching: between phases, construct new agent with new environment config while passing the same brain instance (weights preserved in memory)

## 3. Evaluation Blocks

- [ ] 3.1 Implement eval mode toggle: before eval block, save optimizer state and set brain to no-learning mode; after eval block, restore optimizer state and clear any experience buffers
- [ ] 3.2 Implement eval block runner: run `eval_episodes` episodes on a specified objective, collect per-episode success rate, reward, and steps, return mean metrics
- [ ] 3.3 Implement transition-point evaluation schedule: run eval on objective A at all 5 transition points (pre-training, post-A, post-B, post-C, post-A'), plus eval on each phase's own objective after that phase completes

## 4. Metrics Computation

- [ ] 4.1 Implement backward forgetting (BF): `post_A_score - post_C_score_on_A` per seed, with mean ± std aggregation
- [ ] 4.2 Implement forward transfer (FT): `post_A_eval_on_B - random_baseline_on_B` per seed
- [ ] 4.3 Implement plasticity retention (PR): compare convergence rate during phase A' vs original phase A (episodes to reach threshold)
- [ ] 4.4 Implement quantum vs classical forgetting ratio (FR): `mean_BF_quantum / mean_BF_classical` with two-sample t-test (scipy.stats.ttest_ind)

## 5. Checkpointing

- [ ] 5.1 Implement generic brain checkpoint save: extract `state_dict()` from all PyTorch modules in the brain (actor, critic, optimizer) and save to `exports/{session_id}/plasticity/seed_{seed}/checkpoint_post_{phase_name}.pt`
- [ ] 5.2 Handle architecture-specific save for HybridQuantum/HybridClassical (reflex + cortex + critic components via existing `save_*_weights()` helpers)

## 6. Results Export

- [ ] 6.1 Implement per-seed phase results CSV: write per-episode training metrics and per-transition eval metrics to `exports/{session_id}/plasticity/seed_{seed}/phase_results.csv`
- [ ] 6.2 Implement aggregate metrics CSV: after all seeds complete, write mean ± std for BF, FT, PR, and per-phase eval scores to `exports/{session_id}/plasticity/aggregate_metrics.csv`
- [ ] 6.3 Print summary table to console (Rich) showing per-architecture metrics and quantum vs classical comparison with p-values

## 7. Experiment Tracking Extension

- [ ] 7.1 Extend experiment metadata to include `plasticity` section with protocol parameters, per-phase results, and aggregate plasticity metrics when run type is `plasticity_evaluation`
- [ ] 7.2 Tag plasticity runs with experiment type `"plasticity_evaluation"` in metadata JSON

## 8. Testing

- [ ] 8.1 Add unit test for PlasticityConfig validation (valid config loads, missing fields rejected)
- [ ] 8.2 Add unit test for metrics computation (BF, FT, PR, FR with known inputs)
- [ ] 8.3 Add smoke test: run plasticity protocol with MLP PPO on tiny grid (15×15), 5 training eps, 2 eval eps, 1 seed — verify CSV output, checkpoint files, and metrics computation complete without error
- [ ] 8.4 Run `uv run pre-commit run -a` and fix any lint/type issues
