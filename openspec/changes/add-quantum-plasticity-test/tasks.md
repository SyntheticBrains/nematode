## 1. Dependencies & Configuration

- [x] 1.1 Add `scipy` to project dependencies in `pyproject.toml` (needed for `scipy.stats.ttest_ind`)
- [x] 1.2 Define `PlasticityConfig` Pydantic model with fields: `training_episodes_per_phase`, `eval_episodes`, `seeds`, `convergence_threshold` (default 0.6), and `phases` (list of phase entries each with `name`, `environment`, and `reward` config sections)
- [x] 1.3 Create `configs/studies/plasticity/qrh_plasticity.yml` — QRH architecture with 4-phase protocol (foraging → pursuit → thermotaxis+pursuit → foraging return) on 100×100 grid, 200 training eps, 50 eval eps, 8 seeds
- [x] 1.4 Create `configs/studies/plasticity/crh_plasticity.yml` — CRH (classical ablation control), matching QRH config structure
- [x] 1.5 Create `configs/studies/plasticity/hybridquantum_plasticity.yml` — HybridQuantum in Stage 3 mode with pre-trained weights, 100×100 grid, matching phase structure
- [x] 1.6 Create `configs/studies/plasticity/hybridclassical_plasticity.yml` — HybridClassical (classical ablation control), matching HybridQuantum config
- [x] 1.7 Create `configs/studies/plasticity/mlpppo_plasticity.yml` — MLP PPO pure classical baseline, matching phase structure

## 2. Core Protocol Script

- [x] 2.1 Create `scripts/run_plasticity_test.py` with CLI argument parsing (`--config` path to plasticity YAML, single architecture per invocation)
- [x] 2.2 Implement config loading: parse the plasticity YAML, validate via PlasticityConfig, construct brain config and per-phase environment/reward configs
- [x] 2.3 Implement seed loop: for each seed, construct a fresh brain via `setup_brain_model()` with `brain_config.seed = seed`, run the full 4-phase protocol, collect per-seed results
- [x] 2.4 Implement phase execution loop: for each phase, construct a fresh `QuantumNematodeAgent` with the existing brain + phase-specific environment/reward config, run `training_episodes_per_phase` episodes
- [x] 2.5 Implement environment switching: between phases, construct new agent with new environment config while passing the same brain instance (weights preserved in memory within a seed)

## 3. Evaluation Blocks

- [x] 3.1 Implement state snapshot helper: collect all `state_dict()`s from the brain's PyTorch modules (actor, critic, optimizer, normalisation layers) into an in-memory dict. Implement corresponding restore helper that calls `load_state_dict()` on each module and clears PPO buffer via `buffer.reset()`
- [x] 3.2 Implement eval block runner: snapshot brain state, run `eval_episodes` episodes through the standard agent episode loop (learning may occur internally), collect per-episode success rate/reward/steps, restore brain state from snapshot, return mean metrics
- [x] 3.3 Implement full evaluation matrix at each transition point: 9 eval blocks total — eval on A at all 5 points (pre-training, post-A, post-B, post-C, post-A'); eval on B at 3 points (pre-training, post-A, post-B); eval on C at 1 point (post-C). Each eval records mean success rate, mean reward, mean steps.

## 4. Metrics Computation

- [x] 4.1 Implement backward forgetting (BF): `post_A_score - post_C_score_on_A` per seed, with mean ± std aggregation
- [x] 4.2 Implement forward transfer (FT): `post_A_eval_on_B - random_baseline_on_B` per seed
- [x] 4.3 Implement plasticity retention (PR): convergence defined as first episode where trailing-20-episode mean success rate exceeds `convergence_threshold` (default 0.6). `PR = convergence_episodes_A / convergence_episodes_A'`. Report `N/A` if a phase does not converge, exclude from aggregation.
- [x] 4.4 Implement per-architecture aggregate: compute mean ± std across seeds for BF, FT, PR

## 5. Checkpointing

- [x] 5.1 Implement generic brain checkpoint save: extract `state_dict()` from all PyTorch modules in the brain (actor, critic, optimizer) and save via `torch.save` to `exports/{session_id}/plasticity/seed_{seed}/checkpoint_post_{phase_name}.pt`
- [x] 5.2 Handle architecture-specific save for HybridQuantum/HybridClassical (reflex + cortex + critic components via existing `save_*_weights()` helpers)

## 6. Results Export

- [x] 6.1 Implement per-seed phase results CSV: write per-episode training metrics and per-transition eval metrics to `exports/{session_id}/plasticity/seed_{seed}/phase_results.csv`
- [x] 6.2 Implement aggregate metrics CSV: after all seeds complete, write mean ± std for BF, FT, PR, and per-phase eval scores to `exports/{session_id}/plasticity/aggregate_metrics.csv`
- [x] 6.3 Print per-architecture summary table to console (Rich) showing BF, FT, PR metrics across seeds

## 7. Cross-Architecture Comparison (Post-Hoc)

- [x] 7.1 Create `scripts/compare_plasticity_results.py` with `--results path1.csv path2.csv ...` CLI interface: reads aggregate CSVs from multiple architecture runs, matches quantum/classical pairs by architecture name, computes forgetting ratios (FR = mean_BF_quantum / mean_BF_classical) and two-sample t-test p-values (`scipy.stats.ttest_ind`)
- [x] 7.2 Print cross-architecture comparison table (Rich) with FR, p-values, and hypothesis verdict (FR ≤ 0.5 with p < 0.05 = confirmed)

## 8. Experiment Tracking Extension

- [x] 8.1 Extend experiment metadata to include `plasticity` section with protocol parameters, per-phase results, and aggregate plasticity metrics when run type is `plasticity_evaluation`
- [x] 8.2 Tag plasticity runs with experiment type `"plasticity_evaluation"` in metadata JSON

## 9. Testing

- [x] 9.1 Add unit test for PlasticityConfig validation (valid config loads, missing fields rejected, convergence_threshold defaults to 0.6)
- [x] 9.2 Add unit test for metrics computation (BF, FT, PR with known inputs, including N/A handling for non-converging PR)
- [x] 9.3 Add unit test for state snapshot/restore helper (verify brain state is identical after snapshot → modify → restore cycle)
- [x] 9.4 Add smoke test: run plasticity protocol with MLP PPO on tiny grid (15×15), 5 training eps, 2 eval eps, 1 seed — verify CSV output, checkpoint files, and metrics computation complete without error
- [x] 9.5 Run `uv run pre-commit run -a` and fix any lint/type issues
