# Proposal: Minimal-RNN (minGRU / minLSTM) Policy Arms

## Why

The bit-memory positive control fired ([Logbook 030](../../../docs/experiments/logbooks/030-bit-memory-positive-control.md)): the architecture comparison **does** resolve working memory (CfC / Transformer / LSTM beat the memoryless MLP; the connectome sits at chance), which unblocks [`T7.separation.new_arch_candidates`](../phase6-tracking/tasks.md) — bringing up the strongest memory-axis architectures *not* already in the repo. The candidate research ([docs/research/policy-architecture-candidates.md](../../../docs/research/policy-architecture-candidates.md)) ranks **minGRU / minLSTM** ([Feng et al. 2024, arXiv:2410.01201](https://arxiv.org/abs/2410.01201)) the **cheapest** Tier-1 candidate: a minimal RNN that drops the hidden-state dependence of its gates so the recurrence is an associative (parallel-scannable) scan, competitive with Transformers/Mamba at a fraction of the parameters. It is a near-trivial extension of the existing `lstmppo` arm, and carries **dual value**: (1) a new memory-axis arm to evaluate against the existing recurrent arms on a memory-demanding cell, and (2) a **stability-upgrade A/B vs `lstmppo`** on the reactive cell — independent of memory — where the 029 LSTM arm was the laggard (60.1, 5/8 seeds converged; [Logbook 029](../../../docs/experiments/logbooks/029-continuous-architecture-ranking.md)). The minimal RNN's gates do not feed back through a saturating hidden-to-hidden matrix, which is a principled candidate fix for that seed-fragility.

## What Changes

- **Two new registered classical PPO arms** — `mingruppo` and `minlstmppo` — sharing one **parallel-form minimal-RNN recurrent core** (the only genuinely new computation). minGRU uses a single update gate `h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t`; minLSTM uses two gates (forget/input) with `f/(f+i)` normalization. Both have **input-only gates** (no hidden-to-hidden dependence) and a single recurrent state — the property that removes the saturating recurrent matrix and makes the recurrence an associative scan. Both reuse the existing `lstmppo` PPO machinery **verbatim** — the rollout buffer, chunk-based truncated BPTT, separate actor/critic optimizers, continuous tanh-Gaussian + discrete heads, entropy/LR scheduling, and weight persistence — via a small extract-method refactor of `LSTMPPOBrain` that exposes the recurrent-core construction as an override hook.
- **No new PPO/training code, no new dependency, no new architecture paradigm** — the arms are a new recurrent *cell* plus two thin registrations. The dispatch is registry-based and these arms match the default infrastructure shape, so no `brain_factory` / `_registry` hand-edits are needed.
- **Evaluation on two prongs**: **(a) the memory cell** — the merged bit-memory delayed-match-to-cue control — to confirm the minimal RNNs solve a task the memoryless MLP cannot (validating them as genuine memory arms, with the existing LSTM / CfC as the yardstick); **(b) the reactive-cell stability A/B** — the 029 continuous C3 cell — to test the stability-upgrade hypothesis against `lstmppo`, independent of memory. Both reuse the committed paired-seed Wilcoxon + bootstrap + BH-FDR statistics layer.
- **Per-arm scenario configs** for both prongs (the `bit_memory` family for the memory cell; the continuous-2D klinotaxis cell for the stability A/B).
- **Explicitly non-gating** for Phase 6 Gate 3: the MUST integrated-C3 ranking (029) is unchanged. This is a memory-axis follow-on, evaluated against the existing arms — not a re-opening of the frozen MUST set.

## Capabilities

### New Capabilities

- **`minimal-rnn-brain`** — the minGRU / minLSTM PPO arms: the parallel-form minimal-RNN recurrent core (minGRU single-state and minLSTM cell-state variants), its drop-in integration into the shared recurrent-PPO training pipeline, the two registered arms (`mingruppo`, `minlstmppo`) with discrete and continuous action heads, and their evaluation on the memory cell and the reactive-cell stability A/B. Mirrors the per-arm capability convention (`lstm-ppo-brain`, `cfc-liquid-brain`, `transformer-brain`).

### Modified Capabilities

None. The arms are **additive**: a new recurrent cell + two registrations that reuse the existing recurrent-PPO pipeline. The `lstm-ppo-brain` capability's stated requirements do not change — the extract-method refactor that exposes the recurrent-core hook is byte-identical for the existing `lstm` / `gru` paths (a no-requirement-change implementation detail). No existing arm, config, or training behaviour is altered.

## Impact

- **New brain module** — `brain/arch/minimal_rnn_ppo.py`: the `MinimalRNN` cell (minGRU / minLSTM), the `MinGRUPPOBrainConfig` / `MinLSTMPPOBrainConfig` configs, and the `MinGRUPPOBrain` / `MinLSTMPPOBrain` arms (subclassing `LSTMPPOBrain`, overriding only the recurrent-core construction + its initialisation).
- **Recurrent-core hook** — a small extract-method refactor in `brain/arch/lstmppo.py` (`_build_recurrent_core` / `_init_recurrent_weights`) so the new arms reuse the pipeline without copy-paste; the existing `lstm` / `gru` / LayerNorm-cell paths are unchanged.
- **Registration touchpoints (the ≤6-file plugin budget — here ~4)** — `BrainType.MINGRU_PPO` / `MINLSTM_PPO` in `brain/arch/dtypes.py`; exports in `brain/arch/__init__.py`; the config-type union in `utils/config_loader.py` (`BRAIN_CONFIG_MAP` auto-derives from the registry). No `brain_factory.py` branch (default infra shape) and no manual `_registry.py` edit (`@register_brain` self-registers).
- **Configs** — per-arm scenario YAMLs: `configs/scenarios/bit_memory/{mingruppo,minlstmppo}_small_bit_memory.yml` (memory cell) and the continuous-2D klinotaxis cell for the stability A/B.
- **Evaluation** — **extends the existing harness arm-rosters** to include the new arms (`MEMORY_ARMS` in `scripts/analysis/bit_memory_separation.py` for the memory cell; `PPO_ARCHS` in `scripts/analysis/t7_continuous_ranking.py`, the 029 reactive-cell orchestrator), reusing the shared paired-seed statistics layer. No new harness or statistics code.
- **Tests** — the minimal-RNN cell recurrence (shapes, sequence roll, minGRU-vs-minLSTM state), config validation, registry round-trip (name ↔ `BrainType` ↔ config), and a continuous + discrete forward/learn smoke.
- **No breaking changes; no new dependencies.** Existing arms and the training/episode plumbing are reused unchanged.
- **Tracker** — ticks `T7.separation.new_arch_candidates` (minGRU/minLSTM portion) and records the two-prong result; modified-S5 (the other Tier-1 candidate) remains a separate follow-on.
