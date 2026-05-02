#!/usr/bin/env bash
# Lamarckian inheritance pilot — LSTMPPO + klinotaxis + pursuit predators
# ============================================================================
#
# Per-genome warm-start from the prior generation's elite parent.  Same
# brain (LSTMPPO + klinotaxis sensing + 2 pursuit predators), same
# K=50/L=25 episode budget, same TPE base optimiser, same 4 seeds
# (42-45) as the no-inheritance control — only inheritance is added.
#
# The pilot config drops rnn_type + lstm_hidden_dim from the schema
# (those are architecture-changing; per-genome warm-start cannot load a
# parent's LSTM weights into a child with a different shape — the
# validator on SimulationConfig._validate_hyperparam_schema rejects the
# combination).  rnn_type defaults to "gru" and lstm_hidden_dim to 64
# in the brain block.
#
# The lamarckian-vs-control comparison runs against the sibling script
# phase5_m3_lamarckian_lstmppo_klinotaxis_predator_control.sh, which
# re-runs the same config with inheritance: none on the same code
# revision so the comparison is confounder-free.  The aggregator
# scripts/campaigns/aggregate_m3_pilot.py reads both arms head-to-head.
#
# Wall-time (estimated): ~12-15 min/seed (~50 min total at parallel=4).
# Per-genome torch.save adds ~10 ms per evaluation — negligible vs the
# K=50 train phase.
#
# Outputs land under ``${OUTPUT_ROOT}/seed-${SEED}/<session>/``.  Each
# session produces best_params.json, history.csv, lineage.csv,
# checkpoint.pkl, and an inheritance/ subdirectory containing the final
# winner's weight checkpoint (intermediate checkpoints are GC'd).
#
# Baseline is the run_simulation.py-driven hand-tuned run, expected at
# evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline/
# (produced by the lstmppo-klinotaxis-predator baseline campaign script
# on the same code revision before running the aggregator).
#
# Usage:
#   scripts/campaigns/phase5_m3_lamarckian_lstmppo_klinotaxis_predator.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m3_lamarckian_lstmppo_klinotaxis_predator}"
CONFIG="configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml"
SEEDS=(42 43 44 45)

mkdir -p "${OUTPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Config: ${CONFIG}"
echo "Seeds: ${SEEDS[*]}"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

for SEED in "${SEEDS[@]}"; do
    echo "============================================================"
    echo "Seed ${SEED} — start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "============================================================"
    SEED_DIR="${OUTPUT_ROOT}/seed-${SEED}"
    mkdir -p "${SEED_DIR}"
    uv run python scripts/run_evolution.py \
        --config "${CONFIG}" \
        --fitness learned_performance \
        --seed "${SEED}" \
        --log-level WARNING \
        --output-dir "${SEED_DIR}"
    echo "Seed ${SEED} — done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
done

echo "============================================================"
echo "Campaign complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Artefacts under: ${OUTPUT_ROOT}/"
echo "============================================================"
