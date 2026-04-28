#!/usr/bin/env bash
# Hyperparameter-evolution pilot — MLPPPO arm
# ============================================================================
#
# Runs the MLPPPO hyperparameter-evolution pilot across 4 seeds
# sequentially.  Each seed trains a CMA-ES population of 12 genomes for
# 20 generations, K=30 train + L=5 eval episodes per genome.
#
# Per-seed wall time: ~30 minutes at parallel=4.
# Total campaign wall time: ~2 hours for 4 seeds.
#
# Outputs land under ``${OUTPUT_ROOT}/seed-${SEED}/<session>/``.  Each
# session produces best_params.json, history.csv, lineage.csv, and
# checkpoint.pkl.
#
# Usage:
#   scripts/campaigns/phase5_m2_hyperparam_mlpppo.sh
#
# Override the output root via OUTPUT_ROOT env var:
#   OUTPUT_ROOT=evolution_results scripts/campaigns/phase5_m2_hyperparam_mlpppo.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m2_hyperparam_mlpppo}"
CONFIG="configs/evolution/hyperparam_mlpppo_pilot.yml"
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
