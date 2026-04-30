#!/usr/bin/env bash
# Hyperparameter-evolution pilot — LSTMPPO + klinotaxis arm
# ============================================================================
#
# Runs the LSTMPPO+klinotaxis hyperparameter-evolution pilot across 2
# seeds sequentially.  Each seed trains a CMA-ES population of 12
# genomes for 20 generations, K=50 train + L=25 eval episodes per
# genome.
#
# Wall-time estimate (post bug-fixes for body_length / sensing-mode /
# per-episode reseed): ~2-3 hours per seed at parallel=4.
# Outputs land under ``${OUTPUT_ROOT}/seed-${SEED}/<session>/``.  Each
# session produces best_params.json, history.csv, lineage.csv, and
# checkpoint.pkl.  ``checkpoint_every: 2`` in the YAML caps crash-loss
# to ≤2 generations.
#
# Usage:
#   scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis.sh
#
# Override the output root via OUTPUT_ROOT env var:
#   OUTPUT_ROOT=evolution_results scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m2_hyperparam_lstmppo_klinotaxis}"
CONFIG="configs/evolution/hyperparam_lstmppo_klinotaxis_pilot.yml"
SEEDS=(42 43)

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
