#!/usr/bin/env bash
# Hyperparameter-evolution pilot — LSTMPPO + klinotaxis + pursuit predators (TPE)
# ============================================================================
#
# M2.12.  Re-runs the M2.11 predator-arm pilot under Optuna's TPE
# sampler instead of CMA-ES.  Identical config (brain, sensing, schema,
# K=50/L=25, 4 seeds 42-45) — only the optimiser changes.  Comparison
# vs M2.11's CMA-ES results closes RQ1 (optimiser-portfolio
# re-evaluation).
#
# Wall-time (estimated): comparable to the M2.11 run at ~10-15 min/seed
# (~50 min total at parallel=4).  Predator deaths shorten episodes
# regardless of which optimiser is in use.
#
# Outputs land under ``${OUTPUT_ROOT}/seed-${SEED}/<session>/``.  Each
# session produces best_params.json, history.csv, lineage.csv, and
# checkpoint.pkl.
#
# Baseline is reused from the M2.11 run (the baseline is
# run_simulation.py-driven and has no optimiser dependency); see
# ``scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis_predator_baseline.sh``.
#
# Usage:
#   scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis_predator_tpe.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_tpe}"
CONFIG="configs/evolution/hyperparam_lstmppo_klinotaxis_predator_pilot_tpe.yml"
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
