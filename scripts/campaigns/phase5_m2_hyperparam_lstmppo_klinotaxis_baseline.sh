#!/usr/bin/env bash
# Hand-tuned LSTMPPO+klinotaxis baseline for the M2 pilot — PR 3 arm
# ============================================================================
#
# Trains a single LSTMPPO+klinotaxis agent on the small-foraging
# klinotaxis baseline config for 100 episodes per seed across 2
# matched seeds.  This is the comparison point for the LSTMPPO pilot's
# decision gate (≥3pp over baseline mean → GO).
#
# Usage:
#   scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis_baseline.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m2_hyperparam_lstmppo_klinotaxis_baseline}"
CONFIG="configs/scenarios/foraging/lstmppo_small_klinotaxis.yml"
SEEDS=(42 43)
EPISODES=100

mkdir -p "${OUTPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Config: ${CONFIG}"
echo "Episodes per seed: ${EPISODES}"
echo "Seeds: ${SEEDS[*]}"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

for SEED in "${SEEDS[@]}"; do
    echo "============================================================"
    echo "Baseline seed ${SEED} — start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "============================================================"
    SEED_LOG="${OUTPUT_ROOT}/seed-${SEED}.log"
    uv run python scripts/run_simulation.py \
        --config "${CONFIG}" \
        --runs "${EPISODES}" \
        --seed "${SEED}" \
        --theme headless \
        --log-level WARNING \
        > "${SEED_LOG}" 2>&1
    SUCCESS_RATE=$(grep -E "^Success rate:" "${SEED_LOG}" | tail -1 || echo "Success rate: PARSE_FAIL")
    echo "Seed ${SEED}: ${SUCCESS_RATE}"
    echo "Seed ${SEED} — done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
done

echo "============================================================"
echo "Baseline campaign complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Per-seed logs under: ${OUTPUT_ROOT}/"
echo "============================================================"
