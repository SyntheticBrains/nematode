#!/usr/bin/env bash
# Lamarckian control arm — same config as lamarckian pilot, but inheritance: none
# ============================================================================
#
# Within-experiment from-scratch control for the lamarckian pilot.
# Identical config to lamarckian_lstmppo_klinotaxis_predator_pilot.yml
# (same brain, sensing, K=50/L=25 episode budget, schema, 4 seeds
# 42-45) EXCEPT ``evolution.inheritance: none``.
#
# Run on the same code revision as the lamarckian arm so the
# lamarckian-vs-control comparison is confounder-free — both arms
# exercise the same evolution loop code with the only meaningful
# difference being whether children warm-start from the prior
# generation's elite.
#
# This is NOT the run_simulation.py-driven hand-tuned baseline (that's
# the lstmppo-klinotaxis-predator baseline campaign script, re-run on
# the same code revision before running the aggregator).
#
# Wall-time (estimated): ~12-15 min/seed (~50 min total at parallel=4).
#
# Outputs land under ``${OUTPUT_ROOT}/seed-${SEED}/<session>/``.  No
# inheritance/ subdirectory is created (no-inheritance code path).
#
# Usage:
#   scripts/campaigns/phase5_m3_lamarckian_lstmppo_klinotaxis_predator_control.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m3_lamarckian_lstmppo_klinotaxis_predator_control}"
CONFIG="configs/evolution/lamarckian_lstmppo_klinotaxis_predator_control.yml"
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
        --inheritance none \
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
