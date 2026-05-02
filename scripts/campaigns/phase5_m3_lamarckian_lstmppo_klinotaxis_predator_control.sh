#!/usr/bin/env bash
# M3 control arm — same config as M3 lamarckian, but inheritance: none
# ============================================================================
#
# Within-experiment from-scratch control for the M3 lamarckian pilot.
# Identical config to lamarckian_lstmppo_klinotaxis_predator_pilot.yml
# (same brain, sensing, K=50/L=25, schema, 4 seeds 42-45) EXCEPT
# ``evolution.inheritance: none``.
#
# Run under the M3 revision (not M2.12's published numbers) so the
# lamarckian-vs-control comparison is confounder-free — both arms
# exercise the same M3 loop code with the only meaningful difference
# being whether children warm-start from the prior generation's elite.
#
# This is NOT the run_simulation.py-driven hand-tuned baseline (that's
# scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis_predator_baseline.sh,
# re-run under the M3 revision per task 9.6).
#
# Wall-time (estimated): comparable to M2.12 at ~12-15 min/seed
# (~50 min total at parallel=4).
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
