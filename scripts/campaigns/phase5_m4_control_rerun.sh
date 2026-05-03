#!/usr/bin/env bash
# From-scratch control rerun on the M4 code revision
# ============================================================================
#
# Re-runs the M3 control config
# (configs/evolution/lamarckian_lstmppo_klinotaxis_predator_control.yml)
# under the M4 code revision.  Same brain (LSTMPPO + klinotaxis +
# pursuit predators), same K=50/L=25 episode budget, same 4 seeds
# (42-45), same TPE + 4-field hyperparam_schema as the M3 control —
# only the code revision differs.
#
# This is the from-scratch baseline for the M4 speed gate
# (Baldwin reaches 0.92 ≥2 generations earlier than this control to
# pass the speed gate per the aggregator's decision rule).
#
# Early-stop is enabled at runtime via --early-stop-on-saturation 5
# (matches the Baldwin pilot YAML and the Lamarckian rerun script).
#
# Output to a distinct directory from M3's so the M3 artefacts are
# preserved.  Wall-time per seed comparable to M3 (~12-15 min).
#
# Usage:
#   scripts/campaigns/phase5_m4_control_rerun.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m4_control_lstmppo_klinotaxis_predator}"
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
        --early-stop-on-saturation 5 \
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
