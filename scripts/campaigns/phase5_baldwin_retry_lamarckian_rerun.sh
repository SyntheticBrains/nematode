#!/usr/bin/env bash
# Lamarckian RERUN — LSTMPPO + klinotaxis + pursuit predators (M4.5)
# ============================================================================
#
# Re-runs the M3 lamarckian config (4-field schema) under the M4.5 code
# revision at n=8 seeds.  Two purposes:
#
# 1. PRIMARY: provide the comparative-gate baseline at n=8.  Decision 5
#    (per add-baldwin-retry design) sets the comparative gate as
#    mean_gen_baldwin_to_092 ≤ mean_gen_lamarckian_to_092 + 4.  For
#    statistical apples-to-apples on this gate, Lamarckian must run at
#    the same n as Baldwin/Control.
#
# 2. SECONDARY: reproducibility check on the n=4 subset (seeds 42-45) —
#    Lamarckian numbers SHOULD match M3's published [3, 4, 4, 7] mean
#    4.50, confirming the M4.5 code revision is byte-equivalent for
#    the M3 path.  M4 already showed this property; M4.5 confirms it
#    holds across the M3 → M4 → M4.5 revisions.
#
# Note the schema-asymmetry: Lamarckian uses M3's 4-field schema, while
# Baldwin/Control use M4.5's 8-field schema.  This is intentional —
# Lamarckian doesn't participate in the audit-A1 schema-equalisation
# check (only Baldwin vs Control matters there); its sole role is the
# comparative-gate reference.  The 4-field choice is the existing
# M3 contract; running Lamarckian at 8-field would conflate the
# comparative-gate signal with schema-shift effects.
#
# Wall-time (estimated): same envelope as Baldwin/Control
# (~2.5-3 hours at parallel=4, n=8, early_stop_on_saturation=5
# enabled at runtime via --early-stop-on-saturation 5 since the M3
# config doesn't set it).
#
# Outputs land under ${OUTPUT_ROOT}/seed-${SEED}/ — distinct from M4
# Lamarckian rerun artefacts (preserved under the M4 directories) so
# the two reruns can be compared in logbook 015 if needed.
#
# Usage:
#   scripts/campaigns/phase5_baldwin_retry_lamarckian_rerun.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/baldwin_retry_lamarckian_lstmppo_klinotaxis_predator}"
CONFIG="configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml"
SEEDS=(42 43 44 45 46 47 48 49)

mkdir -p "${OUTPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Config: ${CONFIG}"
echo "Seeds: ${SEEDS[*]} (n=${#SEEDS[@]})"
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
        --inheritance lamarckian \
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
