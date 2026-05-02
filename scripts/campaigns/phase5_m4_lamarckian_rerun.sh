#!/usr/bin/env bash
# Lamarckian rerun on the M4 code revision — confounder-free comparison
# ============================================================================
#
# Re-runs the M3 lamarckian config
# (configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml)
# under the M4 code revision so the 4-arm comparison
# (Lamarckian / Baldwin / control / F1) shares one code revision —
# no Python/dep/machine drift between M3's published numbers and M4's.
#
# M4 introduces a kind() Protocol method, an early-stop monitor, and
# a weight_init_scale brain field.  The M3 lamarckian YAML doesn't
# evolve weight_init_scale, so the brain-config default 1.0 is used
# (byte-equivalent to the M3 init).  The kind()-based loop branching
# is a pure-additive Protocol extension; M3's lamarckian path is
# byte-equivalent.  Early-stop is enabled at runtime via
# --early-stop-on-saturation 5 (matches the Baldwin pilot).
#
# Same 4 seeds as M3 (42-45).  Output to a distinct directory from
# M3's so the M3 artefacts are preserved.  Wall-time per seed
# comparable to M3 (~12-15 min).
#
# This is NOT the run_simulation.py-driven hand-tuned baseline (that
# stays at the M2.11 published artefacts under
# evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline/
# — optimiser- and inheritance-independent, so no re-run needed).
#
# Usage:
#   scripts/campaigns/phase5_m4_lamarckian_rerun.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m4_lamarckian_lstmppo_klinotaxis_predator}"
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
