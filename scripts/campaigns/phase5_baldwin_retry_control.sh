#!/usr/bin/env bash
# From-scratch CONTROL RETRY pilot — LSTMPPO + klinotaxis + pursuit predators (M4.5)
# ============================================================================
#
# The matching control arm for the M4.5 Baldwin retry pilot.  Identical
# to phase5_baldwin_retry_baldwin.sh's run shape EXCEPT for the config
# (control YAML uses inheritance: none) and the OUTPUT_ROOT.
#
# Why this arm exists: audit finding A1 (schema-shift confounder)
# requires that compared arms evolve identical schemas under the same
# TPE seed so gen-0 starting populations are byte-identical.  M4
# violated this by running Baldwin (6-field) against a Control (4-field)
# — Baldwin's gen-0 was systematically -0.14pp weaker before any
# inheritance signal could fire.  M4.5 fixes this by running both
# arms on the SAME 8-field schema (only the inheritance field differs).
#
# The aggregator's pre-flight check (task group 5) verifies the
# schema-equalisation property at runtime: gen-0 mean fitness across
# this arm and the Baldwin arm SHALL converge within |Δ| ≤ 0.05; if
# not, the verdict is forced to INCONCLUSIVE (audit A1 still
# unresolved despite the schema equalisation).
#
# Wall-time (estimated): same envelope as the Baldwin arm
# (~2.5-3 hours at parallel=4, n=8 seeds, early_stop_on_saturation=5).
# Run in parallel with phase5_baldwin_retry_baldwin.sh and
# phase5_baldwin_retry_lamarckian_rerun.sh for total pilot wall
# ~3-4 hours including baseline + F1 post-pilot.
#
# Outputs land under ${OUTPUT_ROOT}/seed-${SEED}/.  Same artefacts as
# the Baldwin arm; NO inheritance/ subdirectory (inheritance: none
# is the no-op path).  Lineage's inherited_from is empty for all rows.
#
# Usage:
#   scripts/campaigns/phase5_baldwin_retry_control.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/baldwin_retry_control_lstmppo_klinotaxis_predator}"
CONFIG="configs/evolution/control_lstmppo_klinotaxis_predator_retry_pilot.yml"
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
