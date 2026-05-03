#!/usr/bin/env bash
# Baldwin RETRY pilot — LSTMPPO + klinotaxis + pursuit predators (M4.5)
# ============================================================================
#
# Trait-only inheritance with the M4.5 8-field schema (see
# configs/evolution/baldwin_lstmppo_klinotaxis_predator_retry_pilot.yml
# header).  This is the M4.5 retry of the original Baldwin pilot
# (logbook 014) — the first valid attempt to measure the Baldwin
# Effect on this testbed after the post-pilot audit downgraded M4 to
# INCONCLUSIVE.
#
# Differences from M4's Baldwin script:
#   - 8-field schema (M4's 6 + 2 NEW arch knobs per design Decision 1)
#   - n=8 seeds (42-49) per design Decision 4
#   - Output to a distinct directory so M4 artefacts are preserved
#
# The 4-arm comparison runs against three sibling scripts:
#   - phase5_baldwin_retry_control.sh: matching 8-field control
#     (audit A1 closure — same TPE prior at gen-0 across arms)
#   - phase5_baldwin_retry_lamarckian_rerun.sh: M3 lamarckian config
#     under the M4.5 code revision at n=8 — comparative-gate baseline
#   - scripts/campaigns/baldwin_f1_postpilot_eval.py: post-hoc K'-train
#     learning-acceleration evaluation per Decision 3 (replaces M4's
#     biologically incoherent K=0 frozen-eval per audit A2 + A3)
# The aggregator scripts/campaigns/aggregate_baldwin_retry_pilot.py
# (created in task group 5) reads all four arms head-to-head and
# emits the recalibrated 3-gate verdict per Decision 5.
#
# Wall-time (estimated): ~2.5-3 hours total at parallel=4 with
# early_stop_on_saturation=5 enabled.  8-field TPE may need more
# generations to saturate than 6-field — gen budget pinned in the
# config (default 20) per smoke-test results from task group 6.
#
# Outputs land under ${OUTPUT_ROOT}/seed-${SEED}/.  Each seed
# produces best_params.json, history.csv, lineage.csv, and
# checkpoint.pkl.  NO inheritance/ subdirectory (Baldwin is
# mechanically a no-op on weight IO).
#
# Usage:
#   scripts/campaigns/phase5_baldwin_retry_baldwin.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/baldwin_retry_baldwin_lstmppo_klinotaxis_predator}"
CONFIG="configs/evolution/baldwin_lstmppo_klinotaxis_predator_retry_pilot.yml"
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
        --inheritance baldwin \
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
