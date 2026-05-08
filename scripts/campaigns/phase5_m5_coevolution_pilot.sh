#!/usr/bin/env bash
# M5 co-evolution pilot — arm A (heuristic-imitation pretrain) + arm B (cold-start).
# ============================================================================
#
# Sequentially runs the two pilot arms documented in
# `openspec/changes/add-coevolution-arms-race/tasks.md` (tasks 7.1a +
# 7.1b + 9.1) so the aggregator (task 9.2) can compare them
# head-to-head and lock the bootstrap choice for the full run
# (task 9.4 / 10.1).
#
# Wall-time per arm: ~14-28 hours per seed under the loop's current
# sequential dispatch (4x the design.md D4 estimate, which assumed
# parallel_workers=4 — a pool dispatch hook is documented but not yet
# wired in `CoevolutionLoop`; per-evaluation parallelism lands in a
# follow-up). Pilot ships 1 seed per arm to amortise the
# heuristic-imitation pretrain cost (~30 sec) — full run uses 4 seeds.
#
# Outputs land under ${OUTPUT_ROOT}/{arm_a,arm_b}/<session>/.
# Each session produces five checkpoint files
# (prey/checkpoint.pkl, predator/checkpoint.pkl,
# coevolution_state.json, coevolution_rng.pkl, champion_history.json),
# both per-side lineage CSVs, and the top-level
# generality_probe.csv for post-hoc Red Queen analysis.
#
# Usage::
#
#   scripts/campaigns/phase5_m5_coevolution_pilot.sh
#
# Override compute knobs via env vars before launch:
#
#   OUTPUT_ROOT=evolution_results/m5_pilot_dryrun \
#       scripts/campaigns/phase5_m5_coevolution_pilot.sh
#
# Pilot decision-gate evaluation (per task 9.3): aggregator output
# (`scripts/campaigns/aggregate_m5_pilot.py` — PR 5) determines
# whether cycling OR escalation fires in ≥1 of the 2 pilot seeds.
# Yes → calibrate full-run thresholds + lock bootstrap; proceed to
# full. No → +1 seed before commit; if still no signal → STOP M5 /
# pivot config.

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m5_coevolution_pilot}"
DRIVER="scripts/run_coevolution.py"
ARM_A_CONFIG="configs/evolution/coevolution_pilot_arm_a.yml"
ARM_B_CONFIG="configs/evolution/coevolution_pilot_arm_b.yml"
ARM_A_SEED=42
ARM_B_SEED=43

mkdir -p "${OUTPUT_ROOT}/arm_a" "${OUTPUT_ROOT}/arm_b"
echo "Output root: ${OUTPUT_ROOT}"
echo "Driver:      ${DRIVER}"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

run_arm() {
    local arm_label="$1"
    local config="$2"
    local seed="$3"
    local out_dir="$4"

    echo "============================================================"
    echo "${arm_label} — seed=${seed} — start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Config: ${config}"
    echo "Output: ${out_dir}"
    echo "============================================================"
    uv run python "${DRIVER}" \
        --config "${config}" \
        --seed "${seed}" \
        --output-dir "${out_dir}" \
        --log-level INFO
    echo "${arm_label} — seed=${seed} — end:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
}

# Arm A first (the more compute-expensive arm because of the ~30-sec
# pretrain inside __init__). Arm B's cold-start avoids it.
run_arm "Arm A (heuristic-imitation pretrain)" "${ARM_A_CONFIG}" "${ARM_A_SEED}" "${OUTPUT_ROOT}/arm_a"
run_arm "Arm B (cold-start)"                   "${ARM_B_CONFIG}" "${ARM_B_SEED}" "${OUTPUT_ROOT}/arm_b"

echo "============================================================"
echo "Pilot complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo
echo "Next step: run the aggregator (task 8.x / 9.2) on ${OUTPUT_ROOT}"
echo "to evaluate the cycling/escalation Red Queen criterion + lock"
echo "the full-run bootstrap choice (task 9.4 / 10.1)."
