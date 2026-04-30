#!/usr/bin/env bash
# Hand-tuned MLPPPO baseline for the M2 hyperparameter pilot
# ============================================================================
#
# Trains a single MLPPPO agent on the small-foraging baseline config for
# 100 episodes per seed across 4 matched seeds.  This is the comparison
# point for the hyperparameter pilot's decision gate (≥3pp over baseline
# mean → GO).  STATIC measurement; not a 1:1 episode-budget match with
# the pilot — the baseline has no genomes, so episode-volume parity is
# not meaningful.
#
# Usage:
#   scripts/campaigns/phase5_m2_hyperparam_baseline.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m2_hyperparam_baseline}"
CONFIG="configs/scenarios/foraging/mlpppo_small_oracle.yml"
SEEDS=(42 43 44 45)
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
    # Extract success rate from the run summary tail.  Fail fast if the
    # parse fails (most likely a pre-summary crash or log-format change);
    # the previous `|| echo "PARSE_FAIL"` swallowed the failure silently.
    SUCCESS_RATE=$(grep -E "^Success rate:" "${SEED_LOG}" | tail -1 || true)
    if [ -z "${SUCCESS_RATE}" ]; then
        echo "ERROR: failed to parse 'Success rate:' line from ${SEED_LOG} (seed ${SEED})." >&2
        echo "  Inspect the log for crashes or format changes; aborting campaign." >&2
        exit 1
    fi
    echo "Seed ${SEED}: ${SUCCESS_RATE}"
    echo "Seed ${SEED} — done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
done

echo "============================================================"
echo "Baseline campaign complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Per-seed logs under: ${OUTPUT_ROOT}/"
echo "============================================================"
