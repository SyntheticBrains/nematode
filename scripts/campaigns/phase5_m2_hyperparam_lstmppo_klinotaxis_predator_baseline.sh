#!/usr/bin/env bash
# Hand-tuned LSTMPPO+klinotaxis+predator baseline for M2.11
# ============================================================================
#
# Trains a single LSTMPPO+klinotaxis+predator agent on the small-foraging
# pursuit baseline config for 100 episodes per seed across 4 matched
# seeds (42-45).  This is the comparison point for the predator pilot's
# decision gate (>=3pp over baseline mean -> GO).
#
# Reference baseline config: configs/scenarios/pursuit/lstmppo_small_klinotaxis.yml
# — same predator block (count=2, pursuit, detection_radius=6) and same
# health/reward/satiety blocks as the pilot, so pilot vs baseline
# differs only on the hyperparameter axis CMA-ES is evolving.
#
# Usage:
#   scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis_predator_baseline.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator_baseline}"
CONFIG="configs/scenarios/pursuit/lstmppo_small_klinotaxis.yml"
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
    # Fail fast on parse failure: previously this was `grep ... || echo
    # "PARSE_FAIL"` which swallowed grep's non-zero exit so the campaign
    # silently continued past a broken seed.  An empty SUCCESS_RATE means
    # the simulation didn't print the expected summary line — most likely
    # the run crashed before completion or the log format changed.  Either
    # way we want the script to halt rather than report a stale aggregate.
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
