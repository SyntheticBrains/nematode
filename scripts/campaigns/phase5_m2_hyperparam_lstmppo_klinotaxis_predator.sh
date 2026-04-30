#!/usr/bin/env bash
# Hyperparameter-evolution pilot — LSTMPPO + klinotaxis + pursuit predators (M2.11)
# ============================================================================
#
# The third M2 arm.  Both prior arms (MLPPPO+oracle, LSTMPPO+klinotaxis
# foraging) saturated at 1.000 from gen 1 — the schemas were too easy
# at K=50 from-scratch training, so the pilots' GO decisions were
# mechanically clean but scientifically uninformative.
#
# This arm adds **pursuit predators** to LSTMPPO+klinotaxis so the
# brain has to balance foraging and survival.  That's the regime
# where gamma/entropy/lr settings genuinely differentiate policies.
# Same brain block, same schema, same K=50/L=25 budget as the M2
# part-2 arm — only the env's predator/nociception/health blocks
# differ.
#
# Wall-time estimate (post bug-fixes for body_length / sensing-mode /
# per-episode reseed): ~2-3 hr per seed at parallel=4 — predators add
# nociception sensing channels and active enemies that may slow per-step
# vs. foraging-only, but body=2 and the post-fix env keep cost bounded.
# Outputs land under ``${OUTPUT_ROOT}/seed-${SEED}/<session>/``.  Each
# session produces best_params.json, history.csv, lineage.csv, and
# checkpoint.pkl.  ``checkpoint_every: 2`` in the YAML caps crash-loss
# to <=2 generations.
#
# Usage:
#   scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis_predator.sh
#
# Override the output root via OUTPUT_ROOT env var:
#   OUTPUT_ROOT=evolution_results scripts/campaigns/phase5_m2_hyperparam_lstmppo_klinotaxis_predator.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m2_hyperparam_lstmppo_klinotaxis_predator}"
CONFIG="configs/evolution/hyperparam_lstmppo_klinotaxis_predator_pilot.yml"
SEEDS=(42 43)

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
