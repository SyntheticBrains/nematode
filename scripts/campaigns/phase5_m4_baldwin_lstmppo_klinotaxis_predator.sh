#!/usr/bin/env bash
# Baldwin inheritance pilot — LSTMPPO + klinotaxis + pursuit predators
# ============================================================================
#
# Trait-only inheritance: the prior generation's elite genome ID is
# recorded as ``inherited_from`` for every child of the next generation
# (so the lineage CSV captures the evolutionary trace), but no
# per-genome weight checkpoints are written.  Each child trains
# from-scratch; over generations TPE selects genomes with hyperparams
# that produce faster K-train convergence — the Baldwin Effect.
#
# Same brain (LSTMPPO + klinotaxis sensing + 2 pursuit predators), same
# K=50/L=25 episode budget, same TPE base optimiser, same 4 seeds
# (42-45) as the M3 control + Lamarckian arms.  Differs from M3 control
# in three ways (per the Baldwin pilot YAML's header comments):
# inheritance: baldwin; hyperparam_schema gains weight_init_scale +
# entropy_decay_episodes; early_stop_on_saturation: 5.
#
# The 4-arm comparison runs against three sibling scripts:
#   - phase5_m4_lamarckian_rerun.sh: M3 lamarckian config under the M4
#     code revision (confounder-free comparison).
#   - phase5_m4_control_rerun.sh: M3 control config under the M4 code
#     revision (the from-scratch baseline for the speed gate).
#   - scripts/campaigns/baldwin_f1_postpilot_eval.py: post-hoc
#     evaluation of each Baldwin seed's gen-N elite genome with K=0
#     (frozen-eval only) — tests whether the bias has been encoded
#     into the genome itself (genetic assimilation gate).
# The aggregator scripts/campaigns/aggregate_m4_pilot.py reads all
# four arms head-to-head.
#
# Wall-time (estimated): ~12-15 min/seed (~50 min total at parallel=4).
# Baldwin has no save_weights/load_weights overhead; early-stop should
# reduce wall-time on saturating arms.
#
# Outputs land under ``${OUTPUT_ROOT}/seed-${SEED}/<session>/``.  Each
# session produces best_params.json, history.csv, lineage.csv, and
# checkpoint.pkl.  NO inheritance/ subdirectory (Baldwin is mechanically
# a no-op on weight IO).
#
# Usage:
#   scripts/campaigns/phase5_m4_baldwin_lstmppo_klinotaxis_predator.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m4_baldwin_lstmppo_klinotaxis_predator}"
CONFIG="configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml"
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
