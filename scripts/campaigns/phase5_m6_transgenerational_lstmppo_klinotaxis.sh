#!/usr/bin/env bash
# Transgenerational Memory campaign — LSTMPPO + klinotaxis + stationary pathogens
# ============================================================================
#
# Three subcommands map to the three execution stages of the M6
# verification protocol:
#
#   --smoke   F0 calibration smoke. 1 seed × pop 6 × ~50 episodes.
#             Single generation, TEI-on arm only. Used to verify the
#             F0 calibration envelope (mean choice_index ∈ [0.45, 0.85])
#             BEFORE unblocking the pilot. ~30 minutes.
#
#   --pilot   Pilot. 1 seed × pop 6 × 4 generations. Paired arms
#             (TEI-on + TEI-off). ~4 wall-hours. Aggregator produces a
#             preliminary verdict; pause for user review before full.
#
#   --full    Full campaign. 4 seeds (42, 43, 44, 45) × pop 16 × 4
#             generations. Paired arms. ~16 wall-hours. Aggregator
#             produces per-seed + cross-seed verdict.
#
# Output directory convention:
#   evolution_results/m6_transgenerational/{tei_on|tei_off}/seed-{N}/<session_id>/
#
# Each session produces best_params.json, history.csv, lineage.csv,
# checkpoint.pkl, and (for TEI-on) the F0 substrate at
# inheritance/gen-000/genome-<id>.tei.pt.
#
# The paired arms are byte-equivalent except for two fields:
# ``inheritance`` and ``transgenerational.enabled``. Both are pinned
# by the EvolutionConfig pairing validator at config-load — the
# control arm uses a sibling YAML
# (``transgenerational_pathogen_avoidance_lstmppo_klinotaxis_tei_off.yml``)
# rather than CLI overrides because the ``transgenerational:`` block
# is not currently CLI-overridable. Maintain the two YAMLs in lockstep
# (see the TEI-off YAML's header).
#
# Usage:
#   scripts/campaigns/phase5_m6_transgenerational_lstmppo_klinotaxis.sh --smoke
#   scripts/campaigns/phase5_m6_transgenerational_lstmppo_klinotaxis.sh --pilot
#   scripts/campaigns/phase5_m6_transgenerational_lstmppo_klinotaxis.sh --full
#
# Outputs land under ``${OUTPUT_ROOT:-evolution_results/m6_transgenerational}/``.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 --smoke|--pilot|--full" >&2
    exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m6_transgenerational}"
CONFIG_TEI_ON="configs/evolution/transgenerational_pathogen_avoidance_lstmppo_klinotaxis.yml"
CONFIG_TEI_OFF="configs/evolution/transgenerational_pathogen_avoidance_lstmppo_klinotaxis_tei_off.yml"
CONFIG_SMOKE="configs/evolution/transgenerational_pathogen_avoidance_lstmppo_klinotaxis_smoke.yml"

# Resolve subcommand → (seeds, population, generations, arms, config-on).
# The TEI-on config differs between ``--smoke`` (F0-only, 1 gen, 1
# schedule entry — for the calibration gate) and the multi-gen runs
# (4 gens, 4 schedule entries). The schedule's coverage validator
# requires exactly one entry per gen in [0, generations), so CLI
# ``--generations`` cannot shrink a multi-gen schedule to 1 gen
# without a separate YAML.
case "$1" in
    --smoke)
        MODE="smoke"
        SEEDS=(42)
        POPULATION=6
        GENERATIONS=1
        ARMS=("tei_on")
        CONFIG_TEI_ON_RESOLVED="${CONFIG_SMOKE}"
        ;;
    --pilot)
        MODE="pilot"
        SEEDS=(42)
        POPULATION=6
        GENERATIONS=4
        ARMS=("tei_on" "tei_off")
        CONFIG_TEI_ON_RESOLVED="${CONFIG_TEI_ON}"
        ;;
    --full)
        MODE="full"
        SEEDS=(42 43 44 45)
        POPULATION=16
        GENERATIONS=4
        ARMS=("tei_on" "tei_off")
        CONFIG_TEI_ON_RESOLVED="${CONFIG_TEI_ON}"
        ;;
    *)
        echo "usage: $0 --smoke|--pilot|--full" >&2
        exit 1
        ;;
esac

mkdir -p "${OUTPUT_ROOT}"
echo "============================================================"
echo "M6 transgenerational campaign — mode: ${MODE}"
echo "============================================================"
echo "Output root: ${OUTPUT_ROOT}"
echo "Seeds:       ${SEEDS[*]}"
echo "Population:  ${POPULATION}"
echo "Generations: ${GENERATIONS}"
echo "Arms:        ${ARMS[*]}"
echo "Start:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

for ARM in "${ARMS[@]}"; do
    if [[ "${ARM}" == "tei_on" ]]; then
        CONFIG="${CONFIG_TEI_ON_RESOLVED}"
    else
        CONFIG="${CONFIG_TEI_OFF}"
    fi
    # ``inheritance`` and ``transgenerational.enabled`` are pinned by
    # the YAMLs (and tied by the pairing validator); no CLI overrides
    # for them. Population / generations are CLI-overridable for the
    # multi-gen modes but the schedule coverage validator means
    # --smoke uses a dedicated YAML rather than --generations 1.
    for SEED in "${SEEDS[@]}"; do
        echo "------------------------------------------------------------"
        echo "Arm: ${ARM}  Seed: ${SEED}  Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "------------------------------------------------------------"
        SEED_DIR="${OUTPUT_ROOT}/${ARM}/seed-${SEED}"
        mkdir -p "${SEED_DIR}"
        uv run python scripts/run_evolution.py \
            --config "${CONFIG}" \
            --fitness learned_performance \
            --population "${POPULATION}" \
            --generations "${GENERATIONS}" \
            --seed "${SEED}" \
            --log-level WARNING \
            --output-dir "${SEED_DIR}"
        echo "Arm: ${ARM}  Seed: ${SEED}  Done:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo
    done
done

echo "============================================================"
echo "Campaign complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Artefacts under: ${OUTPUT_ROOT}/"
echo "============================================================"
