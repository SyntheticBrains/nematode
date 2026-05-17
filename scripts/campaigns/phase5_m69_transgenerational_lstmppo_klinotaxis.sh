#!/usr/bin/env bash
# M6.9+ TEI re-evaluation campaign — three-arm campaign + tripwires
# ============================================================================
#
# Three subcommands map to the verification protocol described in
# the OpenSpec change ``add-transgenerational-memory-redesign``:
#
#   --smoke   F0 calibration smoke. 4 seeds × pop 6 × 1 gen × tei_on
#             only. Runs the four mid-flight tripwires (T1-T4) BEFORE
#             unblocking pilot. ~2 wall-hours.
#
#   --pilot   Pilot. 1 seed × pop 8 × 4 gens × 3 arms (tei_on,
#             weights_only, control). ~3 wall-hours. Aggregator
#             produces ``pilot_pivot_decision.md``; pause for user
#             review before unblocking full.
#
#   --full    Full campaign. 4 seeds (42, 43, 44, 45) × pop 16 ×
#             4 gens × 3 arms. ~22-28 wall-hours. Aggregator produces
#             per-seed + cross-seed + cross-arm verdict + PR-B
#             trigger decision.
#
# Launch-time sanity checks (per configuration-system spec scenarios
# § "campaign shell sanity-checks parity at launch"):
#
#   1. All three arm YAMLs MUST set the same ``fitness_survival_weight``
#      (composite fitness parity). Mismatch aborts before any worker
#      is dispatched.
#   2. The ``analysis`` extras (scipy) MUST be available. Required by
#      the aggregator at commit 6 for Wilcoxon + bootstrap CI. Missing
#      → exits with ``uv sync --extra analysis`` pointer.
#
# Output directory convention:
#   evolution_results/m69_transgenerational/{tei_on|weights_only|control}/seed-{N}/<session_id>/
#
# Each session produces best_params.json, history.csv, lineage.csv,
# checkpoint.pkl, and (for tei_on) the F0 substrate at
# inheritance/gen-000/genome-<id>.tei.pt.
#
# Tripwires (run at --smoke completion, BEFORE pilot is unblocked;
# all four MUST pass — failure triggers the design.md § D6 pivot table):
#
#   T1 F0 envelope: mean F0 survival_rate ∈ [0.30, 0.70] across the
#      4 calibration seeds. Verified post-smoke by the aggregator.
#   T2 Substrate diversity: pairwise coefficient-of-variation across
#      the 4 seeds' extracted bias_network state_dicts > 5%. Verified
#      by ``scripts/campaigns/m69_substrate_diversity.py`` (commit 7).
#   T3 M6-floor-to-beat: F0 survival > M6 "circle right always"
#      baseline on the new env (recomputable from M6 artefacts at K=0).
#      Operator-verified at smoke review.
#   T4 Substrate magnitude: mean absolute bias_network output > 0.1
#      on probes. Operator-verified at smoke review.
#
# Usage:
#   scripts/campaigns/phase5_m69_transgenerational_lstmppo_klinotaxis.sh --smoke
#   scripts/campaigns/phase5_m69_transgenerational_lstmppo_klinotaxis.sh --pilot
#   scripts/campaigns/phase5_m69_transgenerational_lstmppo_klinotaxis.sh --full
#
# Outputs land under ``${OUTPUT_ROOT:-evolution_results/m69_transgenerational}/``.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 --smoke|--pilot|--full" >&2
    exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m69_transgenerational}"
CONFIG_TEI_ON="configs/evolution/transgenerational_m69_tei_on.yml"
CONFIG_WEIGHTS_ONLY="configs/evolution/transgenerational_m69_weights_only.yml"
CONFIG_CONTROL="configs/evolution/transgenerational_m69_control.yml"
CONFIG_SMOKE="configs/evolution/transgenerational_m69_smoke.yml"

# Launch-time sanity check #1: composite-fitness parity across arms.
# Reads ``evolution.fitness_survival_weight`` from each arm YAML in a
# single Python invocation (one interpreter startup vs three) and
# diagnoses missing fields with a clean named-field message rather
# than a raw KeyError traceback. Aborts before any worker is
# dispatched if values diverge or any arm is missing the field.
check_fitness_survival_weight_parity() {
    local parity_out
    parity_out=$(uv run python -c "
import sys, yaml
arms = {
    'tei_on': '${CONFIG_TEI_ON}',
    'weights_only': '${CONFIG_WEIGHTS_ONLY}',
    'control': '${CONFIG_CONTROL}',
}
values = {}
for arm, path in arms.items():
    cfg = yaml.safe_load(open(path))
    evolution = cfg.get('evolution') or {}
    values[arm] = evolution.get('fitness_survival_weight', 'MISSING')
unique = set(values.values())
if len(unique) > 1 or 'MISSING' in unique:
    print('MISMATCH', values, file=sys.stderr)
    sys.exit(1)
print('OK', list(unique)[0])
") || {
        echo "ERROR: fitness_survival_weight parity violated across the three arm YAMLs." >&2
        echo "${parity_out}" >&2
        echo "All three arms MUST share the same fitness_survival_weight" >&2
        echo "for the M3 reproduction check via weights_only to be" >&2
        echo "uncorrupted by elite-selection-rule mismatch. 'MISSING'" >&2
        echo "means an arm's YAML omits the field; add it or fix the" >&2
        echo "diverging value, then re-launch." >&2
        exit 1
    }
    echo "Parity check: ${parity_out}"
}

# Launch-time sanity check #2: scipy availability. EAGER environment
# audit — scipy is only required by the commit-6 aggregator + commit-7
# diversity script, not by --smoke / --pilot / --full themselves. We
# check unconditionally so the operator's environment is verified
# before any compute is spent; better to know now than after 2 wall-h
# of smoke. scipy is in the ``analysis`` extras of
# packages/quantum-nematode/pyproject.toml — not core.
check_scipy_available() {
    if ! uv run python -c "import scipy.stats" 2>/dev/null; then
        echo "ERROR: scipy is not importable in the current uv env." >&2
        echo "The M6.9+ aggregator requires scipy for Wilcoxon +" >&2
        echo "bootstrap CI computation. Install via:" >&2
        echo "  uv sync --extra analysis" >&2
        echo "before re-launching." >&2
        exit 1
    fi
    echo "Scipy import check: OK"
}

case "$1" in
    --smoke)
        MODE="smoke"
        SEEDS=(42 43 44 45)
        POPULATION=6
        GENERATIONS=1
        ARMS=("tei_on")
        CONFIG_TEI_ON_RESOLVED="${CONFIG_SMOKE}"
        ;;
    --pilot)
        MODE="pilot"
        SEEDS=(42)
        POPULATION=8
        GENERATIONS=4
        ARMS=("tei_on" "weights_only" "control")
        CONFIG_TEI_ON_RESOLVED="${CONFIG_TEI_ON}"
        ;;
    --full)
        MODE="full"
        SEEDS=(42 43 44 45)
        POPULATION=16
        GENERATIONS=4
        ARMS=("tei_on" "weights_only" "control")
        CONFIG_TEI_ON_RESOLVED="${CONFIG_TEI_ON}"
        ;;
    *)
        echo "usage: $0 --smoke|--pilot|--full" >&2
        exit 1
        ;;
esac

check_fitness_survival_weight_parity
check_scipy_available

mkdir -p "${OUTPUT_ROOT}"
echo "============================================================"
echo "M6.9+ TEI re-evaluation — mode: ${MODE}"
echo "============================================================"
echo "Output root: ${OUTPUT_ROOT}"
echo "Seeds:       ${SEEDS[*]}"
echo "Population:  ${POPULATION}"
echo "Generations: ${GENERATIONS}"
echo "Arms:        ${ARMS[*]}"
echo "Start:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

for ARM in "${ARMS[@]}"; do
    case "${ARM}" in
        tei_on)        CONFIG="${CONFIG_TEI_ON_RESOLVED}" ;;
        weights_only)  CONFIG="${CONFIG_WEIGHTS_ONLY}" ;;
        control)       CONFIG="${CONFIG_CONTROL}" ;;
        *)             echo "ERROR: unknown arm ${ARM}" >&2; exit 1 ;;
    esac
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

if [[ "${MODE}" == "smoke" ]]; then
    echo
    echo "Next: run the four tripwires (T1-T4) before unblocking pilot."
    echo "  T1 F0 envelope:  inspect ${OUTPUT_ROOT}/tei_on/seed-*/per_gen_elites.jsonl"
    echo "  T2 Diversity:    uv run python scripts/campaigns/m69_substrate_diversity.py \\"
    echo "                     --campaign-root ${OUTPUT_ROOT}"
    echo "  T3 M6 floor:     recompute M6 'circle right always' baseline at K=0 on new env"
    echo "  T4 Magnitude:    inspect mean |bias_network output| via the diversity script"
    echo "All four MUST pass before launching --pilot. Pivot per design.md § D6 if any fails."
elif [[ "${MODE}" == "pilot" ]]; then
    echo
    echo "Next: review pilot_pivot_decision.md emitted by the aggregator."
    echo "  uv run python scripts/campaigns/aggregate_m69_pilot.py \\"
    echo "    --per-gen-csv <eval-dir>/per_gen_choice_index.csv \\"
    echo "    --output-dir <eval-dir> \\"
    echo "    --mode pilot"
    echo "User MUST review before unblocking --full."
fi
