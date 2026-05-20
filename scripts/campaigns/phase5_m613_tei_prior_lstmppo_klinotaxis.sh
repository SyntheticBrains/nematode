#!/usr/bin/env bash
# TEI-as-prior-on-Lamarckian campaign — three-arm campaign + tripwires
# ============================================================================
#
# Three subcommands map to the verification protocol described in
# the OpenSpec change ``add-tei-prior-on-m3``:
#
#   --smoke   K_test calibration smoke. 1-2 seeds × pop 6 × 2 gens ×
#             weights_only ONLY (the Lamarckian baseline arm —
#             calibration target is "find K_test where Lamarckian
#             retraining has headroom AND beats control"). Operator
#             reviews T1'-T4' tripwires before unblocking pilot.
#             ~2-4 wall-hours.
#
#   --pilot   Pilot. 1 seed × pop 8 × 4 gens × 3 arms (tei_weights,
#             weights_only, control). ~3 wall-hours. Aggregator
#             emits ``pilot_pivot_decision.md``; pause for user
#             review before unblocking full per
#             feedback_logbook_review_before_verdict.md.
#
#   --full    Full campaign (only if pilot GO). 4 seeds × pop 16 × 4
#             gens × 3 arms. ~14-18 wall-hours. Aggregator emits
#             per-seed + cross-seed + cross-arm verdict + follow-up
#             trigger (or null-finding note if STOP).
#
# Launch-time sanity checks (per configuration-system spec scenarios
# § "launch-time parity check fires on env-config divergence"):
#
#   1. fitness_survival_weight + fitness_metric MUST match across arms.
#   2. env fields MUST match across arms: grid_size, predators.count,
#      predators.predator_damage, foraging.min_food_predator_distance.
#   3. K_test alignment MUST hold: weights_only.learn_episodes_per_eval
#      AND control.learn_episodes_per_eval MUST match tei_weights F1+
#      lawn_schedule entries' ppo_train_episodes. This is the
#      load-bearing check — the cross-arm primary verdict
#      tei_weights − weights_only is only meaningful when both arms
#      use the same compute budget.
#   4. scipy availability (aggregator dependency).
#
# Output directory convention:
#   evolution_results/m613_tei_prior/{tei_weights|weights_only|control}/seed-{N}/<session_id>/
#
# Tripwires (run at --smoke completion, BEFORE pilot is unblocked;
# all four MUST pass — failure triggers the design.md § D6 pivot table):
#
#   T1' F0 envelope: mean F0 survival_rate ∈ [0.30, 0.70] across
#       smoke seeds. Reads from eval_diagnostics.jsonl.
#   T2' Substrate diversity: pairwise CoV > 5% across calibration
#       seeds' bias_network state_dicts. Reuses the existing
#       m69_substrate_diversity.py script unchanged — composed-mode
#       substrate extraction is byte-identical to pure-TEI extraction.
#       (Note: T2'/T4' require a tei_weights-arm smoke run on top of
#       the weights_only K_test calibration; see "Smoke flow" below.)
#   T3' Lamarckian-headroom (load-bearing):
#       ``weights_only F1+ ≤ 0.95 × F0`` (Lamarckian retraining has
#       headroom) AND ``weights_only F1+ ≥ 1.2 × control F1+`` (it is
#       doing useful
#       work). Operator-verified at smoke review.
#   T4' Substrate magnitude: mean |bias_network output| > 0.1.
#
# Smoke flow:
#   1. Run weights_only smoke at K_test=1000. Read F0+F1 survival_rate
#      from eval_diagnostics.jsonl. Check T1'+T3'.
#      - If T3' fails high (F1 ≥ 0.95×F0): edit smoke YAML to K=500, re-run.
#      - If T3' fails low (F1 < 0.80×F0): edit smoke YAML to K=1500, re-run.
#      - Cap at 2 passes; STOP if neither lands.
#   2. Once K_test is fixed by T3', update weights_only/control YAML
#      learn_episodes_per_eval AND tei_weights F1+ lawn_schedule to
#      that K_test value (the launcher parity check will assert).
#   3. Run a tei_weights smoke (1 seed × pop 6 × 1 gen) to harvest
#      the substrate and verify T2'+T4'.
#   4. All four tripwires PASS → unblock --pilot.
#
# Usage:
#   scripts/campaigns/phase5_m613_tei_prior_lstmppo_klinotaxis.sh --smoke
#   scripts/campaigns/phase5_m613_tei_prior_lstmppo_klinotaxis.sh --pilot
#   scripts/campaigns/phase5_m613_tei_prior_lstmppo_klinotaxis.sh --full
#
# Outputs land under ``${OUTPUT_ROOT:-evolution_results/m613_tei_prior}/``.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 --smoke|--pilot|--full" >&2
    exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m613_tei_prior}"
CONFIG_TEI_WEIGHTS="configs/evolution/tei_prior_m613_tei_weights.yml"
CONFIG_WEIGHTS_ONLY="configs/evolution/tei_prior_m613_weights_only.yml"
CONFIG_CONTROL="configs/evolution/tei_prior_m613_control.yml"
CONFIG_SMOKE="configs/evolution/tei_prior_m613_smoke.yml"

# Launch-time sanity check #1+#2+#3: cross-arm parity audit.
#
# Verifies fitness, env, AND K_test parity in a single Python
# invocation so an inconsistent campaign can't reach a worker
# (better to fail at launch than after 2 wall-h of pilot).
#
# Audit dimensions:
#   - fitness_survival_weight: same scalar across arms (composite
#     fitness parity, inherited contract from the pure-TEI campaign).
#   - fitness_metric: same primary metric across arms (campaign
#     specifies ``survival_rate``).
#   - env.grid_size + predators.count + predators.predator_damage +
#     foraging.min_food_predator_distance: same env across arms (the
#     audit-B/C corrections must apply uniformly).
#   - K_test alignment: weights_only.learn_episodes_per_eval AND
#     control.learn_episodes_per_eval AND tei_weights F1+
#     lawn_schedule entries' ppo_train_episodes ALL match.
#
# stderr merged into stdout via ``2>&1`` so MISMATCH diagnostics
# reach the failure branch (lesson from the prior campaign's
# logbook 019 review).
check_cross_arm_parity() {
    local parity_out
    parity_out=$(uv run python -c "
import sys, yaml

arms = {
    'tei_weights': '${CONFIG_TEI_WEIGHTS}',
    'weights_only': '${CONFIG_WEIGHTS_ONLY}',
    'control': '${CONFIG_CONTROL}',
}

# Load all three arm YAMLs once.
parsed = {arm: yaml.safe_load(open(path)) for arm, path in arms.items()}

# (a) fitness_survival_weight + fitness_metric parity.
fsw = {arm: (cfg.get('evolution') or {}).get('fitness_survival_weight', 'MISSING')
       for arm, cfg in parsed.items()}
fm = {arm: (cfg.get('evolution') or {}).get('fitness_metric', 'composite')
      for arm, cfg in parsed.items()}
if len(set(fsw.values())) > 1 or 'MISSING' in set(fsw.values()):
    print('MISMATCH fitness_survival_weight=' + str(fsw), file=sys.stderr)
    sys.exit(1)
if len(set(fm.values())) > 1:
    print('MISMATCH fitness_metric=' + str(fm), file=sys.stderr)
    sys.exit(2)

# (b) Env-field parity. Reach into nested structure with defensive
# .get() chains so a malformed YAML yields a 'MISSING' rather than
# an AttributeError.
def env_field(cfg, *path):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict):
            return 'MISSING'
        cur = cur.get(key)
        if cur is None:
            return 'MISSING'
    return cur

env_fields = {
    'grid_size': ('environment', 'grid_size'),
    'predators.count': ('environment', 'predators', 'count'),
    'predator_damage': ('environment', 'health', 'predator_damage'),
    'min_food_predator_distance': ('environment', 'foraging', 'min_food_predator_distance'),
}
for label, path in env_fields.items():
    vals = {arm: env_field(cfg, *path) for arm, cfg in parsed.items()}
    if len(set(map(str, vals.values()))) > 1 or 'MISSING' in vals.values():
        print('MISMATCH ' + label + '=' + str(vals), file=sys.stderr)
        sys.exit(3)

# (c) K_test alignment. weights_only AND control use
# learn_episodes_per_eval as the per-gen K (no schedule); tei_weights
# uses lawn_schedule F1+ entries' ppo_train_episodes (F0 is K_full
# 2000, F1+ is K_test). All three MUST match at K_test.
wo_k = env_field(parsed['weights_only'], 'evolution', 'learn_episodes_per_eval')
ctrl_k = env_field(parsed['control'], 'evolution', 'learn_episodes_per_eval')
tg_block = env_field(parsed['tei_weights'], 'evolution', 'transgenerational')
tw_schedule = tg_block.get('lawn_schedule') if isinstance(tg_block, dict) else None
if not isinstance(tw_schedule, list):
    print('MISMATCH tei_weights.transgenerational.lawn_schedule missing', file=sys.stderr)
    sys.exit(4)
# F1+ entries (gen >= 1) MUST all share K_test.
f1plus_ks = sorted({
    entry.get('ppo_train_episodes')
    for entry in tw_schedule
    if isinstance(entry, dict) and (entry.get('generation') or 0) >= 1
})
if len(f1plus_ks) != 1:
    print('MISMATCH tei_weights F1+ ppo_train_episodes diverge: ' + str(f1plus_ks), file=sys.stderr)
    sys.exit(5)
tw_k = f1plus_ks[0]
if not (wo_k == ctrl_k == tw_k):
    print('MISMATCH K_test: weights_only=' + str(wo_k)
          + ' control=' + str(ctrl_k) + ' tei_weights F1+=' + str(tw_k), file=sys.stderr)
    sys.exit(6)

print('OK fitness_survival_weight=' + str(list(set(fsw.values()))[0])
      + ' fitness_metric=' + str(list(set(fm.values()))[0])
      + ' K_test=' + str(tw_k))
" 2>&1) || {
        echo "ERROR: cross-arm parity violated." >&2
        echo "${parity_out}" >&2
        echo "The cross-arm primary verdict (tei_weights − weights_only)" >&2
        echo "requires all three arms to share env config, fitness metric," >&2
        echo "and K_test compute budget. Fix the diverging field and re-launch." >&2
        exit 1
    }
    echo "Parity check: ${parity_out}"
}

# Launch-time sanity check #4: scipy availability (aggregator
# dependency). EAGER environment audit — better to fail at launch
# than after 2 wall-h of pilot.
check_scipy_available() {
    if ! uv run python -c "import scipy.stats" 2>/dev/null; then
        echo "ERROR: scipy is not importable in the current uv env." >&2
        echo "The aggregator requires scipy for Wilcoxon +" >&2
        echo "bootstrap CI computation. Install via:" >&2
        echo "  uv sync --extra analysis" >&2
        echo "before re-launching." >&2
        exit 1
    fi
    echo "Scipy import check: OK"
}

case "$1" in
    --smoke)
        # K_test calibration smoke: weights_only arm only, 1 seed at
        # the K specified in the smoke YAML. Operator reviews
        # eval_diagnostics.jsonl for T1'+T3' BEFORE re-tuning K_test
        # or unblocking --pilot. The substrate-diversity tripwires
        # T2'+T4' require a separate tei_weights smoke pass after
        # K_test is locked in (see header comment "Smoke flow").
        MODE="smoke"
        SEEDS=(42)
        POPULATION=6
        GENERATIONS=2
        ARMS=("weights_only")
        CONFIG_WEIGHTS_ONLY_RESOLVED="${CONFIG_SMOKE}"
        CONFIG_TEI_WEIGHTS_RESOLVED="${CONFIG_TEI_WEIGHTS}"
        ;;
    --pilot)
        MODE="pilot"
        SEEDS=(42)
        POPULATION=8
        GENERATIONS=4
        ARMS=("tei_weights" "weights_only" "control")
        CONFIG_WEIGHTS_ONLY_RESOLVED="${CONFIG_WEIGHTS_ONLY}"
        CONFIG_TEI_WEIGHTS_RESOLVED="${CONFIG_TEI_WEIGHTS}"
        ;;
    --full)
        MODE="full"
        SEEDS=(42 43 44 45)
        POPULATION=16
        GENERATIONS=4
        ARMS=("tei_weights" "weights_only" "control")
        CONFIG_WEIGHTS_ONLY_RESOLVED="${CONFIG_WEIGHTS_ONLY}"
        CONFIG_TEI_WEIGHTS_RESOLVED="${CONFIG_TEI_WEIGHTS}"
        ;;
    *)
        echo "usage: $0 --smoke|--pilot|--full" >&2
        exit 1
        ;;
esac

check_cross_arm_parity
check_scipy_available

# Smoke-mode K disclosure: the parity check above audits the PRODUCTION
# trio (CONFIG_TEI_WEIGHTS / CONFIG_WEIGHTS_ONLY / CONFIG_CONTROL) —
# correct for pilot/full because production is what gets dispatched
# there. Under --smoke the dispatched YAML is CONFIG_SMOKE (a
# weights_only variant whose K is calibration-tunable per design.md
# § D5). The operator may edit CONFIG_SMOKE between passes without
# updating production until T3' locks K_test in — that's the
# expected calibration flow. To prevent "parity passed but smoke ran
# at a different K" confusion, surface the smoke YAML's K and flag
# any divergence from production as INFORMATIONAL (not an error).
if [[ "${MODE}" == "smoke" ]]; then
    SMOKE_K=$(uv run python -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG_SMOKE}'))
print((cfg.get('evolution') or {}).get('learn_episodes_per_eval', 'MISSING'))
")
    PROD_WO_K=$(uv run python -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG_WEIGHTS_ONLY}'))
print((cfg.get('evolution') or {}).get('learn_episodes_per_eval', 'MISSING'))
")
    if [[ "${SMOKE_K}" != "${PROD_WO_K}" ]]; then
        echo "INFO: smoke YAML K=${SMOKE_K} differs from production weights_only K=${PROD_WO_K}."
        echo "      This is expected during K_test calibration; the smoke dispatches at K=${SMOKE_K}."
        echo "      After T3' locks K_test, update the production trio to match before --pilot."
    else
        echo "Smoke K=${SMOKE_K} matches production K=${PROD_WO_K}."
    fi
fi

mkdir -p "${OUTPUT_ROOT}"
echo "============================================================"
echo "TEI-as-prior-on-Lamarckian — mode: ${MODE}"
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
        tei_weights)   CONFIG="${CONFIG_TEI_WEIGHTS_RESOLVED}" ;;
        weights_only)  CONFIG="${CONFIG_WEIGHTS_ONLY_RESOLVED}" ;;
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
    echo "Next: review T1'/T3' from the weights_only smoke before unblocking pilot."
    echo "  T1' F0 envelope: inspect ${OUTPUT_ROOT}/weights_only/seed-*/eval_diagnostics.jsonl"
    echo "                   for mean F0 survival_rate ∈ [0.30, 0.70]."
    echo "  T3' Lamarckian-headroom: F1 ≤ 0.95×F0 (Lamarckian has headroom)"
    echo "                   AND F1 ≥ 1.2×control_F1 (Lamarckian doing useful"
    echo "                   work). If T3' fails high, edit"
    echo "                   tei_prior_m613_smoke.yml learn_episodes_per_eval to 500"
    echo "                   and re-run --smoke. If T3' fails low, bump to 1500."
    echo "                   Caps at 2 calibration passes."
    echo
    echo "After K_test is fixed by T3', update tei_weights YAML F1+ lawn_schedule"
    echo "+ weights_only/control YAML learn_episodes_per_eval to that K_test value."
    echo "Then run a tei_weights smoke pass to verify T2'+T4':"
    echo "  uv run python scripts/campaigns/m69_substrate_diversity.py \\"
    echo "    --campaign-root ${OUTPUT_ROOT} --arm tei_weights"
    echo
    echo "All four tripwires (T1'-T4') MUST pass before launching --pilot."
elif [[ "${MODE}" == "pilot" ]]; then
    echo
    echo "Next: aggregator emits pilot_pivot_decision.md classifying outcome"
    echo "against design.md § D6's six-row pre-declared pivot table."
    echo "  uv run python scripts/campaigns/aggregate_m613_pilot.py \\"
    echo "    --campaign-root ${OUTPUT_ROOT} \\"
    echo "    --output-dir <aggregate-output-dir> \\"
    echo "    --mode pilot"
    echo "User MUST review pilot_pivot_decision.md before unblocking --full"
    echo "per feedback_logbook_review_before_verdict.md."
fi
