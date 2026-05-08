#!/usr/bin/env bash
# M5 co-evolution FULL RUN — 4 seeds × 50 generations.
# ============================================================================
#
# Per task 10.2: 4 seeds (42-45) × 50 gens × pop 24/16 × K=10.
# Wall-time ~30-60 hours total at parallel_workers=4 per side × 4
# sequential seeds (per design.md D4; lock from pilot's actual
# per-episode wall before launch).
#
# Verdict gate (task 10.4): GO if cycling OR escalation fires in ≥2 of
# 4 seeds; STOP if zero seeds; PIVOT if exactly 1.
#
# Per-seed warmstart: each seed loads its own warmstart anchor from
# `configs/evolution/coevolution_warmstart_prey/seed_{42..45}.json`.
# The full-run config templates `prey_gen0_seed_path: ...seed_42.json`
# as a placeholder; this wrapper rewrites it per seed via a tmp YAML
# so each seed boots from a different M3-elite anchor (independence
# across full-run seeds — see design.md D12).
#
# Outputs land under ${OUTPUT_ROOT}/seed-${SEED}/<session>/.
#
# Usage::
#
#   scripts/campaigns/phase5_m5_coevolution_full.sh
#
# Override compute knobs via env vars before launch:
#
#   OUTPUT_ROOT=evolution_results/m5_full_dryrun \
#       SEEDS="42 43" \
#       scripts/campaigns/phase5_m5_coevolution_full.sh
#
# Resume a single seed (continues from the last K-block checkpoint):
#
#   uv run python scripts/run_coevolution.py \
#       --config configs/evolution/coevolution_full.yml \
#       --seed 44 \
#       --resume "$(ls -dt evolution_results/m5_coevolution_full/seed-44/*/ | head -1)"

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-evolution_results/m5_coevolution_full}"
DRIVER="scripts/run_coevolution.py"
BASE_CONFIG="configs/evolution/coevolution_full.yml"
SEEDS="${SEEDS:-42 43 44 45}"

mkdir -p "${OUTPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Driver:      ${DRIVER}"
echo "Base config: ${BASE_CONFIG}"
echo "Seeds:       ${SEEDS}"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# Per-seed config rewrite directory. Tmp YAMLs that swap in the right
# warmstart bundle entry per seed. Lives under the output root so it's
# cleaned up alongside the campaign artefacts.
TMP_CONFIG_DIR="${OUTPUT_ROOT}/_per_seed_configs"
mkdir -p "${TMP_CONFIG_DIR}"

for SEED in ${SEEDS}; do
    SEED_OUTPUT="${OUTPUT_ROOT}/seed-${SEED}"
    mkdir -p "${SEED_OUTPUT}"
    PER_SEED_CONFIG="${TMP_CONFIG_DIR}/coevolution_full_seed_${SEED}.yml"

    # sed-rewrite the warmstart path. The original YAML uses
    # `seed_42.json` as a placeholder; this swap makes each seed load
    # its own M3-elite anchor.
    sed -e "s|coevolution_warmstart_prey/seed_42\.json|coevolution_warmstart_prey/seed_${SEED}.json|" \
        "${BASE_CONFIG}" > "${PER_SEED_CONFIG}"
    # Verify the substitution actually fired — silent no-ops would
    # otherwise leave every seed loading seed_42.json.
    if ! grep -q "coevolution_warmstart_prey/seed_${SEED}.json" "${PER_SEED_CONFIG}"; then
        echo "ERROR: warmstart substitution failed for seed ${SEED}." >&2
        echo "  expected `seed_42.json` placeholder in ${BASE_CONFIG}; not found." >&2
        echo "  edit BASE_CONFIG to restore the placeholder, or update the sed pattern." >&2
        exit 1
    fi

    echo "============================================================"
    echo "Seed ${SEED} — start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Per-seed config: ${PER_SEED_CONFIG}"
    echo "Output:          ${SEED_OUTPUT}"
    echo "============================================================"
    uv run python "${DRIVER}" \
        --config "${PER_SEED_CONFIG}" \
        --seed "${SEED}" \
        --output-dir "${SEED_OUTPUT}" \
        --log-level INFO
    echo "Seed ${SEED} — end:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
done

echo "============================================================"
echo "Full run complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo
echo "Next step: run the aggregator on ${OUTPUT_ROOT} to compute the"
echo "Red Queen verdict (task 10.3 + 10.4)."
