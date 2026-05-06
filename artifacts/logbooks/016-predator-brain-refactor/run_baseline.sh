#!/usr/bin/env bash
# M1 regression baseline runner.
#
# Invoked once on M1 base commit (pre-refactor) and once on M1 head commit
# (post-refactor). Writes per-(arm, config, seed) summary rows to a CSV
# used by logbook 016 to gate the multi-metric regression criterion.
#
# Two arms:
#   - multi-agent: 200 ep × 3 multi-agent pursuit configs (5 agents each)
#   - single-agent: 100 ep × 2 single-agent pursuit configs
# Single-agent and multi-agent paths exercise different orchestration code
# paths through env.update_predators(); covering both is needed for a
# complete refactor regression check.
#
# Usage:
#   tmp/m1_regression_baseline/run_baseline.sh <pre|post>
#
# Output: tmp/m1_regression_baseline/baseline_<pre|post>.csv with header
#   arm,config,seed,mean_success,mean_total_food,mean_steps,
#   mean_predator_engagement,n_episodes,session_id
set -euo pipefail

LABEL="${1:?missing label arg (pre|post)}"
# Use git to resolve the repo root regardless of where this script lives.
# (After archival the script moved from tmp/m1_regression_baseline/ to
# artifacts/logbooks/016-predator-brain-refactor/, so a relative ../.. walk
# is fragile.) Falls back to a path-relative resolution if not in a git tree.
if REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
OUT_DIR="${REPO_ROOT}/tmp/m1_regression_baseline"
mkdir -p "${OUT_DIR}"
OUT_CSV="${OUT_DIR}/baseline_${LABEL}.csv"
SUMMARY_LOG="${OUT_DIR}/run_${LABEL}.log"

# Multi-agent arm: (arm_name, config_path, runs_per_seed)
MULTI_CONFIGS=(
  "configs/scenarios/multi_agent_pursuit/mlpppo_small_5agents_pursuit_oracle.yml"
  "configs/scenarios/multi_agent_pursuit/mlpppo_small_5agents_pursuit_no_alarm_oracle.yml"
  "configs/scenarios/multi_agent_pursuit/lstmppo_small_5agents_pursuit_alarm_klinotaxis.yml"
)
MULTI_RUNS=200

# Single-agent arm
SINGLE_CONFIGS=(
  "configs/scenarios/pursuit/mlpppo_small_oracle.yml"
  "configs/scenarios/pursuit/lstmppo_small_klinotaxis.yml"
)
SINGLE_RUNS=100

SEEDS=(42 43 44 45)

echo "arm,config,seed,mean_success,mean_total_food,mean_steps,mean_predator_engagement,n_episodes,session_id" >"${OUT_CSV}"
{
  echo "[$(date -u +%FT%TZ)] M1 regression baseline ${LABEL}"
  echo "Repo root: ${REPO_ROOT}"
  echo "Multi-agent configs: ${MULTI_CONFIGS[*]}"
  echo "Single-agent configs: ${SINGLE_CONFIGS[*]}"
  echo "Seeds: ${SEEDS[*]}"
  echo "Multi-agent runs/seed: ${MULTI_RUNS}"
  echo "Single-agent runs/seed: ${SINGLE_RUNS}"
} >"${SUMMARY_LOG}"

cd "${REPO_ROOT}"

run_one() {
  local arm="$1"
  local cfg="$2"
  local seed="$3"
  local runs="$4"

  local cfg_name
  cfg_name="$(basename "${cfg}" .yml)"
  local label="${arm}_${cfg_name}_seed${seed}"
  echo "[$(date -u +%FT%TZ)] start ${label}" | tee -a "${SUMMARY_LOG}"

  local sim_log="${OUT_DIR}/sim_${LABEL}_${label}.log"
  uv run scripts/run_simulation.py \
    --config "${cfg}" \
    --seed "${seed}" \
    --runs "${runs}" \
    --device cpu \
    --theme headless \
    --log-level INFO \
    >"${sim_log}" 2>&1 || {
      echo "[$(date -u +%FT%TZ)] FAILED ${label} (see ${sim_log})" | tee -a "${SUMMARY_LOG}"
      return 0
    }

  local session_id
  # Multi-agent emits "Session: <id>"; single-agent emits "Session ID: <id>".
  # Match either form.
  session_id="$(grep -oE 'Session( ID)?: [0-9a-f_]+' "${sim_log}" | head -1 | awk '{print $NF}' || true)"
  if [[ -z "${session_id}" ]]; then
    echo "[$(date -u +%FT%TZ)] no session_id for ${label}" | tee -a "${SUMMARY_LOG}"
    return 0
  fi

  local data_dir="${REPO_ROOT}/exports/${session_id}/session/data"
  local csv_target
  if [[ "${arm}" == "multi" ]]; then
    csv_target="${data_dir}/multi_agent_summary.csv"
  else
    csv_target="${data_dir}/simulation_results.csv"
  fi
  if [[ ! -f "${csv_target}" ]]; then
    echo "[$(date -u +%FT%TZ)] missing ${csv_target} for ${label}" | tee -a "${SUMMARY_LOG}"
    return 0
  fi

  # Extract metrics; schema differs between the two arms.
  # Multi-agent (multi_agent_summary.csv): mean_success in row, total_food,
  #   proximity_events, agents_alive_at_end. We read mean_success,
  #   total_food, proximity_events; steps are not in this CSV (single
  #   row per run aggregates per-agent steps); we read agents_alive_at_end
  #   as a saturation tell.
  # Single-agent (simulation_results.csv): success (0/1), foods_collected,
  #   steps, predator_encounters, successful_evasions per run.
  # If the metric extraction fails (parse error, empty CSV), abort the
  # whole baseline rather than silently writing zeros for this cell.
  if ! stats="$(uv run python -c "
import csv, statistics, sys
import yaml
from pathlib import Path
arm = '${arm}'
cfg_path = Path('${cfg}')
data = yaml.safe_load(cfg_path.read_text())
n_agents = int(data.get('multi_agent', {}).get('count', 1))
csv_path = '${csv_target}'

def to_bool(s):
    return 1.0 if str(s).strip().lower() == 'true' else 0.0

succ, food, steps, eng = [], [], [], []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        try:
            if arm == 'multi':
                # multi-agent summary: mean_success is per-run already
                succ.append(float(row.get('mean_success', 0) or 0))
                food.append(float(row.get('total_food', 0) or 0))
                # No per-run aggregate steps column in this CSV; emit 0 so
                # the field stays in the per-cell schema. Logbook calls out
                # this is a known data-availability gap, not a real metric.
                steps.append(0.0)
                eng.append(float(row.get('proximity_events', 0) or 0))
            else:
                # single-agent simulation_results.csv: success is the string
                # 'True'/'False' so coerce via to_bool
                succ.append(to_bool(row.get('success', 'False')))
                food.append(float(row.get('foods_collected', 0) or 0))
                steps.append(float(row.get('steps', 0) or 0))
                pe = float(row.get('predator_encounters', 0) or 0)
                ev = float(row.get('successful_evasions', 0) or 0)
                eng.append(pe + ev)
        except (KeyError, ValueError) as e:
            print(f'parse error in row {row}: {e}', file=sys.stderr)
            sys.exit(1)

# Fail loudly on missing or empty metric series — silent zero-rows
# defeat the point of the regression baseline.
if not succ or not food or not eng:
    print(
        f'no rows parsed from {csv_path} (succ={len(succ)} food={len(food)} eng={len(eng)})',
        file=sys.stderr,
    )
    sys.exit(1)
# Single-agent arm should also have non-empty steps; multi-agent intentionally fills 0.
if arm != 'multi' and not steps:
    print(f'no steps rows parsed from {csv_path}', file=sys.stderr)
    sys.exit(1)

def s(xs):
    return statistics.fmean(xs) if xs else 0.0

print(f'{s(succ):.6f},{s(food):.6f},{s(steps):.6f},{s(eng):.6f},{len(succ)}', end='')
")"; then
    echo "[$(date -u +%FT%TZ)] FAILED metric extraction for ${label}" | tee -a "${SUMMARY_LOG}"
    return 1
  fi
  echo "${arm},${cfg_name},${seed},${stats},${session_id}" >>"${OUT_CSV}"
  echo "[$(date -u +%FT%TZ)] done ${label} -> ${stats}" | tee -a "${SUMMARY_LOG}"
}

# Multi-agent arm
for cfg in "${MULTI_CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "multi" "${cfg}" "${seed}" "${MULTI_RUNS}"
  done
done

# Single-agent arm
for cfg in "${SINGLE_CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "single" "${cfg}" "${seed}" "${SINGLE_RUNS}"
  done
done

echo "[$(date -u +%FT%TZ)] M1 regression baseline ${LABEL} complete" | tee -a "${SUMMARY_LOG}"
echo "Wrote ${OUT_CSV}" | tee -a "${SUMMARY_LOG}"
