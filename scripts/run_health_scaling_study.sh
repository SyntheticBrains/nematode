#!/bin/bash
# Health System Predator Scaling Study - Parallel Execution Script
# Experiment 005: Evaluating agent adaptation to increasing predator density
#
# Runs 4 sessions in parallel, cycling through all conditions
# Total: 120 sessions (12 conditions x 10 sessions each)
#
# Usage:
#   ./scripts/run_health_scaling_study.sh
#
# Output:
#   - Session logs: logs/health_scaling_study/
#   - Summary CSV: logs/health_scaling_study/summary.csv
#   - Experiment data: experiments/*.json (auto-tracked)
#   - Exports: exports/<session_id>/

set -e

# Configuration
RUNS_PER_SESSION=50
SESSIONS_PER_CONDITION=10
PARALLEL_JOBS=4
STUDY_NAME="health_scaling_study"
CONFIG_DIR="configs/studies/health_scaling"
LOG_DIR="logs/${STUDY_NAME}"
SUMMARY_FILE="${LOG_DIR}/summary.csv"

# Conditions to run (order: all health-enabled first, then controls)
CONDITIONS=(
    "health_p1"
    "health_p2"
    "health_p3"
    "health_p5"
    "health_p7"
    "health_p10"
    "control_p1"
    "control_p2"
    "control_p3"
    "control_p5"
    "control_p7"
    "control_p10"
)

# Create log directory
mkdir -p "${LOG_DIR}"

# Initialize summary CSV (always recreate in test mode to avoid stale data)
if [ "$TEST_MODE" -eq 1 ] || [ ! -f "${SUMMARY_FILE}" ]; then
    echo "condition,session,session_id,success_rate,starved,health_depleted,max_steps,avg_reward,avg_foods,timestamp" > "${SUMMARY_FILE}"
fi

# Function to run a single session and extract key metrics
run_session() {
    local condition=$1
    local session_num=$2
    local config_file="${CONFIG_DIR}/${condition}.yml"
    local log_file="${LOG_DIR}/${condition}_session${session_num}.log"

    echo "[$(date '+%H:%M:%S')] Starting ${condition} session ${session_num}..."

    # Run simulation and capture output
    if uv run python scripts/run_simulation.py \
        --config "${config_file}" \
        --runs "${RUNS_PER_SESSION}" \
        --log-level INFO \
        --show-last-frame-only \
        --track-per-run \
        --track-experiment \
        > "${log_file}" 2>&1; then

        # Extract session ID from log
        session_id=$(grep "Session ID:" "${log_file}" | tail -1 | awk '{print $NF}')

        # Extract metrics from log (using grep and awk)
        # Format: "Success rate: 20.00%"
        success_rate=$(grep "^Success rate:" "${log_file}" | tail -1 | awk '{print $3}' | tr -d '%')

        # Format: "Failed runs - Starved: 1 (20.0%)"
        starved=$(grep "Failed runs - Starved:" "${log_file}" | tail -1 | awk '{print $5}')
        [ -z "$starved" ] && starved="0"

        # Format: "Failed runs - Health Depleted: 1 (20.0%)"
        health_depleted=$(grep "Failed runs - Health Depleted:" "${log_file}" | tail -1 | awk '{print $6}')
        [ -z "$health_depleted" ] && health_depleted="0"

        # Format: "Failed runs - Max Steps: 2 (40.0%)"
        max_steps=$(grep "Failed runs - Max Steps:" "${log_file}" | tail -1 | awk '{print $6}')
        [ -z "$max_steps" ] && max_steps="0"

        # Format: "Average reward per run: 10.36"
        avg_reward=$(grep "^Average reward per run:" "${log_file}" | tail -1 | awk '{print $NF}')

        # Format: "Average foods collected per run: 5.40"
        avg_foods=$(grep "^Average foods collected per run:" "${log_file}" | tail -1 | awk '{print $NF}')

        timestamp=$(date '+%Y-%m-%d %H:%M:%S')

        # Append to summary
        echo "${condition},${session_num},${session_id},${success_rate},${starved},${health_depleted},${max_steps},${avg_reward},${avg_foods},${timestamp}" >> "${SUMMARY_FILE}"

        echo "[$(date '+%H:%M:%S')] Completed ${condition} session ${session_num}: ${success_rate}% success (ID: ${session_id})"
    else
        echo "[$(date '+%H:%M:%S')] FAILED ${condition} session ${session_num} - check ${log_file}"
        echo "${condition},${session_num},FAILED,,,,,,,$(date '+%Y-%m-%d %H:%M:%S')" >> "${SUMMARY_FILE}"
    fi
}

# Export function for parallel execution
export -f run_session
export CONFIG_DIR LOG_DIR SUMMARY_FILE RUNS_PER_SESSION

# Build job list
echo "=========================================="
echo "Health System Predator Scaling Study"
echo "=========================================="
echo "Conditions: ${#CONDITIONS[@]}"
echo "Sessions per condition: ${SESSIONS_PER_CONDITION}"
echo "Runs per session: ${RUNS_PER_SESSION}"
echo "Parallel jobs: ${PARALLEL_JOBS}"
echo "Total sessions: $((${#CONDITIONS[@]} * SESSIONS_PER_CONDITION))"
echo "Total runs: $((${#CONDITIONS[@]} * SESSIONS_PER_CONDITION * RUNS_PER_SESSION))"
echo ""
echo "Log directory: ${LOG_DIR}"
echo "Summary file: ${SUMMARY_FILE}"
echo "=========================================="
echo ""

# Create jobs file
JOBS_FILE="${LOG_DIR}/jobs.txt"
> "${JOBS_FILE}"

for condition in "${CONDITIONS[@]}"; do
    for session in $(seq 1 ${SESSIONS_PER_CONDITION}); do
        echo "${condition} ${session}" >> "${JOBS_FILE}"
    done
done

# Count total and remaining jobs
TOTAL_JOBS=$(wc -l < "${JOBS_FILE}")
COMPLETED_JOBS=$(grep -c "^${CONDITIONS[0]}," "${SUMMARY_FILE}" 2>/dev/null || echo "0")

echo "Starting execution..."
echo "Progress will be logged to ${LOG_DIR}/"
echo ""

# Run jobs in parallel using xargs
# Each job runs run_session with condition and session number
cat "${JOBS_FILE}" | xargs -P ${PARALLEL_JOBS} -L 1 bash -c 'run_session $0 $1'

echo ""
echo "=========================================="
echo "Study Complete!"
echo "=========================================="
echo "Summary: ${SUMMARY_FILE}"
echo ""
echo "Quick stats:"
echo ""
column -t -s',' "${SUMMARY_FILE}" | head -20
echo ""
echo "Full results in: ${SUMMARY_FILE}"
echo "Individual logs in: ${LOG_DIR}/"
echo "Experiment JSONs in: experiments/"
