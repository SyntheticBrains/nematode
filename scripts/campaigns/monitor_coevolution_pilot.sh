#!/usr/bin/env bash
# Monitor an in-flight (or completed) co-evolution pilot/full run.
# ============================================================================
#
# Usage::
#
#   scripts/campaigns/monitor_coevolution_pilot.sh <output-root>
#
# `<output-root>` can point at any of:
#   - A pilot wrapper output root (`<root>/pilot_run/{arm_a,arm_b}/<session>/`)
#   - A per-arm output dir (`<root>/{arm_a,arm_b}/<session>/`)
#   - A single session dir (`<root>/<session>/` containing `prey/`, `predator/`)
#   - A campaign root with `seed-<N>/` subdirs (full-run layout)
#
# The script auto-discovers any session dir under `<output-root>` that
# contains `prey/lineage.csv` and reports per-session:
#   - Progress (latest prey/predator generation, target=30 at pilot scale)
#   - Per-generation mean fitness (prey + predator)
#   - Generality probe rows (raw — interpret per the design doc)
#   - Wall-time so far (sum of `wall_seconds` from walltime.csv)
#   - Champion-history depth (K-blocks completed per side)
#
# Read-only — safe to run while the pilot is still writing.

set -euo pipefail

if [ $# -ne 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $(basename "$0") <output-root>" >&2
    echo "  Reports progress + fitness + probe + wall-time for every" >&2
    echo "  co-evolution session found under <output-root>." >&2
    exit 1
fi

ROOT="$1"
if [ ! -d "$ROOT" ]; then
    echo "Error: $ROOT is not a directory." >&2
    exit 1
fi

# Discover every session dir (one that contains `prey/lineage.csv`).
# `find` is the simplest way to handle all four layouts uniformly.
mapfile -t SESSIONS < <(find "$ROOT" -type f -name "lineage.csv" -path "*/prey/lineage.csv" -exec dirname {} \; | xargs -I{} dirname {} | sort -u)

if [ ${#SESSIONS[@]} -eq 0 ]; then
    echo "No co-evolution sessions found under $ROOT (looked for prey/lineage.csv)."
    # If a wrapper log exists at the root, surface its tail so the user
    # can see whether the run hasn't started yet vs has crashed.
    if [ -f "$ROOT/pilot.log" ]; then
        echo
        echo "=== Tail of $ROOT/pilot.log ==="
        tail -10 "$ROOT/pilot.log"
    fi
    exit 0
fi

echo "Discovered ${#SESSIONS[@]} session(s) under $ROOT:"
for s in "${SESSIONS[@]}"; do
    # Print the path relative to ROOT for readability.
    rel="${s#"$ROOT"/}"
    echo "  - $rel"
done
echo

# ---------------------------------------------------------------------------
# Per-session report
# ---------------------------------------------------------------------------

format_seconds() {
    # Print N.NN seconds as "MMm SSs" or "HHh MMm" for readability.
    local total="$1"
    if [ -z "$total" ] || [ "$total" = "0" ]; then
        echo "0s"
        return
    fi
    awk -v s="$total" 'BEGIN {
        h = int(s/3600); m = int((s%3600)/60); sec = int(s%60);
        if (h > 0) printf "%dh %02dm %02ds", h, m, sec;
        else if (m > 0) printf "%dm %02ds", m, sec;
        else printf "%ds", sec;
    }'
}

mean_fitness_per_gen() {
    # awk a `generation,...,fitness,...` CSV (col 1 = gen, col 4 =
    # fitness in the standard lineage schema). Skip header.
    local csv="$1"
    [ ! -f "$csv" ] && { echo "  (no lineage CSV)"; return; }
    awk -F, 'NR>1 {sum[$1]+=$4; n[$1]++}
        END {
            if (length(n) == 0) { print "  (no rows yet)"; exit }
            for (g in sum) gens[g+0] = g;
            for (g in gens) printf "  gen %2d: mean=%.3f (n=%d)\n", g, sum[g]/n[g], n[g];
        }' "$csv" | sort -k 2n
}

probe_summary() {
    # Pretty-print generality_probe.csv with one block per (gen, side).
    local csv="$1"
    [ ! -f "$csv" ] && { echo "  (no probe CSV)"; return; }
    local rows
    rows=$(awk -F, 'NR>1' "$csv" | wc -l | tr -d ' ')
    if [ "$rows" = "0" ]; then
        echo "  (no probe rows yet — first probe fires at the K-block boundary that satisfies generality_probe_every cadence)"
        return
    fi
    echo "  $rows probe row(s):"
    awk -F, 'NR>1 {printf "    gen=%s side=%s opp=%s fitness=%s\n", $1, $2, $3, $4}' "$csv"
}

wall_summary() {
    local csv="$1"
    [ ! -f "$csv" ] && { echo "  (no walltime CSV)"; return; }
    awk -F, 'NR>1 && $1 == "evaluation" {n_eval++; eval_sum+=$6}
             NR>1 && $1 == "generation" {n_gen++; gen_sum+=$6}
        END {
            printf "  evaluations: %d rows, sum=%.1fs (mean %.2fs/eval)\n",
                n_eval, eval_sum, (n_eval>0 ? eval_sum/n_eval : 0);
            printf "  generations: %d rows, sum=%.1fs (mean %.1fs/gen)\n",
                n_gen, gen_sum, (n_gen>0 ? gen_sum/n_gen : 0);
        }' "$csv"
}

champion_depth() {
    local json="$1"
    [ ! -f "$json" ] && { echo "  (no champion_history.json)"; return; }
    # Use python to extract depth — `jq` may not be installed everywhere.
    python3 - "$json" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    data = json.load(fh)
prey = len(data.get("prey", []))
predator = len(data.get("predator", []))
k_block_index = data.get("k_block_index", "?")
print(f"  prey K-blocks completed: {prey}; predator K-blocks completed: {predator}; loop k_block_index: {k_block_index}")
PY
}

now_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "Report timestamp: $now_utc"
echo

for session in "${SESSIONS[@]}"; do
    rel="${session#"$ROOT"/}"
    echo "============================================================"
    echo "Session: $rel"
    echo "============================================================"

    # --- Progress: latest gen on each side ---
    prey_csv="$session/prey/lineage.csv"
    predator_csv="$session/predator/lineage.csv"
    prey_gen=$(tail -1 "$prey_csv" 2>/dev/null | cut -d, -f1)
    pred_gen=$(tail -1 "$predator_csv" 2>/dev/null | cut -d, -f1)
    [ "$prey_gen" = "generation" ] && prey_gen="(no rows)"
    [ "$pred_gen" = "generation" ] && pred_gen="(no rows)"
    echo "Latest generation: prey=$prey_gen, predator=$pred_gen"
    echo

    # --- K-block completion ---
    echo "K-block completion (champion_history.json):"
    champion_depth "$session/champion_history.json"
    echo

    # --- Per-generation mean fitness ---
    echo "Prey per-generation mean fitness:"
    mean_fitness_per_gen "$prey_csv"
    echo
    echo "Predator per-generation mean fitness:"
    mean_fitness_per_gen "$predator_csv"
    echo

    # --- Generality probe rows ---
    echo "Generality probe (post-K-block, every generality_probe_every gens):"
    probe_summary "$session/generality_probe.csv"
    echo

    # --- Wall-time so far ---
    echo "Wall-time accumulated:"
    wall_summary "$session/walltime.csv"
    echo
done

# ---------------------------------------------------------------------------
# Wrapper-script tail (only meaningful when ROOT is a pilot wrapper root)
# ---------------------------------------------------------------------------

if [ -f "$ROOT/pilot.log" ]; then
    echo "============================================================"
    echo "Pilot wrapper log tail ($ROOT/pilot.log):"
    echo "============================================================"
    tail -8 "$ROOT/pilot.log"
fi
