# Legacy Benchmarks

This folder contains benchmark results from before the reproducibility infrastructure
was implemented (pre-Phase 0 completion).

## Important Notes

- These benchmarks **lack per-run seed tracking** and cannot be reproduced exactly
- They will be **removed** once re-run with the new tracking system
- Do not use these as reference for new submissions

## Migration Status

| Benchmark | Status | New Location |
|-----------|--------|--------------|
| dynamic_predator_small/classical | Pending re-run | - |
| dynamic_predator_small/quantum | Pending re-run | - |
| dynamic_small/classical | Pending re-run | - |
| dynamic_small/quantum | Pending re-run | - |
| static_maze/classical | Pending re-run | - |
| static_maze/quantum | Pending re-run | - |

## Re-running Legacy Benchmarks

To re-run a legacy benchmark with full reproducibility:

1. Use the original config file from the experiment exports
2. Run 10+ independent sessions with the new seeding infrastructure
3. Submit using `benchmark_submit.py` which will:
   - Validate seed uniqueness across all sessions
   - Move experiments to `artifacts/experiments/`
   - Create aggregate benchmark in new format
