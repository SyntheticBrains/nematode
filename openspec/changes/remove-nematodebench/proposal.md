## Why

NematodeBench — the curated benchmark submission, validation and leaderboard system — was demoted from a Phase 7 public-launch deliverable to "internal tooling" at the v4 roadmap rewrite (2026-05-23), on the stated justification that it remains "useful for reproducibility and for the architecture-comparison protocol itself" ([roadmap.md](../../../docs/roadmap.md) § NematodeBench). Two full phases of evidence say that justification did not hold.

The architecture-comparison protocol never routed through it. Phases 5 and 6 ranked architectures with [`scripts/analysis/weight_search_architecture_ranking.py`](../../../scripts/analysis/weight_search_architecture_ranking.py) reading per-seed `--track-experiment` output directly; the submission pipeline, the aggregation-across-sessions machinery and the leaderboard generator were never invoked by a single logbook. The `benchmarks/` corpus is six submissions authored by one contributor on 2025-12-28/29 and never regenerated — no entry exists for any of the 21 brain architectures added since. `BENCHMARKS.md` has not had a content commit since 2025-12-28 and still advertises `static_maze/quantum|classical` categories, an environment deleted from the code in `9a452fd5` (2026-01-30). Every commit touching the system since 2026-01 has been drive-by maintenance dragging it through unrelated refactors (brain renames, config reorg, lint fixes).

The cost is ~2.4k lines of unmaintained, un-exercised surface plus 78 LFS JSONs, carried into Phase 7 for no consumer. The one component the protocol genuinely depends on — the convergence detector and composite score in `benchmark/convergence.py`, which received real feature work as recently as 2026-06-21 ([add-level-agnostic-convergence-metric](../archive/2026-06-21-add-level-agnostic-convergence-metric/)) — is not benchmark-submission code at all. It is the ranked-metric producer for `architecture-comparison-protocol`, and it is retained.

## What Changes

- **Remove the submission/leaderboard/validation system**: `benchmark/{leaderboard,categorization,validation}.py`, `experiment/{submission,validation}.py`, `BenchmarkMetadata`, `scripts/{benchmark_submit,evaluate_submission}.py`, the `benchmarks/` and `artifacts/benchmarks/` corpora, `BENCHMARKS.md`, and `docs/nematodebench/`.
- **Retain and relocate the convergence detector**: `benchmark/convergence.py` → `experiment/convergence.py`, co-located with its sole consumer (`experiment/tracker.py`) and the `ResultsMetadata` model it populates. The `benchmark/` package disappears. No symbol renames — `analyze_convergence`, `detect_convergence`, `ConvergenceMetrics`, `calculate_learning_speed`, `calculate_stability` and the `composite_benchmark_score` field are all referenced by live specs, historical artifacts, or both.
- **Migrate three genuinely-live requirements out of `benchmark-management` before retiring it.** This is the part that makes this change a removal-plus-migration rather than a straight deletion: `Reproducibility Through Seeding` is the *only* place in the live spec set that specifies single-agent experiment seeding (`multi-agent` covers only the multi-agent case), and reproducibility is load-bearing for the paired-seed statistics the whole research programme rests on. Two further requirements are part-live and are split scenario-by-scenario.
- **Correct a drift discovered during the migration**: `experiment-tracking`'s `Experiment Storage` scenario specifies a flat `experiments/{experiment_id}.json`, but the code writes the folder form `experiments/<id>/<id>.json` ([`storage.py:73,98`](../../../packages/quantum-nematode/quantumnematode/experiment/storage.py)). The migrated `Experiment Folder Structure` scenario is the accurate one; the stale scenario is replaced rather than duplicated.
- **Record the reversal** in the roadmap and in the ADR that originally chose NematodeBench over an external framework, so this reads as a decision with evidence rather than silent bit-rot.

## Capabilities

### New Capabilities

<!-- none -->

### Removed Capabilities

- `benchmark-management` — the submission workflow, categorization, leaderboard generation, quality gates, verification, contributor attribution, multi-session aggregation and public-documentation requirements are removed with their implementing code. The capability is **tombstoned rather than deleted** (see design.md D3): the spec file and directory are retained carrying a retirement notice and no live requirements.

### Modified Capabilities

- `experiment-tracking` — **gains** the three migrated requirements (`Reproducibility Through Seeding` verbatim; `Convergence-Derived Metrics`, being the two convergence scenarios of the former `Enhanced Metrics for Benchmarks`; `Experiment Folder Structure` and `Ad-hoc Experiment Storage`), and has its stale flat-file `Experiment Storage` scenario corrected to the folder layout the code implements.
- `cli-interface` — loses the `Benchmark Management CLI` requirement. `Experiment Tracking CLI Flags` is modified to drop the `--save-benchmark` and `--benchmark-notes` scenarios and their help-text bullets; note these flags **were never implemented** (`grep` over `scripts/*.py` returns zero hits), so this is drift correction as much as removal.
- `environment-simulation` — loses `Predator-Enabled Benchmark Categories` and the `Benchmark Category Name Verification` scenario of `Rendering Symbol Verification`; both specified `benchmark/categorization.py` behaviour. The rest of `Rendering Symbol Verification` is untouched.

## Impact

**Code (removed):** `packages/quantum-nematode/quantumnematode/benchmark/{leaderboard,categorization,validation}.py`, `experiment/{submission,validation}.py`, `BenchmarkMetadata` in `experiment/metadata.py`, `scripts/{benchmark_submit,evaluate_submission}.py`, and 3 test modules (~44 tests).

**Code (moved):** `benchmark/convergence.py` → `experiment/convergence.py`; `tests/.../benchmark/test_convergence.py` → `tests/.../experiment/test_convergence.py`. One import line changes (`experiment/tracker.py:9`).

**Data:** `benchmarks/` (6 JSON) and `artifacts/benchmarks/` (72 JSON, 3.8 MB) removed; `.gitattributes` LFS rule for `benchmarks/**/*.json` and the `!benchmarks/` negation in `.gitignore` become orphans and are dropped. The generic `artifacts/**/*.json` LFS rule stays.

**Docs:** `BENCHMARKS.md` and `docs/nematodebench/` deleted; NematodeBench sections stripped from `README.md`, `CONTRIBUTING.md`, `docs/experiments/README.md` and `AGENTS.md`; reversal recorded in `docs/roadmap.md` and `docs/STANDARDIZATION.md`.

**Config:** three orphaned `[tool.ruff.lint.per-file-ignores]` blocks in `pyproject.toml` and the `benchmark/leaderboard.py` entry in `codecov.yml`.

**Explicitly NOT touched** — three systems whose names collide but which are unrelated: `scripts/benchmarks/bench_evolution_smoke.py` (evolution wall-clock harness), `validation/datasets.py::ChemotaxisValidationBenchmark` (biological-literature validation, consumed by `tracker.py`), and `tests/.../e2e_benchmarks.json` (nightly regression ceilings).

## Breaking Changes

Yes. `quantumnematode.benchmark` is removed as an import path; `quantumnematode.experiment` no longer exports `NematodeBenchSubmission`, `SessionReference`, `AggregateMetrics`, `BenchmarkMetadata`, `validate_submission`, `MIN_SESSIONS_REQUIRED` or `MIN_RUNS_PER_SESSION`. Two CLI scripts are deleted. This is a research platform with no external API consumers, so no deprecation window is offered.

## Backward Compatibility

Historical experiment artifacts remain readable. 421 tracked JSONs under `artifacts/` carry a `"benchmark"` key (420 null, one populated) and all 421 carry `composite_benchmark_score`. `ExperimentMetadata.from_dict` ends in `cls(**data)` with Pydantic v2's default `extra="ignore"`, so the now-unknown `benchmark` key is dropped on load without error. Because that behaviour is inherited from a default rather than declared, a regression test pins it — see tasks §2.4. The `composite_benchmark_score` field is deliberately **not** renamed (design.md D2).
