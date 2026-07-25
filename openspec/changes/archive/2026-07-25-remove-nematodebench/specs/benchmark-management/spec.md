# benchmark-management Specification

## REMOVED Requirements

All requirements below are removed with their implementing code. The three requirements of this capability that specified genuinely-live behaviour — `Reproducibility Through Seeding`, `Enhanced Metrics for Benchmarks` (partly) and `Experiment Storage and Tracking` (partly) — are **not** listed here; they are migrated to `experiment-tracking` under this same change (see that delta, and design.md D4 for the per-requirement audit).

The shared reason for every removal below: NematodeBench's submission, validation, categorization, leaderboard and verification layers had no runtime consumer. The architecture-comparison protocol — the one thing the 2026-05-23 roadmap decision cited to justify keeping them — read `--track-experiment` output directly via `scripts/analysis/weight_search_architecture_ranking.py` and never invoked this pipeline through Phases 5 or 6.

The shared migration path: nothing. No replacement capability is offered. Per-experiment metric capture, which is what the project actually needed, is `experiment-tracking`; cross-architecture comparison, including its statistical layer, is `architecture-comparison-protocol`.

### Requirement: Benchmark Submission Workflow

**Reason**: The curated submission workflow (validation, quality checks, contributor attribution) was exercised six times, all by one contributor on 2025-12-28/29, and never again. `scripts/benchmark_submit.py` is deleted.

**Migration**: None. Use `scripts/run_simulation.py --track-experiment` for per-experiment capture.

### Requirement: Benchmark Categorization

**Reason**: `benchmark/categorization.py` derived category strings (`foraging_small/quantum` etc.) that only the leaderboard consumed. It also still emitted `static_maze/*` categories for an environment deleted from the code in `9a452fd5` (2026-01-30).

**Migration**: None. Experiment configs are identified by their config path and brain type in the tracked metadata.

### Requirement: Benchmark Leaderboard Generation

**Reason**: `benchmark/leaderboard.py` regenerated `README.md`, `BENCHMARKS.md` and `docs/nematodebench/LEADERBOARD.md` from the submission corpus. Eight of its twelve category slots were never populated at all, and the four that were covered 3 of the 27 brain architectures.

**Migration**: None. Results are reported per-logbook under `docs/experiments/logbooks/`.

### Requirement: Benchmark Quality Metrics

**Reason**: Success-rate thresholds, consistency checks and configuration validation were warnings emitted by `benchmark/validation.py` during submission only.

**Migration**: None. Statistical rigour for comparative claims is specified by `architecture-comparison-protocol` (paired-seed Wilcoxon, bootstrap CIs, BH-FDR).

### Requirement: Benchmark Comparison Tools

**Reason**: Category leaderboard queries, cross-architecture comparison and personal-best tracking were `scripts/benchmark_submit.py` subcommands over the submission corpus.

**Migration**: None. Cross-architecture comparison is `architecture-comparison-protocol`; per-experiment comparison is `experiment-tracking`'s `Compare Experiments` scenario.

### Requirement: Benchmark Verification

**Reason**: The maintainer reproduce-and-verify workflow was never run; no submission in the corpus carries a verification record.

**Migration**: None. Reproducibility is specified by the migrated `Reproducibility Through Seeding` requirement in `experiment-tracking`.

### Requirement: Benchmark CLI Tools

**Reason**: `scripts/benchmark_submit.py` is deleted.

**Migration**: None. `scripts/experiment_query.py` remains for querying tracked experiments.

### Requirement: Documentation Integration

**Reason**: Specified auto-regeneration of `README.md` and `BENCHMARKS.md` leaderboard sections. Both the generator and `BENCHMARKS.md` are deleted.

**Migration**: None.

### Requirement: Contributor Attribution

**Reason**: `BenchmarkMetadata` (contributor name, notes, verification status) existed only for submissions. The field was never populated by any production code path — `grep "benchmark="` over `packages/` and `scripts/` returns zero.

**Migration**: None. Authorship is recorded by git history and in logbooks.

### Requirement: Predator-Enabled Benchmark Categories

**Reason**: Duplicated in `environment-simulation` (removed there under this same change). Both specified `benchmark/categorization.py` behaviour.

**Migration**: None.

### Requirement: Predator Benchmark Metrics Tracking

**Reason**: Every scenario is scoped to "a benchmark submission for category `predator_*`". The live obligation to capture predator metrics in experiment metadata is already held by `experiment-tracking`'s `Predator Experiment Metadata Capture` and is unaffected.

**Migration**: `experiment-tracking` § Predator Experiment Metadata Capture (already live, no change needed).

### Requirement: NematodeBench Public Documentation

**Reason**: `BENCHMARKS.md` and `docs/nematodebench/` are deleted. The public launch these documents anticipated was deferred to Future Directions on 2026-05-23 and is now not planned at all.

**Migration**: None.

### Requirement: Benchmark Submission Evaluation Script

**Reason**: `scripts/evaluate_submission.py` is deleted with the submission schema it validated.

**Migration**: None.

### Requirement: NematodeBench Multi-Session Architecture

**Reason**: The multi-session aggregation model (10+ sessions, `StatValue` roll-up, `artifacts/benchmarks/` promotion, seed-uniqueness validation across sessions) dies with `experiment/submission.py`. Its one live scenario — that `--track-experiment` writes to `experiments/<id>/` with the config copied alongside — is migrated, not removed.

**Migration**: `experiment-tracking` § Experiment Folder Structure (migrated under this change). Multi-seed aggregation for comparative claims is `architecture-comparison-protocol`.
