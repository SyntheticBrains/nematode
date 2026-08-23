# benchmark-management Specification

## Purpose

This capability specified NematodeBench: the curated benchmark submission workflow, quality validation, categorisation, leaderboard generation, verification, contributor attribution and multi-session aggregation. It was **removed on 2026-07-25** by [`remove-nematodebench`](../../changes/archive/2026-07-25-remove-nematodebench/).

## Status

> **Retired — capability removed.** No implementing code remains. The requirements this capability held are no longer live obligations, and nothing should be built against them. The full 534-line original text is preserved in git history and in the archived change's `specs/benchmark-management/spec.md` delta, which lists every removed requirement with its reason.
>
> The file and directory are retained rather than deleted because no capability spec has ever been deleted in this repository, and a reader arriving from an old link or an archived change needs to find out *where the live parts went* rather than a 404. Do not reintroduce this capability without a new proposal.

**Why it was removed.** The 2026-05-23 roadmap rewrite demoted NematodeBench from a Phase 7 public-launch deliverable to internal tooling, justified by its usefulness to the architecture-comparison protocol. Two phases of evidence contradicted that: the protocol read per-seed `--track-experiment` output directly via `scripts/analysis/weight_search_architecture_ranking.py` and never invoked the submission pipeline. The corpus stopped at six submissions from 2025-12-28/29 covering 3 of the eventual 27 architectures.

## What survives, and where it went

Three requirements of this capability specified genuinely-live behaviour and were **migrated to `experiment-tracking`**, not removed:

| Was | Now |
|---|---|
| `Reproducibility Through Seeding` | `experiment-tracking` § Reproducibility Through Seeding — verbatim. The only live-spec coverage of single-agent experiment seeding; `multi-agent` § Reproducible Seeding covers only the multi-agent case. |
| `Enhanced Metrics for Benchmarks` (learning-speed + stability scenarios) | `experiment-tracking` § Convergence-Derived Metrics. The third scenario, `Statistical Aggregation`, was submission-scoped and was removed. |
| `Experiment Storage and Tracking` (folder-structure + ad-hoc scenarios) | `experiment-tracking` § Experiment Storage and Retrieval. The third scenario, `Benchmark Artifact Storage`, described `artifacts/benchmarks/` and was removed. |

Also surviving in code: the convergence detector and composite score, moved to [`experiment/convergence.py`](../../../packages/quantum-nematode/quantumnematode/experiment/convergence.py) and governed by `architecture-comparison-protocol` (ranked-metric plateau detection) and `experiment-tracking`. The `composite_benchmark_score` field keeps its name — 421 historical artifacts carry the key and renaming it would read as `null` without raising. The 72 session experiments behind the six submissions were migrated into `artifacts/experiments/`; no experimental data was destroyed.

**Three unrelated systems in this repository contain the word "benchmark" and were untouched** — a future reader grepping for it will find them:

- `scripts/benchmarks/bench_evolution_smoke.py` — a wall-clock performance harness for the evolution fitness-eval path.
- `validation/datasets.py::ChemotaxisValidationBenchmark` — real-worm biological-literature validation, consumed by `experiment/tracker.py`.
- `tests/.../e2e_benchmarks.json` — success-rate ceilings for the nightly regression suite.

## Requirements

None. This capability has no live requirements.

> **Expected validator consequence.** `openspec validate --specs` reports this spec
> as failing (`Spec must have at least one requirement`), so the repository-wide
> total is **46 passed / 1 failed**, not 47/47 — and this spec is the *only*
> failure, under both plain and `--strict` validation. That is the intended cost of
> the retention decision above, not drift: the validator (as of openspec 1.10) has
> no notion of a retired capability and offers no ignore or exclusion mechanism,
> and the alternatives are deleting the file (losing the redirect for anyone
> arriving from an old link) or inventing a placeholder requirement (a fake
> obligation in a spec that has none). Neither is better. Do not "fix" this by
> adding a requirement. If a future openspec release adds a retired/ignored spec
> status, adopt it here and this note can go.
