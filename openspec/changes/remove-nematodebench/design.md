## Context

NematodeBench spans five layers: a submission schema (`experiment/submission.py`), a validator (`experiment/validation.py`, `benchmark/validation.py`), a category deriver (`benchmark/categorization.py`), a leaderboard/markdown generator (`benchmark/leaderboard.py`), and two CLI entry points. Sitting inside the same Python package — but belonging to none of those layers — is `benchmark/convergence.py`, the detector that turns a run's episode sequence into `post_convergence_success_rate` and `composite_benchmark_score`.

That last module is the reason this cannot be a `git rm -r`. It is the ranked-metric producer for `architecture-comparison-protocol`, it was materially rewritten on 2026-06-21 for the T7 continuous band, and every `--track-experiment` invocation calls it via `experiment/tracker.py`. The other 2.4k lines have no runtime consumer at all.

A second complication surfaced while drafting the spec delta: `benchmark-management` accumulated requirements that were never about benchmarks. It is the only live spec covering single-agent experiment seeding, and it carries the accurate description of the on-disk experiment layout — more accurate than `experiment-tracking`'s own. Retiring it naively would silently drop live obligations.

## Goals / Non-Goals

**Goals:**

- Delete the submission/leaderboard system and its data with no residual references.
- Preserve the convergence detector's behaviour byte-for-byte and its artifact-compatibility exactly.
- Migrate every genuinely-live requirement out of `benchmark-management` before retiring it, with an explicit audit rather than a judgement call per requirement.
- Leave a record of *why* the 2026-05-23 "demote rather than delete" decision was reversed.

**Non-Goals:**

- Not a rewrite or re-tune of convergence detection — the file moves unchanged apart from its module docstring.
- Not a rename of `composite_benchmark_score` (D2).
- Not a purge of LFS history (`git lfs prune` is out of scope and unsafe here).
- Not a change to the nightly e2e regression ceilings, which are a separate system that shares the word "benchmark".

## Decisions

### D1 — `convergence.py` lands in `experiment/`, not a new `analysis/` or `metrics/` package

The obvious move is a neutral `quantumnematode/analysis/` package. It is wrong here: `scripts/analysis/` already exists (9 harnesses), and `tests/quantumnematode_tests/analysis/` already exists and — despite its location under the package's test tree — `sys.path`-injects `scripts/analysis/` and tests *those* harnesses by bare module name. Adding `quantumnematode/analysis/` would give that test directory two meanings and put a package unit test among four script tests. `metrics/` is taken twice over (`agent/metrics.py`, `plasticity/metrics.py`).

`experiment/` is where the consumer and the output model already live: `experiment/tracker.py` is the only caller, and the 13 fields it writes are on `experiment/metadata.py::ResultsMetadata`. The move adds zero new import edges — `tracker.py` already imports `report.dtypes`, which is `convergence.py`'s only non-stdlib dependency — and creates no cycle, since `report/csv_export.py` imports `experiment.metadata` only under `TYPE_CHECKING`.

*Rejected:* `report/convergence.py`. `report/` is the output/serialisation layer (plots, CSV, summaries); convergence is an analysis step that *feeds* metadata which `report` later serialises. Putting it there inverts the layering.

`analyze_convergence` is deliberately **not** re-exported from `experiment/__init__.py`. It has exactly one call site. A package-level re-export is precisely the habit that let `benchmark/__init__.py` accumulate 15 of them.

### D2 — `composite_benchmark_score` keeps its name

Tempting to rename now that "benchmark" no longer names a subsystem. Four reasons not to:

1. **421 tracked artifacts on disk carry the key.** A rename means every historical read populates the new field from a missing key with its `None` default and discards the old value — wrong numbers, no exception, no warning.
2. **The failure mode is silent by construction.** `from_dict` is `cls(**data)` and no model in `metadata.py` declares `model_config`, so Pydantic v2's `extra="ignore"` applies.
3. **`report/csv_export.py:415` emits the string as a literal CSV row label**, so exported CSVs and the logbook appendices built from them use it too.
4. **The name never meant NematodeBench.** "Benchmark" here denotes the composite scoring formula (`brain-architecture/spec.md:337` calls it "composite benchmark score"), whose multi-objective weighting was extended by the thermotaxis work independently of the submission system.

Only the two stale comments at `metadata.py:410` ("added for benchmark v2") and `:419` ("added for NematodeBench format") change. If a reviewer still wants the rename, the correct implementation is a compatibility shim in `from_dict`, not a field rename — and it is not worth it for a word that was never load-bearing.

### D3 — `benchmark-management` is tombstoned, not deleted

`git log --diff-filter=D -- 'openspec/specs/*/spec.md'` is empty: no capability spec has ever been deleted in this repo. The nearest precedent is `phase6-tracking`, retained with a "**Frozen — historical record.** … no longer impose live obligations" banner.

But that froze a *completed* capability whose requirements still describe reality. This is a *retired* one whose implementing code will not exist, and leaving 14 live-looking `SHALL` statements pointed at deleted modules is worse than deleting the file. So: keep the file and directory, replace the body with a short retirement notice — Purpose (past tense), Status, an explicit "what survives and where it went" section so a future reader does not re-derive the split, and `## Requirements` reading `None`. The full 534-line original is preserved in git history and in this change's archived delta.

The "what survives" section is load-bearing documentation, not courtesy: three unrelated systems in this repo contain the word "benchmark" (`bench_evolution_smoke.py`, `ChemotaxisValidationBenchmark`, `e2e_benchmarks.json`) and a future agent grepping for it will find them.

### D4 — the live-requirement audit, and where each one goes

Every one of the 17 requirements was read rather than assumed removable. Fourteen are submission-scoped and are removed. Three are not:

| Requirement | Verdict | Destination |
|---|---|---|
| `Reproducibility Through Seeding` (4 scenarios) | **Fully live.** Automatic seed generation, env/brain determinism, per-run seed tracking. Nothing to do with submissions. The only live-spec coverage of single-agent seeding — `multi-agent/spec.md:216` covers only the multi-agent case. Dropping it would leave the paired-seed statistical protocol resting on an unspecified foundation. | → `experiment-tracking`, verbatim |
| `Enhanced Metrics for Benchmarks` (3 scenarios) | **Part-live.** `Learning Speed Calculation` and `Stability Metric Calculation` specify `calculate_learning_speed` / `calculate_stability`, which survive in `experiment/convergence.py`. `Statistical Aggregation` is explicitly "for NematodeBench submission" and dies with `AggregateMetrics`. | → `experiment-tracking` as `Convergence-Derived Metrics` (2 scenarios); third scenario removed |
| `Experiment Storage and Tracking` (3 scenarios) | **Part-live.** `Experiment Folder Structure` and `Ad-hoc Experiment Storage` describe `experiments/` and `artifacts/experiments/`, both live. `Benchmark Artifact Storage` describes `artifacts/benchmarks/`, deleted here. | → `experiment-tracking` (2 scenarios); third removed |

Two requirements deserve a note on why they *are* removable despite looking live. `Predator Benchmark Metrics Tracking` reads like live metric capture, but every scenario is scoped to "a benchmark submission for category `predator_*`"; the equivalent live obligation already exists as `experiment-tracking`'s `Predator Experiment Metadata Capture`. `Benchmark Quality Metrics` specifies validator warnings emitted only by `benchmark/validation.py`.

### D5 — the migration corrects a spec/code drift rather than duplicating it

`experiment-tracking`'s existing `Experiment Storage` scenario specifies `experiments/{experiment_id}.json` (flat). The code writes `experiments/<id>/<id>.json` and discovers experiments by scanning subdirectories (`storage.py:73,98`), which is what the migrated `Experiment Folder Structure` scenario describes. Rather than land two contradictory scenarios in one capability, the delta **replaces** the stale scenario. This is in-scope because the migration is what surfaced it; leaving a known-wrong scenario in place while editing the requirement around it would be worse.

## Risks / Trade-offs

- **Silent artifact-read regression** — the one real risk, mitigated by D2 plus an explicit round-trip test (tasks §2.4) against the shape of the single artifact on disk with a populated `benchmark` object. Without the test, a later `extra="forbid"` would turn a passing load into a `ValidationError` with nothing to catch it.
- **Losing the option to revive NematodeBench.** Accepted. A revival would be a from-scratch build against the current architecture set, not a reactivation of a corpus whose six entries predate 21 of the 27 brains. The full spec and code remain in git history.
- **Reversing a recorded roadmap decision.** Mitigated by making the reversal explicit in the roadmap's own reversal table (as a new row, not an edit to the existing one) and by superseding rather than deleting the ADR — so the original reasoning stays legible next to the evidence that overturned it.
- **Coverage-gate noise.** Deleting three modules and one `codecov.yml` ignore on a ~2.4k-line-deletion PR can make the patch check behave oddly even as project coverage rises. Read the Codecov comment; do not assume.

## Migration Plan

Seven commits, spec-first, each independently green under `uv run pytest -m "not nightly"` and `uv run pre-commit run -a`. Order is load-bearing in two places: the code deletion (§2) precedes the `convergence.py` move (§3) so the move is a clean two-file rename against an already-pruned package; and the LFS data deletion (§4) precedes dropping the `.gitattributes` rule (§5) so nothing can be re-added as a plain blob in between. Full sequence in tasks.md.

## Open Questions

None. The convergence-detector destination, the field-name question and the spec-retirement form are all settled above; the live-requirement audit is complete and enumerated in D4.
