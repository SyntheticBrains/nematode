# Tasks

## 1. Spec proposal

- [x] 1.1 Author `proposal.md`, `design.md`, `tasks.md` for `remove-nematodebench`.
- [x] 1.2 Author the four spec deltas: `benchmark-management` (REMOVED × 14), `experiment-tracking` (ADDED × 3 migrated + MODIFIED storage scenario), `cli-interface` (REMOVED × 1 + MODIFIED × 1), `environment-simulation` (REMOVED × 1 + MODIFIED × 1).
- [x] 1.3 Audit all 17 `benchmark-management` requirements individually for live content (design.md D4) — do not assume removability from the capability name.

## 2. Remove the submission and leaderboard code

- [ ] 2.1 Delete `benchmark/{leaderboard,categorization,validation}.py`; reduce `benchmark/__init__.py` to the convergence re-exports only (transitional — the file is deleted in §3).
- [ ] 2.2 Delete `experiment/submission.py` and `experiment/validation.py`; drop their imports and the 7 corresponding `__all__` entries from `experiment/__init__.py`.
- [ ] 2.3 Delete `class BenchmarkMetadata` from `experiment/metadata.py` (:475-489) with its `ExperimentMetadata.benchmark` field (:607) and docstring line (:588). Reword the stale comments at :410 and :419 — comment text only, no field, attribute-docstring or CSV-label changes (design.md D2).
- [ ] 2.4 **Add the artifact round-trip regression test**: assert `load_experiment` succeeds on a fixture carrying a populated `"benchmark": {...}` plus `composite_benchmark_score`, and that the score round-trips to the expected float. Build the fixture from the shape of `artifacts/experiments/20251219_105232/20251219_105232.json` (inline the dict — do not read an LFS file from a unit test). This pins the Pydantic `extra="ignore"` behaviour the whole change rests on.
- [ ] 2.5 Delete `scripts/benchmark_submit.py` and `scripts/evaluate_submission.py`. In `scripts/run_simulation.py`, delete the "To submit as benchmark" hint (`:1172`) and reword the stale comment (`:1146`). **Leave `:1205` alone** — "Use dynamic literature source from benchmark" refers to `ChemotaxisValidationBenchmark`, the real-worm biological validation, which is a different system.
- [ ] 2.6 Delete `tests/.../benchmark/{test_categorization,test_validation}.py` and `tests/.../experiment/test_validation.py`; prune the `BenchmarkMetadata` import, `TestBenchmarkMetadata` class and `benchmark=` kwarg from `tests/.../experiment/test_metadata.py`.
- [ ] 2.7 Drop the three orphaned `[tool.ruff.lint.per-file-ignores]` blocks from `pyproject.toml` and the `benchmark/leaderboard.py` entry from `codecov.yml`.
- [ ] 2.8 Verify: `uv run pytest -m "not nightly"`, `uv run pre-commit run -a` (pyright catches dangling imports), and a grep for the 7 removed symbols returning zero hits in `packages/` + `scripts/`.

## 3. Relocate the convergence detector

- [ ] 3.1 `git mv benchmark/convergence.py experiment/convergence.py`; `git mv tests/.../benchmark/test_convergence.py tests/.../experiment/test_convergence.py`; `git rm` both now-empty `__init__.py` files so the `benchmark/` package disappears.
- [ ] 3.2 Repoint the import in `experiment/tracker.py:9` and in the moved test. Let ruff-isort re-sort — do not hand-place.
- [ ] 3.3 Drop "for benchmark evaluation" from the module docstring. Change no symbol names.
- [ ] 3.4 Update the two live `convergence.py` path references in `openspec/specs/architecture-comparison-protocol/spec.md:52` (link target + display text). Do **not** touch the identical paths under `openspec/changes/archive/**` — archived changes correctly record the path as it was.
- [ ] 3.5 Verify: the 17 convergence tests pass at their new path; a repo-wide grep for `quantumnematode.benchmark` / `quantumnematode/benchmark` returns zero outside `openspec/changes/archive/`.

## 4. Preserve the experiment data, remove the submission wrapper

The 72 experiment JSONs under `artifacts/benchmarks/` have **no duplicate anywhere** — their IDs and `artifacts/experiments/`'s 12 IDs are disjoint. They are primary experiment records, not submission tooling, so they are migrated rather than deleted (design.md D6). Only the six submission manifests are removed.

- [ ] 4.1 Migrate each `artifacts/benchmarks/<submission_id>/<experiment_id>.json` to `artifacts/experiments/<experiment_id>/<experiment_id>.json`, copying that submission's `config.yml` alongside each — the live folder layout specified by the `Experiment Folder Structure` scenario being migrated in §7.1. Use `git mv` where possible so LFS pointers move rather than re-add.
- [ ] 4.2 Verify the migration before deleting anything: 72 new experiment folders exist, each contains exactly one JSON plus one YAML, every JSON parses, and `artifacts/experiments/` has gone 12 → 84 entries with no ID collisions.
- [ ] 4.3 `git rm -r benchmarks/` (the 6 submission manifests) and remove the now-empty `artifacts/benchmarks/` tree — before the `.gitattributes` edit, so nothing can be re-added as a plain blob.
- [ ] 4.4 Drop the `benchmarks/**/*.json` LFS rule from `.gitattributes` and the `!benchmarks/` negation from `.gitignore`. Keep the generic `artifacts/**/*.json` rule — it now covers the migrated files — and the `.bench_evolution_tmp/` entry, a different system.
- [ ] 4.5 Verify: `git lfs status` shows the migrated JSONs still LFS-tracked (not converted to plain blobs); `check-added-large-files` green. Do **not** run `git lfs prune`.

## 5. Remove the documentation

- [ ] 5.1 Delete `BENCHMARKS.md` and `docs/nematodebench/` (5 files).
- [ ] 5.2 Strip the `## 🏆 Top Benchmarks` section from `README.md` (the block the deleted `leaderboard.py::update_readme()` used to regenerate — it anchors on literal strings, not HTML markers, so nothing else keys off it).
- [ ] 5.3 Strip the NematodeBench cluster from `CONTRIBUTING.md` (submission/leaderboard subsections, the `benchmarks/` table row, the scripts bullet). Leave the `#### Nightly E2E Tests` section and its `e2e_benchmarks.json` reference — different system.
- [ ] 5.4 Remove the `benchmark_submit.py` line from the workflow diagram in `docs/experiments/README.md`.
- [ ] 5.5 Update `AGENTS.md`: `experiment/` description now names convergence analysis; drop `benchmark_submit.py` from the scripts list; delete the `benchmarks/` directory entry.
- [ ] 5.6 Mark the orphaned action item in [Logbook 009](../../../docs/experiments/logbooks/009-temporal-sensing-evaluation.md) (`- [ ] Formal NematodeBench submission for lstmppo configs`) as obsolete **in place** — strike it and note the removal date. Logbooks are immutable records of what was true, but an open checkbox is a forward commitment, not a record, and this one now points at a deleted system. Precedent: `phase6-tracking`'s T8 section became a pointer stub with no checkboxes when its scope moved. Do not delete the line and do not touch anything else in the logbook.
- [ ] 5.7 Verify: grep for `BENCHMARKS.md`, `docs/nematodebench`, `benchmark_submit`, `evaluate_submission` across `*.md` returns zero outside the archive. No link-checking hook exists, so this grep is the only guard against dead links.

## 6. Record the reversal

- [ ] 6.1 `docs/roadmap.md` § NematodeBench (Future Directions) — retitle to `(removed)`, state what was removed and what survives, and give the evidence: the protocol used `weight_search_architecture_ranking.py` over tracked experiments and never invoked the submission pipeline.
- [ ] 6.2 `docs/roadmap.md` § NematodeBench public launch — the internal tooling is now removed too; a future launch is a from-scratch build, not a reactivation. Keep the standing "benchmarks crystallise mature communities" rationale.
- [ ] 6.3 `docs/roadmap.md` § Scoping Changes from v3 — **add a new row** recording the second step down. Do not edit the existing row; that table is a record of what changed when.
- [ ] 6.4 `docs/roadmap.md` principle 4 ("Demote rather than delete") — append the corollary: where a demoted component accrues no use across a full phase, deletion follows and is recorded.
- [ ] 6.5 `docs/STANDARDIZATION.md` § Benchmarking — supersede in place. Keep the original Decision/Rationale verbatim as the historical record; add `**Status**: Superseded.` and a `**Why superseded**` block; update the summary-table row.

## 7. Apply the specs and archive

- [ ] 7.1 Apply the `experiment-tracking` delta: add the three migrated requirements, replace the stale flat-file `Experiment Storage` scenario with the folder-layout one (design.md D5).
- [ ] 7.2 Apply the `cli-interface` and `environment-simulation` deltas.
- [ ] 7.3 Replace `openspec/specs/benchmark-management/spec.md` with the tombstone (design.md D3) — Purpose in past tense, Status notice, "what survives and where", `## Requirements` = None.
- [ ] 7.4 `openspec/config.yaml` needs no edit — verified to contain zero benchmark references. Recorded so a reviewer does not re-check.
- [ ] 7.5 Tick this checklist; `git mv openspec/changes/remove-nematodebench openspec/changes/archive/2026-07-25-remove-nematodebench`.
- [ ] 7.6 Final verification: full `uv run pytest -m "not nightly"` (against the recorded baseline — see below), `uv run pre-commit run -a`, an end-to-end `--track-experiment` smoke run confirming `composite_benchmark_score` is still written, and a repo-wide case-insensitive `nematodebench` grep. Expected surviving hits, exactly four sites: the roadmap reversal prose, the `STANDARDIZATION.md` supersession, the `benchmark-management` tombstone, and the struck-through Logbook 009 line. Anything else is a miss.

## Verification baseline

The dev laptop cannot run the pinned numpy/torch (`numpy>=2.2.4`, `torch>=2.7.0`); it has numpy 1.26.4 / torch 2.2.0 installed. CI runs the pinned versions. Two gates are therefore **red before this change starts** and must be compared against baseline, not against zero:

- `pytest -m "not nightly"`: **3 failed, 3978 passed** — `test_qef.py::test_cry_crz_deterministic` and two `test_modules.py` bounds tests, all float32-vs-float64 boundary precision (e.g. float32 `-1.5707964` failing a `>= -π/2` float64 bound by ~1e-7).
- `pyright`: **61 errors** across 22 files — 37 `reportArgumentType` (numpy 1.x scalar typing), 14 `reportMissingImports` (`sklearn`, `qiskit_ibm_runtime` — optional extras not installed locally; CI installs all extras).

The bar for this change is that neither count increases and no new file appears in the pyright set.
