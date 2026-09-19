# Tasks

## 1. The gate

- [x] 1.1 `--allow-unsafe-resume` on `scripts/run_evolution.py` and `scripts/run_coevolution.py`,
  `action="store_true"`, defaulting off, with help text naming the risk rather than only the flag.
- [x] 1.2 `--resume` without it refuses and returns 1, **after argument parsing and before any file
  access or loop construction**, so a missing and a hostile checkpoint are indistinguishable in the
  refusal.
- [x] 1.3 The refusal is **printed to stderr**, not logged: the gate runs before
  `configure_file_logging()` installs a handler, so a logged message would be silent in production.
- [x] 1.4 `--resume`'s own help text states that the flag is required.

## 2. The spec

- [x] 2.1 **MODIFIED** `Evolution Loop Checkpoint and Resume`: the resume scenario carries the flag,
  since `--resume` alone no longer resumes.
- [x] 2.2 **ADDED** a requirement for the gate: default off, refuse before unpickling, refusal visible
  with no configured log handler, and the format left unchanged.

## 3. Call sites

- [x] 3.1 `test_smoke.py`'s resume test passes the flag — it writes the checkpoint it resumes, so it is
  trusted by construction, which is exactly the intended use.
- [x] 3.2 `docs/usage.md`: the flag table gains `--allow-unsafe-resume`, and the resume example uses it.
- [x] 3.3 Both drivers' module-docstring usage examples.
- [x] 3.4 `scripts/campaigns/phase5_m5_coevolution_full.sh`'s commented resume line.
- [x] 3.5 `run_coevolution.py`'s "re-run with --resume ..." hint names the flag, so the message it
  prints is a command that actually works.
- [x] 3.6 Grep the repository for remaining `--resume` invocations without the flag, excluding the
  gate's own negative tests.

## 4. Tests

- [x] 4.1 Refusal without the flag, on both drivers, with a payload that **executes on load** rather
  than one that fails to pickle — the first draft's payload raised at dump time and tested nothing.
- [x] 4.2 The refusal fires with no checkpoint present at all, proving it precedes file access.
- [x] 4.3 `_run` strips `PYTEST_CURRENT_TEST` and `TESTING` so the tests exercise the production
  logging path; one test asserts the refusal is visible with no handler configured.
- [x] 4.4 The refusal names the risk (pickle, arbitrary code), not just the flag.
- [x] 4.5 Both drivers' `--help` list the flag.
- [x] 4.6 The full smoke suite passes, including the resume test that this change initially broke.

## 5. Bookkeeping

- [x] 5.1 `# noqa: S301` justifications name the gate instead of asserting the file is trusted.
- [x] 5.2 `CHANGELOG.md` entry.
- [x] 5.3 Issue #16 records why the format question stays open rather than being closed by this.
