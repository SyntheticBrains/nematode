# GitHub Actions Workflows

This directory contains GitHub Actions workflows for continuous integration and code quality checks.

## Workflows

### `pre-commit.yml`

Runs all pre-commit hooks on every push and pull request to `main`.

**What it does:**

- ✅ Code formatting with ruff
- ✅ Static type checking with pyright
- ✅ YAML and TOML validation
- ✅ Checks for large files
- ✅ Ensures files end with newline
- ✅ Verifies no uncommitted changes after hooks run

**Note:** Skips the `pytest` hook as tests run in a separate workflow.

______________________________________________________________________

### `tests.yml`

Runs the test suite on Python 3.13, the only version the main project supports (`requires-python = ">=3.13,<3.14"`), and tests the `wormbench` package alone on 3.10 and 3.13 (it supports Python 3.10 or later).

**What it does:**

- ✅ Runs pytest across five parallel `pytest-split` shards on Python 3.13, aggregated behind a single `Tests` check
- ✅ Uploads a coverage report per shard to Codecov, which merges them by commit (optional)
- ✅ Pins BLAS/OpenMP to one thread per xdist worker so the runner is not oversubscribed
- ✅ Uploads each shard's measured test durations as the artifact `test-durations-shard-<n>` (14 days), from which `.test_durations` is refreshed
- ✅ **`wormbench alone`**: builds the `wormbench` wheel from its sdist, installs it alone into a fresh environment (Python 3.10 at the declared minimums, numpy 1.24.4 and scipy 1.10.1, and Python 3.13 at the latest), checks that it imports from the wheel without `quantumnematode`, and runs the package's tests. The required `Tests` check needs it to pass with the shards
- ✅ **Skips the shard matrix on docs-only PRs**: a `changes` job classifies the PR's file list (via the GitHub API) against a docs allowlist (`docs/`, `openspec/`, `.claude/`, markdown, licence/lint metadata); known code trees (`artifacts/`, `packages/`, `configs/`, `data/`, `scripts/`, `.github/`), unrecognised paths, and empty file lists all fail open to a full run. The required `Tests` check still reports — it passes explicitly when shards were deliberately skipped. Pushes to `main` are never filtered (the coverage record stays unbroken), and Codecov posts no status/comment on docs-only PRs (neither codecov context is a required check).

______________________________________________________________________

### `nightly-tests.yml`

Runs the nightly end-to-end regression suite on a schedule (`cron: '0 3 * * *'`),
on Python 3.13.

**What it does:**

- ✅ Runs the `nightly`-marked tests, which are excluded from every other workflow
- ✅ Restores Git LFS objects from cache so the fixture data is available
- ✅ Surfaces its status through the **Nightly Tests** badge in the root `README.md`

______________________________________________________________________

### `release-wormbench.yml`

Publishes the `wormbench` package by PyPI's trusted publishing, so no token is stored. See CONTRIBUTING.md § Releasing wormbench for the release steps.

**What it does:**

- ✅ A manual run on `main` uploads to TestPyPI; pushing a `wormbench-vX.Y.Z` tag on a commit in `main` uploads to PyPI. It refuses a tag that does not name the package's version, or that is not on `main`
- ✅ Builds the wheel from its sdist and tests it alone on Python 3.10 at the declared minimums before any upload
- ✅ Uploads from the `testpypi` or `pypi` environment, each of which can require a reviewer's approval
- ✅ Pins every action to a commit, since the upload jobs hold an OIDC token

______________________________________________________________________

### `literature-watch.yml`

Builds a weekly digest of new research relevant to the project, every Monday at 08:00 UTC (`cron: '0 8 * * 1'`), or on a manual run.

**What it does:**

- ✅ Collects the ids earlier digests reported from the `literature-watch` issues, so a paper is reported once; no state is committed
- ✅ Runs `scripts/literature_watch.py`, which fetches candidates and scores and digests them with the Anthropic API (`ANTHROPIC_API_KEY` secret)
- ✅ Opens the digest as an issue labelled `literature-watch`
- ✅ A manual run takes a window end (`until`) and a `dry_run` that lists candidates only, with no model calls and no issue

______________________________________________________________________

## Running Locally

Before pushing, run pre-commit hooks locally:

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files

# Run tests
uv run pytest packages/quantum-nematode/tests/
```

## Workflow Configuration

All workflows use:

- **Python version**: 3.13, except the `wormbench alone` jobs (3.10 and 3.13) and the release's wheel test (3.10)
- **Package manager**: `uv`, with caching enabled except in the release workflow
- **Pre-commit**: v3.0.1 action

## Skipping CI

To skip CI on a commit:

```bash
git commit -m "docs: update readme [skip ci]"
```

## Troubleshooting

If pre-commit checks fail:

1. Run `pre-commit run --all-files` locally
2. Fix any issues reported
3. Commit the changes
4. Push again

If tests fail:

1. Check the workflow logs in GitHub Actions
2. Download test artifacts for detailed error information
3. Run tests locally: `uv run pytest packages/quantum-nematode/tests/ -v`
