# GitHub Actions Workflows

This directory contains GitHub Actions workflows for continuous integration and code quality checks.

## Workflows

### `pre-commit.yml`

Runs all pre-commit hooks on every push and pull request to `main` and `develop` branches.

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

Runs the test suite on Python 3.13 (the only version the project supports — `requires-python = ">=3.13,<3.14"`).

**What it does:**

- ✅ Runs pytest across five parallel `pytest-split` shards on Python 3.13, aggregated behind a single `Tests` check
- ✅ Uploads a coverage report per shard to Codecov, which merges them by commit (optional)
- ✅ Pins BLAS/OpenMP to one thread per xdist worker so the runner is not oversubscribed

______________________________________________________________________

### `nightly-tests.yml`

Runs the nightly end-to-end regression suite on a schedule (`cron: '0 3 * * *'`),
on Python 3.13.

**What it does:**

- ✅ Runs the `nightly`-marked tests, which are excluded from every other workflow
- ✅ Restores Git LFS objects from cache so the fixture data is available
- ✅ Surfaces its status through the **Nightly Tests** badge in the root `README.md`

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

- **Python version**: 3.13
- **Package manager**: `uv` with caching enabled
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
