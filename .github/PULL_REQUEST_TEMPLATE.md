<!-- PR title: Conventional Commits prefix — feat:, fix:, docs:, chore:, refactor:, test: (add ! for a breaking change). -->

## Summary

<!-- What changes and why. For non-trivial work link the OpenSpec change (openspec/changes/<name>/); link the issue if there is one. -->

## Verification

<!-- What you ran and what it showed. Experimental results go in a numbered logbook (docs/experiments/README.md). -->

## Checklist

- [ ] `uv run pytest -m "not nightly"` passes
- [ ] `uv run pre-commit run -a` passes
- [ ] Docs updated (`docs/`, and `AGENTS.md` if commands or layout changed)
- [ ] `CHANGELOG.md` line under *Unreleased* for any user-facing change
- [ ] Disabled-by-default features are byte-identical no-ops when off
- [ ] Nightly benchmark ranges updated with evidence if training behaviour changed
