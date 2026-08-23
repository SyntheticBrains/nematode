# Security Policy

Quantum Nematode is research software: a local simulation and analysis toolkit with no network service of its own. Its security-relevant surface is small, but it exists. This page says how to report a problem and what to be careful with.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability reporting — the **Report a vulnerability** button under the repository's [Security tab](https://github.com/SyntheticBrains/nematode/security) — which opens an advisory visible only to the maintainer. The project has a single maintainer, so responses are best-effort: you should hear back within about a week, and confirmed issues are fixed on `main` and in the next tagged release. Reporters are credited in the release notes unless they ask otherwise.

## Supported versions

Security fixes land on `main` and in the next tagged release. Only `main` and the latest tagged release receive fixes; there are no maintenance branches for earlier tags.

## What to be careful with

- **Checkpoints and substrates deserialise arbitrary objects.** Evolution checkpoints (`evolution_results/**/checkpoint.pkl`) are pickle files, and the transgenerational-memory substrate (`*.tei.pt`) is loaded with `torch.load(weights_only=False)`. Loading such a file from an untrusted source can execute code. Only load checkpoints and substrates you produced yourself. Brain weights (`weights/final.pt`) are loaded with `weights_only=True`.
- **Credentials live in `.env`.** IBM Quantum and Q-CTRL keys are read from the git-ignored `.env` (see `.env.template`). Never commit them; if a key is pushed by mistake, rotate it immediately.
- **Configs are trusted input.** Scenario and evolution YAML is parsed with `yaml.safe_load` and validated by Pydantic, so it cannot execute code, but the paths in it (weights to load, output directories) are used as given.
- **Dependencies.** The lock file is audited with `pip-audit` before each release and Dependabot proposes updates weekly. A vulnerable dependency that affects a code path this project exercises is in scope for the private channel above.
