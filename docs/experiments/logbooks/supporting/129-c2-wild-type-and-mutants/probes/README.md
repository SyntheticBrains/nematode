# Logbook 129's probes

Each `*.py.txt` is a probe script as it was run, beside the record it wrote. They are description on discarded seeds. They run from the repository root as `build/c2-run-mode/<name>.py`, alongside Logbooks 126–128's probes (see [127's README](../../127-c2-readability/probes/README.md) and [128's](../../128-c2-identifiability/probes/README.md)).

## The order, and the inputs to copy into `build/c2-run-mode/`

1. **`overnight`** starts from Logbook 128's `fit_mutants.json`, and writes `overnight.json`.
2. **`overnight_grade`** reads `overnight.json`.
3. **`speednoise`** reads `overnight.json`.
4. **`continuation`** reads `overnight.json`, and writes `continuation.json`.

`overnight.py` and `continuation.py` save their state every generation and resume from it.
