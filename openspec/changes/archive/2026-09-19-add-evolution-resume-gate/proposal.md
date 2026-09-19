# Gate checkpoint resume behind an explicit opt-in

## Why

Both evolution drivers resume from Python pickles. `pickle.load` executes whatever the file tells it
to, so `--resume <path>` on an untrusted file is arbitrary code execution. Issue #16 has tracked this
since 2025-12-13; a warning sits in the CLI help text and the vulnerability is unchanged, because a
warning is not a control.

The three fixes the issue proposes are not equal. `weights_only=True` **does not apply**: the payload
carries CMA-ES optimiser state and a numpy bit-generator state, not tensors. A structured format
would mean re-implementing that serialisation and rewriting a five-file checkpoint protocol whose
torn-save detection depends on deliberate write ordering — the RNG pickle is written last precisely so
a partial save is detectable. That is a large change against a correctness property, for a threat whose
actual shape is "someone hands you a checkpoint file".

The issue's third option addresses that shape directly: **require an explicit opt-in per invocation**.
The provenance of the file is the thing at risk, the program cannot check it, and only the caller
knows it.

## What changes

- **`--allow-unsafe-resume` on both drivers**, defaulting off. `--resume` without it refuses and exits
  1, **before anything is unpickled**.
- **The refusal is printed to stderr, not logged.** The gate runs before `configure_file_logging()`
  installs any handler, so a logged refusal reaches a `NullHandler` and the user gets a bare exit 1.
- **The spec's resume scenarios are modified** to include the flag, because they currently say
  `--resume <path>` alone resumes, which this makes false.
- **Every call site is updated**: the resume smoke test, `docs/usage.md`'s flag table and example, both
  drivers' module docstrings, the co-evolution campaign shell script's commented resume line, and the
  co-evolution "re-run with --resume" hint.
- **The `# noqa: S301` justifications name the gate** rather than asserting the file is simply trusted.

## What this does not change

The checkpoint **format**, which stays pickle. Issue #16 stays open on that question rather than being
closed by a partial fix, and the reasoning above is recorded on it.

Nor does it change resume's behaviour once permitted: same payload, same version checks, same
inheritance-mismatch rejection, same torn-save detection.

## Impact

- Affected specs: `evolution-framework` (one requirement modified, one added)
- Affected code: `scripts/run_evolution.py`, `scripts/run_coevolution.py`,
  `packages/quantum-nematode/quantumnematode/evolution/{loop,coevolution}.py` (comments only)
- Affected docs: `docs/usage.md`, `scripts/campaigns/phase5_m5_coevolution_full.sh`
- Affected tests: `test_smoke.py`'s resume test gains the flag; a new
  `evolution/test_resume_gate.py`
