# The neural readout's ceiling: the spike's code

Closed 2026-11-16. The code behind [Logbook 135](../../docs/experiments/logbooks/135-neural-readout-ceiling.md),
registered in
[launch record 135](../../docs/experiments/logbooks/supporting/135-neural-readout-ceiling/launch.md).
Every Python file is kept as `.py.txt`, so the repository's linters and test runner leave it alone.
They are identical to what ran, except that the end-of-file hook removed one trailing blank line
from `ceiling/pipeline.py.txt`. The registered reading used commit `869c5659`, before the rename.

## What is here

- `ceiling/`: the package. `config` holds every registered constant and literal reading;
  `stages` runs each stage; `scorer` is the only reader of S, logging every access to the launch
  record's `READS.md` before decoding it.
- `run.py.txt`: the stages, in order: setup, preflight, oracles, real, score.
- `summarise.py.txt`: writes Logbook 135's supporting data from a finished run directory. It reads
  no S.
- `tests/`: 176 tests on synthetic data. No test may open the real S.
- `IMPLEMENTATION_NOTES.md`: the literal readings of ambiguous rules, registered by the amendments.
- `DATA_CHECK.md`: the desk check of datasets that pair controlled inputs with neural activity and
  behaviour.

## Running it again

On a scratch branch, rename every `.py.txt` back to `.py` in place: `ceiling/config` finds the
repository and Creamer's matrices from its own path, so the folder must stay where it is. Commit
the rename before running `preflight`, `oracles` or `score`: the scorer refuses any read of S while
`ceiling/` has an uncommitted change. The commands are in `run.py`'s docstring. They expect the repository's `.venv` with the `cpu`, `torch`, `pixel`,
`analysis` and `qpu` extras, and write into a run directory (`runs/` is gitignored). Run the tests
with `OMP_NUM_THREADS=1 NC_PACKAGE_TESTS=1 python -m pytest -q` from the folder.

The scorer refuses a second registered reading of S unless a committed amendment names it. A new
reading needs its own registration.

## Known limits, if the code is reused

Found in review after the run. None was exercised by run 135. Every checkpoint line parses, and m
is finite on every split and oracle cell. The code is left as it ran.

- The cost projection does not count the one-time λ-grid extensions (Logbook 135, Corrections).
- The branch mapper reads an undefined m (every reference unreadable) as "fails" in R-B, R-C and
  the denoising reading, rather than as unreadable.
- A record appended after a line torn by a stop joins that line and is lost on the next load.
- Task identities do not digest their data set, so a rewritten `datasets/*.npz` can reuse stale
  checkpoints on resume.
- The once-only rule reads the working copy of `READS.md`, with no lock and no check against its
  committed history.
- `summarise.py.txt` assumes a complete run with no cuts and every arm readable.
