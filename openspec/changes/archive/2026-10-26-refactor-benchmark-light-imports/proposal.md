## Why

Importing the benchmark's grader (`quantumnematode.benchmark.grading`) loads 2,511 modules, `torch` and `qiskit` among them, and takes 1.5 s. The benchmark's own modules need 0.08 s. Two imports cause it:

- `report/dtypes.py`, home of `BehaviourStep`, imports the brain's base class;
- `validation/__init__.py` eagerly imports `adaptive_sensor`, the one submodule that loads the agent, the environment and `torch`.

`torch` is an optional extra, so a lab that installs the package to grade its own tracks (B.5, [Logbook 114](../../../docs/experiments/logbooks/114-track-import.md)) needs PyTorch it never uses. B.5's spec review scoped the "independent of the simulator" requirement to run time for this reason, and recorded the decoupling as a later refactor (the maintainer's decision of 2026-10-26).

## What Changes

- **`BehaviourStep` moves** to `quantumnematode/report/behaviour.py`, a module with no package imports. `report.dtypes` re-exports it, so every importer is unchanged and it is the same class.
- **`validation/__init__.py` imports `adaptive_sensor` lazily**, through a module-level `__getattr__`. Its four names (`StepInputResult`, `run_step_input`, `weber_invariance_peaks`, `weber_spread`) still import from `quantumnematode.validation`, and `__all__` is unchanged. Importing `roaming_dwelling`, `chemotaxis` or `datasets` no longer loads the agent.
- **`benchmark.track_import`** imports `BehaviourStep` from the new module.
- **The `track-import` spec** adds module-load independence: importing the grader and the importer loads no `torch` and no brain, agent or environment module.

## Capabilities

### Modified Capabilities

- `track-import`: the independence requirement covers what is imported, not only what runs.

## Impact

- **Code:**
  - `report/behaviour.py` (new);
  - `report/dtypes.py`, `validation/__init__.py` and `benchmark/track_import.py`;
  - a test.
- **Docs:** `data/benchmark/IMPORT.md` and Logbook 114's stated limit, now lifted. The tracker's B.5 note and the CHANGELOG.
- **Behaviour:** no change. The full suite is the check.
