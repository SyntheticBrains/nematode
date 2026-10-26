## Context

The measurements below come from the worktree, each submodule imported without running its package's `__init__`:

| Module | Loads `torch`/`qiskit` | Brain, agent or env modules |
|---|---|---|
| `quantumnematode.dtypes` | no | 0 |
| `validation.chemotaxis` | no | 0 |
| `validation.datasets` | no | 0 |
| `validation.adaptive_sensor` | yes | 69 |
| `report.dtypes` | yes, through `brain.arch._brain` | 60 |

## Decisions

1. **`BehaviourStep` moves verbatim**, with its docstring and the comment on why it is a slotted dataclass, to `report/behaviour.py`, which imports only `dataclasses`. `report.dtypes` re-exports it as `from quantumnematode.report.behaviour import BehaviourStep as BehaviourStep`.

   It is the same class object, so the following are unaffected:

   - `isinstance` checks;
   - the behaviour-capture JSON, which uses `dataclasses.asdict` and `BehaviourStep(**d)`;
   - every existing import.

2. **Only `adaptive_sensor` becomes lazy.** `validation/__init__.py` keeps its eager imports of `chemotaxis`, `datasets` and `quantumnematode.dtypes`. Its module-level `__getattr__(name)` imports `adaptive_sensor` on first access to one of that submodule's four names, then returns the attribute. A `TYPE_CHECKING` block keeps the four names visible to the type checker, and `__all__` is unchanged. An unknown name raises `AttributeError`, as Python requires.

3. **Callers inside `benchmark/`** import `BehaviourStep` from `report.behaviour`. `grading.py` already imports it under `TYPE_CHECKING` only, and is left as is.

4. **The test.** A fresh interpreter (`subprocess`) imports `quantumnematode.benchmark.grading` and `quantumnematode.benchmark.track_import`. The test asserts that neither `torch` nor any `quantumnematode.brain`, `quantumnematode.agent` or `quantumnematode.env` module is loaded. A second assertion checks that `quantumnematode.validation.run_step_input` still resolves.

## Risks

- **\[A caller relied on importing `validation` to load `adaptive_sensor`'s side effects\]** → `adaptive_sensor` registers nothing, since it defines functions and a dataclass. The full suite checks this.
- **[Pickles name the class's module]** → `BehaviourStep` would then pickle as `quantumnematode.report.behaviour.BehaviourStep`. Nothing pickles it: captures are JSON.
