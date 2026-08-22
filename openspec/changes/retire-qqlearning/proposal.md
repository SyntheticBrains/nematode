## Why

`QQLearningBrain` is finished as a research direction and unfinished as code, and
the two records disagree. The roadmap already closed it —
*"~~QQLearningBrain completion~~ — evaluated, not competitive; deprioritised"*
([`docs/roadmap.md:1216`](../../../docs/roadmap.md)) — while the module still reads
as work in progress:

- **9 of the 11 `TODO`s in the entire `quantumnematode` package** live in this one
  782-line file (82%), including two explicit "use or remove" dead-parameter flags.
- **`parameter_initializer` is dead**: accepted at `qqlearning.py:126` with an
  explicit `# noqa: ARG002`, never assigned or read.
- **`learning_rate` is dead in a subtler way**: assigned at `:177` and never read
  anywhere in the class. It looks wired up and is not.
- **`copy()` restores 1 of 32 instance attributes** (plus 7 via the constructor).
  It drops `epsilon`, `experience_buffer`, `step_count`, `update_count`, `rng`,
  `parameters` — the entire learning state — behind a `# TODO: Copy entire state`.
  `copy()` is live: many-worlds mode calls it at `runners.py:1313` and `:1367`.

Nothing depends on it. **0** `configs/scenarios/**/qqlearning*` files, **0**
experiment artifacts, and the only live references outside its own module are an
enum member, a factory branch, a config-union entry and two docstring mentions.

Keeping it costs a maintenance surface that every future cross-brain change has to
carry — the `_policy.py` consolidation had to reason about it, and the L4 plasticity
work in Phase 7 would have to again — in exchange for a brain the project has
already decided not to pursue.

## What Changes

- **Remove `brain/arch/qqlearning.py`** and its 30-test module.
- **Remove `BrainType.QQLEARNING`** and its `BRAIN_TYPES` literal entry; the
  registry self-registration goes with the module, keeping
  `assert_registry_matches_enum` satisfied.
- **Remove the `qqlearning` branch in `brain_factory._build_infra_kwargs`**, its
  `config_loader` import + `BrainConfigType` union entry, and the
  `brain/arch/__init__.py` import and two `__all__` entries.
- **Amend the `configuration-system` capability**: its Purpose and its
  *Brain Type Validation* requirement both name `qqlearning` as a valid type.
- **Update the architecture count 27 → 26** in `README.md`, `AGENTS.md`,
  `CONTRIBUTING.md` and `openspec/config.yaml`, and drop the two docstring
  mentions in `brain/modules.py` and `brain/arch/_policy.py`.
- **Record the retirement in `docs/roadmap.md`** alongside the existing
  deprioritisation line, and drop the decision-tree mention in
  `docs/OPTIMIZATION_METHODS.md`.
- **No tombstone module.** Unlike `remove-nematodebench`, no capability disappears
  here — `configuration-system` survives and is merely amended — so there is no
  spec to retire and nothing for a stale link to land on. Git history is the
  record.

## Capabilities

### New Capabilities

<!-- none -->

### Modified Capabilities

- `configuration-system`: `qqlearning` is removed from the set of valid brain
  types, and the requirement's stale legacy-alias list is corrected (see design D4
  — `modular`, `qmodular`, `mlp`, `qmlp`, `ppo` are named in the spec but exist
  nowhere in the code).

## Impact

- **Code (7 files):** `brain/arch/qqlearning.py` (deleted), `brain/arch/__init__.py`,
  `brain/arch/dtypes.py`, `utils/brain_factory.py`, `utils/config_loader.py`,
  `brain/modules.py`, `brain/arch/_policy.py`.
- **Tests:** `tests/.../brain/arch/test_qqlearning.py` (deleted, 30 tests).
- **Spec:** `openspec/specs/configuration-system/spec.md`.
- **Docs (6):** `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `openspec/config.yaml`,
  `docs/roadmap.md`, `docs/OPTIMIZATION_METHODS.md`.
- **Deliberately untouched:** `docs/experiments/logbooks/023-architecture-plugin-interface.md`,
  whose architecture inventory is a historical record of what existed at the time.
- **No behavioural change to any other brain.** Nothing imports `QQLearningBrain`
  outside its own module, its test, and the wiring listed above; `initializers` and
  `DynamicLearningRate` both have other consumers, so nothing is orphaned.
