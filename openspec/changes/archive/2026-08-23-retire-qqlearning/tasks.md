# Tasks

Line numbers are as of `main` at authoring time. Re-verify before editing.

## 1. Remove the brain

- [x] 1.1 Delete `packages/quantum-nematode/quantumnematode/brain/arch/qqlearning.py` (782 lines). The registry entry goes with it — registration is via the `@register_brain` decorator in the module.
- [x] 1.2 Delete `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qqlearning.py` (30 tests).
- [x] 1.3 `brain/arch/dtypes.py`: remove `QQLEARNING = "qqlearning"` (`:25`) and the `BrainType.QQLEARNING` entry in the `BRAIN_TYPES` literal (`:96`).
- [x] 1.4 `brain/arch/__init__.py`: remove the `from .qqlearning import ...` (`:47`) and both `__all__` entries (`:105-106`).
- [x] 1.5 `utils/config_loader.py`: remove the `QQLearningBrainConfig` import (`:35`) and its `BrainConfigType` union entry (`:121`).
- [x] 1.6 `utils/brain_factory.py`: remove the `if brain_type is BrainType.QQLEARNING:` branch — it runs **`:96-111`**, not `:96-107`; the returned dict closes at `:111` and stopping at `:107` leaves a dangling `),\n}`. Note it is the only branch requiring a `DynamicLearningRate` *exclusively* — `qvarcircuit` accepts it too (`:77`), so the import stays.
- [x] 1.7 Run `assert_registry_matches_enum` (invoked from the package `__init__`; covered by `test_registry_enum_consistency.py`) — it fails fast if the enum and registry disagree.

## 2. Counts and docstrings

- [x] 2.1 Re-derive the architecture count and the full name list **from the live `_REGISTRY`**, not by editing 27 → 26 by hand (D5 — the `openspec/config.yaml` enumeration had already drifted once, found in WS4).
- [x] 2.2 Update the count in **five** live sites — `README.md:38`, `AGENTS.md:34`, `CONTRIBUTING.md:140`, `openspec/config.yaml:18` and **`docs/roadmap.md:98`** (*"The platform **now supports**: 27 brain architectures"* — present tense, so a live claim, unlike the historical `:102` and `:1142`). Regenerate `config.yaml`'s enumeration and verify programmatically that it matches the registry exactly.
- [x] 2.3 Remove the `QQLearningBrain` entry from `README.md`'s **Quantum:** list and from `CONTRIBUTING.md`'s numbered list. It is entry **`02`** there, so all 25 later entries shift; the numbering is **zero-padded** (`01.`…`27.`), so verify it is contiguous **`01..26`** in that form.
- [x] 2.4 Drop **three** docstring mentions, not two:
  - `brain/modules.py:596`.
  - `brain/arch/_policy.py:6` — names it among *"the **four** non-PG brains"*; the word **four must become three**, not just the name removed.
  - `utils/brain_factory.py:198` — *"architectures (QVARCIRCUIT, QQLEARNING) consume this"*, in the `_build_infra_kwargs` parameter docs. This survives the branch deletion in 1.6 and becomes false.
- [x] 2.5 `experiment/metadata.py:264` and `:484` list the non-existent legacy aliases (`modular`, `mlp`, `qmodular`, `qmlp`, `ppo`) in `brain_type` docstrings — the same drift D4 corrects in the spec. Fix alongside.

## 3. Docs

- [x] 3.1 `docs/roadmap.md`: record the retirement alongside the existing `:1216` deprioritisation line, so the two records agree. Also check `:102` ("6 brain architectures shipped: … QQLearningBrain …") — that is a **historical** statement about Phase 0 and should be left as-is; confirm the reading before editing.
- [x] 3.2 `docs/OPTIMIZATION_METHODS.md` (`:152`): drop `QQLearningBrain` from the CMA-ES decision-tree line, leaving `QVarCircuitBrain`.
- [x] 3.3 **Do not edit** `docs/experiments/logbooks/023-architecture-plugin-interface.md`. Its architecture inventory records what existed at the time and is a historical record, following the same rule applied to the archived OpenSpec task list in #280.

## 4. Spec

- [x] 4.1 Land the `configuration-system` RENAMED + MODIFIED delta. It restates the requirement in terms of the registry rather than re-enumerating it, and corrects the stale legacy-alias clause (D4).

- [x] 4.2 Land the `brain-architecture` RENAMED + MODIFIED delta (D5a) — `spec.md:582` names `QQLEARNING` in a live `SHALL` set titled *"Other 17 Architectures"*. A retired brain cannot satisfy a `SHALL`.

- [x] 4.3 **Hand-edit `openspec/specs/configuration-system/spec.md` line 5** (the Purpose) to drop `qqlearning` and the five non-existent aliases. `openspec archive` **ignores a delta Purpose** when the target already has one and only emits a warning (D6) — so without this step the change silently half-lands.

- [x] 4.4 `openspec validate retire-qqlearning --strict` passes.

- [x] 4.5 Add a test for the delta's new rejection obligation. The behaviour is real (`config_loader.py:2982` and `:3007` raise `ValueError` naming the unknown type) but `grep -rn "Unknown brain type" packages/quantum-nematode/tests/` returns nothing — the existing unknown-brain tests cover the hyperparameter schema and the encoder, not `configure_brain`. A live `SHALL` with no coverage.

## 5. Audit and gate

- [x] 5.1 Grep audit **with a declared expected answer**, following the WS2 precedent where an open-ended sweep missed a site. After the removal, `qqlearning` / `QQLearning` / `QQ_LEARNING` shall appear in exactly these places, and nowhere else:

  | File | Why it survives |
  |---|---|
  | `docs/experiments/logbooks/023-...md` | historical inventory |
  | `docs/roadmap.md` | the Phase 0 shipped-list (`:102`) and the deprioritisation/retirement lines (`:1216`) |
  | `docs/STANDARDIZATION.md`, `openspec/specs/benchmark-management/spec.md` | historical "3 of the eventual 27" phrasing — count only, no brain name |
  | `openspec/specs/configuration-system/spec.md` | the delta **deliberately** writes `qqlearning` into the rejection example and the retirement note |
  | `openspec/specs/brain-architecture/spec.md` | same — the amended bar records why `QQLEARNING` left |
  | `openspec/changes/archive/**` | archived changes are history |
  | `openspec/changes/retire-qqlearning/**` | this change |
  | `tests/.../utils/test_brain_type_validation.py` | added by task 4.5; names the retired brain deliberately, as a type that must be **rejected** |
  | `htmlcov/`, `__pycache__/` | build output, not tracked |

  Any other hit is a miss, not a judgement call. **The first version of this table
  omitted the two live spec files that this change's own deltas write `qqlearning`
  into, so the audit would have failed on its own work product** — the exact
  false-signal the declared-answer mechanism exists to prevent.

- [x] 5.2 Confirm nothing was orphaned. `initializers` has **ten** other consumers (`spiking_ppo`, `qvarcircuit`, `lstmppo`, `config_loader`, `cfc_ppo`, `equivariant_quantum`, `mlpdqn`, `mlpppo`, `mlpreinforce`, `transformer_ppo`) and `DynamicLearningRate` is also used by `qvarcircuit`, `learning_rate.py`, `brain_factory`, `config_loader`, `scripts/run_simulation.py` and one test module (`test_qvarcircuit.py`). Re-derive both lists rather than trusting these — the first draft of this line said *two* test modules, and the re-derivation found one.

- [x] 5.3 Confirm the compatibility check still holds (D3): `ExperimentMetadata.brain_type` is `str`, not the enum, and **0** artifacts carry the string — so no historical record fails to load.

- [x] 5.4 Full `uv run pytest -m "not nightly"`: **4136 (main) − 30 (deleted) + 7 (added by task 4.5) = 4113**, and nothing else fails. The count is stated as an arithmetic identity rather than "drops by 30" because this change both removes and adds tests; a bare subtraction would have flagged a passing gate as a failure. `uv run pyright` 0 errors. `uv run pre-commit run -a` clean.

- [x] 5.5 Close [#282](https://github.com/SyntheticBrains/nematode/issues/282) via the PR.

- [x] 5.6 Archive to `openspec/changes/archive/<YYYY-MM-DD>-retire-qqlearning/`.
