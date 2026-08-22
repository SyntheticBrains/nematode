# Tasks

Line numbers are as of `main` at authoring time. Re-verify before editing.

## 1. Remove the brain

- [ ] 1.1 Delete `packages/quantum-nematode/quantumnematode/brain/arch/qqlearning.py` (782 lines). The registry entry goes with it — registration is via the `@register_brain` decorator in the module.
- [ ] 1.2 Delete `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_qqlearning.py` (30 tests).
- [ ] 1.3 `brain/arch/dtypes.py`: remove `QQLEARNING = "qqlearning"` (`:25`) and the `BrainType.QQLEARNING` entry in the `BRAIN_TYPES` literal (`:96`).
- [ ] 1.4 `brain/arch/__init__.py`: remove the `from .qqlearning import ...` (`:47`) and both `__all__` entries (`:105-106`).
- [ ] 1.5 `utils/config_loader.py`: remove the `QQLearningBrainConfig` import (`:35`) and its `BrainConfigType` union entry (`:121`).
- [ ] 1.6 `utils/brain_factory.py`: remove the `if brain_type is BrainType.QQLEARNING:` branch (`:96-107`). Note it is the only branch requiring a `DynamicLearningRate` *exclusively* — `qvarcircuit` accepts it too (`:77`), so the import stays.
- [ ] 1.7 Run `assert_registry_matches_enum` (invoked from the package `__init__`; covered by `test_registry_enum_consistency.py`) — it fails fast if the enum and registry disagree.

## 2. Counts and docstrings

- [ ] 2.1 Re-derive the architecture count and the full name list **from the live `_REGISTRY`**, not by editing 27 → 26 by hand (D5 — the `openspec/config.yaml` enumeration had already drifted once, found in WS4).
- [ ] 2.2 Update the count in `README.md`, `AGENTS.md`, `CONTRIBUTING.md` and `openspec/config.yaml`, and regenerate `config.yaml`'s enumeration. Verify programmatically that the enumeration matches the registry exactly.
- [ ] 2.3 Remove the `QQLearningBrain` entry from `README.md`'s **Quantum:** list and from `CONTRIBUTING.md`'s numbered list; re-derive `CONTRIBUTING.md`'s numbering and verify it is contiguous `1..26`.
- [ ] 2.4 Drop the docstring mentions in `brain/modules.py` (`:596`) and `brain/arch/_policy.py` — the latter names it among the non-policy-gradient brains, a list that must stay accurate.

## 3. Docs

- [ ] 3.1 `docs/roadmap.md`: record the retirement alongside the existing `:1216` deprioritisation line, so the two records agree. Also check `:102` ("6 brain architectures shipped: … QQLearningBrain …") — that is a **historical** statement about Phase 0 and should be left as-is; confirm the reading before editing.
- [ ] 3.2 `docs/OPTIMIZATION_METHODS.md` (`:152`): drop `QQLearningBrain` from the CMA-ES decision-tree line, leaving `QVarCircuitBrain`.
- [ ] 3.3 **Do not edit** `docs/experiments/logbooks/023-architecture-plugin-interface.md`. Its architecture inventory records what existed at the time and is a historical record, following the same rule applied to the archived OpenSpec task list in #280.

## 4. Spec

- [ ] 4.1 Land the `configuration-system` MODIFIED delta. It restates the requirement in terms of the registry rather than re-enumerating it, and corrects the stale legacy-alias clause (D4 — `modular`, `qmodular`, `mlp`, `qmlp`, `ppo` are named in the spec but exist in no `BrainType` value and no alias mapping).
- [ ] 4.2 `openspec validate retire-qqlearning --strict` passes.

## 5. Audit and gate

- [ ] 5.1 Grep audit **with a declared expected answer**, following the WS2 precedent where an open-ended sweep missed a site. After the removal, `qqlearning` / `QQLearning` / `QQ_LEARNING` shall appear in exactly these places, and nowhere else:

  | File | Why it survives |
  |---|---|
  | `docs/experiments/logbooks/023-...md` | historical inventory |
  | `docs/roadmap.md` | the Phase 0 shipped-list and the deprioritisation/retirement lines |
  | `openspec/changes/archive/**` | archived changes are history |
  | `openspec/changes/retire-qqlearning/**` | this change |

  Any other hit is a miss, not a judgement call.

- [ ] 5.2 Confirm nothing was orphaned: `initializers` (other consumers: `spiking_ppo`, `qvarcircuit`, `lstmppo`, `config_loader`) and `DynamicLearningRate` (also used by `qvarcircuit`) both retain live callers.

- [ ] 5.3 Confirm the compatibility check still holds (D3): `ExperimentMetadata.brain_type` is `str`, not the enum, and **0** artifacts carry the string — so no historical record fails to load.

- [ ] 5.4 Full `uv run pytest -m "not nightly"`: the suite drops by exactly the 30 deleted tests and nothing else fails. `uv run pyright` 0 errors. `uv run pre-commit run -a` clean.

- [ ] 5.5 Close [#282](https://github.com/SyntheticBrains/nematode/issues/282) via the PR.

- [ ] 5.6 Archive to `openspec/changes/archive/<YYYY-MM-DD>-retire-qqlearning/`.
