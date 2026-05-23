# Tasks: add-architecture-plugin-interface

Phase 6 Tranche 2 (T2). Closes [Gate 1](../phase6-tracking/tasks.md). Sub-tasks expand T2.1 through T2.10 from [phase6-tracking/tasks.md:106-115](../phase6-tracking/tasks.md) into a sequenced implementation checklist. Implementation order follows [design.md § Decision 7](design.md) (Migration sequencing) and [design.md § Migration Plan](design.md).

## 1. Registry + Protocol scaffolding (T2.1 + T2.2 + T2.3)

- [ ] 1.1 Audit `setup_brain_model()` dispatcher and document the 19 elif branches in a short notes file at `openspec/changes/add-architecture-plugin-interface/notes/dispatcher-audit.md` (per [phase6-tracking/tasks.md T2.1](../phase6-tracking/tasks.md)). Cross-reference to the registry refactor plan.
- [ ] 1.2 Create `packages/quantum-nematode/quantumnematode/brain/arch/_registry.py` with `@register_brain(name, config_cls, brain_type, families, topology_factory=None, rule_factory=None)` decorator, `instantiate_brain(name, config, **infra_kwargs) -> Brain`, `get_registration(name) -> Registration`, `list_registered_brains() -> set[str]`, and a `Registration` dataclass.
- [ ] 1.3 Implement duplicate-name detection in `@register_brain(...)` — raise `ValueError` with identifying message if `name` is already registered.
- [ ] 1.4 Implement unknown-name error in `instantiate_brain(...)` — raise `ValueError` listing available names.
- [ ] 1.5 Create `packages/quantum-nematode/quantumnematode/brain/arch/_topology.py` with `BrainTopology` Protocol (`n_inputs`, `n_outputs`, `n_hidden`, `forward`, `apply_weight_mask`). Provide a default `IdentityMaskTopology` mixin for dense topologies (returns weights unchanged from `apply_weight_mask`).
- [ ] 1.6 Create `packages/quantum-nematode/quantumnematode/brain/arch/_rule.py` with `LearningRule` Protocol (`step(topology, batch) -> RuleStepReport`, `reset_episode()`) and a `RuleStepReport` dataclass with loss + gradient-norm fields.
- [ ] 1.7 Write `tests/quantumnematode_tests/brain/arch/test_registry.py`: register / get / list round-trip; duplicate-name detection; unknown-name lookup; consistency between registry and `BrainType` enum at import time.
- [ ] 1.8 Run `uv run pytest packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_registry.py -v` and confirm green.

## 2. MLPPPO migration + byte-equivalence (T2.4 + T2.5 — Gate 1 G1.d MUST #1)

- [ ] 2.1 Capture pre-refactor MLPPPO byte-equivalence fixture: run [configs/scenarios/foraging/lstmppo_small_klinotaxis.yml](../../../configs/scenarios/foraging/lstmppo_small_klinotaxis.yml) with brain swapped to mlpppo, pinned seed, 10 episodes; serialise per-step action-probability tensors + per-episode parameter tensors as a Pickle fixture under `tests/quantumnematode_tests/brain/arch/test_data/mlpppo_pre_refactor.pkl`.
- [ ] 2.2 Add `@register_brain("mlpppo", MLPPPOBrainConfig, BrainType.MLP_PPO, families=("classical",))` decorator to `MLPPPOBrain` in [packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py).
- [ ] 2.3 Extract internal `topology` + `rule` references in `MLPPPOBrain.__init__` — minimal-edit: wrap the existing actor/critic MLPs in an `MLPTopology` adapter that delegates `forward(x)` to the existing forward path; wrap the existing PPO optimiser + loss code in a `PPORule` adapter that delegates `step(...)` to the existing learn path. **Do not rewrite call sites — wrap them.** Per [design.md § Decision 2](design.md), byte-equivalence falls out from preserving the tensor-op sequence.
- [ ] 2.4 Write `tests/quantumnematode_tests/brain/arch/test_migration_byte_equivalence.py::test_mlpppo_byte_equivalence` — load fixture from 2.1, re-run identically against post-refactor brain via the registry, assert `torch.equal` on every captured tensor.
- [ ] 2.5 Run `uv run pytest packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_migration_byte_equivalence.py::test_mlpppo_byte_equivalence -v` and confirm green. **Migration validation gate**: if this fails, halt the migration and re-design [design.md § Decision 2](design.md) strategy.

## 3. LSTMPPO migration + byte-equivalence (T2.4 + T2.5 — Gate 1 G1.d MUST #2)

- [ ] 3.1 Capture pre-refactor LSTMPPO byte-equivalence fixture: run [configs/scenarios/foraging/lstmppo_small_klinotaxis.yml](../../../configs/scenarios/foraging/lstmppo_small_klinotaxis.yml) unmodified, pinned seed, 10 episodes; serialise per-step action-probability tensors + LSTM hidden state + per-episode parameter tensors as a Pickle fixture under `tests/quantumnematode_tests/brain/arch/test_data/lstmppo_pre_refactor.pkl`.
- [ ] 3.2 Add `@register_brain("lstmppo", LSTMPPOBrainConfig, BrainType.LSTM_PPO, families=("classical",))` decorator to `LSTMPPOBrain` in [packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py).
- [ ] 3.3 Extract internal `topology` + `rule` references in `LSTMPPOBrain.__init__` (same wrap-not-rewrite pattern as MLPPPO); the LSTM hidden-state management remains owned by the topology adapter.
- [ ] 3.4 Add `tests/quantumnematode_tests/brain/arch/test_migration_byte_equivalence.py::test_lstmppo_byte_equivalence` — load fixture from 3.1, re-run identically, assert `torch.equal` on every captured tensor + LSTM hidden state.
- [ ] 3.5 Run the test and confirm green. **Migration validation gate**: if this fails, halt the migration.

## 4. Eight classical brains migration (T2.4 + T2.5 — `atol=1e-7`)

- [ ] 4.1 Migrate `MLPReinforceBrain` ([mlpreinforce.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/mlpreinforce.py)): add `@register_brain("mlpreinforce", MLPReinforceBrainConfig, BrainType.MLP_REINFORCE, families=("classical",))` + extract topology/rule references.
- [ ] 4.2 Migrate `MLPDQNBrain` ([mlpdqn.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/mlpdqn.py)): same pattern.
- [ ] 4.3 Migrate `QRCBrain` ([qrc.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/qrc.py)): same pattern; `families=("classical",)` (QRC is classical reservoir despite the Q prefix).
- [ ] 4.4 Migrate `CRHBrain` ([crh.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/crh.py)): same pattern.
- [ ] 4.5 Migrate `CRHQLSTMBrain` ([crhqlstm.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/crhqlstm.py)): same pattern.
- [ ] 4.6 Migrate `HybridClassicalBrain` ([hybridclassical.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/hybridclassical.py)): same pattern.
- [ ] 4.7 Migrate `SpikingReinforceBrain` ([spikingreinforce.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/spikingreinforce.py)): same pattern; `families=("spiking",)`.
- [ ] 4.8 Write `tests/quantumnematode_tests/brain/arch/test_migration_numerical_equivalence.py` — parametrised over the 8 classical/spiking arch names; for each, run a 5-step smoke training pre- and post-refactor with pinned seed, assert `torch.allclose(post, pre, rtol=0, atol=1e-7)` on parameter tensors.
- [ ] 4.9 Run `uv run pytest packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_migration_numerical_equivalence.py -v` and tabulate per-architecture pass/fail in `notes/migration-equivalence-results.md`. Any arch failing the tolerance: investigate, fix, or widen tolerance with written justification in the notes.

## 5. Nine quantum + spiking-quantum brains migration (T2.4 + T2.5 — `atol=1e-7` with deterministic simulator)

- [ ] 5.1 Migrate `QVarCircuitBrain` ([qvarcircuit.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/qvarcircuit.py)): add `@register_brain("qvarcircuit", QVarCircuitBrainConfig, BrainType.QVARCIRCUIT, families=("quantum",))` + extract topology/rule references.
- [ ] 5.2 Migrate `QQLearningBrain` ([qqlearning.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/qqlearning.py)): same pattern.
- [ ] 5.3 Migrate `QRHBrain` ([qrh.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/qrh.py)): same pattern; `families=("classical", "quantum")` (reservoir + quantum hybrid).
- [ ] 5.4 Migrate `QEFBrain` ([qef.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/qef.py)): same pattern.
- [ ] 5.5 Migrate `QSNNReinforceBrain` ([qsnnreinforce.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/qsnnreinforce.py)): same pattern; `families=("quantum", "spiking")`.
- [ ] 5.6 Migrate `QSNNPPOBrain` ([qsnnppo.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/qsnnppo.py)): same pattern; `families=("quantum", "spiking")`.
- [ ] 5.7 Migrate `HybridQuantumBrain` ([hybridquantum.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/hybridquantum.py)): same pattern.
- [ ] 5.8 Migrate `HybridQuantumCortexBrain` ([hybridquantumcortex.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/hybridquantumcortex.py)): same pattern.
- [ ] 5.9 Migrate `QLIFLSTMBrain` ([qliflstm.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/qliflstm.py)): same pattern.
- [ ] 5.10 Migrate `QRHQLSTMBrain` ([qrhqlstm.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/qrhqlstm.py)): same pattern.
- [ ] 5.11 Extend `test_migration_numerical_equivalence.py` parametrisation to cover the 9 quantum/spiking-quantum arches. Configure their smoke fixtures to use a deterministic statevector simulator (per [design.md § Risk A](design.md)).
- [ ] 5.12 Run the full parametrised test suite; for any arch exceeding `atol=1e-7`, widen tolerance to the fallback tier ("training-curve shape preserved within 5%") with written justification in `notes/migration-equivalence-results.md`.

## 6. Dispatcher + loader collapse (T2.2 + T2.4)

- [ ] 6.1 Refactor [packages/quantum-nematode/quantumnematode/utils/brain_factory.py](../../../packages/quantum-nematode/quantumnematode/utils/brain_factory.py) `setup_brain_model()`: collapse the 19-elif body to a single `instantiate_brain(brain_type.value, brain_config, shots=shots, ...)` call. Keep the public function signature unchanged (callers still pass `BrainType` + config). Expect ~400 LOC removed.
- [ ] 6.2 Refactor [packages/quantum-nematode/quantumnematode/utils/config_loader.py](../../../packages/quantum-nematode/quantumnematode/utils/config_loader.py) `BRAIN_CONFIG_MAP`: derive from registry via `{name: reg.config_cls for name, reg in get_all_registrations().items()}`. Remove the hand-maintained 19-entry literal.
- [ ] 6.3 Refactor [packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py): keep `BrainType` enum (out-of-tree callers depend on it) but derive `BRAIN_TYPES` Literal + `QUANTUM_BRAIN_TYPES` / `CLASSICAL_BRAIN_TYPES` / `SPIKING_BRAIN_TYPES` sets from registry `families` metadata.
- [ ] 6.4 Add startup-time consistency check: in [`packages/quantum-nematode/quantumnematode/brain/arch/__init__.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/__init__.py), after importing all arch modules, assert `set(BrainType.__members__.values()) == set(map(str, list_registered_brains()))`; raise descriptive exception on mismatch.
- [ ] 6.5 Update [`packages/quantum-nematode/quantumnematode/brain/arch/__init__.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/__init__.py) re-exports: keep all existing Brain + Config class re-exports; add `register_brain`, `instantiate_brain`, `BrainTopology`, `LearningRule`, `Registration` to `__all__`.
- [ ] 6.6 Run YAML compatibility regression: write `tests/quantumnematode_tests/utils/test_config_loader_yaml_compat.py` that walks [configs/scenarios/](../../../configs/scenarios/) and asserts every YAML file loads to a valid `Brain` instance via the registry. Pass criterion: zero failures.
- [ ] 6.7 Run the full quantum-nematode test suite: `uv run pytest -m "not nightly" packages/quantum-nematode/` — confirm green or triage failures back to the affected migration step.

## 7. ConnectomePPOBrain implementation (T2.3 + T2.6)

- [ ] 7.1 Create `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py` with `ConnectomePPOBrainConfig` Pydantic model per the [connectome-ppo-brain spec § ConnectomePPOBrainConfig](specs/connectome-ppo-brain/spec.md) (fields: `connectome_source`, `enable_gap_junctions`, `chemical_mask_mode`, `forward_pass_depth`, `freeze_updates`, plus PPO hyperparams).
- [ ] 7.2 Implement `ConnectomeTopology` (a `BrainTopology` Protocol implementation) inside `connectome_ppo.py`: load connectome via `load_cook_2019_hermaphrodite()`, build `W_chem` (302×302 learnable) + `M_chem` (302×302 boolean strict-mask) + `G_gap` (302×302 fixed, fan-in normalised, `requires_grad=False`). `forward(x)` iterates `h = tanh(W_chem.T @ (M_chem * h) + G_gap.T @ h)` for K = `forward_pass_depth` steps. `apply_weight_mask(W)` returns `W * M_chem`.
- [ ] 7.3 Implement sensor projection in `ConnectomeTopology`: map `BrainParams.food_chemotaxis_*` to ASE-left/ASE-right/AWC-left/AWC-right/AWA neurons via additive injection scaled by per-input learnable gains. Map `BrainParams.proprioception_*` to AVA/AVB command interneurons.
- [ ] 7.4 Implement motor readout in `ConnectomeTopology`: pool 302-dim activations by VB/DB/VA/DA motor-class membership (mean pooling within each class → 4-vec class activations); apply learnable 4×4 readout matrix → 4 action logits aligned with `DEFAULT_ACTIONS`.
- [ ] 7.5 Implement `ConnectomePPOBrain` (a `Brain` Protocol implementation) inside `connectome_ppo.py`: composes `ConnectomeTopology` + `PPORule` (reuse from MLPPPO migration). `run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy` delegate to the topology + rule pair.
- [ ] 7.6 Implement `freeze_updates` semantics: when `config.freeze_updates is True`, `PPORule.step(...)` is a no-op (skip gradient compute, skip optimiser invocation). Verify weights remain byte-identical to construction across 100 episodes.
- [ ] 7.7 Add `@register_brain("connectomeppo", ConnectomePPOBrainConfig, BrainType.CONNECTOMEPPO, families=("classical",))` decorator. Add `BrainType.CONNECTOMEPPO = "connectomeppo"` enum member at [packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py).
- [ ] 7.8 Re-export `ConnectomePPOBrain` + `ConnectomePPOBrainConfig` from [`packages/quantum-nematode/quantumnematode/brain/arch/__init__.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/__init__.py).
- [ ] 7.9 Write `tests/quantumnematode_tests/brain/arch/test_connectome_ppo.py`: construction from a fixture `Connectome`; forward-pass shape (4 action logits) + finiteness; strict-mask invariant `(W_chem * ~M_chem).abs().max() == 0.0` after a 5-step training; gap-junction tensor byte-identical across 5 training steps; sensor projection routes inputs to expected neurons (assert non-zero gradient on ASE/AWC/AWA gains, zero gradient on un-targeted neurons); motor readout produces 4-dim logits.
- [ ] 7.10 Add `tests/quantumnematode_tests/brain/arch/test_topology_rule_protocols.py` covering MLPPPO, LSTMPPO, and ConnectomePPO — assert each instance satisfies `BrainTopology` and `LearningRule` Protocol conformance via `isinstance(..., BrainTopology)` (runtime_checkable).
- [ ] 7.11 Run `uv run pytest packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_connectome_ppo.py packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_topology_rule_protocols.py -v` and confirm green.

## 8. Klinotaxis smoke config + Gate 1 G1.c paired control (T2.6)

- [ ] 8.1 Create `configs/scenarios/foraging/connectome_ppo_klinotaxis.yml` — fork [lstmppo_small_klinotaxis.yml](../../../configs/scenarios/foraging/lstmppo_small_klinotaxis.yml): swap `brain.name` to `connectomeppo`; swap brain config block to `ConnectomePPOBrainConfig` defaults (`chemical_mask_mode: strict`, `enable_gap_junctions: true`, `forward_pass_depth: 1`); keep STAM, keep `chemotaxis_mode: klinotaxis`, keep env + reward shape unchanged. Pin a documented seed.
- [ ] 8.2 Run the learning run: `uv run python scripts/run_simulation.py --config configs/scenarios/foraging/connectome_ppo_klinotaxis.yml --runs 1 --theme headless` for 100 episodes. Capture episode-return time series.
- [ ] 8.3 Create `configs/scenarios/foraging/connectome_ppo_klinotaxis_frozen_control.yml` — fork 8.1, set `brain.config.freeze_updates: true`, same pinned seed.
- [ ] 8.4 Run the frozen-random-weights control: same command as 8.2 with the frozen config. Capture episode-return time series.
- [ ] 8.5 Compute G1.c pass conditions in a short evaluation script (suggest `tmp/evaluations/architecture-plugin-interface/g1c_evaluation.py`): (a) no NaNs/Infs across 100 episodes in either run, (b) `learning_run.last25_mean ≥ 1.10 * frozen_control.last25_mean`, (c) `learning_run.last25_mean > learning_run.first25_mean`. Record absolute numbers + verdict.
- [ ] 8.6 If G1.c fails: run the diagnostic sequence per [design.md § Decision 6](design.md) (topology density check, reward-shaping check, RNG-seed sweep). If still failing → PIVOT to hand-curated subset per [phase6-tracking/design.md § Decision 6 § Gate 1 STOP/PIVOT triggers](../phase6-tracking/design.md). Pivot decision recorded in logbook 023.

## 9. Plugin-developer documentation (T2.8)

- [ ] 9.1 Create `docs/architecture/plugin-developer-guide.md` — walkthrough of "how to add a new architecture family": decorator-registration usage, optional topology/rule factoring, config-class pattern, where to add tests, the ≤ 6 files target (per Gate 2 G2.b).
- [ ] 9.2 Add a worked example to the guide: registering a hypothetical `TinyMLPBrain` (or revive a Phase 0-3 architecture that didn't make the MUST set, per [phase6-tracking/tasks.md T2.7](../phase6-tracking/tasks.md)). Record actual files-touched count as the methodology baseline T5 will re-verify against.
- [ ] 9.3 Cross-link the guide from [docs/roadmap.md](../../../docs/roadmap.md) Phase 6 § L1 row and from the `brain/arch/__init__.py` module docstring.

## 10. Logbook 023 + Gate 1 decision (T2.10)

- [ ] 10.1 Create `docs/experiments/logbooks/023-architecture-plugin-interface.md` with sections: Implementation Summary, Migration Regression Bar Results (MLPPPO + LSTMPPO byte-equivalence proof; 17-architecture numerical-equivalence table), Gate 1 G1.c Paired-Control Measurement (learning + frozen control numbers + pass/fail verdict against the three conditions), Gate 1 Decision (GO/PIVOT/STOP with reasoning evaluating all four G1.a–G1.d criteria), T2↔T3 Handshake (the corrected ASH/ADL nociception in T3 consumes the new plugin interface; note any constraints).
- [ ] 10.2 **Pause for user review** of the logbook before finalising the Gate 1 verdict. The project convention is that evaluation results and decision-gate verdicts are reviewed by the user before being written as final into a logbook; the verdict can be deferred to a follow-up edit.
- [ ] 10.3 Finalise the Gate 1 decision verdict in the logbook after user review.

## 11. Tracker + roadmap updates (T2.9)

- [ ] 11.1 Tick [openspec/changes/phase6-tracking/tasks.md](../phase6-tracking/tasks.md) T2.1 through T2.10 sub-tasks as work lands (running edit as each section closes; final pass to confirm all ticked).
- [ ] 11.2 Add Gate 1 decision link in [openspec/changes/phase6-tracking/tasks.md § Gate 1](../phase6-tracking/tasks.md): replace `[add link to the T2 logbook where the decision is recorded]` with the actual logbook 023 URL + section anchor. Tick the "Gate 1 decision recorded" + "Gate 1 decision link" checkboxes.
- [ ] 11.3 Flip [docs/roadmap.md](../../../docs/roadmap.md) Phase 6 Tranche Tracker T2 row: `🟡 in progress` on first commit of this branch → `✅ complete` after Gate 1 decision is finalised. Update the Mid-Phase Decision Gates table with the Gate 1 outcome (GO/PIVOT/STOP) and a one-line summary.

## 12. Pre-merge verification

- [ ] 12.1 Run full pre-commit on all changed files: `uv run pre-commit run --files <changed-files>`.
- [ ] 12.2 Run the full quantum-nematode test suite: `uv run pytest -m "not nightly" packages/quantum-nematode/`. Triage any new failures.
- [ ] 12.3 Run `openspec validate add-architecture-plugin-interface --strict` and confirm clean.
- [ ] 12.4 Audit staged-file sizes vs `.gitattributes` LFS rules — the byte-equivalence Pickle fixtures may approach 100 KB; flag any >100 KB file not covered by existing rules.
- [ ] 12.5 Scan staged content for absolute home paths (e.g. `/Users/...`, `/home/...`, `C:\\Users\\...`) and `file:///` URI prefixes; sanitise to repo-relative references before committing.
- [ ] 12.6 **Pause for user authorisation** before `git push` or `gh pr create`. Per project convention, remote-state mutations require explicit user approval each time.
