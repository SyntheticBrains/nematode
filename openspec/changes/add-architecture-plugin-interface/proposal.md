## Why

Phase 6 Tranche 2 (T2, L1 layer) refactors the brain-architecture dispatcher into a pluggable registry interface and introduces the first connectome-constrained brain. Without T2, the headline Phase 6 platform claim — *first closed-loop learning + evolution on the real *C. elegans* connectome with a pluggable architecture interface* — has a substrate (T1 just shipped) but no plugin contract to consume it through, and no PPO-trainable connectome architecture wired into the env.

T2 closes **Gate 1**. Three coupled outcomes ride on this change:

1. **L1 plugin parity** — the 19-branch (1 `if` + 18 `elif`) `setup_brain_model()` dispatcher at [packages/quantum-nematode/quantumnematode/utils/brain_factory.py](../../../packages/quantum-nematode/quantumnematode/utils/brain_factory.py) (459 LOC, one branch per `BrainType`) is replaced with a decorator-registration registry. Adding a new architecture must touch ≤ 6 files with no per-architecture branches in simulation/training loops (the Gate 2 G2.b + G2.c criteria T5 will verify against).
2. **Topology / learning-rule factoring** — every existing `BrainType` entry is a fused `(topology + rule)` bundle (e.g. `LSTMPPOBrain` = LSTM topology + PPO rule). L0 connectome data must be a *topology* that PPO, spiking, and NEAT-evolved-topology+PPO can all consume; T2 introduces `BrainTopology` + `LearningRule` Protocols so the L0 connectome substrate plugs into PPO without bespoke wiring (and T8's NEAT-evolved topologies will plug in identically).
3. **Connectome-as-brain (`ConnectomePPOBrain`)** — first PPO-trainable connectome-constrained architecture, applying chemical-synapse strict-mask + fixed gap-junction weights per [phase6-tracking/design.md § Decision 7](../phase6-tracking/design.md). Trained on klinotaxis on the existing grid env; the Gate 1 G1.c training-signal check evaluates against a paired frozen-random-weights forward-pass control.

Six Phase 6 design decisions from [phase6-tracking/design.md](../phase6-tracking/design.md) shape this change's scope:

- **Decision 1 (tranche ordering)**: T2 closes Gate 1; env upgrade (T5/T6/T7) is deliberately out of scope so the env-upgrade delta (T4-grid vs T7-upgraded) becomes its own finding.
- **Decision 3 (L1 plugin parity is real refactor work)**: registry refactor + topology/rule factoring; existing 19 architectures migrate behind the new registry with a documented regression bar.
- **Decision 4 (MUST set)**: LSTMPPO + MLPPPO are Gate 1 G1.d MUST architectures (byte-equivalence required); the other 17 are demoted to a looser tolerance.
- **Decision 6 (Gate 1 criteria)**: G1.b plugin registry instantiates connectome + MLPPPO through the same code path; G1.c PPO-on-connectome trains ≥ 100 episodes without NaNs, last-25 mean return ≥ frozen-random-weights forward-pass control by ≥ 10%, AND monotonic improvement first-25 → last-25; G1.d MLPPPO + LSTMPPO byte-equivalence.
- **Decision 7 (connection-type taxonomy)**: chemical synapses get strict-mask + PPO-learnable scalar weights along existing edges; gap junctions get fixed Cook 2019 counts (non-learnable, fan-in normalised); extra-synaptic/peptidergic is Phase 7 L4. Strict-mask vs soft-prior — strict-mask is the headline; soft-prior is a documented T4-scope ablation.
- **What this change does NOT decide** (deferred to later tranches per [phase6-tracking/design.md § What This Change Explicitly Does Not Decide](../phase6-tracking/design.md)): continuous-action policy parameterisation (T5); real-worm validation dataset selection (T7); the sensor-projection + motor-readout ablation choice for the connectome row (T4-scope per T4.0c). T2 picks a sensible default sensor/motor projection and documents it.

The migration regression bar follows Phase 5 M1's PredatorBrain refactor precedent ([logbook 016](../../../docs/experiments/logbooks/016-predator-brain-refactor.md)): 23 byte-equivalence tests + 80/80 metric-cell deltas exactly 0.0. T2 holds MLPPPO + LSTMPPO to that bar; the other 17 declare a `np.allclose(rtol=0, atol=1e-7)` tolerance on parameter tensors after a 5-step smoke training (catches all but floating-point reassociation drift; the only allowable drift from a pure mechanical refactor).

## What Changes

### 1. Brain Plugin Registry (`brain-architecture` MODIFIED)

Three new module files under `packages/quantum-nematode/quantumnematode/brain/arch/`:

- `_registry.py` — `@register_brain(name, config_cls, brain_type, families)` decorator + `instantiate_brain(name, config, **infra_kwargs) -> Brain` + registry inspection helpers (`get_registration`, `list_registered_brains`, `family_members`, `assert_registry_matches_enum`). Each architecture's `brain/arch/<name>.py` self-registers at import time. Single source of truth for the (name, config_cls, brain_cls, brain_type, families) mapping.
- `_topology.py` — `BrainTopology` Protocol exposing `n_inputs` / `n_outputs` / `n_hidden` / `forward(x)` / `apply_weight_mask(weights)`. Forward-compat scaffolding consumed by the new `ConnectomePPOBrain` (delivered in the follow-up change); the existing 19 brains are migrated decorator-only.
- `_rule.py` — `LearningRule` Protocol exposing `step(topology, batch) -> RuleStepReport` / `reset_episode()` + a `RuleStepReport` dataclass. Same forward-compat-scaffolding status as `BrainTopology`.

Three modified module files:

- `utils/brain_factory.py` — the 19-branch (1 `if` + 18 `elif`) `setup_brain_model()` dispatcher collapses to a thin shim that builds per-architecture infrastructure kwargs and delegates to `instantiate_brain(brain_type.value, brain_config, **infra_kwargs)`. The function signature is preserved exactly; the body shrinks from 459 LOC to ~170 LOC.
- `utils/config_loader.py` — `BRAIN_CONFIG_MAP` (formerly 19 hand-maintained entries) is now derived from the registry via `{name: reg.config_cls for name, reg in get_all_registrations().items()}`. `_resolve_brain_config` helper at [config_loader.py:153-216](../../../packages/quantum-nematode/quantumnematode/utils/config_loader.py) is unchanged (still useful for raw-dict → Pydantic resolution).
- `brain/arch/dtypes.py` — `BrainType` migrates from `Enum` to `StrEnum` (member values are already strings; the change makes `BrainType.MLP_PPO == "mlpppo"` evaluate True). `QUANTUM_BRAIN_TYPES` / `CLASSICAL_BRAIN_TYPES` / `SPIKING_BRAIN_TYPES` sets are now derived from registry `families` metadata via lazy module `__getattr__` (PEP 562), so adding a new architecture no longer requires editing these set literals. An architecture may carry multiple family tags (e.g. `QSNN_REINFORCE` is in both `QUANTUM_BRAIN_TYPES` and `SPIKING_BRAIN_TYPES`).

Decorator-only edit per architecture (19 files):

- Each `brain/arch/<name>.py` file gets a `@register_brain(name=..., config_cls=..., brain_type=..., families=...)` decorator on the Brain class. **No other code in the migrated brains changes** — per the scope decision documented in [tasks.md § 2](tasks.md), the per-brain topology/rule extraction is deferred to a follow-up change. Byte-equivalence is preserved by construction because no executing code moves.

Wired into `brain/arch/__init__.py`:

- Re-exports `register_brain`, `instantiate_brain`, `get_registration`, `list_registered_brains`, `get_all_registrations`, `BrainTopology`, `LearningRule`, `Registration`, `RuleStepReport`.
- Invokes `assert_registry_matches_enum()` after all architecture modules have imported, so accidental enum/registry drift fails loudly at import time.

### 2. Connectome-Constrained Brain (`connectome-ppo-brain` NEW) — follow-up PR

**Scope split (implementation-time amendment):** the work described in this subsection lands in a follow-up PR. The first PR ships the `brain-architecture` capability changes (§ 1 above); the follow-up PR ships everything in this subsection together with the klinotaxis smoke config (§ 3), the plugin-developer documentation (§ 5), the implementation logbook + Gate-1 decision (§ 6), and the tracker / roadmap updates (§ 7).

New module `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py`:

- `ConnectomePPOBrain` class implementing the `Brain` Protocol. Topology built from `Connectome` (T1 data model) via the T1↔T2 API sketch iteration pattern: chemical-synapse strict-mask matrix from `Connectome.chemical_synapses` (302 × 302 sparse, weights PPO-learnable along existing edges, non-existent edges pinned to zero), gap-junction matrix from `Connectome.gap_junctions` (symmetric, fixed Cook 2019 counts, fan-in normalised — per the [smoke.py](../../../packages/quantum-nematode/quantumnematode/connectome/smoke.py) precedent).
- `ConnectomePPOBrainConfig` (Pydantic): `connectome_source: Literal["cook_2019_hermaphrodite"]`, `enable_gap_junctions: bool = True`, `chemical_mask_mode: Literal["strict", "soft_prior"] = "strict"`, plus PPO hyperparameters mirroring `MLPPPOBrainConfig`.
- Sensor projection: food-chemotaxis input → ASE/AWC/AWA sensory neurons (canonical Bargmann-lab klinotaxis pathway; cross-references `validate_known_pathways` in [connectome/validate.py](../../../packages/quantum-nematode/quantumnematode/connectome/validate.py)). Motor readout: VB/DB/VA/DA motor classes aggregated to the 4-action `DEFAULT_ACTIONS` set ([brain/actions.py:20](../../../packages/quantum-nematode/quantumnematode/brain/actions.py)). This is the T2 default; the choice itself is a T4-scope ablation per [phase6-tracking/tasks.md T4.0c](../phase6-tracking/tasks.md).
- Strict-mask is the T2 default per Decision 7. Soft-prior is implemented but not used in T2's Gate 1 evaluation (it's a T4-scope ablation).

### 3. Klinotaxis Smoke Config (configs/)

- `configs/scenarios/foraging/connectome_ppo_klinotaxis.yml` — minimal-edit fork of [lstmppo_small_klinotaxis.yml](../../../configs/scenarios/foraging/lstmppo_small_klinotaxis.yml): keeps STAM, keeps `chemotaxis_mode: klinotaxis`, swaps brain name to `connectomeppo` and brain config block to `ConnectomePPOBrainConfig` defaults. Drives the Gate 1 G1.c training-signal check.

### 4. Tests

New tests under `packages/quantum-nematode/tests/quantumnematode_tests/`:

**First PR (registry capability):**

- `brain/arch/test_registry.py` — registry round-trip (register / get / list); duplicate-name detection; unknown-name error path; `instantiate_brain` round-trip with wrong-config-type rejection; `family_members` partitioning; `get_all_registrations` returns-a-copy invariant. Uses an autouse fixture to snapshot-and-clear the module-level registry around each test.
- `brain/arch/test_registry_enum_consistency.py` — separate module (outside the autouse-fixture scope) that asserts `BrainType` enum string values equal the registered names at import time. Confirms the production-state invariant.
- `brain/arch/test_registration_equivalence.py` — in-process equivalence test for MLPPPO + LSTMPPO. Instantiates each brain twice (direct constructor + via `instantiate_brain`) under pinned seeds, asserts byte-identical parameter tensors + matching chosen-action lists with `< 1e-12` probability divergence. Replaces the original pickle-fixture pre/post-refactor capture per the scope decision in [design.md Decision 5](design.md).
- `utils/test_config_loader_yaml_compat.py` — parametrised over every YAML under [configs/scenarios/](../../../configs/scenarios/); asserts each loads to a valid brain config via `configure_brain(load_simulation_config(...))`. 189/191 pass on the post-refactor branch; 2 xfailed (`qrc_small_oracle.yml` and `qsnnreinforce_small_oracle.yml`) document pre-existing stale-YAML breakage on main.

**Follow-up PR (`connectome-ppo-brain` capability):**

- `brain/arch/test_topology_rule_protocols.py` — `BrainTopology` and `LearningRule` Protocol conformance for `ConnectomePPOBrain`.
- `brain/arch/test_connectome_ppo.py` — `ConnectomePPOBrain` construction, forward-pass shape + finiteness, strict-mask invariant (no learnable weight along a non-existent edge), gap-junction weights frozen across PPO updates, sensor/motor projection sanity.

### 5. Plugin-Developer Documentation

- `docs/architecture/plugin-developer-guide.md` — "how to add a new architecture family" walkthrough: decorate the Brain class with `@register_brain`, define a Pydantic config, optionally factor into `Topology` + `Rule`, write tests. Files-touched count called out explicitly (target ≤ 6 files per Gate 2 G2.b). The hypothetical-addition exercise per [phase6-tracking/tasks.md T2.7](../phase6-tracking/tasks.md) lives as a worked example in this guide.

### 6. T2 Logbook + Gate 1 Decision

- `docs/experiments/logbooks/023-architecture-plugin-interface.md` — implementation summary, regression-bar findings (MLPPPO + LSTMPPO byte-equivalence proof, the 17-architecture numerical-equivalence table), Gate 1 G1.c paired-control measurement (frozen-random-weights baseline + PPO learning run, both 100 episodes same seed), Gate 1 decision (GO/PIVOT/STOP) per [phase6-tracking/design.md § Decision 6 § Gate 1](../phase6-tracking/design.md), plus the T2↔T3 handshake note (T3 fixes ASH/ADL nociception — the corrected sensor needs the plugin interface in place).

### 7. Tracking + Roadmap Updates

- `openspec/changes/phase6-tracking/tasks.md` — tick T2.1 through T2.10; add the Gate 1 decision link.
- `docs/roadmap.md` Phase 6 Tranche Tracker — T2 row flipped to `🟡 in progress` on first commit, `✅ complete` on PR merge.

## Capabilities

### New Capabilities

- `connectome-ppo-brain`: PPO-trainable connectome-constrained brain using the Cook 2019 *C. elegans* connectome as fixed topology. Chemical synapses are strict-masked with PPO-learnable scalar weights along the wild-type adjacency; gap junctions carry fixed Cook 2019 counts as non-learnable bidirectional couplings (fan-in normalised). Sensor projection maps env chemotaxis/proprioception inputs to canonical Bargmann-lab sensory-neuron pathways; motor readout aggregates VB/DB/VA/DA motor-class activations to the discrete 4-action set. Soft-prior mode is implemented as an opt-in for T4-scope ablation.

### Modified Capabilities

- `brain-architecture`: dispatcher refactor only in this change. `setup_brain_model()` collapses to a thin shim that builds per-architecture infrastructure kwargs and delegates to `instantiate_brain(name, config, **infra_kwargs)`, which is backed by a decorator-registration registry. `BRAIN_CONFIG_MAP` becomes registry-derived (rather than a hand-maintained 19-entry literal). `BrainType` migrates from `Enum` to `StrEnum` so member values are first-class strings comparable to YAML brain-name keys; the family-set aliases (`QUANTUM_BRAIN_TYPES` / `CLASSICAL_BRAIN_TYPES` / `SPIKING_BRAIN_TYPES`) are derived from the registry's family tags via lazy module `__getattr__`. The 19 existing brain modules are migrated decorator-only — each picks up an `@register_brain(...)` decorator and no other code changes. New `BrainTopology` + `LearningRule` Protocols are introduced as forward-compat scaffolding for the upcoming `ConnectomePPOBrain`; the existing 19 brains do **not** yet factor into topology + rule references — that extraction is a planned future refactor deferred to a follow-up change. The external `Brain` Protocol surface (`run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy` at [brain/arch/\_brain.py:346-368](../../../packages/quantum-nematode/quantumnematode/brain/arch/_brain.py)) is unchanged. Adding a new architecture goes from ~5 files touched + per-arch branches in dispatcher + loader, to ≤ 6 files touched + zero branches in the simulation/training loops (a per-arch infrastructure-kwargs branch in `_build_infra_kwargs` is sometimes still required for architectures whose `__init__` signatures diverge from the default shape; this is declarative dict-literal work, not control flow).

## Impact

**Code (first PR — registry capability):**

- `packages/quantum-nematode/quantumnematode/brain/arch/_registry.py` — new registry module
- `packages/quantum-nematode/quantumnematode/brain/arch/_topology.py` — new `BrainTopology` Protocol (forward-compat scaffolding)
- `packages/quantum-nematode/quantumnematode/brain/arch/_rule.py` — new `LearningRule` Protocol + `RuleStepReport` dataclass (forward-compat scaffolding)
- `packages/quantum-nematode/quantumnematode/brain/arch/__init__.py` — re-exports updated for registry consumers; `assert_registry_matches_enum()` invoked at import
- `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` — dispatcher collapsed (459 LOC → ~170 LOC; signature preserved)
- `packages/quantum-nematode/quantumnematode/utils/config_loader.py` — `BRAIN_CONFIG_MAP` derived from registry
- `packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py` — `BrainType` migrated `Enum → StrEnum`; family sets derived from registry via lazy module `__getattr__`
- 19 files at `packages/quantum-nematode/quantumnematode/brain/arch/<name>.py` — `@register_brain(...)` decorator added above the brain class + 2 import lines each; no other changes

**Code (follow-up PR — `connectome-ppo-brain` capability):**

- `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py` — new `ConnectomePPOBrain` + `ConnectomePPOBrainConfig` + `ConnectomeTopology`
- `packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py` — add `BrainType.CONNECTOMEPPO = "connectomeppo"` enum member (paired with the registration so the consistency check never sees a transient mismatch)
- `packages/quantum-nematode/quantumnematode/brain/arch/__init__.py` — re-export `ConnectomePPOBrain` + `ConnectomePPOBrainConfig`

**Configs:**

- `configs/scenarios/foraging/connectome_ppo_klinotaxis.yml` — new smoke config for the Gate 1 G1.c training-signal check

**Dependencies:** None. The registry + Protocol work is pure-stdlib + existing torch/numpy. ConnectomePPOBrain consumes the T1 `connectome` subpackage (already shipped).

**Tests (first PR — registry capability):**

- `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_registry.py` — new
- `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_registry_enum_consistency.py` — new
- `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_registration_equivalence.py` — new (MLPPPO + LSTMPPO in-process equivalence)
- `packages/quantum-nematode/tests/quantumnematode_tests/utils/test_config_loader_yaml_compat.py` — new (scenario-YAML registry-load regression)

**Tests (follow-up PR — `connectome-ppo-brain` capability):**

- `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_topology_rule_protocols.py` — new
- `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_connectome_ppo.py` — new

**Docs:**

- `docs/architecture/plugin-developer-guide.md` — new plugin-developer walkthrough
- `docs/experiments/logbooks/023-architecture-plugin-interface.md` — T2 logbook + Gate 1 decision
- `openspec/changes/phase6-tracking/tasks.md` — T2.x sub-tasks ticked + Gate 1 decision link
- `docs/roadmap.md` Phase 6 Tranche Tracker — T2 row updated

**Git:** None. No new LFS rules, no submodule changes.

## Untouched (T2 boundary — these are T3+ territory)

- `packages/quantum-nematode/quantumnematode/brain/arch/_brain.py` — `Brain` Protocol surface (`run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy`); reaffirmed by [phase6-tracking/tasks.md § Tranche 2](../phase6-tracking/tasks.md) — "the `Brain` Protocol surface ... does NOT change."
- `packages/quantum-nematode/quantumnematode/brain/actions.py` — discrete `DEFAULT_ACTIONS` 4-action set; T5 replaces with continuous-action heads
- `packages/quantum-nematode/quantumnematode/env/env.py` and the sensory pipeline — env / sensor / ASH/ADL wiring; T3 fixes contact-based nociception, T5 does continuous-2D coordinates, T6 does Rung 2 diffusion
- `packages/quantum-nematode/quantumnematode/connectome/` — T1's L0 substrate is consumed read-only; no T1 modifications

## Breaking Changes

None at the user-facing API level. The `Brain` Protocol surface is unchanged. `BrainType` enum membership is preserved (all 19 values stay). `setup_brain_model()` retains its public signature. YAML configs continue to work without modification (brain names + config blocks parse identically).

Internal-only restructuring: 19 brain modules pick up a `@register_brain(...)` decorator + extract topology/rule references. Any out-of-tree code subclassing one of the 19 Brain implementations should still work, but the migration regression bar tests (MLPPPO + LSTMPPO byte-equivalence, 17-architecture numerical equivalence) are the contract.

## Backward Compatibility

Full at the user-facing level. All existing YAML configs, training scripts, and saved-weight artefacts work without modification. The migration regression bar tests are the proof: MLPPPO + LSTMPPO produce byte-identical training trajectories pre-/post-refactor on at least one smoke config each; the other 17 produce `np.allclose(rtol=0, atol=1e-7)` parameter tensors after a 5-step smoke training. Any architecture that fails its tolerance bar blocks the change.
