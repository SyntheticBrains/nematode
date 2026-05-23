## Why

Phase 6 Tranche 2 (T2, L1 layer) refactors the brain-architecture dispatcher into a pluggable registry interface and introduces the first connectome-constrained brain. Without T2, the headline Phase 6 platform claim — *first closed-loop learning + evolution on the real *C. elegans* connectome with a pluggable architecture interface* — has a substrate (T1 just shipped) but no plugin contract to consume it through, and no PPO-trainable connectome architecture wired into the env.

T2 closes **Gate 1**. Three coupled outcomes ride on this change:

1. **L1 plugin parity** — the 19-elif `setup_brain_model()` dispatcher at [packages/quantum-nematode/quantumnematode/utils/brain_factory.py](../../../packages/quantum-nematode/quantumnematode/utils/brain_factory.py) (459 LOC, one branch per `BrainType`) is replaced with a decorator-registration registry. Adding a new architecture must touch ≤ 6 files with no per-architecture branches in simulation/training loops (the Gate 2 G2.b + G2.c criteria T5 will verify against).
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

Two new module files under `packages/quantum-nematode/quantumnematode/brain/arch/`:

- `_registry.py` — `@register_brain(name, config_cls)` decorator + `instantiate_brain(name, config, **kwargs) -> Brain` + registry inspection helpers. Each architecture's `brain/arch/<name>.py` self-registers at import time. Single source of truth for the (name, config_cls, brain_cls) mapping.
- `_topology.py` + `_rule.py` — `BrainTopology` and `LearningRule` Protocols. `BrainTopology` exposes `n_inputs` / `n_outputs` / `n_hidden` / `forward(x)` / optional `apply_weight_mask(weights)`. `LearningRule` exposes optimiser, value head, replay buffer, gradient computation. The 19 existing brains' `__init__` extracts a `topology` + `rule` reference internally; the external Brain Protocol surface ([brain/arch/\_brain.py:346-368](../../../packages/quantum-nematode/quantumnematode/brain/arch/_brain.py)) does NOT change.

Two modified module files:

- `utils/brain_factory.py` — the 19-elif `setup_brain_model()` dispatcher collapses to a single `instantiate_brain(...)` call. Backward-compatible signature retained (callers still pass `BrainType` + config); internal dispatch goes through the registry.
- `utils/config_loader.py` — `BRAIN_CONFIG_MAP` (currently 19 hand-maintained entries) becomes a registry lookup. `_resolve_brain_config` helper at [config_loader.py:153-216](../../../packages/quantum-nematode/quantumnematode/utils/config_loader.py) is unchanged (still useful for raw-dict → Pydantic resolution).

One modified module — minimal-edit per architecture:

- Each of the 19 `brain/arch/<name>.py` files gets a `@register_brain("<name>", <Name>BrainConfig)` decorator on the Brain class. The `BrainType` enum at [brain/arch/dtypes.py](../../../packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py) stays for typed dispatch in evolution-framework + predator-brain callers, but its membership becomes registry-derived rather than hand-maintained.

### 2. Connectome-Constrained Brain (`connectome-ppo-brain` NEW)

New module `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py`:

- `ConnectomePPOBrain` class implementing the `Brain` Protocol. Topology built from `Connectome` (T1 data model) via the T1↔T2 API sketch iteration pattern: chemical-synapse strict-mask matrix from `Connectome.chemical_synapses` (302 × 302 sparse, weights PPO-learnable along existing edges, non-existent edges pinned to zero), gap-junction matrix from `Connectome.gap_junctions` (symmetric, fixed Cook 2019 counts, fan-in normalised — per the [smoke.py](../../../packages/quantum-nematode/quantumnematode/connectome/smoke.py) precedent).
- `ConnectomePPOBrainConfig` (Pydantic): `connectome_source: Literal["cook_2019_hermaphrodite"]`, `enable_gap_junctions: bool = True`, `chemical_mask_mode: Literal["strict", "soft_prior"] = "strict"`, plus PPO hyperparameters mirroring `MLPPPOBrainConfig`.
- Sensor projection: food-chemotaxis input → ASE/AWC/AWA sensory neurons (canonical Bargmann-lab klinotaxis pathway; cross-references `validate_known_pathways` in [connectome/validate.py](../../../packages/quantum-nematode/quantumnematode/connectome/validate.py)). Motor readout: VB/DB/VA/DA motor classes aggregated to the 4-action `DEFAULT_ACTIONS` set ([brain/actions.py:8-31](../../../packages/quantum-nematode/quantumnematode/brain/actions.py)). This is the T2 default; the choice itself is a T4-scope ablation per [phase6-tracking/tasks.md T4.0c](../phase6-tracking/tasks.md).
- Strict-mask is the T2 default per Decision 7. Soft-prior is implemented but not used in T2's Gate 1 evaluation (it's a T4-scope ablation).

### 3. Klinotaxis Smoke Config (configs/)

- `configs/scenarios/foraging/connectome_ppo_klinotaxis.yml` — minimal-edit fork of [lstmppo_small_klinotaxis.yml](../../../configs/scenarios/foraging/lstmppo_small_klinotaxis.yml): keeps STAM, keeps `chemotaxis_mode: klinotaxis`, swaps brain name to `connectomeppo` and brain config block to `ConnectomePPOBrainConfig` defaults. Drives the Gate 1 G1.c training-signal check.

### 4. Tests

New tests under `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/`:

- `test_registry.py` — registry round-trip (register → instantiate → assert type); duplicate-name detection; unknown-name error path; the 19 existing architectures self-register at import time.
- `test_topology_rule_protocols.py` — `BrainTopology` and `LearningRule` Protocol conformance for MLPPPO + LSTMPPO + ConnectomePPO (the three Gate 1 brains).
- `test_connectome_ppo.py` — `ConnectomePPOBrain` construction, forward-pass shape + finiteness (mirrors connectome smoke test), strict-mask invariant (no learnable weight along a non-existent edge), gap-junction weights frozen across PPO updates, sensor/motor projection sanity.
- `test_migration_byte_equivalence.py` — MLPPPO + LSTMPPO byte-equivalence pre-/post-refactor on one smoke config each. Mirrors [test_predator_brain_byte_equivalence.py](../../../packages/quantum-nematode/tests/quantumnematode_tests/env/test_predator_brain_byte_equivalence.py): pin RNG seeds, capture trajectory + parameter tensors, assert exact equality.
- `test_migration_numerical_equivalence.py` — the other 17 architectures: `np.allclose(rtol=0, atol=1e-7)` on parameter tensors after a 5-step smoke training. Per-architecture pass/fail recorded; any architecture exceeding the tolerance fails this test with a captured tensor diff.

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

- `brain-architecture`: REGISTRY refactor. `setup_brain_model()` dispatcher collapses to a single `instantiate_brain(name, config)` call backed by a decorator-registration registry; `BRAIN_CONFIG_MAP` becomes registry-derived; `BrainType` enum membership is derived from the registry rather than hand-maintained. New `BrainTopology` + `LearningRule` Protocols factor the existing 19 architectures into topology + rule references internally. The external `Brain` Protocol surface (`run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy` at [brain/arch/\_brain.py:346-368](../../../packages/quantum-nematode/quantumnematode/brain/arch/_brain.py)) is unchanged. Adding a new architecture goes from ~5 files touched + per-arch branches in dispatcher + loader, to ≤ 6 files touched + zero branches.

## Impact

**Code:**

- `packages/quantum-nematode/quantumnematode/brain/arch/_registry.py` — new registry module
- `packages/quantum-nematode/quantumnematode/brain/arch/_topology.py` — new `BrainTopology` Protocol
- `packages/quantum-nematode/quantumnematode/brain/arch/_rule.py` — new `LearningRule` Protocol
- `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py` — new `ConnectomePPOBrain` + config
- `packages/quantum-nematode/quantumnematode/brain/arch/__init__.py` — re-exports updated for registry consumers
- `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` — dispatcher collapsed (459 LOC → expected ~50 LOC)
- `packages/quantum-nematode/quantumnematode/utils/config_loader.py` — `BRAIN_CONFIG_MAP` becomes registry lookup
- `packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py` — `BrainType` membership registry-derived (enum kept for typed callers)
- 19 files at `packages/quantum-nematode/quantumnematode/brain/arch/<name>.py` — `@register_brain(...)` decorator added; minimal internal refactor to extract topology + rule references

**Configs:**

- `configs/scenarios/foraging/connectome_ppo_klinotaxis.yml` — new smoke config for the Gate 1 G1.c training-signal check

**Dependencies:** None. The registry + Protocol work is pure-stdlib + existing torch/numpy. ConnectomePPOBrain consumes the T1 `connectome` subpackage (already shipped).

**Tests:**

- `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_registry.py` — new
- `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_topology_rule_protocols.py` — new
- `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_connectome_ppo.py` — new
- `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_migration_byte_equivalence.py` — new (MLPPPO + LSTMPPO regression bar)
- `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_migration_numerical_equivalence.py` — new (17 other architectures, `atol=1e-7`)

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
