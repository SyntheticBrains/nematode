## Context

Phase 6 Tranche 2 (T2) refactors the brain-architecture dispatcher into a registry pattern and introduces the first connectome-constrained brain. Current state, from the [T2.1 audit](../../../packages/quantum-nematode/quantumnematode/utils/brain_factory.py):

- `setup_brain_model()` is a 459-LOC function with 19 branches (1 `if` + 18 `elif`) dispatching on `brain_type == BrainType.<X>`. Every branch repeats a type-check pattern on the brain config and a per-architecture instantiation.
- `BRAIN_CONFIG_MAP` at [config_loader.py:130-150](../../../packages/quantum-nematode/quantumnematode/utils/config_loader.py) is a hand-maintained dict mapping 19 YAML brain names to their Pydantic config classes.
- The `BrainType` enum at [brain/arch/dtypes.py:9-37](../../../packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py) is the canonical typed dispatch key, with 19 hand-maintained members + companion type aliases (`BRAIN_TYPES`, `QUANTUM_BRAIN_TYPES`, `CLASSICAL_BRAIN_TYPES`, `SPIKING_BRAIN_TYPES`).
- The 19 brain modules each pair a *topology* (MLP, LSTM, variational quantum circuit, reservoir, spiking, etc.) with a *learning rule* (PPO, REINFORCE, DQN, Q-learning) in a fused implementation. There is no shared topology abstraction that the T1 connectome data model can be substituted into.
- The external `Brain` Protocol at [brain/arch/\_brain.py:346-368](../../../packages/quantum-nematode/quantumnematode/brain/arch/_brain.py) — `run_brain` / `update_memory` / `prepare_episode` / `post_process_episode` / `copy` + `history_data` / `latest_data` attributes — is the contract every brain conforms to, and is the right plugin interface. It does not need to change.
- The T1 connectome data model ([`connectome/__init__.py`](../../../packages/quantum-nematode/quantumnematode/connectome/__init__.py)) ships `Connectome` / `Neuron` / `ChemicalSynapse` / `GapJunction` types via the T1↔T2 API sketch in [logbook 022](../../../docs/experiments/logbooks/022-connectome-substrate.md). The sketch is the load-bearing handshake — T2's `ConnectomePPOBrain` consumes it through the documented iteration patterns.

Phase 5 M1's PredatorBrain refactor ([logbook 016](../../../docs/experiments/logbooks/016-predator-brain-refactor.md)) established the regression-bar precedent: 23 byte-equivalence unit tests + 80/80 metric-cell deltas exactly 0.0 across a registry-style refactor. T2's migration of 19 architectures is comparable in shape but larger in scope.

Stakeholders / downstream consumers:

- **Evolution framework** ([scripts/run_evolution.py](../../../scripts/run_evolution.py), [evolution/](../../../packages/quantum-nematode/quantumnematode/evolution/)) consumes `BrainType` via `instantiate_brain_from_sim_config()` — must continue to work without modification.
- **PredatorBrain factory** consumes `BrainType` for typed dispatch — must continue to work.
- **YAML config consumers** (every scenario under [configs/scenarios/](../../../configs/scenarios/)) — all existing YAML must parse + load without modification.
- **T3 (corrected ASH/ADL nociception)** ships immediately after T2 and will use the new plugin interface to register the corrected sensor pipeline.
- **T4 (L2 first pass)** sweeps the four MUST architectures × three behaviours on the grid; depends on `ConnectomePPOBrain` being registered.
- **T8 (NEAT topology search)** relies on the topology/rule factoring — NEAT-evolved topologies must plug into the same `LearningRule` (PPO) as fixed-topology MLP/LSTM/connectome.

## Goals / Non-Goals

**Goals:**

- Replace the 19-elif dispatcher with a registry pattern such that adding a new architecture touches ≤ 6 files and introduces zero per-architecture branches in `setup_brain_model()` / `BRAIN_CONFIG_MAP` / simulation loop / training loop. This is the Gate 2 G2.b + G2.c criteria T5 will verify against on a hypothetical new arch.
- Factor *topology* out from *learning rule* so the T1 connectome data model is consumable as a `BrainTopology` by the existing PPO `LearningRule` (and, in T8, by NEAT-evolved topologies through the same `LearningRule`).
- Ship `ConnectomePPOBrain` (chemical-synapse strict-mask + fixed gap-junction weights per Cook 2019 counts) as the first PPO-trainable architecture over the wild-type connectome. Wire it through the existing grid env + klinotaxis behaviour for the Gate 1 G1.c training-signal check.
- Establish the migration regression bar: byte-equivalence for MLPPPO + LSTMPPO (G1.d MUST), `np.allclose(rtol=0, atol=1e-7)` parameter-tensor tolerance after a 5-step smoke training for the other 17.
- Document the plugin-developer experience: a self-contained walkthrough of "how to add a new architecture family" with the ≤ 6 files rule called out explicitly.
- Close Gate 1 with a written GO / PIVOT / STOP decision in [logbook 023](../../../docs/experiments/logbooks/) evaluated against the four G1.a–G1.d criteria in [phase6-tracking/design.md § Decision 6](../phase6-tracking/design.md).

**Non-Goals:**

- **Brain Protocol surface changes.** The five Protocol methods + two attributes are the right contract and stay untouched.
- **Continuous-action heads.** Discrete `DEFAULT_ACTIONS` (4 actions) stays as the action API. T5 introduces continuous-action heads; T2's connectome brain produces logits over the 4-action set via aggregated motor-neuron readout.
- **Continuous-2D coordinates / Rung 2 gradients / corrected ASH/ADL nociception.** All deferred to T3 / T5 / T6 / T7.
- **Soft-prior chemical-mask mode evaluation.** Implemented (so T4 can flip the switch) but Gate 1 G1.c evaluates strict-mask only.
- **Real-worm validation.** Deferred to T7.
- **Sensor projection + motor readout sweep.** T2 picks a sensible default (see Decision 4) and documents it; the ablation across alternative projections is a T4-scope deliverable per [phase6-tracking/tasks.md T4.0c](../phase6-tracking/tasks.md).
- **Removing the `BrainType` enum.** Out-of-tree code (evolution + predator) consumes the enum directly. T2 keeps the enum but makes its membership registry-derived (decorator at registration time appends to the enum).
- **NEAT-evolved topology.** T8 builds on T2's topology/rule factoring; T2 only ships the factoring + a fixed-topology connectome consumer.

## Decisions

### Decision 1 — Decorator-registration registry pattern

`packages/quantum-nematode/quantumnematode/brain/arch/_registry.py` exposes:

```python
# Registration side (decorator on each Brain class)
@register_brain(
    name="mlpppo",
    config_cls=MLPPPOBrainConfig,
    brain_type=BrainType.MLP_PPO,  # for backward-compat with enum consumers
    families=("classical",),
)
class MLPPPOBrain: ...

# Lookup side (replaces the 19-elif dispatcher)
brain = instantiate_brain(name="mlpppo", config=cfg, **infra_kwargs)
```

The registry is a private module-level dict; entries are populated at import time when each `brain/arch/<name>.py` module is loaded. `brain/arch/__init__.py` already imports every architecture module at the top level (to re-export the Brain + Config classes); those imports double as the registration trigger. After every architecture module has been imported, `__init__.py` invokes `assert_registry_matches_enum()` so any accidental enum/registry drift fails loudly at import time rather than at first dispatch.

**Alternatives considered:**

- **Config-driven (YAML-based registry)**. Listed in `configs/brains.yml` or similar; loaded at startup. Rejected: loses static analysis on the (name → config_cls → brain_cls) mapping; harder to debug; adds a YAML schema that has to evolve in lockstep with the codebase. No real benefit in a monorepo where every architecture lives in-tree.
- **Python entry-points (`pyproject.toml` `[project.entry-points]`)**. Standard plugin-discovery mechanism. Rejected: adds packaging complexity (the workspace structure already complicates entry-point discovery); decorator pattern serves the same role with less infrastructure; out-of-tree plugins are not in scope for Phase 6.
- **Abstract base class with class-level `name` + `config_cls` attributes**. Considered briefly. Rejected: the existing 19 Brain implementations are heterogeneous (some inherit from `_reservoir_hybrid_base`, some from `_reservoir_lstm_base`, most from nothing). Forcing them under a common ABC requires touching every brain's class hierarchy; decorator-registration is purely additive and leaves inheritance untouched.

### Decision 2 — `BrainTopology` and `LearningRule` Protocols

```python
@runtime_checkable
class BrainTopology(Protocol):
    """Pure structure: how neurons connect and how a forward pass computes outputs.

    Stateful in `weights` (PPO/REINFORCE update these) but topology-stateless —
    `forward()` does not own optimisers, replay buffers, or value heads.
    """
    n_inputs: int
    n_outputs: int
    n_hidden: int

    def forward(self, x: Tensor) -> Tensor: ...

    def apply_weight_mask(self, weights: Tensor) -> Tensor:
        """Project a candidate weight tensor onto the topology's allowed manifold.

        Default implementation: identity (dense topology). Connectome topology
        applies the chemical-synapse strict-mask here. NEAT-evolved topologies
        will apply their per-genome connectivity mask.
        """
        ...


@runtime_checkable
class LearningRule(Protocol):
    """Pure learning algorithm: how a topology's weights are updated from experience.

    Owns the optimiser, value head (if any), replay buffer (if any),
    advantage estimator (if any), gradient clipper (if any).
    """
    def step(self, topology: BrainTopology, batch: ExperienceBatch) -> RuleStepReport: ...
    def reset_episode(self) -> None: ...
```

**Scope decision (implementation-time amendment):** the per-brain topology/rule extraction originally specified here is deferred to a follow-up change. T2 ships the Protocols as forward-compat scaffolding consumed directly by the new `ConnectomePPOBrain`; the existing 19 brains are migrated decorator-only (no `self.topology` / `self.rule` extraction in their `__init__`). Byte-equivalence then becomes trivially preserved because no executing code changes — the decorator is metadata only. The follow-up change can refactor the 19 fused bundles when there is a concrete consumer of the factoring (e.g. T8 NEAT-evolved topology + PPO sharing the same rule with `ConnectomePPOBrain`).

The Protocols still ship under `_topology.py` and `_rule.py` and remain part of the public `brain.arch` surface. `ConnectomePPOBrain` (delivered in the follow-up change) implements `BrainTopology` directly with a non-trivial `apply_weight_mask` (chemical-synapse strict-mask).

**Alternatives considered:**

- **Abstract method on `Brain` Protocol** (e.g. add `topology` + `rule` properties to the Protocol surface). Rejected: changes the Protocol surface that [phase6-tracking/tasks.md § Tranche 2](../phase6-tracking/tasks.md) explicitly says does NOT change.
- **Mixin-style topology classes**. Rejected: forces multiple-inheritance plumbing that the existing 19 brains don't use anywhere else.
- **Compose at registration time** (registry takes `(topology_cls, rule_cls)`, instantiates a generic `Brain` wrapper). Considered for T8 NEAT support. Rejected for T2: too invasive for the migration; would force every brain into a fixed Topology + Rule slot when several existing brains (e.g. `HybridQuantumBrain`, `QRHQLSTMBrain`) have non-trivially composed internals. T2 ships the Protocols + per-brain internal refactor; T8 can compose generically on top if the abstraction holds up.

### Decision 3 — `BrainType` enum membership becomes registry-derived

The 19-member enum at [brain/arch/dtypes.py:9-37](../../../packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py) and its companion type aliases (`BRAIN_TYPES` Literal, `QUANTUM_BRAIN_TYPES` set, etc.) are consumed by callers outside the brain-architecture subsystem (evolution framework + predator factory). Removing them is out of scope. The compromise:

- `BrainType` migrates from `Enum` to `StrEnum` (already encodes string values like `MLP_PPO = "mlpppo"`; the migration makes `BrainType.MLP_PPO == "mlpppo"` evaluate True). The 19 + 1 (connectome) members remain hand-declared (Python enum members must be declared at class-definition time). The `Enum` → `StrEnum` change is backward-compatible for callers using `BrainType.X.value` and additive for callers wanting string-equality semantics.
- `BRAIN_TYPES` Literal and the `QUANTUM_BRAIN_TYPES` / `CLASSICAL_BRAIN_TYPES` / `SPIKING_BRAIN_TYPES` sets are derived from registry metadata at module-init time, replacing the current hand-maintained literals. The `families=("classical",)` / `("quantum",)` / `("spiking",)` parameter on `@register_brain(...)` is the metadata source. An architecture may belong to multiple families (e.g. `QSNNReinforce` is in both `"quantum"` and `"spiking"`); the family sets are the union of all registrations carrying that family tag, which is a broadening from the current single-family-set semantics that the migration documents explicitly.
- A startup-time consistency check asserts the registry has exactly the same set of names as `BrainType` enum string values (since `BrainType` is now a `StrEnum`, the values ARE strings). Mismatch raises at import — fails loudly if someone adds an enum member without a registration or vice versa.

**Alternatives considered:**

- **Drop the `BrainType` enum entirely**, replace string keys throughout. Rejected: ~20 call sites in evolution + predator code consume the enum; an enum-drop is high-blast-radius for low value when the registry-derived approach gives us the same files-touched count.
- **Make `BrainType` dynamic** (`enum` library `_create_` mechanism). Rejected: too clever; breaks static analysis and IDE autocomplete.

### Decision 4 — `ConnectomePPOBrain` topology specification

The `ConnectomeTopology` consumed by `ConnectomePPOBrain` is constructed from a `Connectome` instance per the T1↔T2 API sketch iteration patterns:

**Chemical-synapse layer** (PPO-learnable along existing edges):

- 302 × 302 weight tensor `W_chem`. Initialised from a `MLPPPOBrainConfig`-style scheme (Xavier/Kaiming over the non-masked positions; documented in `design.md` Decision 4 of the OpenSpec change).
- Boolean strict-mask `M_chem` (302 × 302): `M_chem[pre_idx, post_idx] = True` iff `ChemicalSynapse(pre=p, post=q)` exists in `Connectome.chemical_synapses`.
- `apply_weight_mask(W)` returns `W * M_chem` — applied after every PPO gradient step so non-existent edges stay zero.

**Gap-junction layer** (non-learnable, fan-in normalised):

- 302 × 302 symmetric weight tensor `G_gap`. Built from `Connectome.gap_junctions` with `G_gap[a, b] = G_gap[b, a] = float(gj.weight)` (Cook 2019 counts as fixed weights per [phase6-tracking/design.md § Decision 7](../phase6-tracking/design.md)).
- Fan-in normalisation: each row scaled by `1 / max(1, sum(G_gap[i, :]))` so total gap-junction input per neuron is bounded. Same scheme as the T1 [smoke.py](../../../packages/quantum-nematode/quantumnematode/connectome/smoke.py).
- `G_gap.requires_grad = False`; never updated by the PPO rule.

**Sensor projection** (env → connectome input):

- `food_chemotaxis` env input → `ASE` (left + right) + `AWC` (left + right) + `AWA` (left + right) sensory neurons (canonical Bargmann-lab klinotaxis sensory pathway). The 1D chemotaxis gradient signal is broadcast to all six sensory neurons identically; differential left-right encoding is added in T5/T6 when continuous-2D coordinates make it meaningful.
- `proprioception` env input → `AVA` + `AVB` command interneurons (driven by Bargmann-lab anatomical convention; the actual klinotaxis pathway routes proprioception through interneurons before reaching motor neurons).
- Sensor inputs are placed onto the connectome via additive injection on the sensory neurons' activation vector, scaled by a per-input learnable gain (a single scalar each, registered as PPO-learnable parameters but separate from the chemical-synapse weight matrix).

**Motor readout** (connectome → 4-action logits):

- Motor neurons by class: VB / DB / VA / DA. Activations aggregated by class via mean pooling → 4 motor-class activations.
- Projection to the 4 `DEFAULT_ACTIONS` ([FORWARD, LEFT, RIGHT, STAY]) via a learnable 4×4 readout matrix. The readout layer is part of the PPO-learnable parameter set; it's external to the chemical-synapse matrix per Decision 7 (the strict-mask claim applies to chemical synapses, not to sensor gains or motor readout — both are necessary plumbing).

**Forward pass** (single step):

```text
sensory_input = sensor_projection(env_obs)           # 302-vec, mostly zeros
h = sensory_input
for _ in range(K):                                    # K = 1 in T2 default
    chem_drive = (W_chem * M_chem).T @ h              # 302-vec
    gap_drive  = G_gap.T @ h                          # 302-vec, fixed
    h = activation(chem_drive + gap_drive)            # tanh per neuron
motor_activations = motor_class_pool(h)               # 4-vec
action_logits = readout @ motor_activations           # 4-vec → DEFAULT_ACTIONS
```

K (forward-pass depth) is configurable; T2 default is K = 1 (matches the smoke-test forward-pass shape). T4 may sweep K as part of its sensor-projection ablation.

**Alternatives considered:**

- **No sensor / motor projection layer; require the connectome topology to be self-sufficient**. Rejected: the connectome doesn't natively expose "the env observation" — sensor neurons need an input from outside the connectome to start the forward pass. Some adapter is biologically inevitable.
- **Strict-mask the sensor/motor projection too** (don't allow learnable input gains). Rejected: makes the connectome brain untrainable at K=1 (no learnable parameters reach the chemical synapses through the sensor path). The strict-mask claim is specifically about the chemical-synapse adjacency, not about the env→connectome adapter.
- **Differential left/right chemotaxis encoding at T2 (anticipating continuous-2D)**. Rejected as premature: T2 ships on the discrete grid where left-right gradient differential is not exposed by the env. Add this in T5 when continuous-2D is available.

### Decision 5 — Migration regression-bar declaration

Per [phase6-tracking/tasks.md T2.5](../phase6-tracking/tasks.md):

**Scope decision (implementation-time amendment):** because Decision 2's scope decision reduced the migration to a decorator-only change (no per-brain `__init__` modification), the pickle-fixture pre/post-refactor capture became unnecessary. By construction no executing code moves; byte-equivalence reduces to "the registry-instantiated brain and the directly-constructed brain produce identical results under pinned seeds in the same process." That is what [test_registration_equivalence.py](../../../packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_registration_equivalence.py) verifies.

**MLPPPO + LSTMPPO** (Gate 1 G1.d MUST): in-process equivalence test.

- Instantiate the brain twice in the same Python process: once via `MLPPPOBrain(config=cfg, device=DeviceType.CPU)` (direct), once via `instantiate_brain("mlpppo", cfg, device=DeviceType.CPU)` (registry).
- Pin both `cfg.seed` and the global `torch` + `numpy` RNG before each forward pass.
- Assert `torch.equal` on every actor / critic parameter tensor + (LSTM only) the recurrent module's weights.
- Assert the chosen-action list from `run_brain(...)` matches with action-probability divergence `< 1e-12` per action.
- Failure mode: any divergence fails the test and blocks the change.

**Other 17 architectures** (G1.d not required): no explicit numerical-equivalence test ships. The migration is purely additive (a metadata-only decorator above the class declaration), and the pre-existing per-architecture test suites under `tests/.../brain/arch/test_<name>.py` are the implicit contract — they continued to pass byte-for-byte across the migration commit (1061 brain/arch tests + the full 3245-test suite both green).

**Alternatives considered (no longer load-bearing post-scope-decision):**

- **Pickle-fixture pre/post-refactor capture** — the original design here. Made redundant by the scope decision: nothing moves, so there is nothing for a pre/post-refactor comparison to surface that the in-process equivalence test does not already catch with a smaller blast radius.
- **Parametrised numerical-equivalence test over all 17 non-MUST architectures** — also made redundant. If a brain's behaviour changed under the migration, its existing per-architecture test suite would fail. None did.
- **Byte-equivalence on all 19**. Maximally rigorous; not pursued because the scope decision shifted the migration to decorator-only, removing the need for the rigour.

### Decision 6 — Gate 1 G1.c paired-control procedure

Per [phase6-tracking/design.md § Decision 6 § Gate 1](../phase6-tracking/design.md):

**Smoke config**: `configs/scenarios/foraging/connectome_ppo_klinotaxis.yml`. Forked from [lstmppo_small_klinotaxis.yml](../../../configs/scenarios/foraging/lstmppo_small_klinotaxis.yml) with brain swapped to `connectomeppo`. Keeps STAM, keeps `chemotaxis_mode: klinotaxis`, keeps the same env/sensing/reward shape so the control comparison is apples-to-apples.

**Episode budget**: 100 episodes (the Gate 1 floor). Single seed for the smoke decision (G1.c is a smoke check, not a statistical one; T4 runs the full n ≥ 4 seeds).

**Learning run**: standard PPO config (`learning_rate=3e-4`, batch-style hyperparameters mirrored from MLPPPO).

**Frozen-random-weights control**: same brain config except PPO updates disabled. Implementation strategy:

- A `freeze_updates: bool = False` flag on `ConnectomePPOBrainConfig`. When True, the PPO rule's `step()` is a no-op (gradient computation skipped; optimiser never invoked). Random initial weights persist across all 100 episodes.
- Same RNG seed as the learning run (so the action sampling stream is identical given the same logits).

**Pass conditions** (all required for Gate 1 G1.c GO):

1. No NaNs or Infs in any per-step logit or parameter tensor across all 100 episodes (`torch.isfinite(...).all()` per step).
2. Last-25 mean episode return (learning run) ≥ 1.10 × last-25 mean episode return (frozen control). **The 10% margin is the load-bearing claim that PPO weight tuning is adding signal over the same connectome topology under random weights.**
3. Last-25 mean episode return (learning run) > first-25 mean episode return (learning run). **Anti-collapse check: monotonic improvement signal across the run.**

**Failure-mode handling** per [phase6-tracking/design.md § Decision 6 § Gate 1 STOP/PIVOT triggers](../phase6-tracking/design.md):

- G1.c fails on the 302-neuron full connectome → diagnostic sequence (topology density check, reward-shaping check, RNG-seed sweep). If diagnostic doesn't resolve → hand-curated-subset PIVOT (sensory-interneuron-motor subgraph for klinotaxis, ~50-100 neurons; Gate 1 re-evaluates against the subset).
- G1.c fails on the subset too → STOP (publishable substrate-engineering negative result; Phase 7 L4 inherits the substrate question).

### Decision 7 — Migration sequencing (which brain registers when)

The 19-architecture migration runs in a deliberate order to surface problems early:

1. **MLPPPO first** (sub-task in `tasks.md`). The shortest forward + learning loop; the G1.d MUST that the rest of the Gate 1 decision depends on. If byte-equivalence breaks here, the registry pattern itself is wrong and we re-design before touching the other 18.
2. **LSTMPPO second**. The other G1.d MUST; adds recurrent-state byte-equivalence to the bar.
3. **Seven classical / spiking brains next** (`MLPReinforce`, `MLPDQN`, `QRC`, `CRH`, `CRHQLSTM`, `HybridClassical`, `SpikingReinforce`). Numerical-equivalence at `atol=1e-7`. QRC is a classical-reservoir arch despite the Q prefix (no QPU shot variance); SpikingReinforce is non-quantum spiking.
4. **Ten quantum + spiking-quantum brains last** (`QVarCircuit`, `QQLearning`, `QRH`, `QEF`, `QSNNReinforce`, `QSNNPPO`, `HybridQuantum`, `HybridQuantumCortex`, `QLIFLSTM`, `QRHQLSTM`). Most likely to surface RNG-reassociation drift; if any fails `atol=1e-7`, the diagnosis is contained to that arch.
5. **`ConnectomePPOBrain` last** of all. The first new brain registered via the new interface; its existence is also the Gate 1 G1.b check (registry instantiates connectome + MLPPPO through the same code path).

**Alternatives considered**:

- **All-at-once migration**. Rejected: harder to diagnose which arch broke the byte-equivalence bar.
- **Connectome brain first** (proves the interface before migrating). Rejected: the regression bar on MLPPPO is the contract that holds the registry pattern accountable; building on top of an un-vetted registry pattern multiplies risk.

### Decision 8 — Forward-compatibility for NEAT topology (T8 hook)

T8 (NEAT topology search on the upgraded substrate) will need to register NEAT-evolved topologies through the same plugin interface. T2 anticipates this with two forward-compat hooks:

- `BrainTopology` Protocol exposes `apply_weight_mask(weights)`. NEAT-evolved topologies will subclass / implement this to project candidate weights onto their per-genome connectivity mask. T2 implements the identity case (dense topology) and the strict-mask case (`ConnectomeTopology`); T8 inherits the abstraction for free.
- `@register_brain(...)` decorator accepts `topology_factory` and `rule_factory` parameters (default: the brain's own internal `__init__`). T8 will register a `NEATWeightsBrain` and a future `NEATTopologyBrain` that consume the same `PPORule` factory as `MLPPPOBrain` / `ConnectomePPOBrain`. T2 wires the parameters but does not need to register a NEAT brain itself.

These two hooks are purely additive and cost essentially nothing to T2's scope.

## Risks / Trade-offs

\[**Risk A: Byte-equivalence breaks on a quantum architecture due to QPU shot-RNG reassociation.**\] → The smoke configs for the 8 quantum architectures will use a deterministic statevector simulator (not the noisy AerSimulator), eliminating shot-RNG variance. If a quantum arch still fails `atol=1e-7`, that arch's regression bar widens to the fallback "training-curve shape preserved within 5%" tier with documented justification in [logbook 023](../../../docs/experiments/logbooks/). T2 does not block on quantum-arch byte-equivalence.

\[**Risk B: `BrainType` enum and registry get out of sync (one has a member the other doesn't).**\] → Decision 3's startup-time consistency check raises an exception if the enum and registry disagree at import time. Failure mode is loud and immediate; impossible to ship a release where the two diverge.

\[**Risk C: Existing YAML configs break because YAML parsing assumes hand-maintained `BRAIN_CONFIG_MAP`.**\] → Decision 1's registry-derived `BRAIN_CONFIG_MAP` preserves the dict shape exactly (`name -> config_cls`). The `_resolve_brain_config` helper at [config_loader.py:153-216](../../../packages/quantum-nematode/quantumnematode/utils/config_loader.py) is unchanged. A regression test runs every existing scenario YAML under [configs/scenarios/](../../../configs/scenarios/) through `configure_brain()` and asserts each loads to a valid `Brain` instance — catches any YAML compatibility break.

\[**Risk D: The topology/rule factoring introduces a tensor-op reassociation that breaks MLPPPO + LSTMPPO byte-equivalence.**\] → Decision 2's strategy is to keep existing `forward` / `learn` method bodies intact; the topology + rule references are reference-holders, not call-site rewrites. Byte-equivalence is preserved by construction. The byte-equivalence test catches any accidental reassociation.

\[**Risk E: `ConnectomePPOBrain` trains poorly because the chemical-synapse strict-mask leaves too few learnable parameters for klinotaxis.**\] → Per [phase6-tracking/design.md § Decision 6 § Gate 1 STOP/PIVOT triggers](../phase6-tracking/design.md): if G1.c fails on the full 302-neuron connectome after the diagnostic sequence, pivot to a hand-curated sensory-interneuron-motor subgraph (~50-100 neurons). The PIVOT is itself a documented decision in [logbook 023](../../../docs/experiments/logbooks/); Gate 1 re-evaluates against the subset. STOP only if the subset also fails.

\[**Risk F: Sensor projection / motor readout choice biases the Gate 1 G1.c outcome.**\] → T2 picks a canonical Bargmann-lab default (Decision 4) and documents it. The choice is a known confound; T4 explicitly sweeps it as the [T4.0c](../phase6-tracking/tasks.md) ablation. Gate 1 G1.c is a smoke check, not a definitive ranking — the definitive ranking lands at T4.analysis.connectome_grid_ranking.

\[**Risk G: Decorator-registration import-time ordering issues** (e.g. `BrainType.MLP_PPO` is referenced inside `mlpppo.py` at decoration time but the enum module hasn't loaded yet).\] → The registry module imports `BrainType` from `dtypes.py` first; every `brain/arch/<name>.py` imports `dtypes.py` and the registry module before the decorator runs. Python's import machinery handles the rest. A unit test verifies the registry is fully populated after `import quantumnematode.brain.arch`.

\[**Risk H: The 19-architecture migration takes longer than the [3-5 week T2 estimate](../phase6-tracking/tasks.md).**\] → Decision 7's migration sequencing surfaces blockers early: if MLPPPO migration breaks byte-equivalence, that's surfaced before any other arch is touched. Per-architecture migration is small (decorator + topology/rule extraction); the long tail is debugging RNG/reassociation drift on quantum architectures, which Risk A's fallback tolerance handles. The Phase 6 budget Decision 1 amendment mechanism kicks in if the tranche overruns by > 2×.

## Migration Plan

T2 ships as a single OpenSpec change (single feature branch, single PR series within the change) per the user's confirmed scope decision. Implementation sequencing (drives the `tasks.md` ordering):

1. **Scaffold the registry** — `_registry.py` + `_topology.py` + `_rule.py` modules, unit tests for the registry (register / instantiate / duplicate-name detection / unknown-name error).
2. **Migrate MLPPPO** — decorator + internal topology/rule extraction. Capture byte-equivalence fixture pre-refactor; assert post-refactor. **Migration validation gate: if MLPPPO byte-equivalence fails, halt migration, re-design Decision 2's strategy.**
3. **Migrate LSTMPPO** — same pattern. **Migration validation gate: if LSTMPPO byte-equivalence fails, halt migration.**
4. **Migrate the 7 classical / spiking brains** — batch migration, numerical equivalence at `atol=1e-7`. Per-architecture pass/fail recorded.
5. **Migrate the 10 quantum + spiking-quantum brains** — batch migration with the deterministic-simulator strategy from Risk A. Per-architecture pass/fail recorded; any failure escalates to the fallback tolerance tier.
6. **Implement `ConnectomePPOBrain`** — topology construction from `Connectome`, sensor projection, motor readout, PPO rule wiring. Unit tests for shape + finiteness + strict-mask invariant + gap-junction-frozen invariant.
7. **G1.c smoke campaign** — run the klinotaxis learning + frozen-control pair; record measurements; evaluate Gate 1 G1.c pass conditions.
8. **Plugin-developer documentation** — write the walkthrough; include the hypothetical-addition exercise per [phase6-tracking/tasks.md T2.7](../phase6-tracking/tasks.md).
9. **Gate 1 decision** — write the GO/PIVOT/STOP decision in [logbook 023](../../../docs/experiments/logbooks/), evaluating all four G1.a–G1.d criteria. Tick T2 sub-tasks + Gate 1 decision link in [phase6-tracking/tasks.md](../phase6-tracking/tasks.md). Flip the Phase 6 Tranche Tracker T2 row to ✅ complete in [docs/roadmap.md](../../../docs/roadmap.md).

**Rollback strategy**: T2 lands on a feature branch and merges via PR. Per-architecture migration is in separate commits so a single arch can be reverted independently. The pre-refactor byte-equivalence fixtures are committed under `tests/.../test_data/` and remain valid after a revert. If Gate 1 STOPs, the registry refactor still has standalone value (it's a clean dispatcher → registry refactor regardless of the connectome brain outcome); the connectome brain commits can be reverted, leaving the registry in place.

## Open Questions

- **What's the exact K (forward-pass depth) for `ConnectomeTopology`?** T2 defaults to K=1 for parity with the smoke-test forward-pass; T4's sensor-projection ablation may sweep K. The default is documented but not formally evaluated against alternatives in T2.
- **Should the readout layer be PPO-learnable or frozen-random?** T2 defaults to PPO-learnable (matches the Bargmann-lab convention that motor readout is also plastic in vivo); a frozen-random readout is a possible T4-scope ablation. Decision 4 commits to PPO-learnable.
- **What's the smoke-config seed?** TBD during implementation; whatever seed is chosen is committed to the YAML and used identically for the learning run + frozen control + byte-equivalence captures. Single-seed by design (G1.c is a smoke check).
- **`np.allclose` vs `torch.allclose` for the 17-arch tolerance bar?** Both work; `torch.allclose` matches the existing test style. Will commit to `torch.allclose(rtol=0, atol=1e-7)` during implementation.
