## Context

QRH has proven viable (41.2% avg success, 68% post-convergence on large pursuit predators after R9). To scientifically evaluate whether the quantum reservoir adds value, we need a classical control with identical architecture. Additionally, no existing brain in the suite occupies the "classical fixed reservoir + PPO readout" niche, making CRH valuable as a permanent benchmark architecture.

The existing `qrh.py` is ~1100 lines, of which ~800 are reservoir-agnostic PPO infrastructure (actor/critic MLPs, rollout buffer, PPO training, LR scheduling, episode tracking). The quantum-specific code (~300 lines) covers reservoir circuit construction, statevector simulation, and X/Y/Z+ZZ feature extraction. This clean boundary at `_get_reservoir_features()` makes base class extraction straightforward.

## Goals / Non-Goals

**Goals:**

- Extract `ReservoirHybridBase` from QRH containing all shared PPO readout infrastructure
- Implement `CRHBrain` with classical ESN reservoir and configurable feature channels
- Support two operational modes via config: ablation (matching QRH's 75-feature output) and standalone (optimized for best performance)
- Ensure QRH continues to pass all existing tests after refactoring (zero behavioral changes)
- Enable clean three-way comparison: QRH vs CRH (quantum value) and CRH vs MLPPPO (reservoir architecture value)

**Non-Goals:**

- Training reservoir weights (the ESN reservoir is intentionally fixed, matching QRH's fixed quantum reservoir)
- Temporal/recurrent reservoir processing (each `run_brain()` call is independent — no hidden state carried across steps)
- Online reservoir adaptation or spectral radius tuning during training
- Implementing other classical reservoir types (Liquid State Machines, etc.) — ESN is sufficient
- Changing QRH's behavior or performance in any way

## Decisions

### Decision 1: ReservoirHybridBase Architecture

**Choice**: Extract a `ReservoirHybridBase(ClassicalBrain)` base class from QRH. Both `QRHBrain` and `CRHBrain` inherit from it.

**Rationale**: QRH's ~800 lines of PPO infrastructure (actor/critic MLPs, rollout buffer, PPO training loop, LR scheduling, episode tracking, feature normalization, copy, preprocess) are reservoir-agnostic. The boundary is clean: subclasses implement `_get_reservoir_features(sensory_features) -> np.ndarray` and `_compute_feature_dim() -> int`. Everything downstream is identical.

**What moves to base**:

- `ReservoirHybridBaseConfig(BrainConfig)` — all shared config fields: `readout_hidden_dim`, `readout_num_layers`, `actor_lr`, `critic_lr`, `gamma`, `gae_lambda`, `ppo_clip_epsilon`, `ppo_epochs`, `ppo_minibatches`, `ppo_buffer_size`, `entropy_coeff`, `value_loss_coef`, `max_grad_norm`, LR scheduling fields (`lr_warmup_episodes`, `lr_warmup_start`, `lr_decay_episodes`, `lr_decay_end`), `sensory_modules`
- `_RolloutBuffer` — the rollout buffer class (unchanged, includes `compute_returns_and_advantages()` for GAE)
- Seeding logic (`ensure_seed`, `get_rng`, `set_global_seed`)
- Sensory module setup (`input_dim` computation, `sensory_modules` storage)
- Actor/critic MLP construction via `build_readout_network()` from `_quantum_reservoir.py`, LayerNorm, combined Adam optimizer
- `run_brain()`, `learn()`, `_perform_ppo_update()`
- `preprocess()`, `prepare_episode()`, `post_process_episode()`, `copy()`, `update_memory()`
- LR scheduling logic (`_get_current_lr()`, `_update_learning_rate()`)
- Episode tracking, state tracking (`_episode_count`, `training`, `current_probabilities`, `last_value`, `_pending_*` fields), `history_data`, `latest_data`
- Diagnostic logging with subclass-provided `_brain_name` class attribute (e.g., `"QRH"`, `"CRH"`) for log prefixes

**What stays in QRHBrain**:

- `QRHBrainConfig(ReservoirHybridBaseConfig)` — adds `num_reservoir_qubits`, `reservoir_depth`, `reservoir_seed`, `shots`, `use_random_topology`, `num_sensory_qubits`
- Quantum topology constants (GAP_JUNCTION_CZ_PAIRS, CHEMICAL_SYNAPSE_ROTATIONS)
- `_build_structured_reservoir()`, `_build_random_reservoir()`, `_generate_random_topology()`
- `_encode_and_run()`, `_extract_features()` (Qiskit/statevector code)
- `_get_reservoir_features()` (2-line method calling the above)
- `_compute_feature_dim()` (returns `3*N + N*(N-1)//2`) — NOTE: currently a module-level function `_compute_feature_dim(num_qubits)` in `qrh.py`; refactor to instance method `def _compute_feature_dim(self) -> int` using `self.num_qubits`

**Alternatives considered**:

- Flag on QRH (`reservoir_type: quantum|classical`): Violates single-responsibility, mixes Qiskit+ESN imports
- Standalone CRH with copy-pasted PPO: ~800 lines of duplication, maintenance nightmare
- Shared mixin: Python mixins are fragile with `__init__` chains; inheritance is cleaner here

### Decision 2: Classical ESN Reservoir

**Choice**: Echo State Network with fixed random weight matrices (W_in, W_res), tanh nonlinearity, configurable spectral radius.

**Rationale**: ESN is the simplest well-understood classical reservoir. It provides a fair structural comparison to QRH's fixed quantum reservoir:

- Both have fixed, non-trainable dynamics
- Both transform input via nonlinear mapping
- Both feed into the same PPO readout

**ESN computation** (per `_get_reservoir_features()` call):

```text
h_0 = tanh(W_in @ x)                    # Initial projection
h_l = tanh(W_res @ h_{l-1} + W_in @ x)  # Data re-uploading (layers 1..depth-1)
```

Data re-uploading at each layer matches QRH's pattern where input features are re-encoded before each reservoir layer.

**Configurable parameters**:

- `num_reservoir_neurons` (int, default 10): Size of hidden state h
- `reservoir_depth` (int, default 3): Number of reservoir layers (matches QRH)
- `spectral_radius` (float, default 0.9): Controls W_res eigenvalue scaling — key ESN hyperparameter
- `reservoir_seed` (int, default 42): Deterministic initialization
- `input_connectivity` (str, default "sparse"): "sparse" = only sensory neurons receive input (matches QRH sensory qubit pattern), "dense" = all neurons receive input
- `input_scale` (float, default 1.0): Scaling factor for W_in entries
- `num_sensory_neurons` (int | None): Number of sensory neurons for sparse connectivity (named `num_sensory_neurons` to match other classical brains like QSNN/HybridQuantum; QRH keeps `num_sensory_qubits`)

**Weight initialization**:

- W_in: Uniform [-input_scale, input_scale]. For dense: shape `(num_neurons, input_dim)`. For sparse: shape `(num_neurons, input_dim)` with rows beyond `num_sensory_neurons` zeroed out (non-sensory neurons receive no direct input, only recurrent signal via W_res)
- W_res: Random normal, then scaled so largest eigenvalue magnitude = spectral_radius. Guard: if max eigenvalue magnitude < 1e-10, skip scaling (degenerate matrix)

### Decision 3: Configurable Feature Channels

**Choice**: Feature extraction via configurable channel list, supporting both ablation-matched and standalone-optimized modes.

**Rationale**: QRH extracts three types of features: X-expectations, Y-expectations, Z-expectations (3N features) + ZZ-correlations (N(N-1)/2 features). The classical analog needs to produce comparable information diversity from ESN activations.

**Available channels** (computed from final-layer activations h):

| Channel | Formula | Dim | Analog to |
|---------|---------|-----|-----------|
| `raw` | h_i | N | Z-expectations |
| `cos_sin` | cos(pi*h_i), sin(pi*h_i) | 2N | X/Y-expectations (Fourier basis on [-1,1]) |
| `squared` | h_i^2 | N | Magnitude features |
| `pairwise` | h_i * h_j for i\<j | N(N-1)/2 | ZZ-correlations |

**Config field**: `feature_channels: list[FeatureChannel]` — ordered list of channel names, where `FeatureChannel = Literal["raw", "cos_sin", "squared", "pairwise"]`. Using a `Literal` type catches typos at Pydantic validation time rather than runtime.

**Ablation profile**: `[raw, cos_sin, pairwise]` with N=10 -> 10 + 20 + 45 = 75 features (exact match to QRH's 3\*10 + C(10,2) = 75)

**Standalone profile**: Tunable — e.g., `[raw, cos_sin, squared, pairwise]` with N=14 -> 14 + 28 + 14 + 91 = 147 features, or whatever performs best.

**Feature dim computation**:

```python
def _compute_feature_dim(num_neurons, channels):
    dim = 0
    for ch in channels:
        if ch == "raw": dim += num_neurons
        elif ch == "cos_sin": dim += 2 * num_neurons
        elif ch == "squared": dim += num_neurons
        elif ch == "pairwise": dim += num_neurons * (num_neurons - 1) // 2
    return dim
```

### Decision 4: CRH Brain Type Classification

**Choice**: CRH is classified in `CLASSICAL_BRAIN_TYPES`, not `QUANTUM_BRAIN_TYPES`.

**Rationale**: CRH uses no quantum circuits. It is purely classical. Placing it in `CLASSICAL_BRAIN_TYPES` is semantically correct and enables the right benchmark comparisons. QRH remains in `QUANTUM_BRAIN_TYPES`.

### Decision 5: Base Class Copy Semantics

**Choice**: `ReservoirHybridBase.copy()` uses a construct-then-copy-weights pattern (matching QRH's existing implementation). The base class `copy()`:

1. Serializes `self.config` via `model_dump()` (Pydantic deep copy)
2. Calls subclass's `_create_copy_instance(config_copy)` to construct a fresh instance (re-runs `__init__`, regenerating reservoir from seed)
3. Deep-copies shared network weights into the new instance: actor, critic, feature_norm state dicts + optimizer state dict
4. Copies `_episode_count` (drives LR scheduling)

The new instance gets a fresh empty rollout buffer and reset pending state from its own `__init__` — this matches the current QRH behavior.

Subclasses implement `_create_copy_instance(config)` which just constructs `type(self)(config, ...)`. No additional deep copies are needed because reservoir state is deterministically regenerated from the seed.

**For QRH**: Random topology data (`_random_cz_pairs`, `_random_rotations`) is regenerated from `reservoir_seed` in `__init__`, producing identical results.

**For CRH**: W_in and W_res matrices are regenerated from `reservoir_seed` in `__init__`, producing identical results.

### Decision 6: No Temporal State

**Choice**: CRH reservoir is stateless between `run_brain()` calls — each call computes fresh from sensory input.

**Rationale**: QRH's quantum reservoir is also stateless (fresh circuit per call). Matching this ensures fair comparison. The ESN "depth" parameter controls layered computation within a single call, not temporal unfolding.

This means `h` is NOT carried across time steps. Each `_get_reservoir_features()` call starts from `h_0 = tanh(W_in @ x)`.

### Decision 7: Base Class Init Ordering

**Choice**: Subclasses pass `feature_dim` as a constructor argument to `ReservoirHybridBase.__init__()`.

**Rationale**: The base class needs `feature_dim` to construct actor/critic MLPs, but `feature_dim` depends on subclass-specific computation (quantum feature formula for QRH, channel-based formula for CRH). Two options exist:

- **Option A**: Base calls `self._compute_feature_dim()` (abstract) during its own `__init__`. Requires subclass to set reservoir params *before* calling `super().__init__()`, creating a fragile ordering dependency.
- **Option B**: Subclass computes `feature_dim` first, passes it to `super().__init__(config, feature_dim, ...)`. Clean, explicit, no hidden ordering dependency.

Option B is chosen. The subclass `__init__` pattern becomes:

```python
def __init__(self, config, ...):
    # 1. Set up reservoir-specific state (W_in/W_res or quantum topology)
    # 2. Compute feature_dim from reservoir params
    feature_dim = self._compute_feature_dim()
    # 3. Call base init with feature_dim
    super().__init__(config, feature_dim, num_actions, device, action_set)
```

### Decision 8: Diagnostic Logging Parameterization

**Choice**: Base class uses a `_brain_name` class attribute for log prefixes. Subclasses set `_brain_name = "QRH"` or `_brain_name = "CRH"`.

**Rationale**: QRH's `run_brain()` and `_perform_ppo_update()` include diagnostic logging with "QRH" prefixes (e.g., `"QRH step 50: ..."`, `"QRH PPO update: ..."`). When these methods move to the base class, the prefix must reflect the actual subclass. A class attribute is simpler than passing a string through constructors and avoids `type(self).__name__` which would produce `"QRHBrain"` instead of the short `"QRH"`.

## Risks / Trade-offs

**[Risk] Base class extraction may introduce subtle behavioral changes to QRH**
-> Mitigation: Full QRH test regression after refactoring. All 50+ existing QRH tests must pass without modification. If any fail, the extraction boundary is wrong.

**[Risk] CRH ablation mode may not be a perfectly fair quantum comparison**
-> Mitigation: The comparison is "approximately fair" by design — same architecture, same feature count, same PPO readout. Perfect fairness is impossible (quantum and classical nonlinearities differ fundamentally). The feature channel design (cos_sin as analog to X/Y expectations) maximizes structural correspondence.

**[Risk] ESN spectral radius may need environment-specific tuning**
-> Mitigation: Start with 0.9 (well-established default). Spectral radius is a config parameter, not hardcoded. If ablation results are surprisingly bad, tune before concluding quantum advantage.

**[Risk] ESN W_res eigenvalue degenerate case**
-> Mitigation: If max eigenvalue magnitude < 1e-10 (all-zero or near-zero matrix), skip spectral radius scaling and log a warning. This is extremely unlikely with random normal initialization but prevents division by zero.

**[Trade-off] CRH in ablation mode may underperform a properly optimized classical reservoir**
-> Acceptable: Ablation mode intentionally constrains CRH to match QRH's structure. The standalone mode allows optimized comparison against MLPPPO.

**[Trade-off] Base class adds one level of inheritance**
-> Acceptable: Two concrete subclasses (QRH, CRH) justify the abstraction. The alternative (800 lines of duplication) is far worse.
