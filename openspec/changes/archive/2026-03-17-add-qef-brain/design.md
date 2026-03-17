## Context

QRH is our only architecture with genuine quantum advantage (+9.4pp over CRH on pursuit predators). It uses a fixed random quantum reservoir with X/Y/Z+ZZ feature extraction and a classical PPO readout. QA-5 (QEF) extends this paradigm by replacing the random reservoir entanglement with purposeful cross-modal entanglement between sensory-modality qubit pairs, hypothesizing that entangled features encode predator-prey interaction dynamics that separable circuits miss.

QEF reuses the proven `ReservoirHybridBase` infrastructure (PPO training, actor-critic readout, buffer management, LR scheduling) and follows the same subclassing pattern as QRH/CRH.

During evaluation (~500+ runs across multiple experiment rounds), the architecture was significantly extended with hybrid input, learnable feature gating, curated feature subsets, and classical ablation infrastructure.

## Goals / Non-Goals

**Goals:**

- Implement QEFBrain as a `ReservoirHybridBase` subclass with entangled PQC feature extraction
- Support three entanglement topologies: modality-paired, ring, random
- Provide separable ablation mode via config flag for controlled experiments
- Match all QRH evaluation environments with equivalent YAML configs
- Enable direct performance comparison with QRH using identical PPO readout
- Support hybrid input mode (raw sensory + quantum features)
- Support learnable feature gating (static, context, mixed modes)
- Support curated feature subsets (cross-modal ZZ, optional cos/sin, optional ZZZ)
- Provide classical ablation infrastructure in MLP PPO (polynomial, random projection feature expansion + gating)

**Non-Goals:**

- Trainable entanglement angles (reserved for future work; config field exists but raises NotImplementedError)
- QPU execution support — statevector simulation only (same as QRH)

## Decisions

### 1. Naming: `qef` (Quantum Entangled Features)

Follows the 3-letter lowercase convention (qrh, crh, qrc). Descriptive of the architecture's key differentiator. Avoids tying to research doc numbering (qa5).

### 2. Feature channels: configurable via `feature_mode`

Two modes via `feature_mode: Literal["z_cossin", "xyz"]`:

- **z_cossin** (default): Z + ZZ + cos/sin(Z) = 3N + N(N-1)/2 features
- **xyz**: X + Y + Z + ZZ = 3N + N(N-1)/2 features (matching QRH)

Additional options:

- `include_cossin: bool` — disable cos/sin to reduce dimensionality (used for curated feature set)
- `zz_mode: Literal["all", "cross_modal"]` — compute all ZZ pairs or only cross-modal pairs
- `include_zzz: bool` — add ZZZ three-body correlations (N(N-1)(N-2)/6 features)

### 3. Gate mode: configurable CZ vs CRY/CRZ

`gate_mode: Literal["cz", "cry_crz"]` — initial CZ-only design was extended with CRY/CRZ controlled rotations (seeded random angles) after MI analysis showed CZ-only had insufficient expressivity.

### 4. Encoding mode: uniform vs sparse

`encoding_mode: Literal["uniform", "sparse"]`:

- **uniform** (default): RY on ALL qubits, cycling features
- **sparse**: RY/RZ only on first input_dim qubits (like QRH), leaving remaining qubits as entanglement relays

### 5. Hybrid input: raw + quantum features

`hybrid_input: bool` — concatenates raw sensory features (7-dim) with quantum features (31-52 dim) before the readout MLP. This gives the network direct access to actionable signals while benefiting from quantum correlations. The hybrid approach was the key innovation enabling QEF to approach classical performance.

Optional `hybrid_polynomial: bool` adds classical pairwise products to the hybrid input (tested but found redundant with quantum ZZ correlations).

### 6. Feature gating: learnable quantum feature selection

`feature_gating: Literal["none", "static", "context", "mixed"]`:

- **none**: no gating
- **static**: `sigmoid(w) * quantum_features` — learned per-dimension weights
- **context**: `sigmoid(MLP(raw_features)) * quantum_features` — input-dependent gating via small MLP (raw_dim → 16 → quantum_dim)
- **mixed**: average of static + context gates

Gating only applies to the quantum portion of hybrid features; raw features pass through unchanged. The optimal gating mode is task-dependent: static for weak-signal tasks (stationary), context for strong-signal tasks (pursuit).

The `_apply_feature_gating` method, custom `run_brain()` and `_perform_ppo_update()` overrides handle gating in both inference and PPO training.

### 7. Separate critic (tested, ruled out)

`separate_critic: bool` — gives the critic raw features only (requires hybrid_input). Tested and found harmful — the critic benefits from quantum features.

### 8. Curated feature subsets

`zz_mode: "cross_modal"` computes only cross-modal ZZ pairs (food-noci, food-thermo, noci-thermo), dropping intra-modal correlations. Combined with `include_cossin: false`, this reduces feature dimensionality from 52 to 31 quantum features, improving signal-to-noise on weak-signal tasks.

`_get_cross_modal_pairs()` function determines cross-modal pairs based on the qubit-to-modality mapping derived from sensory module structure.

### 9. Classical ablation infrastructure (MLP PPO)

Added to `MLPPPOBrainConfig`:

- `feature_expansion: Literal["none", "polynomial", "polynomial3", "random_projection"]`
- `feature_gating: bool` — sigmoid gate on expanded features
- `feature_expansion_dim: int` — target dimension for random projection
- `feature_expansion_seed: int` — reproducible random projection

These enable rigorous ablation testing: polynomial features test whether classical pairwise products match quantum ZZ correlations; random projection tests whether any feature expansion suffices.

### 10. Entanglement topology: configurable via Literal field

Three options via `entanglement_topology: Literal["modality_paired", "ring", "random"]`:

- **modality_paired**: CZ gates between cross-modal qubit pairs
- **ring** (default for evaluated configs): CZ in a ring (0-1, 1-2, ..., 7-0). 8 CZ pairs — outperforms modality_paired by +12.5pp on pursuit
- **random**: Seeded random CZ pairs (same count as modality_paired = 4 pairs)

### 11. Extend ReservoirHybridBase (not create new base class)

QEF implements the same 3 abstract methods as QRH: `_get_reservoir_features()`, `_compute_feature_dim()`, `_create_copy_instance()`. Additionally overrides `run_brain()` and `_perform_ppo_update()` when feature gating or separate critic is active.

## Risks / Trade-offs

**Feature wrapping with >8 inputs** → When using all 4 sensory modules (9 features for 8 qubits), the 9th feature wraps onto qubit 0. Handled by modular indexing.

**High feature dimensionality** → Full QEF hybrid input is 59-dim (7 raw + 52 quantum), creating compression bottleneck at the 64-wide readout. Curated mode (38-dim) partially addresses this.

**Classical polynomial self-silencing** → Polynomial features (x_i * x_j) naturally go to zero when either input is zero. Quantum ZZ correlations lack this property, requiring learned gating to achieve similar noise suppression. This structural difference explains why polynomial features outperform QEF on weak-signal tasks.

**Trainability-advantage dilemma** → The 8-qubit depth-2 circuit is classically simulatable. ZZ correlations are expressible as classical functions of input rotation angles. The "quantum" aspect is the entanglement structure selecting which correlations to compute, but polynomial expansion achieves similar effect.

## Evaluation Outcome

12-seed validation across 3 tasks showed QEF is competitive but does not demonstrate statistically significant quantum advantage:

- Stationary: QEF 90.8% vs MLP PPO 89.6% vs A3 Poly 93.8% (QEF vs Poly: p=0.08, ns)
- Pursuit: QEF 93.0% vs MLP PPO 96.0% vs A3 Poly 94.8% (QEF vs MLP: p=0.04, \*)
- Small PP: QEF 98.2% vs MLP PPO 98.6% vs A3 Poly 97.0% (all ns)

See `build/brains/qef/qef_scratchpad.md` for complete evaluation history.
