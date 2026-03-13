## Context

QRH is our only architecture with genuine quantum advantage (+9.4pp over CRH on pursuit predators). It uses a fixed random quantum reservoir with X/Y/Z+ZZ feature extraction and a classical PPO readout. QA-5 (QEF) extends this paradigm by replacing the random reservoir entanglement with purposeful cross-modal entanglement between sensory-modality qubit pairs, hypothesizing that entangled features encode predator-prey interaction dynamics that separable circuits miss.

QEF reuses the proven `ReservoirHybridBase` infrastructure (PPO training, actor-critic readout, buffer management, LR scheduling) and follows the same subclassing pattern as QRH/CRH.

## Goals / Non-Goals

**Goals:**

- Implement QEFBrain as a `ReservoirHybridBase` subclass with entangled PQC feature extraction
- Support three entanglement topologies: modality-paired, ring, random
- Provide separable ablation mode via config flag for controlled experiments
- Match all QRH evaluation environments with equivalent YAML configs
- Enable direct performance comparison with QRH using identical PPO readout

**Non-Goals:**

- Trainable entanglement angles (reserved for future work; config field exists but raises NotImplementedError)
- New PPO variants or readout architectures — identical to QRH
- Classical MLP ablation brain — reuse existing MLPPPOBrain
- QPU execution support — statevector simulation only (same as QRH)

## Decisions

### 1. Naming: `qef` (Quantum Entangled Features)

Follows the 3-letter lowercase convention (qrh, crh, qrc). Descriptive of the architecture's key differentiator. Avoids tying to research doc numbering (qa5).

### 2. Feature channels: Z + ZZ + cos_sin (not X+Y+Z+ZZ like QRH)

QRH uses asymmetric encoding (RY on even features, RZ on odd) on only 2 sensory qubits, making X/Y expectations informative about phase differences. QEF uses uniform RY encoding on all 8 qubits, so X/Y expectations carry less distinct information. Instead, cos/sin nonlinear transforms of Z expectations provide expressivity while keeping the same total feature count (52 for 8 qubits):

- 8 Z expectations (per-qubit)
- 28 ZZ correlations (pairwise, i < j)
- 16 cos/sin features (cos(Z_i) and sin(Z_i) for each qubit)

Formula: `N + N(N-1)/2 + 2N = 3N + N(N-1)/2`

Alternative considered: X+Y+Z+ZZ (same as QRH). Rejected because the uniform encoding makes X/Y less meaningful, and using different channels makes the ablation cleaner — differences between entangled and separable isolate the entanglement effect on the same feature type.

### 3. All qubits are sensory (not 2 of 8 like QRH)

QRH maps the C. elegans connectome where only ASE sensory neurons (2 qubits) receive input. QEF encodes sensory features on all 8 qubits — each qubit maps to a specific sensory feature. This maximizes the information available for entanglement to create cross-modal correlations.

Qubit-to-feature mapping depends on configured `sensory_modules`. Features are assigned to qubits in module order (food_chemotaxis, nociception, thermotaxis, mechanosensation). When `input_dim > num_qubits`, features wrap modularly (qubit index = feature index % num_qubits), matching QRH's existing wrapping behavior.

### 4. Entanglement topology: configurable via Literal field

Three options via `entanglement_topology: Literal["modality_paired", "ring", "random"]`:

- **modality_paired** (default): CZ gates between cross-modal qubit pairs encoding interacting sensory modalities:
  - (0, 2): food_chemotaxis ↔ nociception (approach vs avoidance)
  - (1, 3): food_chemotaxis ↔ nociception (angular correlation)
  - (4, 6): thermotaxis ↔ mechanosensation (environmental sensing)
  - (5, 7): thermotaxis ↔ mechanosensation (angular correlation)
- **ring**: CZ in a ring (0-1, 1-2, ..., 7-0). 8 CZ gates — tests nearest-neighbor entanglement.
- **random**: Seeded random CZ pairs (same count as modality_paired = 4 pairs). Controls for entanglement structure vs presence.

Alternative considered: enum class. Rejected — Literal is simpler and consistent with other string config fields in the codebase.

### 5. Separable ablation: config flag, not separate brain type

`entanglement_enabled: bool = True`. When False, CZ gates are skipped — the circuit becomes a product-state (separable) PQC. This shares all other code, making A/B comparison trivial and avoiding brain type proliferation.

### 6. Circuit structure: H → [RY + CZ]^depth → statevector

Per layer (repeated `circuit_depth` times, default 2):

1. RY(feature × π) on all qubits (data re-uploading)
2. CZ entanglement gates per selected topology (skipped if `entanglement_enabled=False`)

Initial H layer on all qubits for superposition. Exact statevector simulation via `Statevector.from_instruction()`.

Alternative considered: RY+RZ encoding (like QRH). Rejected — uniform RY is simpler and the entanglement topology is the variable under test, not the encoding scheme.

### 7. Extend ReservoirHybridBase (not create new base class)

QEF implements the same 3 abstract methods as QRH: `_get_reservoir_features()`, `_compute_feature_dim()`, `_create_copy_instance()`. All PPO infrastructure is inherited. This is the minimal-risk approach.

## Risks / Trade-offs

**Feature wrapping with >8 inputs** → When using all 4 sensory modules (9 features for 8 qubits), the 9th feature wraps onto qubit 0. This is handled by modular indexing but may dilute the modality-paired topology's semantic meaning. Mitigation: document and test explicitly; evaluation configs use sensory_modules combinations that fit naturally.

**cos/sin features may be less useful than X/Y** → If cos/sin of Z expectations are too smooth (low gradient), the PPO readout may train slower. Mitigation: the decision gate (week 1) checks MI(entangled_features, optimal_action) early. If cos/sin underperforms, we can switch to X/Y+Z+ZZ.

**Modality-paired topology assumes specific qubit-feature ordering** → If sensory_modules are configured differently, the cross-modal CZ pairs may not match intended modality interactions. Mitigation: document the expected module ordering; validate in config.

**Fixed entanglement may underperform random** → QRH's random reservoir already captures sufficient structure. Mitigation: this is exactly what the falsification criteria test — if fixed entangled ≤ QRH random, we stop.

### 8. 8 qubits for all environments (including large)

QRH uses 10 qubits for large environments (7 sensory + 3 interneuron). QEF uses 8 qubits for all configs because: (a) all qubits are sensory — with 7 features from 3 modules, 8 gives near-1:1 mapping with 1 qubit in superposition; (b) the modality-paired topology has 4 CZ pairs designed for 8 qubits — 10 would leave 2 qubits without topology assignment, diluting entanglement; (c) 8 qubits produce 52 features which is sufficient for the readout MLP. The readout_hidden_dim is set to 64 (not 128 as QRH large) since the feature count is lower (52 vs 75).

### 9. MI decision gate script

A `scripts/qef_mi_analysis.py` script adapted from `scripts/qrh_mi_analysis.py` compares MI(entangled_features, optimal_action) vs MI(separable_features, optimal_action) vs MI(qrh_random_features, optimal_action). This implements the Week 1 decision gate from the research doc: if entangled MI ≤ separable MI, try alternative topologies; if entangled MI ≤ QRH random MI, stop.
