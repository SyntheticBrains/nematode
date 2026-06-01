# Proposal: Add Equivariant Quantum Brain (`EquivariantQuantumPPOBrain`)

## Why

The Phase 6 cross-architecture comparison ranks brain architectures on an integrated foraging +
predator-evasion + thermotaxis cell. The architecture-promotion gate selected a **quantum** arm to
complete the comparison's architecture axis (alongside the wild-type connectome, the CfC liquid brain,
and the spiking brain). This change ships that arm as a **genuinely-quantum, bilateral-symmetry-equivariant
parameterised quantum circuit (PQC) policy** — trained quantum parameters with entanglement as a
load-bearing resource — *not* a fixed quantum-reservoir feature map with a classical readout.

A pre-build investigation settled both the architecture choice and the honest framing:

- **Why not reuse an existing quantum brain.** The repo's quantum stack splits into (a) reservoirs
  (`qrh`, `qrc`, `qef`, `qqlearning`) — a *fixed/untrained* quantum circuit + classical readout, i.e.
  "mostly classical"; and (b) the trainable variational circuit `qvarcircuit` — genuinely quantum, but
  REINFORCE (not PPO), parameter-shift gradients (expensive), a crude measurement→action readout, and
  barren-plateau-limited (≈40% online, 88% only via offline CMA-ES per logbook 008). Neither is a clean,
  PPO-trainable, genuinely-quantum arm for this comparison.

- **Why equivariant.** A thorough 2025–2026 literature scan found that for small classical-data RL on a
  simulator, (i) no genuine quantum advantage is expected, (ii) small trainable circuits that avoid
  barren plateaus are generically classically simulable, and (iii) on gridworld-PPO specifically,
  unstructured entanglement has been shown not to pay off. The **one** family with *proven*
  barren-plateau-resistance + load-bearing entanglement at small scale is **symmetry-equivariant**
  circuits (permutation/reflection-equivariant QNNs). Encoding a task symmetry is therefore both the
  trainability fix and the scientific question.

- **Which symmetry.** The actions are egocentric (`FORWARD, LEFT, RIGHT, STAY`) and klinotaxis sensing
  is already egocentric, so the natural symmetry is the **left–right mirror (Z₂)** — which is literally
  *C. elegans*' bilateral body symmetry and the natural symmetry of klinotaxis navigation. Because the
  observation is egocentric, the reflection acts as a **fixed, heading-independent ±1 parity operator**:
  the lateral-gradient *angle* features are Z₂-odd (sign-flip), everything else is Z₂-even; the action
  mirror swaps `LEFT`↔`RIGHT` and fixes `FORWARD`/`STAY`.

**Honest, pre-registered framing.** This arm is built for comparison completeness and a principled
inductive-bias study, **not** as a quantum-advantage claim. The literature-supported expectation is
"competitive-or-below a parameter-matched classical baseline." The scientific value is the measured
**equivariant-vs-non-equivariant** delta (does the bilateral prior help?) and the
**quantum-vs-equivariant-classical** delta (does putting the prior in an entangled circuit do anything a
classical equivariant net does not?) — answered by two ablation baselines shipped with the brain.

This change ships **only** the brain, its two ablation siblings (config-flagged), registry integration,
a smoke config, and tests. The combined-cell C3 config, the ranked evaluation, and the
`phase6-tracking` Decision-4 amendment that formally promotes the quantum family live downstream in the
comparison change.

## What Changes

- **New brain** `EquivariantQuantumPPOBrain` (`name: equivariantquantum`,
  `brain_type: EQUIVARIANT_QUANTUM_PPO`, `families: ("quantum",)`):
  - an **equivariant classical pre-encoder** (a parity-block-structured linear map) compressing the
    sensory observation to `num_qubits` Z₂-typed latent features (split into even and odd channels);
  - a **Z₂-equivariant parameterised quantum circuit** — angle encoding with data re-uploading, a
    variational ansatz drawn from the `U_R`-invariant gate set (entangling `IsingXX` / same-parity
    `IsingZZ` + single-qubit rotations), where `U_R = ⊗ X` over the odd-parity qubits realises the
    reflection — so the circuit is **provably equivariant** and entanglement is load-bearing;
  - an **equivariant readout**: Z₂-even observables produce the `FORWARD`/`STAY` logits, and an
    even+odd observable pair produces `LEFT`/`RIGHT` so the mirror swaps them exactly;
  - a plain-ANN **critic** over the detached pre-encoder latent. Trained with **PPO**.
- **Differentiation: an in-repo torch statevector simulator** (≤ ~10 qubits ⇒ ≤ 1024 amplitudes) giving
  exact **backprop-through-simulator** gradients — *not* Qiskit parameter-shift. **No new dependency**
  (`torch` is already present; Qiskit is not used for this brain).
- **Two ablation siblings, config-flagged on the same brain** (so they share env + readout + PPO loop):
  `equivariant: false` (unstructured PQC actor — the non-equivariant quantum control) and
  `quantum: false` (an equivariant *classical* MLP actor — the classical-equivariant control).
- **Registry integration**: `EQUIVARIANT_QUANTUM_PPO` `BrainType`, `@register_brain` decorator, module
  import in `brain/arch/__init__.py`; loads through the existing plugin registry without launcher
  branches.
- **Smoke config** `configs/scenarios/foraging/equivariantquantum_small_klinotaxis.yml` proving the
  brain trains end-to-end on klinotaxis foraging.
- **Tests** under `tests/quantumnematode_tests/brain/arch/test_equivariantquantum.py`, including a
  **mirror-consistency test** (validates the assigned observation parity vector against the live sensory
  code by reflecting the inputs across all four headings) and an **end-to-end policy-equivariance test**.

## Impact

- **No new dependency**: a small in-repo torch statevector simulator; reuses the `@register_brain`
  plugin registry, the `WeightPersistence` protocol, the classical sensory-feature pipeline
  (`extract_classical_features`), the PPO + rollout-buffer + GAE machinery pattern (mirroring the other
  PPO brains), and the `entropy_coef_end` / `entropy_decay_episodes` anneal hook.
- **Reuses / corrects**: the design depends on a **verified** observation parity vector — the predator
  *mechanosensation* zone feature is fore-aft (Z₂-**even**), not sign-flipping; the parity is assigned
  from the module layout, sized to align with `get_classical_feature_dimension` (STAM absorbs the env-set
  context remainder), and validated against the live sensory code by the mirror-consistency test.
- **Out of scope** (downstream or follow-ups): the combined-behaviour C3 cell config and the ranked
  cross-architecture evaluation (the comparison change); the `phase6-tracking` Decision-4 amendment that
  promotes the quantum family from SHOULD-in-T7 to an evaluated arm here; richer symmetry groups (the
  egocentric frame admits only Z₂ — no rotational symmetry); amplitude encoding (this cut uses
  angle encoding + re-uploading); a Qiskit/PennyLane backend (the in-repo torch simulator is the first
  cut); evolving the quantum weights (this brain is PPO-trained, not evolved).
