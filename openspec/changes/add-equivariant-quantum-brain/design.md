# Design: Add Equivariant Quantum Brain (`EquivariantQuantumPPOBrain`)

## Context

The Phase 6 comparison needs a **genuinely-quantum** arm: trained quantum parameters, entanglement as a
load-bearing resource — not a fixed reservoir + classical readout. A pre-build investigation (local
quantum-stack survey + verified logbook 008 + a thorough 2025–2026 literature scan) established three
constraints that jointly determine the architecture:

1. **No quantum advantage is expected** on small classical-data RL at simulator scale, and small
   trainable circuits that avoid barren plateaus are generically classically simulable (Cerezo et al.,
   *Nat. Commun.* 2025). So the arm is framed as an honest comparison/inductive-bias study, not an
   advantage claim.
2. **Unstructured entanglement does not pay off on gridworld-PPO** (Kruse et al. 2025; "Dissecting QRL"
   2025). So a vanilla entangled PQC actor would be quantum-on-paper but unmotivated.
3. **Symmetry-equivariant circuits are the one family with proven barren-plateau-resistance +
   load-bearing entanglement at small scale.** So encoding a task symmetry is both the trainability fix
   and the scientific question.

The task's symmetry is fixed by its frame: actions are **egocentric** (`FORWARD, LEFT, RIGHT, STAY` —
`LEFT`/`RIGHT` rotate the heading; verified in `env.py`'s `DIRECTION_MAP`) and klinotaxis sensing is
**egocentric** (lateral gradients `right − left` relative to heading; `agent/agent.py:_compute_lateral_offsets`).
The only spatial symmetry preserving the egocentric frame is the **left–right mirror, Z₂** — *C. elegans*'
bilateral symmetry and the natural symmetry of klinotaxis. There is **no rotational symmetry** (the agent
always faces "forward"), so Z₂ — not grid-D4 — is correct.

## Decision 1 — Bilateral Z₂ symmetry, acting as a fixed parity operator on the observation

The symmetry group is `Z₂ = {e, r}` (`r` = left–right mirror, `r² = e`). It acts on the policy as
`π(ρ_A(r)·a | ρ_S(r)·s) = π(a | s)` for the optimal policy, where:

- **`ρ_A(r)` (action mirror)**: swaps `LEFT`↔`RIGHT`, fixes `FORWARD` and `STAY`.
- **`ρ_S(r)` (observation mirror)**: because the observation is **egocentric**, `r` acts as a *fixed,
  heading-independent* signed-permutation `R = diag(p)`, `p ∈ {+1,−1}^d`:
  - **Z₂-odd (`p = −1`, sign-flip)**: the lateral-gradient *angle* features — `food_angle`,
    `predator_chemosensation_angle`, `thermotaxis_angle` (`tanh(right − left)` → sign-flips under
    `left↔right`).
  - **Z₂-even (`p = +1`, invariant)**: all *strength* and *temporal-derivative* features; the predator
    **mechanosensation zone** angle (anterior/posterior is a **fore-aft** axis — unchanged by a
    left–right mirror; this corrects a naive "sign-flip" assignment); proprioception (the mirror fixes
    the agent's own heading axis by construction); and STAM channels per their buffered quantity's parity.

**The parity vector `p` is not hand-assigned — it is derived empirically** by a **mirror-consistency
test** (Decision 10): construct env states **across all four headings (UP/DOWN/LEFT/RIGHT)**, reflect
each across the agent's forward axis, recompute the observation, and read off which features are
invariant (`+1`), sign-flipped (`−1`), or neither. The derived `p` SHALL be **identical across all four
headings** — this is what makes `R = diag(p)` a valid *heading-independent* operator (it empirically
confirms the proprioception-is-even claim rather than assuming it); a feature whose parity is
heading-dependent SHALL fail the test. Any feature that does not transform as a clean `±1` (e.g. an
ambiguous STAM channel) is
**symmetrised** (projected onto its even part) or routed as an even side-input, with the deviation logged.
This makes the construction robust to the exact sensory implementation.

## Decision 2 — Equivariant classical pre-encoder (parity-block linear map)

The observation (~25-D for the C3 stack) is compressed to `num_qubits` Z₂-typed latent features by a
linear map `W` that is **Z₂-equivariant**: `W·diag(p_in) = diag(q_out)·W`, which forces
`W[i,j] = 0` unless `p_in[j] = q_out[i]` — i.e. `W` is **block-diagonal in parity** (even-inputs → even
latents, odd-inputs → odd latents), with a bias only on even outputs. The latent split is
`num_qubits = k_even + k_odd` (default `k_even = 5`, `k_odd = 3`). For a linear map the odd block has
rank ≤ the number of odd input features (3 lateral gradients in the full C3 stack; only 1 in
foraging-only sensing), so construction SHALL warn when `k_odd > num_odd_inputs` (the surplus odd latents
are linearly dependent). This is the standard equivariant-linear construction; an optional shallow
parity-respecting MLP is held in reserve. Keeping the pre-encoder **minimal** keeps the quantum circuit
load-bearing.

## Decision 3 — Z₂-equivariant parameterised quantum circuit (Pauli-X representation)

The reflection is represented on the `num_qubits` qubits as `U_R = ⊗_{q ∈ odd} X_q` (an involution).
One qubit per latent feature; even latents on "even" qubits (`U_R = I`), odd latents on "odd" qubits
(`U_R = X`).

- **`U_R`-invariant reference state (REQUIRED for end-to-end equivariance)**: before encoding, prepare
  the odd-parity qubits in `|+⟩` (one Hadamard each; `X|+⟩ = |+⟩`) and the even-parity qubits in `|0⟩`,
  so the start state satisfies `U_R|ψ₀⟩ = |ψ₀⟩`. This is essential: operator-level equivariance
  (`U_R·C(x)·U_R† = C(R·x)`) only yields the policy-level `LEFT`↔`RIGHT` swap if `U_R†|ψ₀⟩ = |ψ₀⟩`;
  with the naive `|0…0⟩` start `U_R|0…0⟩ ≠ |0…0⟩` (the odd qubits flip) and the swap **fails** despite a
  correct circuit. The `H`-on-odd layer is even (it does not break the readout typing).
- **Equivariant encoding** (angle encoding, `L` data-re-uploading layers): each latent `x_j` is encoded
  as `RY_q(x_j)`. On an even qubit `U_R RY_q(x) U_R† = RY_q(x)` (invariant input ✓); on an odd qubit
  `U_R RY_q(x) U_R† = RY_q(−x)` (since `X·RY(θ)·X = RY(−θ)`), matching the odd input's sign-flip ✓. So
  `U_R · Enc(x) · U_R† = Enc(R·x)` exactly.
- **Invariant variational ansatz** (entanglement load-bearing): gates drawn from the `U_R`-commuting set
  — `RX_q` (all qubits), `RZ_q` (even qubits), **`IsingXX(θ)` on any pair**, **`IsingZZ(θ)` on
  same-parity pairs**. Each commutes with `U_R`, so `U_R·A(θ)·U_R† = A(θ)`; the `IsingXX`/`IsingZZ`
  couplings entangle qubits across the even/odd blocks, so removing them collapses the policy (the
  entanglement is functional, verified by an ablation test).

Trainability rests on the equivariant gauge (proven BP-resistance for this construction class), shallow
depth (`L`, `num_layers` small), and backprop-through-simulator gradients (Decision 6).

## Decision 4 — Equivariant readout (action logits transform as `ρ_A(r)`)

Logits are expectation values of Pauli observables chosen so the mirror acts correctly:

- `l_FORWARD = ⟨O_F⟩`, `l_STAY = ⟨O_S⟩` with `O_F, O_S` **Z₂-even** (`U_R O U_R† = O`; e.g. `⟨X_q⟩`, or
  `⟨Z_q⟩` on even qubits) → invariant under `r` ✓.
- `l_LEFT = ⟨O_e⟩ + ⟨O_o⟩`, `l_RIGHT = ⟨O_e⟩ − ⟨O_o⟩` with `O_e` **even** and `O_o` **Z₂-odd**
  (`U_R O_o U_R† = −O_o`; e.g. `⟨Z_q⟩` on a designated odd qubit). Under `r`, `⟨O_o⟩ → −⟨O_o⟩`, so
  `l_LEFT ↔ l_RIGHT` swap exactly ✓.

This makes the **whole policy** Z₂-equivariant by construction (pre-encoder + circuit + readout), not
approximately — verified end-to-end by test (Decision 10). A small learnable per-logit scale/bias
(even-typed) is allowed without breaking equivariance.

## Decision 5 — Plain-ANN critic on the detached pre-encoder latent

The critic is a 2-layer MLP estimating state value from the **detached** even/odd latent (value need not
be equivariant — it is a scalar invariant; feeding the detached latent avoids the critic gradient
perturbing the quantum actor's encoder). Mirrors the critic pattern of the other PPO arms.

## Decision 6 — Differentiation: in-repo torch statevector simulator (not parameter-shift)

For `num_qubits ≤ ~10`, the statevector is ≤ 1024 complex amplitudes. A small in-repo simulator applies
gates as `torch` tensor contractions, so gradients flow by **backprop-through-the-simulator** —
exact, and per-step cost comparable to a small MLP. This is the literature-recommended path for
trainable PQC-RL at simulator scale; Qiskit **parameter-shift** (2 circuit evals/param/step) would make
1000s of episodes intractable and is explicitly *not* used. No new dependency (vs adding PennyLane
`lightning.qubit`, which also risks the `qiskit<2.0` / `torch` pin). The simulator is validated against
Qiskit on a handful of fixed circuits in tests.

## Decision 7 — Two ablation siblings, config-flagged on the same brain

To attribute any measured effect, the same brain exposes two boolean flags sharing the env, readout
shape, and PPO loop:

- `equivariant: false` → an **unstructured** PQC actor (arbitrary `RX/RY/RZ` + `CZ` ansatz, free
  measurement→logit linear head). The **non-equivariant quantum** control → isolates the symmetry prior.
- `quantum: false` → an **equivariant classical** MLP actor (parity-block-structured weights, same
  readout typing). The **classical-equivariant** control → isolates the "quantum" contribution.

The two scientific deltas the comparison reports: *equivariant − non-equivariant* (does the bilateral
prior help?) and *equivariant-quantum − equivariant-classical* (does the entangled circuit add anything
over a classical equivariant net?).

## Decision 8 — PPO with the entropy-coefficient anneal wired in from day one

Trained on-policy with PPO (clipped surrogate + GAE + value loss + entropy bonus), mirroring the other
PPO arms. The `entropy_coef → entropy_coef_end over entropy_decay_episodes` anneal (proven on LSTM / CfC
/ spiking) is included from the start. No truncated-BPTT is needed — the actor is feedforward per step
(temporal memory enters via STAM features, not recurrence).

## Decision 9 — Honest ceiling: competitive-or-below, pre-registered

Per the scan, the literature-supported expectation is **competitive-or-below a parameter-matched
classical baseline** — no quantum advantage, and the bilateral prior *may* help sample efficiency (as it
does in classical equivariant RL) but the entanglement *may not* pay off. Success is pre-registered as:
the brain **trains to a documented result**, and the two ablation deltas (Decision 7) are **measured and
reported** — not "beats the field". This protects against goalpost-moving and frames the arm honestly.

## Decision 10 — Equivariance is verified by test, not asserted

Two tests are load-bearing design artifacts:

- **Mirror-consistency** (Decision 1): derive `p` empirically from the real sensory code; fail if a
  feature claimed even/odd does not transform as `±1` (catches sensory-code drift).
- **End-to-end policy equivariance**: for random inputs `s`, assert
  `softmax(logits(R·s)) == ρ_A(r)·softmax(logits(s))` within tolerance (the `LEFT`/`RIGHT` entries swap,
  `FORWARD`/`STAY` fixed). This must hold for the `equivariant: true` brain and the equivariant-classical
  ablation, and must **fail** for the non-equivariant ablation (a guard that the ablation is genuinely
  unstructured).

## Decision 11 — WeightPersistence yes, evolution encoder no

Implements `WeightPersistence` (checkpoint round-trip of pre-encoder, quantum parameters, readout
scales, critic). No `ENCODER_REGISTRY` entry — this brain is PPO-trained, not evolved.

## Risks

- **The brain trains but lands clearly below the field.** Pre-registered as an acceptable, honest
  outcome (Decision 9); the value is the symmetry/quantum deltas, not the rank.
- **Entanglement turns out non-load-bearing** (the `IsingXX/ZZ` ablation matches the product-state
  circuit). This is itself a reportable finding (consistent with the gridworld-PPO literature), not a
  failure — documented, not hidden.
- **STAM channels lack clean parity.** Mitigated by the mirror-consistency test's symmetrise-or-route
  fallback (Decision 1); worst case the quantum cell uses a STAM-free observation, documented as a
  per-brain interface choice driven by the qubit budget.
- **Statevector simulator correctness.** Mitigated by validating against Qiskit on fixed circuits and by
  the equivariance tests (a buggy simulator would break exact equivariance).
- **Qubit budget vs observation width.** The equivariant pre-encoder compresses to `num_qubits`; if
  `k_odd` is starved (few odd inputs), the odd channel is thin — acceptable (the 3 lateral gradients are
  the symmetry-bearing signals), documented.
