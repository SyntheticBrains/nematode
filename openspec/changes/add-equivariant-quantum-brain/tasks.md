# Tasks: Add Equivariant Quantum Brain (`EquivariantQuantumPPOBrain`)

## 1. Symmetry derivation + enum

- [x] 1.1 Implement the **mirror-consistency helper**: for **each of the four headings
  (UP/DOWN/LEFT/RIGHT)**, build an env state, reflect it across the agent's egocentric forward axis,
  recompute the observation via the live sensory pipeline, and derive the parity vector `p ∈ {+1,−1}^d`
  per feature. Assert `p` is **identical across all four headings** (fail if any feature's parity is
  heading-dependent). Flag any feature that transforms as neither `+1` nor `−1`; symmetrise it (project
  onto its even part) or route it as an even side-input, logging the deviation. Confirm the expected
  assignment: the 3 lateral-gradient *angle* features → `−1`; strengths, temporal-derivatives,
  predator-mechano fore-aft zone, and proprioception → `+1`.
- [x] 1.2 Add `EQUIVARIANT_QUANTUM_PPO` to the `BrainType` enum in `brain/arch/dtypes.py` (value
  `"equivariantquantum"`) and to the `BRAIN_TYPES` `Literal`. (`families=("quantum",)` auto-populates
  the quantum family set via the registry.)

## 2. Config

- [x] 2.1 Add `EquivariantQuantumPPOBrainConfig(BrainConfig)` with: `sensory_modules` (validated
  non-empty); circuit — `num_qubits: int = 8` (interpreted as `k_even + k_odd`), `k_odd: int = 3`,
  `num_layers: int = 3` (data-re-uploading layers); ablation flags — `equivariant: bool = True`,
  `quantum: bool = True`; critic — `critic_hidden_dim: int = 64`, `critic_num_layers: int = 2`;
  PPO — `actor_lr: float = 0.0003`, `critic_lr: float = 0.0003`, `clip_epsilon: float = 0.2`,
  `gamma: float = 0.99`, `gae_lambda: float = 0.95`, `value_loss_coef: float = 0.5`,
  `entropy_coef: float = 0.01`, `entropy_coef_end: float | None = None`,
  `entropy_decay_episodes: int | None = None`, `rollout_buffer_size: int = 512`, `num_epochs: int = 4`,
  `max_grad_norm: float = 0.5`; plus `device_type` and the `BrainConfig` base fields.
- [x] 2.2 `@model_validator(mode="after")` checks: `sensory_modules` non-empty; `0 < k_odd < num_qubits`;
  positive/range guards on `num_qubits` (≤ 10, matching the ≤ 1024-amplitude statevector budget),
  `num_layers`, lrs `> 0`, `clip_epsilon ∈ (0,1)`; and the **paired-field** validator for
  `entropy_coef_end` / `entropy_decay_episodes` (set both or neither, mirroring the CfC/spiking entropy
  validator). At construction (where the sensory dim is known), **warn** when `k_odd > num_odd_inputs`
  (the surplus odd latents are linearly dependent under the equivariant linear map).

## 3. In-repo statevector simulator (`brain/arch/_quantum_statevector.py`)

- [x] 3.1 A small differentiable `torch` statevector simulator: state as a complex `torch.Tensor` of
  `2**num_qubits` amplitudes; gate application (`RX/RY/RZ`, `CZ`, `IsingXX`, `IsingZZ`) as tensor
  contractions; Pauli-observable expectation values (`⟨X_q⟩`, `⟨Z_q⟩`, and the even/odd readout
  observables). All ops autograd-differentiable.
- [x] 3.2 Define the reflection unitary `U_R = ⊗_{q ∈ odd} X_q` and helpers to classify observables as
  Z₂-even / Z₂-odd, plus the `U_R`-invariant ansatz gate set (`RX` all, `RZ` even, `IsingXX` any pair,
  `IsingZZ` same-parity pair). Provide the **`U_R`-invariant reference-state preparation** (Hadamard on
  every odd qubit → `|+⟩`; even qubits `|0⟩`) applied before encoding — required for end-to-end
  equivariance (`|0…0⟩` is not `U_R`-invariant and breaks the `LEFT`/`RIGHT` swap).

## 4. Brain implementation (`brain/arch/equivariant_quantum.py`)

- [x] 4.1 `EquivariantQuantumPPOBrain(ClassicalBrain)` with
  `@register_brain(name="equivariantquantum", config_cls=EquivariantQuantumPPOBrainConfig, brain_type=BrainType.EQUIVARIANT_QUANTUM_PPO, families=("quantum",))`.
- [x] 4.2 **Equivariant pre-encoder**: a parity-block-structured linear map `W` (`W[i,j]=0` unless
  `p_in[j]=q_out[i]`; bias on even outputs only) → `k_even` even + `k_odd` odd latents. Build the
  parity mask from §1.1's `p`.
- [x] 4.3 **Equivariant circuit**: start from the `U_R`-invariant reference state (§3.2; `H` on odd
  qubits), then angle-encode latents (`RY` per qubit; even latents on even qubits, odd latents on odd
  qubits), `num_layers` re-uploading layers interleaved with the `U_R`-invariant variational ansatz
  (entangling `IsingXX`/`IsingZZ`).
- [x] 4.4 **Equivariant readout**: even observables → `FORWARD`/`STAY` logits; even+odd observable pair →
  `LEFT = ⟨O_e⟩+⟨O_o⟩`, `RIGHT = ⟨O_e⟩−⟨O_o⟩`; optional even-typed per-logit learnable scale/bias.
  Logit order = `[FORWARD, LEFT, RIGHT, STAY]`.
- [x] 4.5 **Ablation branches**: `equivariant=False` → unstructured PQC (`RX/RY/RZ`+`CZ`, free linear
  head); `quantum=False` → equivariant classical MLP (parity-block weights, same readout typing).
- [x] 4.6 **Critic**: 2-layer MLP over the detached latent.
- [x] 4.7 `run_brain()` / `update_memory()` / `prepare_episode()` / `post_process_episode()` / `copy()`
  per the `Brain`/`ClassicalBrain` protocol, plus stub `build_brain` / `update_parameters` and an
  `action_set` setter (mirroring cfc/spiking; `copy()` may raise `NotImplementedError` per those
  precedents); PPO `learn()` with rollout buffer + GAE + clipped surrogate + entropy bonus (+ anneal) +
  value loss + grad-norm clip, mirroring the other PPO arms.
- [x] 4.8 `WeightPersistence`: `get_weight_components()` / `load_weight_components()` over pre-encoder,
  quantum params, readout scales, critic.

## 5. Registration

- [x] 5.1 Import the module in `brain/arch/__init__.py` so the `@register_brain` decorator runs (the
  `assert_registry_matches_enum()` call there will fail loudly if the enum/import are out of sync). Add
  `EquivariantQuantumPPOBrainConfig` (+ import) to the `BrainConfigType` union in `config_loader.py`
  (required for pyright — the runtime `BRAIN_CONFIG_MAP` is registry-derived, but the union is the static
  type of the parsed brain-config field; the spiking precedent did this). `brain_factory.py` needs **no**
  per-architecture branch (deriving `input_dim` from `sensory_modules` lets it fall through to the
  default `{"num_actions": 4, "device": device}`, as cfcppo/spikingppo do).

## 6. Smoke config

- [x] 6.1 `configs/scenarios/foraging/equivariantquantum_small_klinotaxis.yml` — klinotaxis foraging,
  `name: equivariantquantum`, small circuit. Foraging-only klinotaxis has exactly **one** odd feature
  (food lateral gradient), so set `k_odd: 1` and a small `num_qubits` (e.g. `4`) to avoid a
  rank-deficient odd block. Header documents the honest framing (no advantage claim) and the two ablation
  flags.

## 7. Tests (`tests/quantumnematode_tests/brain/arch/test_equivariantquantum.py`)

- [x] 7.1 **Mirror-consistency** (all four headings): reflected env ⇒ recomputed observation equals
  `R·obs`; lateral-gradient angles flip, the rest invariant; the derived `p` is identical across
  UP/DOWN/LEFT/RIGHT; any non-`±1` feature is flagged/handled.
- [x] 7.2 **End-to-end policy equivariance** (`equivariant: true`): `P(LEFT|s)=P(RIGHT|R·s)` &
  `FORWARD`/`STAY` fixed, over ≥100 random inputs; **and** the classical-equivariant ablation passes the
  same check; **and** the non-equivariant ablation does **not** (guard that it is genuinely unstructured).
- [x] 7.3 **Simulator vs reference**: torch statevector matches Qiskit `Statevector` on fixed circuits;
  outputs differentiable (finite autograd gradient w.r.t. circuit params).
- [x] 7.4 **Entanglement load-bearing**: removing `IsingXX`/`IsingZZ` ⇒ separable state + different logit
  distribution on a fixed batch.
- [x] 7.5 Construction (shapes, finite logits + value, `DEFAULT_ACTIONS` order); non-degenerate logit
  variance over ≥100 passes; PPO update runs and leaves params finite; paired entropy-field validator;
  `WeightPersistence` round-trip equality; registry load via config `name`.

## 8. Validation + docs

- [x] 8.1 `openspec validate add-equivariant-quantum-brain --strict` clean.
- [x] 8.2 `uv run pytest -m "not nightly and not slow" .../test_equivariantquantum.py` green; targeted
  pre-commit on changed files clean (ruff check + format, pyright).
- [x] 8.3 Smoke run: `uv run python scripts/run_simulation.py --config configs/scenarios/foraging/equivariantquantum_small_klinotaxis.yml --theme headless --runs <n> --seed 2026` trains end-to-end (non-degenerate learning curve).
- [x] 8.4 Note in the brain's module docstring + config header: genuinely-quantum (trained + entangled),
  bilateral-Z₂-equivariant, honest non-advantage framing — no milestone/tranche labels in code.
- [x] 8.5 Architecture-count bookkeeping (mirroring the spiking precedent): append `equivariantquantum`
  to the brain list and bump the architecture count in **both** `AGENTS.md` and `openspec/config.yaml`
  (verify the current count and increment by one).
