# Tasks: Configurable MLP actor head for `SpikingPPOBrain`

## 1. Config

- [x] 1.1 Add to `SpikingPPOBrainConfig`: `actor_head: str = "spike"`, `actor_hidden_dim: int = 64`,
  `actor_num_layers: int = 2`.
- [x] 1.2 In `_validate_config`: reject `actor_head` outside `{"spike", "mlp"}` with a clear error;
  guard `actor_hidden_dim >= 1` and `actor_num_layers >= 1`.

## 2. Brain implementation (`brain/arch/spiking_ppo.py`)

- [x] 2.1 In `__init__`, when `config.actor_head == "mlp"`, build `self.actor_mlp` — an `nn.Sequential`
  `Linear(hidden_size → actor_hidden_dim) → ReLU` repeated `actor_num_layers - 1` times, then
  `Linear(→ num_actions)`. Otherwise `self.actor_mlp = None`. Build it before the actor optimizer so
  it is included.
- [x] 2.2 In `_core_forward`, when `self.actor_mlp is not None`, set `logits = self.actor_mlp(new_v[-1]).squeeze(0)`
  (the top hidden layer's membrane, non-detached); else keep the leaky-integrator readout membrane as
  logits. The readout is still advanced (carried neuron state) in both modes.
- [x] 2.3 In `_actor_parameters()`, append `self.actor_mlp.parameters()` when present (so it is in the
  actor optimizer + `max_grad_norm` clip).
- [x] 2.4 In `WeightPersistence` (`get_weight_components` / `load_weight_components`), expose/load an
  `"actor_mlp"` component when present; `load_weight_components` SHALL tolerate an absent `actor_mlp`
  entry (same-mode round-trip assumed; restoring into spike mode must not error). Per-step neuron
  state unchanged.
- [x] 2.5 No milestone/tracking labels in code or docstrings.

## 3. Tests (`tests/quantumnematode_tests/brain/arch/test_spikingppo.py`)

- [x] 3.1 Default `actor_head: "spike"` builds no `actor_mlp`; forward-pass + PPO update unchanged
  (existing tests still green).
- [x] 3.2 `actor_head: "mlp"`: builds the actor MLP; forward-pass shapes (logits `(num_actions,)`, finite);
  a PPO update runs and gradient flows through the surrogate into the recurrent core — assert the
  learnable membrane decay (`raw_membrane_decay`) grad is finite + non-zero (spike-density-independent,
  per the multi-layer-test lesson); the recurrent weight grad need only be finite. Params stay finite.
- [x] 3.3 Weight `get`/`load` round-trip in `mlp` mode → identical logits for the same input + neuron state
  (covers the `"actor_mlp"` component).
- [x] 3.4 `_validate_config` rejects `actor_head` not in `{"spike", "mlp"}` with a clear error.

## 4. Verification

- [x] 4.1 `uv run pytest -m "not nightly and not slow" tests/quantumnematode_tests/brain/arch/test_spikingppo.py` — green.
- [x] 4.2 `uv run ruff check` + `uv run pyright` clean on the changed files.
- [x] 4.3 Short MLP-head smoke: load a config with `actor_head: "mlp"` through the registry and confirm it
  trains non-degenerately on klinotaxis foraging.

## 5. Close-out

- [x] 5.1 `openspec validate add-spiking-mlp-actor-head --strict` clean.
- [x] 5.2 No brain-count change (this enhances an existing brain, does not add one); no `AGENTS.md` /
  `openspec/config.yaml` count edit.

## Implementation notes (deviations from the as-drafted spec)

- **`load_weight_components` carries `# noqa: C901`** — the added `actor_mlp` branch pushed mccabe
  complexity 10 → 11; the noqa matches the `_validate_config` precedent (a flat sequence of
  conditional loads, not genuinely complex).
- **Cross-mode full loads are unsupported by construction** (the actor optimizer's parameter set is
  mode-specific: an `mlp` brain's actor optimizer covers the actor MLP, a `spike` brain's does not). The
  S2 graceful contract is therefore scoped to an **absent `actor_mlp` component** (skipped, MLP keeps its
  init) — the test exercises exactly that, not a full `mlp`→`spike` restore.
- **The actor MLP is built inline** in `__init__` (kept self-contained) rather than reusing CfC's
  `_build_actor_mlp`, to avoid cross-brain coupling.
- **End-to-end smoke (seed 42, 300 ep foraging, `actor_head: "mlp"`): 90.67% success** — trains
  non-degenerately and faster/higher than the spike head's foraging smoke (31% overall).
