# Tasks: Configurable MLP actor head for `SpikingPPOBrain`

## 1. Config

- [ ] 1.1 Add to `SpikingPPOBrainConfig`: `actor_head: str = "spike"`, `actor_hidden_dim: int = 64`,
      `actor_num_layers: int = 2`.
- [ ] 1.2 In `_validate_config`: reject `actor_head` outside `{"spike", "mlp"}` with a clear error;
      guard `actor_hidden_dim >= 1` and `actor_num_layers >= 1`.

## 2. Brain implementation (`brain/arch/spiking_ppo.py`)

- [ ] 2.1 In `__init__`, when `config.actor_head == "mlp"`, build `self.actor_mlp` — an `nn.Sequential`
      `Linear(hidden_size → actor_hidden_dim) → ReLU` repeated `actor_num_layers - 1` times, then
      `Linear(→ num_actions)`. Otherwise `self.actor_mlp = None`. Build it before the actor optimizer so
      it is included.
- [ ] 2.2 In `_core_forward`, when `self.actor_mlp is not None`, set `logits = self.actor_mlp(new_v[-1]).squeeze(0)`
      (the top hidden layer's membrane, non-detached); else keep the leaky-integrator readout membrane as
      logits. The readout is still advanced (carried neuron state) in both modes.
- [ ] 2.3 In `_actor_parameters()`, append `self.actor_mlp.parameters()` when present (so it is in the
      actor optimizer + `max_grad_norm` clip).
- [ ] 2.4 In `WeightPersistence` (`get_weight_components` / `load_weight_components`), expose/load an
      `"actor_mlp"` component when present. Per-step neuron state unchanged.
- [ ] 2.5 No milestone/tracking labels in code or docstrings.

## 3. Tests (`tests/quantumnematode_tests/brain/arch/test_spikingppo.py`)

- [ ] 3.1 Default `actor_head: "spike"` builds no `actor_mlp`; forward-pass + PPO update unchanged
      (existing tests still green).
- [ ] 3.2 `actor_head: "mlp"`: builds the actor MLP; forward-pass shapes (logits `(num_actions,)`, finite);
      a PPO update runs and gradient flows through the surrogate into the recurrent core (recurrent
      weight grad finite/non-zero), params finite.
- [ ] 3.3 Weight `get`/`load` round-trip in `mlp` mode → identical logits for the same input + neuron state
      (covers the `"actor_mlp"` component).
- [ ] 3.4 `_validate_config` rejects `actor_head` not in `{"spike", "mlp"}` with a clear error.

## 4. Verification

- [ ] 4.1 `uv run pytest -m "not nightly and not slow" tests/quantumnematode_tests/brain/arch/test_spikingppo.py` — green.
- [ ] 4.2 `uv run ruff check` + `uv run pyright` clean on the changed files.
- [ ] 4.3 Short MLP-head smoke: load a config with `actor_head: "mlp"` through the registry and confirm it
      trains non-degenerately on klinotaxis foraging.

## 5. Close-out

- [ ] 5.1 `openspec validate add-spiking-mlp-actor-head --strict` clean.
- [ ] 5.2 No brain-count change (this enhances an existing brain, does not add one); no `AGENTS.md` /
      `openspec/config.yaml` count edit.
