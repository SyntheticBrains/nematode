# Tasks: Add CfC / Liquid Brain (`CfCPPOBrain`)

## 1. Dependency + substrate

- [x] 1.1 Add `ncps` to `packages/quantum-nematode/pyproject.toml` and verify the de-risked API on the project's torch (2.7): `ncps.torch.CfC` + `ncps.wirings.AutoNCP`, single-step `(1,1,F)→(1,1,A)` + sequence `(B,T,F)→(B,T,A)` with hidden `(B,units)`. **Done in pre-flight**: `ncps==1.0.1`; `AutoNCP(32,4)` → 32 units / 116 synapses; motor-output = action logits; critic head on the 32-unit hidden state confirmed.
- [ ] 1.2 Add `CFC_PPO` to the `BrainType` enum in `brain/arch/dtypes.py` (value `"cfcppo"`).

## 2. Config

- [ ] 2.1 Add `CfCBrainConfig(BrainConfig)` with: `sensory_modules` (validated non-empty), `units: int = 32`, `ncp_sparsity: float = 0.5`, `cfc_mode: str = "default"`, `actor_head: str = "motor"` (one of `"motor"` | `"mlp"`; validated), `motor_logit_scale_init: float = 1.0` (init for the learnable motor-head logit temperature; used only when `actor_head == "motor"`), `actor_hidden_dim: int = 64`, `actor_num_layers: int = 2` (used only when `actor_head == "mlp"`), `critic_hidden_dim: int = 64`, `critic_num_layers: int = 2`, `actor_lr: float = 0.0003`, `critic_lr: float = 0.0003`, `clip_epsilon: float = 0.2`, `gamma: float = 0.99`, `gae_lambda: float = 0.95`, `value_loss_coef: float = 0.5`, `entropy_coef: float = 0.01`, `rollout_buffer_size: int = 512`, `num_epochs: int = 4`, `max_grad_norm: float = 0.5`, `bptt_chunk_length: int = 64`, plus the `BrainConfig` base fields (seed, etc.).

## 3. Brain implementation (`brain/arch/cfc_ppo.py`)

- [ ] 3.1 `CfCPPOBrain(ClassicalBrain)` with `@register_brain(name="cfcppo", config_cls=CfCBrainConfig, brain_type=BrainType.CFC_PPO, families=("classical",))`.
- [ ] 3.2 Networks: validate `units > num_actions + 2` at construction (AutoNCP requirement) with a clear error message. `feature_norm = nn.LayerNorm(input_dim)`; `cfc = CfC(input_dim, AutoNCP(units, num_actions, sparsity_level=ncp_sparsity), mode=cfc_mode)` (the `AutoNCP` wiring is used in BOTH head modes). **Actor head** per `actor_head`: `"motor"` → action logits are the CfC output scaled by a learnable temperature, `logits = motor_out * logit_scale` where `logit_scale` is an `nn.Parameter` initialised to `motor_logit_scale_init` (no actor MLP); `"mlp"` → an actor MLP `units → actor_hidden_dim (×actor_num_layers) → num_actions` on the (non-detached) recurrent hidden state, ignoring the motor-neuron output. Critic MLP `units → critic_hidden_dim (×critic_num_layers) → 1` on the **detached** hidden state. Separate Adam optimizers (actor: cfc + feature_norm + `logit_scale` in motor mode / actor MLP in mlp mode; critic: critic MLP).
- [ ] 3.3 Single hidden-state management: `h` shape `(1, units)`, zeroed in `prepare_episode()`, carried across steps.
- [ ] 3.4 `preprocess` (reuse `extract_classical_features`), `run_brain` (feature_norm → CfC single-step → logits → softmax → Categorical sample; stash pending features/action/log_prob/value/hidden-state), `learn` (append to buffer; on full → PPO update; reset), `post_process_episode`, `update_memory`, `copy`, `action_set` getter/setter with length validation, `build_brain`/`update_parameters` as the base requires.
- [ ] 3.5 Rollout buffer + `_perform_ppo_update` mirroring `LSTMPPOBrain` (GAE returns/advantages; truncated-BPTT replay of the CfC over `bptt_chunk_length` chunks to recompute log-probs/values; PPO clip objective + entropy bonus + value loss; `max_grad_norm`; `num_epochs`). Single hidden state stored per step (no cell state).
- [ ] 3.6 `WeightPersistence`: `get_weight_components` / `load_weight_components` for `cfc`, `critic`, `feature_norm`, `actor_optimizer`, `critic_optimizer`, `training_state`.
- [ ] 3.7 No TEI-prior path, no LayerNorm recurrent cell, no LR schedule (flat LRs). No milestone/tracking labels in code or docstrings.

## 4. Registry integration

- [ ] 4.1 Import the new module in `brain/arch/__init__.py` so the decorator runs (match how `lstmppo` / `feedforwardga` are imported).
- [ ] 4.2 Verify `utils/brain_factory.py` + `utils/config_loader.py` load `cfcppo` through the registry without a new per-architecture branch (the smoke run in §6 is the end-to-end check). Add a branch only if genuinely required, and document why.

## 5. Tests (`tests/quantumnematode_tests/brain/arch/test_cfcppo.py`)

- [ ] 5.1 Registry/Protocol conformance (brain builds via the registry with a minimal config).
- [ ] 5.2 Forward-pass shapes: logits `(num_actions,)`, finite; critic value scalar, finite. Output dim equals `num_actions`.
- [ ] 5.3 Hidden-state carry within an episode + reset on `prepare_episode()`.
- [ ] 5.4 A PPO update runs without error on a small synthetic rollout and leaves params finite.
- [ ] 5.5 Weight `get`/`load` round-trip → identical logits for the same input + hidden state.
- [ ] 5.6 Determinism under a fixed seed.
- [ ] 5.7 The `actor_head: "mlp"` variant: builds, forward-passes (logits `(num_actions,)`, finite), and runs a PPO step without error — both head modes exercised, with the actor MLP present only in `"mlp"` mode.
- [ ] 5.8 Motor-head logit temperature: with `logit_scale` raised to a large value, the motor head produces a peaked policy (one action softmax probability > 0.9) — guards against the bounded-motor-output saturation. Construction with `units <= num_actions + 2` raises a clear error.

## 6. Smoke config + verification

- [ ] 6.1 Create `configs/scenarios/foraging/cfcppo_small_klinotaxis.yml` mirroring `configs/scenarios/foraging/lstmppo_small_klinotaxis.yml` (identical env/reward/sensing; brain block `cfcppo`, units 32).
- [ ] 6.2 `uv run pytest -m "not nightly and not slow" tests/quantumnematode_tests/brain/arch/test_cfcppo.py` — green.
- [ ] 6.3 Short training smoke: `uv run ./scripts/run_simulation.py --log-level INFO --show-last-frame-only --runs 300 --config configs/scenarios/foraging/cfcppo_small_klinotaxis.yml --theme headless --seed 42` — loads through the registry and trains non-degenerately (report last-25 foraging success; should climb meaningfully above random).
- [ ] 6.4 `uv run ruff check` + `uv run pyright` clean on the new files.

## 7. Close-out

- [ ] 7.1 `openspec validate add-cfc-liquid-brain --strict` clean.
- [ ] 7.2 Add `cfcppo` to the brain enumeration + bump the count (20 → 21) in BOTH `AGENTS.md` (the `brain/arch/` list) and `openspec/config.yaml` (the architecture list); also update `docs/architecture/plugin-developer-guide.md` if it enumerates registered brains.
