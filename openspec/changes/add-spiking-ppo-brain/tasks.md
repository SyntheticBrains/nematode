# Tasks: Add Spiking Brain (`SpikingPPOBrain`)

## 1. Substrate + enum

- [ ] 1.1 Extend the in-repo spiking substrate (`brain/arch/_spiking_layers.py` or a sibling module):
      reuse `SurrogateGradientSpike` (the in-repo sigmoid-derivative surrogate; pass the slope α per
      forward so it is runtime-schedulable) as-is; add a
      **recurrent adaptive-LIF** cell — learnable per-neuron decay `β = σ(raw_β)`, adaptive threshold
      `θ = v_threshold + adapt_scale ⊙ a`, recurrent spike-feedback current `W_rec · s_prev`, soft reset
      — that carries `(v, a, s)` across calls (one tick per call). Add a non-spiking leaky-integrator
      readout cell carrying `m`. No SNN-library dependency.
- [ ] 1.2 Add `SPIKING_PPO` to the `BrainType` enum in `brain/arch/dtypes.py` (value `"spikingppo"`) and
      to the `BRAIN_TYPES` `Literal`. (`families=("spiking",)` auto-populates `SPIKING_BRAIN_TYPES` via
      the registry — no manual set edit.)

## 2. Config

- [ ] 2.1 Add `SpikingPPOBrainConfig(BrainConfig)` with: `sensory_modules` (validated non-empty);
      core — `hidden_size: int = 64`, `num_hidden_layers: int = 1`, `timesteps_per_step: int = 1`,
      `v_threshold: float = 1.0`, `membrane_decay_init: float = 0.9`, `adaptation_decay_init: float = 0.9`,
      `adapt_scale_init: float = 0.1`, `readout_decay_init: float = 0.9`;
      surrogate — `surrogate_slope: float = 2.0`, `surrogate_slope_end: float | None = None`,
      `surrogate_slope_anneal_episodes: int | None = None`; critic — `critic_hidden_dim: int = 64`,
      `critic_num_layers: int = 2`; PPO — `actor_lr: float = 0.0003`, `critic_lr: float = 0.0003`,
      `clip_epsilon: float = 0.2`, `gamma: float = 0.99`, `gae_lambda: float = 0.95`,
      `value_loss_coef: float = 0.5`, `entropy_coef: float = 0.01`, `entropy_coef_end: float | None = None`,
      `entropy_decay_episodes: int | None = None`, `rollout_buffer_size: int = 512`, `num_epochs: int = 4`,
      `max_grad_norm: float = 0.5`, `bptt_chunk_length: int = 64`; plus `device_type` and the `BrainConfig`
      base fields.
- [ ] 2.2 `@model_validator(mode="after")` checks: `sensory_modules` non-empty; positive/range guards on
      `hidden_size`, `timesteps_per_step`, `bptt_chunk_length ≤ rollout_buffer_size`, decays in `[0, 1)`,
      lrs `> 0`; and **paired-field** validators (set both or neither, mirroring the CfC entropy
      validator) for `entropy_coef_end` / `entropy_decay_episodes` **and** for `surrogate_slope_end` /
      `surrogate_slope_anneal_episodes`.

## 3. Brain implementation (`brain/arch/spiking_ppo.py`)

- [ ] 3.1 `SpikingPPOBrain(ClassicalBrain)` with
      `@register_brain(name="spikingppo", config_cls=SpikingPPOBrainConfig, brain_type=BrainType.SPIKING_PPO, families=("spiking",))`.
- [ ] 3.2 Networks: `feature_norm = nn.LayerNorm(input_dim)`; learnable direct-current encoder
      `Linear(input_dim → hidden_size)`; the recurrent adaptive-LIF hidden layer(s) (§1.1); the
      non-spiking leaky-integrator readout `→ num_actions`; critic MLP
      `hidden_size → critic_hidden_dim (×critic_num_layers) → 1` on the **detached** hidden membrane.
      Separate Adam optimizers (actor: encoder + recurrent core + readout; critic: critic MLP).
- [ ] 3.3 Single carried neuron state `(v, a, s, m)` zeroed in `prepare_episode()`, carried across steps,
      one LIF update per env-step (or `timesteps_per_step` inner ticks with a constant input current).
- [ ] 3.4 `preprocess` (reuse `extract_classical_features`), `run_brain` (feature_norm → encode current →
      recurrent LIF → readout membrane → logits → softmax → Categorical sample; stash pending
      features/action/log_prob/value/neuron-state), `learn` (append to buffer; on full → PPO update;
      reset), `post_process_episode` (increments the `_episode_count` the schedules read, mirroring CfC),
      `update_memory`, `copy` (may raise `NotImplementedError`, mirroring CfC), `action_set` getter/setter
      with length validation, `build_brain`/`update_parameters` as the base requires.
- [ ] 3.5 `SpikingPPORolloutBuffer` + `_perform_ppo_update` mirroring `CfCPPOBrain` (GAE; truncated-BPTT
      replay of the spiking core over `bptt_chunk_length` chunks recomputing log-probs/values through the
      surrogate; PPO clip objective + entropy bonus + value loss; `max_grad_norm`; `num_epochs`). Single
      per-step neuron state stored (detached at chunk boundaries). Reset neuron state at episode `dones`.
- [ ] 3.6 `_get_entropy_coef()` (reuse the CfC linear-anneal schedule) and `_get_surrogate_slope()`
      (analogous episode-based schedule; flat when the pair is unset): the entropy coef is consumed in the
      PPO update; the surrogate slope α is applied in the spike surrogate's backward, exercised during the
      BPTT replay.
- [ ] 3.7 `WeightPersistence`: `get_weight_components` / `load_weight_components` for the input encoder,
      recurrent core (recurrent weights + learnable decay/adaptation params), readout, critic, both
      optimizers, and a training-state counter. Per-step neuron state is NOT persisted.
- [ ] 3.8 No milestone/tracking labels in code or docstrings.

## 4. Registry integration

- [ ] 4.1 Import the new module in `brain/arch/__init__.py` (and add the class + config to `__all__`) so
      the decorator runs (match how `cfcppo` / `lstmppo` are imported).
- [ ] 4.2 Add `SpikingPPOBrainConfig` to the `BrainConfigType` union in `utils/config_loader.py` (+ its
      import). Verify `utils/brain_factory.py` loads `spikingppo` through the registry without a new
      per-architecture branch (modular `input_dim` ⇒ no branch needed); add one only if genuinely
      required, and document why.

## 5. Tests (`tests/quantumnematode_tests/brain/arch/test_spikingppo.py`)

- [ ] 5.1 Registry/Protocol conformance (brain builds via the registry with a minimal config).
- [ ] 5.2 Forward-pass shapes: logits `(num_actions,)`, finite; critic value scalar, finite; output dim
      equals `num_actions`.
- [ ] 5.3 Recurrent neuron state carries within an episode + resets on `prepare_episode()`; two identical
      inputs at different in-episode steps can differ (carried state) — i.e. genuinely recurrent.
- [ ] 5.4 A PPO update runs without error on a small synthetic rollout, with gradient flowing through the
      surrogate, leaving params finite.
- [ ] 5.5 Weight `get`/`load` round-trip → identical logits for the same input + neuron state.
- [ ] 5.6 Determinism under a fixed seed.
- [ ] 5.7 Substrate-level tests for the recurrent adaptive-LIF cell (decay, soft reset, adaptation,
      recurrent current, surrogate gradient is finite/non-zero); carry over the existing
      `SurrogateGradientSpike` / `LIFLayer` tests unchanged.
- [ ] 5.8 Schedule validators: `entropy_coef_end` / `entropy_decay_episodes` and `surrogate_slope_end` /
      `surrogate_slope_anneal_episodes` each reject a half-set pair with a clear error; flat when unset;
      anneal-then-hold when set.
- [ ] 5.9 Non-degenerate variance: over a sample of ≥ 100 forward passes with non-zero gradients, the
      4-action logit variance is strictly > 0 and the policy does not collapse to a constant action
      (covers the spec's variance scenario; mirror CfC's `TestVariance`).

## 6. Smoke config + verification

- [ ] 6.1 Create `configs/scenarios/foraging/spikingppo_small_klinotaxis.yml` mirroring
      `configs/scenarios/foraging/cfcppo_small_klinotaxis.yml` (identical env/reward/sensing; brain block
      `spikingppo`).
- [ ] 6.2 `uv run pytest -m "not nightly and not slow" tests/quantumnematode_tests/brain/arch/test_spikingppo.py` — green.
- [ ] 6.3 Short training smoke:
      `uv run ./scripts/run_simulation.py --log-level INFO --show-last-frame-only --runs 300 --config configs/scenarios/foraging/spikingppo_small_klinotaxis.yml --theme headless --seed 42` —
      loads through the registry and trains non-degenerately (report last-25 foraging success; should
      climb meaningfully above random).
- [ ] 6.4 `uv run ruff check` + `uv run pyright` clean on the new files.

## 7. Close-out

- [ ] 7.1 `openspec validate add-spiking-ppo-brain --strict` clean.
- [ ] 7.2 Add `spikingppo` to the brain enumeration + bump the count to **23** in BOTH `AGENTS.md` (the
      `brain/arch/` list) and `openspec/config.yaml` (the architecture list).
