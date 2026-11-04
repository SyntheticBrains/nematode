## 1. Baselines

- [x] 1.1 `connectome_identity_check.py baseline --build build/c2-speed-noise/identity` on the code before any change.

## 2. The policy

- [x] 2.1 `_policy.py`: `ou_noise_step(eps, prev, rho)`, and `continuous_sample_tanh_gaussian_ou(mean, log_std, low, high, prev, rho)`, which draws the same five normals as the independent sampler, steps speed's alone, and returns the noise state. Also `continuous_evaluate_tanh_gaussian_ou(mean, log_std, pre, prev_mean, prev_speed, live, rho)`, the exact conditional of D3, falling back to the marginal where `live` is false. The independent functions stay as they are.
- [x] 2.2 Unit tests:
  - the stationary variance and the lag-1 autocorrelation;
  - an episode's first step equals the independent sampler, and outputs 1–4 equal it at every step;
  - the evaluator at the sampling parameters returns the sampler's log-prob;
  - under changed parameters it returns the analytic exact conditional.

## 3. The brain

- [x] 3.1 `ConnectomePPOBrainConfig.encoder_speed_noise: Literal["none", "ou"] = "none"`. Add `_speed_noise_refusal` to the validator loop, and a call at construction beside the other shared refusals (D6).
- [x] 3.2 The noise state: `_speed_noise_rho` (set by `write_encoder` and on load) and the per-episode state, reset in `prepare_episode`. `_run_brain_continuous` samples through the OU sampler and keeps the step's previous pre-squash speed, input and live flag for the buffer.
- [x] 3.3 Persistence: training_state records the mode, `_reject_speed_noise_mismatch` refuses the other mode on load, and ρ is recomputed beside the modulator decay.
- [x] 3.4 `utils/step_duration.py`: classify the field, with an audit note and the pinned field count.

## 4. PPO

- [x] 4.1 `_ppo_buffer.py`: `add(..., previous=None)` stores the previous pre-squash speed, the previous input (the full state vector the forward reads: food features, modulator concentrations and drive) and the live flag. `get_minibatches` returns them under the same permutation; with slow noise off, the keys and the permutation are unchanged.
- [x] 4.2 `ppo.py`: `_evaluate` recomputes the previous mean with one batched forward and uses the exact conditional where the minibatch carries a previous step.
- [x] 4.3 Tests:
  - the first-epoch ratio is 1;
  - the gradient of the evaluated log-prob matches autograd through the exact density;
  - τ is not learnable;
  - a short PPO run with slow noise on completes, with finite losses.

## 5. The encoder

- [x] 5.1 `encoder_entries(..., speed_noise=False)`, through `encoder_parameters`, `box_centre`, `tax4_vector` and `CompactEncoder`. The entry `timescale.speed_noise` (log, 0.25–300 s) is appended last, and `write_encoder`/`read_encoder` carry it.
- [x] 5.2 Tests:
  - 197/133 parameters with the option, and 196/132 unchanged without;
  - the prefix property;
  - the centre writes √75 s;
  - a vector round-trips;
  - the learnable count is n − 3.

## 6. The batched evaluator

- [x] 6.1 `BatchedConnectome` keeps each animal's speed-noise state, resets it at `reset(n)` and carries it across phases. It samples through `ou_noise_step` with the same float32 operations as the per-step runner. With the option off, it calls `_sample` unchanged.
- [x] 6.2 Parity tests:
  - layered parity on seeds 5001–5032 for the encoded wild type on Scheer's small lawn;
  - one animal alone against the same animal in a batch of 64;
  - a temporary assay config with short phases crossing a phase start;
  - the two tampered controllers failing.

## 7. The probe and the positive control

- [x] 7.1 `c2_encoder_probe.py`: thread `encoder_speed_noise` through `entries()` and `encoded_brain()`, and add the `speed-noise` stage of D7, written to `docs/experiments/logbooks/supporting/123-c2-speed-noise/positive_control.json`. Declare the diagnosis block (88,000,000) and the control's block (88,100,000) among the probe's seed blocks, so its overlap test protects them.
- [x] 7.2 Run it; record whether the bar is met. *(It is not: the stage-2 answer's speed output is pinned, so the control is re-run on the first fit whose speed responds; Logbook 123.)*

## 8. Checks and records

- [x] 8.1 `connectome_identity_check.py check` against 1.1: every job identical. Recorded in `supporting/123-c2-speed-noise/identity.json`. If `evaluation/batched.py` changes, also run `speedup_identity_check.py` (baseline in 1.1).
- [x] 8.2 `uv run pytest -m "not nightly"` and `uv run pre-commit run -a`.
- [x] 8.3 Logbook 123: the diagnosis, the mechanism, the positive control, and what the readiness restarts inherit. Add an AGENTS.md line, a CHANGELOG line and the tracker, which notes that slow noise gives every W arm the same handle on the kinematics family.
