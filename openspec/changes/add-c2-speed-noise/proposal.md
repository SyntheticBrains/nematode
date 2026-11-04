## Why

C.2's staged fit (Logbook 122) misses on the kinematics family. On 200 fresh animals its answer's 10-s window speeds, against real worms, are:

| Window speed | Real | Answer | Off by |
|---|---|---|---|
| Dwelling median | 0.0128 mm/s | 0.0153 | +18% |
| Dwelling 90th percentile | 0.0282 | 0.0355 | +27% |
| Roaming median | 0.0373 | 0.0341 | −9% |
| Roaming 90th percentile | 0.0733 | 0.0520 | −29% |

The record is `docs/experiments/logbooks/supporting/123-c2-speed-noise/diagnosis/stage2_answer_variants.json`.

The planned fix was a state-dependent per-second std. The diagnosis rules it out: setting the answer's speed noise to its floor (log-SD +0.116 → −4) leaves every window statistic in place, for example a dwelling median of 0.0153 in both. Independent per-second noise averages out inside a 10-s window. What the window statistics need is speed variation that persists across seconds.

The maintainer's decisions (2026-11-04):

- slow Ornstein–Uhlenbeck noise on the speed output, with a fitted timescale;
- supported under PPO now, with the exact conditional likelihood;
- a timescale box of 0.25–300 s, which contains the independent model;
- encoder only, with the timescale held under PPO;
- the mutants, which all fail on the stage-2 answer, decided after the readiness restarts.

## What Changes

- **`encoder_speed_noise: none | ou`** on `ConnectomePPOBrainConfig`, default `none`. It requires `encoder: compact`. With `none`, every output is byte-identical, and the encoder keeps 196 parameters (132 per ligand).
- **With `ou`, the speed output's standardised noise is slow:**
  - x_t = ρ·x\_{t−1} + √(1−ρ²)·ε_t, with ρ = exp(−step/τ) and ε_t the step's own draw, so no random number is added;
  - x_0 = ε_0 at each episode's start, carried across an assay's phases;
  - the other four outputs keep independent noise.
- **The encoder gains one parameter, `timescale.speed_noise`.** It is log-scaled over 0.25–300 s and appended last, so it is the 197th (or 133rd); any stored vector stays a prefix. CMA writes it, and under PPO it is a held buffer.
- **Both runners** sample the same way: the per-step runner, and the batched evaluator's `BatchedConnectome`.
- **PPO evaluates the exact conditional.** The rollout buffer stores each step's previous pre-squash speed and previous input. The update recomputes the previous step's mean under the current parameters, so the speed dimension's pre-squash density is N(m_θ(s_t) + ρ(u\_{t−1} − m_θ(s\_{t−1})), σ²(1−ρ²)). An episode's first step uses the marginal N(m_θ(s_t), σ²).
- **Refusals,** in the validator and again at construction: slow noise without the encoder, without continuous primitives, with a state-dependent std, with leaky dynamics, or with activity traces. A weight file recorded under the other mode is refused.
- **Checks:**
  - parity between the two runners with slow noise on;
  - byte-identity of every existing connectome output with it off;
  - policy and buffer unit tests;
  - a feasibility probe, the component's positive control: the slow noise must raise the roaming speed's 90th percentile by at least 3 band scales.
- **Logbook 123.**

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `connectome-encoder`: slow speed noise, its timescale parameter, its PPO likelihood and its refusals.
- `batched-evaluation`: parity with the per-step runner under slow speed noise.

## Impact

- `quantumnematode/brain/arch/_policy.py`: an OU sampler and its conditional evaluator, beside the unchanged independent ones.
- `quantumnematode/brain/arch/connectome_ppo.py`: the field, refusals, sampling, noise state, buffer hand-off and persistence.
- `quantumnematode/learning_rules/ppo.py` and `_ppo_buffer.py`: the stored previous speed and input, and the conditional evaluation.
- `quantumnematode/connectome/encoder.py`: the appended entry and its write and read.
- `quantumnematode/evaluation/controllers.py`: `BatchedConnectome`'s per-animal noise state.
- `utils/step_duration.py`: the field's classification.
- `scripts/analysis/c2_encoder_probe.py`: the option threaded through `entries()`, and a `speed-noise` stage.
- Tests, AGENTS.md, CHANGELOG, Logbook 123.
