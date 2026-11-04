## ADDED Requirements

### Requirement: Slow speed noise behind a byte-identical default

`ConnectomePPOBrainConfig` SHALL accept `encoder_speed_noise`, `none` by default or `ou`. With `none`
every output SHALL be byte-identical to the brain before the option, and the encoder SHALL keep its 196
parameters (132 under per-ligand tying). With `ou` the speed output's standardised noise SHALL follow
x_t = ρ·x\_{t−1} + √(1−ρ²)·ε_t, ρ = exp(−Δ/τ), ε_t the step's own draw and x_0 = ε_0 at each episode's
start; the other outputs SHALL keep independent noise; and the encoder SHALL append one parameter,
`timescale.speed_noise`, log-scaled over 0.25–300 s, after every existing one.

#### Scenario: Off is byte-identical

- **WHEN** a connectome brain is built with `encoder_speed_noise: none`
- **THEN** its actions, log-probs and encoder vector SHALL equal those of the brain before the option

#### Scenario: On appends one parameter

- **WHEN** an encoded brain is built with `encoder_speed_noise: ou`
- **THEN** it SHALL have 197 parameters (133 under per-ligand tying), the last `timescale.speed_noise`
- **AND** a 196-entry vector SHALL be the first 196 entries of a 197-entry one, writing the same tensors

#### Scenario: The process is a stationary OU on speed alone

- **WHEN** the slow-noise sampler runs for 20,000 steps at ρ = 0.9
- **THEN** the speed noise's variance SHALL be 1 and its lag-1 autocorrelation 0.9, each within its
  sampling error
- **AND** the other four outputs, and every output on an episode's first step, SHALL equal the
  independent sampler's on the same draws

#### Scenario: PPO evaluates the exact conditional

- **WHEN** a rollout under slow noise is re-evaluated before any update
- **THEN** every log-prob SHALL equal the one stored at sampling, so every ratio is 1
- **AND** after the parameters change, the speed dimension's log-density SHALL be that of
  N(m(s_t) + ρ·(u\_{t−1} − m(s\_{t−1})), σ²(1−ρ²)), with both means recomputed under the new parameters

#### Scenario: τ is held under PPO

- **WHEN** the encoded brain with slow noise is trained by PPO
- **THEN** `timescale.speed_noise` SHALL not be among its learnable parameters

#### Scenario: Unsupported settings are refused

- **WHEN** `encoder_speed_noise: ou` is set without `encoder: compact`, without continuous primitives,
  with leaky dynamics or with activity traces, in a validated config or one derived by `model_copy`
- **THEN** the brain SHALL refuse it, naming the setting
- **AND** a weight file recorded under the other speed-noise mode SHALL be refused on load

### Requirement: The slow noise is shown to reach the window statistics before it is relied on

The positive control SHALL grade an answer with independent noise and with slow noise at τ ∈ {0.25, 2,
8.7, 30, 100, 300} s on the same 200 animals, and SHALL be met before the readiness restarts rely on
slow noise. It is read only on an answer whose speed output responds: one whose speed output's
pre-squash median over a reference episode lies within ±2, where the squash is not flat. On the
stage-2 answer, whose speed output's median is −3.9, it was run as first registered and failed; that
reading SHALL be reported as failed, and the control SHALL be re-run on the first fit whose speed
output responds.

#### Scenario: The positive control on a responsive answer

- **WHEN** the sweep is graded on an answer whose speed output responds
- **THEN** the roaming speed's 90th percentile SHALL rise by at least 3 band scales above its value at
  τ = 0.25 at some τ, with the dwelling speed's 90th percentile at that τ reported beside it
- **AND** at τ = 0.25 every kinematics statistic SHALL equal the independent model's, on the same
  seeds, within twice its band scale

#### Scenario: An answer whose speed is pinned cannot carry it

- **WHEN** an answer's speed output's pre-squash median lies outside ±2
- **THEN** the control's reading on it SHALL be reported as failed, not passed and not re-read, with
  the speed output's median beside it
