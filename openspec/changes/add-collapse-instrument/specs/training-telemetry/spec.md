## ADDED Requirements

### Requirement: Per-update PPO telemetry

With `--update-telemetry`, every PPO update on the MLP-PPO and connectome PPO brains SHALL append one
JSON line to `exports/<session>/telemetry/updates.jsonl` carrying the episode and update index; policy
entropy and `log_std`; approximate KL and clip fraction over the final epoch; value loss and explained
variance; pre-clip gradient norm, the fraction of minibatches clipped, and actor and critic gradient
norms; the raw advantages' mean, std and maximum magnitude before normalisation; the pre-squash action
mean's magnitude and the fraction of steps with `|u| > 2`; layer-wise weight norms; the saturated-unit
fraction and effective rank of the features the heads read; and, on the connectome, the settling
residual. Every quantity SHALL be computed from tensors the update already holds, without drawing from
any random generator and without writing to any parameter or buffer.

#### Scenario: Telemetry does not change the run

- **WHEN** one seed is run with `--update-telemetry` and again without it
- **THEN** every episode's outcome and reward, the final weights and the next value of every random
  generator SHALL be identical between the two runs

#### Scenario: One line per update

- **WHEN** a run with `--update-telemetry` performs `n` PPO updates
- **THEN** `updates.jsonl` SHALL hold `n` lines, each with every field above, in update order

#### Scenario: Off writes nothing

- **WHEN** a run is made without `--update-telemetry`
- **THEN** no telemetry directory SHALL be created and the run SHALL be byte-identical to a run made
  before the flag existed

### Requirement: Periodic checkpoints

With `--checkpoint-every N` (a positive integer), a run SHALL save its weights every `N` episodes to
`exports/<session>/weights/ep<NNNNN>.pt` in the existing weight format, in addition to `final.pt`. Saving
SHALL NOT change the run. A non-positive `N` SHALL be refused.

#### Scenario: Checkpoints appear on schedule

- **WHEN** a 1,000-episode run is made with `--checkpoint-every 250`
- **THEN** `ep00250.pt`, `ep00500.pt`, `ep00750.pt` and `ep01000.pt` SHALL exist beside `final.pt`, and
  `ep01000.pt` SHALL load to the same weights as `final.pt`
