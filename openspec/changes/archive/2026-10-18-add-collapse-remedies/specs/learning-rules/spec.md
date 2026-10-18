## ADDED Requirements

### Requirement: PPO stabilisers and the placebo shuffle

The connectome PPO rule and the MLP-PPO brain SHALL offer, each off by default:

- `kl_stop_factor`: before each minibatch's optimiser step, if that minibatch's approximate KL exceeds
  the factor times the median per-update KL over the learner's last 100 updates, the step SHALL NOT be
  taken and the update's remaining minibatches and epochs SHALL be skipped; no stop SHALL occur before
  100 updates;
- `weight_clip`: after every optimiser step, each learnable tensor SHALL be clipped elementwise to ±κ
  times its largest absolute value at construction, tensors whose value at construction is zero being
  left unclipped;
- `adam_eps` and `adam_beta2`, passed to the optimiser and kept after an optimiser state is loaded;
- `rollout_shuffle_stream`, under which the rollout buffer SHALL draw its minibatch permutations from a
  generator keyed on the run seed and a fixed tag, changing nothing else.

With every option off, every output, weight, optimiser state, saved key and the next value of every random
generator SHALL equal the brain's before the options existed.

#### Scenario: The KL stop ends the update before the crossing step

- **WHEN** after at least 100 updates a minibatch's approximate KL exceeds the factor times the running
  median
- **THEN** that minibatch SHALL take no optimiser step, neither SHALL any later minibatch of the update,
  and the update's telemetry SHALL record the stop and the number of minibatches stepped

#### Scenario: Weight clipping bounds each tensor by its construction scale

- **WHEN** an optimiser step moves a weight beyond κ times its tensor's largest magnitude at construction
- **THEN** the weight SHALL be clipped to that bound before the next forward pass

#### Scenario: Adam's settings survive a load

- **WHEN** a brain built with `adam_eps` and `adam_beta2` loads an optimiser state saved without them
- **THEN** its optimiser SHALL carry the configured settings after the load

#### Scenario: The placebo changes only the minibatch order

- **WHEN** a run is made with `rollout_shuffle_stream` and again without it
- **THEN** the first update's minibatches SHALL be drawn in a different order, and the brain's own
  generator SHALL be unaffected by the buffer's draws

#### Scenario: Off is byte-identical

- **WHEN** every option is at its default
- **THEN** a run SHALL be identical, output for output, weight for weight and key for key, to the brain
  before the options existed
