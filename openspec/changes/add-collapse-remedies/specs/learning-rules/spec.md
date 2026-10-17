## ADDED Requirements

### Requirement: PPO stabilisers

The connectome PPO rule and the MLP-PPO brain SHALL offer, each off by default: `target_kl`, which
stops an update's remaining minibatches and epochs once a minibatch's approximate KL exceeds 1.5 times
the target; `weight_clip`, which after every optimiser step clips each learnable weight tensor
elementwise to ±κ times its largest absolute value at initialisation, leaving tensors whose initial
largest value is zero unclipped; `adam_eps` and `adam_beta2`, passed to the optimiser; and
`advantage_norm: running`, which normalises advantages by an exponentially weighted running mean and
standard deviation updated once per update. With every stabiliser off, every output, weight, optimiser
state and the next value of every random generator SHALL equal the brain's before the stabilisers
existed.

#### Scenario: KL early stopping ends the update

- **WHEN** a minibatch's approximate KL exceeds 1.5 times `target_kl`
- **THEN** no further minibatch of that update SHALL take an optimiser step, and the stop SHALL be
  counted in the update's telemetry

#### Scenario: Weight clipping bounds each tensor by its initial scale

- **WHEN** an optimiser step moves a weight beyond κ times its tensor's largest initial magnitude
- **THEN** the weight SHALL be clipped to that bound before the next forward pass

#### Scenario: Off is byte-identical

- **WHEN** every stabiliser is at its default
- **THEN** a run SHALL be identical, output for output and weight for weight, to the brain before the
  stabilisers existed
