## ADDED Requirements

### Requirement: Rollout recorder

A rollout recorder SHALL write, per environment step, one JSON line with the episode index, the
step index, the `BrainParams` the brain read (with unset fields and the embedded action
omitted), the sampled continuous action, the action mean and the sample's probability. It SHALL
flush at the end of each episode and close at session end. The runners SHALL invoke it
immediately after the brain returns its action and SHALL not invoke anything when no recorder is
attached.

#### Scenario: The record matches what the brain saw

- **GIVEN** a headless run with a recorder attached
- **WHEN** the run ends
- **THEN** the recorded `BrainParams` on every step SHALL equal the parameters the brain's history
  recorded for that step, and the recorded action SHALL equal the brain's sampled action

### Requirement: Behavioural-cloning trainer

The trainer SHALL build a student brain from a configuration at a seed through the same factory
the entry point uses, reconstruct each recorded observation, preprocess it through the student's
own feature path, compute the student's squashed and rescaled action mean in batches, and
minimise the mean squared error against the teacher's recorded action mean in the action space
by Adam over a chosen parameter set. Under `plastic` the set SHALL be the chemical weights alone,
with the update masked to the wiring, behind whatever readout the configuration built; under
`full` it SHALL be every parameter PPO trains. A seeded fraction of episodes SHALL be held out.
The trainer SHALL report the initial, final and held-out losses and the parameter set's norm
change, SHALL refuse to save when the final loss is not below the initial, and SHALL otherwise
save through `save_weights` with a clone record beside the file (named after the weight file,
so several clones can share a directory) recording the arguments, the rollout file's hash, the
losses and the seed. The trainer SHALL never run the environment.

#### Scenario: Self-cloning recovers the teacher

- **GIVEN** rollouts recorded from a frozen connectome on a small foraging configuration
- **WHEN** a fresh student of the same wiring at another seed is cloned under either parameter set
- **THEN** the held-out loss SHALL fall by at least an order of magnitude and the saved file SHALL
  load into a brain from the same configuration and reproduce the student's mean on a recorded
  observation

#### Scenario: The plastic set touches only the chemical weights

- **WHEN** a student is cloned under `plastic`
- **THEN** its sensory gains, readout and noise parameter SHALL be bit-identical before and after,
  and every changed chemical weight SHALL lie on the wiring

#### Scenario: The full set trains what PPO trains

- **WHEN** a student is cloned under `full`
- **THEN** its chemical weights, sensory gains, readout and noise parameter SHALL all change

#### Scenario: No improvement, no file

- **WHEN** the final loss is not below the initial
- **THEN** the trainer SHALL write nothing and exit non-zero
