# bit-memory-positive-control Specification

## Purpose
TBD - created by archiving change add-bit-memory-positive-control. Update Purpose after archive.
## Requirements
### Requirement: Delayed-match-to-cue task structure

The system SHALL provide a config-gated bit-memory task composed of one or more independent trials per episode. Each trial SHALL progress through three phases in order — a **cue phase**, a **delay phase**, and a **response phase** — whose lengths are configurable. At the start of each trial the system SHALL sample a binary cue uniformly at random, independently of prior trials.

#### Scenario: Phases progress in order within a trial

- **WHEN** a bit-memory trial runs with `cue_steps`, `delay_steps`, `response_steps` configured
- **THEN** the first `cue_steps` steps are the cue phase, the next `delay_steps` steps are the delay phase, and the next `response_steps` steps are the response phase, after which the next trial begins (or the episode ends)

#### Scenario: Cue is freshly sampled per trial and independent

- **WHEN** an episode contains multiple trials
- **THEN** each trial draws its own binary cue uniformly at random, and the cue value of one trial does not determine the cue value of any other trial

### Requirement: Cue and go-signal observation channels

The system SHALL expose two task observation channels as sensory modules: a **cue** channel and a **go-signal** channel. The cue channel SHALL carry the trial's cue value during the cue phase and SHALL be zero during the delay and response phases. The go-signal channel SHALL be one during the response phase and zero otherwise.

#### Scenario: Cue is present only during the cue phase

- **WHEN** the trial is in the cue phase
- **THEN** the cue channel carries the sampled cue value
- **WHEN** the trial is in the delay or response phase
- **THEN** the cue channel is exactly zero

#### Scenario: Go-signal marks the response phase

- **WHEN** the trial is in the response phase
- **THEN** the go-signal channel is one
- **WHEN** the trial is in the cue or delay phase
- **THEN** the go-signal channel is zero

### Requirement: No external memory aids in the task observation

The bit-memory task observation SHALL contain only the cue and go-signal channels. It SHALL NOT include the short-term associative memory (STAM) buffer, gradient/klinotaxis sensing, or any other recency buffer that would let a memoryless policy recover the cue after the cue phase. This guarantees that retaining the cue across the delay requires internal recurrent state.

#### Scenario: A memoryless policy cannot recover the cue at response time

- **WHEN** the agent's observation is assembled during the response phase
- **THEN** the observation contains no channel encoding the current trial's cue value (the cue channel is zero and no STAM/recency channel is present), so the cue is recoverable only from the policy's internal state

### Requirement: Cue-conditioned response reward

The system SHALL grant reward based on the cue only during the response phase. A response action that matches the trial's cue SHALL receive the correct-response reward; a non-matching response SHALL receive the wrong-response outcome (zero or a configured penalty). No cue-based reward SHALL be granted during the cue or delay phases, and agent movement SHALL NOT contribute reward.

#### Scenario: Correct response is rewarded

- **WHEN** the agent emits a response matching the trial's cue during the response phase
- **THEN** the agent receives the configured correct-response reward

#### Scenario: Incorrect response is not rewarded

- **WHEN** the agent emits a response not matching the trial's cue during the response phase
- **THEN** the agent receives the wrong-response outcome (zero reward or the configured penalty) and not the correct-response reward

#### Scenario: No cue reward outside the response phase

- **WHEN** the trial is in the cue or delay phase
- **THEN** no cue-conditioned reward is granted regardless of the agent's action

### Requirement: Binary action readout

The agent's response SHALL be derived from its existing policy action with no new output head. For a continuous-action arm the binary response SHALL be the sign of the normalized turn component; for a discrete-action arm it SHALL be the left/right action. The mapping from cue value to the matching response SHALL be fixed for the task.

#### Scenario: Continuous turn sign selects the response

- **WHEN** a continuous-action arm emits a normalized turn during the response phase
- **THEN** a turn of one sign is read as one binary response and the opposite sign as the other, and this is compared against the cue to score correctness

### Requirement: Config-gated, off by default

The bit-memory task SHALL be controlled by a configuration flag that is disabled by default. When disabled, the environment, observation assembly, reward calculation, and configuration loading SHALL behave exactly as before this change. When enabled, foraging, predator, thermotaxis, satiety, and health dynamics SHALL be inactive so they do not interact with the task.

#### Scenario: Disabled leaves existing behaviour unchanged

- **WHEN** a configuration does not enable the bit-memory task
- **THEN** the environment runs its normal foraging/predator/thermotaxis behaviour with no cue/go channels and no phase machine

#### Scenario: Enabling activates only the task dynamics

- **WHEN** a configuration enables the bit-memory task
- **THEN** the foraging, predator, thermotaxis, satiety, and health dynamics are inactive and the episode is driven by the cue/delay/response phase machine

### Requirement: Architecture-separation evaluation

The system SHALL provide an evaluation that reports each arm's cue-match success rate over response steps across paired seeds and computes the pairwise separation between arms using the committed paired-seed statistics layer (one-sided Wilcoxon, bootstrap confidence interval, BH-FDR). The evaluation SHALL report whether the recurrent/attention arms exceed both chance and the memoryless arm, which is the separation verdict.

#### Scenario: Separation is reported when memory arms beat a memoryless arm at chance

- **WHEN** the evaluation is run over the arm panel and seeds
- **THEN** it reports per-arm cue-match success, the pairwise deltas with BH-FDR-adjusted significance, and a verdict of "separation confirmed" when the recurrent/attention arms are significantly above both chance and the memoryless MLP

#### Scenario: A null result is reported as such

- **WHEN** no arm exceeds chance, or the recurrent/attention arms do not significantly exceed the memoryless MLP
- **THEN** the evaluation reports a null verdict (the comparison does not resolve working memory) rather than a separation

