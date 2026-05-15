# LSTM PPO Brain — TEI Prior Delta

## ADDED Requirements

### Requirement: TEI Logit-Bias Prior Attribute

`LSTMPPOBrain` SHALL expose a `tei_prior: torch.Tensor | None` attribute, defaulted to `None` in `__init__`. When `tei_prior is not None`, `run_brain()` SHALL add the prior tensor to the actor's logits before the softmax / categorical sampling step, at every step of every episode for the duration the attribute is set.

The attribute SHALL be a read-write public attribute (not a property), so the runner can set or clear it without going through a setter method. Setting `tei_prior = None` SHALL be the disable path; the brain SHALL then behave byte-equivalently to its pre-TEI behaviour.

The `Brain` Protocol in `quantumnematode/brain/arch/_brain.py` SHALL NOT be modified — `tei_prior` is an LSTMPPO-specific attribute that other brain subtypes do not need to implement. Non-LSTMPPO brains SHALL ignore any attempt by the runner to set `tei_prior` on them.

#### Scenario: Default tei_prior is None and brain behaviour is unchanged

- **GIVEN** a freshly constructed `LSTMPPOBrain` with no `tei_prior` set
- **WHEN** `run_brain` is invoked
- **THEN** `self.tei_prior` SHALL be `None`
- **AND** the actor logit computation SHALL be byte-equivalent to the pre-TEI implementation (no additive offset applied)
- **AND** action sampling SHALL produce the same distribution as before TEI was introduced (modulo unrelated RNG state)

#### Scenario: TEI prior is applied at every step of an episode

- **GIVEN** an `LSTMPPOBrain` with `tei_prior = torch.tensor([2.0, 0.0, 0.0])` (strong bias toward action 0)
- **WHEN** the brain runs 100 sequential rollout steps via `run_brain` (with `prepare_episode` called once at the start)
- **THEN** at every step, the actor logits SHALL include the additive `[+2.0, 0.0, 0.0]` offset before softmax
- **AND** the empirical probability of action 0 across the 100 steps SHALL be strictly higher than a no-prior baseline run with the same seed
- **AND** the prior SHALL NOT be drowned out by LSTM hidden-state evolution (i.e. the elevated probability persists across all 100 steps, not just step 1)

#### Scenario: Setting tei_prior to None restores baseline behaviour

- **GIVEN** an `LSTMPPOBrain` with `tei_prior = torch.tensor([2.0, 0.0, 0.0])`
- **WHEN** the runner sets `brain.tei_prior = None` and then invokes `run_brain`
- **THEN** the actor logits SHALL be byte-equivalent to the pre-TEI implementation
- **AND** action sampling SHALL match a no-prior baseline with the same seed

#### Scenario: prepare_episode does not clear tei_prior

- **GIVEN** an `LSTMPPOBrain` with `tei_prior` set to a non-None tensor
- **WHEN** `prepare_episode` is called (which zeros the LSTM hidden state)
- **THEN** `self.tei_prior` SHALL be unchanged
- **AND** the LSTM hidden state SHALL be zeroed as before
- **AND** the prior SHALL remain in effect for the next episode's `run_brain` calls

### Requirement: Runner Sets TEI Prior Before prepare_episode

The episode runner (`StandardEpisodeRunner.run` in `quantumnematode/agent/runners.py`) SHALL set `agent.brain.tei_prior` to either the inherited substrate's `logit_bias` tensor or `None` immediately BEFORE calling `agent.brain.prepare_episode()`. The runner SHALL NOT set the attribute on non-LSTMPPO brain types (the attribute is LSTMPPO-specific).

When no TEI substrate is configured for the current generation (e.g. `inheritance: none|lamarckian|baldwin`), the runner SHALL set `tei_prior = None` (explicit clear), ensuring no stale prior survives across configuration changes.

#### Scenario: Runner passes the substrate's logit_bias to the brain

- **GIVEN** an evolution generation under `inheritance: transgenerational` where the child genome has been assigned a substrate with `logit_bias = torch.tensor([0.5, -0.5, 0.3])`
- **WHEN** the runner begins each evaluation episode
- **THEN** the runner SHALL set `agent.brain.tei_prior = torch.tensor([0.5, -0.5, 0.3])` (or a clone thereof) BEFORE calling `agent.brain.prepare_episode()`
- **AND** the prior SHALL remain in effect for the entire episode

#### Scenario: Runner clears tei_prior when no substrate is configured

- **GIVEN** an evolution run under `inheritance: none|lamarckian|baldwin`
- **WHEN** the runner begins each evaluation episode
- **THEN** the runner SHALL set `agent.brain.tei_prior = None` BEFORE calling `agent.brain.prepare_episode()` (defensive clear)
- **AND** the actor logit computation SHALL be byte-equivalent to the pre-TEI implementation
