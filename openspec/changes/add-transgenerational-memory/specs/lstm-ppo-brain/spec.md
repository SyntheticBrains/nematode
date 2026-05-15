# LSTM PPO Brain — TEI Prior Delta

## ADDED Requirements

### Requirement: TEI Logit-Bias Prior Attribute

`LSTMPPOBrain` SHALL expose a `tei_prior: torch.Tensor | None` attribute, defaulted to `None` in `__init__`. When `tei_prior is not None`, `run_brain()` SHALL add the prior tensor to the actor's logits before the softmax / categorical sampling step, at every step of every episode for the duration the attribute is set. `learn()` SHALL ALSO add the same prior tensor to the actor's logits at its training-time forward pass (see "PPO Sampling/Update Distribution Consistency" requirement below).

The attribute SHALL be a read-write public attribute (not a property), so the runner can set or clear it without going through a setter method. Setting `tei_prior = None` SHALL be the disable path; the brain SHALL then behave byte-equivalently to its pre-TEI behaviour.

The `Brain` Protocol in `quantumnematode/brain/arch/_brain.py` SHALL NOT be modified — `tei_prior` is an LSTMPPO-specific attribute that other brain subtypes do not need to implement. Non-LSTMPPO brains SHALL ignore any attempt by the runner to set `tei_prior` on them.

#### Scenario: Default tei_prior is None and brain behaviour is unchanged

- **GIVEN** a freshly constructed `LSTMPPOBrain` with no `tei_prior` set
- **WHEN** `run_brain` is invoked
- **THEN** `self.tei_prior` SHALL be `None`
- **AND** the actor logit computation SHALL be byte-equivalent to the pre-TEI implementation (no additive offset applied)
- **AND** action sampling SHALL produce the same distribution as before TEI was introduced (modulo unrelated RNG state)

#### Scenario: TEI prior is applied at every step of an episode

- **GIVEN** an `LSTMPPOBrain` with `tei_prior = torch.tensor([2.0, 0.0, 0.0, 0.0])` (strong bias toward action 0)
- **WHEN** the brain runs 100 sequential rollout steps via `run_brain` (with `prepare_episode` called once at the start)
- **THEN** at every step, the actor logits SHALL include the additive `[+2.0, 0.0, 0.0]` offset before softmax
- **AND** the empirical probability of action 0 across the 100 steps SHALL be strictly higher than a no-prior baseline run with the same seed
- **AND** the prior SHALL NOT be drowned out by LSTM hidden-state evolution (i.e. the elevated probability persists across all 100 steps, not just step 1)

#### Scenario: Setting tei_prior to None restores baseline behaviour

- **GIVEN** an `LSTMPPOBrain` with `tei_prior = torch.tensor([2.0, 0.0, 0.0, 0.0])`
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

- **GIVEN** an evolution generation under `inheritance: transgenerational` where the child genome has been assigned a substrate with `logit_bias = torch.tensor([0.5, -0.5, 0.3, 0.0])` (4-action default)
- **WHEN** the runner begins each evaluation episode
- **THEN** the runner SHALL set `agent.brain.tei_prior = torch.tensor([0.5, -0.5, 0.3, 0.0])` (or a clone thereof) BEFORE calling `agent.brain.prepare_episode()`
- **AND** the prior SHALL remain in effect for the entire episode

#### Scenario: Runner clears tei_prior when no substrate is configured

- **GIVEN** an evolution run under `inheritance: none|lamarckian|baldwin`
- **WHEN** the runner begins each evaluation episode
- **THEN** the runner SHALL set `agent.brain.tei_prior = None` BEFORE calling `agent.brain.prepare_episode()` (defensive clear)
- **AND** the actor logit computation SHALL be byte-equivalent to the pre-TEI implementation

#### Scenario: Multi-agent runner applies the same set/clear symmetry

- **GIVEN** a multi-agent evolution episode under `inheritance: transgenerational` where multiple agents share the same generation's substrate
- **WHEN** `MultiAgentSimulation.run_episode` reaches each agent's per-episode `prepare_episode()` call site (`agent/multi_agent.py:588`)
- **THEN** the runner SHALL set `agent.brain.tei_prior` for that agent (to its assigned substrate's `logit_bias`, or `None` when no substrate is configured) BEFORE calling `agent.brain.prepare_episode()`
- **AND** the single-agent and multi-agent paths SHALL be observationally symmetric for the same substrate configuration: a single-agent episode and a multi-agent episode running an LSTMPPO with the same `tei_prior` SHALL apply the prior at every step of every episode in both paths
- **AND** the multi-agent path SHALL skip the set on non-LSTMPPO brain types (the attribute is LSTMPPO-specific, mirroring the single-agent rule)

### Requirement: PPO Sampling/Update Distribution Consistency

The `LSTMPPOBrain.learn()` method SHALL apply the same `tei_prior` to logits in its training-time forward pass (currently at `lstmppo.py:747`, before the softmax at line 748) that `run_brain()` applied in its sampling-time forward pass (currently at `lstmppo.py:601`, before the softmax at line 602). The bias SHALL be added at both call sites whenever `self.tei_prior is not None`.

This requirement exists because PPO's policy-update step computes a probability ratio `exp(new_log_probs - chunk["old_log_probs"])` over the rollout batch. `chunk["old_log_probs"]` is the log-probability recorded at action-sampling time (under the biased distribution when TEI is active). `new_log_probs` is recomputed inside `learn()` from the current policy's forward pass. If the training forward pass omits the bias while the sampling forward pass includes it, `new_log_probs` and `old_log_probs` reflect different distributions and the PPO ratio is systematically wrong — silently corrupting F0 training under any non-zero `ppo_train_episodes`.

`tei_prior` SHALL be constant across an episode (the runner sets it once pre-`prepare_episode` and does not mutate it during the step loop). The `learn()` method SHALL include a defensive assertion at entry that `self.tei_prior` is either `None` or a tensor with the same shape and dtype it had at the most recent `run_brain` call whose rollout data is in the current update batch; mismatch SHALL raise an explicit `RuntimeError` rather than silently degrading the ratio.

#### Scenario: TEI prior is applied at the PPO training-time forward pass

- **GIVEN** an `LSTMPPOBrain` with `tei_prior = torch.tensor([+1.0, 0.0, 0.0, 0.0])` set across both rollout and update windows
- **WHEN** `run_brain()` is called to sample N steps (recording `_pending_log_prob` from the biased distribution) and then `learn()` is called on those N steps
- **THEN** the training-time forward pass at `learn()` SHALL include the same `+[1.0, 0.0, 0.0, 0.0]` additive offset to logits before its softmax
- **AND** for the first PPO update on freshly-rolled-out data (when `new_log_probs ≈ old_log_probs`), the PPO ratio SHALL be approximately 1.0 (within numerical noise tolerance) — confirming sampling and update distributions match

#### Scenario: TEI prior absent yields byte-equivalent baseline training

- **GIVEN** an `LSTMPPOBrain` with `tei_prior = None` across rollout and update
- **WHEN** `run_brain()` and `learn()` are called with seed-controlled RNG
- **THEN** the PPO update SHALL be byte-equivalent to the pre-TEI baseline (no observable change to gradients, losses, or actor/critic parameter updates)

#### Scenario: learn() rejects mid-update tei_prior mutation

- **GIVEN** an `LSTMPPOBrain` whose `tei_prior` was a non-None tensor during the rollout window
- **WHEN** `tei_prior` is mutated (e.g., set to `None` or to a different-shaped tensor) BEFORE `learn()` is called on that rollout's data
- **THEN** `learn()` SHALL raise a `RuntimeError` at entry stating that the `tei_prior` shape/dtype must remain stable across the rollout-to-update window
- **AND** the message SHALL include both the rollout-time tensor metadata and the current `tei_prior` metadata so the operator can diagnose
