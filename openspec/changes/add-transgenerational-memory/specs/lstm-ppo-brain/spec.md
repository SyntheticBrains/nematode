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
- **THEN** at every step, the actor logits SHALL include the additive `[+2.0, 0.0, 0.0, 0.0]` offset before softmax (4-action default)
- **AND** the empirical probability of action 0 across the 100 steps SHALL be strictly higher than a no-prior baseline run with the same seed
- **AND** the prior SHALL NOT be drowned out by LSTM hidden-state evolution (i.e. the elevated probability persists across all 100 steps, not just step 1)

#### Scenario: Setting tei_prior to None restores baseline behaviour

- **GIVEN** an `LSTMPPOBrain` with `tei_prior = torch.tensor([2.0, 0.0, 0.0, 0.0])`
- **WHEN** `brain.tei_prior` is set to `None` (e.g., by a test or by a non-TEI `fitness.evaluate` call) and `run_brain` is then invoked
- **THEN** the actor logits SHALL be byte-equivalent to the pre-TEI implementation
- **AND** action sampling SHALL match a no-prior baseline with the same seed

#### Scenario: prepare_episode does not clear tei_prior

- **GIVEN** an `LSTMPPOBrain` with `tei_prior` set to a non-None tensor
- **WHEN** `prepare_episode` is called (which zeros the LSTM hidden state)
- **THEN** `self.tei_prior` SHALL be unchanged
- **AND** the LSTM hidden state SHALL be zeroed as before
- **AND** the prior SHALL remain in effect for the next episode's `run_brain` calls

### Requirement: Fitness Evaluator Sets TEI Prior on Decoded Brain

`LearnedPerformanceFitness.evaluate` in `quantumnematode/evolution/fitness.py` SHALL be the single integration point for TEI substrate application. The runner and the worker SHALL NOT apply the substrate directly. This mirrors the existing `warm_start_path_override` / `weight_capture_path` pattern: the worker (`_evaluate_in_worker` in `evolution/loop.py`) forwards TEI configuration as keyword arguments to `fitness.evaluate`, and `fitness.evaluate` applies the substrate immediately after constructing the brain (currently `brain = encoder.decode(...)` at `fitness.py:429`) and BEFORE invoking any episode runner (`StandardEpisodeRunner` / `FrozenEvalRunner` / `MultiAgentSimulation`).

`LearnedPerformanceFitness.evaluate` SHALL accept a new keyword argument:

- `tei_prior_source: tuple[Path, float, int] | None = None` — the triple `(f0_substrate_path, decay_factor, lineage_depth)`. When `None` (the default), no TEI substrate is loaded and `brain.tei_prior` is not touched, preserving byte-equivalent behaviour for `inheritance: none|lamarckian|baldwin` runs.

When `tei_prior_source is not None`, `fitness.evaluate` SHALL:

1. Load the F0 substrate via `TransgenerationalMemory.load(f0_substrate_path)`.
2. Apply `inherit_from([f0_substrate], decay_factor)` `lineage_depth` times to produce a depth-N substrate (F0 → 0 applications, F1 → 1, F2 → 2, F3 → 3).
3. Dispatch via `hasattr(brain, "tei_prior")`:
   - When True (e.g. `LSTMPPOBrain`): set `brain.tei_prior = depth_n_substrate.logit_bias`.
   - When False: emit a `logger.warning` stating the substrate is inert for this brain type and proceed without setting the attribute.

The runner code (`StandardEpisodeRunner`, `FrozenEvalRunner`, `MultiAgentSimulation`) SHALL be unchanged — runners are TEI-agnostic. Single-agent and multi-agent paths apply the prior identically because both paths read `brain.tei_prior` from the same brain instance constructed by `fitness.evaluate`.

#### Scenario: Fitness evaluator loads and applies the depth-N substrate

- **GIVEN** a call to `LearnedPerformanceFitness.evaluate(genome, sim_config, encoder, episodes=N, seed=S, tei_prior_source=(f0_substrate_path, 0.6, 2))` (F2: 2 decay applications)
- **WHEN** `fitness.evaluate` executes
- **THEN** the F0 substrate at `f0_substrate_path` SHALL be loaded via `TransgenerationalMemory.load(...)`
- **AND** `inherit_from([f0], 0.6)` SHALL be applied twice, producing a depth-2 substrate with `logit_bias ≈ f0.logit_bias * 0.36`
- **AND** if `hasattr(brain, "tei_prior") == True`, `brain.tei_prior` SHALL be set to the depth-2 substrate's `logit_bias` immediately after `brain = encoder.decode(...)` and BEFORE `_build_agent(brain, env, sim_config)` is called
- **AND** the prior SHALL remain in effect for the full train+eval cycle in `fitness.evaluate` (`run_brain` reads `self.tei_prior` at every actor forward pass)

#### Scenario: Default tei_prior_source preserves byte-equivalent fitness behaviour

- **GIVEN** a call to `LearnedPerformanceFitness.evaluate(...)` without `tei_prior_source` (or with `tei_prior_source=None`)
- **WHEN** `fitness.evaluate` executes
- **THEN** no substrate SHALL be loaded
- **AND** `brain.tei_prior` SHALL NOT be touched (LSTMPPO's `__init__` default of `None` is preserved; non-LSTMPPO brains never get the attribute)
- **AND** the actor logit computation and PPO update SHALL be byte-equivalent to the pre-TEI baseline for that genome/seed

#### Scenario: Non-LSTMPPO brains under transgenerational config emit a defensive warning

- **GIVEN** a call to `fitness.evaluate(..., tei_prior_source=(path, 0.6, 1))` where the brain type produced by `encoder.decode(...)` is NOT `LSTMPPOBrain` (a future config experiments with applying TEI to a different brain)
- **WHEN** `fitness.evaluate` executes
- **THEN** the substrate SHALL be loaded and decayed
- **AND** `hasattr(brain, "tei_prior")` SHALL return False
- **AND** `fitness.evaluate` SHALL NOT set the attribute on the brain
- **AND** `fitness.evaluate` SHALL emit a `logger.warning` stating that a TEI substrate was assigned but the brain type does not support `tei_prior`, so the substrate is inert for this run
- **AND** `fitness.evaluate` SHALL proceed with train+eval (defensive — do not crash the worker pool over a config-level brain/substrate mismatch)

#### Scenario: Single-agent and multi-agent paths apply the prior identically

- **GIVEN** two separate `fitness.evaluate` invocations on `LSTMPPOBrain` instances, one configured to use `StandardEpisodeRunner` (single-agent) and one configured to use `MultiAgentSimulation` (multi-agent), both passing the same `tei_prior_source`
- **WHEN** each invocation executes
- **THEN** the prior SHALL be applied at every step of every episode in both paths (because `LSTMPPOBrain.run_brain` reads `self.tei_prior` at every actor forward pass, regardless of which runner orchestrates the episode)
- **AND** the runners themselves SHALL contain no TEI-specific code
- **AND** the per-step actor logits in both paths SHALL include the same additive offset (the substrate's `logit_bias`)

### Requirement: PPO Sampling/Update Distribution Consistency

The `LSTMPPOBrain.learn()` method SHALL apply the same `tei_prior` to logits in its training-time forward pass (currently at `lstmppo.py:747`, before the softmax at line 748) that `run_brain()` applied in its sampling-time forward pass (currently at `lstmppo.py:601`, before the softmax at line 602). The bias SHALL be added at both call sites whenever `self.tei_prior is not None`.

This requirement exists because PPO's policy-update step computes a probability ratio `exp(new_log_probs - chunk["old_log_probs"])` over the rollout batch. `chunk["old_log_probs"]` is the log-probability recorded at action-sampling time (under the biased distribution when TEI is active). `new_log_probs` is recomputed inside `learn()` from the current policy's forward pass. If the training forward pass omits the bias while the sampling forward pass includes it, `new_log_probs` and `old_log_probs` reflect different distributions and the PPO ratio is systematically wrong — silently corrupting F0 training under any non-zero `ppo_train_episodes`.

`tei_prior` SHALL be constant across an episode (it is set once by `fitness.evaluate` post-decode and SHALL NOT be mutated during the step loop). The `learn()` method SHALL include a defensive assertion at entry that `self.tei_prior` is either `None` or a tensor with the same shape and dtype it had at the most recent `run_brain` call whose rollout data is in the current update batch; mismatch SHALL raise an explicit `RuntimeError` rather than silently degrading the ratio.

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
