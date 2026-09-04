# Spec: brain-architecture

## ADDED Requirements

### Requirement: Continuous Action Std Modes

`BrainConfig` SHALL expose `continuous_std_mode: Literal["state_independent", "state_dependent"] = "state_independent"`. In the default mode every continuous brain keeps today's free `log_std` parameter, byte-identically. In `state_dependent` mode (meaningful only with `action_mode: continuous`) each continuous brain SHALL compute `log_std` from a per-brain **std head** — `nn.Linear(trunk_dim, CONTINUOUS_ACTION_DIM)` off the same trunk feature that feeds its mean head (mlpppo: gated features; lstmppo: `h_out`; cfcppo: `h_new` under both `actor_head` modes; transformerppo: pooled `d_model`; connectome: the pooled motor 4-vec, with the head owned by `ConnectomeTopology` and surfaced through `learnable_parameters`). The head SHALL be allocated only when the mode is on, after all existing parameters, and zero-initialised (weight and bias) so `std = 1` for every state at step 0 — matching the state-independent init exactly. In state-dependent mode the `log_std` parameter SHALL NOT be allocated; optimiser and grad-clip parameter lists and weight persistence (`log_std_head` component in place of `log_std`) SHALL swap accordingly. The shared `_policy.py` helpers SHALL be unchanged (they are shape-generic over `log_std`); the frozen `_legacy_connectome_update_reference.py` is exempt (byte-equivalence runs mode-off only).

#### Scenario: Default mode is byte-identical

- **WHEN** a continuous brain is constructed with `continuous_std_mode` unset or `state_independent`
- **THEN** every parameter SHALL be byte-identical to a pre-change construction at the same seed
- **AND** no std-head attribute SHALL exist

#### Scenario: State-dependent mode matches at initialisation

- **WHEN** two brains differing only in `continuous_std_mode` act at step 0 under pinned RNG
- **THEN** the sampled action SHALL be byte-identical (the zero-initialised head reproduces `log_std ≡ 0`)
- **AND** the `log_std` parameter SHALL NOT exist on the state-dependent brain

#### Scenario: Batched state-dependent log-std evaluates correctly

- **WHEN** `continuous_evaluate_tanh_gaussian` receives a `(B, 2)` `log_std`
- **THEN** log-probs SHALL match a per-row change-of-variables computation
- **AND** the returned entropy SHALL be the batch mean of per-state entropies

#### Scenario: D7 klinokinesis validation gate

- **WHEN** the 036 thermal pair re-runs with the mode on (n=4, 300ep, `--theta-sharp 0.45`, tail 100)
- **THEN** the substrate SHALL be declared frozen only if the klinotaxis arm's combined klinokinesis verdict is **PRESENT** (both ratio statistics' 80% CI lower bounds > 1.0), weathervane does not regress below its 036 grades, and the derivative control stays non-PRESENT
- **AND** a failed gate after one entropy-only tuning pass SHALL produce a dated D7 amendment, never a silent freeze

#### Scenario: Post-D7 re-baseline is the L4 reference frame

- **WHEN** the gate passes
- **THEN** mlpppo, cfcppo, transformerppo, and connectomeppo SHALL re-run on `_sdstd` C3 variants at n=8 under the 029 protocol (frozen recipes, uniform budget + top-up, convergence audit before ranking)
- **AND** the result SHALL supersede Logbook 029 as the L4 panel's descriptive reference frame, with the 029 comparison reported qualitatively
