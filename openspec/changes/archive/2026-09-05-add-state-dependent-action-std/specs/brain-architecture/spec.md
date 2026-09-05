# Spec: brain-architecture

## ADDED Requirements

### Requirement: Continuous Action Std Modes

`BrainConfig` SHALL expose `continuous_std_mode: Literal["state_independent", "state_dependent"] = "state_independent"`. In the default mode every continuous brain keeps today's free `log_std` parameter, byte-identically. In `state_dependent` mode (meaningful only with `action_mode: continuous`) each continuous brain SHALL compute `log_std` from a per-brain **std head** — `nn.Linear(trunk_dim, CONTINUOUS_ACTION_DIM)` off the same trunk feature that feeds its mean head (mlpppo: gated features; lstmppo: `h_out`; cfcppo: `h_new` under both `actor_head` modes; transformerppo: pooled `d_model`; connectome: the pooled motor 4-vec, with the head owned by `ConnectomeTopology` and surfaced through `learnable_parameters`). The head's weight and bias SHALL be built directly as zero `nn.Parameter`s — the allocation SHALL consume **no RNG draws** — so `std = 1` for every state at step 0, matching the state-independent init exactly, and every other parameter draw stays matched across modes. The connectome's accessor is the **public** `ConnectomeTopology.state_dependent_log_std(hidden)` (both `(302,)` and `(B, 302)`); `forward_with_hidden*` signatures SHALL be unchanged. In state-dependent mode the `log_std` parameter SHALL NOT be allocated; optimiser and grad-clip parameter lists and weight persistence (`log_std_head` component in place of `log_std`) SHALL swap accordingly, with a cross-mode weight load raising a descriptive mode-mismatch error. `continuous_std_mode: state_dependent` with `action_mode: discrete` SHALL fail at load (pydantic model validation). Mode-on runs SHALL record per-update mean/max of the clamped `log_std` batch (the per-state clamp-ceiling guard). The shared `_policy.py` helpers SHALL be unchanged (they are shape-generic over `log_std`); the frozen `_legacy_connectome_update_reference.py` is exempt (byte-equivalence runs mode-off only).

#### Scenario: Default mode is byte-identical

- **WHEN** a continuous brain is constructed with `continuous_std_mode` unset or `state_independent`
- **THEN** every parameter SHALL be byte-identical to a pre-change construction at the same seed
- **AND** no std-head attribute SHALL exist

#### Scenario: State-dependent mode matches at initialisation

- **WHEN** two brains differing only in `continuous_std_mode` are constructed at the same seed and act at step 0
- **THEN** the sampled action SHALL be byte-identical **as a run property** — the RNG-free head allocation leaves the generator state matched, so no test-side re-seed is needed
- **AND** the `log_std` parameter SHALL NOT exist on the state-dependent brain

#### Scenario: Batched state-dependent log-std evaluates correctly

- **WHEN** `continuous_evaluate_tanh_gaussian` receives a `(B, 2)` `log_std`
- **THEN** log-probs SHALL match a per-row change-of-variables computation
- **AND** the returned entropy SHALL be the batch mean of per-state entropies

#### Scenario: D7 klinokinesis validation gate

- **WHEN** the 036 thermal pair re-runs with the mode on (n=4, 300ep, `--theta-sharp 0.45`, tail 100)
- **THEN** the substrate SHALL be declared frozen only if the klinotaxis arm's combined klinokinesis verdict is **PRESENT** (both ratio statistics' 80% CI lower bounds > 1.0), weathervane does not regress below its 036 grades (both slope statistics REPRODUCED), and the derivative control's combined klinokinesis verdict is neither PRESENT nor PRESENT_PARTIAL (no REPRODUCED ratio statistic in the control)
- **AND** a failed gate after one entropy-only tuning pass SHALL produce a dated D7 amendment, never a silent freeze

#### Scenario: Post-D7 re-baseline is the L4 reference frame

- **WHEN** the gate passes
- **THEN** mlpppo, cfcppo, transformerppo, and connectomeppo SHALL re-run on `_sdstd` C3 variants at n=8 under the 029 protocol (frozen recipes, uniform budget + top-up, convergence audit before ranking)
- **AND** the result SHALL supersede Logbook 029 as the L4 panel's descriptive reference frame, with the 029 comparison reported qualitatively
- **AND** the frozen substrate SHALL be mode-on: the L4 panel configs derive from the `_sdstd` variants, with std-head ownership (PPO vs the three-factor rule) decided in the panel change
