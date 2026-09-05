# Add the matched-rule MLP arm

## Why

The Phase 7 panel's ranking test asks whether the plastic wild-type connectome reaches or exceeds the **matched-rule MLP**'s band. That arm is the yardstick that makes "recovery" well-defined: comparing a plastic connectome against PPO-trained arms would be a cross-regime comparison the project's own commensurability rule forbids, so the only quantitative standard is a different substrate trained by the *same* rule.

The rule was written to be substrate-generic — the roadmap withdrew the earlier claim that an MLP has no biologically-plausible-plasticity analogue precisely because a rate-based three-factor rule has no dependence on the substrate it updates. But the implementation does not yet honour that: `ConnectomeThreeFactorRule` reads five connectome-specific attributes by name (`w_chem`, `m_chem`, `activity_traces`, `enable_activity_traces`, `apply_weight_mask`). "Substrate-generic" is currently a property of the mathematics, not of the code. This change makes it a property of the code, and uses that to build the arm.

## What Changes

### 1. The rule becomes generic over a plastic-topology seam

A small Protocol surface — the plastic weight tensors, their aligned eligibility traces, their edge masks, and the mask projector — is what the rule genuinely touches. `ConnectomeTopology` already provides all of it under connectome-specific names; the change exposes those through the seam and rewrites the rule to read the seam. The rule class is renamed to reflect what it now is, with the connectome-named class kept as an alias so nothing that imports it breaks.

Multiple plastic tensors are supported from the outset: the connectome has one (`w_chem`), the MLP has one per layer. Telemetry aggregates across them — mean absolute change over all plastic entries, saturated fraction over all masked entries — so the two arms report the same quantities with the same meaning.

### 2. An MLP topology that wraps the existing actor

`MLPPPOBrain` gains a topology object holding references to the actor's `Linear` layers and one eligibility trace per layer. **It wraps the existing actor rather than rebuilding it.** That is the load-bearing implementation constraint: the actor's construction order fixes the torch-RNG stream every PPO result depends on, and its `state_dict` keys are what the MLP's weight persistence stores as the `policy` component. Wrapping leaves both untouched, so the PPO path stays byte-identical and existing weight files still load.

The traced forward iterates the same layers in the same order as `self.actor(features)` and is pinned bitwise-equal to it, so a plastic-arm forward is the PPO forward with observation.

### 3. Eligibility for feedforward layers

Per layer, `E_l ← λ·E_l + post_l ⊗ pre_l`, oriented `(out, in)` to match `nn.Linear.weight`, with `pre_l` the layer's input and `post_l` its output after the nonlinearity where one exists.

The pre-synaptic term is the **same step's** input, not the previous step's. The connectome needed a previous-step term because its same-step product `h ⊗ h` is symmetric — the two directions of a reciprocal edge were indistinguishable. A feedforward layer has no such degeneracy: `pre_l` and `post_l` are different populations, and the layer itself orders them causally within the step. The previous-step form would credit each layer's synapses for an activation they had no path to.

### 4. What is plastic on the MLP

**Every `Linear` weight matrix**, under the same rule with the same hyperparameters. Biases stay frozen: the connectome has no bias terms, and a Hebbian rule has no pre-synaptic partner for one. `log_std` stays frozen and the critic stays dormant, exactly as on the connectome.

This is an asymmetry in the MLP's favour, and it is deliberate. Freezing the MLP's output layer would look symmetric with the connectome's frozen readout, but the two frozen objects are not alike: the connectome's readout is the anatomical contrast, the MLP's is an arbitrary orthogonal draw. Freezing the latter would revive on the yardstick the very incapacity the anatomical readout removed from the connectome. Letting it learn instead makes the MLP a **conservative** yardstick — if the plastic connectome reaches the band of an arm with strictly more plastic freedom, the ranking result is stronger, not weaker.

### 5. Shared plasticity configuration

The plasticity fields — `learning_rule`, `plasticity_rate`, `plasticity_weight_decay`, `plasticity_weight_bound`, `plasticity_baseline_rate`, `enable_activity_traces`, `trace_decay` — move to a mixin both brain configs inherit, with their validators. "Matched" has to mean *the same defaults from one definition*; two copies of the same numbers would be matched only until someone edited one of them.

### 6. Rule selection, dispatch, and the value-head skip on the MLP brain

`MLPPPOBrain` gains the same `learning_rule` selection, per-step dispatch, and dormant PPO machinery as the connectome brain. Its action path computes a state value on every step through the critic; under a plastic rule that call is skipped, not stubbed, for the reason recorded when the connectome got the same treatment.

### 7. Configuration and coverage

One plastic MLP config, derived from the C3 MLP cell by the minimal key delta, plus its minimal-delta test and a smoke entry — the plastic MLP path should run end to end through the entry point before the panel depends on it.

## Impact

- **Affected specs**: `learning-rules` — the rule's requirement is modified to read the seam, and the seam itself is added; the MLP brain's plastic mode is specified under the capability that holds its requirements.
- **Affected code**: `learning_rules/three_factor.py` (generic rule); a new plastic-topology Protocol; `ConnectomeTopology` exposing the seam; a new MLP topology; `mlpppo.py` (selection, dispatch, value skip, traces); a shared plasticity-config mixin; one config.
- **Default behaviour unchanged**: `learning_rule` defaults to PPO on both brains; the MLP topology wraps rather than rebuilds; the connectome rule is an alias. Every existing config, weight file, and recorded result is untouched.
- **Not in scope**: the rewired-null plastic arm (A.6), the panel (A.7), any run, any claim. The CfC port D10 marks MAY is not attempted.
