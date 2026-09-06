# Design: rule robustness before the panel

## Context

Three defaults were never examined until pilot 2 ran long enough to show them: an action noise
frozen at standard deviation 1.0 on every plastic arm, a decay far too weak to hold a coherent
Hebbian drive, and a yardstick whose units are unbounded under a rule that only bounded units
survive. Each is fixed by a mechanism that is off, or unchanged, by default, so nothing measured
so far moves; the panel change then chooses the values by probe and records them.

## Goals / Non-Goals

**Goals**

- One shared initial action noise for every plastic arm and floor.
- A runaway control that keeps the plateau off the clamp, biologically motivated, on both
  substrates through the seam.
- A yardstick whose units are bounded like the connectome's.
- Byte-identity for every existing build.

**Non-Goals**

- Picking the panel's values here (probe and dated amendment in the panel change).
- Any change to the trace, modulator, scales, environment, reward, or PPO training.
- A learnable `log_std` under the plastic rule: that would train a non-plastic parameter and
  break the "only the seam's plastic weights change" invariant the arms are compared under.

## Decisions

### D1. `initial_log_std` on the mixin, one number for every arm

The state-independent Gaussian head of both brains initialises `log_std` from the config
instead of from zero (`torch.full` instead of `torch.zeros`), default `0.0`. Under PPO the
parameter still trains from that value; under the plastic rule it stays there. Placed on the
shared mixin so the two brains cannot drift apart, and so every panel arm — plastic, floor,
rewired, MLP — carries the same value and every contrast stays paired.

Ratified with Chris over letting the plastic rule adapt `log_std` (a non-plastic parameter
would then be learning) and over an exploration schedule (a second free-form recipe axis). A
frozen but sensible noise is the smallest change that stops the noise from capping the plateau
while keeping the periphery frozen.

### D2. Homeostatic incoming-norm scaling

After each plastic update the rule rescales, for every unit, the vector of its incoming plastic
weights so that its norm over the unit's edge set returns to the norm it had at initialisation:
`w_j ← w_j · t_j / max(‖w_j‖, floor)`, with `t_j` captured from the topology at rule
construction. Units with no incoming edges (`t_j = 0`) are skipped. The clamp is applied after
the rescale, so the bound still holds; the decay term stays available but is no longer what
holds the substrate.

Why this and not a stronger decay: the runaway is Hebbian positive feedback along
reward-correlated directions, and the standard remedy is multiplicative normalisation of each
neuron's synaptic budget — Oja's rule in the abstract, synaptic scaling in the animal. It
targets the mechanism, holds the norm exactly rather than fighting it, and needs no new constant
beyond the floor the scales already have. A stronger decay would need its own grid.

The target is the *initial* norm, not a fixed number, so the connectome's per-neuron scale
(`1/√k` over `k` inputs, giving an incoming norm near 1) and the MLP's orthogonal rows (norm 1)
are each their own reference. On the connectome the target is therefore about 1 for every
neuron by construction, which is a pleasant coincidence rather than a design input.

**Seam**: `PlasticTopology` gains `plastic_fan_in_axes: list[int]`, aligned with the other
lists — the axis to reduce over to obtain each unit's incoming norm. The connectome's chemical
matrix is `[pre, post]`, so a neuron's incoming synapses are a column and the axis is `0`; a
`Linear` weight is `[out, in]`, so a unit's incoming weights are a row and the axis is `1`.
The rule never names either.

**Telemetry**: `plasticity_norm_drift`, the mean over units with a target of `|‖w_j‖ / t_j − 1|`
measured before the rescale — how hard the homeostasis worked that step. NaN when off. Under a
freeze nothing is rescaled and the drift is still reported from the unchanged weights.

### D3. `activation` on the MLP brain config

`relu` (default, byte-identical: the builder emits the same modules and the same orthogonal
gain `√2`) or `tanh` (Tanh modules, gain `5/3` per the standard recommendation). It applies to
the actor and critic builders alike, so the PPO parent stays exactly as it is and the plastic
MLP arm can be built with bounded units. Ratified with Chris: an unbounded yardstick under a
local Hebbian rule cannot learn, and a tanh MLP is the honest matched-rule comparator to a tanh
connectome — same rule, same scaling, same kind of unit.

### D4. What the panel change does with these

Nothing here changes a config. The panel change runs three short probes at the selected rate
and seed 101 over 600 episodes — an `initial_log_std` grid read on success and action std,
homeostasis on against off read on the saturated fraction, and the tanh MLP read on trace scale
and success — records them under the supporting directory, and pins the values into all seven
arm configs by dated amendment before pilot 3.

### D5. Byte-identity

Every mechanism is behind a default that reproduces today's arithmetic: `initial_log_std = 0`
gives `torch.zeros`; homeostasis off adds no operation to the update; `activation = relu`
builds the same modules with the same gain. The existing frozen-reference tests for the
three-factor rule and for the MLP PPO path keep proving it; new tests pin each mechanism on
both substrates.

## Risks / Trade-offs

- **Homeostasis constrains what the rule can express.** A neuron cannot grow its total input,
  only redistribute it. That is the biological constraint too, and it is the same for both
  wirings and both substrates, so the contrasts stay fair.
- **A frozen noise chosen by probe on one seed is a recipe choice.** It is shared by every arm
  and pinned before the pilot, like the rate; the probe's selection rule is stated in the panel
  change before it runs.
- **Tanh changes the MLP's PPO behaviour if ever enabled there.** It is not enabled there; the
  PPO parent keeps `relu`, and the plastic arm's delta from its parent grows by one key, which
  its variant test records.

## Open Questions

None.
