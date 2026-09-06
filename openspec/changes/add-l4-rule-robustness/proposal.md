# Rule robustness before the panel: exploration noise, runaway control, a bounded yardstick

## Why

Pilot 2 of the L4 panel ran to completion on the re-registered grid with the centred modulator
and its rules produced a pin. The pin was withheld, ratified with Chris, because the pilot's
telemetry exposes three properties of *our rule and arms* — not of the worm — that a referee
could use to dismiss the panel whichever way it came out. Each is a default nobody examined
until a run was long enough to show it. Records:
`docs/experiments/logbooks/supporting/040-l4-panel/pilot-2-notes.md`.

1. **Every plastic arm explores at full noise forever.** The continuous policy's `log_std` is
   initialised to zero — an action standard deviation of 1.0 — and under the plastic rule that
   parameter is frozen by design, so the plastic arms and all four floors sample actions at unit
   noise for the whole run. PPO arms learn to shrink it, and that is part of how the PPO
   connectome reached 52%. A policy whose deterministic part is decent but is drowned in unit
   noise plateaus low whatever the synapses learn; it fits the pilot's picture of a wild-type arm
   that is above its floor from the first block and then flat.
2. **Weights run away onto the bound.** The decay term is `η · 0.001 · w` per step against a
   coherent Hebbian drive of order `η`; a fifth of the synapses sit clamped at ±3 by the plateau
   tail at every rate, and the Hebbian floors saturate into a constant policy within a few
   hundred episodes. The plateau is measured on a partly constant network.
3. **The matched-rule yardstick has unbounded units.** The MLP's activation is hard-coded ReLU
   while the connectome's units are tanh. Under a local Hebbian rule unbounded units explode
   (trace scale to `1e7`) or die (updates to `1e-7`); every MLP run so far has done one or the
   other. D2 test (ii) is vacuous until the yardstick can learn at all.

Fixing these before the panel is the same discipline that fixed the rate scale and the
modulator's mean: the panel must not be run, twice, on a rule with a known defect. Ratified
with Chris as one bundled change rather than three cycles, each mechanism default-off or
default-unchanged so every existing build stays byte-identical, and each value for the panel
chosen afterwards by a short telemetry probe in the panel change's re-registration.

## What Changes

- **Configurable initial action noise**: `initial_log_std` on the shared plasticity mixin,
  default `0.0` (byte-identical), applied by both brains' state-independent Gaussian head at
  construction. One number for every panel arm.
- **Homeostatic incoming-norm scaling**: `plasticity_homeostasis` on the mixin, default off.
  When on, after each plastic update every unit's incoming plastic weights are rescaled to the
  incoming norm they had at initialisation (about 1 on the connectome by construction), over
  the masked entries only. The plastic-topology seam gains `plastic_fan_in_axes` so the rule
  knows which axis holds a unit's incoming weights on each substrate. One telemetry key,
  `plasticity_norm_drift`. The decay term and the clamp stay as they are.
- **Configurable MLP activation**: `activation` on the MLP brain config, `relu` (default,
  byte-identical) or `tanh`, with the orthogonal initialisation gain following the choice.
- Tests for each mechanism on both substrates and for byte-identity with defaults; docs.

Out of scope: choosing the panel's values (the panel change's probes and dated amendment), any
change to the trace, the modulator, the scales, the environment, or the PPO path.

## Capabilities

**Modified**: `learning-rules` — gains homeostatic incoming-norm scaling; the plastic-topology
seam exposes fan-in axes. **Modified**: `brain-architecture` — configurable initial action
noise on the plastic brains and a configurable activation on the MLP brain.

## Impact

- Edited: `brain/arch/_plasticity_config.py`, `brain/arch/_topology.py`,
  `brain/arch/connectome_ppo.py`, `brain/arch/_mlp_topology.py`, `brain/arch/mlpppo.py`,
  `learning_rules/three_factor.py`, `brain/arch/_brain.py`, tests, `docs/architectures.md`,
  `CHANGELOG.md`.
- No config changes here; PPO, trace-off and default builds are byte-identical.
