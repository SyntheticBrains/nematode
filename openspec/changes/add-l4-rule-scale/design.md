# Design: substrate-invariant scaling of the three-factor rule

## Context

The rule today is `Δw = η · δ · E − η · λ_w · w`, `δ = r − b`, applied once per environment
step under `torch.no_grad()` over the plastic-topology seam, then clamped to `±bound`. Its
state is one scalar, the baseline `b`. The panel's pilot showed `η` is not one number across
substrates or across steps: the terminal prediction error is ~200× an ordinary one, and the
MLP's trace is ~1000× smaller per weight than the connectome's. This change gives the rule two
running scales so that `η` means the same thing everywhere, and keeps every existing behaviour
available bit for bit behind default-off switches.

## Goals / Non-Goals

**Goals**

- A bounded, scale-free modulator and a per-tensor scale-free trace, each opt-in.
- Byte-identity with both switches off, proven against the frozen reference.
- The same definition on every plastic brain (the mixin), and the same telemetry meaning.

**Non-Goals**

- Re-registering the panel's grid (the panel change does that, dated, after this lands).
- Touching the trace substrate, the MLP's activation, or any arm config.
- Persisting the scales in a state dict (they are rule state like the baseline; a resumed run
  re-warms them).

## Decisions

### D1. The modulator: `δ̃ = tanh(δ / σ)`

`σ` is the running root-mean-square of the raw prediction error: `σ² ← (1 − r_s) σ² + r_s δ²`
with `r_s = plasticity_scale_rate`. The current step is scored against the scale estimated
**before** it is absorbed, the same convention the baseline uses, so a surprising step is
scored as surprising rather than against a scale it has already inflated. `σ` is floored at
`plasticity_scale_floor` before division.

Ratified with Chris over a clip and over scale-only division: bounded in `[−1, 1]`, monotone,
sign-preserving, one fewer constant than a clip, and linear near zero. The death penalty still
carries the strongest signal of an episode (`δ̃ ≈ −1`), but it can no longer be a hundred times
an ordinary step. The raw `δ` is still reported so the compression is visible.

### D2. The trace: `E / ρ` per plastic tensor

For each plastic tensor `ρ²` is the running mean square of the trace over that tensor's edge
set (the masked entries, so off-edge zeros do not dilute a sparse substrate): `ρ² ← (1 − r_s) ρ² + r_s · mean(E[mask]²)`, again with the pre-update value used for the current step, floored
before division. The Hebbian term becomes `η · δ̃ · E / ρ`; the decay term and the clamp are
unchanged.

Ratified with Chris over instantaneous normalisation (which would make every step the same
size regardless of co-activity, discarding what the trace carries) and over a weight-relative
LARS-style scale (steps that grow with the weights fight the bound and the decay). With a
running scale, `η` is the mean absolute Hebbian step per unit modulator on every substrate,
and a step with more co-activity than usual still moves more.

Substrates differ in initialisation scale (connectome ~0.3, MLP ~0.6), so identical absolute
steps are not identical relative steps. That factor of two is stated, not corrected: the
panel's yardstick claim is "same rule, same hyperparameters", and a rule that scaled itself by
each substrate's weights would be a different rule on each.

### D3. Warm start, freeze, unmodulated mode

- **Warm start.** Both scales start unset; the first observation sets them to its own value
  (`σ = |δ|`, `ρ = RMS(E)` per tensor, each floored). Without it the first steps would divide
  by the floor and every early modulator would sit at ±1 while the trace step would be
  enormous. A zero first trace leaves `ρ` unset until a non-zero one arrives.
- **Freeze.** The scales update under a freeze exactly as the baseline does: the frozen arm
  must report the same telemetry the plastic arm would, or the two stop being comparable
  step for step. Nothing is written to a weight.
- **Unmodulated.** The modulator is `1.0` regardless of the switch; `σ` is still tracked and
  reported, mirroring how the baseline and prediction error are already reported in that mode.
  Trace normalisation applies, so the Hebbian floor's steps are matched to the plastic arm's.

### D4. Configuration and telemetry

Four mixin fields, so every plastic brain shares one definition:

| field | default | meaning |
|---|---|---|
| `plasticity_normalise_modulator` | `false` | apply D1 |
| `plasticity_normalise_trace` | `false` | apply D2 |
| `plasticity_scale_rate` | `0.01` | EMA rate of both scales; bounded to `(0, 1]` at load |
| `plasticity_scale_floor` | `1e-6` | floor under both scales before division; `> 0` at load |

Three new history fields recorded beside the existing four, through the same shared recorder:
`plasticity_modulator` (the effective third factor after D1, or `1.0` unmodulated),
`plasticity_modulator_scale` (`σ`), `plasticity_trace_scale` (mean of `ρ` over plastic
tensors). The CSV export is generic over history fields, so no export plumbing changes.

### D5. Byte-identity and the frozen reference

With both switches off the code path must not change a single operation: the new branches
sit inside `if` guards, and the existing bit-identity test against
`_legacy_three_factor_reference.py` keeps proving it across modulated/unmodulated and
frozen/plastic. A new test asserts that a config with the defaults produces a rule whose
scaling switches are off.

## Risks / Trade-offs

- **Dead ReLU units on the MLP are less likely, not impossible.** A bounded modulator and a
  normalised trace cap the first kick, but a large enough `η` can still kill units. The
  re-pilot's grid is chosen in normalised units, and saturation and the trace scale are
  reported so a dead network shows as a collapsing `ρ`.
- **Two more hyperparameters.** Both are shared, bounded, and default to values that make
  the scales slow and the floors inert; neither is swept.
- **The panel's configs change after this lands.** Turning the switches on for every arm is a
  one-key-each edit at the panel's pin step, recorded there by dated amendment.

## Open Questions

None. The re-registered grid is the panel change's decision.
