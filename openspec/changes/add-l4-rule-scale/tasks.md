# Tasks: substrate-invariant scaling of the three-factor rule

## 1. Configuration

- [ ] 1.1 Add `plasticity_normalise_modulator`, `plasticity_normalise_trace`,
  `plasticity_scale_rate` (`(0, 1]`) and `plasticity_scale_floor` (`> 0`) to the plasticity
  config mixin with load-time bounds; defaults off / `0.01` / `1e-6`.
- [ ] 1.2 Both brains pass the four fields to the rule at construction.

## 2. The rule

- [ ] 2.1 Running modulator scale `σ` with warm start, pre-update use, EMA absorb, floor;
  `δ̃ = tanh(δ / σ)` when enabled; raw `δ` still reported.
- [ ] 2.2 Per-tensor running trace scale `ρ` over the masked entries with warm start, pre-update
  use, EMA absorb, floor; Hebbian term divided by `ρ` when enabled; decay and clamp unchanged.
- [ ] 2.3 Scales update under a freeze and in unmodulated mode; nothing is written under a freeze;
  the unmodulated modulator stays `1.0`.
- [ ] 2.4 Three telemetry keys (`plasticity_modulator`, `plasticity_modulator_scale`,
  `plasticity_trace_scale`), three history fields, recorded by the shared recorder.

## 3. Tests

- [ ] 3.1 Byte-identity with both switches off against the frozen reference (existing test keeps
  passing) and a default-config rule has both switches off.
- [ ] 3.2 Modulator: bounded in `[−1, 1]`; equals `tanh(δ / σ)` with the pre-update `σ`; warm start
  from the first `|δ|`; floor applied.
- [ ] 3.3 Trace: the Hebbian step is invariant to a constant rescaling of the trace once `ρ` has
  warmed; per-tensor scales differ when tensors differ; the mask excludes off-edge entries;
  a zero first trace defers the warm start.
- [ ] 3.4 Freeze: scales advance, weights do not. Unmodulated: modulator `1.0`, `σ` reported,
  trace normalisation applied.
- [ ] 3.5 Telemetry: the three keys are present and the recorder appends them.
- [ ] 3.6 Matched rule across substrates: with normalisation on, the connectome and the MLP take
  Hebbian steps of the same mean magnitude per unit modulator from traces that differ in
  scale by orders of magnitude.

## 4. Docs and close-out

- [ ] 4.1 `docs/architectures.md` plasticity row; `CHANGELOG.md`.
- [ ] 4.2 Pre-commit gate on all files exit 0; full suite green.
- [ ] 4.3 No implementation code or docstring references a planning document.
- [ ] 4.4 Re-review for drift, archive, review the branch, open the PR.
