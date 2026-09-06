# Tasks: substrate-invariant scaling of the three-factor rule

## 1. Configuration

- [x] 1.1 Add `plasticity_normalise_modulator`, `plasticity_normalise_trace`,
  `plasticity_scale_rate` (`(0, 1]`) and `plasticity_scale_floor` (`> 0`) to the plasticity
  config mixin with load-time bounds; defaults off / `0.01` / `1e-6`.
- [x] 1.2 Both brains pass the four fields to the rule at construction.

## 2. The rule

- [x] 2.1 Running modulator scale `σ`: bias-corrected EMA, pre-update use, floor;
  `δ̃ = tanh(δ / σ)` when enabled; raw `δ` still reported.
- [x] 2.2 Per-tensor running trace scale `ρ` over the masked entries: bias-corrected EMA over
  non-zero traces, pre-update use, floor; Hebbian term divided by `ρ` when enabled; decay and
  clamp unchanged.
- [x] 2.3 Scales update under a freeze and in unmodulated mode; nothing is written under a freeze;
  the unmodulated modulator stays `1.0`; each estimator runs only when its switch is on.
- [x] 2.4 Three telemetry keys (`plasticity_modulator`, `plasticity_modulator_scale`,
  `plasticity_trace_scale`), three history fields, recorded by the shared recorder.
- [x] 2.5 Group the scaling options in a small dataclass handed to the rule, so the constructor
  does not grow four more parameters.

## 3. Tests

- [x] 3.1 Byte-identity with both switches off against the frozen reference (existing test keeps
  passing) and a default-config rule has both switches off.
- [x] 3.2 Modulator: bounded in `[−1, 1]`; equals `tanh(δ / σ)` with the pre-update `σ`;
  bias-corrected: equals the first `|δ|` at the first step and the corrected average after;
  floor applied.
- [x] 3.3 Trace: the Hebbian step is invariant to a constant rescaling of the trace once `ρ` has
  warmed; per-tensor scales differ when tensors differ; the mask excludes off-edge entries;
  a zero trace neither updates `ρ` nor advances its count.
- [x] 3.4 Freeze: scales advance, weights do not. Unmodulated: modulator `1.0`, `σ` reported,
  trace normalisation applied.
- [x] 3.5 Telemetry: the three keys are present and the recorder appends them.
- [x] 3.6 Matched rule across substrates: with normalisation on, the connectome and the MLP take
  Hebbian steps of the same root-mean-square magnitude per unit modulator from traces that
  differ in scale by orders of magnitude.
- [x] 3.7 Config: a scale rate outside `(0, 1]` or a non-positive floor fails at load; the
  shared-values test lists the four new fields for both brains.

## 4. Docs and close-out

- [x] 4.1 `docs/architectures.md` plasticity row; `CHANGELOG.md`.
- [x] 4.2 Pre-commit gate on all files exit 0; full suite green. (4578 passed, 2 skipped, 2 xfailed, full suite including slow tests)
- [x] 4.3 No implementation code or docstring references a planning document.
- [x] 4.4 Re-review for drift, archive, review the branch, open the PR. (all seven scenarios map to a named test)
