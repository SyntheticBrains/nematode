# Tasks: rule robustness before the panel

## 1. Initial action noise

- [ ] 1.1 `initial_log_std: float = 0.0` on the plasticity mixin; both brains build the
  state-independent `log_std` from it (`torch.full`); default byte-identical.
- [ ] 1.2 Tests: the parameter equals the configured value on both brains; default builds are
  bit-identical to today's (existing frozen-reference tests keep passing); the two plastic
  wiring arms still share `log_std` at one seed.

## 2. Homeostatic incoming-norm scaling

- [ ] 2.1 `plastic_fan_in_axes` on the seam: connectome `[0]`, MLP `[1]` per layer.
- [ ] 2.2 `plasticity_homeostasis: bool = False` on the mixin; the rule captures per-unit initial
  incoming norms over the masked entries at construction, rescales after each update, skips
  zero-target units, floors the norm, clamps after the rescale; nothing under a freeze.
- [ ] 2.3 Telemetry `plasticity_norm_drift` (NaN when off); history field; recorded.
- [ ] 2.4 Tests: incoming norms equal their targets after updates on both substrates; masked-only
  norms on the connectome; zero-target units untouched; the bound still holds; off is
  bit-identical to the frozen reference; freeze writes nothing and still reports; the key.
- [ ] 2.5 Test: on the connectome the captured targets are near 1 for every neuron with inputs.

## 3. MLP activation

- [ ] 3.1 `activation: Literal["relu", "tanh"] = "relu"` on the MLP brain config; the builder emits
  the matching module and the initialiser the matching gain.
- [ ] 3.2 Tests: `relu` builds the same modules and weights as today (the MLP frozen-reference test
  keeps passing); `tanh` builds Tanh modules with gain `5/3`; the plastic MLP arm's variant test
  accounts for the extra key once the panel change sets it.

## 4. Docs and close-out

- [ ] 4.1 `docs/architectures.md` (plasticity row, MLP row); `CHANGELOG.md`.
- [ ] 4.2 Pre-commit gate on all files exit 0; full suite green.
- [ ] 4.3 No implementation code or docstring references a planning document.
- [ ] 4.4 Re-review for drift, archive, review the branch, open the PR.
