# Tasks: the body drive's gain vector

- [x] 1. **The gain**: 25 learnable log-gains on the connectome's body drive, starting at zero, in the
  readout's place among the learnable parameters. Tests: the gain scales the anatomy; the trainable
  count does not depend on the wiring.
- [ ] 2. **Re-probe**: the wild type and the chemical-only null, learning and frozen, through the body at
  500 steps, seeds 1601–1604 (16 runs); plateaus, floors, trained means and noise, against the first
  probe.
- [ ] 3. **Records**: the probes' readings committed beside C.1e's plan; tracker C.1; CHANGELOG.
- [ ] 4. **Close-out**: full suite; hooks by exit code; `openspec validate --strict`; archive; PR.
