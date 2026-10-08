# Tasks: the body drive's gain vector

- [x] 1. **The gain**: 25 learnable log-gains on the connectome's body drive, starting at zero, in the
  readout's place among the learnable parameters. Tests: the gain scales the anatomy; the trainable
  count does not depend on the wiring.
- [x] 2. **Re-probe** — **done 2026-10-09: three probes; with the gain and a dimension-matched entropy bonus (0.004) every seed of both wirings learns (wild type 66.7%, chemical-only null 52.3%, 0% floors, best 83.5%).** Original scope:: the wild type and the chemical-only null, learning and frozen, through the body at
  500 steps, seeds 1601–1604 (16 runs); plateaus, floors, trained means and noise, against the first
  probe.
- [x] 3. **Records** — **done: the probes in the design; tracker C.1e; CHANGELOG. The probe readings are committed with C.1e's pilot registration.** Original scope:: the probes' readings committed beside C.1e's plan; tracker C.1; CHANGELOG.
- [x] 4. **Close-out** — **done 2026-10-09: full suite 7425 passed, 34 skipped; hooks; validate; archived.** Original scope: full suite; hooks by exit code; `openspec validate --strict`; archive; PR.
