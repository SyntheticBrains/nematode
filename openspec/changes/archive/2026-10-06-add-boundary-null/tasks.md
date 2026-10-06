# Tasks: A.3 — the boundary-preserving null and the registered hop predictor

Phase 8b carried control A.3. Registered in
`docs/experiments/logbooks/supporting/079-boundary-null/launch.md` before any scored run and before the
predictor is computed.

- [x] 1. **The null**: `hold_boundary` in the rewiring, `rewired_boundary_held` and `boundary_neurons()`
  in the brain; tests for the held boundary, exact degrees, held gap junctions and autapses, kept routes,
  and the pinned default null.
- [x] 2. **Configs**: two boundary-null configs through the generator; loader tests.
- [x] 3. **Analysis**: `boundary_null.py` and `hop_predictor.py`, with tests.
- [x] 4. **Pilot and preflight** — **done: 8/8 pilot runs; both levels readable; dry run clean, no outcome read.** Original scope: the boundary null learning and frozen on seeds 305–308 (A.6's, disjoint
  from the band), read with A.6's wild-type and chemical-only runs through the gate preflight; the panel
  dry-run on them.
- [x] 5. **Registration** — **done: committed; spec review added a base-effect gate, the predictor's interpretation caveat and a CHANGELOG line, before launch and before the predictor was computed.** Original scope: the launch record, then `/nematode-review-spec`.
- [x] 6. **Predictor** — **done: unresolved (rho −0.154, p 0.29, 80% CI [−0.336, +0.042], n 48).** Original scope: computed once, after the registration is committed.
- [x] 7. **Campaign** — **done: 768/768, 11.8 h.** Original scope: seeds 641–768, 768 runs.
- [x] 8. **Readout** — **done: Logbook 079; `boundary`, `no_lead`; tracker A.3 met; block V's conditions and D21 updated.** Original scope: Logbook 079; tracker A.3; D21's third null in the roadmap.
- [x] 9. **Close-out** — **done: full suite 7,094 passed; hooks pass with everything staged, judged by exit code; validated; archived.** Original scope: full suite; `git add -A` then `uv run pre-commit run --all-files`, judged by exit
  code; `openspec validate --strict`; archive and PR.
