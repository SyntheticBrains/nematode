## 1. Reference copies and identity tests

- [ ] 1.1 Keep a reference copy of the current `to_behaviour_steps`, `measure` and `_uniforms` path in the tests. Add the torch block-draw property test.
- [ ] 1.2 `scripts/analysis/speedup_identity_check.py baseline`, run on `main`'s code before any edit: v2 without and with presence, and v1, at the box centre and two F.0a vectors (base 99,700,000), written to `build/v2-speedup/baseline.jsonl`.

## 2. The speed-ups

- [ ] 2.1 `to_behaviour_steps`: columns via `.tolist()`, the signed speed with `math.cos`/`math.sin`, positional construction; the exact comparison test on Scheer and Chen runs.
- [ ] 2.2 `measure`, `NoiseLibrary.draw` and `_splice`: column-wise rebuild and overhead removal with the same draws and tie rule; the exact comparison test with splices and lost seconds.
- [ ] 2.3 `BatchedMechanistic`'s block uniforms, tied to the generator list, and its docstring; tests across a run boundary and a re-run.

## 3. Identity on stored records and the timing

- [ ] 3.1 `speedup_identity_check.py check`: re-simulate the baseline's jobs, attempt 2's probe answers, three observations from `candidates.jsonl` graded with presence, and three of F.0a's v1 simulations. Compare every statistic, D and presence reading exactly, and write `docs/experiments/logbooks/supporting/118-v2-speedup/identity.json`.
- [ ] 3.2 Re-time the 32-simulation bench on 16 workers in High Power mode, and one simulation on one core; record both.

## 4. Close-out

- [ ] 4.1 The Phase 10 tracker: the compute plan (High Power on; this speed-up; whole-generation batching before C.2's cost pilot; sequential stopping of W's null draws in W's design; surrogate-assisted CMA dropped). Logbook 118, the index row, AGENTS.md and the CHANGELOG.
- [ ] 4.2 Full suite; `pre-commit run --all-files` after `git add -A`; `openspec validate --strict`; archive; PR.
