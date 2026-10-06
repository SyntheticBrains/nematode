# Tasks: B.2a — across-step state, and its positive controls

Phase 8b carried control B.2a. Registered in
`docs/experiments/logbooks/supporting/080-across-step-state/launch.md` before any scored run.

- [x] 1. **The substrate**: `dynamics`, `membrane_tau_steps` and `bptt_chunk_length` with load-time
  refusals; the semi-implicit leaky step with `M⁻¹` precomputed; the sustained sensor current; the membrane reset at episode start and the per-step detach; the state kept out of checkpoints and
  `copy()`; a CHANGELOG line. Tests: settling byte-identical (a pinned
  short run), contraction and boundedness on the raw wild-type gap weights, carry and reset, refusals.
- [x] 2. **The replay**: start states in the rollout buffer; the chunk iterator; the sequence forward;
  the rule's chunked path; the end-of-episode update needing as many chunks as minibatches. Tests: replay equals rollout at unchanged parameters, across an episode
  boundary and with a partial final chunk; the settling minibatch draws unchanged.
- [x] 3. **Configs**: a generator for the dynamical wild type, learning and frozen, at each pilot τ on
  both cells, each its settling parent with only the three new keys; the MLP-PPO thermal target-35
  config. Loader tests.
- [x] 4. **Analysis**: `across_step_control.py` — the pilot rule, the MLP competence gate, the non-inferiority reading and the per-seed CSV, with cells as levels in the preflight's `STEMS`
  shape — with tests of every verdict.
- [x] 5. **Identity check** — **done: 12/12 identical on every Run: line and the final w_chem.** Original scope: four learning and two frozen settling runs per cell from the bands, re-run
  on this code, bit for bit against the committed logs.
- [x] 5b. **Recalibration** (added 2026-10-07 after the first pilot): `input_gain` and its refusals;
  `across_step_calibration.py` on untrained brains; the input-gain configs; the pilot repeated on 1105–1108.
- [x] 6. **Pilot and MLP control** — **done: first pilot no tau eligible; recalibrated; repeat pilot chose tau 0.2; both MLP controls pass; preflight readable.** Original scope: the τ pilot on seeds 1101–1104 (64 runs) and MLP-PPO on both cells on 1201–1208 (Logbook 060 committed only per-width means, so hard350's
  control is run rather than cited);
  the rule applied; the gate preflight on the chosen τ; cost measured.
- [x] 7. **Registration** — **done: launch.md committed after the gate preflight; readiness review found the MLP verdict recorded but not applied, fixed before any panel run.** Original scope: the launch record, then `/nematode-review-spec`.
- [ ] 8. **Campaign**: the dynamical wild type, learning and frozen, at the chosen τ on seeds 641–768
  (hard350) and 513–640 (thermal target 35).
- [ ] 9. **Readout**: Logbook 080; tracker B.2a; the roadmap's B.2 entry; what B.2b may run on, per cell.
- [ ] 10. **Close-out**: full suite; `git add -A` then `uv run pre-commit run --all-files`, judged by
  exit code; `openspec validate --strict`; archive and PR.
