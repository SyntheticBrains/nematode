# Tasks: C.1e — the wiring contrast through the body

- [x] 1. **Substrate**: `freeze_wiring` (refused but under PPO, and with `freeze_updates`); the
  body-drive boundary. Tests.
- [x] 1b. **Frozen-wiring probe** — **done 2026-10-09: `fails_floor` (wild type 0% on all seeds, null 0, 0, 25.3, 0); the learner leaves, and the pilot is PPO only, 64 runs.** Original scope:: the wild type and the chemical-only null under `freeze_wiring`,
  seeds 1601–1604, 8 runs; the learner is dropped before the pilot if neither learns.
- [x] 2. **Configs and analysis** — **done: the generator (pilot configs identical to the probes'); `body_wiring.py` reads the pilot (reference, minimum and floor, n, gates, McNemar); the panel's readings and verdicts are added with its registration (task 5), once the pilot fixes the map.** Original scope: the generator; `body_wiring.py` (the pilot's reference, minimum,
  sizing and gates; the panel's readings and verdicts). Tests.
- [x] 3. **Pilot registration** — **done 2026-10-09: launch.md, probes.json, probe-preflight.json (PPO `readable`); spec review applied (panel n never below 16; the floor's share of the wild type's auc reported; the reading learner's closure conditioned; write_csv tested).** Original scope:: `084-body-wiring-pilot/launch.md`, with the three gain probes' and the
  frozen-wiring probe's readings committed beside it; then `/nematode-review-spec`.
- [x] 4. **Pilot** — **done 2026-10-09: readable; reference +0.0229 [−0.025, +0.071]; minimum 0.0367 (the floor); sd 0.161, so n = 64 (capped, MDE 0.050). Logbook 084.** Original scope:: seeds 1701–1716, 96 runs; Logbook 084 fixes the minimum and the panel's size.
- [x] 5. **Panel registration** — **done 2026-10-09: launch.md with both verdict maps (the boundary stage's registered before launch, fixed-sequence); preflight readable, launch true; spec review applied.** Original scope:: `085-body-wiring/launch.md` with the verdict map (primary contrast only; the boundary null gated on `move_wt`), the panel's scoring in `body_wiring.py`; then review and gate preflight.
- [x] 6. **Panel** — **done 2026-10-10: 320/320 runs in 20.9 h; `unresolved_at_this_sensitivity`, +0.016 [−0.012, +0.044]; the boundary stage does not run.** Original scope:, seeds 1801–1864; the boundary stage only on `move_wt`.
- [x] 7. **Readout** — **done: Logbook 085; tracker C.1e; roadmap C.1; index.** Original scope:: Logbook 085; tracker C.1e; roadmap.
- [x] 8. **Close-out** — **done 2026-10-10: full suite 7440 passed, 34 skipped; hooks; validate; archived.** Original scope: full suite; hooks by exit code; validate; archive; PR.
