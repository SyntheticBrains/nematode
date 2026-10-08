# Tasks: C.1e — the wiring contrast through the body

- [x] 1. **Substrate**: `freeze_wiring` (refused but under PPO, and with `freeze_updates`); the
  body-drive boundary. Tests.
- [ ] 1b. **Frozen-wiring probe**: the wild type and the chemical-only null under `freeze_wiring`,
  seeds 1601–1604, 8 runs; the learner is dropped before the pilot if neither learns.
- [x] 2. **Configs and analysis** — **done: the generator (pilot configs identical to the probes'); `body_wiring.py` reads the pilot (reference, minimum and floor, n, gates, McNemar); the panel's readings and verdicts are added with its registration (task 5), once the pilot fixes the map.** Original scope: the generator; `body_wiring.py` (the pilot's reference, minimum,
  sizing and gates; the panel's readings and verdicts). Tests.
- [ ] 3. **Pilot registration**: `084-body-wiring-pilot/launch.md`, with the three gain probes' and the
  frozen-wiring probe's readings committed beside it; then `/nematode-review-spec`.
- [ ] 4. **Pilot**: seeds 1701–1716, 96 runs; Logbook 084 fixes the minimum and the panel's size.
- [ ] 5. **Panel registration**: `085-body-wiring/launch.md` with the verdict map, then review and
  gate preflight.
- [ ] 6. **Panel**, from seed 1801.
- [ ] 7. **Readout**: Logbook 085; tracker C.1e; roadmap.
- [ ] 8. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
