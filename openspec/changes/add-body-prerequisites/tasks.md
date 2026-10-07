# Tasks: C.0 — body prerequisites

Phase 8b C.0: the step constant, signed speed, the substrate freeze and D19 recorded, with a gate-only
validation pilot.

- [x] 1. **Step constant (C.0a)**: `env/worm_time.py` and its test on both block-V cells.
- [x] 2. **Signed speed in the environment (C.0b)**: `allow_reversal`, the signed clamp and move, and
  `speed_signed` in behaviour capture under reversal. Tests: off is byte-identical; backward motion;
  sensing and contact zones under reversal; the capture field.
- [x] 3. **Signed speed in the brains**: `signed_speed` with its refusal under discrete actions, the
  shared bounds helper, the five continuous brains reading it, and the brain–environment agreement
  check at load. Tests for each.
- [x] 4. **Emmons 2024 source**: `connectome_source`, the loader branch, and a test that Emmons differs
  from Cook in exactly the four gap pairs.
- [ ] 5. **D19 recorded (C.0d)**: roadmap D19 marked decided, with the generator's form, parameter
  sources, interface and calibration protocol; tracker C.0d.
- [ ] 6. **Pilot configs and analysis**: MLP-PPO and the Emmons settling wild type on hard350 with
  reversal, learning and frozen, through a generator with loader tests;
  `scripts/analysis/body_prerequisites.py` (the floor gate, the beside-measures) with tests.
- [ ] 7. **Pilot**: seeds 1301–1308, 32 runs; the gate read; the record committed.
- [ ] 8. **Readout**: Logbook 081; tracker C.0a, C.0b, C.0d and the substrate freeze; roadmap C.0, D19
  and D22; CHANGELOG.
- [ ] 9. **Close-out**: full suite; `git add -A` then `uv run pre-commit run --all-files`, judged by
  exit code; `openspec validate --strict`; archive and PR.
