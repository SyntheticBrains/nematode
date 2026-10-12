# Tasks: the realistic replay renderer

- [ ] 1. **Recording**: `report/replay.py` (the format, the writer, the loader); `--record-replay` in
  `run_simulation.py` with its refusals. Tests: sub-step and step counts; recording does not change
  a seeded run; refusals.
- [ ] 2. **Geometry and time**: Catmull–Rom through the joints; the elliptical width profile;
  sub-step interpolation; the pacer. Tests.
- [ ] 3. **Drawing**: `env/realistic_renderer.py`: the agar, the lawn film and rim, the body tube and
  bulbs, the label and status. Tests under the dummy driver.
- [ ] 4. **Player**: `scripts/replay_viewer.py` with the keys and `--frames`. Tests.
- [ ] 5. **Demo configs**: the lawn-body MLP arms. Loader tests.
- [ ] 6. **Docs**: `docs/visualization.md` (recording and replaying, keys, a frame); the CHANGELOG; a
  NOTICE entry for Wormlight.
- [ ] 7. **Close-out**: full suite; hooks by exit code; validate; archive; PR.
