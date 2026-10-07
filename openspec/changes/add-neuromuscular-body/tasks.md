# Tasks: C.1a–c — the neuromuscular readout and the kinematic body

Phase 8b C.1, the substrate. C.1d (the MLP positive control) and C.1e (the wiring contrast) follow as
their own changes.

- [x] 1. **Muscle anatomy and the drive map (C.1a)**: the name parser, the segment map and the signed,
  column-normalised map, with the zero-weight cells listed. Tests: parsing, segment counts, signs,
  normalisation, the 162 cells.
- [x] 2. **The body (C.1c)**: `env/body.py` with the generator, the drive modulation and the
  resistive-force step; `body_model` and its dispatch in the environment. Tests: forward and backward
  motion, turning by bias, the head's period, the arena clamp, the point worm unchanged.
- [ ] 3. **The action space (C.1b)**: `action_space`, the body-drive bounds, the action type and the
  runner dispatch; the connectome readout; MLP-PPO's width; the agreement checks and refusals. Tests
  for each.
- [ ] 4. **A smoke run**: a few episodes of each brain through the body on hard350 (Emmons, reversal on).
- [ ] 5. **Close-out**: tracker C.1a–c; CHANGELOG; full suite; `git add -A` then
  `uv run pre-commit run --all-files`, judged by exit code; `openspec validate --strict`; archive and
  PR.
