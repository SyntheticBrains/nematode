## Why

`pixel_continuous` draws the kinematic body and lawns faithfully but schematically. It shows one
frame per environment step, and a step is 5 worm-seconds against an undulation of about 1.6 s, so
the body moves in jumps.

Human reviewers, and anyone shown the project, need to see a run as a worm on a plate looks: a
smooth crawl, a translucent body, lawns being grazed, and control over playback speed.

Decided with the user on 2026-10-12 (option A of three):

- a replay recorder and a realistic pygame player;
- the kinematic body, on lawns where present;
- a bright-field look;
- speed control;
- no browser viewer, no live realistic mode, and no other scenarios.

## What Changes

- **A replay recorder.** `run_simulation.py --record-replay <dir>` writes one compressed `.npz` per
  run with:

  - every body sub-step: time, segment curvatures, head and frame angle;
  - each step's lawn densities, intake, lawn occupancy, satiety, modulator concentrations and
    reversal;
  - the run's static geometry: arena, lawns and body length.

  It requires the kinematic body. It changes nothing in the simulation, since it reads the body's
  existing sub-step record.

- **A replay player.** `scripts/replay_viewer.py <file.npz>` plays a recording in a pygame window:

  - **Time:** in worm-seconds, interpolated between sub-steps.
  - **Speed:** 1×, 10× and 60× (keys `1`, `2`, `3`), pause (space), step (→), and restart.
  - **Camera:** follows by default, with zoom (`+`/`−`); `C` toggles it.
  - **Label:** an always-on label naming the model and the speed.
  - `--frames <dir>` saves frames instead of opening a window.

- **A bright-field look,** adapted from Wormlight (Apache-2.0, attributed):

  - **Body:** a Catmull–Rom tube through the body's 13 joints, with an elliptical width profile
    (Boyle, Berri & Cohen 2012), shaded as a translucent body with darker edges. Pharyngeal bulbs
    appear when the worm is large on screen.
  - **Agar:** pre-baked mottle and specks, and a darker meniscus at the walls.
  - **Lawns:** a film whose opacity follows each cell's density, smoothed, with a thicker rim.

- **Demo configs:** MLP-PPO on D.1's lawn cell through the kinematic body, learning and frozen, so a
  body-on-lawns run can be trained and replayed.

## Capabilities

**New**: `realistic-replay`: recording and replaying kinematic-body runs.

## Impact

- `scripts/run_simulation.py`: the flag, which sets the environment's posture log.
- `packages/quantum-nematode/quantumnematode/report/replay.py`: the recording format and writer.
- `packages/quantum-nematode/quantumnematode/env/realistic_renderer.py`: the drawing.
- `scripts/replay_viewer.py`: the player.
- Demo configs, tests, `docs/visualization.md`, the CHANGELOG, and a NOTICE entry for Wormlight.
- Headless runs are unchanged.
