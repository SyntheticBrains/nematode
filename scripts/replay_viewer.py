#!/usr/bin/env python
r"""Replay a recorded kinematic-body run as a worm on an agar plate, at any speed.

Record a run first with ``run_simulation.py --record-replay <dir>``, which writes
``<dir>/<session>-run<NNNN>.npz`` for every run of a kinematic-body configuration.

Keys: ``1`` ``2`` ``3`` set 1x, 10x, 60x worm time; space pauses; the right arrow steps one sub-step
while paused; ``R`` restarts; ``C`` toggles the camera between following the worm and the whole
plate; ``+`` and ``-`` zoom the followed view; ``Esc`` quits.

``--frames <dir>`` renders without a window and saves every frame as a PNG, at ``--fps`` frames per
wall-second of playback; assemble them with ffmpeg if a video is wanted.

Usage::

    uv run python scripts/replay_viewer.py recordings/<session>-run0001.npz
    SDL_VIDEODRIVER=dummy uv run python scripts/replay_viewer.py recordings/<file>.npz \\
        --frames frames/ --speed 10
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The brain package first: importing the env package first creates an import cycle that this order
# breaks.
import quantumnematode.brain  # noqa: F401
from quantumnematode.env.realistic_renderer import SPEEDS, RealisticRenderer, advance
from quantumnematode.report.replay import load_replay

ZOOM_STEP = 1.25
MIN_VIEW_BODY_LENGTHS = 1.5
MAX_VIEW_BODY_LENGTHS = 40.0


def export_frames(path: Path, out: Path, *, speed: float, fps: float, size: int) -> int:
    """Render every frame of the replay at ``speed`` without a window; return the frame count."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    replay = load_replay(path)
    pygame.display.init()
    surface = pygame.Surface((size, size))
    renderer = RealisticRenderer(replay, size)
    out.mkdir(parents=True, exist_ok=True)
    clock, frame = 0.0, 0
    while True:
        renderer.draw(surface, clock, speed)
        pygame.image.save(surface, str(out / f"{path.stem}-frame{frame:05d}.png"))
        frame += 1
        if clock >= replay.duration:
            return frame
        clock = min(replay.duration, clock + speed / fps)


@dataclass
class Controls:
    """The player's state: the worm-time clock, the speed and whether it is paused."""

    clock: float = 0.0
    speed: float = SPEEDS[1]
    paused: bool = False


def apply_key(  # noqa: PLR0913 - the key, the controls and what the key may change
    key: int,
    keys: Any,  # noqa: ANN401 - the pygame module, whose key constants are plain ints
    controls: Controls,
    renderer: RealisticRenderer,
    duration: float,
    substep: float,
) -> None:
    """Apply one key press to the controls and the renderer's camera."""
    if key in (keys.K_1, keys.K_2, keys.K_3):
        controls.speed = SPEEDS[key - keys.K_1]
    elif key == keys.K_SPACE:
        controls.paused = not controls.paused
    elif key == keys.K_RIGHT and controls.paused:
        controls.clock = min(duration, controls.clock + substep)
    elif key == keys.K_r:
        controls.clock = 0.0
    elif key == keys.K_c:
        renderer.follow = not renderer.follow
    elif key in (keys.K_PLUS, keys.K_EQUALS, keys.K_KP_PLUS):
        renderer.view_body_lengths = max(
            MIN_VIEW_BODY_LENGTHS,
            renderer.view_body_lengths / ZOOM_STEP,
        )
    elif key in (keys.K_MINUS, keys.K_KP_MINUS):
        renderer.view_body_lengths = min(
            MAX_VIEW_BODY_LENGTHS,
            renderer.view_body_lengths * ZOOM_STEP,
        )


def play(path: Path, *, speed: float, size: int) -> int:
    """Play the replay in a window until it is closed."""
    import pygame

    replay = load_replay(path)
    pygame.display.init()
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption(f"Replay: {path.name}")
    renderer = RealisticRenderer(replay, size)
    ticker = pygame.time.Clock()
    controls = Controls(speed=speed)
    times = replay.substep_time
    substep = float(times[1] - times[0]) if len(times) > 1 else 0.25
    while True:
        wall = ticker.tick(60) / 1000.0
        for event in pygame.event.get():
            quit_key = event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            if event.type == pygame.QUIT or quit_key:
                pygame.quit()
                return 0
            if event.type == pygame.KEYDOWN:
                apply_key(event.key, pygame, controls, renderer, replay.duration, substep)
        if not controls.paused:
            controls.clock = advance(controls.clock, wall, controls.speed, replay.duration)
        renderer.draw(screen, controls.clock, controls.speed)
        pygame.display.flip()


def main(argv: list[str] | None = None) -> int:
    """CLI: play a replay, or export its frames."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("recording", type=Path)
    ap.add_argument("--speed", type=float, choices=SPEEDS, default=SPEEDS[1])
    ap.add_argument("--size", type=int, default=800, help="window size in pixels")
    ap.add_argument("--frames", type=Path, help="save frames here instead of opening a window")
    ap.add_argument("--fps", type=float, default=30.0, help="frames per wall-second with --frames")
    args = ap.parse_args(argv)
    try:
        if args.frames is not None:
            count = export_frames(
                args.recording,
                args.frames,
                speed=args.speed,
                fps=args.fps,
                size=args.size,
            )
            print(f"wrote {count} frames to {args.frames}")
            return 0
        return play(args.recording, speed=args.speed, size=args.size)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
