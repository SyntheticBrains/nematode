"""Export a screenshot of the Pixel theme renderer.

Creates a staged environment with food, predators, temperature zones,
and a nematode with body segments, then saves a single rendered frame
as a PNG image. Useful for README and documentation.

Usage:
    python scripts/export_screenshot.py [output_path]

Default output: docs/assets/images/pixel_theme.png
"""

from __future__ import annotations

import sys


def main(output_path: str = "docs/assets/images/pixel_theme.png") -> None:
    """Render a single representative frame and save it as PNG."""
    import os

    # Use dummy video driver so no window actually appears
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from quantumnematode.brain.actions import Action
    from quantumnematode.env.env import (
        DynamicForagingEnvironment,
        ForagingParams,
        HealthParams,
        Predator,
        PredatorParams,
        PredatorType,
        ThermotaxisParams,
    )
    from quantumnematode.env.pygame_renderer import PygameRenderer
    from quantumnematode.env.theme import Theme

    # Create an environment with all features enabled.
    # Use a hot spot near the viewport to create visible temperature zones.
    agent_x, agent_y = 15, 15
    env = DynamicForagingEnvironment(
        grid_size=30,
        start_pos=(agent_x, agent_y),
        viewport_size=(11, 11),
        max_body_length=6,
        theme=Theme.PIXEL,
        action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        seed=42,
        foraging=ForagingParams(
            foods_on_grid=3,
            target_foods_to_collect=10,
        ),
        predator=PredatorParams(
            enabled=True,
            count=1,
            predator_type=PredatorType.STATIONARY,
            damage_radius=2,
        ),
        health=HealthParams(
            enabled=True,
            max_hp=100.0,
        ),
        thermotaxis=ThermotaxisParams(
            enabled=True,
            cultivation_temperature=20.0,
            base_temperature=20.0,
            gradient_strength=0.3,
            gradient_direction=45.0,
            hot_spots=[
                (agent_x + 4, agent_y + 3, 25.0),
            ],
            cold_spots=[
                (agent_x - 4, agent_y - 3, 20.0),
            ],
            comfort_delta=3.0,
            discomfort_delta=6.0,
            danger_delta=10.0,
        ),
    )

    # Walk the nematode a few steps to create body segments
    moves = [
        Action.FORWARD,
        Action.FORWARD,
        Action.FORWARD,
        Action.LEFT,
        Action.FORWARD,
        Action.FORWARD,
    ]
    for action in moves:
        env.move_agent(action)

    # Manually place predators within the viewport.
    # Note: after moves, agent has shifted from start_pos, so use env.agent_pos.
    ax, ay = env.agent_pos[0], env.agent_pos[1]
    env.predators = [
        Predator(
            position=(ax + 3, ay + 3),
            predator_type=PredatorType.STATIONARY,
            speed=0.0,
            detection_radius=5,
            damage_radius=2,
        ),
        Predator(
            position=(ax - 3, ay - 2),
            predator_type=PredatorType.RANDOM,
            speed=1.0,
            detection_radius=5,
            damage_radius=0,
        ),
        Predator(
            position=(ax - 4, ay + 2),
            predator_type=PredatorType.PURSUIT,
            speed=1.0,
            detection_radius=8,
            damage_radius=0,
        ),
    ]

    # Re-place food relative to current agent position
    env.foods = [
        (ax + 1, ay + 2),
        (ax - 2, ay + 4),
        (ax + 3, ay - 1),
        (ax - 4, ay - 3),
    ]

    # Build renderer (headless via dummy driver)
    pygame.init()
    renderer = PygameRenderer(
        viewport_size=env.viewport_size,
    )

    # Get temperature info for the status bar
    temperature = env.get_temperature()
    zone = env.get_temperature_zone()
    zone_name = zone.value.upper().replace("_", " ") if zone else None

    # Sample session text
    session_text = (
        "Session:\nRun:\t\t3/10\nWins:\t\t1/10\nEaten:\t\t12/30\nSteps(Avg):\t250.00/500\n"
    )

    # Render one frame
    renderer.render_frame(
        env=env,
        step=45,
        max_steps=500,
        foods_collected=2,
        target_foods=10,
        health=75.0,
        max_health=100.0,
        satiety=60.0,
        max_satiety=100.0,
        in_danger=True,
        temperature=temperature,
        zone_name=zone_name,
        session_text=session_text,
    )

    # Save the screen surface as PNG
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(renderer._screen, output_path)  # noqa: SLF001
    print(f"Screenshot saved to {output_path}")

    renderer.close()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/images/pixel_theme.png"
    main(path)
