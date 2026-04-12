"""Export screenshots of the Pixel theme renderer.

Creates staged environments with food, predators, temperature zones,
and nematodes with body segments, then saves rendered frames as PNG
images. Supports both single-agent and multi-agent modes.

Usage:
    python scripts/export_screenshot.py [output_path]
    python scripts/export_screenshot.py --multi-agent [output_path]

Default output:
    Single-agent: docs/assets/images/pixel_theme.png
    Multi-agent:  docs/assets/images/pixel_theme_multi_agent.png
"""

from __future__ import annotations

import argparse


def _export_single_agent(output_path: str) -> None:
    """Render a single-agent frame and save it as PNG."""
    import os

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
            predator_type=PredatorType.PURSUIT,
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

    env.foods = [
        (ax + 1, ay + 2),
        (ax - 2, ay + 4),
        (ax + 3, ay - 1),
        (ax - 4, ay - 3),
    ]

    pygame.init()
    renderer = PygameRenderer(viewport_size=env.viewport_size)

    temperature = env.get_temperature()
    zone = env.get_temperature_zone()
    zone_name = zone.value.upper().replace("_", " ") if zone else None

    session_text = (
        "Session:\nRun:\t\t3/10\nWins:\t\t1/10\nEaten:\t\t12/30\nSteps(Avg):\t250.00/500\n"
    )

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

    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(renderer._screen, output_path)
    print(f"Screenshot saved to {output_path}")
    renderer.close()


def _export_multi_agent(output_path: str) -> None:
    """Render a multi-agent frame and save it as PNG."""
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from quantumnematode.brain.actions import Action
    from quantumnematode.env.env import (
        DynamicForagingEnvironment,
        ForagingParams,
        HealthParams,
    )
    from quantumnematode.env.pygame_renderer import AgentRenderState, PygameRenderer
    from quantumnematode.env.theme import Theme

    env = DynamicForagingEnvironment(
        grid_size=30,
        start_pos=(15, 15),
        viewport_size=(11, 11),
        max_body_length=5,
        theme=Theme.PIXEL,
        action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        seed=42,
        foraging=ForagingParams(
            foods_on_grid=5,
            target_foods_to_collect=15,
        ),
        health=HealthParams(max_hp=100.0),
    )

    # Add 4 agents at different positions within the viewport
    agent_positions = [
        ("agent_0", (14, 16)),
        ("agent_1", (17, 14)),
        ("agent_2", (12, 13)),
        ("agent_3", (16, 18)),
    ]
    for aid, pos in agent_positions:
        env.add_agent(aid, position=pos, max_body_length=5)

    # Walk each agent a few steps to create body segments
    agent_moves = {
        "agent_0": [Action.FORWARD, Action.FORWARD, Action.LEFT, Action.FORWARD],
        "agent_1": [Action.FORWARD, Action.RIGHT, Action.FORWARD, Action.FORWARD],
        "agent_2": [Action.LEFT, Action.FORWARD, Action.FORWARD, Action.RIGHT],
        "agent_3": [Action.FORWARD, Action.FORWARD, Action.FORWARD, Action.LEFT],
    }
    for aid, moves in agent_moves.items():
        for action in moves:
            env.move_agent_for(aid, action)

    # Place food near the viewport center
    center_x, center_y = 15, 15
    env.foods = [
        (center_x + 2, center_y + 1),
        (center_x - 1, center_y + 3),
        (center_x + 3, center_y - 2),
        (center_x - 3, center_y - 1),
        (center_x, center_y + 4),
    ]

    # Build AgentRenderState snapshots
    agents_state = []
    for i, (aid, _) in enumerate(agent_positions):
        state = env.agents[aid]
        agents_state.append(
            AgentRenderState(
                agent_id=aid,
                position=state.position,
                body=list(state.body),
                direction=state.direction.value,
                alive=i != 3,  # agent_3 is dead for demo
                hp=80.0 - i * 15,
                max_hp=100.0,
                foods_collected=3 - i if i < 3 else 0,
                satiety=70.0 - i * 10,
                max_satiety=100.0,
                color_index=i,
            ),
        )

    pygame.init()
    renderer = PygameRenderer(viewport_size=env.viewport_size)

    renderer.render_multi_agent_frame(
        env=env,
        agents=agents_state,
        followed_agent_id="agent_0",
        step=85,
        max_steps=300,
        current_step=85,
    )

    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(renderer._screen, output_path)
    print(f"Multi-agent screenshot saved to {output_path}")
    renderer.close()


def main() -> None:
    """Parse arguments and export screenshot(s)."""
    parser = argparse.ArgumentParser(description="Export Pixel theme screenshots.")
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Output PNG path (default depends on mode).",
    )
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Export a multi-agent screenshot instead of single-agent.",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Export both single-agent and multi-agent screenshots.",
    )
    args = parser.parse_args()

    if args.both:
        _export_single_agent(args.output_path or "docs/assets/images/pixel_theme.png")
        _export_multi_agent(
            "docs/assets/images/pixel_theme_multi_agent.png",
        )
    elif args.multi_agent:
        _export_multi_agent(
            args.output_path or "docs/assets/images/pixel_theme_multi_agent.png",
        )
    else:
        _export_single_agent(args.output_path or "docs/assets/images/pixel_theme.png")


if __name__ == "__main__":
    main()
