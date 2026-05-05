"""Frozen reference of the pre-M1 Predator movement logic.

This module preserves a byte-faithful copy of the legacy
`Predator._update_pursuit`, `Predator._update_random`, and the
branching logic in `Predator.update_position` exactly as they
existed in env.py before M1's PredatorBrain refactor. It exists
ONLY for the byte-equivalence parametrised test in
`test_predator_brain_byte_equivalence.py` — never imported by
production code.

If the legacy logic is ever re-derived (e.g. for a future M5
audit), this file is the canonical reference. Do NOT modify the
movement semantics here under any circumstances; the whole point
is that this is a frozen snapshot.

Lines below mirror env.py:529-681 (pre-M1) verbatim, with only
the class extraction so it can run independently of the rest of
the env module's state.
"""

from __future__ import annotations

import numpy as np

from quantumnematode.env import PredatorType


class _LegacyPredatorReference:
    """Frozen pre-M1 Predator implementation for byte-equivalence testing.

    Constructed identically to the new Predator (without brain or
    predator_id fields, since those were added in M1.2). The legacy
    code path used inline helpers for movement.
    """

    def __init__(  # noqa: PLR0913
        self,
        position: tuple[int, int],
        predator_type: PredatorType = PredatorType.PURSUIT,
        speed: float = 1.0,
        movement_accumulator: float = 0.0,
        detection_radius: int = 8,
        damage_radius: int = 0,
    ) -> None:
        self.position = position
        self.predator_type = predator_type
        self.speed = speed
        self.movement_accumulator = movement_accumulator
        self.detection_radius = detection_radius
        self.damage_radius = damage_radius

    def update_position(
        self,
        grid_size: int,
        rng: np.random.Generator,
        agent_pos: tuple[int, int] | None = None,
        agent_positions: list[tuple[int, int]] | None = None,
    ) -> None:
        """Verbatim copy of pre-M1 Predator.update_position (env.py:529-568)."""
        if self.predator_type == PredatorType.STATIONARY:
            return

        chase_target: tuple[int, int] | None = None
        if agent_positions is not None and len(agent_positions) > 0:
            px, py = self.position
            chase_target = min(
                agent_positions,
                key=lambda ap: abs(px - ap[0]) + abs(py - ap[1]),
            )
        elif agent_pos is not None:
            chase_target = agent_pos

        if self.predator_type == PredatorType.PURSUIT and chase_target is not None:
            self._update_pursuit(grid_size, rng, chase_target)
        else:
            self._update_random(grid_size, rng)

    def _update_random(self, grid_size: int, rng: np.random.Generator) -> None:
        """Verbatim copy of pre-M1 Predator._update_random (env.py:570-617)."""
        self.movement_accumulator += self.speed
        if self.movement_accumulator < 1.0:
            return

        max_steps_per_update = 10
        steps_taken = 0

        while self.movement_accumulator >= 1.0 and steps_taken < max_steps_per_update:
            self.movement_accumulator -= 1.0
            steps_taken += 1

            up = 0
            down = 1
            left = 2
            right = 3

            direction_choice = rng.integers(4)
            x, y = self.position

            if direction_choice == up:
                y = max(0, y - 1)
            elif direction_choice == down:
                y = min(grid_size - 1, y + 1)
            elif direction_choice == left:
                x = max(0, x - 1)
            elif direction_choice == right:
                x = min(grid_size - 1, x + 1)

            self.position = (x, y)

    def _update_pursuit(
        self,
        grid_size: int,
        rng: np.random.Generator,
        agent_pos: tuple[int, int],
    ) -> None:
        """Verbatim copy of pre-M1 Predator._update_pursuit (env.py:619-681)."""
        px, py = self.position
        ax, ay = agent_pos
        distance = abs(px - ax) + abs(py - ay)

        if distance > self.detection_radius:
            self._update_random(grid_size, rng)
            return

        self.movement_accumulator += self.speed
        if self.movement_accumulator < 1.0:
            return

        max_steps_per_update = 10
        steps_taken = 0

        while self.movement_accumulator >= 1.0 and steps_taken < max_steps_per_update:
            self.movement_accumulator -= 1.0
            steps_taken += 1

            x, y = self.position

            dx = ax - x
            dy = ay - y

            if abs(dx) >= abs(dy):
                if dx > 0:
                    x = min(grid_size - 1, x + 1)
                elif dx < 0:
                    x = max(0, x - 1)
            elif dy > 0:
                y = min(grid_size - 1, y + 1)
            elif dy < 0:
                y = max(0, y - 1)

            self.position = (x, y)
