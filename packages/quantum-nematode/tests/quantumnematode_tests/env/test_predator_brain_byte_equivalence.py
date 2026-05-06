"""Byte-equivalence tests: new Predator (with brain) vs frozen legacy reference.

This is the PRIMARY regression gate for the PredatorBrain refactor.
For every supported `PredatorType` x `speed` combination, a Predator
with `HeuristicPredatorBrain` and a `_LegacyPredatorReference` (frozen
pre-refactor implementation) MUST produce step-by-step identical
position trajectories AND identical RNG-state advancement when given
the same seed and inputs.

If any of these tests fail, the refactor introduced a behavioural
divergence -- either in branch ordering (chase_target resolution),
RNG-draw ordering, accumulator timing, or the action-to-clamp mapping.

Per the design's frozen-branch invariant: the in-range/out-of-range
decision is made ONCE per `update_position` call (not per accumulator-
step), so multi-step movement at speed > 1.0 stays committed to one
branch. Test parametrisation includes speed in {0.5, 1.0, 2.0} to
exercise fractional, single-step, and multi-step regimes.
"""

import numpy as np
import pytest
from quantumnematode.env import PredatorType
from quantumnematode.env.env import Predator

from ._legacy_predator_reference import _LegacyPredatorReference


def _run_pair_for_steps(  # noqa: PLR0913 — test harness; threads many fixtures
    legacy: _LegacyPredatorReference,
    new: Predator,
    *,
    grid_size: int,
    steps: int,
    rng_legacy: np.random.Generator,
    rng_new: np.random.Generator,
    agent_positions_factory,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Run both predators for N steps with identical inputs.

    Returns a per-step list of (legacy_pos, new_pos) for trajectory diff.
    """
    history: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for step in range(steps):
        agent_positions = agent_positions_factory(step)
        legacy.update_position(grid_size, rng_legacy, agent_positions=agent_positions)
        new.update_position(
            grid_size,
            rng_new,
            agent_positions=agent_positions,
            step_index=step,
        )
        history.append((legacy.position, new.position))
    return history


@pytest.mark.parametrize("predator_type", [PredatorType.STATIONARY, PredatorType.PURSUIT])
@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
class TestByteEquivalence:
    """Position trajectories MUST match legacy step-for-step across param combos."""

    def test_static_agent_byte_equivalent(
        self,
        predator_type: PredatorType,
        speed: float,
    ) -> None:
        """Agent at fixed position; both predators should produce identical paths."""
        grid_size = 20
        rng_legacy = np.random.default_rng(42)
        rng_new = np.random.default_rng(42)
        legacy = _LegacyPredatorReference(
            position=(5, 5),
            predator_type=predator_type,
            speed=speed,
            detection_radius=8,
            damage_radius=0,
        )
        new = Predator(
            position=(5, 5),
            predator_type=predator_type,
            speed=speed,
            detection_radius=8,
            damage_radius=0,
            predator_id="predator_test",
        )

        history = _run_pair_for_steps(
            legacy,
            new,
            grid_size=grid_size,
            steps=1000,
            rng_legacy=rng_legacy,
            rng_new=rng_new,
            agent_positions_factory=lambda step: [(15, 5)],
        )

        for step, (legacy_pos, new_pos) in enumerate(history):
            assert legacy_pos == new_pos, (
                f"divergence at step {step}: legacy={legacy_pos}, new={new_pos}"
            )

    def test_moving_agent_byte_equivalent(
        self,
        predator_type: PredatorType,
        speed: float,
    ) -> None:
        """Agent moves around the grid; predators must track legacy semantics."""
        grid_size = 20
        rng_legacy = np.random.default_rng(123)
        rng_new = np.random.default_rng(123)
        legacy = _LegacyPredatorReference(
            position=(10, 10),
            predator_type=predator_type,
            speed=speed,
            detection_radius=6,
            damage_radius=0,
        )
        new = Predator(
            position=(10, 10),
            predator_type=predator_type,
            speed=speed,
            detection_radius=6,
            damage_radius=0,
            predator_id="predator_test",
        )

        # Agent traces a deterministic circular-ish path through the grid,
        # touching both in-range and out-of-range zones for the pursuit case.
        def agent_at(step: int) -> list[tuple[int, int]]:
            radius = 5
            cx, cy = 10, 10
            angle_steps = 16
            theta = (step % angle_steps) * (2 * np.pi / angle_steps)
            x = int(cx + radius * np.cos(theta))
            y = int(cy + radius * np.sin(theta))
            return [(x, y)]

        history = _run_pair_for_steps(
            legacy,
            new,
            grid_size=grid_size,
            steps=1000,
            rng_legacy=rng_legacy,
            rng_new=rng_new,
            agent_positions_factory=agent_at,
        )

        for step, (legacy_pos, new_pos) in enumerate(history):
            assert legacy_pos == new_pos, (
                f"divergence at step {step}: legacy={legacy_pos}, new={new_pos}"
            )

    def test_multi_agent_targeting_byte_equivalent(
        self,
        predator_type: PredatorType,
        speed: float,
    ) -> None:
        """Multi-agent: predators must target nearest by Manhattan, in env's iter order."""
        grid_size = 30
        rng_legacy = np.random.default_rng(7)
        rng_new = np.random.default_rng(7)
        legacy = _LegacyPredatorReference(
            position=(15, 15),
            predator_type=predator_type,
            speed=speed,
            detection_radius=10,
            damage_radius=0,
        )
        new = Predator(
            position=(15, 15),
            predator_type=predator_type,
            speed=speed,
            detection_radius=10,
            damage_radius=0,
            predator_id="predator_test",
        )

        # 5 agents at varying distances; their relative positions shift over
        # time so the nearest-by-Manhattan target changes mid-episode.
        def agents_at(step: int) -> list[tuple[int, int]]:
            return [
                (5 + (step % 10), 5),
                (25, 5 + (step % 10)),
                (25, 25 - (step % 10)),
                (5, 25),
                (15, 5 + ((step + 3) % 10)),
            ]

        history = _run_pair_for_steps(
            legacy,
            new,
            grid_size=grid_size,
            steps=1000,
            rng_legacy=rng_legacy,
            rng_new=rng_new,
            agent_positions_factory=agents_at,
        )

        for step, (legacy_pos, new_pos) in enumerate(history):
            assert legacy_pos == new_pos, (
                f"divergence at step {step}: legacy={legacy_pos}, new={new_pos}"
            )


class TestRngStateAdvancement:
    """RNG state advancement must match legacy one-for-one.

    This is the deeper equivalence guarantee: not just final positions
    matching, but the env's RNG state evolving identically. Otherwise
    downstream consumers of the same RNG (food spawning, agent decisions)
    would silently desynchronise.
    """

    @pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
    def test_rng_state_identical_across_steps_pursuit(self, speed: float) -> None:
        """Verify rng state identical across steps pursuit."""
        rng_legacy = np.random.default_rng(99)
        rng_new = np.random.default_rng(99)
        legacy = _LegacyPredatorReference(
            position=(10, 10),
            predator_type=PredatorType.PURSUIT,
            speed=speed,
            detection_radius=6,
            damage_radius=0,
        )
        new = Predator(
            position=(10, 10),
            predator_type=PredatorType.PURSUIT,
            speed=speed,
            detection_radius=6,
            damage_radius=0,
            predator_id="predator_test",
        )

        for step in range(500):
            # Agent oscillates between in-range and out-of-range to exercise
            # both RNG-consuming (random) and RNG-free (greedy) branches.
            agent = (10 + (step % 14), 10)  # max dx 13 > detection_radius 6
            legacy.update_position(20, rng_legacy, agent_positions=[agent])
            new.update_position(20, rng_new, agent_positions=[agent], step_index=step)
            # After every call, the two RNGs must have advanced identically.
            assert rng_legacy.bit_generator.state == rng_new.bit_generator.state, (
                f"RNG state diverged at step {step}"
            )

    def test_rng_state_identical_across_steps_random(self) -> None:
        """No agents alive → both predators take random branch every step."""
        rng_legacy = np.random.default_rng(5)
        rng_new = np.random.default_rng(5)
        legacy = _LegacyPredatorReference(
            position=(10, 10),
            predator_type=PredatorType.PURSUIT,
            speed=1.0,
        )
        new = Predator(
            position=(10, 10),
            predator_type=PredatorType.PURSUIT,
            speed=1.0,
            predator_id="predator_test",
        )

        for step in range(200):
            legacy.update_position(20, rng_legacy, agent_positions=[])
            new.update_position(20, rng_new, agent_positions=[], step_index=step)
            assert rng_legacy.bit_generator.state == rng_new.bit_generator.state, (
                f"RNG state diverged at step {step}"
            )


class TestUpdatePredatorsOrderingInvariant:
    """update_predators MUST pass agent_positions in agents.values() insertion order.

    Guards against future refactors that might sort, reverse, or
    otherwise reorder the agent_positions tuple before passing to
    predator brains. Ordering matters because nearest-target tie-
    breaking on equal Manhattan distances uses Python's stable `min()`
    which returns the first element with the minimum value.
    """

    def test_predator_targets_lex_first_equidistant_agent(self) -> None:
        """With two equidistant agents, the predator chases the one inserted first.

        Verifies the ordering invariant indirectly but meaningfully:
        the env's `update_predators` builds `agent_positions` from
        `self.agents.values()` (insertion-stable since Python 3.7) and
        passes it to `Predator.update_position`, which uses Python's
        stable `min(... key=Manhattan)` to resolve `chase_target`. With
        two agents equidistant from the predator, the lex-first inserted
        agent wins — proving the env passed the positions in insertion
        order, not (e.g.) a sorted or reversed order which would change
        which agent gets targeted on a tie.
        """
        from quantumnematode.brain.actions import Action
        from quantumnematode.env import (
            DynamicForagingEnvironment,
            ForagingParams,
            PredatorParams,
        )

        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(
                foods_on_grid=3,
                target_foods_to_collect=5,
                min_food_distance=2,
                agent_exclusion_radius=2,
                gradient_decay_constant=4.0,
                gradient_strength=1.0,
            ),
            viewport_size=(11, 11),
            max_body_length=2,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            predator=PredatorParams(
                enabled=True,
                count=1,
                predator_type=PredatorType.PURSUIT,
                speed=1.0,
                detection_radius=20,  # large enough to cover both
                damage_radius=0,
            ),
            seed=42,
        )

        # The env auto-creates a default agent at start_pos (10, 10) which
        # would coincide with the predator and get auto-targeted at distance
        # 0; mark it not-alive so it's excluded from `agents.values() if alive`.
        from quantumnematode.env import DEFAULT_AGENT_ID

        env.agents[DEFAULT_AGENT_ID].alive = False

        # Add two agents at positions equidistant (Manhattan) from the
        # predator. Ordering: first_agent inserted before second_agent,
        # so insertion-order dict iteration yields first_agent first.
        # If the env ever sorts, reverses, or otherwise reorders the
        # agent_positions tuple, the predator will chase the wrong agent
        # and this test will fail.
        env.add_agent("first_agent", position=(2, 10))
        env.add_agent("second_agent", position=(18, 10))
        # Pin the predator at (10, 10): Manhattan distance 8 to each.
        env.predators[0].position = (10, 10)

        # Step the predator once. Greedy chase of the lex-first inserted
        # agent (at (2, 10)) means the predator should move LEFT (x-1).
        env.update_predators(step_index=0)
        assert env.predators[0].position == (9, 10), (
            "predator chased wrong agent: insertion order is "
            "[first_agent (2,10), second_agent (18,10)], so equidistant "
            "min() should pick first_agent and the predator should move "
            f"LEFT to (9,10), but went to {env.predators[0].position}"
        )
