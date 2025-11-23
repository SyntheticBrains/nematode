"""Performance tests for predator-enabled environment."""

import time

import pytest
from quantumnematode.brain.actions import Action
from quantumnematode.env import DynamicForagingEnvironment


class TestPredatorPerformance:
    """Test predator environment performance."""

    @pytest.fixture
    def large_predator_env(self):
        """Create large environment with 5 predators (worst case scenario)."""
        return DynamicForagingEnvironment(
            grid_size=100,
            start_pos=(50, 50),
            foods_on_grid=50,
            target_foods_to_collect=75,
            min_food_distance=5,
            agent_exclusion_radius=10,
            gradient_decay_constant=10.0,
            gradient_strength=1.0,
            viewport_size=(11, 11),
            max_body_length=6,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            predators_enabled=True,
            num_predators=5,
            predator_speed=1.0,
            predator_detection_radius=8,
            predator_kill_radius=1,
            predator_gradient_decay=5.0,
            predator_gradient_strength=1.0,
        )

    def test_step_time_with_predators(self, large_predator_env):
        """Test that environment step time is < 100ms with predators enabled."""
        env = large_predator_env

        # Warmup
        for _ in range(10):
            env.get_state(env.agent_pos)
            env.update_predators()

        # Time 100 steps
        num_steps = 100
        times = []

        for _ in range(num_steps):
            start = time.perf_counter()

            # Simulate a full environment step
            env.get_state(env.agent_pos)
            env.update_predators()
            env.check_predator_collision()
            env.is_agent_in_danger()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds

        # Calculate statistics
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]

        # Print performance stats for visibility
        print("\nPerformance Test Results (100x100 grid, 5 predators):")  # noqa: T201
        print(f"  Average step time: {avg_time:.2f}ms")  # noqa: T201
        print(f"  Min step time:     {min_time:.2f}ms")  # noqa: T201
        print(f"  Max step time:     {max_time:.2f}ms")  # noqa: T201
        print(f"  95th percentile:   {p95_time:.2f}ms")  # noqa: T201

        # Assert performance requirement
        assert avg_time < 100, f"Average step time ({avg_time:.2f}ms) exceeds 100ms requirement"
