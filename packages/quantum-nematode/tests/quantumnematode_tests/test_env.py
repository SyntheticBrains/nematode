"""Unit tests for nematode environments."""

import numpy as np
import pytest
from quantumnematode.brain.actions import Action
from quantumnematode.env import (
    Direction,
    DynamicForagingEnvironment,
    MazeEnvironment,
    Theme,
)


class TestMazeEnvironment:
    """Test cases for MazeEnvironment (backward compatibility)."""

    @pytest.fixture
    def maze_env(self):
        """Create a test maze environment."""
        return MazeEnvironment(
            grid_size=10,
            start_pos=(1, 1),
            food_pos=(8, 8),
            max_body_length=5,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

    def test_maze_initialization(self, maze_env):
        """Test maze environment initialization."""
        assert maze_env.grid_size == 10
        assert maze_env.agent_pos == (1, 1)
        assert maze_env.goal == (8, 8)
        assert maze_env.current_direction == Direction.UP
        # Body starts with just head position
        assert len(maze_env.body) >= 1

    def test_maze_get_state(self, maze_env):
        """Test gradient state retrieval in maze environment."""
        strength, direction = maze_env.get_state((1, 1))

        assert isinstance(strength, float)
        assert isinstance(direction, float)
        assert strength >= 0.0
        assert -np.pi <= direction <= np.pi

    def test_maze_goal_reached(self, maze_env):
        """Test goal detection in maze environment."""
        # Move agent to goal
        maze_env.agent_pos = maze_env.goal

        assert maze_env.reached_goal()

    def test_maze_copy(self, maze_env):
        """Test maze environment copying."""
        # Modify original
        maze_env.move_agent(Action.FORWARD)

        # Copy
        copied_env = maze_env.copy()

        assert copied_env.grid_size == maze_env.grid_size
        assert copied_env.agent_pos == maze_env.agent_pos
        assert copied_env.goal == maze_env.goal

        # Modify copy - should not affect original
        original_pos = maze_env.agent_pos
        copied_env.move_agent(Action.FORWARD)
        assert maze_env.agent_pos == original_pos


class TestDynamicForagingEnvironmentInit:
    """Test cases for DynamicForagingEnvironment initialization."""

    def test_default_initialization(self):
        """Test default dynamic foraging environment initialization."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            num_initial_foods=5,
            max_active_foods=10,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        assert env.grid_size == 20
        assert env.num_initial_foods == 5
        assert env.max_active_foods == 10
        assert len(env.foods) == 5
        assert env.agent_pos == (10, 10)
        assert len(env.visited_cells) == 1
        assert (env.agent_pos[0], env.agent_pos[1]) in env.visited_cells

    def test_custom_start_position(self):
        """Test initialization with custom start position."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(5, 5),
            num_initial_foods=5,
            max_active_foods=10,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        assert env.agent_pos == (5, 5)
        assert (5, 5) in env.visited_cells

    def test_custom_viewport(self):
        """Test initialization with custom viewport size."""
        env = DynamicForagingEnvironment(
            grid_size=50,
            num_initial_foods=10,
            max_active_foods=15,
            viewport_size=(15, 15),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        assert env.viewport_size == (15, 15)


class TestGradientSuperposition:
    """Test cases for gradient superposition logic."""

    @pytest.fixture
    def env(self):
        """Create environment for gradient testing."""
        return DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            num_initial_foods=0,  # Start with no food, we'll add manually
            max_active_foods=10,
            gradient_decay_constant=5.0,
            gradient_strength=1.0,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

    def test_single_food_gradient(self, env):
        """Test gradient from single food source."""
        # Add single food at known location
        env.foods = [(15, 10)]  # Food to the right

        strength, direction = env.get_state((10, 10))

        # Should point toward food (east direction, 0 radians)
        assert strength > 0.0
        assert np.isclose(direction, 0.0, atol=0.1)  # East

    def test_two_foods_gradient_superposition(self, env):
        """Test gradient superposition from two food sources."""
        # Add two foods equidistant from agent
        env.foods = [(15, 10), (5, 10)]  # East and west

        strength, _direction = env.get_state((10, 10))

        # Gradients should cancel out (opposite directions)
        # Strength should be near zero or very small
        assert strength < 0.1

    def test_gradient_decay_with_distance(self, env):
        """Test that gradient strength decays with distance."""
        env.foods = [(15, 10)]

        # Measure gradient at increasing distances
        strength_near, _ = env.get_state((14, 10))
        strength_far, _ = env.get_state((5, 10))

        # Closer position should have stronger gradient
        assert strength_near > strength_far

    def test_gradient_direction_accuracy(self, env):
        """Test gradient direction points toward food."""
        test_cases = [
            ((15, 10), 0.0),  # Food to east
            ((5, 10), np.pi),  # Food to west
            ((10, 15), np.pi / 2),  # Food to north
            ((10, 5), -np.pi / 2),  # Food to south
        ]

        for food_pos, expected_angle in test_cases:
            env.foods = [food_pos]
            _, direction = env.get_state((10, 10))

            # Allow some tolerance for angle comparison
            # Normalize angles to [-pi, pi]
            diff = abs(direction - expected_angle)
            if diff > np.pi:
                diff = 2 * np.pi - diff

            assert diff < 0.2, f"Food at {food_pos}: expected {expected_angle}, got {direction}"

    def test_multiple_foods_superposition(self, env):
        """Test gradient superposition with multiple foods."""
        # Three foods in a cluster to the east
        env.foods = [(14, 10), (15, 11), (16, 10)]

        strength, direction = env.get_state((10, 10))

        # Should point generally east
        assert strength > 0.0
        assert -np.pi / 4 < direction < np.pi / 4  # Roughly eastward

    def test_no_food_gradient(self, env):
        """Test gradient when no food is present."""
        env.foods = []

        strength, direction = env.get_state((10, 10))

        # No food means no gradient
        assert strength == 0.0
        assert direction == 0.0

    def test_gradient_exponential_decay(self, env):
        """Test exponential decay characteristic of gradient."""
        env.foods = [(15, 10)]
        env.gradient_decay_constant = 3.0

        # Sample at different distances
        distances = [1, 3, 5, 7, 9]
        strengths = []

        for dist in distances:
            pos = (15 - dist, 10)
            strength, _ = env.get_state(pos)
            strengths.append(strength)

        # Verify exponential decay: each step should decay by roughly constant factor
        for i in range(len(strengths) - 1):
            if strengths[i] > 0 and strengths[i + 1] > 0:
                ratio = strengths[i + 1] / strengths[i]
                # Ratio should be consistent (exponential decay)
                assert 0.0 < ratio < 1.0


class TestPoissonDiskSampling:
    """Test cases for Poisson disk sampling food distribution."""

    def test_initial_food_count(self):
        """Test that correct number of foods are spawned initially."""
        env = DynamicForagingEnvironment(
            grid_size=50,
            num_initial_foods=10,
            max_active_foods=20,
            min_food_distance=3,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        assert len(env.foods) == 10

    def test_minimum_food_distance(self):
        """Test that foods respect minimum distance constraint."""
        # Use conservative parameters to reliably satisfy constraints
        env = DynamicForagingEnvironment(
            grid_size=80,
            start_pos=(40, 40),
            num_initial_foods=5,
            max_active_foods=10,
            min_food_distance=8,
            agent_exclusion_radius=15,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        # Check all pairs of foods
        for i, food1 in enumerate(env.foods):
            for food2 in env.foods[i + 1 :]:
                distance = np.sqrt((food1[0] - food2[0]) ** 2 + (food1[1] - food2[1]) ** 2)
                # Allow small tolerance for floating point
                assert distance >= 7.99, (
                    f"Foods too close: {food1} and {food2} (distance={distance})"
                )

    def test_agent_exclusion_radius(self):
        """Test that foods don't spawn too close to agent."""
        start_pos = (40, 40)
        env = DynamicForagingEnvironment(
            grid_size=80,
            start_pos=start_pos,
            num_initial_foods=5,
            max_active_foods=10,
            min_food_distance=8,
            agent_exclusion_radius=15,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        # Check all foods are outside exclusion radius
        for food in env.foods:
            distance = np.sqrt((food[0] - start_pos[0]) ** 2 + (food[1] - start_pos[1]) ** 2)
            # Allow small tolerance for floating point
            assert distance >= 14.99, (
                f"Food {food} too close to agent at {start_pos} (distance={distance})"
            )

    def test_foods_within_grid_bounds(self):
        """Test that all foods are within grid boundaries."""
        env = DynamicForagingEnvironment(
            grid_size=30,
            num_initial_foods=20,
            max_active_foods=30,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        for food in env.foods:
            assert 0 <= food[0] < 30, f"Food x-coordinate {food[0]} out of bounds"
            assert 0 <= food[1] < 30, f"Food y-coordinate {food[1]} out of bounds"

    def test_spawn_food_respects_constraints(self):
        """Test that spawned food respects all constraints."""
        env = DynamicForagingEnvironment(
            grid_size=50,
            start_pos=(25, 25),
            num_initial_foods=5,
            max_active_foods=10,
            min_food_distance=5,
            agent_exclusion_radius=8,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        initial_count = len(env.foods)

        # Spawn new food
        success = env.spawn_food()

        if success:
            assert len(env.foods) == initial_count + 1
            new_food = env.foods[-1]

            # Check minimum distance to other foods
            for other_food in env.foods[:-1]:
                distance = np.sqrt(
                    (new_food[0] - other_food[0]) ** 2 + (new_food[1] - other_food[1]) ** 2,
                )
                assert distance >= 5

            # Check agent exclusion
            agent_dist = np.sqrt((new_food[0] - 25) ** 2 + (new_food[1] - 25) ** 2)
            assert agent_dist >= 8


class TestSatietySystem:
    """Test cases for satiety and food consumption."""

    @pytest.fixture
    def env(self):
        """Create environment for satiety testing."""
        return DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            num_initial_foods=5,
            max_active_foods=10,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

    def test_food_consumption(self, env):
        """Test food consumption when agent reaches food."""
        # Place food at agent position
        env.foods = [(10, 10), (15, 15)]
        initial_food_count = len(env.foods)

        # Consume food
        consumed = env.consume_food()

        assert consumed is not None
        assert consumed == (10, 10)
        assert len(env.foods) == initial_food_count  # Should spawn new food immediately

    def test_no_consumption_when_no_food(self, env):
        """Test that consume returns None when no food at position."""
        env.foods = [(15, 15)]  # Food elsewhere
        env.agent_pos = (10, 10)

        consumed = env.consume_food()

        assert consumed is None
        assert len(env.foods) == 1  # No change

    def test_food_respawn_after_consumption(self, env):
        """Test that food respawns after being consumed."""
        initial_foods = [(10, 10), (5, 5), (15, 15)]
        env.foods = initial_foods.copy()
        env.agent_pos = (10, 10)

        # Consume food
        env.consume_food()

        # Should still have same number of foods (one consumed, one spawned)
        assert len(env.foods) == len(initial_foods)

        # Original food should be gone
        assert (10, 10) not in env.foods

    def test_max_active_foods_limit(self, env):
        """Test that food count doesn't exceed max_active_foods."""
        env.max_active_foods = 5

        # Manually set foods to max
        env.foods = [(i, i) for i in range(5)]

        # Try to spawn more
        for _ in range(10):
            env.spawn_food()

        # Should not exceed max
        assert len(env.foods) <= 5


class TestViewportCalculations:
    """Test cases for viewport rendering system."""

    @pytest.fixture
    def env(self):
        """Create large environment for viewport testing."""
        return DynamicForagingEnvironment(
            grid_size=50,
            start_pos=(25, 25),
            num_initial_foods=10,
            max_active_foods=20,
            viewport_size=(11, 11),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

    def test_viewport_centered_on_agent(self, env):
        """Test that viewport is centered on agent."""
        # Agent at (25, 25), viewport 11x11
        # Viewport should be (20, 20) to (30, 30) - centered on agent

        # Access the _render_grid method to get viewport
        viewport = env._get_viewport_bounds()
        grid = env._render_grid(env.foods, viewport=viewport)

        # Grid should be 11x11 (viewport size)
        assert len(grid) == 11
        assert len(grid[0]) == 11

    def test_viewport_at_grid_edge(self):
        """Test viewport clamping at grid edges."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(2, 2),  # Near edge
            num_initial_foods=5,
            max_active_foods=10,
            viewport_size=(11, 11),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        # Should clamp to grid boundaries - viewport will be smaller
        viewport = env._get_viewport_bounds()
        grid = env._render_grid(env.foods, viewport=viewport)

        # At position (2,2) with viewport 11x11, we're near edge
        # Viewport will be clamped to available space
        assert len(grid) <= 11
        assert len(grid[0]) <= 11
        assert len(grid) > 0
        assert len(grid[0]) > 0

    def test_viewport_corner_case(self):
        """Test viewport at grid corner."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(0, 0),  # Corner
            num_initial_foods=5,
            max_active_foods=10,
            viewport_size=(11, 11),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        viewport = env._get_viewport_bounds()
        grid = env._render_grid(env.foods, viewport=viewport)

        # At corner, viewport will be clamped to grid size
        assert len(grid) <= 11
        assert len(grid[0]) <= 11
        assert len(grid) > 0
        assert len(grid[0]) > 0

    def test_full_grid_with_none_viewport(self):
        """Test rendering full grid when viewport is None."""
        env = DynamicForagingEnvironment(
            grid_size=10,
            start_pos=(5, 5),
            num_initial_foods=3,
            max_active_foods=5,
            viewport_size=(11, 11),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        # Pass viewport=None to render full grid
        grid = env._render_grid(env.foods, viewport=None)

        # Should be full grid size
        assert len(grid) == 10
        assert len(grid[0]) == 10


class TestExplorationTracking:
    """Test cases for visited cell tracking."""

    @pytest.fixture
    def env(self):
        """Create environment for exploration testing."""
        return DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            num_initial_foods=5,
            max_active_foods=10,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

    def test_initial_position_visited(self, env):
        """Test that initial position is marked as visited."""
        assert (10, 10) in env.visited_cells
        assert len(env.visited_cells) == 1

    def test_visited_cells_tracking(self, env):
        """Test that visited cells are tracked as agent moves."""
        # Manually add visited cells (simulating agent movement)
        env.visited_cells.add((11, 10))
        env.visited_cells.add((12, 10))

        assert len(env.visited_cells) == 3
        assert (10, 10) in env.visited_cells
        assert (11, 10) in env.visited_cells
        assert (12, 10) in env.visited_cells

    def test_visited_cells_no_duplicates(self, env):
        """Test that visiting same cell twice doesn't duplicate."""
        env.visited_cells.add((10, 10))  # Already visited
        env.visited_cells.add((11, 10))
        env.visited_cells.add((11, 10))  # Duplicate

        assert len(env.visited_cells) == 2


class TestEnvironmentCopy:
    """Test cases for environment copying."""

    def test_dynamic_env_copy(self):
        """Test deep copy of dynamic foraging environment."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(5, 5),
            num_initial_foods=5,
            max_active_foods=10,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        # Modify original
        env.move_agent(Action.FORWARD)
        original_pos = env.agent_pos
        original_foods = env.foods.copy()

        # Copy
        copied_env = env.copy()

        assert copied_env.grid_size == env.grid_size
        assert copied_env.agent_pos == env.agent_pos
        assert copied_env.foods == env.foods

        # Modify copy
        copied_env.move_agent(Action.FORWARD)

        # Original should be unchanged
        assert env.agent_pos == original_pos
        assert env.foods == original_foods


class TestEnvironmentIntegration:
    """Integration tests for environment functionality."""

    def test_complete_foraging_workflow(self):
        """Test complete workflow: spawn, navigate, consume, respawn."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            num_initial_foods=3,
            max_active_foods=5,
            min_food_distance=3,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        initial_food_count = len(env.foods)

        # Find a food position
        if len(env.foods) > 0:
            target_food = env.foods[0]

            # Move agent to food
            env.agent_pos = target_food

            # Consume
            consumed = env.consume_food()

            assert consumed == target_food
            assert target_food not in env.foods
            assert len(env.foods) == initial_food_count  # Respawned

    def test_multiple_food_consumption(self):
        """Test consuming multiple foods in sequence."""
        env = DynamicForagingEnvironment(
            grid_size=30,
            start_pos=(15, 15),
            num_initial_foods=5,
            max_active_foods=10,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        foods_consumed = 0

        # Consume up to 3 foods
        for _ in range(3):
            if len(env.foods) > 0:
                target = env.foods[0]
                env.agent_pos = target
                consumed = env.consume_food()
                if consumed:
                    foods_consumed += 1

        assert foods_consumed > 0

    def test_environment_state_consistency(self):
        """Test that environment maintains consistent state."""
        env = DynamicForagingEnvironment(
            grid_size=25,
            start_pos=(12, 12),
            num_initial_foods=7,
            max_active_foods=10,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        # Perform various operations
        for _ in range(10):
            # Get state
            strength, direction = env.get_state(env.agent_pos)
            assert np.isfinite(strength)
            assert np.isfinite(direction)

            # Try to consume
            env.consume_food()

            # Try to spawn
            env.spawn_food()

            # Verify invariants
            assert len(env.foods) <= env.max_active_foods
            assert all(0 <= f[0] < env.grid_size and 0 <= f[1] < env.grid_size for f in env.foods)


class TestNearestFoodDistance:
    """Test cases for nearest food distance calculation."""

    def test_nearest_food_distance(self):
        """Test Manhattan distance to nearest food."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            num_initial_foods=0,
            max_active_foods=10,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        # Add foods at known positions
        env.foods = [(15, 10), (10, 5), (8, 8)]

        nearest_dist = env.get_nearest_food_distance()

        # Nearest should be (8, 8) with distance 4
        expected_dist = abs(10 - 8) + abs(10 - 8)
        assert nearest_dist == expected_dist

    def test_no_food_distance(self):
        """Test distance when no food exists."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            num_initial_foods=0,
            max_active_foods=10,
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        env.foods = []

        nearest_dist = env.get_nearest_food_distance()
        assert nearest_dist is None
