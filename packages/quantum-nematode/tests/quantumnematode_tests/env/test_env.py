"""Unit tests for nematode environments."""

import numpy as np
import pytest
from quantumnematode.brain.actions import Action
from quantumnematode.env import (
    Direction,
    DynamicForagingEnvironment,
    ForagingParams,
    HealthParams,
    PredatorParams,
    PredatorType,
    StaticEnvironment,
)
from quantumnematode.env.env import Predator, ThermotaxisParams
from quantumnematode.env.temperature import TemperatureZone
from quantumnematode.env.theme import THEME_SYMBOLS, Theme
from quantumnematode.utils.seeding import get_rng


class TestStaticEnvironment:
    """Test cases for StaticEnvironment."""

    @pytest.fixture
    def maze_env(self):
        """Create a test maze environment."""
        return StaticEnvironment(
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
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        assert env.grid_size == 20
        assert env.foraging.foods_on_grid == 5
        assert env.foraging.target_foods_to_collect == 10
        assert len(env.foods) == 5
        assert env.agent_pos == (10, 10)
        assert len(env.visited_cells) == 1
        assert (env.agent_pos[0], env.agent_pos[1]) in env.visited_cells

    def test_custom_start_position(self):
        """Test initialization with custom start position."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(5, 5),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        assert env.agent_pos == (5, 5)
        assert (5, 5) in env.visited_cells

    def test_custom_viewport(self):
        """Test initialization with custom viewport size."""
        env = DynamicForagingEnvironment(
            grid_size=50,
            foraging=ForagingParams(foods_on_grid=10, target_foods_to_collect=15),
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
            foraging=ForagingParams(
                foods_on_grid=0,  # Start with no food, we'll add manually
                target_foods_to_collect=10,
                gradient_decay_constant=5.0,
                gradient_strength=1.0,
            ),
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
        env.foraging = ForagingParams(gradient_decay_constant=3.0)

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
            foraging=ForagingParams(
                foods_on_grid=10,
                target_foods_to_collect=20,
                min_food_distance=3,
            ),
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
            foraging=ForagingParams(
                foods_on_grid=5,
                target_foods_to_collect=10,
                min_food_distance=8,
                agent_exclusion_radius=15,
            ),
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
            foraging=ForagingParams(
                foods_on_grid=5,
                target_foods_to_collect=10,
                min_food_distance=8,
                agent_exclusion_radius=15,
            ),
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
            foraging=ForagingParams(foods_on_grid=20, target_foods_to_collect=30),
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
            foraging=ForagingParams(
                foods_on_grid=5,
                target_foods_to_collect=10,
                min_food_distance=5,
                agent_exclusion_radius=8,
            ),
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
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
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

    def test_target_foods_to_collect_limit(self, env):
        """Test that food count doesn't exceed foods_on_grid (constant supply)."""
        env.foraging = ForagingParams(
            target_foods_to_collect=10,  # Victory condition
            foods_on_grid=3,  # Constant foods maintained on grid
        )

        # Manually set foods to foods_on_grid count
        env.foods = [(i, i) for i in range(3)]

        # Try to spawn more - should not exceed foods_on_grid
        for _ in range(10):
            env.spawn_food()

        # Should not exceed foods_on_grid (constant supply maintained)
        assert len(env.foods) <= env.foraging.foods_on_grid
        assert len(env.foods) == 3  # Maintains exactly foods_on_grid


class TestViewportCalculations:
    """Test cases for viewport rendering system."""

    @pytest.fixture
    def env(self):
        """Create large environment for viewport testing."""
        return DynamicForagingEnvironment(
            grid_size=50,
            start_pos=(25, 25),
            foraging=ForagingParams(foods_on_grid=10, target_foods_to_collect=20),
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
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
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
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
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
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
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
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
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
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
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
            foraging=ForagingParams(
                foods_on_grid=3,
                target_foods_to_collect=5,
                min_food_distance=3,
            ),
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
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
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
            foraging=ForagingParams(foods_on_grid=7, target_foods_to_collect=10),
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

            # Verify invariants
            assert len(env.foods) <= env.foraging.foods_on_grid
            assert all(0 <= f[0] < env.grid_size and 0 <= f[1] < env.grid_size for f in env.foods)


class TestNearestFoodDistance:
    """Test cases for nearest food distance calculation."""

    def test_nearest_food_distance(self):
        """Test Manhattan distance to nearest food."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=0, target_foods_to_collect=10),
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
            foraging=ForagingParams(foods_on_grid=0, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        env.foods = []

        nearest_dist = env.get_nearest_food_distance()
        assert nearest_dist is None


class TestPredatorMechanics:
    """Test cases for Predator class and predator mechanics."""

    @pytest.fixture
    def predator_env(self):
        """Create a test environment with predators enabled."""
        return DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            predator=PredatorParams(
                enabled=True,
                count=2,
                speed=1.0,
                detection_radius=8,
                kill_radius=0,
            ),
        )

    def test_predator_initialization(self, predator_env):
        """Test predator initialization."""
        assert predator_env.predator.enabled is True
        assert len(predator_env.predators) == 2

        for predator in predator_env.predators:
            # Check predator has valid position within grid
            assert 0 <= predator.position[0] < predator_env.grid_size
            assert 0 <= predator.position[1] < predator_env.grid_size

            # Check predator has correct speed
            assert predator.speed == 1.0

    def test_predator_movement(self, predator_env):
        """Test predator random movement."""
        predator = predator_env.predators[0]
        rng = get_rng(42)

        # Move predator multiple times
        for _ in range(10):
            predator.update_position(predator_env.grid_size, rng)

            # Check position is still valid
            assert 0 <= predator.position[0] < predator_env.grid_size
            assert 0 <= predator.position[1] < predator_env.grid_size

        # Movement is random, just verify it doesn't crash
        assert True

    def test_gradient_with_predators(self, predator_env):
        """Test that state includes gradient information with predators."""
        # Place agent at known position
        predator_env.agent_pos = (10, 10)

        # Get state which includes gradients (pass agent position)
        state = predator_env.get_state(predator_env.agent_pos)

        # State should be a tuple with gradient information (x, y components)
        assert isinstance(state, tuple)
        assert len(state) == 2

        # Verify all values are finite (no NaN or Inf)
        assert all(np.isfinite(val) for val in state)

    def test_collision_detection_kill_radius(self, predator_env):
        """Test collision detection with non-zero kill_radius using Manhattan distance."""
        predator_env.agent_pos = (10, 10)
        predator_env.predator = PredatorParams(
            enabled=True,
            count=2,
            speed=1.0,
            detection_radius=8,
            kill_radius=1,
        )

        # Distance 0: same cell → kill
        predator_env.predators[0].position = (10, 10)
        assert predator_env.check_predator_collision() is True

        # Distance 1 (cardinal neighbour) → kill
        predator_env.predators[0].position = (11, 10)
        assert predator_env.check_predator_collision() is True

        # Distance 2 → no kill
        predator_env.predators[0].position = (12, 10)
        assert predator_env.check_predator_collision() is False

        # Distance 2 (diagonal) → no kill
        predator_env.predators[0].position = (11, 11)
        assert predator_env.check_predator_collision() is False

    def test_proximity_detection(self, predator_env):
        """Test proximity detection with detection_radius."""
        # Place agent at (10, 10)
        predator_env.agent_pos = (10, 10)

        # Place predator within detection radius (8 units)
        predator_env.predators[0].position = (10, 17)  # 7 units away
        predator_env.predators = [predator_env.predators[0]]  # Keep only one

        in_danger = predator_env.is_agent_in_danger()
        assert in_danger is True

        # Place predator outside detection radius
        predator_env.predators[0].position = (10, 19)  # 9 units away

        in_danger = predator_env.is_agent_in_danger()
        assert in_danger is False

    def test_predators_disabled_by_default(self):
        """Test that predators are disabled by default."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        assert env.predator.enabled is False
        assert len(env.predators) == 0

    def test_predator_update_positions(self, predator_env):
        """Test that predator positions update each step."""
        # Update predator positions
        predator_env.update_predators()

        # Verify the method runs without error and predators remain valid
        assert len(predator_env.predators) == 2
        for predator in predator_env.predators:
            assert 0 <= predator.position[0] < predator_env.grid_size
            assert 0 <= predator.position[1] < predator_env.grid_size

    def test_predator_proximity_detection(self, predator_env):
        """Test agent danger detection when within predator detection radius.

        Note: The actual proximity penalty (-0.1) is applied by the reward calculator,
        not the environment. This test validates the environment's danger detection helper.
        """
        # Place agent at (10, 10)
        predator_env.agent_pos = (10, 10)

        # Place predator within detection radius
        predator_env.predators[0].position = (10, 15)  # 5 units away, within detection_radius=8
        predator_env.predators = [predator_env.predators[0]]

        # Agent should be in danger (within detection radius)
        assert predator_env.is_agent_in_danger() is True

        # Place predator outside detection radius
        predator_env.predators[0].position = (10, 20)  # 10 units away, outside detection_radius=8

        # Agent should not be in danger
        assert predator_env.is_agent_in_danger() is False

    def test_full_episode_with_predators(self, predator_env):
        """Integration test: Verify predators work with environment operations."""
        # Verify environment has predators
        assert len(predator_env.predators) == 2
        assert predator_env.predator.enabled is True

        # Run a few update cycles
        for _ in range(10):
            # Get current state
            state = predator_env.get_state(predator_env.agent_pos)
            assert isinstance(state, tuple)
            assert len(state) == 2

            # Update predators
            predator_env.update_predators()

            # Verify all predator positions are valid
            for predator in predator_env.predators:
                assert 0 <= predator.position[0] < predator_env.grid_size
                assert 0 <= predator.position[1] < predator_env.grid_size

            # Check collision detection works
            collision = predator_env.check_predator_collision()
            assert isinstance(collision, bool)

            # Check danger detection works
            in_danger = predator_env.is_agent_in_danger()
            assert isinstance(in_danger, bool)

    def test_config_backward_compatibility(self):
        """Test that predators are disabled by default for backward compatibility."""
        # Create environment without predator config
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        # Predators should be disabled
        assert env.predator.enabled is False
        assert len(env.predators) == 0
        assert env.predator.detection_radius == 8  # Default value
        assert env.predator.kill_radius == 0  # Default value

        # is_agent_in_danger should return False
        assert env.is_agent_in_danger() is False

        # check_predator_collision should return False
        assert env.check_predator_collision() is False

    def test_predators_spawn_outside_detection_radius(self, predator_env):
        """Test that predators spawn outside detection/damage radius of agent at initialization."""
        agent_pos = predator_env.agent_pos
        detection_radius = predator_env.predator.detection_radius
        damage_radius = predator_env.predator.damage_radius
        # Predators should spawn outside both detection and damage radius
        min_spawn_distance = max(detection_radius, damage_radius)

        # Verify all predators spawn outside the minimum safe distance
        for predator in predator_env.predators:
            # Calculate Euclidean distance (consistent with implementation)
            distance = np.sqrt(
                (predator.position[0] - agent_pos[0]) ** 2
                + (predator.position[1] - agent_pos[1]) ** 2,
            )
            assert distance > min_spawn_distance, (
                f"Predator at {predator.position} spawned within min spawn distance "
                f"({distance:.2f} <= {min_spawn_distance}) of agent at {agent_pos}"
            )

        # Agent should not be in danger at initialization
        assert predator_env.is_agent_in_danger() is False

    def test_predator_fractional_speed(self):
        """Test predator with fractional speed (< 1.0)."""
        rng = get_rng(42)
        # Speed 0.5 should move every 2 updates
        predator = Predator(position=(5, 5), speed=0.5)
        initial_pos = predator.position

        # First update: accumulator = 0.5, no movement
        predator.update_position(grid_size=10, rng=rng)
        assert predator.position == initial_pos
        assert predator.movement_accumulator == 0.5

        # Second update: accumulator = 1.0, movement occurs
        predator.update_position(grid_size=10, rng=rng)
        assert predator.position != initial_pos  # Should have moved
        assert predator.movement_accumulator == 0.0

    def test_predator_normal_speed(self):
        """Test predator with normal speed (1.0)."""
        rng = get_rng(42)
        predator = Predator(position=(5, 5), speed=1.0)
        initial_pos = predator.position

        # Each update should move exactly once
        predator.update_position(grid_size=10, rng=rng)
        first_pos = predator.position
        assert first_pos != initial_pos  # Should have moved
        assert predator.movement_accumulator == 0.0

        # Second update should also move once
        predator.update_position(grid_size=10, rng=rng)
        assert predator.position != first_pos  # Should have moved again
        assert predator.movement_accumulator == 0.0

    def test_predator_double_speed(self):
        """Test predator with double speed (2.0) takes two movement steps per update."""
        rng = get_rng(42)
        predator = Predator(position=(5, 5), speed=2.0)

        # Single update should take 2 steps (movement is random, check accumulator)
        predator.update_position(grid_size=10, rng=rng)

        # After speed 2.0, accumulator should be 0.0 (2 steps taken)
        assert predator.movement_accumulator == 0.0, (
            f"Expected accumulator 0.0 after 2 steps (speed=2.0), "
            f"got {predator.movement_accumulator}"
        )

    def test_predator_triple_speed(self):
        """Test predator with triple speed (3.0) takes three movement steps per update."""
        rng = get_rng(42)
        predator = Predator(position=(5, 5), speed=3.0)

        # Single update should take 3 steps (movement is random, check accumulator)
        predator.update_position(grid_size=10, rng=rng)

        # After speed 3.0, accumulator should be 0.0 (3 steps taken)
        assert predator.movement_accumulator == 0.0, (
            f"Expected accumulator 0.0 after 3 steps (speed=3.0), "
            f"got {predator.movement_accumulator}"
        )

    def test_predator_fractional_multi_speed(self):
        """Test predator with fractional multi-step speed (2.5)."""
        rng = get_rng(42)
        predator = Predator(position=(5, 5), speed=2.5)

        # First update: 2 steps, 0.5 remaining
        # Note: Movement is random, but accumulator should be decremented correctly
        predator.update_position(grid_size=10, rng=rng)
        assert predator.movement_accumulator == 0.5, (
            f"Expected accumulator 0.5 after speed 2.5 (2 steps taken), "
            f"got {predator.movement_accumulator}"
        )

        # Second update: accumulator 0.5 + 2.5 = 3.0, so 3 steps
        predator.update_position(grid_size=10, rng=rng)
        assert predator.movement_accumulator == 0.0, (
            f"Expected accumulator 0.0 after 3 steps, got {predator.movement_accumulator}"
        )

    def test_predator_high_speed_capped(self):
        """Test predator with very high speed is capped at 10 steps per update."""
        rng = get_rng(42)
        predator = Predator(position=(5, 5), speed=15.0)

        # Single update should be capped at 10 steps
        # Movement is random, but accumulator should show capping occurred
        predator.update_position(grid_size=20, rng=rng)

        # Accumulator should have 5.0 remaining (15.0 - 10.0 cap)
        assert predator.movement_accumulator == 5.0, (
            f"Expected accumulator 5.0 after speed 15.0 capped at 10 steps, "
            f"got {predator.movement_accumulator}"
        )

    def test_predator_speed_boundary_clamping(self):
        """Test that high-speed predators respect grid boundaries."""
        rng = get_rng(42)
        # Place predator near edge with high speed
        predator = Predator(position=(1, 1), speed=5.0)

        # Update position - should not go out of bounds
        predator.update_position(grid_size=10, rng=rng)

        # Verify position is still within grid
        assert 0 <= predator.position[0] < 10
        assert 0 <= predator.position[1] < 10

    def test_predator_zero_speed(self):
        """Test predator with zero speed never moves."""
        rng = get_rng(42)
        predator = Predator(position=(5, 5), speed=0.0)
        initial_pos = predator.position

        # Multiple updates should never move
        for _ in range(10):
            predator.update_position(grid_size=10, rng=rng)
            assert predator.position == initial_pos
            assert predator.movement_accumulator == 0.0

    def test_predator_very_slow_speed(self):
        """Test predator with very slow speed (0.25)."""
        rng = get_rng(42)
        # Use 0.25 to avoid floating point precision issues (4 * 0.25 = 1.0 exactly)
        predator = Predator(position=(5, 5), speed=0.25)
        initial_pos = predator.position

        # Should not move for 3 updates (accumulator below 1.0)
        for i in range(3):
            predator.update_position(grid_size=10, rng=rng)
            assert predator.position == initial_pos
            expected_accumulator = (i + 1) * 0.25
            assert abs(predator.movement_accumulator - expected_accumulator) < 0.001

        # 4th update should trigger movement (accumulator reaches 1.0)
        predator.update_position(grid_size=10, rng=rng)
        # Movement is random, but accumulator should reset to 0.0
        assert predator.movement_accumulator == 0.0, (
            f"Expected accumulator 0.0 after reaching 1.0, got {predator.movement_accumulator}"
        )


class TestHealthSystem:
    """Test cases for HP-based health system."""

    @pytest.fixture
    def health_env(self):
        """Create environment with health system enabled."""
        return DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            health=HealthParams(
                enabled=True,
                max_hp=100.0,
                predator_damage=10.0,
                food_healing=5.0,
            ),
        )

    @pytest.fixture
    def health_predator_env(self):
        """Create environment with both health and predators enabled."""
        return DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            health=HealthParams(
                enabled=True,
                max_hp=100.0,
                predator_damage=25.0,
                food_healing=10.0,
            ),
            predator=PredatorParams(
                enabled=True,
                count=2,
                speed=1.0,
                detection_radius=8,
                kill_radius=1,
            ),
        )

    def test_health_disabled_by_default(self):
        """Test that health system is disabled by default."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        assert env.health.enabled is False
        assert env.agent_hp == 0.0  # HP is 0 when disabled

    def test_health_initialization(self, health_env):
        """Test health system initialization."""
        assert health_env.health.enabled is True
        assert health_env.health.max_hp == 100.0
        assert health_env.health.predator_damage == 10.0
        assert health_env.health.food_healing == 5.0
        assert health_env.agent_hp == 100.0  # Starts at max HP

    def test_health_params_defaults(self):
        """Test HealthParams default values."""
        params = HealthParams()
        assert params.enabled is False
        assert params.max_hp == 100.0
        assert params.predator_damage == 10.0
        assert params.food_healing == 5.0

    def test_health_params_custom_values(self):
        """Test HealthParams with custom values."""
        params = HealthParams(
            enabled=True,
            max_hp=200.0,
            predator_damage=50.0,
            food_healing=25.0,
        )
        assert params.enabled is True
        assert params.max_hp == 200.0
        assert params.predator_damage == 50.0
        assert params.food_healing == 25.0

    def test_apply_predator_damage(self, health_env):
        """Test applying predator damage."""
        initial_hp = health_env.agent_hp
        damage = health_env.apply_predator_damage()

        assert damage == 10.0
        assert health_env.agent_hp == initial_hp - 10.0
        assert health_env.agent_hp == 90.0

    def test_apply_predator_damage_multiple_times(self, health_env):
        """Test applying predator damage multiple times."""
        for i in range(5):
            health_env.apply_predator_damage()
            expected_hp = 100.0 - (i + 1) * 10.0
            assert health_env.agent_hp == expected_hp

    def test_apply_predator_damage_does_not_go_negative(self, health_env):
        """Test that HP cannot go below zero."""
        # Apply damage 15 times (150 damage, but max HP is 100)
        for _ in range(15):
            health_env.apply_predator_damage()

        assert health_env.agent_hp == 0.0
        assert health_env.is_health_depleted() is True

    def test_apply_predator_damage_when_disabled(self):
        """Test that damage is not applied when health system is disabled."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            health=HealthParams(enabled=False),
        )

        damage = env.apply_predator_damage()
        assert damage == 0.0
        assert env.agent_hp == 0.0

    def test_apply_food_healing(self, health_env):
        """Test applying food healing."""
        # First reduce HP
        health_env.agent_hp = 80.0

        healing = health_env.apply_food_healing()

        assert healing == 5.0
        assert health_env.agent_hp == 85.0

    def test_apply_food_healing_caps_at_max(self, health_env):
        """Test that healing does not exceed max HP."""
        # Set HP close to max
        health_env.agent_hp = 98.0

        healing = health_env.apply_food_healing()

        assert healing == 2.0  # Only heals 2, not full 5
        assert health_env.agent_hp == 100.0

    def test_apply_food_healing_at_max_hp(self, health_env):
        """Test healing when already at max HP."""
        assert health_env.agent_hp == 100.0

        healing = health_env.apply_food_healing()

        assert healing == 0.0
        assert health_env.agent_hp == 100.0

    def test_apply_food_healing_when_disabled(self):
        """Test that healing is not applied when health system is disabled."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            health=HealthParams(enabled=False),
        )

        healing = env.apply_food_healing()
        assert healing == 0.0

    def test_is_health_depleted(self, health_env):
        """Test health depletion check."""
        assert health_env.is_health_depleted() is False

        health_env.agent_hp = 0.0
        assert health_env.is_health_depleted() is True

        health_env.agent_hp = 0.1
        assert health_env.is_health_depleted() is False

    def test_is_health_depleted_when_disabled(self):
        """Test that health depletion returns False when disabled."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            health=HealthParams(enabled=False),
        )

        # Even with HP at 0, should return False when disabled
        assert env.agent_hp == 0.0
        assert env.is_health_depleted() is False

    def test_reset_health(self, health_env):
        """Test resetting health to max."""
        health_env.agent_hp = 25.0
        health_env.reset_health()
        assert health_env.agent_hp == 100.0

    def test_reset_health_when_disabled(self):
        """Test that reset_health does nothing when disabled."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            health=HealthParams(enabled=False),
        )

        env.reset_health()
        assert env.agent_hp == 0.0  # Should remain 0 when disabled

    def test_health_with_predator_damage_workflow(self, health_predator_env):
        """Test complete health + predator workflow."""
        env = health_predator_env
        assert env.agent_hp == 100.0

        # Apply damage from predator contact
        damage = env.apply_predator_damage()
        assert damage == 25.0
        assert env.agent_hp == 75.0
        assert env.is_health_depleted() is False

        # Apply more damage
        env.apply_predator_damage()
        env.apply_predator_damage()
        assert env.agent_hp == 25.0

        # Heal with food
        healing = env.apply_food_healing()
        assert healing == 10.0
        assert env.agent_hp == 35.0

        # Final damage to deplete
        env.apply_predator_damage()
        assert env.agent_hp == 10.0
        env.apply_predator_damage()
        assert env.agent_hp == 0.0
        assert env.is_health_depleted() is True

    def test_environment_copy_preserves_health(self, health_env):
        """Test that environment copy preserves health state."""
        health_env.agent_hp = 50.0

        copied_env = health_env.copy()

        assert copied_env.health.enabled is True
        assert copied_env.health.max_hp == 100.0
        assert copied_env.agent_hp == 50.0

        # Modify original should not affect copy
        health_env.agent_hp = 25.0
        assert copied_env.agent_hp == 50.0

    def test_environment_copy_preserves_health_config(self, health_env):
        """Test that environment copy preserves health config."""
        copied_env = health_env.copy()

        assert copied_env.health.predator_damage == health_env.health.predator_damage
        assert copied_env.health.food_healing == health_env.health.food_healing

    def test_custom_health_values(self):
        """Test environment with custom health values."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            health=HealthParams(
                enabled=True,
                max_hp=50.0,
                predator_damage=5.0,
                food_healing=15.0,
            ),
        )

        assert env.agent_hp == 50.0
        assert env.health.max_hp == 50.0

        # Damage
        env.apply_predator_damage()
        assert env.agent_hp == 45.0

        # Heal (should cap at max)
        env.apply_food_healing()
        assert env.agent_hp == 50.0  # 45 + 15 = 60, but capped at 50


class TestPredatorTypes:
    """Test cases for different predator movement behaviors."""

    def test_predator_type_defaults_to_random(self):
        """Test that predator type defaults to RANDOM."""
        predator = Predator(position=(5, 5))
        assert predator.predator_type == PredatorType.RANDOM

    def test_stationary_predator_does_not_move(self):
        """Test that stationary predators never move."""
        predator = Predator(
            position=(5, 5),
            predator_type=PredatorType.STATIONARY,
            speed=1.0,
        )
        initial_pos = predator.position
        rng = get_rng(42)

        # Update position many times
        for _ in range(100):
            predator.update_position(grid_size=20, rng=rng, agent_pos=(10, 10))
            assert predator.position == initial_pos

    def test_pursuit_predator_moves_toward_agent_when_in_range(self):
        """Test that pursuit predators move toward agent when within detection radius."""
        # Start predator at (5, 5) with detection radius 10
        # Agent at (8, 5) - distance 3, within detection radius
        predator = Predator(
            position=(5, 5),
            predator_type=PredatorType.PURSUIT,
            speed=1.0,
            detection_radius=10,
        )
        rng = get_rng(42)

        agent_pos = (8, 5)
        initial_distance = abs(predator.position[0] - agent_pos[0]) + abs(
            predator.position[1] - agent_pos[1],
        )

        # Move predator
        predator.update_position(grid_size=20, rng=rng, agent_pos=agent_pos)

        # Should have moved closer to agent
        new_distance = abs(predator.position[0] - agent_pos[0]) + abs(
            predator.position[1] - agent_pos[1],
        )
        assert new_distance < initial_distance

    def test_pursuit_predator_moves_randomly_when_out_of_range(self):
        """Test that pursuit predators move randomly when outside detection radius."""
        # Start predator at (0, 0) with detection radius 5
        # Agent at (15, 15) - distance 30, outside detection radius
        predator = Predator(
            position=(5, 5),
            predator_type=PredatorType.PURSUIT,
            speed=1.0,
            detection_radius=5,
        )
        rng = get_rng(42)

        agent_pos = (18, 18)  # Far from predator

        # Move predator multiple times - movement should be random
        positions = [predator.position]
        for _ in range(10):
            predator.update_position(grid_size=20, rng=rng, agent_pos=agent_pos)
            positions.append(predator.position)

        # Should have moved (not stationary)
        assert len(set(positions)) > 1

    def test_pursuit_predator_catches_agent(self):
        """Test that pursuit predator eventually catches stationary agent."""
        # Predator starts close to agent
        predator = Predator(
            position=(5, 5),
            predator_type=PredatorType.PURSUIT,
            speed=1.0,
            detection_radius=10,
        )
        rng = get_rng(42)

        agent_pos = (8, 5)

        # Move predator until it reaches agent (max 10 steps should be enough)
        for _ in range(10):
            predator.update_position(grid_size=20, rng=rng, agent_pos=agent_pos)
            if predator.position == agent_pos:
                break

        assert predator.position == agent_pos

    def test_environment_with_stationary_predators(self):
        """Test environment initialization with stationary predators."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            predator=PredatorParams(
                enabled=True,
                count=3,
                predator_type=PredatorType.STATIONARY,
                speed=1.0,
            ),
        )

        assert len(env.predators) == 3
        for pred in env.predators:
            assert pred.predator_type == PredatorType.STATIONARY

        # Store initial positions
        initial_positions = [p.position for p in env.predators]

        # Update predators
        env.update_predators()

        # Stationary predators should not move
        for i, pred in enumerate(env.predators):
            assert pred.position == initial_positions[i]

    def test_environment_with_pursuit_predators(self):
        """Test environment initialization with pursuit predators."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            predator=PredatorParams(
                enabled=True,
                count=2,
                predator_type=PredatorType.PURSUIT,
                speed=1.0,
                detection_radius=15,
            ),
        )

        assert len(env.predators) == 2
        for pred in env.predators:
            assert pred.predator_type == PredatorType.PURSUIT
            assert pred.detection_radius == 15

    def test_predator_copy_preserves_type(self):
        """Test that environment copy preserves predator type."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            predator=PredatorParams(
                enabled=True,
                count=2,
                predator_type=PredatorType.PURSUIT,
                speed=0.5,
                detection_radius=12,
            ),
        )

        copied_env = env.copy()

        assert len(copied_env.predators) == 2
        for i, pred in enumerate(copied_env.predators):
            assert pred.predator_type == env.predators[i].predator_type
            assert pred.speed == env.predators[i].speed
            assert pred.detection_radius == env.predators[i].detection_radius

    def test_pursuit_greedy_movement_horizontal(self):
        """Test that pursuit uses greedy movement (larger axis first)."""
        # Predator at (0, 0), agent at (5, 2)
        # Horizontal distance (5) > vertical (2), so should move right first
        predator = Predator(
            position=(0, 0),
            predator_type=PredatorType.PURSUIT,
            speed=1.0,
            detection_radius=10,
        )
        rng = get_rng(42)

        predator.update_position(grid_size=20, rng=rng, agent_pos=(5, 2))

        # Should have moved right (x increased)
        assert predator.position == (1, 0)

    def test_pursuit_greedy_movement_vertical(self):
        """Test that pursuit uses greedy movement (larger axis first)."""
        # Predator at (0, 0), agent at (2, 5)
        # Vertical distance (5) > horizontal (2), so should move down first
        predator = Predator(
            position=(0, 0),
            predator_type=PredatorType.PURSUIT,
            speed=1.0,
            detection_radius=10,
        )
        rng = get_rng(42)

        predator.update_position(grid_size=20, rng=rng, agent_pos=(2, 5))

        # Should have moved down (y increased)
        assert predator.position == (0, 1)

    def test_predator_type_in_predator_params(self):
        """Test that PredatorParams includes predator_type field."""
        params = PredatorParams(
            enabled=True,
            count=2,
            predator_type=PredatorType.PURSUIT,
        )
        assert params.predator_type == PredatorType.PURSUIT

        # Default should be RANDOM
        default_params = PredatorParams(enabled=True, count=2)
        assert default_params.predator_type == PredatorType.RANDOM


class TestPredatorTypeSymbols:
    """Tests for predator-type-specific rendering symbols."""

    def test_theme_has_predator_type_symbols(self):
        """Test that ThemeSymbolSet includes symbols for each predator type."""
        for theme in Theme:
            symbols = THEME_SYMBOLS[theme]
            assert hasattr(symbols, "predator")
            assert hasattr(symbols, "predator_stationary")
            assert hasattr(symbols, "predator_pursuit")
            # All symbols should be non-empty strings
            assert symbols.predator
            assert symbols.predator_stationary
            assert symbols.predator_pursuit

    def test_predator_symbols_are_distinct(self):
        """Test that different predator types have distinct symbols in ASCII theme."""
        symbols = THEME_SYMBOLS[Theme.ASCII]
        # All three predator symbols should be different
        assert symbols.predator != symbols.predator_stationary
        assert symbols.predator != symbols.predator_pursuit
        assert symbols.predator_stationary != symbols.predator_pursuit

    def test_render_predators_uses_correct_symbols(self):
        """Test that rendering uses type-specific symbols."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            viewport_size=(11, 11),
            predator=PredatorParams(
                enabled=True,
                count=1,
                predator_type=PredatorType.RANDOM,
            ),
        )
        # Set predator position within viewport
        env.predators[0].position = (8, 8)

        rendered = env.render()
        rendered_str = "\n".join(rendered)
        # ASCII theme uses '#' for random predators
        assert "#" in rendered_str

    def test_render_stationary_predator_symbol(self):
        """Test that stationary predators render with their specific symbol."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            viewport_size=(11, 11),
            predator=PredatorParams(
                enabled=True,
                count=1,
                predator_type=PredatorType.STATIONARY,
            ),
        )
        # Set predator position within viewport
        env.predators[0].position = (8, 8)

        rendered = env.render()
        rendered_str = "\n".join(rendered)
        # ASCII theme uses 'X' for stationary predators
        assert "X" in rendered_str

    def test_render_pursuit_predator_symbol(self):
        """Test that pursuit predators render with their specific symbol."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            viewport_size=(11, 11),
            predator=PredatorParams(
                enabled=True,
                count=1,
                predator_type=PredatorType.PURSUIT,
            ),
        )
        # Set predator position within viewport
        env.predators[0].position = (8, 8)

        rendered = env.render()
        rendered_str = "\n".join(rendered)
        # ASCII theme uses '@' for pursuit predators
        assert "@" in rendered_str

    def test_get_predator_symbol_helper(self):
        """Test the _get_predator_symbol helper method."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            viewport_size=(11, 11),
        )
        symbols = THEME_SYMBOLS[Theme.ASCII]

        # Create predators of each type
        random_pred = Predator(
            position=(0, 0),
            predator_type=PredatorType.RANDOM,
            speed=1.0,
            detection_radius=8,
        )
        stationary_pred = Predator(
            position=(1, 1),
            predator_type=PredatorType.STATIONARY,
            speed=0.0,
            detection_radius=8,
        )
        pursuit_pred = Predator(
            position=(2, 2),
            predator_type=PredatorType.PURSUIT,
            speed=1.0,
            detection_radius=8,
        )

        assert env._get_predator_symbol(random_pred, symbols) == "#"
        assert env._get_predator_symbol(stationary_pred, symbols) == "X"
        assert env._get_predator_symbol(pursuit_pred, symbols) == "@"


class TestDamageRadius:
    """Tests for per-predator damage_radius functionality."""

    def test_predator_has_damage_radius(self):
        """Test that Predator class has damage_radius attribute."""
        pred = Predator(
            position=(5, 5),
            predator_type=PredatorType.RANDOM,
            speed=1.0,
            detection_radius=8,
            damage_radius=1,
        )
        assert pred.damage_radius == 1

    def test_predator_params_has_damage_radius(self):
        """Test that PredatorParams includes damage_radius field."""
        params = PredatorParams(
            enabled=True,
            count=2,
            damage_radius=3,
        )
        assert params.damage_radius == 3

        # Default should be 0
        default_params = PredatorParams(enabled=True, count=2)
        assert default_params.damage_radius == 0

    def test_is_agent_in_damage_radius_true(self):
        """Test that is_agent_in_damage_radius returns True when within radius."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            viewport_size=(11, 11),
            predator=PredatorParams(
                enabled=True,
                count=1,
                damage_radius=2,
            ),
        )
        # Place predator 1 cell away from agent (within damage_radius=2)
        env.predators[0].position = (11, 10)

        assert env.is_agent_in_damage_radius() is True

    def test_is_agent_in_damage_radius_false(self):
        """Test that is_agent_in_damage_radius returns False when outside radius."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            viewport_size=(11, 11),
            predator=PredatorParams(
                enabled=True,
                count=1,
                damage_radius=1,
            ),
        )
        # Place predator 3 cells away from agent (outside damage_radius=1)
        env.predators[0].position = (13, 10)

        assert env.is_agent_in_damage_radius() is False

    def test_is_agent_in_damage_radius_uses_per_predator_radius(self):
        """Test that is_agent_in_damage_radius uses each predator's own radius."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            viewport_size=(11, 11),
            predator=PredatorParams(
                enabled=True,
                count=2,
                damage_radius=1,  # Default radius
            ),
        )
        # Place first predator far away
        env.predators[0].position = (0, 0)
        env.predators[0].damage_radius = 1

        # Place second predator closer with larger radius
        env.predators[1].position = (13, 10)  # 3 cells away
        env.predators[1].damage_radius = 3  # Now agent is within this radius

        assert env.is_agent_in_damage_radius() is True

    def test_stationary_predator_large_damage_radius(self):
        """Test stationary predator can have larger damage_radius (toxic zone)."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            viewport_size=(11, 11),
            predator=PredatorParams(
                enabled=True,
                count=1,
                predator_type=PredatorType.STATIONARY,
                damage_radius=3,  # Larger toxic zone
            ),
        )
        # Place stationary predator 2 cells away (within damage_radius=3)
        env.predators[0].position = (12, 10)

        assert env.is_agent_in_damage_radius() is True
        # Verify it's stationary
        assert env.predators[0].predator_type == PredatorType.STATIONARY

    def test_damage_radius_copied_in_env_copy(self):
        """Test that damage_radius is preserved when copying environment."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            viewport_size=(11, 11),
            predator=PredatorParams(
                enabled=True,
                count=1,
                damage_radius=5,
            ),
        )
        env.predators[0].damage_radius = 7  # Set different value

        env_copy = env.copy()

        assert env_copy.predators[0].damage_radius == 7

    def test_damage_radius_initialized_from_params(self):
        """Test that predators get damage_radius from PredatorParams."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            viewport_size=(11, 11),
            predator=PredatorParams(
                enabled=True,
                count=3,
                damage_radius=4,
            ),
        )

        for pred in env.predators:
            assert pred.damage_radius == 4


class TestMechanosensation:
    """Tests for mechanosensation (boundary and predator contact detection)."""

    @pytest.fixture
    def env(self):
        """Create environment for mechanosensation testing."""
        return DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

    def test_is_agent_at_boundary_center(self, env):
        """Test agent at center is not at boundary."""
        env.agent_pos = (10, 10)
        assert env.is_agent_at_boundary() is False

    def test_is_agent_at_boundary_left_edge(self, env):
        """Test agent at left edge (x=0) is at boundary."""
        env.agent_pos = (0, 10)
        assert env.is_agent_at_boundary() is True

    def test_is_agent_at_boundary_right_edge(self, env):
        """Test agent at right edge (x=max) is at boundary."""
        env.agent_pos = (19, 10)  # grid_size=20, so max is 19
        assert env.is_agent_at_boundary() is True

    def test_is_agent_at_boundary_top_edge(self, env):
        """Test agent at top edge (y=max) is at boundary."""
        env.agent_pos = (10, 19)
        assert env.is_agent_at_boundary() is True

    def test_is_agent_at_boundary_bottom_edge(self, env):
        """Test agent at bottom edge (y=0) is at boundary."""
        env.agent_pos = (10, 0)
        assert env.is_agent_at_boundary() is True

    def test_is_agent_at_boundary_corner(self, env):
        """Test agent at corner is at boundary."""
        env.agent_pos = (0, 0)
        assert env.is_agent_at_boundary() is True

        env.agent_pos = (19, 19)
        assert env.is_agent_at_boundary() is True

        env.agent_pos = (0, 19)
        assert env.is_agent_at_boundary() is True

        env.agent_pos = (19, 0)
        assert env.is_agent_at_boundary() is True

    def test_is_agent_at_boundary_one_off_edge(self, env):
        """Test agent one cell from edge is not at boundary."""
        env.agent_pos = (1, 10)
        assert env.is_agent_at_boundary() is False

        env.agent_pos = (18, 10)
        assert env.is_agent_at_boundary() is False

        env.agent_pos = (10, 1)
        assert env.is_agent_at_boundary() is False

        env.agent_pos = (10, 18)
        assert env.is_agent_at_boundary() is False

    def test_is_agent_in_predator_contact_no_predators(self, env):
        """Test predator contact returns False when predators disabled."""
        assert env.predator.enabled is False
        assert env.is_agent_in_predator_contact() is False

    def test_is_agent_in_predator_contact_with_health(self):
        """Test predator contact uses damage radius when health system enabled."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            health=HealthParams(enabled=True, max_hp=100.0),
            predator=PredatorParams(
                enabled=True,
                count=1,
                damage_radius=2,
            ),
        )

        # Place predator within damage radius
        env.predators[0].position = (11, 10)  # 1 cell away, within damage_radius=2
        assert env.is_agent_in_predator_contact() is True

        # Place predator outside damage radius
        env.predators[0].position = (13, 10)  # 3 cells away, outside damage_radius=2
        assert env.is_agent_in_predator_contact() is False

    def test_is_agent_in_predator_contact_without_health(self):
        """Test predator contact uses kill radius when health system disabled."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            predator=PredatorParams(
                enabled=True,
                count=1,
                kill_radius=1,
            ),
        )

        # Place predator within kill radius
        env.predators[0].position = (11, 10)  # 1 cell away, within kill_radius=1
        assert env.is_agent_in_predator_contact() is True

        # Place predator outside kill radius
        env.predators[0].position = (12, 10)  # 2 cells away, outside kill_radius=1
        assert env.is_agent_in_predator_contact() is False

    def test_is_agent_in_predator_contact_exact_boundary(self):
        """Test predator contact at exact radius boundary."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=5, target_foods_to_collect=10),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            health=HealthParams(enabled=True, max_hp=100.0),
            predator=PredatorParams(
                enabled=True,
                count=1,
                damage_radius=2,
            ),
        )

        # Place predator exactly at damage radius boundary (2 cells away)
        env.predators[0].position = (12, 10)  # Manhattan distance = 2 = damage_radius
        assert env.is_agent_in_predator_contact() is True

        # Place predator just outside damage radius (3 cells away)
        env.predators[0].position = (13, 10)  # Manhattan distance = 3 > damage_radius
        assert env.is_agent_in_predator_contact() is False

    def test_wall_collision_flag_initialized_false(self, env):
        """Test wall_collision_occurred starts as False."""
        assert env.wall_collision_occurred is False

    def test_wall_collision_flag_set_on_wall_hit(self):
        """Test wall_collision_occurred is set when agent tries to move into wall."""
        env = DynamicForagingEnvironment(
            grid_size=10,
            start_pos=(0, 5),  # Start at left edge
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        # Agent faces UP by default, turn to face LEFT
        env.current_direction = Direction.LEFT
        assert env.agent_pos == (0, 5)
        assert env.wall_collision_occurred is False

        # Try to move forward (into left wall)
        env.move_agent(Action.FORWARD)

        # Position shouldn't change, collision flag should be set
        assert env.agent_pos == (0, 5)
        assert env.wall_collision_occurred is True

    def test_wall_collision_flag_not_set_on_normal_move(self):
        """Test wall_collision_occurred is False when move succeeds."""
        env = DynamicForagingEnvironment(
            grid_size=10,
            start_pos=(5, 5),  # Start in center
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        assert env.wall_collision_occurred is False

        # Move forward (should succeed, agent faces UP)
        env.move_agent(Action.FORWARD)

        # Position should change, collision flag should be False
        assert env.agent_pos == (5, 6)
        assert env.wall_collision_occurred is False

    def test_wall_collision_flag_reset_each_move(self):
        """Test wall_collision_occurred is reset at start of each move."""
        env = DynamicForagingEnvironment(
            grid_size=10,
            start_pos=(0, 5),  # Start at left edge
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        # Face left and try to move into wall
        env.current_direction = Direction.LEFT
        env.move_agent(Action.FORWARD)
        assert env.wall_collision_occurred is True

        # Turn and move successfully (face UP and move forward)
        env.current_direction = Direction.UP
        env.move_agent(Action.FORWARD)

        # Flag should be reset to False
        assert env.wall_collision_occurred is False
        assert env.agent_pos == (0, 6)

    def test_wall_collision_flag_not_set_on_stay(self):
        """Test wall_collision_occurred is False when agent stays."""
        env = DynamicForagingEnvironment(
            grid_size=10,
            start_pos=(0, 5),  # At left edge
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

        # STAY action shouldn't trigger wall collision
        env.move_agent(Action.STAY)
        assert env.wall_collision_occurred is False
        assert env.agent_pos == (0, 5)

    def test_wall_collision_flag_not_set_on_body_collision(self):
        """Test wall_collision_occurred is False when agent hits its own body.

        This is the key distinction: body collisions should NOT trigger the
        boundary penalty. Only wall collisions should.
        """
        env = DynamicForagingEnvironment(
            grid_size=10,
            start_pos=(5, 5),  # Start in center
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            theme=Theme.ASCII,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
            max_body_length=3,  # Long enough body to collide with
        )

        # Manually set up a body collision scenario
        env.agent_pos = (5, 5)
        env.body = [(4, 5), (3, 5)]  # Body to the left
        env.current_direction = Direction.LEFT  # Facing left toward body

        # Try to move forward into body at (4, 5)
        pos_before = env.agent_pos
        env.move_agent(Action.FORWARD)

        # Position shouldn't change (body collision), but flag should be False
        assert env.agent_pos == pos_before  # Didn't move due to body collision
        assert env.wall_collision_occurred is False  # NOT a wall collision


class TestThermotaxisIntegration:
    """Test thermotaxis integration with DynamicForagingEnvironment."""

    def test_thermotaxis_disabled_by_default(self):
        """Test that thermotaxis is disabled when not configured."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
        )
        assert env.thermotaxis.enabled is False
        assert env.temperature_field is None
        assert env.get_temperature() is None
        assert env.get_temperature_gradient() is None
        assert env.get_temperature_zone() is None

    def test_thermotaxis_enabled_creates_temperature_field(self):
        """Test that enabling thermotaxis creates a temperature field."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=20.0,
                gradient_direction=0.0,
                gradient_strength=0.5,
            ),
        )
        assert env.thermotaxis.enabled is True
        assert env.temperature_field is not None
        assert env.temperature_field.grid_size == 20
        assert env.temperature_field.base_temperature == 20.0
        assert env.temperature_field.gradient_strength == 0.5

    def test_get_temperature_returns_value_when_enabled(self):
        """Test that get_temperature returns a value when thermotaxis is enabled."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                base_temperature=20.0,
                gradient_direction=0.0,  # Increasing to the right
                gradient_strength=0.5,
            ),
        )
        temp = env.get_temperature()
        assert temp is not None
        # At position (10, 10) with gradient strength 0.5: 20 + 10*0.5 = 25
        assert temp == pytest.approx(25.0)

    def test_get_temperature_gradient_returns_polar_coordinates(self):
        """Test that get_temperature_gradient returns (magnitude, direction)."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                gradient_direction=0.0,  # Increasing to the right
                gradient_strength=1.0,
            ),
        )
        gradient = env.get_temperature_gradient()
        assert gradient is not None
        magnitude, direction = gradient
        # With linear gradient pointing right, direction should be ~0
        assert direction == pytest.approx(0.0, abs=0.1)
        assert magnitude > 0

    def test_get_temperature_zone_comfort(self):
        """Test zone classification returns COMFORT at cultivation temperature."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(0, 0),  # At origin
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=20.0,  # Base = Tc, no gradient offset at origin
                gradient_strength=0.0,  # No gradient
            ),
        )
        zone = env.get_temperature_zone()
        assert zone == TemperatureZone.COMFORT

    def test_get_temperature_zone_discomfort_hot(self):
        """Test zone classification returns DISCOMFORT_HOT when above comfort."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(15, 0),  # Right side of grid
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=20.0,
                gradient_direction=0.0,  # Increases to right
                gradient_strength=0.5,  # At x=15: 20 + 15*0.5 = 27.5°C
                comfort_delta=5.0,  # Comfort: 15-25°C
            ),
        )
        zone = env.get_temperature_zone()
        # 27.5°C is in discomfort zone (25-30°C)
        assert zone == TemperatureZone.DISCOMFORT_HOT

    def test_apply_temperature_effects_comfort_reward(self):
        """Test that comfort zone gives positive reward."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(0, 0),
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=20.0,
                gradient_strength=0.0,  # No gradient, always at Tc
                comfort_reward=0.1,
            ),
        )
        reward_delta, hp_damage = env.apply_temperature_effects()
        assert reward_delta == pytest.approx(0.1)
        assert hp_damage == 0.0
        assert env.steps_in_comfort_zone == 1
        assert env.total_thermotaxis_steps == 1

    def test_apply_temperature_effects_danger_damage(self):
        """Test that danger zone applies HP damage when health enabled."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(25, 0),  # Far right, hot zone
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            health=HealthParams(enabled=True, max_hp=100.0),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=20.0,
                gradient_direction=0.0,
                gradient_strength=1.0,  # At x=25: 20 + 25 = 45°C (lethal!)
                danger_hp_damage=5.0,
                lethal_hp_damage=10.0,
            ),
        )
        initial_hp = env.agent_hp
        reward_delta, hp_damage = env.apply_temperature_effects()

        # Should be in lethal zone
        assert hp_damage == 10.0
        assert env.agent_hp == initial_hp - 10.0
        assert reward_delta < 0  # Penalty applied

    def test_temperature_comfort_score_calculation(self):
        """Test comfort score is calculated correctly."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(0, 0),
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            thermotaxis=ThermotaxisParams(
                enabled=True,
                cultivation_temperature=20.0,
                base_temperature=20.0,
                gradient_strength=0.0,  # Always at comfort
            ),
        )

        # Apply effects 4 times (all in comfort)
        for _ in range(4):
            env.apply_temperature_effects()

        assert env.get_temperature_comfort_score() == pytest.approx(1.0)
        assert env.steps_in_comfort_zone == 4
        assert env.total_thermotaxis_steps == 4

    def test_reset_thermotaxis_clears_counters(self):
        """Test that reset_thermotaxis clears the tracking counters."""
        env = DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(0, 0),
            foraging=ForagingParams(foods_on_grid=3, target_foods_to_collect=5),
            thermotaxis=ThermotaxisParams(enabled=True),
        )

        # Accumulate some steps
        env.apply_temperature_effects()
        env.apply_temperature_effects()
        assert env.total_thermotaxis_steps == 2

        # Reset
        env.reset_thermotaxis()
        assert env.steps_in_comfort_zone == 0
        assert env.total_thermotaxis_steps == 0
        assert env.get_temperature_comfort_score() == 0.0
