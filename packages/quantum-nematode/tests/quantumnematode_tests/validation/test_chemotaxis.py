"""Tests for chemotaxis index calculation and metrics."""

from quantumnematode.validation.chemotaxis import (
    CI_THRESHOLD_EXCELLENT,
    CI_THRESHOLD_MINIMUM,
    CI_THRESHOLD_TARGET,
    ChemotaxisMetrics,
    ValidationLevel,
    ValidationResult,
    calculate_chemotaxis_index,
    calculate_chemotaxis_index_stepwise,
    calculate_chemotaxis_metrics,
    calculate_chemotaxis_metrics_stepwise,
    get_validation_level,
)


class TestCalculateChemotaxisIndex:
    """Test the core chemotaxis index calculation."""

    def test_empty_positions_returns_zero(self):
        """Test that empty trajectory returns neutral CI."""
        ci, n_attract, n_control = calculate_chemotaxis_index(
            positions=[],
            food_positions=[(10.0, 10.0)],
        )
        assert ci == 0.0
        assert n_attract == 0
        assert n_control == 0

    def test_all_positions_near_food(self):
        """Test that all positions near food gives CI = 1.0."""
        # All positions within radius 5.0 of food at (10, 10)
        positions = [(10.0, 10.0), (11.0, 11.0), (9.0, 9.0), (10.5, 10.5)]
        food_positions = [(10.0, 10.0)]

        ci, n_attract, n_control = calculate_chemotaxis_index(
            positions,
            food_positions,
            attractant_zone_radius=5.0,
        )

        assert ci == 1.0
        assert n_attract == 4
        assert n_control == 0

    def test_all_positions_far_from_food(self):
        """Test that all positions far from food gives CI = -1.0."""
        # All positions far from food at (10, 10)
        positions = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        food_positions = [(50.0, 50.0)]

        ci, n_attract, n_control = calculate_chemotaxis_index(
            positions,
            food_positions,
            attractant_zone_radius=5.0,
        )

        assert ci == -1.0
        assert n_attract == 0
        assert n_control == 4

    def test_equal_time_near_and_far(self):
        """Test that equal time near/far gives CI = 0.0."""
        # 2 positions near, 2 positions far
        positions = [
            (10.0, 10.0),  # near
            (11.0, 11.0),  # near
            (0.0, 0.0),  # far
            (1.0, 1.0),  # far
        ]
        food_positions = [(10.0, 10.0)]

        ci, n_attract, n_control = calculate_chemotaxis_index(
            positions,
            food_positions,
            attractant_zone_radius=5.0,
        )

        assert ci == 0.0
        assert n_attract == 2
        assert n_control == 2

    def test_custom_attractant_radius(self):
        """Test that attractant zone radius is respected."""
        positions = [(10.0, 10.0), (12.0, 12.0)]  # dist 0, dist ~2.83
        food_positions = [(10.0, 10.0)]

        # With radius 2.0, only first position is near
        ci, n_attract, n_control = calculate_chemotaxis_index(
            positions,
            food_positions,
            attractant_zone_radius=2.0,
        )
        assert n_attract == 1
        assert n_control == 1
        assert ci == 0.0

        # With radius 5.0, both positions are near
        ci, n_attract, n_control = calculate_chemotaxis_index(
            positions,
            food_positions,
            attractant_zone_radius=5.0,
        )
        assert n_attract == 2
        assert n_control == 0
        assert ci == 1.0

    def test_multiple_food_sources(self):
        """Test with multiple food sources - near ANY food counts."""
        positions = [
            (10.0, 10.0),  # near food1
            (30.0, 30.0),  # near food2
            (50.0, 50.0),  # far from all
        ]
        food_positions = [(10.0, 10.0), (30.0, 30.0)]

        ci, n_attract, n_control = calculate_chemotaxis_index(
            positions,
            food_positions,
            attractant_zone_radius=5.0,
        )

        # 2 near (food1 and food2), 1 far
        assert n_attract == 2
        assert n_control == 1
        # CI = (2 - 1) / 3 = 0.333...
        assert abs(ci - (1 / 3)) < 0.001

    def test_position_on_exact_radius_boundary(self):
        """Test that position exactly on radius boundary counts as near."""
        positions = [(15.0, 10.0)]  # exactly 5.0 units from food
        food_positions = [(10.0, 10.0)]

        _ci, n_attract, n_control = calculate_chemotaxis_index(
            positions,
            food_positions,
            attractant_zone_radius=5.0,
        )

        assert n_attract == 1
        assert n_control == 0

    def test_no_food_positions(self):
        """Test with no food - all positions count as control."""
        positions = [(0.0, 0.0), (1.0, 1.0)]
        food_positions = []

        ci, n_attract, n_control = calculate_chemotaxis_index(
            positions,
            food_positions,
            attractant_zone_radius=5.0,
        )

        assert n_attract == 0
        assert n_control == 2
        assert ci == -1.0


class TestCalculateChemotaxisMetrics:
    """Test comprehensive chemotaxis metrics calculation."""

    def test_empty_trajectory_returns_unreliable_metrics(self):
        """Test that empty trajectory returns unreliable metrics."""
        metrics = calculate_chemotaxis_metrics(
            positions=[],
            food_positions=[(10.0, 10.0)],
        )

        assert metrics.chemotaxis_index == 0.0
        assert metrics.time_in_attractant == 0.0
        assert metrics.approach_frequency == 0.0
        assert metrics.path_efficiency == 1.0
        assert metrics.total_steps == 0
        assert metrics.reliable is False

    def test_short_trajectory_unreliable(self):
        """Test that trajectory with fewer than minimum steps is unreliable."""
        metrics = calculate_chemotaxis_metrics(
            positions=[(0.0, 0.0), (1.0, 1.0)],  # 2 positions
            food_positions=[(10.0, 10.0)],
            minimum_reliable_steps=10,
        )

        assert metrics.total_steps == 2
        assert metrics.reliable is False

    def test_long_trajectory_reliable(self):
        """Test that trajectory with enough steps is reliable."""
        positions = [(float(i), float(i)) for i in range(20)]
        metrics = calculate_chemotaxis_metrics(
            positions=positions,
            food_positions=[(100.0, 100.0)],
            minimum_reliable_steps=10,
        )

        assert metrics.total_steps == 20
        assert metrics.reliable is True

    def test_time_in_attractant_calculation(self):
        """Test time in attractant zone calculation."""
        # 3 of 4 positions near food
        positions = [(10.0, 10.0), (11.0, 11.0), (12.0, 12.0), (50.0, 50.0)]
        metrics = calculate_chemotaxis_metrics(
            positions=positions,
            food_positions=[(10.0, 10.0)],
            attractant_zone_radius=5.0,
        )

        assert metrics.time_in_attractant == 0.75  # 3/4

    def test_approach_frequency_all_approaching(self):
        """Test approach frequency when always moving toward food."""
        # Each step gets closer to food at (10, 10)
        positions = [(0.0, 0.0), (3.0, 3.0), (6.0, 6.0), (9.0, 9.0)]
        metrics = calculate_chemotaxis_metrics(
            positions=positions,
            food_positions=[(10.0, 10.0)],
        )

        assert metrics.approach_frequency == 1.0

    def test_approach_frequency_never_approaching(self):
        """Test approach frequency when always moving away from food."""
        # Each step gets further from food at (0, 0)
        positions = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
        metrics = calculate_chemotaxis_metrics(
            positions=positions,
            food_positions=[(0.0, 0.0)],
        )

        assert metrics.approach_frequency == 0.0

    def test_approach_frequency_mixed(self):
        """Test approach frequency with mixed movement."""
        # 2 approaching, 1 retreating
        positions = [(0.0, 0.0), (5.0, 5.0), (8.0, 8.0), (6.0, 6.0)]
        metrics = calculate_chemotaxis_metrics(
            positions=positions,
            food_positions=[(10.0, 10.0)],
        )

        # Steps: 0->5 (approach), 5->8 (approach), 8->6 (retreat)
        # 2/3 = 0.666...
        assert abs(metrics.approach_frequency - (2 / 3)) < 0.001

    def test_path_efficiency_optimal_path(self):
        """Test path efficiency with optimal straight-line path."""
        # Straight line to food
        positions = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        metrics = calculate_chemotaxis_metrics(
            positions=positions,
            food_positions=[(10.0, 0.0)],
        )

        assert metrics.path_efficiency == 1.0

    def test_path_efficiency_inefficient_path(self):
        """Test path efficiency with inefficient zigzag path."""
        # Zigzag path, actual distance much longer than optimal
        positions = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
        metrics = calculate_chemotaxis_metrics(
            positions=positions,
            food_positions=[(10.0, 0.0)],
        )

        # optimal = 10, actual = 10 + 10 + 10 = 30
        assert abs(metrics.path_efficiency - (10.0 / 30.0)) < 0.001

    def test_path_efficiency_no_movement(self):
        """Test path efficiency when agent doesn't move."""
        positions = [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]
        metrics = calculate_chemotaxis_metrics(
            positions=positions,
            food_positions=[(10.0, 10.0)],
        )

        # No movement should return 1.0
        assert metrics.path_efficiency == 1.0


class TestGetValidationLevel:
    """Test validation level determination."""

    def test_excellent_level(self):
        """Test excellent validation level at threshold."""
        assert get_validation_level(CI_THRESHOLD_EXCELLENT) == ValidationLevel.EXCELLENT
        assert get_validation_level(0.9) == ValidationLevel.EXCELLENT

    def test_target_level(self):
        """Test target validation level."""
        assert get_validation_level(CI_THRESHOLD_TARGET) == ValidationLevel.TARGET
        assert get_validation_level(0.7) == ValidationLevel.TARGET

    def test_minimum_level(self):
        """Test minimum validation level."""
        assert get_validation_level(CI_THRESHOLD_MINIMUM) == ValidationLevel.MINIMUM
        assert get_validation_level(0.5) == ValidationLevel.MINIMUM

    def test_none_level(self):
        """Test none validation level for low CI."""
        assert get_validation_level(0.0) == ValidationLevel.NONE
        assert get_validation_level(0.39) == ValidationLevel.NONE
        assert get_validation_level(-0.5) == ValidationLevel.NONE

    def test_boundary_values(self):
        """Test boundary conditions between levels."""
        # Just below excellent threshold
        just_below_excellent = CI_THRESHOLD_EXCELLENT - 0.001
        assert get_validation_level(just_below_excellent) == ValidationLevel.TARGET

        # Just below target threshold
        just_below_target = CI_THRESHOLD_TARGET - 0.001
        assert get_validation_level(just_below_target) == ValidationLevel.MINIMUM

        # Just below minimum threshold
        just_below_minimum = CI_THRESHOLD_MINIMUM - 0.001
        assert get_validation_level(just_below_minimum) == ValidationLevel.NONE


class TestChemotaxisMetricsDataclass:
    """Test ChemotaxisMetrics dataclass behavior."""

    def test_metrics_creation(self):
        """Test creating ChemotaxisMetrics instance."""
        metrics = ChemotaxisMetrics(
            chemotaxis_index=0.65,
            time_in_attractant=0.7,
            approach_frequency=0.8,
            path_efficiency=0.5,
            total_steps=100,
            steps_in_attractant=70,
            steps_in_control=30,
        )

        assert metrics.chemotaxis_index == 0.65
        assert metrics.time_in_attractant == 0.7
        assert metrics.approach_frequency == 0.8
        assert metrics.path_efficiency == 0.5
        assert metrics.total_steps == 100
        assert metrics.steps_in_attractant == 70
        assert metrics.steps_in_control == 30
        assert metrics.reliable is True  # default

    def test_metrics_unreliable_flag(self):
        """Test setting reliable flag to False."""
        metrics = ChemotaxisMetrics(
            chemotaxis_index=0.0,
            time_in_attractant=0.0,
            approach_frequency=0.0,
            path_efficiency=1.0,
            total_steps=0,
            steps_in_attractant=0,
            steps_in_control=0,
            reliable=False,
        )

        assert metrics.reliable is False


class TestValidationResultDataclass:
    """Test ValidationResult dataclass behavior."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult instance."""
        metrics = ChemotaxisMetrics(
            chemotaxis_index=0.65,
            time_in_attractant=0.7,
            approach_frequency=0.8,
            path_efficiency=0.5,
            total_steps=100,
            steps_in_attractant=70,
            steps_in_control=30,
        )

        result = ValidationResult(
            agent_ci=0.65,
            biological_ci_range=(0.5, 0.85),
            biological_ci_typical=0.7,
            matches_biology=True,
            validation_level=ValidationLevel.TARGET,
            literature_source="Bargmann et al. (1993)",
            agent_metrics=metrics,
        )

        assert result.agent_ci == 0.65
        assert result.biological_ci_range == (0.5, 0.85)
        assert result.biological_ci_typical == 0.7
        assert result.matches_biology is True
        assert result.validation_level == ValidationLevel.TARGET
        assert result.literature_source == "Bargmann et al. (1993)"
        assert result.agent_metrics == metrics


class TestCalculateChemotaxisIndexStepwise:
    """Test the step-by-step chemotaxis index calculation."""

    def test_empty_positions_returns_zero(self):
        """Test that empty trajectory returns neutral CI."""
        ci, n_attract, n_control = calculate_chemotaxis_index_stepwise(
            positions=[],
            food_history=[[(10.0, 10.0)]],
        )
        assert ci == 0.0
        assert n_attract == 0
        assert n_control == 0

    def test_empty_food_history_returns_zero(self):
        """Test that empty food history returns neutral CI."""
        ci, n_attract, n_control = calculate_chemotaxis_index_stepwise(
            positions=[(0.0, 0.0), (1.0, 1.0)],
            food_history=[],
        )
        assert ci == 0.0
        assert n_attract == 0
        assert n_control == 0

    def test_static_food_matches_regular_calculation(self):
        """Test that stepwise with static food matches regular calculation."""
        positions = [(10.0, 10.0), (11.0, 11.0), (50.0, 50.0), (51.0, 51.0)]
        food_positions = [(10.0, 10.0)]
        # Same food at each step
        food_history = [food_positions, food_positions, food_positions, food_positions]

        ci_stepwise, n_attract_stepwise, n_control_stepwise = calculate_chemotaxis_index_stepwise(
            positions,
            food_history,
            attractant_zone_radius=5.0,
        )
        ci_regular, n_attract_regular, n_control_regular = calculate_chemotaxis_index(
            positions,
            food_positions,
            attractant_zone_radius=5.0,
        )

        assert ci_stepwise == ci_regular
        assert n_attract_stepwise == n_attract_regular
        assert n_control_stepwise == n_control_regular

    def test_dynamic_food_respawn(self):
        """Test that respawning food is handled correctly step-by-step."""
        # Agent at (10, 10), food moves from (10, 10) to (50, 50) after step 1
        positions = [
            (10.0, 10.0),  # step 0: near food at (10, 10)
            (10.0, 10.0),  # step 1: food moved to (50, 50), now far
            (10.0, 10.0),  # step 2: still far from food at (50, 50)
            (50.0, 50.0),  # step 3: now near food at (50, 50)
        ]
        food_history = [
            [(10.0, 10.0)],  # step 0
            [(50.0, 50.0)],  # step 1
            [(50.0, 50.0)],  # step 2
            [(50.0, 50.0)],  # step 3
        ]

        ci, n_attract, n_control = calculate_chemotaxis_index_stepwise(
            positions,
            food_history,
            attractant_zone_radius=5.0,
        )

        # 2 near (steps 0 and 3), 2 far (steps 1 and 2)
        assert n_attract == 2
        assert n_control == 2
        assert ci == 0.0

    def test_step_with_no_food_counts_as_control(self):
        """Test that a step with no food counts as control zone."""
        positions = [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)]
        food_history = [
            [(10.0, 10.0)],  # step 0: food nearby
            [],  # step 1: no food
            [(10.0, 10.0)],  # step 2: food back
        ]

        ci, n_attract, n_control = calculate_chemotaxis_index_stepwise(
            positions,
            food_history,
            attractant_zone_radius=5.0,
        )

        # 2 near (steps 0 and 2), 1 far (step 1 - no food)
        assert n_attract == 2
        assert n_control == 1
        # CI = (2 - 1) / 3 = 0.333...
        assert abs(ci - (1 / 3)) < 0.001

    def test_food_history_shorter_than_path(self):
        """Test that last food positions are used when history is shorter."""
        positions = [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0), (10.0, 10.0)]
        food_history = [
            [(10.0, 10.0)],  # step 0
            [(10.0, 10.0)],  # steps 1, 2, 3 use this
        ]

        ci, n_attract, n_control = calculate_chemotaxis_index_stepwise(
            positions,
            food_history,
            attractant_zone_radius=5.0,
        )

        # All 4 positions near food
        assert n_attract == 4
        assert n_control == 0
        assert ci == 1.0


class TestCalculateChemotaxisMetricsStepwise:
    """Test comprehensive stepwise chemotaxis metrics calculation."""

    def test_empty_trajectory_returns_unreliable_metrics(self):
        """Test that empty trajectory returns unreliable metrics."""
        metrics = calculate_chemotaxis_metrics_stepwise(
            positions=[],
            food_history=[[(10.0, 10.0)]],
        )

        assert metrics.chemotaxis_index == 0.0
        assert metrics.time_in_attractant == 0.0
        assert metrics.approach_frequency == 0.0
        assert metrics.path_efficiency == 1.0
        assert metrics.total_steps == 0
        assert metrics.reliable is False

    def test_empty_food_history_returns_unreliable_metrics(self):
        """Test that empty food history returns unreliable metrics."""
        metrics = calculate_chemotaxis_metrics_stepwise(
            positions=[(0.0, 0.0), (1.0, 1.0)],
            food_history=[],
        )

        assert metrics.chemotaxis_index == 0.0
        assert metrics.reliable is False

    def test_static_food_returns_correct_metrics(self):
        """Test metrics with static food positions."""
        positions = [(float(i), 0.0) for i in range(15)]  # 15 steps toward (20, 0)
        food_positions = [(20.0, 0.0)]
        food_history = [food_positions] * 15

        metrics = calculate_chemotaxis_metrics_stepwise(
            positions=positions,
            food_history=food_history,
            attractant_zone_radius=5.0,
            minimum_reliable_steps=10,
        )

        assert metrics.total_steps == 15
        assert metrics.reliable is True
        assert metrics.approach_frequency == 1.0  # Always approaching
        # Some steps should be in attractant zone (positions 15-20 are within 5 of 20)

    def test_dynamic_food_ci_differs_from_static(self):
        """Test that dynamic food gives different CI than using all positions."""
        # Agent stays at (0, 0) while food moves around
        positions = [(0.0, 0.0)] * 10

        # Food is near for first 3 steps, far for rest
        food_history = [
            [(2.0, 0.0)],  # near
            [(2.0, 0.0)],  # near
            [(2.0, 0.0)],  # near
            [(50.0, 50.0)],  # far
            [(50.0, 50.0)],
            [(50.0, 50.0)],
            [(50.0, 50.0)],
            [(50.0, 50.0)],
            [(50.0, 50.0)],
            [(50.0, 50.0)],
        ]

        metrics = calculate_chemotaxis_metrics_stepwise(
            positions=positions,
            food_history=food_history,
            attractant_zone_radius=5.0,
        )

        # 3 near, 7 far => CI = (3-7)/10 = -0.4
        assert metrics.steps_in_attractant == 3
        assert metrics.steps_in_control == 7
        assert abs(metrics.chemotaxis_index - (-0.4)) < 0.001

    def test_time_in_attractant_with_dynamic_food(self):
        """Test time in attractant calculation with moving food."""
        positions = [(10.0, 10.0)] * 4
        food_history = [
            [(10.0, 10.0)],  # near
            [(50.0, 50.0)],  # far
            [(10.0, 10.0)],  # near
            [(50.0, 50.0)],  # far
        ]

        metrics = calculate_chemotaxis_metrics_stepwise(
            positions=positions,
            food_history=food_history,
            attractant_zone_radius=5.0,
        )

        assert metrics.time_in_attractant == 0.5  # 2/4
