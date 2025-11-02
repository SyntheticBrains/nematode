"""Tests for episode runners."""

from unittest.mock import Mock

import pytest
from quantumnematode.agent import EpisodeResult, StepResult
from quantumnematode.brain.actions import Action
from quantumnematode.env import DynamicForagingEnvironment, MazeEnvironment
from quantumnematode.metrics import MetricsTracker
from quantumnematode.rendering import EpisodeRenderer
from quantumnematode.report.dtypes import PerformanceMetrics
from quantumnematode.runners import StandardEpisodeRunner
from quantumnematode.step_processor import StepProcessor


class TestStandardEpisodeRunnerInitialization:
    """Test standard episode runner initialization."""

    def test_initialize_with_dependencies(self):
        """Test that runner initializes with all dependencies."""
        step_processor = Mock(spec=StepProcessor)
        metrics_tracker = Mock(spec=MetricsTracker)
        renderer = Mock(spec=EpisodeRenderer)

        runner = StandardEpisodeRunner(
            step_processor=step_processor,
            metrics_tracker=metrics_tracker,
            renderer=renderer,
        )

        assert runner.step_processor is step_processor
        assert runner.metrics_tracker is metrics_tracker
        assert runner.renderer is renderer


class TestStandardEpisodeRunnerExecution:
    """Test standard episode runner execution."""

    def test_run_episode_success_single_goal(self):
        """Test running a successful episode in a single-goal environment."""
        # Setup mocks
        step_processor = Mock(spec=StepProcessor)
        metrics_tracker = Mock(spec=MetricsTracker)
        renderer = Mock(spec=EpisodeRenderer)

        # Create agent mock
        agent = Mock()
        agent.env = Mock(spec=MazeEnvironment)
        agent.env.agent_pos = [1, 1]
        agent.env.get_state = Mock(return_value=(0.5, 1.57))

        # Setup step processor to return success after 3 steps
        step_results = [
            StepResult(
                action=Action.FORWARD,
                reward=0.1,
                done=False,
                info={"goal_reached": False, "food_consumed": False},
            ),
            StepResult(
                action=Action.RIGHT,
                reward=0.2,
                done=False,
                info={"goal_reached": False, "food_consumed": False},
            ),
            StepResult(
                action=Action.FORWARD,
                reward=1.0,
                done=False,
                info={"goal_reached": True, "food_consumed": False},
            ),
        ]
        step_processor.process_step = Mock(side_effect=step_results)

        # Setup metrics tracker
        metrics_tracker.calculate_metrics = Mock(
            return_value=PerformanceMetrics(
                success_rate=1.0,
                average_steps=3.0,
                average_reward=1.3,
                foraging_efficiency=0.0,
            ),
        )

        # Run episode
        runner = StandardEpisodeRunner(step_processor, metrics_tracker, renderer)
        reward_config = Mock()
        result = runner.run(agent, reward_config, max_steps=10)

        # Verify result
        assert isinstance(result, EpisodeResult)
        assert result.success is True
        assert result.steps_taken == 3
        assert result.total_reward == pytest.approx(1.3)
        assert len(result.path) == 4  # Initial position + 3 steps

        # Verify interactions
        assert step_processor.process_step.call_count == 3
        metrics_tracker.track_episode_completion.assert_called_once_with(
            success=True,
            steps=3,
            total_reward=pytest.approx(1.3),
        )

    def test_run_episode_max_steps_reached(self):
        """Test episode termination when max steps reached."""
        step_processor = Mock(spec=StepProcessor)
        metrics_tracker = Mock(spec=MetricsTracker)
        renderer = Mock(spec=EpisodeRenderer)

        agent = Mock()
        agent.env = Mock(spec=MazeEnvironment)
        agent.env.agent_pos = [1, 1]
        agent.env.get_state = Mock(return_value=(0.3, 0.5))

        # All steps fail to reach goal
        step_result = StepResult(
            action=Action.STAY,
            reward=-0.01,
            done=False,
            info={"goal_reached": False, "food_consumed": False},
        )
        step_processor.process_step = Mock(return_value=step_result)

        metrics_tracker.calculate_metrics = Mock(
            return_value=PerformanceMetrics(
                success_rate=0.0,
                average_steps=5.0,
                average_reward=-0.05,
                foraging_efficiency=0.0,
            ),
        )

        runner = StandardEpisodeRunner(step_processor, metrics_tracker, renderer)
        reward_config = Mock()
        result = runner.run(agent, reward_config, max_steps=5)

        # Verify episode ran for max_steps
        assert result.success is False
        assert result.steps_taken == 5
        assert step_processor.process_step.call_count == 5

    def test_run_episode_with_starvation(self):
        """Test episode termination due to starvation."""
        step_processor = Mock(spec=StepProcessor)
        metrics_tracker = Mock(spec=MetricsTracker)
        renderer = Mock(spec=EpisodeRenderer)

        agent = Mock()
        agent.env = Mock(spec=DynamicForagingEnvironment)
        agent.env.agent_pos = [2, 2]
        agent.env.get_state = Mock(return_value=(0.8, 2.0))

        # Agent starves on step 3
        step_results = [
            StepResult(
                action=Action.FORWARD,
                reward=0.05,
                done=False,
                info={"goal_reached": False, "food_consumed": False},
            ),
            StepResult(
                action=Action.LEFT,
                reward=0.03,
                done=False,
                info={"goal_reached": False, "food_consumed": False},
            ),
            StepResult(
                action=Action.STAY,
                reward=-10.0,
                done=True,  # Starvation
                info={"goal_reached": False, "food_consumed": False, "starved": True},
            ),
        ]
        step_processor.process_step = Mock(side_effect=step_results)

        metrics_tracker.calculate_metrics = Mock(
            return_value=PerformanceMetrics(
                success_rate=0.0,
                average_steps=3.0,
                average_reward=-9.92,
                foraging_efficiency=0.0,
            ),
        )

        runner = StandardEpisodeRunner(step_processor, metrics_tracker, renderer)
        reward_config = Mock()
        result = runner.run(agent, reward_config, max_steps=10)

        # Verify early termination
        assert result.success is False
        assert result.steps_taken == 3
        assert step_processor.process_step.call_count == 3

    def test_run_episode_with_food_collection(self):
        """Test episode with food collection in foraging environment."""
        step_processor = Mock(spec=StepProcessor)
        metrics_tracker = Mock(spec=MetricsTracker)
        renderer = Mock(spec=EpisodeRenderer)

        agent = Mock()
        agent.env = Mock(spec=DynamicForagingEnvironment)
        agent.env.agent_pos = [3, 3]
        agent.env.get_state = Mock(return_value=(0.6, 1.0))

        # Agent collects food on steps 2 and 4, then continues until max_steps
        step_results = [
            StepResult(
                action=Action.FORWARD,
                reward=0.1,
                done=False,
                info={"goal_reached": False, "food_consumed": False},
            ),
            StepResult(
                action=Action.FORWARD,
                reward=1.0,
                done=False,
                info={
                    "goal_reached": True,
                    "food_consumed": True,
                    "distance_efficiency": 0.85,
                },
            ),
            StepResult(
                action=Action.RIGHT,
                reward=0.15,
                done=False,
                info={"goal_reached": False, "food_consumed": False},
            ),
            StepResult(
                action=Action.FORWARD,
                reward=1.0,
                done=False,
                info={
                    "goal_reached": True,
                    "food_consumed": True,
                    "distance_efficiency": 0.75,
                },
            ),
        ]
        # Add default result for remaining steps
        default_result = StepResult(
            action=Action.FORWARD,
            reward=0.05,
            done=False,
            info={"goal_reached": False, "food_consumed": False},
        )
        step_results.extend([default_result] * 6)  # Fill to 10 steps

        step_processor.process_step = Mock(side_effect=step_results)

        metrics_tracker.calculate_metrics = Mock(
            return_value=PerformanceMetrics(
                success_rate=1.0,
                average_steps=4.0,
                average_reward=2.25,
                foraging_efficiency=2.0,  # 2 foods collected
            ),
        )

        runner = StandardEpisodeRunner(step_processor, metrics_tracker, renderer)
        reward_config = Mock()
        _ = runner.run(agent, reward_config, max_steps=10)

        # Verify food collection was tracked
        assert metrics_tracker.track_food_collection.call_count == 2
        metrics_tracker.track_food_collection.assert_any_call(distance_efficiency=0.85)
        metrics_tracker.track_food_collection.assert_any_call(distance_efficiency=0.75)

    def test_run_episode_tracks_stuck_position(self):
        """Test that stuck position count is tracked correctly."""
        step_processor = Mock(spec=StepProcessor)
        metrics_tracker = Mock(spec=MetricsTracker)
        renderer = Mock(spec=EpisodeRenderer)

        agent = Mock()
        agent.env = Mock(spec=MazeEnvironment)

        # Agent stays in same position for 2 steps, then moves
        # The runner accesses agent_pos 4 times per step: initial + 3 checks
        agent.env.agent_pos = [1, 1]  # Just use a fixed position
        agent.env.get_state = Mock(return_value=(0.4, 0.8))

        step_result = StepResult(
            action=Action.STAY,
            reward=-0.05,
            done=False,
            info={"goal_reached": False, "food_consumed": False},
        )
        step_processor.process_step = Mock(return_value=step_result)

        metrics_tracker.calculate_metrics = Mock(
            return_value=PerformanceMetrics(
                success_rate=0.0,
                average_steps=3.0,
                average_reward=-0.15,
                foraging_efficiency=0.0,
            ),
        )

        runner = StandardEpisodeRunner(step_processor, metrics_tracker, renderer)
        reward_config = Mock()
        _ = runner.run(agent, reward_config, max_steps=3)

        # Verify stuck_position_count was passed correctly
        calls = step_processor.process_step.call_args_list
        assert calls[0].kwargs["stuck_position_count"] == 0  # First step
        assert calls[1].kwargs["stuck_position_count"] == 1  # Stuck for 1 step
        assert calls[2].kwargs["stuck_position_count"] == 2  # Stuck for 2 steps

    def test_run_episode_with_rendering(self):
        """Test that rendering is called during episode."""
        step_processor = Mock(spec=StepProcessor)
        metrics_tracker = Mock(spec=MetricsTracker)
        renderer = Mock(spec=EpisodeRenderer)

        agent = Mock()
        agent.env = Mock(spec=MazeEnvironment)
        agent.env.agent_pos = [1, 1]
        agent.env.get_state = Mock(return_value=(0.5, 1.0))

        step_result = StepResult(
            action=Action.FORWARD,
            reward=0.1,
            done=False,
            info={"goal_reached": False, "food_consumed": False},
        )
        step_processor.process_step = Mock(return_value=step_result)

        metrics_tracker.calculate_metrics = Mock(
            return_value=PerformanceMetrics(
                success_rate=0.0,
                average_steps=3.0,
                average_reward=0.3,
                foraging_efficiency=0.0,
            ),
        )

        runner = StandardEpisodeRunner(step_processor, metrics_tracker, renderer)
        reward_config = Mock()
        _ = runner.run(
            agent,
            reward_config,
            max_steps=3,
            render_text="Test Episode",
            show_last_frame_only=True,
        )

        # Verify rendering was called for each step
        assert renderer.render_if_needed.call_count == 3
        # Check that kwargs were passed correctly
        first_call = renderer.render_if_needed.call_args_list[0]
        assert first_call.kwargs["text"] == "Test Episode"
        assert first_call.kwargs["show_last_frame_only"] is True
