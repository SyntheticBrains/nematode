"""Tests for the step processing system."""

from unittest.mock import MagicMock, Mock

import pytest
from quantumnematode.agent import StepResult
from quantumnematode.brain.actions import Action, ActionData
from quantumnematode.brain.arch import BrainParams
from quantumnematode.env import Direction
from quantumnematode.food_handler import FoodConsumptionHandler
from quantumnematode.reward_calculator import RewardCalculator
from quantumnematode.satiety import SatietyManager
from quantumnematode.step_processor import StepProcessor


class TestStepProcessorInitialization:
    """Test step processor initialization."""

    def test_initialize_with_dependencies(self):
        """Test that step processor initializes with all dependencies."""
        brain = Mock()
        env = Mock()
        reward_calculator = Mock(spec=RewardCalculator)
        food_handler = Mock(spec=FoodConsumptionHandler)
        satiety_manager = Mock(spec=SatietyManager)

        processor = StepProcessor(
            brain=brain,
            env=env,
            reward_calculator=reward_calculator,
            food_handler=food_handler,
            satiety_manager=satiety_manager,
        )

        assert processor.brain is brain
        assert processor.env is env
        assert processor.reward_calculator is reward_calculator
        assert processor.food_handler is food_handler
        assert processor.satiety_manager is satiety_manager


class TestPrepareBrainParams:
    """Test brain parameter preparation."""

    def test_prepare_params_without_previous_action(self):
        """Test preparing brain params when no previous action exists."""
        brain = Mock()
        env = Mock()
        env.agent_pos = [2, 3, 0]
        env.current_direction = Direction.UP
        reward_calculator = Mock(spec=RewardCalculator)
        food_handler = Mock(spec=FoodConsumptionHandler)
        satiety_manager = Mock(spec=SatietyManager)

        processor = StepProcessor(brain, env, reward_calculator, food_handler, satiety_manager)

        params = processor.prepare_brain_params(
            gradient_strength=0.5,
            gradient_direction=1.57,
            previous_action=None,
        )

        assert isinstance(params, BrainParams)
        assert params.gradient_strength == 0.5
        assert params.gradient_direction == pytest.approx(1.57)
        assert params.agent_position == (2.0, 3.0)
        assert params.agent_direction == Direction.UP
        assert params.action is None

    def test_prepare_params_with_previous_action(self):
        """Test preparing brain params with a previous action."""
        brain = Mock()
        env = Mock()
        env.agent_pos = [1, 2]
        env.current_direction = Direction.DOWN
        reward_calculator = Mock(spec=RewardCalculator)
        food_handler = Mock(spec=FoodConsumptionHandler)
        satiety_manager = Mock(spec=SatietyManager)

        processor = StepProcessor(brain, env, reward_calculator, food_handler, satiety_manager)

        params = processor.prepare_brain_params(
            gradient_strength=0.8,
            gradient_direction=3.14,
            previous_action=Action.FORWARD,
        )

        assert params.action is not None
        assert params.action.action == Action.FORWARD
        assert params.action.state == "forward"
        assert params.action.probability == 1.0

    def test_prepare_params_handles_3d_position(self):
        """Test that 3D positions are correctly reduced to 2D."""
        brain = Mock()
        env = Mock()
        env.agent_pos = [4, 5, 6]  # 3D position
        env.current_direction = Direction.RIGHT
        reward_calculator = Mock(spec=RewardCalculator)
        food_handler = Mock(spec=FoodConsumptionHandler)
        satiety_manager = Mock(spec=SatietyManager)

        processor = StepProcessor(brain, env, reward_calculator, food_handler, satiety_manager)

        params = processor.prepare_brain_params(
            gradient_strength=0.3,
            gradient_direction=0.0,
            previous_action=None,
        )

        assert params.agent_position == (4.0, 5.0)


class TestProcessStep:
    """Test single step processing."""

    def test_process_step_normal_flow(self):
        """Test normal step processing flow."""
        # Setup mocks
        brain = Mock()
        env = Mock()
        env.agent_pos = [1, 1]
        env.current_direction = Direction.UP
        env.reached_goal = Mock(return_value=False)

        reward_calculator = Mock(spec=RewardCalculator)
        reward_calculator.calculate_reward = Mock(return_value=0.1)

        food_handler = Mock(spec=FoodConsumptionHandler)
        food_handler.track_step = Mock()
        food_handler.check_and_consume_food = Mock(
            return_value=MagicMock(
                food_consumed=False,
                satiety_restored=0.0,
                reward=0.0,
                distance_efficiency=None,
            ),
        )

        satiety_manager = Mock(spec=SatietyManager)
        satiety_manager.is_starved = Mock(return_value=False)
        satiety_manager.current_satiety = 1.0

        # Mock brain response
        action_data = ActionData(state="forward", action=Action.FORWARD, probability=0.8)
        brain.run_brain = Mock(return_value=[action_data])

        processor = StepProcessor(brain, env, reward_calculator, food_handler, satiety_manager)

        result = processor.process_step(
            gradient_strength=0.5,
            gradient_direction=1.57,
            previous_action=None,
            previous_reward=0.0,
            path=[(0, 0), (1, 1)],
            stuck_position_count=0,
        )

        # Verify result
        assert isinstance(result, StepResult)
        assert result.action == Action.FORWARD
        assert result.reward == pytest.approx(0.1)
        assert result.done is False

        # Verify interactions
        env.move_agent.assert_called_once_with(Action.FORWARD)
        food_handler.track_step.assert_called_once()
        food_handler.check_and_consume_food.assert_called_once()
        reward_calculator.calculate_reward.assert_called_once()

    def test_process_step_with_food_consumption(self):
        """Test step processing when food is consumed."""
        brain = Mock()
        env = Mock()
        env.agent_pos = [2, 2]
        env.current_direction = Direction.DOWN
        env.reached_goal = Mock(return_value=True)

        reward_calculator = Mock(spec=RewardCalculator)
        reward_calculator.calculate_reward = Mock(return_value=0.1)

        food_handler = Mock(spec=FoodConsumptionHandler)
        food_handler.track_step = Mock()
        food_handler.check_and_consume_food = Mock(
            return_value=MagicMock(
                food_consumed=True,
                satiety_restored=0.2,
                reward=1.0,
                distance_efficiency=0.85,
            ),
        )

        satiety_manager = Mock(spec=SatietyManager)
        satiety_manager.is_starved = Mock(return_value=False)
        satiety_manager.current_satiety = 0.8

        action_data = ActionData(state="forward", action=Action.FORWARD, probability=0.9)
        brain.run_brain = Mock(return_value=[action_data])

        processor = StepProcessor(brain, env, reward_calculator, food_handler, satiety_manager)

        result = processor.process_step(
            gradient_strength=0.2,
            gradient_direction=0.0,
            previous_action=Action.LEFT,
            previous_reward=0.05,
            path=[(1, 1), (2, 2)],
        )

        # Reward should include food reward
        assert result.reward == pytest.approx(1.1)  # 0.1 base + 1.0 food
        assert result.info["food_consumed"] is True
        assert result.info["distance_efficiency"] == pytest.approx(0.85)

    def test_process_step_with_starvation(self):
        """Test step processing when agent starves."""
        brain = Mock()
        env = Mock()
        env.agent_pos = [3, 3]
        env.current_direction = Direction.LEFT
        env.reached_goal = Mock(return_value=False)

        reward_calculator = Mock(spec=RewardCalculator)
        reward_calculator.calculate_reward = Mock(return_value=-0.1)

        food_handler = Mock(spec=FoodConsumptionHandler)
        food_handler.track_step = Mock()
        food_handler.check_and_consume_food = Mock(
            return_value=MagicMock(
                food_consumed=False,
                satiety_restored=0.0,
                reward=0.0,
                distance_efficiency=None,
            ),
        )

        satiety_manager = Mock(spec=SatietyManager)
        satiety_manager.is_starved = Mock(return_value=True)
        satiety_manager.current_satiety = 0.0

        action_data = ActionData(state="stay", action=Action.STAY, probability=0.5)
        brain.run_brain = Mock(return_value=[action_data])

        processor = StepProcessor(brain, env, reward_calculator, food_handler, satiety_manager)

        result = processor.process_step(
            gradient_strength=0.9,
            gradient_direction=2.5,
            previous_action=Action.RIGHT,
            previous_reward=-0.05,
            path=[(2, 2), (3, 3)],
        )

        # Episode should be done due to starvation
        assert result.done is True
        assert result.info["starved"] is True

    def test_process_step_with_no_brain_actions(self):
        """Test step processing when brain returns no actions."""
        brain = Mock()
        env = Mock()
        env.agent_pos = [0, 0]
        env.current_direction = Direction.RIGHT
        env.reached_goal = Mock(return_value=False)

        reward_calculator = Mock(spec=RewardCalculator)
        reward_calculator.calculate_reward = Mock(return_value=0.0)

        food_handler = Mock(spec=FoodConsumptionHandler)
        food_handler.track_step = Mock()
        food_handler.check_and_consume_food = Mock(
            return_value=MagicMock(
                food_consumed=False,
                satiety_restored=0.0,
                reward=0.0,
                distance_efficiency=None,
            ),
        )

        satiety_manager = Mock(spec=SatietyManager)
        satiety_manager.is_starved = Mock(return_value=False)

        # Brain returns empty list
        brain.run_brain = Mock(return_value=[])

        processor = StepProcessor(brain, env, reward_calculator, food_handler, satiety_manager)

        result = processor.process_step(
            gradient_strength=0.0,
            gradient_direction=0.0,
            previous_action=None,
            previous_reward=0.0,
            path=[(0, 0)],
        )

        # Should fallback to FORWARD action
        assert result.action == Action.FORWARD
        env.move_agent.assert_called_once_with(Action.FORWARD)

    def test_process_step_with_top_only_flag(self):
        """Test step processing with top_only=True."""
        brain = Mock()
        env = Mock()
        env.agent_pos = [1, 2]
        env.current_direction = Direction.UP
        env.reached_goal = Mock(return_value=False)

        reward_calculator = Mock(spec=RewardCalculator)
        reward_calculator.calculate_reward = Mock(return_value=0.05)

        food_handler = Mock(spec=FoodConsumptionHandler)
        food_handler.track_step = Mock()
        food_handler.check_and_consume_food = Mock(
            return_value=MagicMock(
                food_consumed=False,
                satiety_restored=0.0,
                reward=0.0,
                distance_efficiency=None,
            ),
        )

        satiety_manager = Mock(spec=SatietyManager)
        satiety_manager.is_starved = Mock(return_value=False)

        action_data = ActionData(state="left", action=Action.LEFT, probability=0.6)
        brain.run_brain = Mock(return_value=[action_data])

        processor = StepProcessor(brain, env, reward_calculator, food_handler, satiety_manager)

        _ = processor.process_step(
            gradient_strength=0.4,
            gradient_direction=1.0,
            previous_action=Action.FORWARD,
            previous_reward=0.02,
            path=[(0, 0), (1, 2)],
            top_only=True,
            top_randomize=False,
        )

        # Verify brain was called with top_only=True
        brain.run_brain.assert_called_once()
        call_kwargs = brain.run_brain.call_args.kwargs
        assert call_kwargs["top_only"] is True
        assert call_kwargs["top_randomize"] is False

    def test_process_step_with_quantum_brain(self):
        """Test step processing with a quantum brain (input_data should be set)."""
        from quantumnematode.brain.arch import QuantumBrain

        brain = Mock(spec=QuantumBrain)
        brain.num_qubits = 4
        env = Mock()
        env.agent_pos = [1, 1]
        env.current_direction = Direction.DOWN
        env.reached_goal = Mock(return_value=False)

        reward_calculator = Mock(spec=RewardCalculator)
        reward_calculator.calculate_reward = Mock(return_value=0.15)

        food_handler = Mock(spec=FoodConsumptionHandler)
        food_handler.track_step = Mock()
        food_handler.check_and_consume_food = Mock(
            return_value=MagicMock(
                food_consumed=False,
                satiety_restored=0.0,
                reward=0.0,
                distance_efficiency=None,
            ),
        )

        satiety_manager = Mock(spec=SatietyManager)
        satiety_manager.is_starved = Mock(return_value=False)

        action_data = ActionData(state="right", action=Action.RIGHT, probability=0.7)
        brain.run_brain = Mock(return_value=[action_data])

        processor = StepProcessor(brain, env, reward_calculator, food_handler, satiety_manager)

        _ = processor.process_step(
            gradient_strength=0.6,
            gradient_direction=2.0,
            previous_action=None,
            previous_reward=0.0,
            path=[(0, 0), (1, 1)],
        )

        # Verify brain was called with input_data
        brain.run_brain.assert_called_once()
        call_kwargs = brain.run_brain.call_args.kwargs
        assert call_kwargs["input_data"] == [0.6, 0.6, 0.6, 0.6]  # 4 qubits

    def test_process_step_tracks_stuck_position(self):
        """Test that stuck position count is passed to reward calculator."""
        brain = Mock()
        env = Mock()
        env.agent_pos = [2, 2]
        env.current_direction = Direction.UP
        env.reached_goal = Mock(return_value=False)

        reward_calculator = Mock(spec=RewardCalculator)
        reward_calculator.calculate_reward = Mock(return_value=-0.5)

        food_handler = Mock(spec=FoodConsumptionHandler)
        food_handler.track_step = Mock()
        food_handler.check_and_consume_food = Mock(
            return_value=MagicMock(
                food_consumed=False,
                satiety_restored=0.0,
                reward=0.0,
                distance_efficiency=None,
            ),
        )

        satiety_manager = Mock(spec=SatietyManager)
        satiety_manager.is_starved = Mock(return_value=False)

        action_data = ActionData(state="stay", action=Action.STAY, probability=0.9)
        brain.run_brain = Mock(return_value=[action_data])

        processor = StepProcessor(brain, env, reward_calculator, food_handler, satiety_manager)

        _ = processor.process_step(
            gradient_strength=0.3,
            gradient_direction=1.5,
            previous_action=Action.STAY,
            previous_reward=-0.1,
            path=[(2, 2), (2, 2), (2, 2)],
            stuck_position_count=5,
        )

        # Verify reward calculator received stuck_position_count
        reward_calculator.calculate_reward.assert_called_once()
        call_kwargs = reward_calculator.calculate_reward.call_args.kwargs
        assert call_kwargs["stuck_position_count"] == 5
