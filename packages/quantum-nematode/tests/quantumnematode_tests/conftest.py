import pytest
from quantumnematode.experiment.metadata import RewardMetadata


@pytest.fixture
def reward_metadata():
    """Create reward metadata for testing."""
    return RewardMetadata(
        reward_goal=2.0,
        reward_distance_scale=0.5,
        reward_exploration=0.05,
        penalty_step=0.005,
        penalty_anti_dithering=0.02,
        penalty_stuck_position=0.1,
        stuck_position_threshold=3,
        penalty_starvation=10.0,
        penalty_predator_death=10.0,
        penalty_predator_proximity=0.1,
    )
