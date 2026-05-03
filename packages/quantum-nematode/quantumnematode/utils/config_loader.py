# pyright: reportUnusedFunction=false
"""Load and configure simulation settings from a YAML file."""

import math
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from quantumnematode.agent import (
    ManyworldsModeConfig,
    RewardConfig,
    SatietyConfig,
)
from quantumnematode.brain.arch import (
    CRHBrainConfig,
    CRHQLSTMBrainConfig,
    HybridClassicalBrainConfig,
    HybridQuantumBrainConfig,
    HybridQuantumCortexBrainConfig,
    LSTMPPOBrainConfig,
    MLPDQNBrainConfig,
    MLPPPOBrainConfig,
    MLPReinforceBrainConfig,
    QEFBrainConfig,
    QLIFLSTMBrainConfig,
    QQLearningBrainConfig,
    QRCBrainConfig,
    QRHBrainConfig,
    QRHQLSTMBrainConfig,
    QSNNPPOBrainConfig,
    QSNNReinforceBrainConfig,
    QVarCircuitBrainConfig,
    SpikingReinforceBrainConfig,
)
from quantumnematode.brain.modules import Modules
from quantumnematode.dtypes import OxygenSpot, TemperatureSpot
from quantumnematode.env.env import (
    DEFAULT_BASE_OXYGEN,
    DEFAULT_COMFORT_REWARD,
    DEFAULT_CULTIVATION_TEMPERATURE,
    DEFAULT_DANGER_HP_DAMAGE,
    DEFAULT_DANGER_PENALTY,
    DEFAULT_DISCOMFORT_PENALTY,
    DEFAULT_LETHAL_HP_DAMAGE,
    DEFAULT_OXYGEN_DANGER_HP_DAMAGE,
    DEFAULT_OXYGEN_DANGER_PENALTY,
    DEFAULT_OXYGEN_GRADIENT_STRENGTH,
    DEFAULT_OXYGEN_LETHAL_HP_DAMAGE,
    DEFAULT_TEMPERATURE_GRADIENT_STRENGTH,
    AerotaxisParams,
    ForagingParams,
    HealthParams,
    PheromoneParams,
    PheromoneTypeConfig,
    PredatorParams,
    PredatorType,
    SocialFeedingParams,
    ThermotaxisParams,
)
from quantumnematode.initializers import (
    ManualParameterInitializer,
    RandomPiUniformInitializer,
    RandomSmallUniformInitializer,
    ZeroInitializer,
)
from quantumnematode.logging_config import (
    logger,
)
from quantumnematode.optimizers.gradient_methods import (
    DEFAULT_MAX_GRADIENT_NORM,
    GradientCalculationMethod,
)
from quantumnematode.optimizers.learning_rate import (
    DEFAULT_ADAM_LEARNING_RATE_BETA1,
    DEFAULT_ADAM_LEARNING_RATE_BETA2,
    DEFAULT_ADAM_LEARNING_RATE_EPSILON,
    DEFAULT_CONSTANT_LEARNING_RATE,
    DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_FACTOR,
    DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_RATE,
    DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_TYPE,
    DEFAULT_DYNAMIC_LEARNING_RATE_MAX_STEPS,
    DEFAULT_DYNAMIC_LEARNING_RATE_MIN_LR,
    DEFAULT_DYNAMIC_LEARNING_RATE_POWER,
    DEFAULT_DYNAMIC_LEARNING_RATE_STEP_SIZE,
    DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_ADJUSTMENT_FACTOR,
    DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_MAX,
    DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_MIN,
    AdamLearningRate,
    ConstantLearningRate,
    DecayType,
    DynamicLearningRate,
    LearningRateMethod,
    PerformanceBasedLearningRate,
)

if TYPE_CHECKING:
    from quantumnematode.env import DynamicForagingEnvironment
    from quantumnematode.env.theme import Theme

BrainConfigType = (
    QVarCircuitBrainConfig
    | MLPReinforceBrainConfig
    | MLPPPOBrainConfig
    | MLPDQNBrainConfig
    | QQLearningBrainConfig
    | QRCBrainConfig
    | QRHBrainConfig
    | CRHBrainConfig
    | QEFBrainConfig
    | QSNNPPOBrainConfig
    | QSNNReinforceBrainConfig
    | HybridClassicalBrainConfig
    | HybridQuantumBrainConfig
    | HybridQuantumCortexBrainConfig
    | QLIFLSTMBrainConfig
    | QRHQLSTMBrainConfig
    | CRHQLSTMBrainConfig
    | LSTMPPOBrainConfig
    | SpikingReinforceBrainConfig
)

# Type alias for predator movement patterns
MovementPattern = Literal["stationary", "pursuit"]

# Mapping of brain names to their config classes
BRAIN_CONFIG_MAP: dict[str, type[BrainConfigType]] = {
    "qvarcircuit": QVarCircuitBrainConfig,
    "qqlearning": QQLearningBrainConfig,
    "mlpreinforce": MLPReinforceBrainConfig,
    "mlpppo": MLPPPOBrainConfig,
    "mlpdqn": MLPDQNBrainConfig,
    "spikingreinforce": SpikingReinforceBrainConfig,
    "qrc": QRCBrainConfig,
    "qrh": QRHBrainConfig,
    "qef": QEFBrainConfig,
    "crh": CRHBrainConfig,
    "qsnnppo": QSNNPPOBrainConfig,
    "qsnnreinforce": QSNNReinforceBrainConfig,
    "hybridquantum": HybridQuantumBrainConfig,
    "hybridclassical": HybridClassicalBrainConfig,
    "hybridquantumcortex": HybridQuantumCortexBrainConfig,
    "qliflstm": QLIFLSTMBrainConfig,
    "qrhqlstm": QRHQLSTMBrainConfig,
    "crhqlstm": CRHQLSTMBrainConfig,
    "lstmppo": LSTMPPOBrainConfig,
}


def _resolve_brain_config[T: BrainConfigType](
    raw_config: BrainConfigType | None,
    config_cls: type[T],
    brain_name: str,
) -> T:
    """Resolve a raw brain config to the expected config class.

    Handles three cases:
    1. None -> raise (brain config is required)
    2. Already correct type -> return as-is
    3. Dict-like object -> extract matching fields and construct

    Parameters
    ----------
    raw_config
        The raw configuration from YAML parsing.
    config_cls
        The expected Pydantic config class.
    brain_name
        The brain type name (for error messages).

    Returns
    -------
    T
        The resolved configuration instance.

    Raises
    ------
    ValueError
        If the raw_config cannot be converted to the expected type.
    """
    if raw_config is None:
        msg = f"Brain config is required for '{brain_name}' — no default config available."
        raise ValueError(msg)
    if isinstance(raw_config, config_cls):
        return raw_config
    if hasattr(raw_config, "__dict__"):
        # Extract fields that exist in both raw_config and target config class
        config_dict = {
            field_name: getattr(raw_config, field_name)
            for field_name in config_cls.model_fields
            if hasattr(raw_config, field_name)
        }
        # Warn about any fields in raw_config that are not in the target config class
        # Prefer model_fields for Pydantic models, fall back to __dict__ for generic objects
        if hasattr(raw_config, "model_fields"):
            raw_fields = set(raw_config.model_fields.keys())
        else:
            raw_fields = set(raw_config.__dict__.keys())
        expected_fields = set(config_cls.model_fields.keys())
        dropped_fields = raw_fields - expected_fields
        if dropped_fields:
            logger.warning(
                f"Configuration for '{brain_name}' brain: ignoring unrecognized fields "
                f"{sorted(dropped_fields)}. Expected fields: {sorted(expected_fields)}. "
                f"This may indicate a misconfigured brain type.",
            )
        return config_cls(**config_dict)
    error_message = (
        f"Invalid brain configuration for '{brain_name}' brain type. "
        f"Expected {config_cls.__name__}, got {type(raw_config)}."
    )
    logger.error(error_message)
    raise ValueError(error_message)


class BrainContainerConfig(BaseModel):
    """Configuration for the brain architecture."""

    name: str
    config: BrainConfigType


class LearningRateParameters(BaseModel):
    """Parameters for configuring the learning rate."""

    # For constant learning rate
    learning_rate: float = DEFAULT_CONSTANT_LEARNING_RATE
    # For dynamic/adam/performance-based learning rates
    initial_learning_rate: float = 0.1
    decay_rate: float = DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_RATE
    decay_type: str = DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_TYPE.value
    decay_factor: float = DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_FACTOR
    step_size: int = DEFAULT_DYNAMIC_LEARNING_RATE_STEP_SIZE
    max_steps: int = DEFAULT_DYNAMIC_LEARNING_RATE_MAX_STEPS
    power: float = DEFAULT_DYNAMIC_LEARNING_RATE_POWER
    min_lr: float = DEFAULT_DYNAMIC_LEARNING_RATE_MIN_LR
    beta1: float = DEFAULT_ADAM_LEARNING_RATE_BETA1
    beta2: float = DEFAULT_ADAM_LEARNING_RATE_BETA2
    epsilon: float = DEFAULT_ADAM_LEARNING_RATE_EPSILON
    min_learning_rate: float = DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_MIN
    max_learning_rate: float = DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_MAX
    adjustment_factor: float = DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_ADJUSTMENT_FACTOR


class LearningRateConfig(BaseModel):
    """Configuration for the learning rate method and its parameters."""

    method: LearningRateMethod = LearningRateMethod.DYNAMIC
    parameters: LearningRateParameters = LearningRateParameters()


class GradientConfig(BaseModel):
    """Configuration for the gradient calculation method."""

    method: GradientCalculationMethod = GradientCalculationMethod.RAW
    max_norm: float | None = None  # For norm_clip method


class ParameterInitializerConfig(BaseModel):
    """Configuration for parameter initialization."""

    type: str = "random_small"
    manual_parameter_values: dict[str, float] | None = None


class ForagingConfig(BaseModel):
    """Configuration for foraging mechanics in dynamic environment."""

    foods_on_grid: int = 10
    target_foods_to_collect: int = 15
    min_food_distance: int = 5
    agent_exclusion_radius: int = 10
    gradient_decay_constant: float = 10.0
    gradient_strength: float = 1.0
    safe_zone_food_bias: float = 0.0
    food_hotspots: list[list[float]] | None = None
    food_hotspot_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    food_hotspot_decay: float = Field(default=8.0, gt=0.0)
    no_respawn: bool = False
    satiety_food_threshold: float | None = Field(default=None, gt=0.0, le=1.0)

    def to_params(self) -> ForagingParams:
        """Convert to ForagingParams for environment initialization."""
        from quantumnematode.dtypes import FoodHotspot

        food_hotspot_tuples: list[FoodHotspot] | None = None
        if self.food_hotspots is not None:
            food_hotspot_tuples = [
                _validate_and_convert_spot(spot, "food_hotspot", i)
                for i, spot in enumerate(self.food_hotspots)
            ]

        return ForagingParams(
            foods_on_grid=self.foods_on_grid,
            target_foods_to_collect=self.target_foods_to_collect,
            min_food_distance=self.min_food_distance,
            agent_exclusion_radius=self.agent_exclusion_radius,
            gradient_decay_constant=self.gradient_decay_constant,
            gradient_strength=self.gradient_strength,
            safe_zone_food_bias=self.safe_zone_food_bias,
            food_hotspots=food_hotspot_tuples,
            food_hotspot_bias=self.food_hotspot_bias,
            food_hotspot_decay=self.food_hotspot_decay,
            no_respawn=self.no_respawn,
            satiety_food_threshold=self.satiety_food_threshold,
        )


class PredatorConfig(BaseModel):
    """Configuration for predator mechanics in dynamic environment.

    Attributes
    ----------
    enabled : bool
        Whether predators are active in the environment.
    count : int
        Number of predators to spawn.
    speed : float
        Movement speed relative to agent.
    movement_pattern : MovementPattern
        Movement behavior: "stationary" or "pursuit".
    detection_radius : int
        Distance at which pursuit predators detect the agent.
    damage_radius : int
        Distance at which predators deal damage.
        Stationary predators typically have larger damage_radius (toxic zones).
    gradient_decay_constant : float
        Controls how quickly predator gradient signal decays with distance.
    gradient_strength : float
        Multiplier for predator gradient signal strength.
    """

    enabled: bool = False
    count: int = 2  # Maps to DynamicForagingEnvironment.num_predators
    speed: float = 1.0  # Maps to DynamicForagingEnvironment.predator_speed
    movement_pattern: MovementPattern = "pursuit"
    # Maps to DynamicForagingEnvironment.predator_detection_radius
    detection_radius: int = 8
    damage_radius: int = 0  # Distance for damage application
    # Maps to DynamicForagingEnvironment.predator_gradient_decay
    gradient_decay_constant: float = 12.0
    # Maps to DynamicForagingEnvironment.predator_gradient_strength
    gradient_strength: float = 1.0

    def to_params(self) -> PredatorParams:
        """Convert to PredatorParams for environment initialization."""
        # Map movement_pattern string to PredatorType enum
        pattern_to_type = {
            "stationary": PredatorType.STATIONARY,
            "pursuit": PredatorType.PURSUIT,
        }
        predator_type = pattern_to_type[self.movement_pattern]

        return PredatorParams(
            enabled=self.enabled,
            count=self.count,
            predator_type=predator_type,
            speed=self.speed,
            detection_radius=self.detection_radius,
            damage_radius=self.damage_radius,
            gradient_decay_constant=self.gradient_decay_constant,
            gradient_strength=self.gradient_strength,
        )


class HealthConfig(BaseModel):
    """Configuration for HP-based health system.

    Predator contact deals damage based on proximity within damage_radius.
    Food consumption restores both HP and satiety.
    """

    max_hp: float = 100.0
    predator_damage: float = 10.0
    food_healing: float = 5.0

    def to_params(self) -> HealthParams:
        """Convert to HealthParams for environment initialization."""
        return HealthParams(
            max_hp=self.max_hp,
            predator_damage=self.predator_damage,
            food_healing=self.food_healing,
        )


# Temperature spot configuration constants
TEMPERATURE_SPOT_ELEMENTS = 3  # [x, y, intensity]


def _validate_and_convert_spot(
    spot: list[float],
    spot_type: str,
    index: int,
) -> TemperatureSpot:
    """Validate and convert a temperature spot from list to tuple.

    Args:
        spot: List of [x, y, intensity] values.
        spot_type: Type of spot for error messages ("hot_spot" or "cold_spot").
        index: Index of the spot in the list for error messages.

    Returns
    -------
        TemperatureSpot tuple of (x, y, intensity) with x and y as integers.

    Raises
    ------
        ValueError: If spot doesn't have exactly 3 elements.
    """
    if len(spot) != TEMPERATURE_SPOT_ELEMENTS:
        msg = f"Invalid {spot_type} at index {index}: expected [x, y, intensity], got {spot}"
        raise ValueError(msg)
    return (int(spot[0]), int(spot[1]), float(spot[2]))


class ThermotaxisConfig(BaseModel):
    """Configuration for thermotaxis (temperature sensing) system.

    When enabled, the environment has a temperature field that the agent
    can sense. Temperature zones affect rewards and HP damage based on
    deviation from the cultivation temperature.

    Attributes
    ----------
    enabled : bool
        Whether thermotaxis is active in the environment.
    cultivation_temperature : float
        The temperature the agent was raised at (°C). Agent is most comfortable here.
    base_temperature : float
        Base temperature of the environment before gradients/spots (°C).
    gradient_direction : float
        Direction of linear temperature gradient (radians, 0 = increases to right).
    gradient_strength : float
        How quickly temperature changes per cell (°C/cell).
    hot_spots : list[list[float]] | None
        Localized hot spots as [x, y, intensity] lists.
    cold_spots : list[list[float]] | None
        Localized cold spots as [x, y, intensity] lists.
    spot_decay_constant : float
        Decay constant for hot/cold spot exponential falloff.
    comfort_delta : float
        Temperature deviation from cultivation temp considered comfortable (°C).
    discomfort_delta : float
        Temperature deviation threshold for discomfort zone (°C).
    danger_delta : float
        Temperature deviation threshold for danger zone (°C).
    comfort_reward : float
        Reward per step when in comfort zone.
    discomfort_penalty : float
        Penalty per step when in discomfort zone (should be negative).
    danger_penalty : float
        Penalty per step when in danger zone (should be negative).
    danger_hp_damage : float
        HP damage per step when in danger zone.
    lethal_hp_damage : float
        HP damage per step when in lethal zone.
    reward_discomfort_food : float
        Bonus reward for collecting food while in a discomfort zone.
        Encourages "brave foraging" - entering uncomfortable but safe zones for food.
    """

    enabled: bool = False
    cultivation_temperature: float = DEFAULT_CULTIVATION_TEMPERATURE
    base_temperature: float = DEFAULT_CULTIVATION_TEMPERATURE
    gradient_direction: float = 0.0
    gradient_strength: float = DEFAULT_TEMPERATURE_GRADIENT_STRENGTH
    hot_spots: list[list[float]] | None = None
    cold_spots: list[list[float]] | None = None
    spot_decay_constant: float = 5.0
    comfort_delta: float = 5.0
    discomfort_delta: float = 10.0
    danger_delta: float = 15.0
    comfort_reward: float = DEFAULT_COMFORT_REWARD
    discomfort_penalty: float = DEFAULT_DISCOMFORT_PENALTY
    danger_penalty: float = DEFAULT_DANGER_PENALTY
    danger_hp_damage: float = DEFAULT_DANGER_HP_DAMAGE
    lethal_hp_damage: float = DEFAULT_LETHAL_HP_DAMAGE
    reward_discomfort_food: float = 0.0

    def to_params(self) -> ThermotaxisParams:
        """Convert to ThermotaxisParams for environment initialization."""
        # Convert list of lists from YAML to list of TemperatureSpot tuples
        hot_spots_tuples: list[TemperatureSpot] | None = None
        if self.hot_spots is not None:
            hot_spots_tuples = [
                _validate_and_convert_spot(spot, "hot_spot", i)
                for i, spot in enumerate(self.hot_spots)
            ]

        cold_spots_tuples: list[TemperatureSpot] | None = None
        if self.cold_spots is not None:
            cold_spots_tuples = [
                _validate_and_convert_spot(spot, "cold_spot", i)
                for i, spot in enumerate(self.cold_spots)
            ]

        return ThermotaxisParams(
            enabled=self.enabled,
            cultivation_temperature=self.cultivation_temperature,
            base_temperature=self.base_temperature,
            gradient_direction=self.gradient_direction,
            gradient_strength=self.gradient_strength,
            hot_spots=hot_spots_tuples,
            cold_spots=cold_spots_tuples,
            spot_decay_constant=self.spot_decay_constant,
            comfort_delta=self.comfort_delta,
            discomfort_delta=self.discomfort_delta,
            danger_delta=self.danger_delta,
            comfort_reward=self.comfort_reward,
            discomfort_penalty=self.discomfort_penalty,
            danger_penalty=self.danger_penalty,
            danger_hp_damage=self.danger_hp_damage,
            lethal_hp_damage=self.lethal_hp_damage,
            reward_discomfort_food=self.reward_discomfort_food,
        )


class AerotaxisConfig(BaseModel):
    """Configuration for aerotaxis (oxygen sensing) system.

    When enabled, the environment has an oxygen field that the agent can sense.
    Oxygen zones affect rewards and HP damage based on O2 percentage.

    Unlike thermotaxis, oxygen zones use absolute percentage thresholds (not
    symmetric deltas) and have 5 zones (no discomfort tier).
    """

    enabled: bool = False
    base_oxygen: float = DEFAULT_BASE_OXYGEN
    gradient_direction: float = 0.0
    gradient_strength: float = DEFAULT_OXYGEN_GRADIENT_STRENGTH
    high_oxygen_spots: list[list[float]] | None = None
    low_oxygen_spots: list[list[float]] | None = None
    spot_decay_constant: float = 5.0
    comfort_reward: float = 0.0
    danger_penalty: float = DEFAULT_OXYGEN_DANGER_PENALTY
    danger_hp_damage: float = DEFAULT_OXYGEN_DANGER_HP_DAMAGE
    lethal_hp_damage: float = DEFAULT_OXYGEN_LETHAL_HP_DAMAGE
    reward_discomfort_food: float = 0.0
    # Zone thresholds (absolute O2 percentages)
    lethal_hypoxia_upper: float = 2.0
    danger_hypoxia_upper: float = 5.0
    comfort_lower: float = 5.0
    comfort_upper: float = 12.0
    danger_hyperoxia_upper: float = 17.0

    def to_params(self) -> AerotaxisParams:
        """Convert to AerotaxisParams for environment initialization."""
        high_spots: list[OxygenSpot] | None = None
        if self.high_oxygen_spots is not None:
            high_spots = [
                _validate_and_convert_spot(spot, "high_oxygen_spot", i)
                for i, spot in enumerate(self.high_oxygen_spots)
            ]

        low_spots: list[OxygenSpot] | None = None
        if self.low_oxygen_spots is not None:
            low_spots = [
                _validate_and_convert_spot(spot, "low_oxygen_spot", i)
                for i, spot in enumerate(self.low_oxygen_spots)
            ]

        return AerotaxisParams(
            enabled=self.enabled,
            base_oxygen=self.base_oxygen,
            gradient_direction=self.gradient_direction,
            gradient_strength=self.gradient_strength,
            high_oxygen_spots=high_spots,
            low_oxygen_spots=low_spots,
            spot_decay_constant=self.spot_decay_constant,
            comfort_reward=self.comfort_reward,
            danger_penalty=self.danger_penalty,
            danger_hp_damage=self.danger_hp_damage,
            lethal_hp_damage=self.lethal_hp_damage,
            reward_discomfort_food=self.reward_discomfort_food,
            lethal_hypoxia_upper=self.lethal_hypoxia_upper,
            danger_hypoxia_upper=self.danger_hypoxia_upper,
            comfort_lower=self.comfort_lower,
            comfort_upper=self.comfort_upper,
            danger_hyperoxia_upper=self.danger_hyperoxia_upper,
        )


class PheromoneTypeConfigYAML(BaseModel):
    """YAML configuration for a single pheromone type."""

    emission_strength: float = Field(default=1.0, ge=0.0)
    spatial_decay_constant: float = Field(default=8.0, gt=0.0)
    temporal_half_life: float = Field(default=50.0, gt=0.0)
    max_sources: int = Field(default=100, gt=0)


class PheromoneConfig(BaseModel):
    """YAML configuration for pheromone communication.

    When enabled, agents emit chemical signals on events (food consumption,
    predator damage) that diffuse and decay over time. Other agents sense
    pheromone concentrations for biologically grounded communication.
    """

    enabled: bool = False
    food_marking: PheromoneTypeConfigYAML = Field(
        default_factory=PheromoneTypeConfigYAML,
    )
    alarm: PheromoneTypeConfigYAML = Field(
        default_factory=lambda: PheromoneTypeConfigYAML(
            emission_strength=2.0,
            spatial_decay_constant=5.0,
            temporal_half_life=20.0,
            max_sources=50,
        ),
    )
    aggregation: PheromoneTypeConfigYAML | None = None

    def to_params(self) -> PheromoneParams:
        """Convert to PheromoneParams for environment initialization."""
        agg_config = None
        if self.aggregation is not None:
            agg_config = PheromoneTypeConfig(
                emission_strength=self.aggregation.emission_strength,
                spatial_decay_constant=self.aggregation.spatial_decay_constant,
                temporal_half_life=self.aggregation.temporal_half_life,
                max_sources=self.aggregation.max_sources,
            )
        return PheromoneParams(
            enabled=self.enabled,
            food_marking=PheromoneTypeConfig(
                emission_strength=self.food_marking.emission_strength,
                spatial_decay_constant=self.food_marking.spatial_decay_constant,
                temporal_half_life=self.food_marking.temporal_half_life,
                max_sources=self.food_marking.max_sources,
            ),
            alarm=PheromoneTypeConfig(
                emission_strength=self.alarm.emission_strength,
                spatial_decay_constant=self.alarm.spatial_decay_constant,
                temporal_half_life=self.alarm.temporal_half_life,
                max_sources=self.alarm.max_sources,
            ),
            aggregation=agg_config,
        )


class SocialFeedingConfig(BaseModel):
    """YAML configuration for social feeding behavior.

    Models C. elegans npr-1-mediated social feeding: social animals conserve
    energy near conspecifics via reduced locomotion and increased pharyngeal
    pumping on bacterial lawns. Detection radius is shared with the multi-agent
    config's ``social_detection_radius``.
    """

    enabled: bool = False
    decay_reduction: float = Field(
        default=0.7,
        gt=0.0,
        description="Satiety decay multiplier when near conspecifics (< 1.0 = slower decay).",
    )
    solitary_decay: float = Field(
        default=1.0,
        gt=0.0,
        description="Satiety decay multiplier for solitary phenotype (> 1.0 = crowding penalty).",
    )

    def to_params(self) -> SocialFeedingParams:
        """Convert to SocialFeedingParams for environment initialization."""
        return SocialFeedingParams(
            enabled=self.enabled,
            decay_reduction=self.decay_reduction,
            solitary_decay=self.solitary_decay,
        )


class SensingMode(StrEnum):
    """Sensing mode for gradient-based sensory modalities."""

    ORACLE = "oracle"  # Spatial gradients (existing behavior)
    TEMPORAL = "temporal"  # Mode A: raw scalar only, agent uses STAM
    DERIVATIVE = "derivative"  # Mode B: scalar + dC/dt
    KLINOTAXIS = "klinotaxis"  # Mode C: head-sweep lateral gradient + dC/dt


class SensingConfig(BaseModel):
    """Configuration for temporal sensing modes and STAM memory.

    Each gradient-based modality (chemotaxis, thermotaxis, nociception,
    pheromones) can independently use oracle, temporal, derivative,
    or klinotaxis mode.
    """

    chemotaxis_mode: SensingMode = SensingMode.ORACLE
    thermotaxis_mode: SensingMode = SensingMode.ORACLE
    nociception_mode: SensingMode = SensingMode.ORACLE
    aerotaxis_mode: SensingMode = SensingMode.ORACLE
    pheromone_food_mode: SensingMode = SensingMode.ORACLE
    pheromone_alarm_mode: SensingMode = SensingMode.ORACLE
    pheromone_aggregation_mode: SensingMode = SensingMode.ORACLE
    stam_enabled: bool = False
    stam_buffer_size: int = Field(default=30, gt=0)
    stam_decay_rate: float = Field(default=0.1, gt=0.0)
    derivative_scale: float = Field(
        default=50.0,
        gt=0.0,
        description=(
            "Scaling factor applied to temporal derivatives before tanh normalization. "
            "Raw derivatives on small grids (20x20) are ~0.001-0.01, which tanh maps to "
            "near-zero. This scale amplifies the signal: tanh(derivative * scale). "
            "Default 50.0 maps a raw derivative of 0.01 to tanh(0.5) ≈ 0.46."
        ),
    )
    lateral_scale: float = Field(
        default=50.0,
        gt=0.0,
        description=(
            "Scaling factor applied to lateral (head-sweep) gradients before tanh "
            "normalization in klinotaxis mode. Applied as tanh((right - left) * scale). "
            "Default 50.0."
        ),
    )


def _module_suffix_for_mode(mode: SensingMode) -> str:
    """Return the module name suffix for a given sensing mode."""
    if mode == SensingMode.KLINOTAXIS:
        return "_klinotaxis"
    if mode != SensingMode.ORACLE:
        return "_temporal"
    return ""


def apply_sensing_mode(
    sensory_modules: list[str],
    sensing: SensingConfig,
) -> list[str]:
    """Auto-replace oracle sensory module names with temporal or klinotaxis variants.

    Handles:
    - food_chemotaxis → food_chemotaxis_temporal (temporal/derivative) or
      food_chemotaxis_klinotaxis (klinotaxis)
    - Same pattern for nociception, thermotaxis, aerotaxis, pheromones
    - Appends stam module when stam_enabled

    Parameters
    ----------
    sensory_modules : list[str]
        Original sensory module names from brain config.
    sensing : SensingConfig
        Sensing configuration.

    Returns
    -------
    list[str]
        Updated sensory module names with mode substitutions applied.
    """
    # Map oracle module names to their mode config attribute
    mode_map: dict[str, SensingMode] = {
        "food_chemotaxis": sensing.chemotaxis_mode,
        "nociception": sensing.nociception_mode,
        "thermotaxis": sensing.thermotaxis_mode,
        "aerotaxis": sensing.aerotaxis_mode,
        "pheromone_food": sensing.pheromone_food_mode,
        "pheromone_alarm": sensing.pheromone_alarm_mode,
        "pheromone_aggregation": sensing.pheromone_aggregation_mode,
    }

    result = []

    for module in sensory_modules:
        if module in mode_map:
            mode = mode_map[module]
            suffix = _module_suffix_for_mode(mode)
            result.append(f"{module}{suffix}" if suffix else module)
        else:
            result.append(module)

    # Append STAM module when enabled (if not already present)
    if sensing.stam_enabled and "stam" not in result:
        result.append("stam")

    return result


def validate_sensing_config(sensing: SensingConfig) -> SensingConfig:
    """Validate sensing config and auto-enable STAM for derivative mode.

    Parameters
    ----------
    sensing : SensingConfig
        Sensing configuration to validate.

    Returns
    -------
    SensingConfig
        Validated/updated sensing configuration.
    """
    all_modes = (
        sensing.chemotaxis_mode,
        sensing.thermotaxis_mode,
        sensing.nociception_mode,
        sensing.aerotaxis_mode,
        sensing.pheromone_food_mode,
        sensing.pheromone_alarm_mode,
        sensing.pheromone_aggregation_mode,
    )
    any_derivative = any(mode == SensingMode.DERIVATIVE for mode in all_modes)
    any_temporal = any(mode == SensingMode.TEMPORAL for mode in all_modes)
    any_klinotaxis = any(mode == SensingMode.KLINOTAXIS for mode in all_modes)

    # Derivative and klinotaxis modes require STAM — auto-enable if not set
    if (any_derivative or any_klinotaxis) and not sensing.stam_enabled:
        mode_name = "Klinotaxis" if any_klinotaxis else "Derivative"
        logger.info(
            f"{mode_name} sensing mode requires temporal history — "
            "auto-enabling STAM with default parameters "
            f"(buffer_size={sensing.stam_buffer_size}, "
            f"decay_rate={sensing.stam_decay_rate}).",
        )
        sensing = sensing.model_copy(update={"stam_enabled": True})

    # Temporal mode without STAM — warn but allow
    if any_temporal and not sensing.stam_enabled:
        logger.warning(
            "Temporal sensing mode (Mode A) without STAM enabled may result in "
            "very limited sensory information. Consider setting stam_enabled: true.",
        )

    return sensing


class EnvironmentConfig(BaseModel):
    """Configuration for the dynamic foraging environment."""

    grid_size: int = 50
    viewport_size: tuple[int, int] = (11, 11)

    # Nested configuration subsections
    foraging: ForagingConfig | None = None
    predators: PredatorConfig | None = None
    health: HealthConfig | None = None
    thermotaxis: ThermotaxisConfig | None = None
    aerotaxis: AerotaxisConfig | None = None
    pheromones: PheromoneConfig | None = None
    social_feeding: SocialFeedingConfig | None = None
    sensing: SensingConfig | None = None

    def get_foraging_config(self) -> ForagingConfig:
        """Get foraging configuration with defaults."""
        return self.foraging or ForagingConfig()

    def get_predator_config(self) -> PredatorConfig:
        """Get predator configuration with defaults."""
        return self.predators or PredatorConfig()

    def get_health_config(self) -> HealthConfig:
        """Get health configuration with defaults."""
        return self.health or HealthConfig()

    def get_thermotaxis_config(self) -> ThermotaxisConfig:
        """Get thermotaxis configuration with defaults."""
        return self.thermotaxis or ThermotaxisConfig()

    def get_aerotaxis_config(self) -> AerotaxisConfig:
        """Get aerotaxis configuration with defaults."""
        return self.aerotaxis or AerotaxisConfig()

    def get_pheromone_config(self) -> PheromoneConfig:
        """Get pheromone configuration with defaults."""
        return self.pheromones or PheromoneConfig()

    def get_social_feeding_config(self) -> SocialFeedingConfig:
        """Get social feeding configuration with defaults."""
        return self.social_feeding or SocialFeedingConfig()

    def get_sensing_config(self) -> SensingConfig:
        """Get sensing configuration with defaults."""
        return self.sensing or SensingConfig()


class AgentConfig(BaseModel):
    """Configuration for a single agent in multi-agent mode."""

    id: str
    brain: BrainContainerConfig
    weights_path: str | None = None
    social_phenotype: Literal["social", "solitary"] = "social"


class MultiAgentConfig(BaseModel):
    """Configuration for multi-agent simulation.

    Either ``count`` (homogeneous population using top-level brain config) or
    ``agents`` (heterogeneous population with per-agent brain configs) must be
    set when ``enabled=True``.
    """

    enabled: bool = False
    count: int | None = None
    agents: list[AgentConfig] | None = None
    food_competition: Literal["first_arrival", "random"] = "first_arrival"
    social_detection_radius: int = 5
    termination_policy: Literal["freeze", "remove", "end_all"] = "freeze"
    min_agent_distance: int = 5  # Best-effort Poisson disk; may be violated on dense grids

    @model_validator(mode="after")
    def _validate_population(self) -> "MultiAgentConfig":
        if not self.enabled:
            return self
        has_count = self.count is not None
        has_agents = self.agents is not None and len(self.agents) > 0
        if has_count and has_agents:
            msg = "Cannot set both 'count' and 'agents' in multi_agent config."
            raise ValueError(msg)
        if not has_count and not has_agents:
            msg = "Must set either 'count' or 'agents' when multi_agent.enabled=True."
            raise ValueError(msg)
        min_agents = 2
        max_agents = 10
        if has_count and not (min_agents <= self.count <= max_agents):  # type: ignore[operator]
            msg = (
                f"multi_agent.count must be between {min_agents} and "
                f"{max_agents}, got {self.count}."
            )
            raise ValueError(msg)
        if has_agents and not (min_agents <= len(self.agents) <= max_agents):  # type: ignore[arg-type]
            msg = (
                f"multi_agent.agents must contain between {min_agents} and {max_agents} entries, "
                f"got {len(self.agents)}."  # type: ignore[arg-type]
            )
            raise ValueError(msg)
        return self


class EvolutionConfig(BaseModel):
    """Configuration for the evolution loop.

    Optional sub-block of :class:`SimulationConfig`.  When absent (the default
    for non-evolution configs), :class:`SimulationConfig.evolution` is
    ``None`` and the evolution-loop CLI must rely on its own defaults.

    Field defaults are chosen for full-scale campaigns (50 generations,
    population 20).  Pilot configs (``configs/evolution/*_small*.yml``)
    typically override these to ``generations: 10, population_size: 8,
    episodes_per_eval: 3`` for fast smoke-testing.
    """

    algorithm: Literal["cmaes", "ga", "tpe"] = "cmaes"
    population_size: int = Field(default=20, ge=1)
    generations: int = Field(default=50, ge=1)
    episodes_per_eval: int = Field(default=15, ge=1)
    sigma0: float = Field(default=math.pi / 2, gt=0.0)  # CMA-ES initial step size
    elite_fraction: float = Field(default=0.2, ge=0.0, le=1.0)  # GA-only
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0)  # GA-only
    crossover_rate: float = Field(default=0.8, ge=0.0, le=1.0)  # GA-only
    parallel_workers: int = Field(default=1, ge=1)
    checkpoint_every: int = Field(default=10, ge=1)
    # CMA-ES-only: restrict covariance to diagonal (sep-CMA-ES; sets
    # cma's CMA_diagonal=True option).  MUST be enabled for any campaign
    # with a genome dim >~1000 — full-cov CMA-ES `tell()` is O(n²) and
    # becomes minutes per generation at the weight-evolution scale of
    # LSTMPPO / large MLPPPO networks.
    #
    # Trade-off: diagonal mode gives up off-diagonal covariance adaptation,
    # so per-generation convergence is slower (typically 2-10x more
    # generations to reach the same fitness target on non-separable
    # problems; comparable or faster on separable ones — Ros & Hansen
    # 2008).  At n>~1000, full-cov is not actually a competing option (it
    # doesn't fit in memory or finish a generation in finite time), so
    # net wall-clock to convergence is dramatically faster with diagonal
    # despite the slower per-generation convergence.
    #
    # Default False to preserve back-compat for small-genome campaigns
    # — at small n (<20) where the HyperparameterEncoder operates,
    # full-cov is cheap and probably preferable.
    cma_diagonal: bool = False
    # Hyperparameter-evolution train/eval split for
    # LearnedPerformanceFitness.  When learn_episodes_per_eval > 0 and
    # the CLI selects --fitness learned_performance, each genome
    # evaluation runs K = learn_episodes_per_eval training episodes
    # (brain.learn() fires) followed by L = eval_episodes_per_eval
    # frozen eval episodes.  Default learn_episodes_per_eval=0 means
    # LearnedPerformanceFitness.evaluate raises — preserving the
    # default behaviour where EpisodicSuccessRate ignores both fields.
    learn_episodes_per_eval: int = Field(default=0, ge=0)
    # None means "fall back to episodes_per_eval" — preserves the
    # default behaviour for that field.
    eval_episodes_per_eval: int | None = Field(default=None, ge=1)
    # Optional warm-start checkpoint for LearnedPerformanceFitness.  When
    # set, every genome's brain is loaded with the checkpoint's weights
    # AFTER ``encoder.decode`` and BEFORE the K train phase, so the K
    # episodes fine-tune the checkpoint under the genome's evolved
    # hyperparameters rather than training from scratch.  Incompatible
    # with hyperparam_schema entries that change tensor shapes
    # (architecture fields like ``actor_hidden_dim``, ``lstm_hidden_dim``,
    # ``rnn_type``); those are rejected at YAML load time by
    # ``SimulationConfig._validate_hyperparam_schema``.  When None
    # (default), behaviour is unchanged: fresh-init weights from
    # ``encoder.decode``, no load step.
    warm_start_path: Path | None = None
    # Per-genome Lamarckian inheritance: when set to "lamarckian", each
    # child of generation N+1 warm-starts its brain from a *selected*
    # parent of generation N (per the InheritanceStrategy in
    # ``quantumnematode.evolution.inheritance``).  Default "none" leaves
    # the loop, fitness, and lineage code paths byte-equivalent to a
    # frozen-weight evolution run.  Mutually exclusive with
    # warm_start_path; requires hyperparam_schema (rejecting weight
    # encoders, which would double-count weights as both genome and
    # substrate); requires learn_episodes_per_eval > 0 (no train phase
    # = no weights to inherit); and incompatible with
    # architecture-changing schema fields.  All checks enforced by the
    # model validators below + ``SimulationConfig._validate_hyperparam_schema``.
    inheritance: Literal["none", "lamarckian", "baldwin"] = "none"
    # Number of prior-generation elites whose checkpoints survive into
    # the next generation.  Default 1 (single-elite-broadcast).  Only
    # 1 is currently accepted when inheritance is "lamarckian"; the
    # field accepts >=1 in the schema so multi-elite parent-selection
    # strategies (round-robin, tournament, soft-elite top-k) can be
    # added later without a config-schema migration.  Under
    # inheritance: baldwin the field has no runtime effect (Baldwin is
    # single-elite by construction — trait inheritance flows through
    # TPE's posterior, which biases sampling toward the prior elite —
    # and ``BaldwinInheritance`` ignores the field entirely), but the
    # structural ``inheritance_elite_count <= population_size`` check
    # in ``_validate_inheritance`` (rule 4) is still enforced for all
    # inheritance modes to keep the schema invariant uniform.
    inheritance_elite_count: int = Field(default=1, ge=1)
    # Optional early-stop on saturation: when set to a positive integer
    # N, the loop exits if best_fitness has not strictly improved for N
    # consecutive generations.  Default None preserves existing
    # full-budget behaviour byte-equivalently.  CLI override:
    # --early-stop-on-saturation N on scripts/run_evolution.py.
    # Persisted in the checkpoint pickle (CHECKPOINT_VERSION 3) so
    # resume preserves the saturation-tracking state.
    early_stop_on_saturation: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_inheritance(self) -> "EvolutionConfig":
        """Enforce inheritance configuration rules at YAML load time.

        Four rules:

        1. ``inheritance != "none"`` AND ``learn_episodes_per_eval == 0``
           — no train phase means no weights to inherit (Lamarckian) or
           no learned-elite signal to propagate (Baldwin).
        2. ``inheritance != "none"`` AND ``warm_start_path is not None``
           — Lamarckian and warm_start both load weights into the same
           brain slot; Baldwin under static warm-start would mean every
           child starts from the same fixed checkpoint, collapsing the
           Baldwin signal.  Exactly one may be set.
        3. ``inheritance == "lamarckian"`` AND ``inheritance_elite_count
           != 1`` — single-elite-broadcast is the only currently-supported
           parent-selection rule for Lamarckian inheritance.  The rule
           does NOT apply under ``inheritance: baldwin`` since Baldwin
           ignores the field.
        4. ``inheritance_elite_count > population_size`` — trivially
           impossible to keep more elites than there are genomes.
           Independent of rule 3 so the multi-elite restriction can
           be lifted without dropping this structural check.
        """
        if self.inheritance != "none":
            if self.learn_episodes_per_eval == 0:
                msg = (
                    f"evolution.inheritance is {self.inheritance!r} but "
                    "evolution.learn_episodes_per_eval is 0. Inheritance "
                    "requires a non-zero train phase: Lamarckian needs "
                    "weights to inherit; Baldwin's whole premise is that "
                    "lifetime learning shapes the gen-N elite that becomes "
                    "the prior for gen-N+1. Either set "
                    "learn_episodes_per_eval > 0 or set inheritance: none."
                )
                raise ValueError(msg)
            if self.warm_start_path is not None:
                msg = (
                    "evolution.warm_start_path and evolution.inheritance "
                    f"({self.inheritance!r}) are mutually exclusive. "
                    "Under Lamarckian, both load weights into the same "
                    "brain slot before the K train phase. Under Baldwin, "
                    "static warm-start would collapse the Baldwin signal "
                    "(every child starts from the same checkpoint regardless "
                    "of the elite's evolved hyperparameters). Drop one of "
                    "the two."
                )
                raise ValueError(msg)
        if self.inheritance == "lamarckian" and self.inheritance_elite_count != 1:
            msg = (
                f"evolution.inheritance_elite_count={self.inheritance_elite_count} "
                f"but evolution.inheritance is {self.inheritance!r}. "
                "inheritance_elite_count MUST be 1 when inheritance: lamarckian. "
                "Multi-elite parent selection (round-robin or tournament) "
                "is not currently supported; the field exists structurally "
                "so future strategies can populate it without a "
                "config-schema migration."
            )
            raise ValueError(msg)
        if self.inheritance_elite_count > self.population_size:
            msg = (
                f"evolution.inheritance_elite_count ({self.inheritance_elite_count}) "
                f"exceeds evolution.population_size ({self.population_size}). "
                "MUST be <= population_size."
            )
            raise ValueError(msg)
        return self


class ParamSchemaEntry(BaseModel):
    """One slot in a hyperparameter-evolution genome.

    Each entry declares which brain-config field is evolved, its type,
    and the type-appropriate metadata (bounds for float/int, values
    for categorical, log_scale for float).  Validated at YAML load
    time both for type-conditional metadata correctness (this
    validator) and for field-name correctness against the resolved
    brain config (the SimulationConfig validator).
    """

    name: str
    type: Literal["float", "int", "bool", "categorical"]
    # float/int: required (low, high).  bool/categorical: must be None.
    bounds: tuple[float, float] | None = None
    # categorical: required, len >= 2.  Other types: must be None.
    values: list[str] | None = None
    # float only: when True, the genome value is in log-space and
    # decode applies exp(value).  bool/categorical/int: must be False.
    log_scale: bool = False

    # PLR0912/C901: explicit per-type dispatch with distinct error
    # messages is clearer here than abstraction.
    @model_validator(mode="after")
    def _validate_type_conditional_metadata(self) -> "ParamSchemaEntry":  # noqa: PLR0912, C901
        if self.type == "float":
            if self.bounds is None:
                msg = (
                    f"hyperparam_schema entry {self.name!r} of type 'float' requires "
                    "'bounds' to be set as (low, high)."
                )
                raise ValueError(msg)
            if self.values is not None:
                msg = (
                    f"hyperparam_schema entry {self.name!r} of type 'float' must not "
                    "set 'values' (use 'bounds' instead)."
                )
                raise ValueError(msg)
            self._validate_bounds_range()
            self._validate_log_scale_positivity()
        elif self.type == "int":
            if self.bounds is None:
                msg = (
                    f"hyperparam_schema entry {self.name!r} of type 'int' requires "
                    "'bounds' to be set as (low, high)."
                )
                raise ValueError(msg)
            if self.values is not None:
                msg = (
                    f"hyperparam_schema entry {self.name!r} of type 'int' must not "
                    "set 'values' (use 'bounds' instead)."
                )
                raise ValueError(msg)
            if self.log_scale:
                msg = (
                    f"hyperparam_schema entry {self.name!r} of type 'int' must not "
                    "set 'log_scale=True' (log_scale is only meaningful for float)."
                )
                raise ValueError(msg)
            self._validate_bounds_range()
        elif self.type == "bool":
            if self.bounds is not None:
                msg = (
                    f"hyperparam_schema entry {self.name!r} of type 'bool' must not "
                    "set 'bounds' (bool is sampled as ±1 then thresholded by '> 0')."
                )
                raise ValueError(msg)
            if self.values is not None:
                msg = (
                    f"hyperparam_schema entry {self.name!r} of type 'bool' must not "
                    "set 'values' (use type 'categorical' for explicit value lists)."
                )
                raise ValueError(msg)
            if self.log_scale:
                msg = (
                    f"hyperparam_schema entry {self.name!r} of type 'bool' must not "
                    "set 'log_scale=True' (log_scale is only meaningful for float)."
                )
                raise ValueError(msg)
        elif self.type == "categorical":
            self._validate_categorical_values()
            if self.bounds is not None:
                msg = (
                    f"hyperparam_schema entry {self.name!r} of type 'categorical' "
                    "must not set 'bounds' (use 'values' instead)."
                )
                raise ValueError(msg)
            if self.log_scale:
                msg = (
                    f"hyperparam_schema entry {self.name!r} of type 'categorical' "
                    "must not set 'log_scale=True' (log_scale is only meaningful for float)."
                )
                raise ValueError(msg)
        return self

    def _validate_bounds_range(self) -> None:
        """Bounds SHALL be a strictly-increasing pair (low < high).

        Pydantic's ``tuple[float, float]`` typing only enforces "two
        numeric items"; ``bounds: [10, 5]`` would type-check but
        produce nonsense at sample/decode time.  Reject explicitly.
        """
        if self.bounds is None:  # pragma: no cover — caller-checked
            return
        low, high = self.bounds
        if low >= high:
            msg = (
                f"hyperparam_schema entry {self.name!r}: bounds ({low}, {high}) "
                "must be strictly increasing (low < high)."
            )
            raise ValueError(msg)

    def _validate_log_scale_positivity(self) -> None:
        """When ``log_scale=True``, both bounds SHALL be > 0.

        log(0) is -inf, log(<0) is NaN — either silently corrupts
        sampling and decode.  Catch at YAML-load time.
        """
        if not self.log_scale or self.bounds is None:
            return
        low, high = self.bounds
        if low <= 0 or high <= 0:
            msg = (
                f"hyperparam_schema entry {self.name!r}: log_scale=True requires "
                f"both bounds to be > 0; got ({low}, {high})."
            )
            raise ValueError(msg)

    def _validate_categorical_values(self) -> None:
        """Categorical entries SHALL have ≥2 distinct values.

        ``values: ["lstm", "lstm"]`` would pass the length-2 check
        but offer no real choice at decode time.
        """
        if self.values is None or len(self.values) < 2:  # noqa: PLR2004
            msg = (
                f"hyperparam_schema entry {self.name!r} of type 'categorical' "
                "requires 'values' to be set with at least 2 distinct items."
            )
            raise ValueError(msg)
        if len(set(self.values)) < 2:  # noqa: PLR2004
            msg = (
                f"hyperparam_schema entry {self.name!r} of type 'categorical': "
                f"values {self.values!r} contains duplicates; need at least 2 "
                "distinct items."
            )
            raise ValueError(msg)


class SimulationConfig(BaseModel):
    """Configuration for the simulation environment."""

    seed: int | None = None
    brain: BrainContainerConfig | None = None
    max_steps: int | None = None
    shots: int | None = None
    body_length: int | None = None
    qubits: int | None = None
    learning_rate: LearningRateConfig | None = None
    gradient: GradientConfig | None = None
    parameter_initializer: ParameterInitializerConfig | None = None
    reward: RewardConfig | None = None
    satiety: SatietyConfig | None = None
    modules: Modules | None = None
    manyworlds_mode: ManyworldsModeConfig | None = None
    environment: EnvironmentConfig | None = None
    multi_agent: MultiAgentConfig | None = None
    evolution: EvolutionConfig | None = None
    # Hyperparameter-evolution schema: top-level list of param-schema
    # entries.  When None (the default), runs use weight-evolution
    # dispatch.  When set, the run is a hyperparameter-evolution run
    # and select_encoder() returns a HyperparameterEncoder regardless
    # of brain.name.  See the evolution-framework capability spec for
    # the full contract.
    hyperparam_schema: list[ParamSchemaEntry] | None = None

    @model_validator(mode="after")
    def _validate_hyperparam_schema(self) -> "SimulationConfig":  # noqa: C901, PLR0912
        # C901/PLR0912 (too complex / too many branches): this validator
        # is a sequence of independent guards (None-skip, empty-list,
        # duplicate-name, brain-block-missing, brain-name-unknown,
        # field-name-unknown, warm-start-arch-incompat, inheritance-no-
        # schema, inheritance-arch-incompat).  Splitting into helpers
        # fragments the contract; each guard is a single small block
        # that's easier to read in sequence.
        """Cross-check hyperparam_schema entries against the brain config.

        Three defensive cases handled up-front (so users get clear
        errors instead of raw AttributeError/KeyError on malformed
        YAML):

        1. brain block missing — hyperparam_schema is meaningless without
           a brain to evolve hyperparameters for.
        2. brain name unknown — the user picked a brain that isn't in
           BRAIN_CONFIG_MAP (typo).
        3. brain name valid — walk the schema entries and check every
           entry.name is a real field on the resolved brain config
           Pydantic model.  Without this, Pydantic v2 model_copy(update=)
           silently no-ops typos and produces unevolved genomes.
        """
        # Inheritance requires hyperparam_schema: weight encoders evolve
        # weights, and Lamarckian on top would double-count weights as
        # both genome and substrate; Baldwin needs a hyperparameter
        # genome to evolve — without one there is no trait substrate to
        # inherit.  Caught here (rather than only in
        # EvolutionConfig._validate_inheritance) because the
        # hyperparam_schema field lives on SimulationConfig, not
        # EvolutionConfig.
        if (
            self.evolution is not None
            and self.evolution.inheritance != "none"
            and self.hyperparam_schema is None
        ):
            msg = (
                f"evolution.inheritance is {self.evolution.inheritance!r} but "
                "no hyperparam_schema is set.  Lamarckian inheritance over "
                "weight encoders would double-count weights as both genome "
                "and substrate; Baldwin needs a hyperparameter genome to "
                "evolve — without one there is no trait substrate to inherit. "
                "Either drop inheritance (returning to weight evolution) "
                "or add a hyperparam_schema (switching to hyperparameter "
                "evolution + inheritance)."
            )
            raise ValueError(msg)

        if self.hyperparam_schema is None:
            return self

        if not self.hyperparam_schema:
            msg = (
                "hyperparam_schema is set but empty.  Either remove the "
                "hyperparam_schema: key entirely (to fall back to weight evolution) "
                "or add at least one entry."
            )
            raise ValueError(msg)

        # Reject duplicate entry names: each schema entry maps to a brain
        # config field, and decode applies values via model_copy(update={...}).
        # Duplicate names would silently let the second entry's value win,
        # making the first slot's evolved genome value invisible.
        names = [entry.name for entry in self.hyperparam_schema]
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in names:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            msg = (
                f"hyperparam_schema contains duplicate entry name(s): {duplicates}.  "
                "Each entry must map to a distinct field on the brain config — "
                "duplicate names would silently let one slot's value override the "
                "other at decode time."
            )
            raise ValueError(msg)

        if self.brain is None:
            msg = (
                "hyperparam_schema requires a 'brain:' block in the YAML to resolve "
                "the field-name validation against. Add a brain block specifying "
                "brain.name and (optionally) brain.config."
            )
            raise ValueError(msg)

        brain_name = self.brain.name
        if brain_name not in BRAIN_CONFIG_MAP:
            registered = sorted(BRAIN_CONFIG_MAP)
            msg = (
                f"hyperparam_schema references brain.name {brain_name!r}, which is "
                f"not a registered brain. Registered brains: {registered}."
            )
            raise ValueError(msg)

        brain_config_cls = BRAIN_CONFIG_MAP[brain_name]
        brain_fields = set(brain_config_cls.model_fields.keys())
        max_alternatives_to_surface = 10
        for entry in self.hyperparam_schema:
            if entry.name not in brain_fields:
                # Surface 3+ valid alternatives so the user can spot a typo.
                alternatives = sorted(brain_fields)
                preview = alternatives[:max_alternatives_to_surface]
                truncated = "..." if len(alternatives) > max_alternatives_to_surface else ""
                msg = (
                    f"hyperparam_schema entry {entry.name!r} is not a field on "
                    f"{brain_config_cls.__name__}. Valid alternatives include: "
                    f"{preview}{truncated}."
                )
                raise ValueError(msg)

        # Warm-start incompatibility: an evolved architecture field would
        # reshape some layer's state_dict, which ``load_weights`` would then
        # either crash on or silently mis-load.  Reject at load time so the
        # 100-genome x 20-generation campaign doesn't discover the problem
        # mid-run.
        if self.evolution is not None and self.evolution.warm_start_path is not None:
            offenders = sorted(
                {entry.name for entry in self.hyperparam_schema} & _ARCHITECTURE_CHANGING_FIELDS,
            )
            if offenders:
                msg = (
                    "evolution.warm_start_path is set but hyperparam_schema "
                    f"contains architecture-changing entries: {offenders}.  "
                    "Warm-start loads a fixed-shape checkpoint; the genome "
                    "cannot reshape the brain at the same time.  Either "
                    "(a) drop the architecture entries from the schema and "
                    "evolve only non-architecture fields (learning rates, "
                    "gamma, entropy, etc.), or (b) drop warm_start_path and "
                    "let each genome train from a fresh init."
                )
                raise ValueError(msg)

        # Inheritance arch-fields incompatibility: same shape-mismatch
        # reasoning as warm-start above, applied to per-genome dynamic
        # checkpoints.  Single source of truth via the same
        # _ARCHITECTURE_CHANGING_FIELDS denylist.  Applies to Lamarckian
        # ONLY: Baldwin doesn't load weights, so shape mismatches are
        # fine — a future Baldwin arm can evolve actor_hidden_dim etc.
        if self.evolution is not None and self.evolution.inheritance == "lamarckian":
            offenders = sorted(
                {entry.name for entry in self.hyperparam_schema} & _ARCHITECTURE_CHANGING_FIELDS,
            )
            if offenders:
                msg = (
                    f"evolution.inheritance is {self.evolution.inheritance!r} but "
                    f"hyperparam_schema contains architecture-changing entries: "
                    f"{offenders}.  Per-genome checkpoints cannot be loaded "
                    "into a child whose architecture differs from the parent's. "
                    "Either drop the architecture entries from the schema and "
                    "evolve only non-architecture fields, or drop inheritance."
                )
                raise ValueError(msg)

        return self


# Brain-config fields whose values change tensor shapes (and therefore the
# ``state_dict`` layout) when the brain is constructed.  A warm-start
# checkpoint is saved with one specific shape; loading it into a brain that
# the genome has reshaped via one of these fields would either crash with a
# state_dict mismatch or silently load the wrong slices.  Rejected at YAML
# load time when ``evolution.warm_start_path`` is set.
#
# This list is intentionally hand-curated rather than introspected.  The
# alternative (instantiate two brains with two values, diff state_dicts)
# is significantly more code for a constant that changes very rarely —
# new architecture fields land via PR review.  Update this set when
# adding architecture-changing fields to a brain config.
_ARCHITECTURE_CHANGING_FIELDS: frozenset[str] = frozenset(
    {
        # MLPPPO
        "actor_hidden_dim",
        "critic_hidden_dim",
        "num_hidden_layers",
        # LSTMPPO
        "actor_num_layers",
        "critic_num_layers",
        "lstm_hidden_dim",
        "rnn_type",
    },
)


class PlasticityPhaseConfig(BaseModel):
    """Configuration for a single phase in the plasticity evaluation protocol."""

    name: str
    environment: EnvironmentConfig
    reward: RewardConfig = RewardConfig()
    satiety: SatietyConfig | None = None
    max_steps: int = 2000


class PlasticityProtocolConfig(BaseModel):
    """Configuration for the plasticity evaluation protocol parameters."""

    training_episodes_per_phase: int = 200
    eval_episodes: int = 50
    seeds: list[int] = [42, 123, 256, 512, 789, 1024, 2048, 4096]
    convergence_threshold: float = Field(default=0.6, gt=0.0, le=1.0)
    phases: list[PlasticityPhaseConfig]


class PlasticityConfig(BaseModel):
    """Top-level configuration for a plasticity evaluation run."""

    brain: BrainContainerConfig
    plasticity: PlasticityProtocolConfig
    # Optional top-level overrides (shared across phases unless phase overrides)
    shots: int | None = None
    qubits: int | None = None
    body_length: int = 2
    learning_rate: LearningRateConfig | None = None
    gradient: GradientConfig | None = None
    parameter_initializer: ParameterInitializerConfig | None = None
    modules: Modules | None = None


def load_plasticity_config(config_path: str) -> PlasticityConfig:
    """Load plasticity evaluation configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    PlasticityConfig
        Parsed configuration as a Pydantic model.
    """
    with Path(config_path).open() as file:
        data = yaml.safe_load(file)
        return PlasticityConfig(**data)


def load_simulation_config(config_path: str) -> SimulationConfig:
    """
    Load simulation configuration from a YAML file and parse it into a SimulationConfig model.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns
    -------
        SimulationConfig: Parsed configuration as a Pydantic model.
    """
    with Path(config_path).open() as file:
        data = yaml.safe_load(file)
        return SimulationConfig(**data)


def configure_brain(
    config: SimulationConfig,
) -> BrainConfigType:
    """
    Configure the brain architecture based on the provided configuration.

    Args:
        config (SimulationConfig): Simulation configuration object.

    Returns
    -------
        BrainConfigType:
            The configured brain architecture.
    """
    if config.brain is None:
        error_message = "No brain configuration found in the simulation config."
        logger.error(error_message)
        raise ValueError(error_message)

    if config.brain.name is None:
        error_message = "No brain name specified in the simulation config."
        logger.error(error_message)
        raise ValueError(error_message)

    brain_name = config.brain.name

    if brain_name not in BRAIN_CONFIG_MAP:
        error_message = f"Unknown brain type: {config.brain.name}."
        logger.error(error_message)
        raise ValueError(error_message)

    config_cls = BRAIN_CONFIG_MAP[brain_name]
    return _resolve_brain_config(config.brain.config, config_cls, brain_name)


def configure_brain_from_container(
    brain_container: BrainContainerConfig,
) -> BrainConfigType:
    """Configure brain from a BrainContainerConfig (for multi-agent per-agent configs).

    Parameters
    ----------
    brain_container : BrainContainerConfig
        Brain container with name and config.

    Returns
    -------
    BrainConfigType
        Parsed brain configuration.
    """
    brain_name = brain_container.name
    if brain_name not in BRAIN_CONFIG_MAP:
        msg = f"Unknown brain type: {brain_name}."
        raise ValueError(msg)
    config_cls = BRAIN_CONFIG_MAP[brain_name]
    return _resolve_brain_config(brain_container.config, config_cls, brain_name)


def configure_learning_rate(
    config: SimulationConfig,
) -> ConstantLearningRate | DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate:
    """
    Configure the learning rate based on the provided configuration.

    Args:
        config (SimulationConfig): Simulation configuration object.

    Returns
    -------
        ConstantLearningRate | DynamicLearningRate |
                AdamLearningRate | PerformanceBasedLearningRate:
            Configured learning rate object.
    """
    lr_cfg = config.learning_rate
    if lr_cfg is None:
        logger.warning(
            "No learning rate configuration found. Using default DynamicLearningRate.",
        )
        return DynamicLearningRate()

    method = lr_cfg.method
    params = lr_cfg.parameters or LearningRateParameters()
    if method == LearningRateMethod.CONSTANT:
        return ConstantLearningRate(
            learning_rate=params.learning_rate,
        )
    if method == LearningRateMethod.DYNAMIC:
        decay_type = _resolve_decay_type(params)
        return DynamicLearningRate(
            initial_learning_rate=params.initial_learning_rate,
            decay_rate=params.decay_rate,
            decay_type=decay_type,
            decay_factor=params.decay_factor,
            step_size=params.step_size,
            max_steps=params.max_steps,
            power=params.power,
            min_lr=params.min_lr,
        )
    if method == LearningRateMethod.ADAM:
        return AdamLearningRate(
            initial_learning_rate=params.initial_learning_rate,
            beta1=params.beta1,
            beta2=params.beta2,
            epsilon=params.epsilon,
        )
    if method == LearningRateMethod.PERFORMANCE_BASED:
        return PerformanceBasedLearningRate(
            initial_learning_rate=params.initial_learning_rate,
            min_learning_rate=params.min_learning_rate,
            max_learning_rate=params.max_learning_rate,
            adjustment_factor=params.adjustment_factor,
        )
    error_message = (
        f"Unknown learning rate method: {method}. "
        f"Supported methods are {[m.value for m in LearningRateMethod]}."
    )
    logger.error(error_message)
    raise ValueError(error_message)


def _resolve_decay_type(learning_rate_parameters: LearningRateParameters) -> DecayType:
    decay_type_value = (
        learning_rate_parameters.decay_type or DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_TYPE.value
    )
    if isinstance(decay_type_value, DecayType):
        decay_type = decay_type_value
    else:
        try:
            decay_type = DecayType(decay_type_value)
        except ValueError:
            logger.warning(
                f"Unknown decay_type '{decay_type_value}', "
                "defaulting to '{DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_TYPE.value}'.",
            )
            decay_type = DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_TYPE
    return decay_type


def configure_gradient_method(
    gradient_method: GradientCalculationMethod,
    config: SimulationConfig,
) -> tuple[GradientCalculationMethod, float | None]:
    """
    Configure the gradient calculation method based on the provided configuration.

    Args:
        gradient_method (GradientCalculationMethod): The default gradient calculation method.
        config (SimulationConfig): Simulation configuration object.

    Returns
    -------
        tuple[GradientCalculationMethod, float | None]: The configured gradient calculation method
            and optional max_norm parameter for norm_clip method.

    Raises
    ------
        ValueError: If an invalid gradient method is specified in the configuration.
    """
    grad_cfg = config.gradient or GradientConfig()
    method = grad_cfg.method or gradient_method
    max_norm = grad_cfg.max_norm
    if method == GradientCalculationMethod.NORM_CLIP and max_norm is None:
        logger.info(
            "norm_clip method configured without max_norm, "
            f"using default: {DEFAULT_MAX_GRADIENT_NORM}",
        )
    return method, max_norm


def configure_parameter_initializer(
    config: SimulationConfig,
) -> ParameterInitializerConfig:
    """
    Configure the parameter initializer based on the provided configuration.

    Args:
        config (SimulationConfig): Simulation configuration object.

    Returns
    -------
        ParameterInitializerConfig: The configured parameter initializer object.
    """
    return config.parameter_initializer or ParameterInitializerConfig()


def create_parameter_initializer_instance(
    param_config: ParameterInitializerConfig,
) -> (
    ZeroInitializer
    | RandomPiUniformInitializer
    | RandomSmallUniformInitializer
    | ManualParameterInitializer
    | None
):
    """
    Create a parameter initializer instance from configuration.

    Args:
        param_config: Parameter initializer configuration.

    Returns
    -------
        Parameter initializer instance.

    Raises
    ------
        ValueError: If an unknown parameter initializer type is specified.
    """
    init_type = param_config.type.lower()

    if init_type == "manual":
        if param_config.manual_parameter_values is None:
            error_message = "Manual parameter initializer type specified "
            "but no manual_parameter_values provided in config."
            logger.error(error_message)
            raise ValueError(error_message)
        return ManualParameterInitializer(param_config.manual_parameter_values)
    if init_type == "zero":
        return ZeroInitializer()
    if init_type == "random_pi":
        return RandomPiUniformInitializer()
    if init_type == "random_small":
        return RandomSmallUniformInitializer()
    error_message = (
        f"Unknown parameter initializer type: {init_type}. "
        "Valid options are: 'manual', 'zero', 'random_pi', 'random_small'."
    )
    logger.error(error_message)
    raise ValueError(error_message)


def configure_manyworlds_mode(config: SimulationConfig) -> ManyworldsModeConfig:
    """
    Configure the many-worlds mode based on the provided configuration.

    Args:
        config (SimulationConfig): Simulation configuration object.

    Returns
    -------
        ManyworldsModeConfig: The configured many-worlds mode object.
    """
    sp_cfg = config.manyworlds_mode or ManyworldsModeConfig()
    return ManyworldsModeConfig(
        max_superpositions=sp_cfg.max_superpositions,
        max_columns=sp_cfg.max_columns,
        render_sleep_seconds=sp_cfg.render_sleep_seconds,
        top_n_actions=sp_cfg.top_n_actions,
        top_n_randomize=sp_cfg.top_n_randomize,
    )


def configure_reward(config: SimulationConfig) -> RewardConfig:
    """
    Configure the reward function based on the provided configuration.

    Args:
        config (SimulationConfig): Simulation configuration object.

    Returns
    -------
        RewardConfig: The configured reward function object.
    """
    return config.reward or RewardConfig()


def configure_satiety(config: SimulationConfig) -> SatietyConfig:
    """
    Configure the satiety system based on the provided configuration.

    Args:
        config (SimulationConfig): Simulation configuration object.

    Returns
    -------
        SatietyConfig: The configured satiety system object.
    """
    return config.satiety or SatietyConfig()


def configure_environment(config: SimulationConfig) -> EnvironmentConfig:
    """
    Configure the environment based on the provided configuration.

    Args:
        config (SimulationConfig): Simulation configuration object.

    Returns
    -------
        EnvironmentConfig: The configured environment object.
    """
    return config.environment or EnvironmentConfig()


def create_env_from_config(
    env_config: EnvironmentConfig,
    *,
    seed: int | None = None,
    max_body_length: int | None = None,
    theme: "Theme | None" = None,
) -> "DynamicForagingEnvironment":
    """Create a DynamicForagingEnvironment from an EnvironmentConfig.

    This is a convenience factory for creating environments from parsed
    configuration, usable from scripts, notebooks, or tests.

    Parameters
    ----------
    env_config : EnvironmentConfig
        Parsed environment configuration.
    seed : int or None, optional
        Seed for environment RNG.
    max_body_length : int or None, optional
        Max body length for the agent. Defaults to 6.
    theme : Theme or None, optional
        Rendering theme. Defaults to ``Theme.ASCII``.

    Returns
    -------
    DynamicForagingEnvironment
        Configured environment instance.
    """
    from quantumnematode.env import DynamicForagingEnvironment
    from quantumnematode.env.theme import Theme as ThemeEnum

    foraging_config = env_config.get_foraging_config()
    predator_config = env_config.get_predator_config()
    health_config = env_config.get_health_config()
    thermotaxis_config = env_config.get_thermotaxis_config()
    aerotaxis_config = env_config.get_aerotaxis_config()
    pheromone_config = env_config.get_pheromone_config()
    social_feeding_config = env_config.get_social_feeding_config()

    return DynamicForagingEnvironment(
        grid_size=env_config.grid_size,
        viewport_size=env_config.viewport_size,
        max_body_length=max_body_length if max_body_length is not None else 6,
        theme=theme if theme is not None else ThemeEnum.ASCII,
        seed=seed,
        foraging=foraging_config.to_params(),
        predator=predator_config.to_params(),
        health=health_config.to_params(),
        thermotaxis=thermotaxis_config.to_params(),
        aerotaxis=aerotaxis_config.to_params(),
        pheromones=pheromone_config.to_params(),
        social_feeding=social_feeding_config.to_params(),
    )
