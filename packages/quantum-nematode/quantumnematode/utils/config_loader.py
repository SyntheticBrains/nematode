"""Load and configure simulation settings from a YAML file."""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel

from quantumnematode.agent import (
    ManyworldsModeConfig,
    RewardConfig,
    SatietyConfig,
)
from quantumnematode.brain.arch import (
    MLPDQNBrainConfig,
    MLPPPOBrainConfig,
    MLPReinforceBrainConfig,
    QQLearningBrainConfig,
    QVarCircuitBrainConfig,
    SpikingReinforceBrainConfig,
)
from quantumnematode.brain.arch.dtypes import BRAIN_NAME_ALIASES
from quantumnematode.brain.modules import Modules
from quantumnematode.dtypes import TemperatureSpot
from quantumnematode.env.env import (
    DEFAULT_COMFORT_REWARD,
    DEFAULT_CULTIVATION_TEMPERATURE,
    DEFAULT_DANGER_HP_DAMAGE,
    DEFAULT_DANGER_PENALTY,
    DEFAULT_DISCOMFORT_PENALTY,
    DEFAULT_LETHAL_HP_DAMAGE,
    DEFAULT_TEMPERATURE_GRADIENT_STRENGTH,
    ForagingParams,
    HealthParams,
    PredatorParams,
    PredatorType,
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
    | SpikingReinforceBrainConfig
)

# Type alias for predator movement patterns
MovementPattern = Literal["random", "stationary", "pursuit"]


class BrainContainerConfig(BaseModel):
    """Configuration for the brain architecture."""

    name: str
    config: BrainConfigType | None = None


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

    def to_params(self) -> ForagingParams:
        """Convert to ForagingParams for environment initialization."""
        return ForagingParams(
            foods_on_grid=self.foods_on_grid,
            target_foods_to_collect=self.target_foods_to_collect,
            min_food_distance=self.min_food_distance,
            agent_exclusion_radius=self.agent_exclusion_radius,
            gradient_decay_constant=self.gradient_decay_constant,
            gradient_strength=self.gradient_strength,
            safe_zone_food_bias=self.safe_zone_food_bias,
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
        Movement behavior: "random", "stationary", or "pursuit".
    detection_radius : int
        Distance at which pursuit predators detect the agent.
    kill_radius : int
        Distance for instant death (when health system disabled).
    damage_radius : int
        Distance at which predators deal damage (when health system enabled).
        Stationary predators typically have larger damage_radius (toxic zones).
    gradient_decay_constant : float
        Controls how quickly predator gradient signal decays with distance.
    gradient_strength : float
        Multiplier for predator gradient signal strength.
    """

    enabled: bool = False
    count: int = 2  # Maps to DynamicForagingEnvironment.num_predators
    speed: float = 1.0  # Maps to DynamicForagingEnvironment.predator_speed
    movement_pattern: MovementPattern = "random"
    # Maps to DynamicForagingEnvironment.predator_detection_radius
    detection_radius: int = 8
    kill_radius: int = 0  # Maps to DynamicForagingEnvironment.predator_kill_radius
    damage_radius: int = 0  # Distance for damage application (health system)
    # Maps to DynamicForagingEnvironment.predator_gradient_decay
    gradient_decay_constant: float = 12.0
    # Maps to DynamicForagingEnvironment.predator_gradient_strength
    gradient_strength: float = 1.0

    def to_params(self) -> PredatorParams:
        """Convert to PredatorParams for environment initialization."""
        # Map movement_pattern string to PredatorType enum
        pattern_to_type = {
            "random": PredatorType.RANDOM,
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
            kill_radius=self.kill_radius,
            damage_radius=self.damage_radius,
            gradient_decay_constant=self.gradient_decay_constant,
            gradient_strength=self.gradient_strength,
        )


class HealthConfig(BaseModel):
    """Configuration for HP-based health system.

    When enabled, predator contact deals damage instead of instant death.
    Food consumption restores both HP and satiety (when both systems are enabled).
    """

    enabled: bool = False
    max_hp: float = 100.0
    predator_damage: float = 10.0
    food_healing: float = 5.0

    def to_params(self) -> HealthParams:
        """Convert to HealthParams for environment initialization."""
        return HealthParams(
            enabled=self.enabled,
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


class EnvironmentConfig(BaseModel):
    """Configuration for the dynamic foraging environment."""

    grid_size: int = 50
    viewport_size: tuple[int, int] = (11, 11)
    use_separated_gradients: bool = False  # Whether to use separated food/predator gradients

    # Nested configuration subsections
    foraging: ForagingConfig | None = None
    predators: PredatorConfig | None = None
    health: HealthConfig | None = None
    thermotaxis: ThermotaxisConfig | None = None

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


# Backward compatibility alias
DynamicEnvironmentConfig = EnvironmentConfig


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


def configure_brain(  # noqa: C901, PLR0911, PLR0912, PLR0915
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

    # Resolve deprecated names to canonical names
    brain_name = BRAIN_NAME_ALIASES.get(config.brain.name, config.brain.name)

    match brain_name:
        case "qvarcircuit":
            if config.brain.config is None:
                return QVarCircuitBrainConfig()
            if isinstance(config.brain.config, QVarCircuitBrainConfig):
                return config.brain.config
            if hasattr(config.brain.config, "__dict__"):
                return QVarCircuitBrainConfig(**config.brain.config.__dict__)
            error_message = (
                "Invalid brain configuration for 'qvarcircuit' brain type. "
                f"Expected QVarCircuitBrainConfig, got {type(config.brain.config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        case "qqlearning":
            if config.brain.config is None:
                return QQLearningBrainConfig()
            if isinstance(config.brain.config, QQLearningBrainConfig):
                return config.brain.config
            if hasattr(config.brain.config, "__dict__"):
                config_dict = {}
                for field_name in QQLearningBrainConfig.model_fields:
                    if hasattr(config.brain.config, field_name):
                        config_dict[field_name] = getattr(config.brain.config, field_name)
                return QQLearningBrainConfig(**config_dict)
            error_message = (
                "Invalid brain configuration for 'qqlearning' brain type. "
                f"Expected QQLearningBrainConfig, got {type(config.brain.config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        case "mlpreinforce":
            if config.brain.config is None:
                return MLPReinforceBrainConfig()
            if isinstance(config.brain.config, MLPReinforceBrainConfig):
                return config.brain.config
            if hasattr(config.brain.config, "__dict__"):
                config_dict = {}
                for field_name in MLPReinforceBrainConfig.model_fields:
                    if hasattr(config.brain.config, field_name):
                        config_dict[field_name] = getattr(config.brain.config, field_name)
                return MLPReinforceBrainConfig(**config_dict)
            error_message = (
                "Invalid brain configuration for 'mlpreinforce' brain type. "
                f"Expected MLPReinforceBrainConfig, got {type(config.brain.config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        case "mlpppo":
            if config.brain.config is None:
                return MLPPPOBrainConfig()
            if isinstance(config.brain.config, MLPPPOBrainConfig):
                return config.brain.config
            if hasattr(config.brain.config, "__dict__"):
                config_dict = {}
                for field_name in MLPPPOBrainConfig.model_fields:
                    if hasattr(config.brain.config, field_name):
                        config_dict[field_name] = getattr(config.brain.config, field_name)
                return MLPPPOBrainConfig(**config_dict)
            error_message = (
                "Invalid brain configuration for 'mlpppo' brain type. "
                f"Expected MLPPPOBrainConfig, got {type(config.brain.config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        case "mlpdqn":
            if config.brain.config is None:
                return MLPDQNBrainConfig()
            if isinstance(config.brain.config, MLPDQNBrainConfig):
                return config.brain.config
            if hasattr(config.brain.config, "__dict__"):
                config_dict = {}
                for field_name in MLPDQNBrainConfig.model_fields:
                    if hasattr(config.brain.config, field_name):
                        config_dict[field_name] = getattr(config.brain.config, field_name)
                return MLPDQNBrainConfig(**config_dict)
            error_message = (
                "Invalid brain configuration for 'mlpdqn' brain type. "
                f"Expected MLPDQNBrainConfig, got {type(config.brain.config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        case "spikingreinforce":
            if config.brain.config is None:
                return SpikingReinforceBrainConfig()
            if isinstance(config.brain.config, SpikingReinforceBrainConfig):
                return config.brain.config
            if hasattr(config.brain.config, "__dict__"):
                config_dict = {}
                for field_name in SpikingReinforceBrainConfig.model_fields:
                    if hasattr(config.brain.config, field_name):
                        config_dict[field_name] = getattr(config.brain.config, field_name)
                return SpikingReinforceBrainConfig(**config_dict)
            error_message = (
                "Invalid brain configuration for 'spikingreinforce' brain type. "
                f"Expected SpikingReinforceBrainConfig, got {type(config.brain.config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        case _:
            error_message = f"Unknown brain type: {config.brain.name}."
            logger.error(error_message)
            raise ValueError(error_message)


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

    Args:
        env_config: Parsed environment configuration.
        seed: Optional seed for environment RNG.
        max_body_length: Optional max body length for the agent.
        theme: Optional rendering theme.

    Returns
    -------
        Configured DynamicForagingEnvironment instance.
    """
    from quantumnematode.env import DynamicForagingEnvironment
    from quantumnematode.env.theme import Theme as ThemeEnum

    foraging_config = env_config.get_foraging_config()
    predator_config = env_config.get_predator_config()
    health_config = env_config.get_health_config()
    thermotaxis_config = env_config.get_thermotaxis_config()

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
    )
