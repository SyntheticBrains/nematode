"""Load and configure simulation settings from a YAML file."""

from pathlib import Path

import yaml
from pydantic import BaseModel

from quantumnematode.agent import (
    ManyworldsModeConfig,
    RewardConfig,
)
from quantumnematode.brain.arch import (
    MLPBrainConfig,
    ModularBrainConfig,
    QMLPBrainConfig,
    QModularBrainConfig,
    SpikingBrainConfig,
)
from quantumnematode.brain.modules import Modules
from quantumnematode.initializers import (
    ManualParameterInitializer,
    RandomPiUniformInitializer,
    RandomSmallUniformInitializer,
    ZeroInitializer,
)
from quantumnematode.logging_config import (
    logger,
)
from quantumnematode.optimizers.gradient_methods import GradientCalculationMethod
from quantumnematode.optimizers.learning_rate import (
    DEFAULT_ADAM_LEARNING_RATE_BETA1,
    DEFAULT_ADAM_LEARNING_RATE_BETA2,
    DEFAULT_ADAM_LEARNING_RATE_EPSILON,
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
    DecayType,
    DynamicLearningRate,
    LearningRateMethod,
    PerformanceBasedLearningRate,
)

DEFAULT_LEARNING_RATE_INITIAL = 0.1
DEFAULT_LEARNING_RATE_METHOD = LearningRateMethod.DYNAMIC

BrainConfigType = (
    ModularBrainConfig | MLPBrainConfig | QMLPBrainConfig | QModularBrainConfig | SpikingBrainConfig
)


class BrainContainerConfig(BaseModel):
    """Configuration for the brain architecture."""

    name: str
    config: BrainConfigType | None = None


class LearningRateParameters(BaseModel):
    """Parameters for configuring the learning rate."""

    initial_learning_rate: float = DEFAULT_LEARNING_RATE_INITIAL
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

    method: LearningRateMethod = DEFAULT_LEARNING_RATE_METHOD
    parameters: LearningRateParameters = LearningRateParameters()


DEFAULT_GRADIENT_CALCULATION_METHOD = GradientCalculationMethod.RAW


class GradientConfig(BaseModel):
    """Configuration for the gradient calculation method."""

    method: GradientCalculationMethod = DEFAULT_GRADIENT_CALCULATION_METHOD


DEFAULT_PARAMETER_INITIALIZER_TYPE = "random_small"


class ParameterInitializerConfig(BaseModel):
    """Configuration for parameter initialization."""

    type: str = DEFAULT_PARAMETER_INITIALIZER_TYPE
    manual_parameter_values: dict[str, float] | None = None


class DynamicEnvironmentConfig(BaseModel):
    """Configuration for dynamic foraging environment."""

    grid_size: int = 50
    num_initial_foods: int = 10
    max_active_foods: int = 15
    min_food_distance: int = 5
    agent_exclusion_radius: int = 10
    gradient_decay_constant: float = 10.0
    gradient_strength: float = 1.0
    viewport_size: tuple[int, int] = (11, 11)


class EnvironmentConfig(BaseModel):
    """Configuration for environment selection and parameters."""

    type: str = "static"  # "static" or "dynamic"
    dynamic: DynamicEnvironmentConfig | None = None


class SimulationConfig(BaseModel):
    """Configuration for the simulation environment."""

    brain: BrainContainerConfig | None = None
    max_steps: int | None = None
    maze_grid_size: int | None = None
    shots: int | None = None
    body_length: int | None = None
    qubits: int | None = None
    learning_rate: LearningRateConfig | None = None
    gradient: GradientConfig | None = None
    parameter_initializer: ParameterInitializerConfig | None = None
    reward: RewardConfig | None = None
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

    match config.brain.name:
        case "modular":
            if config.brain.config is None:
                return ModularBrainConfig()
            if isinstance(config.brain.config, ModularBrainConfig):
                return config.brain.config
            # Handle case where YAML parsed as wrong type - reconstruct as ModularBrainConfig
            if hasattr(config.brain.config, "__dict__"):
                return ModularBrainConfig(**config.brain.config.__dict__)
            error_message = (
                "Invalid brain configuration for 'modular' brain type. "
                f"Expected ModularBrainConfig, got {type(config.brain.config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        case "qmodular":
            if config.brain.config is None:
                return QModularBrainConfig()
            if isinstance(config.brain.config, QModularBrainConfig):
                return config.brain.config
            # Handle case where YAML parsed as wrong type - reconstruct as QModularBrainConfig
            if hasattr(config.brain.config, "__dict__"):
                # Filter only attributes that exist in QModularBrainConfig
                config_dict = {}
                for field_name in QModularBrainConfig.model_fields:
                    if hasattr(config.brain.config, field_name):
                        config_dict[field_name] = getattr(config.brain.config, field_name)
                return QModularBrainConfig(**config_dict)
            error_message = (
                "Invalid brain configuration for 'qmodular' brain type. "
                f"Expected QModularBrainConfig, got {type(config.brain.config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        case "mlp":
            if config.brain.config is None:
                return MLPBrainConfig()
            if isinstance(config.brain.config, MLPBrainConfig):
                return config.brain.config
            # Handle case where YAML parsed as wrong type - reconstruct as MLPBrainConfig
            if hasattr(config.brain.config, "__dict__"):
                config_dict = {}
                for field_name in MLPBrainConfig.model_fields:
                    if hasattr(config.brain.config, field_name):
                        config_dict[field_name] = getattr(config.brain.config, field_name)
                return MLPBrainConfig(**config_dict)
            error_message = (
                "Invalid brain configuration for 'mlp' brain type. "
                f"Expected MLPBrainConfig, got {type(config.brain.config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        case "qmlp":
            if config.brain.config is None:
                return QMLPBrainConfig()
            if isinstance(config.brain.config, QMLPBrainConfig):
                return config.brain.config
            # Handle case where YAML parsed as wrong type - reconstruct as QMLPBrainConfig
            if hasattr(config.brain.config, "__dict__"):
                config_dict = {}
                for field_name in QMLPBrainConfig.model_fields:
                    if hasattr(config.brain.config, field_name):
                        config_dict[field_name] = getattr(config.brain.config, field_name)
                return QMLPBrainConfig(**config_dict)
            error_message = (
                "Invalid brain configuration for 'qmlp' brain type. "
                f"Expected QMLPBrainConfig, got {type(config.brain.config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        case "spiking":
            if config.brain.config is None:
                return SpikingBrainConfig()
            if isinstance(config.brain.config, SpikingBrainConfig):
                return config.brain.config
            # Handle case where YAML parsed as wrong type - reconstruct as SpikingBrainConfig
            if hasattr(config.brain.config, "__dict__"):
                config_dict = {}
                for field_name in SpikingBrainConfig.model_fields:
                    if hasattr(config.brain.config, field_name):
                        config_dict[field_name] = getattr(config.brain.config, field_name)
                return SpikingBrainConfig(**config_dict)
            error_message = (
                "Invalid brain configuration for 'spiking' brain type. "
                f"Expected SpikingBrainConfig, got {type(config.brain.config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        case _:
            error_message = f"Unknown brain type: {config.brain.name}."
            logger.error(error_message)
            raise ValueError(error_message)


def configure_learning_rate(
    config: SimulationConfig,
) -> DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate:
    """
    Configure the learning rate based on the provided configuration.

    Args:
        config (SimulationConfig): Simulation configuration object.

    Returns
    -------
        DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate:
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
) -> GradientCalculationMethod:
    """
    Configure the gradient calculation method based on the provided configuration.

    Args:
        gradient_method (GradientCalculationMethod): The default gradient calculation method.
        config (SimulationConfig): Simulation configuration object.

    Returns
    -------
        GradientCalculationMethod: The configured gradient calculation method.

    Raises
    ------
        ValueError: If an invalid gradient method is specified in the configuration.
    """
    grad_cfg = config.gradient or GradientConfig()
    return grad_cfg.method or gradient_method


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
