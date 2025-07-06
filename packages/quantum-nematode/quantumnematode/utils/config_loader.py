"""Load and configure simulation settings from a YAML file."""

from pathlib import Path

import yaml
from pydantic import BaseModel

from quantumnematode.agent import (
    DEFAULT_SUPERPOSITION_MODE_MAX_COLUMNS,
    DEFAULT_SUPERPOSITION_MODE_MAX_SUPERPOSITIONS,
    DEFAULT_SUPERPOSITION_MODE_RENDER_SLEEP_SECONDS,
    DEFAULT_SUPERPOSITION_MODE_TOP_N_ACTIONS,
    DEFAULT_SUPERPOSITION_MODE_TOP_N_RANDOMIZE,
)
from quantumnematode.logging_config import (
    logger,
)
from quantumnematode.optimizers.gradient_methods import GradientCalculationMethod
from quantumnematode.optimizers.learning_rate import (
    DEFAULT_ADAM_LEARNING_RATE_BETA1,
    DEFAULT_ADAM_LEARNING_RATE_BETA2,
    DEFAULT_ADAM_LEARNING_RATE_EPSILON,
    DEFAULT_ADAM_LEARNING_RATE_INITIAL,
    DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_FACTOR,
    DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_RATE,
    DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_TYPE,
    DEFAULT_DYNAMIC_LEARNING_RATE_INITIAL,
    DEFAULT_DYNAMIC_LEARNING_RATE_MAX_STEPS,
    DEFAULT_DYNAMIC_LEARNING_RATE_MIN_LR,
    DEFAULT_DYNAMIC_LEARNING_RATE_POWER,
    DEFAULT_DYNAMIC_LEARNING_RATE_STEP_SIZE,
    DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_ADJUSTMENT_FACTOR,
    DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_INITIAL,
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


class LearningRateParameters(BaseModel):
    """Parameters for configuring the learning rate."""

    initial_learning_rate: float | None = DEFAULT_LEARNING_RATE_INITIAL
    decay_rate: float | None = DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_RATE
    decay_type: str | None = DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_TYPE.value
    decay_factor: float | None = DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_FACTOR
    step_size: int | None = DEFAULT_DYNAMIC_LEARNING_RATE_STEP_SIZE
    max_steps: int | None = DEFAULT_DYNAMIC_LEARNING_RATE_MAX_STEPS
    power: float | None = DEFAULT_DYNAMIC_LEARNING_RATE_POWER
    min_lr: float | None = DEFAULT_DYNAMIC_LEARNING_RATE_MIN_LR
    beta1: float | None = DEFAULT_ADAM_LEARNING_RATE_BETA1
    beta2: float | None = DEFAULT_ADAM_LEARNING_RATE_BETA2
    epsilon: float | None = DEFAULT_ADAM_LEARNING_RATE_EPSILON
    min_learning_rate: float | None = DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_MIN
    max_learning_rate: float | None = DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_MAX
    adjustment_factor: float | None = DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_ADJUSTMENT_FACTOR


class LearningRateConfig(BaseModel):
    """Configuration for the learning rate method and its parameters."""

    method: LearningRateMethod = DEFAULT_LEARNING_RATE_METHOD
    parameters: LearningRateParameters | None = LearningRateParameters()


DEFAULT_GRADIENT_CALCULATION_METHOD = GradientCalculationMethod.RAW


class GradientConfig(BaseModel):
    """Configuration for the gradient calculation method."""

    method: GradientCalculationMethod | None = DEFAULT_GRADIENT_CALCULATION_METHOD


class SuperpositionModeConfig(BaseModel):
    """Configuration for the superposition mode."""

    max_superpositions: int = DEFAULT_SUPERPOSITION_MODE_MAX_SUPERPOSITIONS
    max_columns: int = DEFAULT_SUPERPOSITION_MODE_MAX_COLUMNS
    render_sleep_seconds: float = DEFAULT_SUPERPOSITION_MODE_RENDER_SLEEP_SECONDS
    top_n_actions: int = DEFAULT_SUPERPOSITION_MODE_TOP_N_ACTIONS
    top_n_randomize: bool = DEFAULT_SUPERPOSITION_MODE_TOP_N_RANDOMIZE


class SimulationConfig(BaseModel):
    """Configuration for the simulation environment."""

    max_steps: int | None = None
    maze_grid_size: int | None = None
    brain: str | None = None
    shots: int | None = None
    body_length: int | None = None
    qubits: int | None = None
    learning_rate: LearningRateConfig | None = None
    gradient: GradientConfig | None = None
    superposition_mode: SuperpositionModeConfig | None = None


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
            initial_learning_rate=params.initial_learning_rate
            if params.initial_learning_rate is not None
            else DEFAULT_DYNAMIC_LEARNING_RATE_INITIAL,
            decay_rate=params.decay_rate
            if params.decay_rate is not None
            else DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_RATE,
            decay_type=decay_type,
            decay_factor=params.decay_factor
            if params.decay_factor is not None
            else DEFAULT_DYNAMIC_LEARNING_RATE_DECAY_FACTOR,
            step_size=params.step_size
            if params.step_size is not None
            else DEFAULT_DYNAMIC_LEARNING_RATE_STEP_SIZE,
            max_steps=params.max_steps
            if params.max_steps is not None
            else DEFAULT_DYNAMIC_LEARNING_RATE_MAX_STEPS,
            power=params.power if params.power is not None else DEFAULT_DYNAMIC_LEARNING_RATE_POWER,
            min_lr=params.min_lr
            if params.min_lr is not None
            else DEFAULT_DYNAMIC_LEARNING_RATE_MIN_LR,
        )
    if method == LearningRateMethod.ADAM:
        return AdamLearningRate(
            initial_learning_rate=params.initial_learning_rate
            if params.initial_learning_rate is not None
            else DEFAULT_ADAM_LEARNING_RATE_INITIAL,
            beta1=params.beta1 if params.beta1 is not None else DEFAULT_ADAM_LEARNING_RATE_BETA1,
            beta2=params.beta2 if params.beta2 is not None else DEFAULT_ADAM_LEARNING_RATE_BETA2,
            epsilon=params.epsilon
            if params.epsilon is not None
            else DEFAULT_ADAM_LEARNING_RATE_EPSILON,
        )
    if method == LearningRateMethod.PERFORMANCE_BASED:
        return PerformanceBasedLearningRate(
            initial_learning_rate=params.initial_learning_rate
            if params.initial_learning_rate is not None
            else DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_INITIAL,
            min_learning_rate=params.min_learning_rate
            if params.min_learning_rate is not None
            else DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_MIN,
            max_learning_rate=params.max_learning_rate
            if params.max_learning_rate is not None
            else DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_MAX,
            adjustment_factor=params.adjustment_factor
            if params.adjustment_factor is not None
            else DEFAULT_PERFORMANCE_BASED_LEARNING_RATE_ADJUSTMENT_FACTOR,
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
    grad_cfg = config.gradient
    if grad_cfg and grad_cfg.method is not None:
        return grad_cfg.method
    return gradient_method


def configure_superposition_mode(config: SimulationConfig) -> SuperpositionModeConfig:
    """
    Configure the superposition mode based on the provided configuration.

    Args:
        config (SimulationConfig): Simulation configuration object.

    Returns
    -------
        SuperpositionModeConfig: The configured superposition mode object.
    """
    sp_cfg = config.superposition_mode or SuperpositionModeConfig()
    return SuperpositionModeConfig(
        max_superpositions=sp_cfg.max_superpositions,
        max_columns=sp_cfg.max_columns,
        render_sleep_seconds=sp_cfg.render_sleep_seconds,
        top_n_actions=sp_cfg.top_n_actions,
        top_n_randomize=sp_cfg.top_n_randomize,
    )
