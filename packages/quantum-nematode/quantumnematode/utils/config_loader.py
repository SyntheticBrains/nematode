"""Load and configure simulation settings from a YAML file."""

from pathlib import Path

import yaml

from quantumnematode.logging_config import (
    logger,
)
from quantumnematode.optimizer.gradient_methods import GradientCalculationMethod
from quantumnematode.optimizer.learning_rate import (
    AdamLearningRate,
    DecayType,
    DynamicLearningRate,
    PerformanceBasedLearningRate,
)


def load_simulation_config(config_path: str) -> dict:
    """
    Load simulation configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns
    -------
        dict: Parsed configuration as a dictionary.
    """
    with Path(config_path).open() as file:
        return yaml.safe_load(file)


def configure_learning_rate(
    config: dict,
) -> DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate:
    """
    Configure the learning rate based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing learning rate settings.

    Returns
    -------
        DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate:
            Configured learning rate object.
    """
    learning_rate_config = config.get("learning_rate", {})

    if not learning_rate_config:
        logger.warning(
            "No learning rate configuration found. Using default DynamicLearningRate.",
        )
        return DynamicLearningRate()

    learning_rate_method = learning_rate_config.get("method", "default")
    learning_rate_parameters = learning_rate_config.get("parameters", {})
    if learning_rate_method == "dynamic":
        decay_type = _resolve_decay_type(learning_rate_parameters)
        return DynamicLearningRate(
            initial_learning_rate=learning_rate_parameters.get("initial_learning_rate", 0.1),
            decay_rate=learning_rate_parameters.get("decay_rate", 0.01),
            decay_type=decay_type,
            decay_factor=learning_rate_parameters.get("decay_factor", 0.5),
            step_size=learning_rate_parameters.get("step_size", 10),
            max_steps=learning_rate_parameters.get("max_steps", 1000),
            power=learning_rate_parameters.get("power", 1.0),
            min_lr=learning_rate_parameters.get("min_lr", 0.0001),
        )
    if learning_rate_method == "adam":
        return AdamLearningRate(
            initial_learning_rate=learning_rate_parameters.get("initial_learning_rate", 0.1),
            beta1=learning_rate_parameters.get("beta1", 0.9),
            beta2=learning_rate_parameters.get("beta2", 0.999),
            epsilon=learning_rate_parameters.get("epsilon", 1e-8),
        )
    if learning_rate_method == "performance_based":
        return PerformanceBasedLearningRate(
            initial_learning_rate=learning_rate_parameters.get("initial_learning_rate", 0.1),
            min_learning_rate=learning_rate_parameters.get("min_learning_rate", 0.001),
            max_learning_rate=learning_rate_parameters.get("max_learning_rate", 0.5),
            adjustment_factor=learning_rate_parameters.get("adjustment_factor", 1.1),
        )
    error_message = (
        f"Unknown learning rate method: {learning_rate_method}. "
        "Supported methods are 'dynamic' and 'adam'."
    )
    logger.error(error_message)
    raise ValueError(error_message)


def _resolve_decay_type(learning_rate_parameters: dict) -> DecayType:
    decay_type_value = learning_rate_parameters.get("decay_type", "inverse_time")
    if isinstance(decay_type_value, DecayType):
        decay_type = decay_type_value
    else:
        try:
            decay_type = DecayType(decay_type_value)
        except ValueError:
            logger.warning(
                f"Unknown decay_type '{decay_type_value}', defaulting to 'inverse_time'.",
            )
            decay_type = DecayType.INVERSE_TIME
    return decay_type


def configure_gradient_method(
    gradient_method: GradientCalculationMethod,
    config: dict,
) -> GradientCalculationMethod:
    """
    Configure the gradient calculation method based on the provided configuration.

    Args:
        gradient_method (GradientCalculationMethod): The default gradient calculation method.
        config (dict): Configuration dictionary containing gradient method settings.

    Returns
    -------
        GradientCalculationMethod: The configured gradient calculation method.

    Raises
    ------
        ValueError: If an invalid gradient method is specified in the configuration.
    """
    gradient_config = config.get("gradient", {})
    method_name = gradient_config.get("method", gradient_method.name).upper()
    if method_name not in GradientCalculationMethod.__members__:
        error_message = (
            f"Invalid gradient method: {method_name}. "
            f"Valid options are: {list(GradientCalculationMethod.__members__.keys())}"
        )
        logger.error(error_message)
        raise ValueError(error_message)
    return GradientCalculationMethod[method_name]
