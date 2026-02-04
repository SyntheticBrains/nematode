"""Factory for creating brain instances from configuration.

Consolidates brain instantiation logic used across entrypoint scripts
(run_simulation.py, run_evolution.py) into a single reusable module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantumnematode.brain.arch import (
    Brain,
    MLPDQNBrainConfig,
    MLPPPOBrainConfig,
    MLPReinforceBrainConfig,
    QQLearningBrainConfig,
    QRCBrainConfig,
    QVarCircuitBrainConfig,
    SpikingReinforceBrainConfig,
)
from quantumnematode.brain.arch.dtypes import BrainType, DeviceType
from quantumnematode.logging_config import logger
from quantumnematode.optimizers.gradient_methods import GradientCalculationMethod  # noqa: TC001
from quantumnematode.optimizers.learning_rate import (
    AdamLearningRate,
    ConstantLearningRate,
    DynamicLearningRate,
    PerformanceBasedLearningRate,
)
from quantumnematode.utils.config_loader import (
    ParameterInitializerConfig,
    create_parameter_initializer_instance,
)

if TYPE_CHECKING:
    from qiskit_serverless.core.function import RunnableQiskitFunction


def setup_brain_model(  # noqa: C901, PLR0912, PLR0913, PLR0915
    brain_type: BrainType,
    brain_config: QVarCircuitBrainConfig
    | MLPReinforceBrainConfig
    | MLPPPOBrainConfig
    | MLPDQNBrainConfig
    | QQLearningBrainConfig
    | QRCBrainConfig
    | SpikingReinforceBrainConfig,
    shots: int,
    qubits: int,  # noqa: ARG001
    device: DeviceType,
    learning_rate: ConstantLearningRate
    | DynamicLearningRate
    | AdamLearningRate
    | PerformanceBasedLearningRate,
    gradient_method: GradientCalculationMethod,
    gradient_max_norm: float | None,
    parameter_initializer_config: ParameterInitializerConfig,
    perf_mgmt: RunnableQiskitFunction | None = None,
) -> Brain:
    """Set up the brain model based on the specified brain type.

    Args:
        brain_type: The type of brain architecture to use.
        brain_config: Configuration for the brain architecture.
        shots: The number of shots for quantum circuit execution.
        qubits: The number of qubits (only for quantum brain architectures).
        device: The device to use for simulation.
        learning_rate: The learning rate configuration for the brain.
        gradient_method: The gradient calculation method.
        gradient_max_norm: Maximum gradient norm for clipping.
        parameter_initializer_config: Configuration for parameter initialization.
        perf_mgmt: Q-CTRL performance management function instance.

    Returns
    -------
        Brain: An instance of the selected brain model.

    Raises
    ------
        ValueError: If an unknown brain type is provided.
    """
    if brain_type in (BrainType.QVARCIRCUIT, BrainType.MODULAR):
        if not isinstance(brain_config, QVarCircuitBrainConfig):
            error_message = (
                "The 'qvarcircuit' brain architecture requires a QVarCircuitBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        if not isinstance(learning_rate, (DynamicLearningRate, ConstantLearningRate)):
            error_message = (
                "The 'qvarcircuit' brain architecture requires a "
                "DynamicLearningRate or ConstantLearningRate. "
                f"Provided learning rate type: {type(learning_rate)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        from quantumnematode.brain.arch.qvarcircuit import QVarCircuitBrain

        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = QVarCircuitBrain(
            config=brain_config,
            device=device,
            shots=shots,
            learning_rate=learning_rate,
            parameter_initializer=parameter_initializer,
            gradient_method=gradient_method,
            gradient_max_norm=gradient_max_norm,
            perf_mgmt=perf_mgmt,
        )
    elif brain_type in (BrainType.QQLEARNING, BrainType.QMODULAR):
        if not isinstance(brain_config, QQLearningBrainConfig):
            error_message = (
                "The 'qqlearning' brain architecture requires a QQLearningBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        if not isinstance(learning_rate, DynamicLearningRate):
            error_message = (
                "The 'qqlearning' brain architecture requires a DynamicLearningRate. "
                f"Provided learning rate type: {type(learning_rate)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        from quantumnematode.brain.arch.qqlearning import QQLearningBrain

        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = QQLearningBrain(
            config=brain_config,
            device=device,
            shots=shots,
            learning_rate=learning_rate,
            parameter_initializer=parameter_initializer,
        )

    elif brain_type in (BrainType.MLP_REINFORCE, BrainType.MLP):
        from quantumnematode.brain.arch.mlpreinforce import MLPReinforceBrain

        if not isinstance(brain_config, MLPReinforceBrainConfig):
            error_message = (
                "The 'mlpreinforce' brain architecture requires an MLPReinforceBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = MLPReinforceBrain(
            config=brain_config,
            input_dim=2,
            num_actions=4,
            lr_scheduler=True,
            device=device,
            parameter_initializer=parameter_initializer,
        )
    elif brain_type in (BrainType.MLP_PPO, BrainType.PPO):
        from quantumnematode.brain.arch.mlpppo import MLPPPOBrain

        if not isinstance(brain_config, MLPPPOBrainConfig):
            error_message = (
                "The 'mlpppo' brain architecture requires a MLPPPOBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = MLPPPOBrain(
            config=brain_config,
            input_dim=2,
            num_actions=4,
            device=device,
            parameter_initializer=parameter_initializer,
        )
    elif brain_type in (BrainType.MLP_DQN, BrainType.QMLP):
        from quantumnematode.brain.arch.mlpdqn import MLPDQNBrain

        if not isinstance(brain_config, MLPDQNBrainConfig):
            error_message = (
                "The 'mlpdqn' brain architecture requires a MLPDQNBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        parameter_initializer = create_parameter_initializer_instance(parameter_initializer_config)

        brain = MLPDQNBrain(
            config=brain_config,
            input_dim=2,
            num_actions=4,
            device=device,
            parameter_initializer=parameter_initializer,
        )
    elif brain_type in (BrainType.SPIKING_REINFORCE, BrainType.SPIKING):
        from quantumnematode.brain.arch.spikingreinforce import SpikingReinforceBrain

        if not isinstance(brain_config, SpikingReinforceBrainConfig):
            error_message = (
                "The 'spikingreinforce' brain architecture requires a SpikingReinforceBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        input_dim = 4 if brain_config.use_separated_gradients else 2

        brain = SpikingReinforceBrain(
            config=brain_config,
            input_dim=input_dim,
            num_actions=4,
            device=device,
        )
    elif brain_type == BrainType.QRC:
        from quantumnematode.brain.arch.qrc import QRCBrain

        if not isinstance(brain_config, QRCBrainConfig):
            error_message = (
                "The 'qrc' brain architecture requires a QRCBrainConfig. "
                f"Provided brain config type: {type(brain_config)}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        brain = QRCBrain(
            config=brain_config,
            num_actions=4,
            device=device,
        )
    else:
        error_message = f"Unknown brain type: {brain_type}"
        logger.error(error_message)
        raise ValueError(error_message)

    return brain
