"""Factory for creating brain instances from configuration.

Consolidates brain instantiation logic used across entrypoint scripts
(run_simulation.py, run_evolution.py) into a single reusable module.

The ``setup_brain_model`` entry point composes two responsibilities:

1. **Registry dispatch** — class lookup, config-type validation, and
   instantiation are delegated to ``instantiate_brain`` in
   :mod:`quantumnematode.brain.arch._registry`. Each architecture
   self-registers via the ``@register_brain`` decorator on its Brain
   class.
2. **Infrastructure kwargs** — each architecture accepts a slightly
   different subset of the global infrastructure context (``device``,
   ``shots``, ``learning_rate``, ``parameter_initializer``,
   ``gradient_method``, ``gradient_max_norm``, ``perf_mgmt``,
   ``num_actions``, ``input_dim``). The kwargs subset is encoded
   per-architecture in :func:`_build_infra_kwargs`; this is the one place
   where the per-arch ``__init__`` signature shape is reflected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantumnematode.brain.arch import (
    Brain,
    QVarCircuitBrainConfig,
)
from quantumnematode.brain.arch._registry import instantiate_brain
from quantumnematode.brain.arch.dtypes import BrainConfig, BrainType, DeviceType
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


_LearningRate = (
    ConstantLearningRate | DynamicLearningRate | AdamLearningRate | PerformanceBasedLearningRate
)


def _build_infra_kwargs(  # noqa: PLR0911, PLR0913
    brain_type: BrainType,
    *,
    brain_config: BrainConfig,
    shots: int,
    device: DeviceType,
    learning_rate: _LearningRate,
    gradient_method: GradientCalculationMethod,
    gradient_max_norm: float | None,
    parameter_initializer_config: ParameterInitializerConfig,
    perf_mgmt: RunnableQiskitFunction | None,
) -> dict[str, Any]:
    """Build the per-architecture infrastructure kwargs dict.

    Each architecture's ``__init__`` accepts a slightly different subset
    of the global infrastructure context. This function is the canonical
    place where those subsets are declared. Adding a new architecture
    typically means adding one branch here OR — when the new architecture
    matches the default shape — adding nothing here at all.
    """
    # Quantum-circuit architectures: validate learning-rate type, then
    # forward the full quantum-infrastructure surface.
    if brain_type is BrainType.QVARCIRCUIT:
        if not isinstance(learning_rate, (DynamicLearningRate, ConstantLearningRate)):
            msg = (
                "The 'qvarcircuit' brain architecture requires a "
                "DynamicLearningRate or ConstantLearningRate. "
                f"Provided learning rate type: {type(learning_rate)}."
            )
            logger.error(msg)
            raise ValueError(msg)
        return {
            "device": device,
            "shots": shots,
            "learning_rate": learning_rate,
            "parameter_initializer": create_parameter_initializer_instance(
                parameter_initializer_config,
            ),
            "gradient_method": gradient_method,
            "gradient_max_norm": gradient_max_norm,
            "perf_mgmt": perf_mgmt,
        }
    if brain_type is BrainType.QQLEARNING:
        if not isinstance(learning_rate, DynamicLearningRate):
            msg = (
                "The 'qqlearning' brain architecture requires a DynamicLearningRate. "
                f"Provided learning rate type: {type(learning_rate)}."
            )
            logger.error(msg)
            raise ValueError(msg)
        return {
            "device": device,
            "shots": shots,
            "learning_rate": learning_rate,
            "parameter_initializer": create_parameter_initializer_instance(
                parameter_initializer_config,
            ),
        }

    # Classical PPO / REINFORCE / DQN: parameter-initialiser plumbing.
    if brain_type is BrainType.MLP_REINFORCE:
        return {
            "input_dim": 2,
            "num_actions": 4,
            "lr_scheduler": True,
            "device": device,
            "parameter_initializer": create_parameter_initializer_instance(
                parameter_initializer_config,
            ),
        }
    if brain_type is BrainType.MLP_PPO:
        return {
            "num_actions": 4,
            "device": device,
            "parameter_initializer": create_parameter_initializer_instance(
                parameter_initializer_config,
            ),
        }
    if brain_type is BrainType.MLP_DQN:
        return {
            "input_dim": 2,
            "num_actions": 4,
            "device": device,
            "parameter_initializer": create_parameter_initializer_instance(
                parameter_initializer_config,
            ),
        }

    # Spiking REINFORCE takes an explicit ``input_dim=4``.
    if brain_type is BrainType.SPIKING_REINFORCE:
        return {"input_dim": 4, "num_actions": 4, "device": device}

    # Connectome-constrained PPO: 4-action discrete output via the
    # topology's motor readout; the action count is fixed by the readout
    # matrix shape (not configurable via ``num_actions``).
    if brain_type is BrainType.CONNECTOMEPPO:
        return {"device": device}

    # Default shape for every other architecture: 4-action discrete output
    # on the configured device. Suppresses the unused-parameter warning
    # for the kwargs the default-shape brains don't consume.
    del brain_config, shots, learning_rate, gradient_method
    del gradient_max_norm, parameter_initializer_config, perf_mgmt
    return {"num_actions": 4, "device": device}


def setup_brain_model(  # noqa: PLR0913
    brain_type: BrainType,
    brain_config: BrainConfig,
    shots: int,
    qubits: int,  # noqa: ARG001  — kept in signature for backward-compat with external callers
    device: DeviceType,
    learning_rate: _LearningRate,
    gradient_method: GradientCalculationMethod,
    gradient_max_norm: float | None,
    parameter_initializer_config: ParameterInitializerConfig,
    perf_mgmt: RunnableQiskitFunction | None = None,
) -> Brain:
    """Instantiate a brain by ``BrainType`` enum member.

    Public entry point preserved from the pre-registry signature. The
    actual dispatch is delegated to the plugin registry; this function's
    only job is mapping the global infrastructure context to the per-arch
    ``__init__`` kwargs subset.

    Parameters
    ----------
    brain_type
        The brain architecture to instantiate. Must be a member of
        :class:`BrainType` and registered via ``@register_brain``.
    brain_config
        Configuration for the brain architecture. Must be an instance of
        the architecture's registered ``config_cls``.
    shots
        The number of shots for quantum circuit execution. Consumed only
        by quantum-circuit architectures.
    qubits
        Unused at this layer (kept for backward-compat with external
        callers; the per-architecture configs carry their own qubit count
        when relevant).
    device
        The device to use for simulation.
    learning_rate
        The learning rate configuration for the brain. Only quantum-circuit
        architectures (QVARCIRCUIT, QQLEARNING) consume this at the
        infrastructure layer; classical brains read learning-rate fields
        from their own config.
    gradient_method
        The gradient calculation method. Consumed only by QVARCIRCUIT.
    gradient_max_norm
        Maximum gradient norm for clipping. Consumed only by QVARCIRCUIT.
    parameter_initializer_config
        Configuration for parameter initialisation. Consumed by the
        quantum-circuit and classical MLP families.
    perf_mgmt
        Q-CTRL performance management function instance. Consumed only by
        QVARCIRCUIT.

    Returns
    -------
    Brain
        An instance of the selected brain model.

    Raises
    ------
    ValueError
        If ``brain_type`` is unknown to the registry, or ``brain_config``
        is not an instance of the registered config class.
    """
    infra_kwargs = _build_infra_kwargs(
        brain_type,
        brain_config=brain_config,
        shots=shots,
        device=device,
        learning_rate=learning_rate,
        gradient_method=gradient_method,
        gradient_max_norm=gradient_max_norm,
        parameter_initializer_config=parameter_initializer_config,
        perf_mgmt=perf_mgmt,
    )
    return instantiate_brain(brain_type.value, brain_config, **infra_kwargs)


# Keep the qvarcircuit-config import alive so type-checkers reading legacy
# call sites that reference ``brain_factory.QVarCircuitBrainConfig`` continue
# to type-check. It is also forwarded by some scripts via this module path.
__all__ = ["QVarCircuitBrainConfig", "setup_brain_model"]
