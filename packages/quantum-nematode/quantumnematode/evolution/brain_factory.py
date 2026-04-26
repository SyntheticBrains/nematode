"""Thin wrapper around ``utils.brain_factory.setup_brain_model`` for evolution.

The wrapper centralises evolution-specific concerns: patching ``BrainConfig.seed``
with the per-evaluation seed, forcing ``BrainConfig.weights_path = None`` (the
genome is the sole weight source for evolution), hardcoding ``device=DeviceType.CPU``
for fitness eval, and converting config-shaped fields to runtime objects via the
existing ``configure_*`` helpers in :mod:`quantumnematode.utils.config_loader`.

Encoders call this — they do not call ``setup_brain_model`` directly.

See ``Decision 0`` and ``Decision 3a`` in the change's ``design.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantumnematode.brain.arch.dtypes import (
    DEFAULT_QUBITS,
    DEFAULT_SHOTS,
    BrainType,
    DeviceType,
)
from quantumnematode.optimizers.gradient_methods import GradientCalculationMethod
from quantumnematode.utils.brain_factory import setup_brain_model
from quantumnematode.utils.config_loader import (
    configure_brain,
    configure_gradient_method,
    configure_learning_rate,
    configure_parameter_initializer,
)

if TYPE_CHECKING:
    from quantumnematode.brain.arch._brain import Brain
    from quantumnematode.utils.config_loader import SimulationConfig


def instantiate_brain_from_sim_config(
    sim_config: SimulationConfig,
    *,
    seed: int | None = None,
) -> Brain:
    """Build a fresh brain for evolution from a simulation config.

    Single source of truth for how the evolution framework constructs brains.
    Patches ``BrainConfig.seed`` with the per-evaluation ``seed`` (so the brain's
    ``__init__`` calls ``set_global_seed(seed)`` and ``self.rng = get_rng(seed)``,
    seeding all three RNG sources — global numpy, global torch, brain-local
    Generator), forces ``BrainConfig.weights_path = None`` (the genome is the
    sole weight source), and dispatches to ``setup_brain_model``.

    Parameters
    ----------
    sim_config
        Parsed simulation config.  Must have ``brain.name`` set.
    seed
        Per-evaluation seed.  When provided, overrides the YAML-configured
        ``BrainConfig.seed``.  When None, the brain's seed comes from
        ``sim_config.brain.config.seed`` (or auto-generated if that is None too).

    Returns
    -------
    Brain
        Fresh brain instance with weights initialised by the brain's own
        constructor (genome will be applied later via ``load_weight_components``).
    """
    if sim_config.brain is None or sim_config.brain.name is None:
        msg = "instantiate_brain_from_sim_config requires sim_config.brain.name to be set."
        raise ValueError(msg)

    # Patch BrainConfig.seed (NOT SimulationConfig.seed) and force weights_path=None.
    # The brain reads from BrainConfig.seed, matching the canonical pattern in
    # scripts/run_simulation.py.
    brain_config = configure_brain(sim_config)
    overrides: dict[str, object] = {"weights_path": None}
    if seed is not None:
        overrides["seed"] = seed
    brain_config = brain_config.model_copy(update=overrides)

    # Convert config-shaped fields to runtime objects via existing helpers.
    # configure_gradient_method takes a default + sim_config and returns a
    # (method, max_norm) tuple — matching the canonical pattern in run_simulation.py.
    # GradientCalculationMethod.RAW is a no-op default; classical brains in
    # evolution don't actually use gradient methods (no .learn() in fitness eval),
    # so any sensible default works.  The helper extracts any user-configured
    # max_norm from sim_config.gradient.
    learning_rate = configure_learning_rate(sim_config)
    gradient_method, gradient_max_norm = configure_gradient_method(
        GradientCalculationMethod.RAW,
        sim_config,
    )
    parameter_initializer_config = configure_parameter_initializer(sim_config)

    return setup_brain_model(
        brain_type=BrainType(sim_config.brain.name),
        brain_config=brain_config,
        shots=sim_config.shots or DEFAULT_SHOTS,
        qubits=sim_config.qubits or DEFAULT_QUBITS,
        device=DeviceType.CPU,
        learning_rate=learning_rate,
        gradient_method=gradient_method,
        gradient_max_norm=gradient_max_norm,
        parameter_initializer_config=parameter_initializer_config,
    )
