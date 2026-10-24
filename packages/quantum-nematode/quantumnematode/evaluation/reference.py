"""The per-step runner's record of one episode, for comparison with the batched evaluator.

One episode of a config under a controller, seeded as an evaluation episode is seeded: the
environment built at the seed, and the global generators set to it just before the episode runs.
Each step logs:

- what the controller saw: the brain's own feature vector, or nothing for a hazard rule;
- the action's noiseless mean;
- the modulators the step started with;
- the fold-change derivative;
- the trajectory, as the behaviour capture records it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from quantumnematode.agent import QuantumNematodeAgent
from quantumnematode.env.theme import Theme
from quantumnematode.utils.config_loader import (
    configure_environment,
    configure_reward,
    configure_satiety,
    create_env_from_config,
)
from quantumnematode.utils.seeding import set_global_seed

if TYPE_CHECKING:
    from quantumnematode.report.dtypes import BehaviourStep
    from quantumnematode.utils.config_loader import SimulationConfig


@dataclass
class ReferenceEpisode:
    """One per-step episode's per-step log."""

    features: list[np.ndarray] = field(default_factory=list)
    means: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    modulators: list[np.ndarray] = field(default_factory=list)
    derivatives: list[float] = field(default_factory=list)
    densities: list[np.ndarray] = field(default_factory=list)
    behaviour: list[BehaviourStep] = field(default_factory=list)
    final_satiety: float = 0.0


def run_reference(
    config: SimulationConfig,
    controller: Any,  # noqa: ANN401 - a brain or a hazard controller, both on the brain protocol
    seed: int,
    steps: int | None = None,
    *,
    record_density: bool = False,
) -> ReferenceEpisode:
    """Run one per-step episode of ``config`` under ``controller`` at ``seed`` and log it."""
    environment = configure_environment(config)
    sensing = environment.get_sensing_config().model_copy(update={"capture_behaviour": True})
    env = create_env_from_config(environment, seed=seed, theme=Theme.HEADLESS)
    agent = QuantumNematodeAgent(
        brain=controller,
        env=env,
        theme=Theme.HEADLESS,
        satiety_config=configure_satiety(config),
        sensing_config=sensing,
    )
    log = ReferenceEpisode()
    act = controller.run_brain
    preprocess = getattr(controller, "preprocess", None)

    def run_brain(params: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if preprocess is not None:
            log.features.append(np.asarray(preprocess(params), dtype=np.float64))
        concentrations = getattr(controller, "modulator_concentrations", None)
        if concentrations is not None:
            log.modulators.append(np.asarray(concentrations, dtype=np.float64))
        log.derivatives.append(float(params.food_dconcentration_dt or 0.0))
        if record_density:
            field_ = agent.env.lawn_field
            log.densities.append(np.zeros(0) if field_ is None else field_.density.copy())
        actions = act(params, *args, **kwargs)
        action = actions[0]
        log.actions.append(np.asarray(action.continuous, dtype=np.float64))
        mean = action.continuous_mean if action.continuous_mean is not None else action.continuous
        log.means.append(np.asarray(mean, dtype=np.float64))
        return actions

    controller.run_brain = run_brain
    try:
        set_global_seed(seed)
        horizon = steps if steps is not None else config.max_steps
        if horizon is None:
            msg = "the reference episode needs steps or the config's max_steps"
            raise ValueError(msg)
        agent.run_episode(configure_reward(config), max_steps=horizon)
    finally:
        controller.run_brain = act
    log.behaviour = list(agent.behaviour)
    log.final_satiety = float(agent.current_satiety)
    return log
