"""Derive every step-coupled constant from one configured step duration.

A config's per-step values are written for its *reference step*: the duration of a full-speed
step of its ``max_step_mm`` at the crawl speed, 5 worm-seconds at 1 mm. Setting the top-level
``step_seconds`` converts each of them from the reference step to the configured one, by the rule
for its kind, so that the quantity it stands for keeps its value per worm-second. With ``k`` the
configured step over the reference step and ``r = 1/k``, a whole number:

- the full-speed step becomes the configured step times the crawl speed;
- an amount per step scales by ``k`` (turn, satiety decay, regrowth, per-step reward terms, the
  modulator step), keeping its rate per second;
- a fraction removed per step becomes ``1 - (1 - p)^k`` (lawn intake, the adaptive sensor's
  smoothing), so ``r`` short steps remove what one reference step did;
- a per-step discount becomes ``d^k`` (``gamma``, ``gae_lambda``), keeping the discount over a
  span of worm time;
- the STAM decay rate scales by ``k``, so a sample's weight is the same function of its age in
  seconds;
- a length in steps scales by ``r`` (``max_steps``, the STAM buffer, the stuck threshold), as does
  a membrane time constant in steps under leaky dynamics;
- the derivative scale scales by ``r``: the rate-of-change feature is a per-step difference, so a
  shorter step shrinks it by ``k``, and the scale keeps it a change per reference step.

The conversion covers one surface: the continuous point worm on lawns, driven by ``connectomeppo``
or ``mlpppo`` without within-lifetime plasticity. Everything outside it has per-step constants with
no rule here, so a non-reference step is refused there rather than run with values half converted.

Unset, nothing is derived and the config is returned as it is. A derived config's reference step
is its configured step, so deriving it again changes nothing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quantumnematode.env.worm_time import CRAWL_SPEED_MM_PER_S, step_worm_seconds

if TYPE_CHECKING:
    from pydantic import BaseModel

    from quantumnematode.utils.config_loader import SimulationConfig

logger = logging.getLogger(__name__)

# The largest whole ratio of reference step to configured step accepted.
MAX_STEP_RATIO = 100
# How far the ratio may sit from a whole number and still count as one.
_RATIO_TOLERANCE = 1e-9

# Brains whose step-coupled settings have a rule here.
STEP_DERIVED_BRAINS = frozenset({"connectomeppo", "mlpppo"})

# Reward terms that shape a state, or whose meaning depends on the step itself ("where it was two
# steps before"); the lawn model already requires each to be zero, and no rule converts them.
_UNCONVERTED_REWARD_TERMS = (
    "penalty_stuck_position",
    "penalty_anti_dithering",
    "reward_exploration",
    "reward_distance_scale",
)


@dataclass(frozen=True)
class DerivedConstant:
    """One converted constant: where it lives, as written, and as derived."""

    path: str
    written: float
    derived: float


@dataclass(frozen=True)
class StepDerivation:
    """What a derivation did: the configured and reference steps, their ratio, every conversion."""

    step_seconds: float
    reference_seconds: float
    ratio: int
    constants: tuple[DerivedConstant, ...]

    def lines(self) -> list[str]:
        """Return the record: a ``Step:`` line, then one line per derived constant."""
        head = (
            f"Step: {self.step_seconds:g} s (reference {self.reference_seconds:g} s, x{self.ratio})"
        )
        return [head, *(f"  {c.path}: {c.written!r} -> {c.derived!r}" for c in self.constants)]


def amount_per_step(value: float, k: float) -> float:
    """Return an amount per step at a step ``k`` times the reference step."""
    return value * k


def fraction_per_step(fraction: float, k: float) -> float:
    """Return the per-step fraction that, compounded, removes ``fraction`` per reference step."""
    return 1.0 - (1.0 - fraction) ** k


def discount_per_step(discount: float, k: float) -> float:
    """Return the per-step discount that, compounded, discounts ``discount`` per reference step."""
    return discount**k


def step_ratio(config: SimulationConfig) -> int | None:
    """Return the whole ratio of reference step to configured step, or None with no step configured.

    Raises
    ------
    ValueError
        If a step is configured on the grid, with a zero full-speed step, or at a ratio that is not
        a whole number from 1 to ``MAX_STEP_RATIO``.
    """
    if config.step_seconds is None:
        return None
    environment = config.environment
    if environment is None or environment.env_type != "continuous_2d":
        msg = (
            "step_seconds is set but the substrate is not continuous_2d: the grid has no step "
            "duration to configure."
        )
        raise ValueError(msg)
    max_step_mm = environment.get_continuous_config().max_step_mm
    if max_step_mm <= 0.0:
        msg = (
            "step_seconds is set but continuous.max_step_mm is 0, so the config's reference step, "
            "the duration of its full-speed step, is undefined."
        )
        raise ValueError(msg)
    reference = step_worm_seconds(max_step_mm)
    quotient = reference / config.step_seconds
    ratio = round(quotient)
    if not 1 <= ratio <= MAX_STEP_RATIO or abs(quotient - ratio) > _RATIO_TOLERANCE * ratio:
        msg = (
            f"step_seconds {config.step_seconds:g} does not divide the config's {reference:g}-s "
            f"reference step (max_step_mm {max_step_mm:g} at the crawl speed) into a whole number "
            f"of steps from 1 to {MAX_STEP_RATIO}: the ratio is {quotient:g}."
        )
        raise ValueError(msg)
    return ratio


def _refuse_outside_surface(config: SimulationConfig) -> None:  # noqa: C901, PLR0912
    """Refuse a non-reference step wherever a step-coupled constant has no rule.

    Raises
    ------
    ValueError
        Naming the first setting outside the converted surface, and why.
    """
    environment = config.environment
    assert environment is not None  # step_ratio has refused it otherwise  # noqa: S101
    scope = (
        "a non-reference step_seconds is converted only for the continuous point worm on lawns, "
        f"driven by {sorted(STEP_DERIVED_BRAINS)} without within-lifetime plasticity"
    )

    def refuse(what: str) -> None:
        msg = f"step_seconds is set and {what}: {scope}."
        raise ValueError(msg)

    foraging = environment.foraging
    if foraging is None or foraging.food_model != "lawns":
        refuse("the food model is not lawns")
    for name in ("thermotaxis", "aerotaxis", "predators", "pheromones", "social_feeding"):
        block = getattr(environment, name)
        if block is not None and block.enabled:
            refuse(f"{name} is enabled, and its per-step constants have no rule")
    if config.multi_agent is not None and config.multi_agent.enabled:
        refuse("this is a multi-agent run")
    for name in ("bit_memory_task", "associative_memory_task"):
        task = getattr(environment, name)
        if task is not None and task.enabled:
            refuse(f"{name} is enabled, and its phases are counted in steps")
    if environment.get_continuous_config().body_model == "kinematic":
        refuse("the body model is kinematic, whose reversal and refractory counts have no rule")
    brain = config.brain
    if brain is None or brain.name not in STEP_DERIVED_BRAINS:
        refuse(f"the brain is {brain.name if brain else None!r}")
    assert brain is not None  # noqa: S101
    if getattr(brain.config, "learning_rule", "ppo") != "ppo" or getattr(
        brain.config,
        "enable_activity_traces",
        False,
    ):
        refuse("within-lifetime plasticity is on, and its trace decay and rate are per step")
    if config.hyperparam_schema is not None:
        refuse(
            "hyperparam_schema is set, and hyperparameter evolution rewrites brain values after "
            "load, at the configured step rather than the reference",
        )
    if config.max_steps is None:
        refuse("max_steps is unset, so the episode's length cannot be converted")
    if config.reward is not None:
        for key in _UNCONVERTED_REWARD_TERMS:
            if getattr(config.reward, key) != 0:
                refuse(f"reward.{key} is {getattr(config.reward, key)}, and has no rule")


def _rebuilt[M: BaseModel](model: M, updates: dict[str, Any]) -> M:
    """Return ``model`` with ``updates`` applied, re-validated so every field bound still holds."""
    return type(model).model_validate({**dict(model), **updates})


def derive_step_coupled(config: SimulationConfig) -> SimulationConfig:
    """Return ``config`` with every step-coupled constant converted to its configured step.

    A config without ``step_seconds``, or whose step equals its reference step, is returned
    unchanged. The returned config records the conversion as ``config.step_derivation``.

    Raises
    ------
    ValueError
        If the step cannot be configured here (see ``step_ratio``), or the config reaches outside
        the converted surface.
    """
    # Imported here: config_loader imports this module.
    from quantumnematode.agent.agent import RewardConfig, SatietyConfig
    from quantumnematode.utils.config_loader import BRAIN_CONFIG_MAP, _resolve_brain_config

    ratio = step_ratio(config)
    if ratio is None or ratio == 1:
        return config
    _refuse_outside_surface(config)
    step = config.step_seconds
    assert step is not None  # noqa: S101
    k = 1.0 / ratio
    constants: list[DerivedConstant] = []

    def convert(path: str, written: float, derived: float) -> float:
        constants.append(DerivedConstant(path=path, written=written, derived=derived))
        return derived

    environment = config.environment
    assert environment is not None  # noqa: S101
    continuous = environment.get_continuous_config()
    reference = step_worm_seconds(continuous.max_step_mm)
    continuous = _rebuilt(
        continuous,
        {
            "max_step_mm": convert(
                "environment.continuous.max_step_mm",
                continuous.max_step_mm,
                step * CRAWL_SPEED_MM_PER_S,
            ),
            "max_turn_rad": convert(
                "environment.continuous.max_turn_rad",
                continuous.max_turn_rad,
                amount_per_step(continuous.max_turn_rad, k),
            ),
            "step_seconds": step,
        },
    )

    foraging = environment.foraging
    assert foraging is not None  # noqa: S101
    assert foraging.lawns is not None  # noqa: S101
    lawns = foraging.lawns
    lawns = _rebuilt(
        lawns,
        {
            "intake_fraction": convert(
                "environment.foraging.lawns.intake_fraction",
                lawns.intake_fraction,
                fraction_per_step(lawns.intake_fraction, k),
            ),
            "regrowth_per_step": convert(
                "environment.foraging.lawns.regrowth_per_step",
                lawns.regrowth_per_step,
                amount_per_step(lawns.regrowth_per_step, k),
            ),
        },
    )
    foraging = _rebuilt(foraging, {"lawns": lawns})

    sensing = environment.get_sensing_config()
    sensing = _rebuilt(
        sensing,
        {
            "stam_buffer_size": int(
                convert(
                    "environment.sensing.stam_buffer_size",
                    sensing.stam_buffer_size,
                    sensing.stam_buffer_size * ratio,
                ),
            ),
            "stam_decay_rate": convert(
                "environment.sensing.stam_decay_rate",
                sensing.stam_decay_rate,
                amount_per_step(sensing.stam_decay_rate, k),
            ),
            "adaptive_chemosensor_alpha": convert(
                "environment.sensing.adaptive_chemosensor_alpha",
                sensing.adaptive_chemosensor_alpha,
                fraction_per_step(sensing.adaptive_chemosensor_alpha, k),
            ),
            "derivative_scale": convert(
                "environment.sensing.derivative_scale",
                sensing.derivative_scale,
                sensing.derivative_scale * ratio,
            ),
        },
    )
    environment = _rebuilt(
        environment,
        {"continuous": continuous, "foraging": foraging, "sensing": sensing},
    )

    satiety = config.satiety or SatietyConfig()
    satiety = _rebuilt(
        satiety,
        {
            "satiety_decay_rate": convert(
                "satiety.satiety_decay_rate",
                satiety.satiety_decay_rate,
                amount_per_step(satiety.satiety_decay_rate, k),
            ),
        },
    )

    reward = config.reward or RewardConfig()
    reward = _rebuilt(
        reward,
        {
            "penalty_step": convert(
                "reward.penalty_step",
                reward.penalty_step,
                amount_per_step(reward.penalty_step, k),
            ),
            "penalty_boundary_collision": convert(
                "reward.penalty_boundary_collision",
                reward.penalty_boundary_collision,
                amount_per_step(reward.penalty_boundary_collision, k),
            ),
            "stuck_position_threshold": int(
                convert(
                    "reward.stuck_position_threshold",
                    reward.stuck_position_threshold,
                    reward.stuck_position_threshold * ratio,
                ),
            ),
        },
    )

    brain = config.brain
    assert brain is not None  # noqa: S101
    brain_config: Any = _resolve_brain_config(
        brain.config,
        BRAIN_CONFIG_MAP[brain.name],
        brain.name,
    )
    brain_updates: dict[str, Any] = {
        "gamma": convert(
            "brain.config.gamma",
            brain_config.gamma,
            discount_per_step(brain_config.gamma, k),
        ),
        "gae_lambda": convert(
            "brain.config.gae_lambda",
            brain_config.gae_lambda,
            discount_per_step(brain_config.gae_lambda, k),
        ),
    }
    if brain.name == "connectomeppo":
        brain_updates["modulator_step_seconds"] = convert(
            "brain.config.modulator_step_seconds",
            brain_config.modulator_step_seconds,
            amount_per_step(brain_config.modulator_step_seconds, k),
        )
        if brain_config.dynamics == "leaky":
            brain_updates["membrane_tau_steps"] = convert(
                "brain.config.membrane_tau_steps",
                brain_config.membrane_tau_steps,
                brain_config.membrane_tau_steps * ratio,
            )
    brain = _rebuilt(brain, {"config": _rebuilt(brain_config, brain_updates)})

    assert config.max_steps is not None  # noqa: S101
    max_steps = int(convert("max_steps", config.max_steps, config.max_steps * ratio))

    derived = _rebuilt(
        config,
        {
            "environment": environment,
            "satiety": satiety,
            "reward": reward,
            "brain": brain,
            "max_steps": max_steps,
        },
    )
    derived._step_derivation = StepDerivation(  # noqa: SLF001
        step_seconds=step,
        reference_seconds=reference,
        ratio=ratio,
        constants=tuple(constants),
    )
    for constant in constants:
        logger.info(
            "step_seconds %g: %s %r -> %r",
            step,
            constant.path,
            constant.written,
            constant.derived,
        )
    if not math.isclose(step_worm_seconds(continuous.max_step_mm), step, rel_tol=1e-12):
        msg = f"derived max_step_mm {continuous.max_step_mm} does not give the {step:g}-s step"
        raise AssertionError(msg)
    return derived
