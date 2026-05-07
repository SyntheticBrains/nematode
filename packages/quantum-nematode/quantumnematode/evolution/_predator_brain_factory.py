"""Predator-side brain factory for the evolution loop.

Mirrors `evolution.brain_factory.instantiate_brain_from_sim_config` but for
the predator stack. Required because `_ClassicalPPOEncoder.decode` calls
`instantiate_brain_from_sim_config`, which only knows the agent-side
`setup_brain_model` dispatch (19 registered agent brains, no predator brain).
A predator-side encoder must call THIS factory to construct an
`MLPPPOPredatorBrain` from `sim_config.environment.predators.brain_config`.

Why a separate factory rather than registering the predator brain in
`setup_brain_model`?

- `setup_brain_model` consumes `BrainConfig` (sensory_modules, gradient_method,
  parameter_initializer_config, etc.) — none of which apply to the predator
  brain. Forcing `MLPPPOPredatorBrain` through that dispatch would either
  bloat its constructor surface or require null-shaped agent fields.
- Predator config lives at `sim_config.environment.predators.brain_config`,
  NOT `sim_config.brain` (which is the AGENT brain). Re-using the agent
  factory would require passing the wrong sub-tree of the config.
- Keeping the two factories separate makes the agent-side / predator-side
  boundary explicit at module-import time — cross-side leaks fail loudly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain
from quantumnematode.utils.seeding import set_global_seed

if TYPE_CHECKING:
    from quantumnematode.utils.config_loader import SimulationConfig


def instantiate_predator_brain_from_sim_config(
    sim_config: SimulationConfig,
    *,
    seed: int | None = None,
) -> MLPPPOPredatorBrain:
    """Build a fresh `MLPPPOPredatorBrain` for evolution from a sim config.

    Reads predator brain config from
    ``sim_config.environment.predators.brain_config`` (a Pydantic
    `PredatorBrainConfigSchema`). Honours the architectural ``extra``
    overrides — `actor_hidden_dim`, `critic_hidden_dim`,
    `num_hidden_layers` — so a YAML-tuned predator brain shape flows
    through the encoder unchanged. The genome encoder overwrites weights
    via `WeightPersistence` after construction so the brain's
    orthogonal-init is throwaway.

    The ``extra["sample"]`` flag is intentionally IGNORED here and the
    constructor pins ``sample=False``: this factory is exclusively used
    by the genome encoder, which decodes a genome's weights into a
    fresh brain that the fitness function then evaluates with argmax
    actions (no sampling — frozen-weight per design.md D13). Sampling-
    mode is reachable via the env-side dispatcher
    (``DynamicForagingEnvironment._build_predator_brain``) for
    standalone scenarios and via the pretrain helper for exploration
    noise during behavioural cloning, but neither path passes through
    this factory.

    Parameters
    ----------
    sim_config
        Parsed simulation config. Must have
        ``sim_config.environment.predators.brain_config`` set with
        ``kind == "mlpppo_predator"``.
    seed
        Per-evaluation seed. When provided, seeds numpy + torch globals
        via :func:`set_global_seed` BEFORE constructing the brain (so
        orthogonal-init is reproducible) AND is passed through to
        `MLPPPOPredatorBrain.__init__` as its torch-RNG seed for the
        same reason. The two paths are belt-and-braces: the brain seeds
        torch internally, and `set_global_seed` covers numpy globals
        used by any future code path (e.g. dropout, future RNG-driven
        init schemes) that consults the global generator before the
        brain's local seed takes effect.
        When None, the brain is constructed with default (non-deterministic)
        orthogonal init.

    Returns
    -------
    MLPPPOPredatorBrain
        Fresh brain instance. Weights are about to be overwritten by
        the encoder's `load_weight_components` round-trip; the seed
        controls only the throwaway pre-load init values.

    Raises
    ------
    ValueError
        If the sim_config lacks an environment block, predator block,
        or predator brain_config; or if `kind` is not the supported
        `"mlpppo_predator"` (the heuristic kind is not learnable and
        has no encoder counterpart).
    """
    if sim_config.environment is None:
        msg = (
            "instantiate_predator_brain_from_sim_config requires sim_config.environment to be set."
        )
        raise ValueError(msg)
    predators = sim_config.environment.predators
    if predators is None or predators.brain_config is None:
        msg = (
            "instantiate_predator_brain_from_sim_config requires "
            "sim_config.environment.predators.brain_config to be set with "
            "kind='mlpppo_predator'."
        )
        raise ValueError(msg)
    brain_config = predators.brain_config
    if brain_config.kind != "mlpppo_predator":
        msg = (
            f"instantiate_predator_brain_from_sim_config only supports "
            f"kind='mlpppo_predator', got kind={brain_config.kind!r}. "
            "Heuristic predators are not learnable and have no encoder."
        )
        raise ValueError(msg)

    if seed is not None:
        # Seed numpy + torch globals so any global-RNG-consuming code
        # before the brain's local torch.manual_seed call (e.g. future
        # init schemes or dropout) is reproducible. The brain itself
        # also seeds torch internally via its `seed=` kwarg below.
        set_global_seed(seed)

    extra = brain_config.extra or {}
    return MLPPPOPredatorBrain(
        actor_hidden_dim=int(extra.get("actor_hidden_dim", 64)),
        critic_hidden_dim=int(extra.get("critic_hidden_dim", 64)),
        num_hidden_layers=int(extra.get("num_hidden_layers", 2)),
        seed=seed,
        # Frozen-weight evaluation in fitness uses argmax (sample=False)
        # by default. Pretrain helper sets sample=True directly on its
        # brain instance; encoders never need the sampled path because
        # weights, not actions, are what flow through the genome.
        sample=False,
    )
