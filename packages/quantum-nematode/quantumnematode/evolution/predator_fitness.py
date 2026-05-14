"""Predator-side fitness functions for the co-evolution loop.

Two implementations conforming to the canonical
:class:`~quantumnematode.evolution.fitness.FitnessFunction` Protocol:

- :class:`PredatorEpisodicKillRate` — primary, frozen-weight kill-rate
  measurement (no inner-loop training; the outer-loop CMA-ES owns the
  gradient on the predator's small policy space).
- :class:`PredatorLearnedPerformanceFitness` — secondary variant that
  trains the predator brain inner-loop before evaluation. NOT used in
  the main co-evolution flow, but available as a follow-up ablation
  if pilot evidence motivates it.

Both are designed to run inside the existing
:meth:`EvolutionLoop._evaluate_in_worker` 11-tuple worker, so they are
plain classes with no per-instance shared state and pickle cleanly.

The fitness function does NOT patch `sim_config` to mount frozen prey
opponents — that is the CALLER's responsibility (the
:class:`CoevolutionLoop` does this by replacing the agent block in
`sim_config` with the opposing-side champion before invoking
`evaluate`). This keeps the fitness function brain-agnostic on the
prey side: it simply runs whatever multi-agent env the patched
`sim_config` describes and aggregates predator-side metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from quantumnematode.agent.multi_agent import MultiAgentSimulation
from quantumnematode.brain.weights import load_weights, save_weights
from quantumnematode.utils.config_loader import create_env_from_config
from quantumnematode.utils.seeding import derive_run_seed, set_global_seed

if TYPE_CHECKING:
    from quantumnematode.agent.agent import QuantumNematodeAgent
    from quantumnematode.brain.arch._brain import Brain
    from quantumnematode.env import DynamicForagingEnvironment
    from quantumnematode.env.predator_brain import PredatorBrain
    from quantumnematode.evolution.encoders import GenomeEncoder
    from quantumnematode.evolution.genome import Genome
    from quantumnematode.utils.config_loader import SimulationConfig


# Fallback ceiling (exclusive) for the proximity signal: any non-zero
# kill count across N episodes scores >= 1/N, so capping the proximity
# fallback strictly below 1/N preserves the "one kill always beats
# zero kills" invariant. Concretely the raw cross-slot proximity
# ratio lives in [0, 1]; we scale it by this factor to land in
# [0, ~1/N).
_PROXIMITY_FALLBACK_HEADROOM = 0.99  # conservative — strictly < 1/N

# Shared default mirroring `EpisodicSuccessRate` / `LearnedPerformanceFitness`.
_DEFAULT_MAX_STEPS = 500


def _resolve_max_steps(sim_config: SimulationConfig) -> int:
    """Return `sim_config.max_steps` or the canonical default.

    Single source of truth so the kill-rate path and the proximity-
    fallback path use the same `max_steps` for normalisation — an
    accidental drift here would change the [0, 1] range of the
    fallback.
    """
    return sim_config.max_steps if sim_config.max_steps is not None else _DEFAULT_MAX_STEPS


def _build_env_with_genome_predators(
    sim_config: SimulationConfig,
    encoder: GenomeEncoder,
    genome: Genome,
    seed: int,
) -> DynamicForagingEnvironment:
    """Build a fresh env and install the genome-decoded brain on every predator.

    Spec contract:
    "all predator slots in the env using the same decoded brain (the
    genome under evaluation is the strategy being measured, not just
    slot 0)". The env's `_build_predator_brain` dispatcher constructs
    one brain per predator (via `_make_predator`); to override that with
    the genome's brain we replace `Predator.brain` on each slot
    post-construction. A fresh brain instance is created per slot via
    `encoder.decode` so weight tensors are not shared (avoids accidental
    in-place mutation if a future code path mutates brain state during
    evaluation).

    The env is built with HEADLESS theme — worker processes have no
    terminal, and any other theme would render to a discarded stdout
    (matching the pattern in
    :class:`~quantumnematode.evolution.fitness.EpisodicSuccessRate`).
    """
    if sim_config.environment is None:
        msg = (
            "Predator fitness requires sim_config.environment to be set "
            "(the env block describes both prey opponents and predator slot count)."
        )
        raise ValueError(msg)
    from quantumnematode.env.theme import Theme

    env = create_env_from_config(
        sim_config.environment,
        seed=seed,
        theme=Theme.HEADLESS,
        max_body_length=sim_config.body_length,
    )

    # Fail fast on a misconfigured env. With `predators.enabled=False` (or
    # `count=0`) the env builds an empty `predators` list — every episode
    # then yields zero kills + zero proximity, and fitness silently
    # collapses to 0.0 with no diagnostic. Co-evolution YAMLs always
    # enable predators, but a programmatic caller that patches the
    # sim_config wrong (or a typo in a config) would otherwise see a
    # flat-zero gradient with no signal as to why.
    if not env.predators:
        msg = (
            "Predator fitness requires at least one predator slot in the env. "
            "Got 0 — check that sim_config.environment.predators.enabled is "
            "True and count >= 1."
        )
        raise ValueError(msg)

    # One fresh brain per predator slot — same weights, independent
    # tensors. A single shared `brain` would also satisfy the spec, but
    # independent instances guard against future hooks that might
    # mutate `brain` state per slot (e.g. a per-slot RNG state).
    #
    # `encoder.decode` is annotated to return `Brain` (the agent-side
    # Protocol surface from `GenomeEncoder.decode`), but the predator
    # encoder concretely returns `MLPPPOPredatorBrain` which satisfies
    # the *separate* `PredatorBrain` Protocol. Cast at the assignment
    # boundary — runtime is correct, the cast just narrows the
    # static-type signature mismatch between the two Protocols.
    for predator in env.predators:
        predator.brain = cast(
            "PredatorBrain",
            encoder.decode(genome, sim_config, seed=seed),
        )

    return env


def _build_prey_agents(
    env: DynamicForagingEnvironment,
    sim_config: SimulationConfig,
) -> list[QuantumNematodeAgent]:
    """Build the prey agent list from `sim_config.multi_agent.agents`.

    Mirrors the multi-agent path used by :func:`scripts.run_simulation`
    (see [run_simulation.py:1740-1766]) but stripped to the bare
    minimum: the caller's `sim_config` already describes the FROZEN
    prey opponents (the CoevolutionLoop patches `sim_config.multi_agent`
    before invoking `evaluate`); here we just instantiate them.

    Returns the list of `QuantumNematodeAgent` ready for
    `MultiAgentSimulation`. Each agent's brain is built from its own
    `BrainContainerConfig` via the agent factory; predator brains are
    already installed on `env.predators` by
    :func:`_decode_brain_into_all_predators`.
    """
    if sim_config.multi_agent is None or not sim_config.multi_agent.agents:
        msg = (
            "Predator fitness requires sim_config.multi_agent.agents to be "
            "populated with frozen prey opponents. The CoevolutionLoop "
            "patches this before invoking evaluate; programmatic callers "
            "must do the same."
        )
        raise ValueError(msg)

    # Local imports keep this module's top-level import graph small —
    # agent + brain machinery only needs to load when fitness is
    # actually invoked, not when the FitnessFunction Protocol is
    # imported by other modules.
    from quantumnematode.agent.agent import (
        DEFAULT_MAX_AGENT_BODY_LENGTH,
        QuantumNematodeAgent,
    )
    from quantumnematode.evolution.brain_factory import (
        instantiate_brain_from_sim_config,
    )
    from quantumnematode.utils.config_loader import validate_sensing_config

    sensing = (
        validate_sensing_config(sim_config.environment.sensing)
        if sim_config.environment is not None and sim_config.environment.sensing is not None
        else None
    )
    body_length = (
        sim_config.body_length
        if sim_config.body_length is not None
        else DEFAULT_MAX_AGENT_BODY_LENGTH
    )

    agents = []
    for agent_config in sim_config.multi_agent.agents:
        # Build each prey agent's brain from its OWN BrainContainerConfig
        # by patching the agent-singleton sim_config with the per-agent
        # brain block, then dispatching through the canonical
        # `instantiate_brain_from_sim_config`. This keeps frozen prey
        # opponents on the SAME brain construction path that the prey
        # side of CoevolutionLoop uses (via the agent encoder's
        # `decode`), so any divergence between them would surface as
        # behavioural mismatch — not silently honoured by parallel
        # construction code.
        per_agent_sim_config = sim_config.model_copy(
            update={"brain": agent_config.brain},
        )
        brain = instantiate_brain_from_sim_config(per_agent_sim_config)

        # Per-agent `weights_path` overrides the freshly-init weights
        # with a saved checkpoint. `CoevolutionLoop._evaluate_candidate`
        # uses this to inject opposition prey genome weights when the
        # predator side is training: each opposition prey's flattened
        # genome is materialised to a tmp .pt and the path is set here.
        # Mirrors the pattern at `scripts/run_simulation.py:1633-1641`
        # for parity with the standalone multi-agent runner.
        if agent_config.weights_path:
            from quantumnematode.brain.weights import (
                WeightPersistence,
                load_weights,
            )

            if not isinstance(brain, WeightPersistence):
                msg = (
                    f"Agent {agent_config.id!r} brain {type(brain).__name__} "
                    "does not implement WeightPersistence; cannot load "
                    f"weights from {agent_config.weights_path!r}."
                )
                raise TypeError(msg)
            load_weights(brain, Path(agent_config.weights_path))

        if agent_config.id not in env.agents:
            env.add_agent(agent_id=agent_config.id, position=None)

        agents.append(
            QuantumNematodeAgent(
                brain=brain,
                env=env,
                agent_id=agent_config.id,
                max_body_length=body_length,
                satiety_config=sim_config.satiety,
                sensing_config=sensing,
            ),
        )
    return agents


class PredatorEpisodicKillRate:
    """Frozen-weight predator fitness: per-episode mean of total kills.

    Per-evaluation flow:

    1. Decode the genome into a predator brain via `encoder.decode`.
    2. Build a multi-agent env (HEADLESS) with frozen prey opponents
       described by the patched `sim_config`.
    3. Install the SAME decoded brain on every `Predator` slot in the
       env (so fitness measures the *strategy*, not slot-0 luck).
    4. Run N = `episodes` episodes via :class:`MultiAgentSimulation`;
       each episode aggregates `sum(per_predator_kills.values())`
       across slots.
    5. Return the per-episode mean of those sums (so a strategy that
       reliably kills ≥1 prey/episode scores ≥ 1.0).

    When `secondary_signal=True` and ALL N episodes recorded zero kills
    across all slots, fall back to a normalised proximity signal so
    early-generation flat fitness curves still expose a gradient. The
    fallback is strictly bounded below `1/N` so the kill-rate ordering
    is preserved (one kill in any episode beats any all-zero strategy).
    """

    def __init__(self, *, secondary_signal: bool = True) -> None:
        """Construct the fitness.

        Parameters
        ----------
        secondary_signal
            When True (default), zero-kill all-N-episodes evaluations
            fall back to a normalised proximity ratio bounded below
            `1/N`. When False, all-zero evaluations score exactly 0.0
            (matching the literal kill-rate definition; useful for
            ablations that disable the proximity assist).
        """
        self.secondary_signal = secondary_signal

    def evaluate(  # noqa: C901, PLR0912, PLR0913 - train+eval branches mirror LearnedPerformanceFitness shape
        self,
        genome: Genome,
        sim_config: SimulationConfig,
        encoder: GenomeEncoder,
        *,
        episodes: int,
        seed: int,
        warm_start_path_override: Path | None = None,
        weight_capture_path: Path | None = None,
    ) -> float:
        """Return the per-episode mean kill count (with proximity fallback).

        Returns 0.0 immediately when `episodes == 0` (the literal
        kill-rate definition with no episodes yields no signal).

        Optional inheritance kwargs (mirror `LearnedPerformanceFitness`'s
        surface for symmetric per-side Lamarckian inheritance):
        - `warm_start_path_override`: when set, after decoding the
          genome into a fresh brain, load weights from this `.pt` path
          INSTEAD of using the genome-encoded weights. Used by the
          co-evolution loop's Lamarckian inheritance step to seed each
          child from its parent's K-block-end checkpoint.
        - `weight_capture_path`: when set, save the brain's weights
          (which may have been updated during evaluation if PPO inner
          loop fired via the multi-agent runner's per-step hook) to
          this `.pt` path BEFORE returning. Used by the co-evolution
          loop's Lamarckian inheritance step to capture each genome's
          post-train weights for next-generation children to inherit.

        When `warm_start_path_override` or `weight_capture_path` is
        set, the encoder is invoked with `enable_learning=True` so the
        constructed brain has the PPO machinery; the multi-agent runner
        then drives `predator.brain.learn(reward, episode_done)` per
        step via `MultiAgentSimulation`'s predator-learning pass (which
        composes a per-step reward from kill-deltas + proximity flags
        and calls `predator.brain.learn` for every predator brain that
        exposes the `learn` method).

        Note (asymmetry vs `LearnedPerformanceFitness`): the prey-side
        fitness does an explicit train-then-eval split (K train + L
        frozen eval). The predator side returns the mean kill-rate
        across the FULL N-episode trajectory (training-included). This
        keeps the encoder/fitness surface simple and uses the
        within-episode-learning average as the optimisation signal;
        the trade-off is that early-episode-no-learning episodes
        contribute to the mean. Acceptable for screening; can be
        upgraded to a train-then-eval split if pilot evidence motivates.
        """
        if episodes < 0:
            msg = f"episodes must be >= 0, got {episodes}"
            raise ValueError(msg)
        if episodes == 0:
            return 0.0
        if sim_config.reward is None:
            msg = "PredatorEpisodicKillRate.evaluate requires sim_config.reward to be set."
            raise ValueError(msg)

        # Inheritance kwargs (either set) flips the encoder/factory into
        # learning-enabled mode so PPO can fire during the multi-agent
        # runner's per-step hook. When BOTH are None, behaviour is
        # byte-equivalent to the original frozen-weight contract.
        enable_learning = warm_start_path_override is not None or weight_capture_path is not None

        max_steps = _resolve_max_steps(sim_config)
        total_kills = 0
        total_proximity = 0
        # Tracks the slot count from the FIRST built env; all subsequent
        # rebuilds use the same env config so the count is invariant
        # across episodes. Used for proximity normalisation below.
        num_predator_slots: int | None = None
        # When learning is enabled, keep a reference to the brain so we
        # can `save_weights` after training. The env is rebuilt per
        # episode but we want weight continuity across episodes — so
        # we build the brain ONCE and re-attach it to each new env's
        # predator slot.
        persistent_brain = None
        if enable_learning:
            # `enable_learning` is a predator-encoder extension not part
            # of the canonical `GenomeEncoder.decode` Protocol surface
            # (the agent-side encoder doesn't need it). Cast to Any so
            # static checkers don't complain about the extra kwarg, but
            # runtime predator encoders accept it. Future revision: if
            # other encoders adopt the kwarg, lift it into the Protocol.
            persistent_brain = cast("Any", encoder).decode(
                genome,
                sim_config,
                seed=seed,
                enable_learning=True,
            )
            if warm_start_path_override is not None:
                load_weights(cast("Brain", persistent_brain), warm_start_path_override)

        for ep_idx in range(episodes):
            # Fresh env per episode — `MultiAgentSimulation.run_episode`
            # does not rebuild the env between calls, so we rebuild here
            # to match the per-episode layout-resampling pattern that
            # EpisodicSuccessRate / LearnedPerformanceFitness use via
            # `agent.reset_environment()`. The cost is one
            # `create_env_from_config` per episode (cheap — env init
            # is bounded by Poisson-disk food/predator placement).
            run_seed = derive_run_seed(seed, ep_idx)
            set_global_seed(run_seed)
            if persistent_brain is None:
                # Frozen-weight path: encoder builds + decodes fresh per episode.
                env = _build_env_with_genome_predators(sim_config, encoder, genome, run_seed)
            else:
                # Learning path: build env, then OVERWRITE the env's
                # predator brains with our persistent one so weights
                # accumulate across episodes within this eval.
                env = _build_env_with_genome_predators(sim_config, encoder, genome, run_seed)
                for predator in env.predators:
                    predator.brain = cast("PredatorBrain", persistent_brain)
            # `_build_env_with_genome_predators` raises ValueError when
            # the env has no predator slots, so `len(env.predators) >= 1`
            # is guaranteed here.
            if num_predator_slots is None:
                num_predator_slots = len(env.predators)
            agents = _build_prey_agents(env, sim_config)
            sim = MultiAgentSimulation(env=env, agents=agents)
            result = sim.run_episode(sim_config.reward, max_steps)
            total_kills += sum(result.per_predator_kills.values())
            total_proximity += sum(result.per_predator_prey_proximity_steps.values())

        # Persist post-training weights for Lamarckian inheritance, if
        # requested. Only meaningful in the learning path — the
        # frozen-weight path's brain is recreated per episode from the
        # genome's encoded weights, so there's nothing extra to capture
        # beyond the genome itself.
        if weight_capture_path is not None and persistent_brain is not None:
            weight_capture_path.parent.mkdir(parents=True, exist_ok=True)
            save_weights(cast("Brain", persistent_brain), weight_capture_path)

        mean_kills = total_kills / episodes
        if mean_kills > 0.0 or not self.secondary_signal:
            return float(mean_kills)

        # All-zero kill path: fall back to normalised proximity.
        # Per-episode raw proximity sums scale with `max_steps *
        # num_predator_slots` (each slot contributes at most one
        # proximity-step per env step), so the mean ratio over N
        # episodes lives in [0, 1]. Multiplying by
        # `_PROXIMITY_FALLBACK_HEADROOM / episodes` produces a value in
        # `[0, 0.99/episodes)`, strictly below `1/episodes` (the lowest
        # non-zero kill-rate).
        if num_predator_slots is None:  # pragma: no cover — episodes>0 guarded above
            return 0.0
        max_proximity_per_episode = max_steps * num_predator_slots
        if max_proximity_per_episode <= 0:  # pragma: no cover — slot count >= 1
            return 0.0
        proximity_ratio = total_proximity / (episodes * max_proximity_per_episode)
        return float(proximity_ratio * _PROXIMITY_FALLBACK_HEADROOM / episodes)


class PredatorLearnedPerformanceFitness:
    """Train-then-eval predator fitness — DEFERRED ablation variant.

    Mirrors :class:`~quantumnematode.evolution.fitness.LearnedPerformanceFitness`
    in shape but for predator-side genomes. The co-evolution loop
    hardcodes :class:`PredatorEpisodicKillRate` (no inner-loop
    training) for predator fitness; this class is reserved for a
    follow-up ablation if pilot evidence shows predator-side weight
    evolution stalls without inner-loop signal.

    Implementation note: the predator brain is currently designed for
    frozen-weight evaluation only — it has no `optimizer` /
    `training_state` weight components and no `.learn()` hook.
    Wiring an inner-loop PPO training stage onto
    `MLPPPOPredatorBrain` is out of scope for the current substrate.
    This stub raises `NotImplementedError` so an accidental config
    selection fails loudly rather than silently degrading to
    frozen-weight behaviour.
    """

    def evaluate(
        self,
        genome: Genome,
        sim_config: SimulationConfig,
        encoder: GenomeEncoder,
        *,
        episodes: int,
        seed: int,
    ) -> float:
        """Stub — see class docstring. Always raises."""
        del genome, sim_config, encoder, episodes, seed
        msg = (
            "PredatorLearnedPerformanceFitness is reserved for a future "
            "follow-up ablation; the co-evolution loop currently uses "
            "PredatorEpisodicKillRate (frozen-weight evaluation). "
            "Wiring the inner-loop PPO training stage requires extending "
            "MLPPPOPredatorBrain with optimizer + training_state weight "
            "components first."
        )
        raise NotImplementedError(msg)
