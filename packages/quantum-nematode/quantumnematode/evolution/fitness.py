"""Fitness functions for the evolution loop.

This module provides :class:`EpisodicSuccessRate` — a frozen-weight fitness
that decodes a genome into a fresh brain, runs it for ``episodes_per_eval``
complete episodes, counts successes, and returns the ratio.  No learning
happens during evaluation; the brain's weights are fixed by the genome.

Calling :meth:`agent.run_episode` directly is insufficient because
:class:`~quantumnematode.agent.runners.StandardEpisodeRunner._terminate_episode`
defaults ``learn=True`` on the success path ([runners.py:817-823]).  Every
successful episode would call ``brain.learn()`` and mutate weights between
episodes within a single fitness evaluation, breaking the frozen-weight
contract.  :class:`FrozenEvalRunner` subclasses the standard runner and forces
``learn=False, update_memory=False`` on every termination path while
otherwise inheriting all step-loop logic unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from quantumnematode.agent.agent import DEFAULT_MAX_AGENT_BODY_LENGTH, QuantumNematodeAgent
from quantumnematode.agent.runners import StandardEpisodeRunner
from quantumnematode.brain.weights import load_weights, save_weights
from quantumnematode.env.theme import Theme
from quantumnematode.report.dtypes import TerminationReason
from quantumnematode.utils.config_loader import create_env_from_config
from quantumnematode.utils.seeding import derive_run_seed, get_rng, set_global_seed

if TYPE_CHECKING:
    from pathlib import Path

    from quantumnematode.agent.runners import EpisodeResult
    from quantumnematode.brain.arch._brain import Brain
    from quantumnematode.env import DynamicForagingEnvironment
    from quantumnematode.evolution.encoders import GenomeEncoder
    from quantumnematode.evolution.genome import Genome
    from quantumnematode.utils.config_loader import SimulationConfig


# ---------------------------------------------------------------------------
# Fitness protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FitnessFunction(Protocol):
    """Protocol for evaluating the fitness of a genome."""

    def evaluate(
        self,
        genome: Genome,
        sim_config: SimulationConfig,
        encoder: GenomeEncoder,
        *,
        episodes: int,
        seed: int,
    ) -> float:
        """Return a fitness value for ``genome``.

        ``seed`` controls per-evaluation determinism: implementations propagate
        it through ``encoder.decode(seed=seed)`` so the brain's RNG, numpy
        global, and torch global all start from the same point.  ``encoder``
        is responsible for decoding the genome into a brain; the fitness
        function does NOT call ``setup_brain_model`` or other brain-construction
        code directly.
        """
        ...


# ---------------------------------------------------------------------------
# Frozen-weight runner
# ---------------------------------------------------------------------------


class FrozenEvalRunner(StandardEpisodeRunner):
    """Runs an episode without ever calling ``brain.learn()`` or ``update_memory()``.

    Two override points are needed because the standard runner calls ``learn``
    in two places:

    1. **Per-step**, inside the main loop ([runners.py:747]): every step the
       runner calls ``agent.brain.learn(...)`` for ``ClassicalBrain``s.  The
       termination-time override does NOT catch this.
    2. **Per-termination**, via ``_terminate_episode`` ([runners.py:182]): the
       success path defaults ``learn=True``.

    To intercept both, ``run()`` temporarily replaces ``agent.brain.learn`` and
    ``agent.brain.update_memory`` with no-ops for the duration of the episode,
    then restores them.  ``_terminate_episode`` also forces ``learn=False,
    update_memory=False`` as a belt-and-braces guard (and to preserve the
    ``food_history=...`` Ellipsis sentinel by passing kwargs through unchanged).
    """

    def run(
        self,
        agent: QuantumNematodeAgent,
        reward_config: Any,  # noqa: ANN401
        max_steps: int,
        **kwargs: Any,  # noqa: ANN401
    ) -> EpisodeResult:
        """Run an episode with ``brain.learn`` and ``brain.update_memory`` neutered.

        The brain methods are restored after the episode so subsequent uses of
        the same agent (e.g. across multiple episodes in a fitness eval) start
        with the original implementation each time we re-enter ``run``.
        """
        # Save original methods.
        original_learn = agent.brain.learn  # type: ignore[attr-defined]
        original_update_memory = agent.brain.update_memory  # type: ignore[attr-defined]

        def _noop_learn(*_args: Any, **_kw: Any) -> None:  # noqa: ANN401
            return None

        def _noop_update_memory(*_args: Any, **_kw: Any) -> None:  # noqa: ANN401
            return None

        agent.brain.learn = _noop_learn  # type: ignore[attr-defined,method-assign]
        agent.brain.update_memory = _noop_update_memory  # type: ignore[attr-defined,method-assign]
        try:
            return super().run(agent, reward_config, max_steps, **kwargs)
        finally:
            agent.brain.learn = original_learn  # type: ignore[attr-defined,method-assign]
            agent.brain.update_memory = original_update_memory  # type: ignore[attr-defined,method-assign]

    def _terminate_episode(  # type: ignore[override]
        self,
        agent: QuantumNematodeAgent,
        params: Any,  # noqa: ANN401
        reward: float,
        **kwargs: Any,  # noqa: ANN401
    ) -> EpisodeResult:
        # Force frozen behaviour regardless of caller's intent.  All other
        # kwargs (success, termination_reason, food_history sentinel, etc.)
        # pass through unchanged.  This is belt-and-braces — `run()` already
        # neutered the brain's learn/update_memory methods.
        kwargs["learn"] = False
        kwargs["update_memory"] = False
        return super()._terminate_episode(agent, params, reward, **kwargs)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def _build_agent(
    brain: Brain,
    env: DynamicForagingEnvironment,
    sim_config: SimulationConfig,
) -> QuantumNematodeAgent:
    """Centralise QuantumNematodeAgent construction for fitness eval.

    Pulls ``satiety_config`` and ``sensing_config`` from ``sim_config``; other
    constructor kwargs use defaults (theme/rich_style_config are presentation-
    only; agent_id defaults to "default"; maze_grid_size is unused when an
    explicit ``env`` is provided).

    ``max_body_length`` MUST be threaded through from ``sim_config`` because
    although the explicit ``env`` already has the correct body length, the
    agent stores its own ``self.max_body_length`` and uses it when
    ``agent.reset_environment()`` rebuilds the env between episodes
    ([agent.py:1122]).  Omitting this kwarg lets the agent default to
    ``DEFAULT_MAX_AGENT_BODY_LENGTH = 6``, so the FIRST episode runs against
    the configured body length but every SUBSEQUENT episode silently runs
    with body=6 — a fundamentally different (much harder) task.  This
    silently corrupted every multi-episode fitness evaluation prior to this
    fix.
    """
    # Mirror run_simulation.py's pattern: pass the validated sensing
    # config (which auto-enables STAM when klinotaxis/derivative modes
    # are active) rather than the raw env-block sensing.  Most pilot
    # YAMLs set ``stam_enabled: true`` explicitly so this is usually a
    # no-op, but matching the canonical path means evolution configs
    # behave identically to scenario configs even if a user forgets the
    # explicit ``stam_enabled`` line.
    if sim_config.environment is not None and sim_config.environment.sensing is not None:
        from quantumnematode.utils.config_loader import validate_sensing_config

        sensing = validate_sensing_config(sim_config.environment.sensing)
    else:
        sensing = None
    # ``sim_config.body_length`` is ``int | None`` (default ``None``); the
    # agent constructor expects a non-optional ``int`` and applies its own
    # default when the YAML omitted the field.  Mirror that fallback here
    # so pyright stays happy and the runtime behaviour for unset YAML is
    # unchanged.
    body_length = (
        sim_config.body_length
        if sim_config.body_length is not None
        else DEFAULT_MAX_AGENT_BODY_LENGTH
    )
    return QuantumNematodeAgent(
        brain=brain,
        env=env,
        max_body_length=body_length,
        satiety_config=sim_config.satiety,
        sensing_config=sensing,
    )


# ---------------------------------------------------------------------------
# EpisodicSuccessRate
# ---------------------------------------------------------------------------


class EpisodicSuccessRate:
    """Frozen-weight fitness: success rate over ``episodes_per_eval`` episodes.

    Decodes the genome into a fresh brain (with seed propagated via
    ``encoder.decode(seed=seed)`` → ``BrainConfig.seed`` patch), builds the env
    with the same seed, runs ``episodes`` complete episodes via
    :class:`FrozenEvalRunner`, and returns ``successes / episodes`` where success
    is defined as ``result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD``.

    Returns a value in ``[0.0, 1.0]``.

    No learning happens during evaluation — ``FrozenEvalRunner`` forces
    ``learn=False, update_memory=False`` on every termination.  The brain's
    ``post_process_episode`` is still called by the inherited per-step logic,
    advancing ``_episode_count`` between episodes within a single evaluation,
    but this does not affect reproducibility (LR is unused without ``.learn()``,
    actions depend on weights + seeded RNG only).
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
        """Run ``episodes`` complete episodes and return the success ratio."""
        if episodes <= 0:
            msg = f"episodes must be positive, got {episodes}"
            raise ValueError(msg)

        # Pass seed through the encoder.  The wrapper patches BrainConfig.seed
        # before brain construction, so the brain's __init__ calls
        # set_global_seed(seed) and self.rng = get_rng(seed) — seeding numpy
        # global, torch global, and the brain's local RNG to OUR seed.
        brain = encoder.decode(genome, sim_config, seed=seed)
        if sim_config.environment is None:
            msg = "EpisodicSuccessRate.evaluate requires sim_config.environment to be set."
            raise ValueError(msg)
        # HEADLESS theme short-circuits per-step rendering in
        # `agent._render_step` (agent.py:958).  Worker processes have no
        # terminal, so any other theme would render to a discarded stdout
        # — wasted work paid every step.
        env = create_env_from_config(
            sim_config.environment,
            seed=seed,
            theme=Theme.HEADLESS,
            max_body_length=sim_config.body_length,
        )

        agent = _build_agent(brain, env, sim_config)
        runner = FrozenEvalRunner()

        if sim_config.reward is None:
            msg = "EpisodicSuccessRate.evaluate requires sim_config.reward to be set."
            raise ValueError(msg)
        max_steps = sim_config.max_steps if sim_config.max_steps is not None else 500

        successes = 0
        for ep_idx in range(episodes):
            # Per-episode seed derivation matches run_simulation.py's
            # per-run pattern.  Without it the global RNG drifts
            # monotonically and every reset rebuilds the env from the
            # original seed → identical layouts every episode.
            run_seed = derive_run_seed(seed, ep_idx)
            set_global_seed(run_seed)
            # Reset the env between episodes so each starts from a clean
            # initial state (food respawn, agent at start position, full
            # satiety).  Without this, a failed episode leaves the env
            # in a starved/exhausted state and subsequent episodes
            # inherit the failure.  Matches the agent.reset_environment
            # pattern used by run_simulation.py between runs.
            if ep_idx > 0:
                agent.env.seed = run_seed
                agent.env.rng = get_rng(run_seed)
                agent.reset_environment()
            result = runner.run(agent, sim_config.reward, max_steps)
            if result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD:
                successes += 1

        return successes / episodes


# ---------------------------------------------------------------------------
# LearnedPerformanceFitness
# ---------------------------------------------------------------------------

# Default max_steps when sim_config.max_steps is None — matches the
# fallback used by EpisodicSuccessRate.
_DEFAULT_MAX_STEPS = 500


class LearnedPerformanceFitness:
    """Train-then-eval fitness: K train episodes followed by L frozen eval episodes.

    Per-evaluation flow:

    1. Decode the genome into a fresh brain (the encoder builds a brain
       with the genome's hyperparameters; weights are freshly initialised).
    2. Train phase: run K = ``evolution.learn_episodes_per_eval`` episodes
       via :class:`StandardEpisodeRunner` (calls ``brain.learn()`` per-step
       under the standard contract).
    3. Build a SECOND, fresh environment for the eval phase (same seed),
       reusing the trained brain.  Critical: post-train env state is
       arbitrary (food consumed, agent in some corner) and would corrupt
       the eval measurement.  The brain carries over with its learned
       weights; the env starts clean.
    4. Eval phase: run L episodes via :class:`FrozenEvalRunner`.  Count
       successes via
       ``result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD``.
    5. Return ``successes / L``.

    Episode-count resolution:

    - K reads from ``sim_config.evolution.learn_episodes_per_eval`` directly
      (no CLI override exists for it today).
    - L reads from ``sim_config.evolution.eval_episodes_per_eval`` if set
      in YAML, else falls back to the protocol's ``episodes`` kwarg —
      which the loop wires from its CLI-override-aware
      ``evolution_config.episodes_per_eval``.  This makes ``--episodes``
      work consistently for both fitness functions.

    K=0 raises ``ValueError`` with a clear pointer to
    :class:`EpisodicSuccessRate` (the correct fitness for frozen-weight
    evaluation).
    """

    # C901: linear pipeline (decode → train env → train loop → eval env → eval loop)
    # with multiple defensive guards — splitting into helpers fragments the flow.
    def evaluate(  # noqa: C901
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
        """Run K train + L eval and return ``eval_successes / L``.

        Two optional kwargs that the :class:`EvolutionLoop` MAY pass
        per-genome (defaulting to ``None`` when omitted, preserving the
        existing single-arg call shape):

        - ``warm_start_path_override``: when set, the brain is loaded
          from this path INSTEAD of ``evolution_config.warm_start_path``.
          The two are mutually exclusive at YAML load time, so in
          practice only one is ever active.  The override exists so the
          loop's inheritance step can supply per-genome parent
          checkpoints without mutating run-wide config.
        - ``weight_capture_path``: when set, the post-K-train brain
          weights are written to this path via ``save_weights`` AFTER
          the K train loop completes and BEFORE the L eval phase begins.
          This captures the policy as-trained (not as-eval'd).  The
          path's parent directory is created if missing.
        """
        # Defensive guards mirroring EpisodicSuccessRate.evaluate, plus
        # the evolution-block guard specific to learned-performance fitness.
        if sim_config.evolution is None:
            msg = (
                "LearnedPerformanceFitness requires an `evolution:` block in "
                "the YAML to set learn_episodes_per_eval. The loop forwards "
                "the raw sim_config to fitness; sim_config.evolution is None "
                "so we have no K to use."
            )
            raise ValueError(msg)
        if sim_config.environment is None:
            msg = "LearnedPerformanceFitness.evaluate requires sim_config.environment to be set."
            raise ValueError(msg)
        if sim_config.reward is None:
            msg = "LearnedPerformanceFitness.evaluate requires sim_config.reward to be set."
            raise ValueError(msg)

        evolution_config = sim_config.evolution
        if evolution_config.learn_episodes_per_eval == 0:
            msg = (
                "LearnedPerformanceFitness requires learn_episodes_per_eval > 0; "
                "got 0. Use EpisodicSuccessRate for frozen-weight evaluation, "
                "or set learn_episodes_per_eval in the evolution: block."
            )
            raise ValueError(msg)

        max_steps = sim_config.max_steps if sim_config.max_steps is not None else _DEFAULT_MAX_STEPS

        # Decode the genome → fresh brain with the evolved hyperparameters.
        brain = encoder.decode(genome, sim_config, seed=seed)

        # Optional warm-start: load a pre-trained checkpoint AFTER the fresh
        # brain is constructed and BEFORE the train phase.  Each genome's
        # K train episodes then fine-tune the checkpoint under the genome's
        # evolved hyperparameters.  Validator already guarantees that the
        # schema does not contain architecture-changing fields when either
        # source is set, so state_dict shapes match.
        #
        # Two sources, mutually exclusive at YAML load time:
        #   (a) ``warm_start_path_override`` (per-genome, set by the loop's
        #       inheritance step) — wins when present.
        #   (b) ``sim_config.evolution.warm_start_path`` (run-wide static
        #       checkpoint, set in YAML) — used otherwise.
        # ``load_weights`` raises ``FileNotFoundError`` with a clear
        # message if the resolved path doesn't exist.
        warm_start_path = (
            warm_start_path_override
            if warm_start_path_override is not None
            else evolution_config.warm_start_path
        )
        if warm_start_path is not None:
            load_weights(brain, warm_start_path)

        # Train phase: fresh env, brain weights mutate as it learns.
        train_env = create_env_from_config(
            sim_config.environment,
            seed=seed,
            theme=Theme.HEADLESS,
            max_body_length=sim_config.body_length,
        )
        train_agent = _build_agent(brain, train_env, sim_config)
        train_runner = StandardEpisodeRunner()
        for ep_idx in range(evolution_config.learn_episodes_per_eval):
            # Per-episode seed derivation matches run_simulation.py's
            # per-run pattern.  Without it, the global RNG drifts
            # monotonically across all K train episodes and every reset
            # rebuilds the env from the same original seed → identical
            # layouts every episode (food positions, agent start).  The
            # brain trains on one specific layout repeatedly rather than
            # the population of layouts the standard run loop produces.
            run_seed = derive_run_seed(seed, ep_idx)
            set_global_seed(run_seed)
            # Reset env between episodes so training samples come from
            # clean initial states rather than from whatever post-failure
            # state the previous episode left behind.  Brain weights
            # persist (they're learned across episodes); env state does
            # not.  Matches the run_simulation.py per-run reset pattern.
            if ep_idx > 0:
                train_agent.env.seed = run_seed
                train_agent.env.rng = get_rng(run_seed)
                train_agent.reset_environment()
            train_runner.run(train_agent, sim_config.reward, max_steps)

        # Optional weight capture: persist the post-K-train brain weights
        # to ``weight_capture_path`` BEFORE the eval phase begins.  This
        # captures the policy as-trained, not as-eval'd (the eval phase
        # never mutates weights under FrozenEvalRunner, but capturing
        # before eval keeps the contract simple: "trained weights" =
        # weights at the moment train ended).  Used by the Lamarckian
        # inheritance loop to checkpoint each genome for the next
        # generation's children to inherit from.
        if weight_capture_path is not None:
            weight_capture_path.parent.mkdir(parents=True, exist_ok=True)
            save_weights(brain, weight_capture_path)

        # Eval phase: SECOND fresh env (same seed) — post-train env state
        # is arbitrary and would corrupt the eval measurement.  Brain
        # carries over with its learned weights.
        eval_env = create_env_from_config(
            sim_config.environment,
            seed=seed,
            theme=Theme.HEADLESS,
            max_body_length=sim_config.body_length,
        )
        eval_agent = _build_agent(brain, eval_env, sim_config)
        eval_runner = FrozenEvalRunner()

        # L resolution: YAML override wins; else fall back to the
        # protocol's `episodes` kwarg (which is the loop's
        # CLI-override-aware episodes_per_eval).
        eval_eps = evolution_config.eval_episodes_per_eval
        eval_count = eval_eps if eval_eps is not None else episodes
        # Guard against zero/negative eval_count.  evolution.eval_episodes_per_eval
        # is Field(ge=1) so when set it's always positive, but the protocol's
        # `episodes` kwarg has no validation — a programmatic caller could pass
        # 0 or negative.  Mirrors EpisodicSuccessRate.evaluate's guard above.
        if eval_count <= 0:
            msg = f"eval_count must be > 0, got {eval_count}"
            raise ValueError(msg)

        successes = 0
        for ep_idx in range(eval_count):
            # Per-episode seed derivation — same rationale as the train
            # loop above.  Eval episodes use a different base offset
            # (``seed + K + ep_idx``) so eval layouts don't replay the
            # last K train layouts; otherwise the brain would be eval'd
            # on layouts identical to its most recent training, which
            # over-optimistically scores it.
            run_seed = derive_run_seed(
                seed + evolution_config.learn_episodes_per_eval,
                ep_idx,
            )
            set_global_seed(run_seed)
            # Reset env between eval episodes so each starts from a
            # clean initial state — same rationale as the train loop.
            if ep_idx > 0:
                eval_agent.env.seed = run_seed
                eval_agent.env.rng = get_rng(run_seed)
                eval_agent.reset_environment()
            result = eval_runner.run(eval_agent, sim_config.reward, max_steps)
            if result.termination_reason == TerminationReason.COMPLETED_ALL_FOOD:
                successes += 1
        return successes / eval_count
