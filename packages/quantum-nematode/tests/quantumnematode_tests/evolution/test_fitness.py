"""Tests for :mod:`quantumnematode.evolution.fitness`."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from quantumnematode.agent.runners import StandardEpisodeRunner
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain
from quantumnematode.evolution.encoders import MLPPPOEncoder
from quantumnematode.evolution.fitness import (
    _PROGRESS_FOOD_WEIGHT,
    _PROGRESS_SURVIVAL_WEIGHT,
    EpisodicProgressFitness,
    EpisodicSuccessRate,
    FitnessFunction,
    FrozenEvalRunner,
    _progress_score,
)
from quantumnematode.report.dtypes import TerminationReason
from quantumnematode.utils.config_loader import SimulationConfig, load_simulation_config

if TYPE_CHECKING:
    from quantumnematode.evolution.genome import Genome

PROJECT_ROOT = Path(__file__).resolve().parents[5]
MLPPPO_CONFIG = PROJECT_ROOT / "configs/scenarios/foraging/mlpppo_small_oracle.yml"


# ---------------------------------------------------------------------------
# EpisodicSuccessRate basic contract
# ---------------------------------------------------------------------------


def _make_genome_for(
    config_path: Path,
) -> tuple[SimulationConfig, MLPPPOEncoder, Genome]:
    """Build ``(sim_config, encoder, genome)`` from a YAML config path."""
    sim_config = load_simulation_config(str(config_path))
    encoder = MLPPPOEncoder()
    genome = encoder.initial_genome(sim_config, rng=np.random.default_rng(0))
    return sim_config, encoder, genome


def test_episodic_success_rate_returns_float_in_unit_interval() -> None:
    """Fitness SHALL be a finite float in [0.0, 1.0] for an arbitrary genome."""
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    fitness = EpisodicSuccessRate()
    score = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=42)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_episodic_success_rate_deterministic_for_seeded_genome() -> None:
    """Same genome + same seed SHALL produce byte-identical fitness."""
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    fitness = EpisodicSuccessRate()
    a = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=42)
    b = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=42)
    assert a == b


def test_episodic_success_rate_rejects_zero_episodes() -> None:
    """Zero or negative episodes SHALL raise ValueError."""
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    fitness = EpisodicSuccessRate()
    with pytest.raises(ValueError, match="episodes must be positive"):
        fitness.evaluate(genome, sim_config, encoder, episodes=0, seed=42)
    with pytest.raises(ValueError, match="episodes must be positive"):
        fitness.evaluate(genome, sim_config, encoder, episodes=-1, seed=42)


# ---------------------------------------------------------------------------
# Frozen-weight contract
# ---------------------------------------------------------------------------


def test_frozen_eval_runner_never_calls_learn() -> None:
    """``FrozenEvalRunner`` SHALL NOT call ``brain.learn`` during episodes.

    Patches ``MLPPPOBrain.learn`` and ``MLPPPOBrain.update_memory`` and runs
    a short fitness eval; both methods SHALL be untouched.
    """
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    fitness = EpisodicSuccessRate()

    with (
        patch.object(MLPPPOBrain, "learn", autospec=True) as mock_learn,
        patch.object(MLPPPOBrain, "update_memory", autospec=True) as mock_um,
    ):
        fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=42)

    assert mock_learn.call_count == 0
    assert mock_um.call_count == 0


def test_episodic_success_rate_uses_termination_reason_for_success() -> None:
    """Success SHALL be defined as ``COMPLETED_ALL_FOOD`` termination.

    Patch ``StandardEpisodeRunner.run`` to return controlled termination
    reasons; verify the fitness function counts only ``COMPLETED_ALL_FOOD``
    as success.
    """
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    fitness = EpisodicSuccessRate()

    from quantumnematode.agent.runners import EpisodeResult

    completed = EpisodeResult(
        agent_path=[(0, 0)],
        termination_reason=TerminationReason.COMPLETED_ALL_FOOD,
        food_history=None,
    )
    starved = EpisodeResult(
        agent_path=[(0, 0)],
        termination_reason=TerminationReason.STARVED,
        food_history=None,
    )

    # 2 successes, 1 failure → 2/3 fitness
    sequence = [completed, starved, completed]
    with patch.object(FrozenEvalRunner, "run", side_effect=sequence):
        score = fitness.evaluate(genome, sim_config, encoder, episodes=3, seed=42)

    assert score == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# Seed propagation
# ---------------------------------------------------------------------------


def test_evaluate_passes_seed_to_encoder_decode() -> None:
    """``evaluate(seed=X)`` SHALL forward ``X`` to BOTH the encoder and the env factory.

    Specifically, ``encoder.decode`` and ``create_env_from_config`` MUST both
    receive the same ``seed`` value the fitness function was invoked with.
    Also asserts the env factory receives ``theme=Theme.HEADLESS`` so workers
    don't pay per-step rendering cost.
    """
    from quantumnematode.env.theme import Theme

    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    fitness = EpisodicSuccessRate()

    captured_decode: dict[str, int | None] = {}
    captured_env: dict[str, object] = {}
    real_decode = encoder.decode
    from quantumnematode.evolution import fitness as fitness_module

    real_env_factory = fitness_module.create_env_from_config

    def spy_decode(g, cfg, *, seed=None):
        captured_decode["seed"] = seed
        return real_decode(g, cfg, seed=seed)

    def spy_env_factory(env_config, *, seed=None, theme=None, **kwargs):
        captured_env["seed"] = seed
        captured_env["theme"] = theme
        return real_env_factory(env_config, seed=seed, theme=theme, **kwargs)

    with (
        patch.object(MLPPPOEncoder, "decode", side_effect=spy_decode),
        patch.object(fitness_module, "create_env_from_config", side_effect=spy_env_factory),
    ):
        fitness.evaluate(genome, sim_config, encoder, episodes=1, seed=99)

    assert captured_decode["seed"] == 99
    assert captured_env["seed"] == 99
    assert captured_env["theme"] == Theme.HEADLESS


def test_evaluate_fitness_seed_overrides_brain_config_seed() -> None:
    """Fitness ``seed`` overrides ``brain.config.seed`` and reaches the runtime RNG.

    Sets ``brain.config.seed = 0`` in sim_config, then runs evaluate with
    seed=1 and seed=2.  The two results MAY differ but aren't guaranteed to
    — random brains can both score 0.0.  What we DO guarantee, and assert
    here, is (a) seed=1 twice produces byte-identical fitness, and (b) a
    different seed runs without error.  Combined with
    ``test_evaluate_passes_seed_to_encoder_decode``, this proves the fitness
    seed (not the YAML BrainConfig.seed) controls per-evaluation RNG.
    """
    sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
    # Pin the YAML brain config seed to 0; the fitness function's seed kwarg
    # overrides this via encoder.decode → instantiate_brain_from_sim_config.
    assert sim_config.brain is not None  # narrows Optional for type checker
    bogus_brain_cfg = sim_config.brain.config.model_copy(update={"seed": 0})
    sim_config = sim_config.model_copy(
        update={
            "brain": sim_config.brain.model_copy(update={"config": bogus_brain_cfg}),
        },
    )

    fitness = EpisodicSuccessRate()
    a = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=1)
    b = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=2)
    # We can't guarantee they differ — random brains can score 0.0 for both —
    # but we CAN guarantee that with seed=1 twice we get the same value, and
    # that the seed kwarg actually reaches the encoder.  Combined with
    # test_evaluate_passes_seed_to_encoder_decode, this gives us the contract.
    a_again = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=1)
    assert a == a_again
    # And b is computed without error for a different seed.
    assert isinstance(b, float)


# ---------------------------------------------------------------------------
# food_history sentinel (B9)
# ---------------------------------------------------------------------------


def test_frozen_eval_runner_preserves_food_history_sentinel() -> None:
    """``FrozenEvalRunner._terminate_episode`` SHALL pass kwargs through unchanged.

    The override only injects ``learn=False, update_memory=False`` and
    forwards everything else.  In particular, the parent's
    ``food_history=...`` Ellipsis sentinel is preserved (not converted to
    ``None``), so the parent correctly falls back to ``agent.food_history``.

    We verify this by calling the override directly with a stand-in agent,
    capturing what gets forwarded to ``super()._terminate_episode``.
    """
    runner = FrozenEvalRunner()

    # Capture what the parent's _terminate_episode receives.
    captured: dict[str, object] = {}

    def fake_super(self, agent, params, reward, **kwargs):
        captured.update(kwargs)

    with patch.object(
        StandardEpisodeRunner,
        "_terminate_episode",
        autospec=True,
        side_effect=fake_super,
    ):
        runner._terminate_episode(  # type: ignore[arg-type]
            agent=None,  # type: ignore[arg-type]
            params=None,
            reward=0.0,
            success=True,
            termination_reason=TerminationReason.COMPLETED_ALL_FOOD,
        )

    # learn and update_memory were forced to False
    assert captured["learn"] is False
    assert captured["update_memory"] is False
    # success and termination_reason passed through
    assert captured["success"] is True
    assert captured["termination_reason"] == TerminationReason.COMPLETED_ALL_FOOD
    # food_history was NOT injected by the override (sentinel preserved by
    # virtue of not being in kwargs at all)
    assert "food_history" not in captured


# ---------------------------------------------------------------------------
# Regression: ``_build_agent`` threads ``max_body_length`` through.
#
# Pre-fix: ``_build_agent`` did not pass ``max_body_length``, so the agent
# defaulted to ``DEFAULT_MAX_AGENT_BODY_LENGTH = 6``.  Episode 0 ran
# against the explicit env (correct body length) but ``reset_environment``
# rebuilt the env using ``self.max_body_length = 6`` between episodes,
# silently switching to a fundamentally different (much harder) task.
# This test pins the fix in place.
# ---------------------------------------------------------------------------


def test_build_agent_threads_max_body_length() -> None:
    """The agent's ``max_body_length`` SHALL match ``sim_config.body_length``.

    Otherwise multi-episode fitness evaluations silently switch tasks
    between episode 0 (configured body) and episodes 1+ (default body=6)
    via the ``reset_environment`` rebuild path.
    """
    from quantumnematode.evolution.fitness import _build_agent
    from quantumnematode.utils.config_loader import create_env_from_config

    sim_config = load_simulation_config(str(MLPPPO_CONFIG))
    # Force a non-default body length so we can detect the bug —
    # mlpppo_small_oracle defaults to 2; pin to 3 here.
    sim_config = sim_config.model_copy(update={"body_length": 3})
    # ``environment`` is typed ``EnvironmentConfig | None``; the YAML always
    # populates it for evolution configs, so the assert documents the
    # invariant for pyright.
    assert sim_config.environment is not None
    env = create_env_from_config(
        sim_config.environment,
        seed=42,
        max_body_length=sim_config.body_length,
    )
    encoder = MLPPPOEncoder()
    rng = np.random.default_rng(0)
    genome = encoder.initial_genome(sim_config, rng=rng)
    from quantumnematode.evolution.brain_factory import instantiate_brain_from_sim_config

    brain = instantiate_brain_from_sim_config(sim_config, seed=42)
    encoder.decode(genome, sim_config, seed=42)
    agent = _build_agent(brain, env, sim_config)

    # The agent's stored max_body_length governs reset_environment's rebuild.
    # Must match the configured body_length, NOT the agent's default of 6.
    assert agent.max_body_length == sim_config.body_length, (
        f"agent.max_body_length={agent.max_body_length} != "
        f"sim_config.body_length={sim_config.body_length} — "
        "_build_agent regressed: episodes 1+ will silently switch body length"
    )


# ---------------------------------------------------------------------------
# EpisodicProgressFitness — graded scoring (_progress_score, pure)
# ---------------------------------------------------------------------------


class TestProgressScore:
    """The pure per-episode graded scorer ``_progress_score``.

    All grading logic lives here, tested with synthetic numbers (no env/agent),
    so the algorithm is pinned independently of the run wiring.
    """

    def test_full_clear_is_the_apex(self) -> None:
        """``COMPLETED_ALL_FOOD`` SHALL score exactly 1.0 regardless of steps."""
        for steps in (0, 1, 250, 500, 10_000):
            assert (
                _progress_score(
                    termination_reason=TerminationReason.COMPLETED_ALL_FOOD,
                    foods_collected=10,
                    target_foods=10,
                    steps=steps,
                    max_steps=500,
                )
                == 1.0
            )

    def test_nonclear_is_strictly_below_apex(self) -> None:
        """Any non-``COMPLETED_ALL_FOOD`` termination SHALL score < 1.0.

        Even the best non-clear (one food short, full survival) stays below the
        full-clear apex, so a champion that clears is always preferred.
        """
        best_nonclear = _progress_score(
            termination_reason=TerminationReason.HEALTH_DEPLETED,
            foods_collected=9,
            target_foods=10,
            steps=500,
            max_steps=500,
        )
        assert best_nonclear < 1.0

    def test_exact_blend_value(self) -> None:
        """A non-clear SHALL equal ``FOOD_W*food_frac + SURVIVAL_W*survival_frac``."""
        score = _progress_score(
            termination_reason=TerminationReason.HEALTH_DEPLETED,
            foods_collected=5,
            target_foods=10,
            steps=250,
            max_steps=500,
        )
        expected = _PROGRESS_FOOD_WEIGHT * 0.5 + _PROGRESS_SURVIVAL_WEIGHT * 0.5
        assert score == pytest.approx(expected)

    def test_monotonic_in_food(self) -> None:
        """More food (survival held fixed) SHALL score strictly higher."""
        more = _progress_score(
            termination_reason=TerminationReason.HEALTH_DEPLETED,
            foods_collected=5,
            target_foods=10,
            steps=100,
            max_steps=500,
        )
        less = _progress_score(
            termination_reason=TerminationReason.HEALTH_DEPLETED,
            foods_collected=2,
            target_foods=10,
            steps=100,
            max_steps=500,
        )
        assert more > less

    def test_monotonic_in_survival(self) -> None:
        """Longer survival (food held fixed) SHALL score strictly higher."""
        longer = _progress_score(
            termination_reason=TerminationReason.HEALTH_DEPLETED,
            foods_collected=2,
            target_foods=10,
            steps=400,
            max_steps=500,
        )
        shorter = _progress_score(
            termination_reason=TerminationReason.HEALTH_DEPLETED,
            foods_collected=2,
            target_foods=10,
            steps=100,
            max_steps=500,
        )
        assert longer > shorter

    def test_survival_breaks_the_zero_food_deadlock(self) -> None:
        """Zero-food genomes SHALL still be separable by survival time.

        This is the property that lets the GA bootstrap under lethal predators:
        when EVERY genome eats zero food, the survival term keeps the landscape
        non-flat (a genome that survives longer scores higher than one that dies
        instantly), so selection still has a gradient. A genome that dies
        instantly with no food scores exactly 0.0.
        """
        survived = _progress_score(
            termination_reason=TerminationReason.HEALTH_DEPLETED,
            foods_collected=0,
            target_foods=10,
            steps=100,
            max_steps=500,
        )
        died_instantly = _progress_score(
            termination_reason=TerminationReason.HEALTH_DEPLETED,
            foods_collected=0,
            target_foods=10,
            steps=0,
            max_steps=500,
        )
        assert survived > died_instantly
        assert died_instantly == 0.0

    def test_graceful_degradation_on_zero_denominators(self) -> None:
        """``target_foods <= 0`` / ``max_steps <= 0`` SHALL not raise and treat the term as 0."""
        # target_foods == 0 → food term contributes 0; survival still counts.
        only_survival = _progress_score(
            termination_reason=TerminationReason.HEALTH_DEPLETED,
            foods_collected=0,
            target_foods=0,
            steps=250,
            max_steps=500,
        )
        assert only_survival == pytest.approx(_PROGRESS_SURVIVAL_WEIGHT * 0.5)
        # Both denominators zero → score 0.0 (no division error).
        assert (
            _progress_score(
                termination_reason=TerminationReason.HEALTH_DEPLETED,
                foods_collected=0,
                target_foods=0,
                steps=0,
                max_steps=0,
            )
            == 0.0
        )

    def test_fractions_clamped_so_score_never_exceeds_apex(self) -> None:
        """Over-counted foods/steps SHALL clamp so a non-clear never exceeds 1.0."""
        score = _progress_score(
            termination_reason=TerminationReason.HEALTH_DEPLETED,
            foods_collected=15,  # > target — defensive clamp to 1.0
            target_foods=10,
            steps=999,  # > max_steps — defensive clamp to 1.0
            max_steps=500,
        )
        assert score == pytest.approx(_PROGRESS_FOOD_WEIGHT + _PROGRESS_SURVIVAL_WEIGHT)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# EpisodicProgressFitness — wiring (real eval)
# ---------------------------------------------------------------------------


class TestEpisodicProgressFitness:
    """The graded fitness end-to-end against a real foraging eval."""

    def test_satisfies_fitness_protocol(self) -> None:
        """``EpisodicProgressFitness`` SHALL satisfy the runtime-checkable Protocol."""
        assert isinstance(EpisodicProgressFitness(), FitnessFunction)

    def test_rejects_zero_episodes(self) -> None:
        """Zero or negative episodes SHALL raise ValueError (mirrors success_rate)."""
        sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
        fitness = EpisodicProgressFitness()
        with pytest.raises(ValueError, match="episodes must be positive"):
            fitness.evaluate(genome, sim_config, encoder, episodes=0, seed=42)

    def test_returns_unit_interval_float(self) -> None:
        """Fitness SHALL be a finite float in [0.0, 1.0] for an arbitrary genome."""
        sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
        score = EpisodicProgressFitness().evaluate(genome, sim_config, encoder, episodes=2, seed=42)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic_for_seeded_genome(self) -> None:
        """Same genome + same seed SHALL produce byte-identical graded fitness."""
        sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
        fitness = EpisodicProgressFitness()
        a = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=42)
        b = fitness.evaluate(genome, sim_config, encoder, episodes=2, seed=42)
        assert a == b

    def test_grades_at_or_above_success_rate(self) -> None:
        """Graded fitness SHALL be >= binary success on the SAME (genome, seed).

        Per episode the graded score is >= the success indicator (a clear scores
        1.0 under both; a non-clear scores >= 0 graded == 0 success), so the
        means satisfy ``graded >= success``. Combined with ``graded > 0`` (a
        random genome survives some steps ⇒ the survival term fires), this proves
        the wiring reads real per-episode progress — not just the success rate.
        """
        sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
        graded = EpisodicProgressFitness().evaluate(genome, sim_config, encoder, episodes=4, seed=7)
        success = EpisodicSuccessRate().evaluate(genome, sim_config, encoder, episodes=4, seed=7)
        assert graded >= success
        assert graded > 0.0

    def test_per_episode_tracker_does_not_accumulate(self) -> None:
        """Each episode's score SHALL read the per-episode foods/steps, not the cumulative.

        ``EpisodicProgressFitness`` is the first fitness to read
        ``agent.episode_foods_collected`` / ``episode_steps`` across the
        ``episodes_per_eval`` loop (``EpisodicSuccessRate`` reads only the
        returned ``EpisodeResult.termination_reason``), so the per-episode tracker
        reset is a NEW dependency no other test exercises. If
        ``reset_environment`` ever stops zeroing the tracker, foods/steps would
        accumulate (episode 2 reads ep1+ep2 …), silently inflating food-fraction
        toward the 1.0 clamp.

        We patch the runner to INCREMENT the tracker by a fixed amount per call
        (mimicking how a real episode accrues onto a freshly-reset tracker) and
        capture what ``_progress_score`` receives each episode. Correct reset ⇒
        every episode reads the SAME fixed amount; a broken reset ⇒ the captured
        values grow monotonically (2, 4, 6 …).
        """
        from quantumnematode.agent.runners import EpisodeResult
        from quantumnematode.evolution import fitness as fitness_module

        per_call_foods, per_call_steps = 2, 50
        captured: list[tuple[int, int]] = []

        def fake_run(agent, _reward_config, _max_steps):
            # Accrue onto the (freshly reset, for ep>0) tracker, as a real episode would.
            agent._episode_tracker.data.foods_collected += per_call_foods
            agent._episode_tracker.data.steps += per_call_steps
            return EpisodeResult(
                agent_path=[(0, 0)],
                termination_reason=TerminationReason.HEALTH_DEPLETED,
                food_history=None,
            )

        real_progress_score = fitness_module._progress_score

        def spy_progress_score(**kwargs: object) -> float:
            captured.append((kwargs["foods_collected"], kwargs["steps"]))  # type: ignore[arg-type]
            return real_progress_score(**kwargs)  # type: ignore[arg-type]

        sim_config, encoder, genome = _make_genome_for(MLPPPO_CONFIG)
        with (
            patch.object(FrozenEvalRunner, "run", side_effect=fake_run),
            patch.object(fitness_module, "_progress_score", side_effect=spy_progress_score),
        ):
            EpisodicProgressFitness().evaluate(genome, sim_config, encoder, episodes=3, seed=42)

        # Per-episode reset ⇒ every episode reads exactly the fixed increment.
        # A regressed reset would read (2,50),(4,100),(6,150).
        assert captured == [(per_call_foods, per_call_steps)] * 3, (
            f"tracker accumulated across episodes (reset regressed): {captured}"
        )
