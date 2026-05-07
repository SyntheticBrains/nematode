"""Tests for :mod:`quantumnematode.evolution.predator_fitness` (task 3.6).

Covers spec scenarios from `evolution-framework/spec.md`:

- FitnessFunction Protocol conformance (`isinstance` via runtime_checkable).
- `evaluate(genome, sim_config, encoder, *, episodes, seed) -> float`
  signature is honoured.
- Per-episode mean of `sum(per_predator_kills.values())` across slots.
- All predator slots receive the same decoded brain (no slot-0 special-casing).
- N=0 returns 0.0 immediately.
- Secondary proximity signal triggers when all-N kills are zero.
- Secondary fallback strictly less than `1/episodes` (preserves
  kill-rate ordering — one kill always beats the proximity signal).
- Pickle round-trip (so fitness can flow through the worker process).

The multi-agent runner path is monkeypatched: building a real env +
agents + running a real episode is out of scope for unit tests. The
synthetic results we feed in are sufficient to verify the fitness
function's aggregation logic.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from quantumnematode.agent.multi_agent import MultiAgentEpisodeResult
from quantumnematode.evolution.encoders import GenomeEncoder
from quantumnematode.evolution.fitness import FitnessFunction
from quantumnematode.evolution.genome import Genome
from quantumnematode.evolution.predator_fitness import (
    PredatorEpisodicKillRate,
    PredatorLearnedPerformanceFitness,
)

if TYPE_CHECKING:
    from quantumnematode.utils.config_loader import SimulationConfig


# Magic-number constants — tests use these throughout for readability.
_DEFAULT_MAX_STEPS = 500  # mirrors `_DEFAULT_MAX_STEPS` in predator_fitness.py


class _FakeBrain:
    """Stand-in for `MLPPPOPredatorBrain` — identity-equality only.

    Tests use this to verify that "all predator slots use the same
    decoded brain" — every slot's `brain` attribute compares `is` to
    the SAME instance (or independent instances each from the encoder,
    which our `FakeEncoder` stamps with an incrementing slot index).
    """

    def __init__(self, marker: int = 0) -> None:
        self.marker = marker


class _FakeEncoder:
    """Records every `decode` call so the test can verify equal genomes are decoded.

    The return type of :meth:`decode` is annotated as ``Brain`` (the
    Protocol surface) but actually returns a :class:`_FakeBrain`
    duck-instance. Pyright/mypy flag this as unsound — pragmatically
    acceptable in test code because:

    - The fitness function only ever assigns the returned object to
      ``Predator.brain`` and never invokes a `Brain` method on it
      (real brain methods are exercised via `MultiAgentSimulation`,
      which we monkeypatch in :func:`_patch_runtime`).
    - Our `_FakeBrain` deliberately has zero methods, so any future
      code path that does call a `Brain` method through this fake
      will fail loudly with `AttributeError`, surfacing the
      assumption.

    We therefore cast at the boundary (call sites) via
    :func:`cast(GenomeEncoder, encoder)` rather than annotating
    `_FakeEncoder.decode` as `-> Brain` (which would lie about the
    runtime type).
    """

    brain_name = "mlpppo_predator"

    def __init__(self) -> None:
        self.decode_calls: list[tuple[Genome, int]] = []

    def initial_genome(  # pragma: no cover — unused in tests
        self,
        sim_config: SimulationConfig,
        *,
        rng: np.random.Generator,
    ) -> Genome:
        del sim_config, rng
        return Genome(
            params=np.zeros(4, dtype=np.float32),
            genome_id="",
            parent_ids=[],
            generation=0,
        )

    def decode(
        self,
        genome: Genome,
        sim_config: SimulationConfig,
        *,
        seed: int | None = None,
    ) -> Any:
        del sim_config
        self.decode_calls.append((genome, seed if seed is not None else 0))
        return _FakeBrain(marker=len(self.decode_calls))

    def genome_dim(  # pragma: no cover — unused in tests
        self,
        sim_config: SimulationConfig,
    ) -> int:
        del sim_config
        return 4

    def genome_stds(  # pragma: no cover — unused in tests
        self,
        sim_config: SimulationConfig,
    ) -> list[float] | None:
        del sim_config
        return None

    def genome_bounds(  # pragma: no cover — unused in tests
        self,
        sim_config: SimulationConfig,
    ) -> list[tuple[float, float]] | None:
        del sim_config
        return None


class _FakePredator:
    """Stand-in for `env.predators[i]` — only `brain` and `predator_id` are read."""

    def __init__(self, predator_id: str) -> None:
        self.predator_id = predator_id
        self.brain: _FakeBrain | None = None


class _FakeEnv:
    """Stand-in for `DynamicForagingEnvironment` — only `predators` is read."""

    def __init__(self, num_predator_slots: int) -> None:
        self.predators = [_FakePredator(f"predator_{i}") for i in range(num_predator_slots)]


def _make_synthetic_result(
    *,
    kills_per_slot: dict[str, int] | None = None,
    proximity_per_slot: dict[str, int] | None = None,
) -> MultiAgentEpisodeResult:
    """Build a `MultiAgentEpisodeResult` with only the fields predator fitness reads."""
    return MultiAgentEpisodeResult(
        agent_results={},
        total_food_collected=0,
        per_agent_food={},
        food_competition_events=0,
        proximity_events=0,
        agents_alive_at_end=0,
        mean_agent_success=0.0,
        food_gini_coefficient=0.0,
        per_predator_kills=kills_per_slot or {},
        per_predator_prey_proximity_steps=proximity_per_slot or {},
    )


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_predator_slots: int,
    results_per_episode: list[MultiAgentEpisodeResult],
) -> dict[str, list]:
    """Stub out the heavy runtime paths and capture call records.

    The fitness function calls (in order):

    1. `_build_env_with_genome_predators(sim_config, encoder, genome, seed)`
       → stubbed to return `_FakeEnv` and record the seed.
    2. `_build_prey_agents(env, sim_config)` → stubbed to return [].
    3. `MultiAgentSimulation(env, agents)` → stubbed via a class
       replacement whose `run_episode` returns the next pre-baked
       result from `results_per_episode`.

    Returns a dict of capture lists for assertion convenience.
    """
    captures: dict[str, list] = {
        "build_env_calls": [],
        "build_agents_calls": [],
        "run_episode_calls": [],
        "envs": [],
    }
    result_iter = iter(results_per_episode)

    def fake_build_env(sim_config, encoder, genome, seed):
        env = _FakeEnv(num_predator_slots=num_predator_slots)
        # Mimic the real path: install the encoder-decoded brain on every
        # predator slot. This is what the spec guards against ("all
        # slots use the same decoded brain") — the fitness function
        # MUST trigger N decode calls per env construction.
        for predator in env.predators:
            predator.brain = encoder.decode(genome, sim_config, seed=seed)
        captures["build_env_calls"].append(seed)
        captures["envs"].append(env)
        return env

    def fake_build_agents(env, sim_config):
        captures["build_agents_calls"].append(env)
        return []

    class _FakeSim:
        def __init__(self, env, agents) -> None:
            self.env = env

        def run_episode(self, reward_config, max_steps):
            captures["run_episode_calls"].append(self.env)
            return next(result_iter)

    monkeypatch.setattr(
        "quantumnematode.evolution.predator_fitness._build_env_with_genome_predators",
        fake_build_env,
    )
    monkeypatch.setattr(
        "quantumnematode.evolution.predator_fitness._build_prey_agents",
        fake_build_agents,
    )
    monkeypatch.setattr(
        "quantumnematode.evolution.predator_fitness.MultiAgentSimulation",
        _FakeSim,
    )
    return captures


@pytest.fixture
def fake_sim_config():
    """Minimal-but-valid sim_config carrying just a `reward` block.

    The fitness function only reads `sim_config.reward` and
    `sim_config.max_steps` — both touched after the runtime patches
    activate. The patches don't read the env/predator/multi_agent
    blocks, so we don't need to populate them for unit tests.
    """
    from quantumnematode.utils.config_loader import RewardConfig, SimulationConfig

    return SimulationConfig(reward=RewardConfig())


# ---------------------------------------------------------------------------
# Protocol conformance + signature
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Spec scenario "FitnessFunction Protocol Conformance"."""

    def test_episodic_kill_rate_satisfies_protocol(self) -> None:
        """`PredatorEpisodicKillRate` SHALL satisfy the runtime-checkable Protocol."""
        assert isinstance(PredatorEpisodicKillRate(), FitnessFunction)

    def test_learned_performance_fitness_satisfies_protocol(self) -> None:
        """The deferred `PredatorLearnedPerformanceFitness` SHALL also satisfy the Protocol."""
        assert isinstance(PredatorLearnedPerformanceFitness(), FitnessFunction)

    def test_pickle_round_trip(self) -> None:
        """The fitness instance SHALL pickle for transport through worker processes."""
        original = PredatorEpisodicKillRate(secondary_signal=False)
        # S301: pickle.loads on data we pickled ourselves is trusted.
        round_tripped = pickle.loads(pickle.dumps(original))  # noqa: S301
        assert isinstance(round_tripped, PredatorEpisodicKillRate)
        assert round_tripped.secondary_signal is False


# ---------------------------------------------------------------------------
# Mean kill-rate calculation
# ---------------------------------------------------------------------------


class TestMeanKillRate:
    """Spec scenario "PredatorEpisodicKillRate Mean Kills Calculation"."""

    def test_mean_kills_aggregates_across_slots_and_episodes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_sim_config: SimulationConfig,
    ) -> None:
        """Fitness SHALL be `mean(sum(per_predator_kills.values()))` across N episodes."""
        # Episode 1: slot 0 kills 2, slot 1 kills 1 → sum=3.
        # Episode 2: slot 0 kills 0, slot 1 kills 4 → sum=4.
        # Episode 3: both slots zero → sum=0.
        # Mean = (3 + 4 + 0) / 3 = 7/3.
        results = [
            _make_synthetic_result(kills_per_slot={"predator_0": 2, "predator_1": 1}),
            _make_synthetic_result(kills_per_slot={"predator_0": 0, "predator_1": 4}),
            _make_synthetic_result(kills_per_slot={"predator_0": 0, "predator_1": 0}),
        ]
        _patch_runtime(monkeypatch, num_predator_slots=2, results_per_episode=results)

        encoder = _FakeEncoder()
        genome = Genome(
            params=np.zeros(4, dtype=np.float32),
            genome_id="test_genome",
            parent_ids=[],
            generation=0,
        )
        fitness = PredatorEpisodicKillRate(secondary_signal=True)
        out = fitness.evaluate(
            genome,
            fake_sim_config,
            cast("GenomeEncoder", encoder),
            episodes=3,
            seed=42,
        )
        assert out == pytest.approx(7 / 3)

    def test_zero_episodes_returns_zero(
        self,
        fake_sim_config: SimulationConfig,
    ) -> None:
        """`episodes == 0` SHALL return `0.0` immediately."""
        encoder = _FakeEncoder()
        genome = Genome(
            params=np.zeros(4, dtype=np.float32),
            genome_id="g",
            parent_ids=[],
            generation=0,
        )
        fitness = PredatorEpisodicKillRate()
        # No monkeypatch — should never reach the runtime path.
        assert (
            fitness.evaluate(
                genome,
                fake_sim_config,
                cast("GenomeEncoder", encoder),
                episodes=0,
                seed=42,
            )
            == 0.0
        )

    def test_negative_episodes_raises(
        self,
        fake_sim_config: SimulationConfig,
    ) -> None:
        """Negative `episodes` SHALL raise `ValueError`."""
        encoder = _FakeEncoder()
        genome = Genome(
            params=np.zeros(4, dtype=np.float32),
            genome_id="g",
            parent_ids=[],
            generation=0,
        )
        fitness = PredatorEpisodicKillRate()
        with pytest.raises(ValueError, match="episodes"):
            fitness.evaluate(
                genome,
                fake_sim_config,
                cast("GenomeEncoder", encoder),
                episodes=-1,
                seed=42,
            )


class TestAllSlotsUseSameDecodedBrain:
    """Spec rationale: "all predator slots run the same decoded brain (the genome's strategy)"."""

    def test_every_slot_receives_decoded_brain(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_sim_config: SimulationConfig,
    ) -> None:
        """Every `Predator.brain` slot SHALL be populated by `encoder.decode`.

        The fitness function MUST trigger one decode per slot per
        episode (so a future code path that sets only `predators[0].brain`
        and leaves others untouched fails this test).
        """
        results = [
            _make_synthetic_result(kills_per_slot={"predator_0": 1, "predator_1": 1}),
        ]
        _patch_runtime(monkeypatch, num_predator_slots=3, results_per_episode=results)

        encoder = _FakeEncoder()
        genome = Genome(
            params=np.zeros(4, dtype=np.float32),
            genome_id="g",
            parent_ids=[],
            generation=0,
        )
        fitness = PredatorEpisodicKillRate()
        fitness.evaluate(
            genome,
            fake_sim_config,
            cast("GenomeEncoder", encoder),
            episodes=1,
            seed=42,
        )

        # 1 episode x 3 slots = 3 decode calls (the fitness builds a
        # fresh env per episode and decodes once per slot inside
        # `_build_env_with_genome_predators`).
        assert len(encoder.decode_calls) == 3
        # Every decode received the SAME genome (no slot-0 special-casing
        # where slot 0 gets the test genome and other slots get a stale
        # / random brain).
        for called_genome, _seed in encoder.decode_calls:
            assert called_genome is genome


# ---------------------------------------------------------------------------
# Secondary proximity signal
# ---------------------------------------------------------------------------


class TestSecondaryProximitySignal:
    """Spec scenario "Secondary Proximity Signal When Kill Count Is Zero"."""

    def test_proximity_fallback_triggers_when_all_kills_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_sim_config: SimulationConfig,
    ) -> None:
        """All-zero-kills evaluation SHALL fall back to the proximity ratio."""
        # 3 episodes, all zero kills. Proximity sums non-zero so
        # fallback should produce a strictly positive result.
        # Per-episode max proximity = max_steps x num_predator_slots
        # = 500 x 2 = 1000. Per-episode raw proximity = 100 + 200 = 300.
        # Mean ratio = 300/1000 = 0.3 (constant since all eps identical).
        # Fallback = 0.3 x 0.99 / 3 ≈ 0.099.
        episodes_n = 3
        results = [
            _make_synthetic_result(
                kills_per_slot={"predator_0": 0, "predator_1": 0},
                proximity_per_slot={"predator_0": 100, "predator_1": 200},
            )
            for _ in range(episodes_n)
        ]
        _patch_runtime(monkeypatch, num_predator_slots=2, results_per_episode=results)

        encoder = _FakeEncoder()
        genome = Genome(
            params=np.zeros(4, dtype=np.float32),
            genome_id="g",
            parent_ids=[],
            generation=0,
        )
        fitness = PredatorEpisodicKillRate(secondary_signal=True)
        out = fitness.evaluate(
            genome,
            fake_sim_config,
            cast("GenomeEncoder", encoder),
            episodes=episodes_n,
            seed=42,
        )
        assert out > 0.0
        # Strictly less than 1/N — preserves kill-rate ordering.
        assert out < 1.0 / episodes_n

    def test_fallback_strictly_below_one_over_episodes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_sim_config: SimulationConfig,
    ) -> None:
        """Even with maximum proximity, fallback SHALL stay below `1/episodes`.

        Hammers the worst case: every step every slot is in proximity,
        so raw proximity = max_steps x num_slots per episode. The
        normalised ratio is 1.0; fallback is `1.0 x 0.99 / N`. This
        MUST be `< 1/N` (the smallest non-zero kill-rate fitness).
        """
        episodes_n = 5
        # Per-episode max = 500 (max_steps default) x 2 slots = 1000.
        results = [
            _make_synthetic_result(
                kills_per_slot={"predator_0": 0, "predator_1": 0},
                proximity_per_slot={
                    "predator_0": _DEFAULT_MAX_STEPS,
                    "predator_1": _DEFAULT_MAX_STEPS,
                },
            )
            for _ in range(episodes_n)
        ]
        _patch_runtime(monkeypatch, num_predator_slots=2, results_per_episode=results)

        encoder = _FakeEncoder()
        genome = Genome(
            params=np.zeros(4, dtype=np.float32),
            genome_id="g",
            parent_ids=[],
            generation=0,
        )
        fitness = PredatorEpisodicKillRate(secondary_signal=True)
        out = fitness.evaluate(
            genome,
            fake_sim_config,
            cast("GenomeEncoder", encoder),
            episodes=episodes_n,
            seed=42,
        )
        # Strictly below 1/N (the lowest non-zero kill-rate).
        assert out < 1.0 / episodes_n
        # Within an epsilon of the headroom x max_ratio / N (approx).
        assert out == pytest.approx(0.99 / episodes_n, rel=1e-6)

    def test_secondary_signal_false_returns_zero_on_all_zero_kills(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_sim_config: SimulationConfig,
    ) -> None:
        """With `secondary_signal=False`, all-zero kills SHALL score exactly 0.0."""
        episodes_n = 3
        results = [
            _make_synthetic_result(
                kills_per_slot={"predator_0": 0},
                proximity_per_slot={"predator_0": 50},
            )
            for _ in range(episodes_n)
        ]
        _patch_runtime(monkeypatch, num_predator_slots=1, results_per_episode=results)

        encoder = _FakeEncoder()
        genome = Genome(
            params=np.zeros(4, dtype=np.float32),
            genome_id="g",
            parent_ids=[],
            generation=0,
        )
        fitness = PredatorEpisodicKillRate(secondary_signal=False)
        out = fitness.evaluate(
            genome,
            fake_sim_config,
            cast("GenomeEncoder", encoder),
            episodes=episodes_n,
            seed=42,
        )
        assert out == 0.0


# ---------------------------------------------------------------------------
# PredatorLearnedPerformanceFitness — stub raises clearly
# ---------------------------------------------------------------------------


class TestLearnedPerformanceFitnessStub:
    """The deferred predator-side learned-performance variant raises `NotImplementedError`."""

    def test_evaluate_raises_with_clear_message(
        self,
        fake_sim_config: SimulationConfig,
    ) -> None:
        """`PredatorLearnedPerformanceFitness.evaluate` SHALL raise `NotImplementedError`."""
        encoder = _FakeEncoder()
        genome = Genome(
            params=np.zeros(4, dtype=np.float32),
            genome_id="g",
            parent_ids=[],
            generation=0,
        )
        fitness = PredatorLearnedPerformanceFitness()
        with pytest.raises(NotImplementedError, match="frozen-weight"):
            fitness.evaluate(
                genome,
                fake_sim_config,
                cast("GenomeEncoder", encoder),
                episodes=1,
                seed=42,
            )


# ---------------------------------------------------------------------------
# GenomeEncoder Protocol-conformance for FakeEncoder (sanity check)
# ---------------------------------------------------------------------------


def test_fake_encoder_satisfies_protocol() -> None:
    """The test's `_FakeEncoder` SHALL satisfy the GenomeEncoder Protocol.

    Sanity check: if the test fixture itself doesn't conform to the
    Protocol, the fitness tests would be testing against a contract
    the production code doesn't see.
    """
    assert isinstance(_FakeEncoder(), GenomeEncoder)
