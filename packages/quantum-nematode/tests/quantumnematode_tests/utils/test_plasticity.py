"""Tests for the plasticity evaluation protocol.

Covers:
- PlasticityConfig validation (9.1)
- Metrics computation — BF, FT, PR (9.2)
- State snapshot/restore helper (9.3)
"""

from __future__ import annotations

import pytest
from quantumnematode.plasticity import (
    EvalResult,
    PhaseTrainingResult,
    SeedResult,
    compute_convergence_episode,
    compute_seed_metrics,
    restore_brain_state,
    snapshot_brain_state,
)
from quantumnematode.utils.config_loader import (
    EnvironmentConfig,
    PlasticityConfig,
    PlasticityPhaseConfig,
    PlasticityProtocolConfig,
)

# ---------------------------------------------------------------------------
# 9.1 PlasticityConfig validation
# ---------------------------------------------------------------------------


class TestPlasticityConfig:
    """Tests for PlasticityConfig Pydantic model validation."""

    @staticmethod
    def _make_minimal_phases() -> list[dict]:
        return [
            {"name": "foraging", "environment": {"grid_size": 15}},
            {"name": "pursuit", "environment": {"grid_size": 15}},
            {"name": "foraging_return", "environment": {"grid_size": 15}},
        ]

    def test_valid_config_loads(self) -> None:
        """Valid config with all required fields loads successfully."""
        config = PlasticityConfig(
            brain={"name": "mlpppo"},  # type: ignore[arg-type]
            plasticity=PlasticityProtocolConfig(
                training_episodes_per_phase=10,
                eval_episodes=5,
                seeds=[42],
                phases=[
                    PlasticityPhaseConfig(
                        name=p["name"],
                        environment=EnvironmentConfig(**p["environment"]),
                    )
                    for p in self._make_minimal_phases()
                ],
            ),
        )
        assert config.brain.name == "mlpppo"
        assert config.plasticity.training_episodes_per_phase == 10
        assert len(config.plasticity.phases) == 3

    def test_convergence_threshold_defaults_to_0_6(self) -> None:
        """convergence_threshold defaults to 0.6 if not specified."""
        protocol = PlasticityProtocolConfig(
            training_episodes_per_phase=10,
            eval_episodes=5,
            seeds=[42],
            phases=[
                PlasticityPhaseConfig(name="a", environment=EnvironmentConfig()),
                PlasticityPhaseConfig(name="b", environment=EnvironmentConfig()),
                PlasticityPhaseConfig(name="c", environment=EnvironmentConfig()),
            ],
        )
        assert protocol.convergence_threshold == pytest.approx(0.6)

    def test_custom_convergence_threshold(self) -> None:
        """convergence_threshold can be overridden."""
        protocol = PlasticityProtocolConfig(
            training_episodes_per_phase=10,
            eval_episodes=5,
            seeds=[42],
            convergence_threshold=0.8,
            phases=[
                PlasticityPhaseConfig(name="a", environment=EnvironmentConfig()),
                PlasticityPhaseConfig(name="b", environment=EnvironmentConfig()),
                PlasticityPhaseConfig(name="c", environment=EnvironmentConfig()),
            ],
        )
        assert protocol.convergence_threshold == pytest.approx(0.8)

    def test_missing_brain_rejected(self) -> None:
        """Config without brain field raises validation error."""
        with pytest.raises(Exception):  # noqa: B017, PT011
            PlasticityConfig(
                plasticity=PlasticityProtocolConfig(  # type: ignore[call-arg]
                    training_episodes_per_phase=10,
                    eval_episodes=5,
                    seeds=[42],
                    phases=[
                        PlasticityPhaseConfig(name="a", environment=EnvironmentConfig()),
                        PlasticityPhaseConfig(name="b", environment=EnvironmentConfig()),
                        PlasticityPhaseConfig(name="c", environment=EnvironmentConfig()),
                    ],
                ),
            )

    def test_phase_reward_defaults(self) -> None:
        """Phase reward config defaults to RewardConfig() if not specified."""
        from quantumnematode.agent import RewardConfig

        phase = PlasticityPhaseConfig(
            name="test",
            environment=EnvironmentConfig(),
        )
        assert isinstance(phase.reward, RewardConfig)

    def test_phase_satiety_optional(self) -> None:
        """Phase satiety config is optional (defaults to None)."""
        phase = PlasticityPhaseConfig(
            name="test",
            environment=EnvironmentConfig(),
        )
        assert phase.satiety is None


# ---------------------------------------------------------------------------
# 9.2 Metrics computation
# ---------------------------------------------------------------------------


class TestMetricsComputation:
    """Tests for BF, FT, PR metrics with known inputs."""

    def test_backward_forgetting(self) -> None:
        """BF = post_A_score - post_C_score_on_A."""
        seed_result = SeedResult(seed=42)
        seed_result.eval_results = [
            EvalResult("foraging", "post_A", 0.8, 10.0, 100.0),
            EvalResult("foraging", "post_C", 0.3, 5.0, 200.0),
        ]
        seed_result.training_results = []
        compute_seed_metrics(seed_result, convergence_threshold=0.6)
        assert seed_result.backward_forgetting == pytest.approx(0.5)

    def test_forward_transfer(self) -> None:
        """FT = post_A_eval_on_B - random_baseline_on_B."""
        seed_result = SeedResult(seed=42)
        seed_result.eval_results = [
            EvalResult("pursuit_predators", "pre_training", 0.1, 1.0, 300.0),
            EvalResult("pursuit_predators", "post_A", 0.3, 3.0, 250.0),
        ]
        seed_result.training_results = []
        compute_seed_metrics(seed_result, convergence_threshold=0.6)
        assert seed_result.forward_transfer == pytest.approx(0.2)

    def test_plasticity_retention_converging(self) -> None:
        """PR = convergence_episodes_A / convergence_episodes_A' when both converge."""
        # Phase A converges at episode 40 (success rate reaches 0.6 in trailing 20)
        phase_a = PhaseTrainingResult(phase_name="foraging")
        phase_a.episode_successes = [False] * 20 + [True] * 180

        # Phase A' converges at episode 20 (faster relearning)
        phase_a_prime = PhaseTrainingResult(phase_name="foraging_return")
        phase_a_prime.episode_successes = [True] * 200

        seed_result = SeedResult(seed=42)
        seed_result.training_results = [phase_a, phase_a_prime]
        seed_result.eval_results = []
        compute_seed_metrics(seed_result, convergence_threshold=0.6)

        # Phase A: trailing-20 at index 31 = episodes[12..31] = 8 False + 12 True = 60%
        # → converges at episode 32. Phase A' converges at episode 20 (all True).
        # PR = 32/20 = 1.6 > 1.0 (A' relearns faster).
        assert seed_result.plasticity_retention is not None
        assert seed_result.plasticity_retention > 1.0  # A' converges faster

    def test_plasticity_retention_no_convergence(self) -> None:
        """PR is None when a phase does not converge."""
        # Neither phase converges (all failures)
        phase_a = PhaseTrainingResult(phase_name="foraging")
        phase_a.episode_successes = [False] * 200

        phase_a_prime = PhaseTrainingResult(phase_name="foraging_return")
        phase_a_prime.episode_successes = [False] * 200

        seed_result = SeedResult(seed=42)
        seed_result.training_results = [phase_a, phase_a_prime]
        seed_result.eval_results = []
        compute_seed_metrics(seed_result, convergence_threshold=0.6)
        assert seed_result.plasticity_retention is None

    def test_convergence_episode_computation(self) -> None:
        """Verify convergence episode detection with known sequence."""
        # Converge at episode 20 (window [1..20] all True = 100%)
        successes = [True] * 200
        assert compute_convergence_episode(successes, threshold=0.6, window=20) == 20

        # Never converge
        failures = [False] * 200
        assert compute_convergence_episode(failures, threshold=0.6, window=20) is None

        # Converge midway
        mixed = [False] * 30 + [True] * 170
        result = compute_convergence_episode(mixed, threshold=0.6, window=20)
        assert result is not None
        # Window must have ≥12 True out of 20 (60%)
        # At episode 42: window [23..42] = 8 False + 12 True = 60%
        assert result == 42


# ---------------------------------------------------------------------------
# 9.3 State snapshot/restore
# ---------------------------------------------------------------------------


class TestStateSnapshotRestore:
    """Tests for brain state snapshot and restore."""

    def test_snapshot_restore_mlpppo(self) -> None:
        """Verify MLPPPOBrain state is identical after snapshot → modify → restore."""
        import torch
        from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig

        config = MLPPPOBrainConfig(seed=42)
        brain = MLPPPOBrain(config)

        # Snapshot
        snapshot = snapshot_brain_state(brain)

        # Modify weights
        with torch.no_grad():
            for param in brain.actor.parameters():
                param.add_(torch.randn_like(param))

        # Verify weights changed
        current = snapshot_brain_state(brain)
        actor_changed = False
        for key in snapshot.get("actor", {}):
            if not torch.equal(snapshot["actor"][key], current["actor"][key]):
                actor_changed = True
                break
        assert actor_changed, "Weights should have changed after modification"

        # Restore
        restore_brain_state(brain, snapshot)

        # Verify weights restored
        restored = snapshot_brain_state(brain)
        for key in snapshot.get("actor", {}):
            assert torch.equal(snapshot["actor"][key], restored["actor"][key]), (
                f"Actor weight {key} not restored correctly"
            )
