"""Tests for the predator brain abstraction (Protocol + heuristic adapter).

Covers task 1.6 from add-learning-predators OpenSpec change:
- Protocol conformance via @runtime_checkable isinstance
- HeuristicPredatorBrain.copy() returns independent instance
- Stationary predator returns STAY regardless of params
- In-range pursuit greedy axis selection (with horizontal-first tie-break)
- Out-of-range pursuit falls back to random with seeded RNG → expected direction
- Random branch consumes exactly one rng.integers(4) draw per call
- No-op prepare_episode / post_process_episode (lifecycle hooks)

The byte-equivalence parametrised test against the legacy `_update_pursuit` /
`_update_random` helpers lives in M1.2 (task 2.6) — once the legacy code
is replaced. These tests verify the heuristic's standalone semantics.
"""

import numpy as np
import pytest
from quantumnematode.env import (
    HeuristicPredatorBrain,
    PredatorAction,
    PredatorBrain,
    PredatorBrainParams,
    PredatorType,
)


def _make_params(  # noqa: PLR0913
    *,
    predator_type: PredatorType = PredatorType.PURSUIT,
    predator_position: tuple[int, int] = (5, 5),
    chase_target: tuple[int, int] | None = (8, 5),
    is_pursuing: bool = True,
    detection_radius: int = 8,
    damage_radius: int = 0,
    agent_positions: tuple[tuple[int, int], ...] = ((8, 5),),
    grid_size: int = 20,
    rng: np.random.Generator | None = None,
    step_index: int = 0,
    predator_id: str = "predator_0",
) -> PredatorBrainParams:
    """Build a PredatorBrainParams with sensible defaults; override per-test."""
    return PredatorBrainParams(
        predator_id=predator_id,
        predator_position=predator_position,
        predator_type=predator_type,
        detection_radius=detection_radius,
        damage_radius=damage_radius,
        agent_positions=agent_positions,
        chase_target=chase_target,
        is_pursuing=is_pursuing,
        grid_size=grid_size,
        rng=rng if rng is not None else np.random.default_rng(42),
        step_index=step_index,
    )


class TestProtocolConformance:
    """HeuristicPredatorBrain SHALL satisfy the PredatorBrain Protocol."""

    def test_isinstance_via_runtime_checkable(self) -> None:
        """Verify isinstance via runtime checkable."""
        brain = HeuristicPredatorBrain()
        assert isinstance(brain, PredatorBrain)

    def test_protocol_methods_exist(self) -> None:
        """Verify protocol methods exist."""
        brain = HeuristicPredatorBrain()
        # All four required Protocol methods must be callable.
        assert callable(brain.run_brain)
        assert callable(brain.prepare_episode)
        assert callable(brain.post_process_episode)
        assert callable(brain.copy)

    def test_copy_returns_independent_instance(self) -> None:
        """Verify copy returns independent instance."""
        original = HeuristicPredatorBrain()
        clone = original.copy()
        assert clone is not original
        assert isinstance(clone, HeuristicPredatorBrain)
        # Behavioural equality: same params → same action.
        params = _make_params()
        assert original.run_brain(params) == clone.run_brain(params)


class TestStationaryAlwaysStays:
    """STATIONARY predators SHALL return STAY regardless of other params."""

    def test_returns_stay_when_pursuing_in_range(self) -> None:
        """Verify returns stay when pursuing in range."""
        brain = HeuristicPredatorBrain()
        params = _make_params(
            predator_type=PredatorType.STATIONARY,
            is_pursuing=True,
            chase_target=(8, 5),
        )
        assert brain.run_brain(params) == PredatorAction.STAY

    def test_returns_stay_when_no_target(self) -> None:
        """Verify returns stay when no target."""
        brain = HeuristicPredatorBrain()
        params = _make_params(
            predator_type=PredatorType.STATIONARY,
            is_pursuing=False,
            chase_target=None,
            agent_positions=(),
        )
        assert brain.run_brain(params) == PredatorAction.STAY


class TestPursuitGreedyAxisSelection:
    """In-range PURSUIT predators SHALL move greedy on the larger-delta axis."""

    def test_greedy_horizontal_when_dx_larger(self) -> None:
        """Verify greedy horizontal when dx larger."""
        brain = HeuristicPredatorBrain()
        # predator at (5,5), agent at (8,5): dx=3, dy=0 → RIGHT
        params = _make_params(
            predator_position=(5, 5),
            chase_target=(8, 5),
            is_pursuing=True,
        )
        assert brain.run_brain(params) == PredatorAction.RIGHT

    def test_greedy_horizontal_when_dx_negative(self) -> None:
        """Verify greedy horizontal when dx negative."""
        brain = HeuristicPredatorBrain()
        params = _make_params(
            predator_position=(5, 5),
            chase_target=(2, 5),
            is_pursuing=True,
        )
        assert brain.run_brain(params) == PredatorAction.LEFT

    def test_greedy_vertical_when_dy_larger(self) -> None:
        """Verify greedy vertical when dy larger."""
        brain = HeuristicPredatorBrain()
        # predator at (5,5), agent at (5,8): dx=0, dy=3 → DOWN
        params = _make_params(
            predator_position=(5, 5),
            chase_target=(5, 8),
            is_pursuing=True,
        )
        assert brain.run_brain(params) == PredatorAction.DOWN

    def test_greedy_vertical_when_dy_negative(self) -> None:
        """Verify greedy vertical when dy negative."""
        brain = HeuristicPredatorBrain()
        params = _make_params(
            predator_position=(5, 5),
            chase_target=(5, 2),
            is_pursuing=True,
        )
        assert brain.run_brain(params) == PredatorAction.UP

    def test_horizontal_first_tiebreak_when_dx_equals_dy(self) -> None:
        """Verify horizontal first tiebreak when dx equals dy."""
        brain = HeuristicPredatorBrain()
        # predator at (5,5), agent at (8,8): abs(dx)=3 == abs(dy)=3 → RIGHT
        # (legacy `if abs(dx) >= abs(dy)` precedence at env.py:669)
        params = _make_params(
            predator_position=(5, 5),
            chase_target=(8, 8),
            is_pursuing=True,
        )
        assert brain.run_brain(params) == PredatorAction.RIGHT

    def test_returns_stay_when_already_at_target(self) -> None:
        """Verify returns stay when already at target."""
        brain = HeuristicPredatorBrain()
        params = _make_params(
            predator_position=(5, 5),
            chase_target=(5, 5),
            is_pursuing=True,
        )
        assert brain.run_brain(params) == PredatorAction.STAY

    def test_pursuit_does_not_consume_rng(self) -> None:
        """Verify pursuit does not consume rng."""
        brain = HeuristicPredatorBrain()
        rng = np.random.default_rng(42)
        # Capture state BEFORE the call
        state_before = rng.bit_generator.state
        params = _make_params(
            predator_position=(5, 5),
            chase_target=(8, 5),
            is_pursuing=True,
            rng=rng,
        )
        brain.run_brain(params)
        # State unchanged: pursuit branch is RNG-free
        assert rng.bit_generator.state == state_before


class TestRandomBranch:
    """Out-of-range / no-target SHALL trigger random branch with one RNG draw."""

    def test_random_branch_uses_one_rng_draw(self) -> None:
        """Verify random branch uses one rng draw."""
        brain = HeuristicPredatorBrain()
        rng = np.random.default_rng(42)
        # Sister rng: same seed, used to verify draw count
        sister = np.random.default_rng(42)
        params = _make_params(is_pursuing=False, rng=rng)
        brain.run_brain(params)
        # The brain should consume exactly one rng.integers(4) draw.
        # Sister rng draws once; both should now match.
        sister.integers(4)
        assert rng.bit_generator.state == sister.bit_generator.state

    @pytest.mark.parametrize(
        ("draw_value", "expected"),
        [
            (0, PredatorAction.UP),
            (1, PredatorAction.DOWN),
            (2, PredatorAction.LEFT),
            (3, PredatorAction.RIGHT),
        ],
    )
    def test_random_direction_mapping(
        self,
        draw_value: int,
        expected: PredatorAction,
    ) -> None:
        """Direction mapping 0/1/2/3 -> UP/DOWN/LEFT/RIGHT (matches legacy env.py:608-615)."""
        brain = HeuristicPredatorBrain()
        # Seed-search for an rng whose first draw matches draw_value.
        for seed in range(500):
            r_check = np.random.default_rng(seed)
            if int(r_check.integers(4)) == draw_value:
                params = _make_params(
                    is_pursuing=False,
                    rng=np.random.default_rng(seed),
                )
                assert brain.run_brain(params) == expected
                return
        pytest.fail(f"could not find seed producing integers(4) == {draw_value}")

    def test_random_when_no_chase_target(self) -> None:
        """No agents alive ⇒ chase_target=None ⇒ is_pursuing=False ⇒ random branch."""
        brain = HeuristicPredatorBrain()
        rng = np.random.default_rng(42)
        sister = np.random.default_rng(42)
        params = _make_params(
            chase_target=None,
            is_pursuing=False,
            agent_positions=(),
            rng=rng,
        )
        action = brain.run_brain(params)
        sister_draw = int(sister.integers(4))
        expected = [
            PredatorAction.UP,
            PredatorAction.DOWN,
            PredatorAction.LEFT,
            PredatorAction.RIGHT,
        ][sister_draw]
        assert action == expected


class TestLifecycleHooks:
    """prepare_episode and post_process_episode SHALL be no-op for heuristic."""

    def test_prepare_episode_returns_none(self) -> None:
        """Verify prepare episode returns none."""
        brain = HeuristicPredatorBrain()
        assert brain.prepare_episode() is None

    def test_post_process_episode_returns_none(self) -> None:
        """Verify post process episode returns none."""
        brain = HeuristicPredatorBrain()
        assert brain.post_process_episode() is None
        assert brain.post_process_episode(episode_success=True) is None
        assert brain.post_process_episode(episode_success=False) is None
