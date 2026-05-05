"""Predator brain abstraction.

Defines the pluggable policy seam for predators: a `PredatorBrain` Protocol,
a minimal `PredatorBrainParams` input dataclass, the `PredatorAction` action
enum, and a `HeuristicPredatorBrain` adapter that encapsulates the original
heuristic movement logic (stationary / random / greedy-pursuit) that used
to live inline in `Predator._update_pursuit` and `Predator._update_random`.

Future learnable predator brains (e.g. MLPPPO predators for co-evolution)
implement the same Protocol surface and slot in via the env's predator-brain
dispatcher; the harness owns kinematics (accumulator, grid clamp) so brains
only need to produce intent (one of `PredatorAction`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from quantumnematode.env.env import PredatorType


class PredatorAction(Enum):
    """Cardinal-grid action emitted by a `PredatorBrain.run_brain` call.

    The harness (`Predator._apply_action`) translates the action into a
    position delta, advances the movement accumulator, and applies the
    grid clamp. Brains never mutate `Predator.position` directly.
    """

    STAY = "stay"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class PredatorBrainParams:
    """Per-step input bundle passed to `PredatorBrain.run_brain`.

    The frozen-branch invariant: `chase_target` and `is_pursuing` are
    pre-resolved by `Predator.update_position` ONCE per call and passed
    unchanged across every accumulator-loop iteration in that call. This
    matches the legacy semantics where `_update_pursuit`'s in-range check
    happens once at the top of the call and stays committed for all
    multi-step movements within the call.

    Only `predator_position` varies per accumulator-step within a single
    `update_position` call.

    Attributes
    ----------
    predator_id : str
        Synthesised identifier for the predator (e.g. `"predator_0"`).
        Stable across env `reset()` calls within a single env instance,
        and reproducible across env instances given the same config + seed.
    predator_position : tuple[int, int]
        Current `(x, y)` of the predator at the start of THIS accumulator-
        step. Updates between `run_brain` invocations within the same call
        for multi-step movement at `speed > 1.0`.
    predator_type : PredatorType
        The predator's behavior enum (STATIONARY / PURSUIT). Frozen for
        the lifetime of the predator.
    detection_radius : int
        The predator's detection radius (Manhattan).
    damage_radius : int
        The predator's damage radius (Manhattan). Available to learnable
        brains as a feature; the heuristic does not use it directly.
    agent_positions : tuple[tuple[int, int], ...]
        Tuple of alive prey positions, ordered by env's `agents.values()`
        insertion order. Empty tuple if no prey alive. Future learnable
        brains can use this for raw observations; the heuristic uses
        the pre-resolved `chase_target` instead.
    chase_target : tuple[int, int] | None
        Resolved nearest-by-Manhattan agent from `agent_positions` at the
        START of the `update_position` call. Frozen across all accumulator-
        steps in the call. `None` if no agents are alive or for STATIONARY
        predators (which never chase).
    is_pursuing : bool
        Frozen pursuit-mode flag. True iff predator is PURSUIT type AND
        `chase_target is not None` AND Manhattan distance from
        `predator_position` (at start of call) to `chase_target` is
        `<= detection_radius`. Determines branch (greedy chase vs random)
        for the entire accumulator loop.
    grid_size : int
        Env's grid size, available to brains that want normalised position
        features. The harness owns the actual bounds clamp.
    rng : numpy.random.Generator
        The env's RNG. Heuristic random branch consumes one `integers(4)`
        draw per call; future stochastic brains share the same generator
        to keep RNG-state advancement deterministic across downstream
        consumers (food spawning, agent decisions).
    step_index : int
        Episode-level step counter at the start of the `update_position`
        call. Frozen across accumulator-steps within the call. Available
        to time-aware policies (e.g. learning-rate schedules).
    """

    predator_id: str
    predator_position: tuple[int, int]
    predator_type: PredatorType
    detection_radius: int
    damage_radius: int
    agent_positions: tuple[tuple[int, int], ...]
    chase_target: tuple[int, int] | None
    is_pursuing: bool
    grid_size: int
    rng: np.random.Generator
    step_index: int


@runtime_checkable
class PredatorBrain(Protocol):
    """Pluggable predator-policy Protocol.

    Mirrors the agent-side `Brain` Protocol's lifecycle (prepare / run /
    post-process / copy) but with a predator-specific input bundle and
    a discrete cardinal-direction action space.

    `prepare_episode` and `post_process_episode` are no-ops for the
    heuristic but are required by the Protocol so future learnable
    predator brains have guaranteed lifecycle hooks (e.g. RL brains
    that need to reset hidden state at episode start).
    """

    def run_brain(self, params: PredatorBrainParams) -> PredatorAction:
        """Decide one cardinal action given the predator's current params."""
        ...

    def prepare_episode(self) -> None:
        """Reset per-episode state (no-op for stateless brains)."""
        ...

    def post_process_episode(
        self,
        *,
        episode_success: bool | None = None,
    ) -> None:
        """Finalise per-episode state (no-op for stateless brains)."""
        ...

    def copy(self) -> PredatorBrain:
        """Return an independent copy with the same logical state."""
        ...


@dataclass(frozen=True)
class PredatorBrainConfig:
    """Runtime brain-config dataclass referenced by `PredatorParams`.

    YAML loading produces the Pydantic-validated `PredatorBrainConfigSchema`
    in `config_loader`; that schema's `to_params()` returns this dataclass
    for runtime use. Two-type pattern matches the existing
    `PredatorParams` (dataclass) ↔ `PredatorConfig` (Pydantic) split.

    `kind` selects which brain implementation `_build_predator_brain`
    constructs. Currently only `"heuristic"` is honoured; the literal
    type can be extended with learnable kinds (e.g. `"mlpppo"`) when
    learnable predator brains are introduced.
    """

    kind: Literal["heuristic"] = "heuristic"
    extra: dict[str, Any] | None = None


class HeuristicPredatorBrain:
    """Adapter encapsulating the legacy heuristic movement logic.

    Byte-equivalent to the pre-refactor `Predator._update_pursuit` and
    `Predator._update_random` helpers. Action choice depends entirely on
    the pre-resolved `is_pursuing` flag and the current `predator_position`
    in `PredatorBrainParams`:

    - STATIONARY predator → always returns `PredatorAction.STAY`.
    - `is_pursuing=True` → greedy chase: move on the larger-delta axis
      first (`if abs(dx) >= abs(dy)`), tie-breaking horizontal-first.
      No RNG consumption.
    - `is_pursuing=False` → random: a single `rng.integers(4)` draw,
      mapped `0/1/2/3 → UP/DOWN/LEFT/RIGHT` exactly as legacy.

    The frozen-branch invariant ensures multi-step movement at `speed >
    1.0` does not retarget mid-call (legacy `_update_pursuit` decides
    in-range once at the top, then loops accumulator-steps). The
    `params.is_pursuing` flag is what enforces this here.
    """

    def __init__(self) -> None:
        # No state; heuristic is pure function of params.
        # __init__ is kept (rather than removed) so future variants can
        # add state without changing the Protocol surface.
        pass

    def run_brain(self, params: PredatorBrainParams) -> PredatorAction:
        """Return one cardinal action per accumulator-step (pure of self).

        STATIONARY → STAY. is_pursuing=True → greedy axis selection.
        Else → single rng.integers(4) draw mapped to a cardinal direction.
        """
        # STATIONARY predators never move (matches legacy update_position
        # early-return at env.py STATIONARY check).
        if params.predator_type.value == "stationary":
            return PredatorAction.STAY

        if params.is_pursuing and params.chase_target is not None:
            return _greedy_action(params.predator_position, params.chase_target)

        # Random branch: same RNG draw + direction mapping as legacy
        # _update_random. The harness applies the action with grid clamp.
        return _random_action(params.rng)

    def prepare_episode(self) -> None:
        """No-op (stateless brain has no per-episode state to reset)."""
        return

    def post_process_episode(
        self,
        *,
        episode_success: bool | None = None,
    ) -> None:
        """No-op (stateless brain has no per-episode state to finalise)."""
        del episode_success

    def copy(self) -> HeuristicPredatorBrain:
        """Return a fresh independent HeuristicPredatorBrain instance.

        Stateless brain — fresh instance is independent of self. Future
        stateful brains override copy() to deep-copy their state.
        """
        return HeuristicPredatorBrain()


# ---------------------------------------------------------------------------
# Internal helpers — pure functions kept module-private.
# ---------------------------------------------------------------------------

# Index → direction mapping for the random branch. Ordering MUST match the
# legacy random-direction mapping (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT) preserved
# in `_legacy_predator_reference.py`, so RNG-state advancement stays one-
# for-one with the original heuristic.
_RANDOM_DIRECTION_BY_INDEX = (
    PredatorAction.UP,
    PredatorAction.DOWN,
    PredatorAction.LEFT,
    PredatorAction.RIGHT,
)


def _greedy_action(
    predator_position: tuple[int, int],
    target: tuple[int, int],
) -> PredatorAction:
    """Pure-function greedy chase: move on the larger-delta axis first.

    Mirrors the legacy `_update_pursuit` inner-loop branch at env.py
    line 668-679 exactly. Tie-breaking on `abs(dx) == abs(dy)` favours
    horizontal-first (the `if abs(dx) >= abs(dy)` check uses `>=`).
    Returns `STAY` only if predator is already at target — legacy code
    did the same implicitly via the `dx > 0 / dx < 0 / dy > 0 / dy < 0`
    branches all evaluating False.
    """
    px, py = predator_position
    ax, ay = target
    dx = ax - px
    dy = ay - py

    if abs(dx) >= abs(dy):
        if dx > 0:
            return PredatorAction.RIGHT
        if dx < 0:
            return PredatorAction.LEFT
        # dx == 0 and abs(dy) <= abs(dx) == 0, so dy == 0 too.
        return PredatorAction.STAY
    if dy > 0:
        return PredatorAction.DOWN
    if dy < 0:
        return PredatorAction.UP
    return PredatorAction.STAY


def _random_action(rng: np.random.Generator) -> PredatorAction:
    """Pure-function random direction draw — one `integers(4)` per call.

    Direction mapping `0/1/2/3 → UP/DOWN/LEFT/RIGHT` matches the legacy
    `_update_random` order (env.py 600-615). Critical for byte-equivalence:
    the env's RNG state advancement must match the legacy code one-for-one.
    """
    direction_choice = int(rng.integers(4))
    return _RANDOM_DIRECTION_BY_INDEX[direction_choice]
