"""Run many simulated worms under one frozen controller in lockstep, vectorised over animals.

The per-step runner advances one animal at a time through Python objects. For a frozen controller
almost all of a step is per-animal arithmetic that vectorises:

- the lawn odour at the worm and at its two head-sweep points;
- the short-term memory buffer and its derivative;
- the adaptive fold-change readout;
- the primitives' moves, eating and satiety.

This module holds every per-animal state as arrays over N animals and runs a step for all of them
at once, in the per-step runner's order:

1. any assay phase that starts;
2. sensing;
3. the controller;
4. the primitive move;
5. intake, which restores satiety;
6. satiety's decay.

**Randomness per animal.** Animal *i*, with seed ``s_i``, has the environment the per-step runner
would build for an episode with that seed: it is constructed from the config at ``s_i`` and keeps
its own numpy generator. Its policy noise comes from a torch generator seeded ``s_i``, five standard
normals a step, which is what ``set_global_seed(s_i)`` gives the per-step runner's single sample. An
animal therefore behaves the same alone or in any batch.

**Exactness.** The environment objects are used only where exactness is cheap and the draws are
rare: at construction, at phase starts, at the start test of each idle step, and at bout starts.
Everything else is arrays. Float results match the per-step runner to float32 precision. Batched
matrix products round differently in the last bits, so trajectories agree exactly only until one
start test falls on the other side of its threshold.

**The surface it covers** is checked at construction, and anything else is refused:

- the continuous point worm under locomotor primitives, with a step of at most a second;
- lawns placed at the arena's centre, with or without an assay protocol;
- klinotaxis chemosensing through the short-term memory and the fold-change adaptive sensor;
- the Fick odour kernel;
- a satiety that never ends the episode;
- a frozen controller from :mod:`quantumnematode.evaluation.controllers`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

# The brain package first: the environment package imports from it while it initialises.
import quantumnematode.brain  # noqa: F401
from quantumnematode.env.continuous_2d import Continuous2DEnvironment
from quantumnematode.env.env import DEFAULT_AGENT_ID
from quantumnematode.env.primitives import (
    BOUT_KINDS,
    PRIMITIVE_KINDS,
    PrimitiveState,
    _start_bout,
    load_kinematics,
)
from quantumnematode.env.theme import Theme
from quantumnematode.report.dtypes import BehaviourStep
from quantumnematode.utils.config_loader import (
    SensingMode,
    configure_environment,
    configure_satiety,
    create_env_from_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from quantumnematode.evaluation.controllers import BatchedController
    from quantumnematode.utils.config_loader import SimulationConfig

# Codes for the primitive running on a step, in the primitives' own order.
KIND_CODE = {kind: i for i, kind in enumerate(PRIMITIVE_KINDS)}
RUN, REVERSAL, OMEGA, DELTA = (KIND_CODE[k] for k in PRIMITIVE_KINDS)
_BOUT_CODES = np.array([KIND_CODE[k] for k in BOUT_KINDS])
# A bout whose remaining time falls below this has ended, as in the primitive layer.
_END_TOLERANCE_S = 1e-9
# The short-term memory's action index for a step with no previous action, and for a continuous one.
_NO_ACTION, _CONTINUOUS_ACTION = 0, 3


class UnsupportedConfigError(ValueError):
    """The config reaches outside what the batched evaluator reproduces."""


@dataclass
class BatchedSenses:
    """One step's senses for every animal, as the brains' feature extractors read them.

    ``derivative`` is the food channel's rate of change after the fold-change readout, before any
    brain scales it; ``lateral`` is the raw right-minus-left head-sweep difference. ``food_contact``
    is the lawn density under each worm, 0 off a lawn. ``moved`` is 0 on an episode's first step,
    which follows no action, and 1 after.
    """

    concentration: np.ndarray
    lateral: np.ndarray
    derivative: np.ndarray
    food_contact: np.ndarray
    satiety_fraction: np.ndarray
    locomotor: np.ndarray
    stam: np.ndarray
    moved: float
    intake: np.ndarray
    max_intake: float | None
    lateral_scale: float
    derivative_scale: float


@dataclass
class BatchedTrajectories:
    """Every animal's per-step record, as compact arrays over animals and steps.

    Each step is recorded where the per-step runner captures it: as the step's senses are read, so
    position and heading are the state the previous step's move left. ``primitive`` is the code of
    the primitive that moved the worm into the step, and ``phase`` the index of its assay phase.
    """

    seeds: list[int]
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    concentration: np.ndarray
    derivative: np.ndarray
    satiety: np.ndarray
    intake: np.ndarray
    on_lawn: np.ndarray
    primitive: np.ndarray
    phase: np.ndarray
    phase_names: list[str] | None
    assay: dict[str, object] | None
    final_satiety: np.ndarray = field(default_factory=lambda: np.zeros(0))
    total_intake: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # The live food-gradient direction and tanh'd strength at each step, when recorded.
    grad_dir: np.ndarray | None = None
    grad_strength: np.ndarray | None = None

    def to_behaviour_steps(self, animal: int) -> list[BehaviourStep]:
        """Return one animal's record as the per-step runner's ``BehaviourStep`` series.

        The live food-gradient fields are the recorded ones, or 0 when the run did not record them.
        """
        steps: list[BehaviourStep] = []
        previous: tuple[float, float] | None = None
        for t in range(self.x.shape[1]):
            x, y, heading = (
                float(self.x[animal, t]),
                float(self.y[animal, t]),
                float(self.heading[animal, t]),
            )
            if previous is None:
                signed = 0.0
            else:
                signed = (x - previous[0]) * math.cos(heading) + (y - previous[1]) * math.sin(
                    heading,
                )
            previous = (x, y)
            steps.append(
                BehaviourStep(
                    step=t,
                    x=x,
                    y=y,
                    heading_rad=heading,
                    concentration=float(self.concentration[animal, t]),
                    dc_dt=float(self.derivative[animal, t]),
                    grad_dir=0.0 if self.grad_dir is None else float(self.grad_dir[animal, t]),
                    grad_strength=(
                        0.0 if self.grad_strength is None else float(self.grad_strength[animal, t])
                    ),
                    speed_signed=signed,
                    satiety=float(self.satiety[animal, t]),
                    intake=float(self.intake[animal, t]),
                    on_lawn=bool(self.on_lawn[animal, t]),
                    primitive=PRIMITIVE_KINDS[int(self.primitive[animal, t])],
                    phase=(
                        None
                        if self.phase_names is None
                        else self.phase_names[int(self.phase[animal, t])]
                    ),
                ),
            )
        return steps


def _refuse(reason: str) -> None:
    msg = f"the batched evaluator does not reproduce this config: {reason}"
    raise UnsupportedConfigError(msg)


def check_surface(config: SimulationConfig) -> None:  # noqa: C901, PLR0912 - one check per setting
    """Refuse a config outside the surface the batched evaluator reproduces, naming the setting."""
    environment = config.environment
    if environment is None or environment.env_type != "continuous_2d":
        _refuse("the substrate is not continuous_2d")
    assert environment is not None  # noqa: S101
    continuous = environment.get_continuous_config()
    if continuous.locomotion != "primitives":
        _refuse(f"continuous.locomotion is {continuous.locomotion}, not primitives")
    if continuous.body_model != "point":
        _refuse("the body model is not the point worm")
    foraging = environment.foraging
    if foraging is None or foraging.food_model != "lawns" or foraging.lawns is None:
        _refuse("the food model is not lawns")
    assert foraging is not None  # noqa: S101
    assert foraging.lawns is not None  # noqa: S101
    if foraging.lawns.placement != "centre":
        _refuse(f"lawns.placement is {foraging.lawns.placement}; every animal must share one lawn")
    if foraging.gradient_field_mode != "fick":
        _refuse(f"gradient_field_mode is {foraging.gradient_field_mode}, not fick")
    for name in ("thermotaxis", "aerotaxis", "predators", "pheromones", "social_feeding"):
        block = getattr(environment, name)
        if block is not None and block.enabled:
            _refuse(f"{name} is enabled")
    if config.multi_agent is not None and config.multi_agent.enabled:
        _refuse("this is a multi-agent run")
    sensing = environment.get_sensing_config()
    if sensing.chemotaxis_mode != SensingMode.KLINOTAXIS:
        _refuse(f"chemotaxis_mode is {sensing.chemotaxis_mode}, not klinotaxis")
    if not sensing.stam_enabled:
        _refuse("the short-term memory is off")
    if (
        not sensing.adaptive_chemosensor_enabled
        or sensing.adaptive_chemosensor_readout != "fold_change"
    ):
        _refuse("the adaptive chemosensor is not the fold-change readout")
    satiety = configure_satiety(config)
    if satiety.starvation_ends_episode:
        _refuse("starvation_ends_episode is true; every animal must run the whole episode")
    if config.max_steps is None:
        _refuse("max_steps is unset")


@dataclass
class _LawnGrid:
    """A centred lawn's cells laid out on their regular grid, for the separable Fick odour."""

    xs: np.ndarray
    ys: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    weights: np.ndarray
    centre: np.ndarray
    radius: float
    cells: np.ndarray
    quality: float
    cell_mm: float
    # Each grid point's cell index, or -1 where the grid point lies outside the lawn.
    index: np.ndarray

    @classmethod
    def from_field(cls, env: Continuous2DEnvironment) -> _LawnGrid:
        lawn = env.lawn_field
        assert lawn is not None  # noqa: S101
        cells = lawn.cells
        xs = np.unique(cells[:, 0])
        ys = np.unique(cells[:, 1])
        ix = np.searchsorted(xs, cells[:, 0])
        iy = np.searchsorted(ys, cells[:, 1])
        index = np.full((len(xs), len(ys)), -1, dtype=np.int64)
        index[ix, iy] = np.arange(len(cells))
        if not np.all(lawn.weights == lawn.weights[0]):  # one lawn: every cell an equal share
            msg = "a centred lawn's cells must share its area equally"
            raise ValueError(msg)
        return cls(
            xs=xs,
            ys=ys,
            ix=ix,
            iy=iy,
            weights=lawn.weights.copy(),
            centre=lawn.centres[0].copy(),
            radius=float(lawn.radius_mm),
            cells=cells.copy(),
            quality=float(lawn.qualities[0]),
            cell_mm=float(lawn.cell_mm),
            index=index,
        )


class BatchedEvaluator:
    """N animals of one config, run in lockstep under a frozen controller.

    Parameters
    ----------
    config : SimulationConfig
        A loaded config inside the evaluator's surface (see :func:`check_surface`).
    controller : BatchedController
        The frozen controller every animal runs.
    seeds : Sequence[int]
        One seed per animal: the run seed the per-step runner would give its episode.
    mean_action : bool
        Act on the controller's noiseless mean instead of sampling its noise.
    record_gradient : bool
        Also record each step's live food-gradient direction and strength, as the per-step runner's
        capture does. It is a sum over every lawn cell per animal, so it is off by default, and
        fitting has no need of it.
    """

    def __init__(
        self,
        config: SimulationConfig,
        controller: BatchedController,
        seeds: Sequence[int],
        *,
        mean_action: bool = False,
        record_gradient: bool = False,
    ) -> None:
        check_surface(config)
        if not seeds:
            msg = "the batched evaluator needs at least one seed"
            raise ValueError(msg)
        self.config = config
        self.controller = controller
        self.seeds = [int(s) for s in seeds]
        self.mean_action = mean_action
        self.record_gradient = record_gradient
        self.environment_config = configure_environment(config)
        self.satiety_config = configure_satiety(config)
        sensing = self.environment_config.get_sensing_config()
        self.stam_size = sensing.stam_buffer_size
        self.stam_weights = np.array(
            [np.exp(-sensing.stam_decay_rate * i) for i in range(self.stam_size)],
            dtype=np.float64,
        )
        self.alpha = float(sensing.adaptive_chemosensor_alpha)
        self.eps = float(sensing.adaptive_chemosensor_epsilon)
        self.lateral_scale = float(sensing.lateral_scale)
        self.derivative_scale = float(sensing.derivative_scale)
        self.kinematics = load_kinematics()
        self.max_steps = int(config.max_steps)  # type: ignore[arg-type]

    # ----- construction -----

    def _build_animals(self) -> None:
        n = len(self.seeds)
        self.envs: list[Continuous2DEnvironment] = []
        for seed in self.seeds:
            env = create_env_from_config(self.environment_config, seed=seed, theme=Theme.HEADLESS)
            assert isinstance(env, Continuous2DEnvironment)  # noqa: S101
            self.envs.append(env)
        first = self.envs[0]
        self.world = float(first.continuous.world_size_mm)
        self.grid_upper = float(first.grid_size - 1)
        self.circle = first.continuous.arena_shape == "circle"
        self.step_seconds = float(first.step_seconds)
        self.sweep = float(first.continuous.sweep_amplitude_mm)
        foraging = first.foraging
        self.fick_length = float(foraging.fick_length())
        self.field_strength = float(foraging.gradient_strength)
        lawns = foraging.lawns
        assert lawns is not None  # noqa: S101
        self.lawn_params = lawns
        self.assay = first.continuous.assay
        self.phase_names = None if self.assay is None else [p.name for p in self.assay.phases]
        self.phase_index = 0 if self.assay is not None else None
        self.grid: _LawnGrid | None = None
        self.lawn_on = np.zeros(n, dtype=bool)
        self.density = np.zeros((n, 0))
        # The same densities on the lawn's grid, zero off the lawn, for the separable odour.
        self.density_grid = np.zeros((n, 0, 0))
        self._sync_lawns()
        states = [env.agents[DEFAULT_AGENT_ID] for env in self.envs]
        self.pos = np.array([s.pos_continuous for s in states], dtype=np.float64)
        self.heading = np.array([s.heading_rad for s in states], dtype=np.float64)
        satiety = self.satiety_config
        self.max_satiety = float(satiety.initial_satiety)
        self.satiety = np.full(n, float(satiety.initial_satiety))
        self.decay = float(satiety.satiety_decay_rate)
        self.step_intake = np.zeros(n)
        self.total_intake = np.zeros(n)
        # Where each worm was when it last ate's step began; NaN until a lawn is first out.
        self.last_position = np.full((n, 2), np.nan)
        # Primitive state, as arrays; the ventral side is drawn on each animal's first move.
        self.ventral = np.zeros(n, dtype=np.int64)
        self.bout_kind = np.full(n, RUN, dtype=np.int64)
        self.remaining = np.zeros(n)
        self.bout_speed = np.zeros(n)
        self.bout_turn = np.zeros(n)
        self.last_kind = np.full(n, RUN, dtype=np.int64)
        # Short-term memory, newest first, and its fill.
        self.stam_c = np.zeros((n, self.stam_size))
        self.stam_dx = np.zeros((n, self.stam_size))
        self.stam_dy = np.zeros((n, self.stam_size))
        self.stam_action = np.zeros(self.stam_size, dtype=np.int64)
        self.stam_n = 0
        self.previous_sensing: np.ndarray | None = None
        self.background = np.zeros(n)
        self.background_set = False
        self.noise = [torch.Generator().manual_seed(seed) for seed in self.seeds]

    def _sync_lawns(self) -> None:
        """Read each animal's lawn from its environment after construction or a phase start."""
        on = np.array([env.lawn_field is not None for env in self.envs])
        if on.any():
            if self.grid is None:
                self.grid = _LawnGrid.from_field(self.envs[int(np.flatnonzero(on)[0])])
            grid = self.grid
            cells = len(grid.cells)
            if self.density.shape[1] != cells:
                self.density = np.zeros((len(self.envs), cells))
                self.density_grid = np.zeros((len(self.envs), len(grid.xs), len(grid.ys)))
            for i, env in enumerate(self.envs):
                if env.lawn_field is not None and not self.lawn_on[i]:
                    self.density[i] = env.lawn_field.density
                    self.density_grid[i, grid.ix, grid.iy] = self.density[i]
        self.lawn_on = on

    def _advance_assay(self, step_index: int) -> None:
        """Begin any phase starting at this step, through each animal's environment; read back."""
        if self.assay is None or self.phase_index is None:
            return
        target = self.assay.phase_at(step_index * self.step_seconds)
        while self.phase_index < target:
            self.phase_index += 1
            for i, env in enumerate(self.envs):
                state = env.agents[DEFAULT_AGENT_ID]
                state.pos_continuous = (float(self.pos[i, 0]), float(self.pos[i, 1]))
                state.heading_rad = float(self.heading[i])
                env.assay_phase_index = self.phase_index - 1
                env._begin_assay_phase(self.phase_index)  # noqa: SLF001
                self.pos[i] = state.pos_continuous
                self.heading[i] = state.heading_rad
            self.lawn_on[:] = False
            self._sync_lawns()

    # ----- the odour, separable over the lawn's cell grid -----

    def _odour(self, points: np.ndarray) -> np.ndarray:
        """Return the tanh'd lawn odour at one point per animal: 0 where no lawn is out."""
        n = len(points)
        out = np.zeros(n)
        if self.grid is None or not self.lawn_on.any():
            return np.tanh(out)
        grid = self.grid
        rows = np.flatnonzero(self.lawn_on)
        gx = np.exp(-(((points[rows, 0:1] - grid.xs[None, :]) / self.fick_length) ** 2))
        gy = np.exp(-(((points[rows, 1:2] - grid.ys[None, :]) / self.fick_length) ** 2))
        # The Fick kernel is separable on the cell grid: sum_xy gx[x] density[x, y] gy[y].
        along_y = np.matmul(self.density_grid[rows], gy[:, :, None])[:, :, 0]
        raw = grid.weights[0] * np.sum(gx * along_y, axis=1)
        out[rows] = self.field_strength * raw
        return np.tanh(out)

    def _gradient(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the lawn odour's direction and tanh'd strength at each animal's point.

        As the environment computes it: each cell pulls toward itself with its density, its share of
        the lawn and the kernel at its distance. The direction is 0 where the pull is zero.
        """
        n = len(points)
        vx, vy = np.zeros(n), np.zeros(n)
        if self.grid is not None and self.lawn_on.any():
            grid = self.grid
            rows = np.flatnonzero(self.lawn_on)
            dx = grid.cells[None, :, 0] - points[rows, 0:1]
            dy = grid.cells[None, :, 1] - points[rows, 1:2]
            distance = np.hypot(dx, dy)
            away = distance > 0
            kernel = self.field_strength * np.exp(-((distance / self.fick_length) ** 2))
            pull = np.where(away, self.density[rows] * grid.weights[None, :] * kernel, 0.0)
            safe = np.where(away, distance, 1.0)
            vx[rows] = np.sum(pull * dx / safe, axis=1)
            vy[rows] = np.sum(pull * dy / safe, axis=1)
        magnitude = np.hypot(vx, vy)
        direction = np.where(magnitude > 0, np.arctan2(vy, vx), 0.0)
        return direction, np.tanh(magnitude)

    def _cells_under(self, points: np.ndarray) -> np.ndarray:
        """Return each animal's cell index under its point, or -1 off the lawn or with none out."""
        cells = np.full(len(points), -1, dtype=np.int64)
        if self.grid is None:
            return cells
        grid = self.grid
        inside = self.lawn_on & (np.hypot(*(grid.centre[None, :] - points).T) <= grid.radius)
        rows = np.flatnonzero(inside)
        if not len(rows):
            return cells
        # The nearest grid point is the nearest cell whenever it lies on the lawn; only near the
        # rim, where it can fall outside, is every cell searched as the lawn field searches them.
        ix = np.clip(np.rint((points[rows, 0] - grid.xs[0]) / grid.cell_mm), 0, len(grid.xs) - 1)
        iy = np.clip(np.rint((points[rows, 1] - grid.ys[0]) / grid.cell_mm), 0, len(grid.ys) - 1)
        nearest = grid.index[ix.astype(np.int64), iy.astype(np.int64)]
        cells[rows] = nearest
        rim = rows[nearest < 0]
        if len(rim):
            d = np.hypot(
                grid.cells[None, :, 0] - points[rim, 0:1],
                grid.cells[None, :, 1] - points[rim, 1:2],
            )
            cells[rim] = np.argmin(d, axis=1)
        return cells

    def _clamp_sample(self, points: np.ndarray) -> np.ndarray:
        """Clamp head-sweep sample points as the per-step runner does: grid, then wall."""
        clamped = np.clip(points, 0.0, self.grid_upper)
        return self._into_circle(clamped) if self.circle else clamped

    def _into_circle(self, points: np.ndarray) -> np.ndarray:
        centre = radius = self.world / 2.0
        d = points - centre
        distance = np.hypot(d[:, 0], d[:, 1])
        outside = distance > radius
        if outside.any():
            scale = radius / distance[outside]
            points = points.copy()
            points[outside] = centre + d[outside] * scale[:, None]
        return points

    # ----- one step -----

    def _sense(self, t: int) -> BatchedSenses:
        pos = self.pos
        c = self._odour(pos)
        perp = np.stack([-np.sin(self.heading), np.cos(self.heading)], axis=1)
        left = self._clamp_sample(pos + self.sweep * perp)
        right = self._clamp_sample(pos - self.sweep * perp)
        lateral = self._odour(right) - self._odour(left)
        delta = np.zeros_like(pos) if self.previous_sensing is None else pos - self.previous_sensing
        self.previous_sensing = pos.copy()
        # The short-term memory, newest first.
        self.stam_c[:, 1:] = self.stam_c[:, :-1]
        self.stam_dx[:, 1:] = self.stam_dx[:, :-1]
        self.stam_dy[:, 1:] = self.stam_dy[:, :-1]
        self.stam_action[1:] = self.stam_action[:-1]
        self.stam_c[:, 0] = c
        self.stam_dx[:, 0] = delta[:, 0]
        self.stam_dy[:, 0] = delta[:, 1]
        self.stam_action[0] = _NO_ACTION if t == 0 else _CONTINUOUS_ACTION
        self.stam_n = min(self.stam_n + 1, self.stam_size)
        derivative = self._stam_derivative()
        stam = self._stam_state(derivative)
        # The fold-change readout reshapes the derivative; the background is tracked regardless.
        if not self.background_set:
            self.background = c.copy()
            self.background_set = True
        else:
            self.background = (1.0 - self.alpha) * self.background + self.alpha * c
        readout = derivative / (c + self.eps)
        cells = self._cells_under(pos)
        contact = np.zeros(len(pos))
        has = cells >= 0
        contact[has] = self.density[np.flatnonzero(has), cells[has]]
        self._sensed_concentration = c
        self._sensed_readout = readout
        self._sensed_on_lawn = has
        return BatchedSenses(
            concentration=c,
            lateral=lateral,
            derivative=readout,
            food_contact=contact,
            satiety_fraction=np.clip(self.satiety / self.max_satiety, 0.0, 1.0),
            locomotor=self.bout_kind.copy(),
            stam=stam,
            moved=0.0 if t == 0 else 1.0,
            intake=self.step_intake.copy(),
            max_intake=(
                self.lawn_params.intake_fraction * self.grid.quality
                if self.grid is not None and self.lawn_on.any()
                else None
            ),
            lateral_scale=self.lateral_scale,
            derivative_scale=self.derivative_scale,
        )

    def _stam_derivative(self) -> np.ndarray:
        n = self.stam_n
        if n < 2:  # noqa: PLR2004
            return np.zeros(len(self.pos))
        weights = self.stam_weights[: n - 1]
        diffs = self.stam_c[:, : n - 1] - self.stam_c[:, 1:n]
        total = np.sum(weights)
        return np.sum(weights[None, :] * diffs, axis=1) / total

    def _stam_state(self, derivative: np.ndarray) -> np.ndarray:
        """Return the memory state: mean, derivative, movement deviation and action entropy."""
        n = self.stam_n
        state = np.zeros((len(self.pos), 5))
        weights = self.stam_weights[:n]
        total = np.sum(weights)
        state[:, 0] = np.sum(weights[None, :] * self.stam_c[:, :n], axis=1) / total
        state[:, 1] = derivative
        if n >= 2:  # noqa: PLR2004
            mean_dx = np.sum(weights[None, :] * self.stam_dx[:, :n], axis=1) / total
            mean_dy = np.sum(weights[None, :] * self.stam_dy[:, :n], axis=1) / total
            state[:, 2] = self.stam_dx[:, 0] - mean_dx
            state[:, 3] = self.stam_dy[:, 0] - mean_dy
        actions = self.stam_action[:n]
        unique, counts = np.unique(actions, return_counts=True)
        if len(unique) > 1:
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs))
            state[:, 4] = entropy / np.log2(max(len(unique), 4))
        return state.astype(np.float32)

    def _move(self, raw: np.ndarray, t: int) -> None:
        """Advance every animal by the primitives, drawing from each animal's own generator."""
        k = self.kinematics
        dt = self.step_seconds
        if t == 0:
            for i, env in enumerate(self.envs):
                self.ventral[i] = 1 if env.rng.random() < 0.5 else -1  # noqa: PLR2004
        x = np.clip(raw, -1.0, 1.0)
        log_low = math.log(k.hazard_min_per_s)
        log_span = math.log(k.hazard_max_per_s) - log_low
        hazards = np.exp(log_low + (x[:, 2:] + 1.0) / 2.0 * log_span)
        speed = k.v_max_mm_per_s * (x[:, 0] + 1.0) / 2.0
        curvature = k.omega_max_rad_per_s * x[:, 1]
        idle = self.bout_kind == RUN
        total = hazards.sum(axis=1)
        start_p = 1.0 - np.exp(-total * dt)
        displacement = np.zeros(len(raw))
        turn = np.zeros(len(raw))
        kinds = np.full(len(raw), RUN, dtype=np.int64)
        self._start_bouts(
            np.flatnonzero(idle),
            start_p,
            hazards,
            total,
            speed,
            curvature,
            displacement,
            turn,
        )
        in_bout = self.bout_kind != RUN
        if in_bout.any():
            rows = np.flatnonzero(in_bout)
            elapsed = np.minimum(dt, self.remaining[rows])
            self.remaining[rows] -= elapsed
            displacement[rows] = self.bout_speed[rows] * elapsed
            turn[rows] = self.bout_turn[rows] * elapsed
            kinds[rows] = self.bout_kind[rows]
            ended = rows[self.remaining[rows] <= _END_TOLERANCE_S]
            self.bout_kind[ended] = RUN
        self.last_kind = kinds
        heading = (self.heading + turn + math.pi) % (2.0 * math.pi) - math.pi
        self.heading = heading
        new_x = np.clip(self.pos[:, 0] + displacement * np.cos(heading), 0.0, self.world)
        new_y = np.clip(self.pos[:, 1] + displacement * np.sin(heading), 0.0, self.world)
        moved = np.stack([new_x, new_y], axis=1)
        self.pos = self._into_circle(moved) if self.circle else moved

    def _start_bouts(  # noqa: PLR0913 - the step's arrays, filled in place
        self,
        idle: np.ndarray,
        start_p: np.ndarray,
        hazards: np.ndarray,
        total: np.ndarray,
        speed: np.ndarray,
        curvature: np.ndarray,
        displacement: np.ndarray,
        turn: np.ndarray,
    ) -> None:
        """Draw each idle animal's start test; start its bout, or move it as a run."""
        k = self.kinematics
        dt = self.step_seconds
        for i in idle:
            rng = self.envs[i].rng
            if rng.random() < start_p[i]:
                pick = rng.random() * total[i]
                cumulative = 0.0
                chosen = BOUT_KINDS[-1]
                for candidate, hazard in zip(BOUT_KINDS, hazards[i], strict=True):
                    cumulative += float(hazard)
                    if pick < cumulative:
                        chosen = candidate
                        break
                bout = _start_bout(
                    chosen,
                    PrimitiveState(ventral_side=int(self.ventral[i])),
                    k,
                    rng,
                )
                self.bout_kind[i] = KIND_CODE[chosen]
                self.remaining[i] = bout.remaining_s
                self.bout_speed[i] = bout.speed_mm_per_s
                self.bout_turn[i] = bout.turn_rad_per_s
            else:
                displacement[i] = speed[i] * dt
                turn[i] = curvature[i] * dt

    def _eat(self) -> None:
        """Charge movement, eat from the cell under each worm, regrow, and restore satiety."""
        lawns = self.lawn_params
        self.step_intake[:] = 0.0
        if lawns.movement_cost_per_mm > 0:
            charged = self.lawn_on & ~np.isnan(self.last_position[:, 0])
            moved = np.hypot(*(self.pos - self.last_position).T)
            cost = np.where(charged, lawns.movement_cost_per_mm * moved, 0.0)
            spend = self.max_satiety * lawns.satiety_per_intake * cost
            self.satiety = np.maximum(0.0, self.satiety - spend)
        self.last_position = np.where(self.lawn_on[:, None], self.pos, self.last_position)
        if self.grid is None or not self.lawn_on.any():
            return
        cells = self._cells_under(self.pos)
        rows = np.flatnonzero(cells >= 0)
        eaten = cells[rows]
        amount = lawns.intake_fraction * self.density[rows, eaten]
        self.density[rows, eaten] -= amount
        grid = self.grid
        self.density_grid[rows, grid.ix[eaten], grid.iy[eaten]] = self.density[rows, eaten]
        value = amount * self.grid.quality
        if lawns.regrowth_per_step > 0:
            on = np.flatnonzero(self.lawn_on)
            self.density[on] = np.minimum(self.density[on] + lawns.regrowth_per_step, 1.0)
            self.density_grid[on[:, None], grid.ix[None, :], grid.iy[None, :]] = self.density[on]
        self.step_intake[rows] = value
        self.total_intake[rows] += value
        restore = self.max_satiety * lawns.satiety_per_intake * value
        self.satiety[rows] = np.minimum(self.max_satiety, self.satiety[rows] + restore)

    # ----- the run -----

    def run(
        self,
        observer: Callable[[int, BatchedEvaluator, BatchedSenses], None] | None = None,
    ) -> BatchedTrajectories:
        """Run every animal for the config's ``max_steps`` and return their records.

        ``observer``, if given, is called on each step once its senses are read, before the
        controller acts: with the step, the evaluator, and the senses.
        """
        self._build_animals()
        self.controller.reset(len(self.seeds))
        n, steps = len(self.seeds), self.max_steps
        # Position, heading, satiety and intake in double precision, the rest in single.
        wide = ("x", "y", "heading", "satiety", "intake")
        record = {
            name: np.zeros((n, steps), dtype=np.float64 if name in wide else np.float32)
            for name in ("x", "y", "heading", "concentration", "derivative", "satiety", "intake")
        }
        grad_dir = np.zeros((n, steps)) if self.record_gradient else None
        grad_strength = np.zeros((n, steps)) if self.record_gradient else None
        on_lawn = np.zeros((n, steps), dtype=bool)
        primitive = np.zeros((n, steps), dtype=np.int8)
        phase = np.zeros((n, steps), dtype=np.int8)
        previous_kind = np.full(n, RUN, dtype=np.int64)
        for t in range(steps):
            self._advance_assay(t)
            senses = self._sense(t)
            record["x"][:, t] = self.pos[:, 0]
            record["y"][:, t] = self.pos[:, 1]
            record["heading"][:, t] = self.heading
            record["concentration"][:, t] = senses.concentration
            record["derivative"][:, t] = senses.derivative
            record["satiety"][:, t] = self.satiety
            record["intake"][:, t] = self.step_intake
            on_lawn[:, t] = self._sensed_on_lawn
            if grad_dir is not None and grad_strength is not None:
                grad_dir[:, t], grad_strength[:, t] = self._gradient(self.pos)
            primitive[:, t] = previous_kind
            phase[:, t] = 0 if self.phase_index is None else self.phase_index
            if observer is not None:
                observer(t, self, senses)
            raw = self.controller.act(senses, None if self.mean_action else self.noise)
            self._move(np.asarray(raw, dtype=np.float64), t)
            previous_kind = self.last_kind.copy()
            self._eat()
            self.satiety = np.maximum(0.0, self.satiety - self.decay)
        return BatchedTrajectories(
            seeds=self.seeds,
            on_lawn=on_lawn,
            primitive=primitive,
            phase=phase,
            phase_names=self.phase_names,
            assay=None if self.assay is None else self.assay.record(self.step_seconds),
            final_satiety=self.satiety.copy(),
            total_intake=self.total_intake.copy(),
            grad_dir=grad_dir,
            grad_strength=grad_strength,
            **record,
        )
