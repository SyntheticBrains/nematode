"""Replay recordings of kinematic-body runs, at the body's sub-step resolution.

A recording holds every body sub-step a run took -- its time, segment curvatures, head and frame
angle -- and every environment step's status: intake, lawn occupancy, satiety, modulator
concentrations, whether the body reversed, and every lawn cell's density. It also holds the run's
static geometry. A player redraws the run from it at any speed, as smooth as the sub-steps allow.

The body's sub-steps come from the environment's posture log, an evaluation record the body already
writes when asked; turning it on draws nothing, so a recorded run is the run that would have run.
Each step's status is the state its move left: the agent snapshots it as the next step's brain
parameters are built, and once more when the run ends, so the first snapshot, taken before any
move, is dropped.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quantumnematode.env.continuous_2d import Continuous2DEnvironment


@dataclass(frozen=True)
class Replay:
    """A loaded recording: the body's sub-steps, each step's status, and the static geometry."""

    substep_time: np.ndarray
    substep_curvature: np.ndarray
    substep_head: np.ndarray
    substep_frame: np.ndarray
    step_of_substep: np.ndarray
    step_reversed: np.ndarray
    step_intake: np.ndarray
    step_on_lawn: np.ndarray
    step_satiety: np.ndarray
    step_modulators: np.ndarray
    step_lawn_density: np.ndarray | None
    lawn_cells: np.ndarray | None
    lawn_centres: np.ndarray | None
    lawn_radius_mm: float | None
    lawn_cell_mm: float | None
    world_size_mm: float
    body_length_mm: float
    step_seconds: float
    meta: dict[str, Any]

    @property
    def duration(self) -> float:
        """Return the recording's length in worm-seconds."""
        return float(self.substep_time[-1]) if len(self.substep_time) else 0.0


class ReplayRecorder:
    """Record each run of a kinematic-body simulation to ``<directory>/<prefix>run<NNNN>.npz``."""

    def __init__(
        self,
        directory: Path,
        prefix: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.prefix = prefix
        self.meta = dict(meta or {})
        self._env: Continuous2DEnvironment | None = None
        self._run = 0
        self._status: list[tuple[float, bool, float, float, float, np.ndarray | None]] = []

    def begin_run(self, env: Continuous2DEnvironment, run: int) -> None:
        """Start recording a run on ``env``: turn its posture log on and clear the status."""
        if env.continuous.body_model != "kinematic":
            msg = "a replay records the kinematic body; this run is a point worm"
            raise ValueError(msg)
        self._env = env
        self._run = run
        self._status = []
        env.posture_log = []

    def snapshot(
        self,
        *,
        intake: float,
        on_lawn: bool,
        satiety: float,
        modulators: tuple[float, float] | None,
    ) -> None:
        """Record the state the last move left; the lawn densities are read from the environment."""
        if self._env is None:
            return
        field = self._env.lawn_field
        density = None if field is None else field.density.copy()
        serotonin, pdf = modulators if modulators is not None else (math.nan, math.nan)
        self._status.append((intake, on_lawn, satiety, serotonin, pdf, density))

    @property
    def active(self) -> bool:
        """Whether a run is being recorded."""
        return self._env is not None

    def end_run(self) -> Path:
        """Write the run's recording and return its path."""
        env = self._env
        if env is None or env.posture_log is None:
            msg = "end_run without begin_run"
            raise ValueError(msg)
        arrays = _arrays(env, env.posture_log, self._status)
        arrays["meta"] = np.array(json.dumps(self.meta | {"run": self._run}))
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self.prefix}run{self._run:04d}.npz"
        np.savez_compressed(path, **arrays)  # pyright: ignore[reportArgumentType]
        env.posture_log = None
        self._env = None
        return path


def _arrays(
    env: Continuous2DEnvironment,
    log: list[dict[str, Any]],
    status: list[tuple[float, bool, float, float, float, np.ndarray | None]],
) -> dict[str, np.ndarray]:
    """Turn a posture log and the step snapshots into the recording's arrays."""
    steps = len(log)
    # The first snapshot precedes any move; each later one is the state the previous move left.
    status = status[1 : steps + 1]
    times, curvatures, heads, frames, owner = [], [], [], [], []
    for index, entry in enumerate(log):
        for time, curvature, head, frame in entry["substeps"]:
            times.append(float(time))
            curvatures.append(np.asarray(curvature, dtype=np.float32))
            heads.append(np.asarray(head, dtype=np.float32))
            frames.append(float(frame))
            owner.append(index)
    arrays: dict[str, np.ndarray] = {
        "substep_time": np.asarray(times, dtype=np.float64),
        "substep_curvature": np.asarray(curvatures, dtype=np.float32).reshape(len(times), -1),
        "substep_head": np.asarray(heads, dtype=np.float32).reshape(len(times), 2),
        "substep_frame": np.asarray(frames, dtype=np.float64),
        "step_of_substep": np.asarray(owner, dtype=np.int32),
        "step_reversed": np.asarray([bool(e["reversed"]) for e in log], dtype=bool),
        "step_intake": np.asarray([s[0] for s in status], dtype=np.float32),
        "step_on_lawn": np.asarray([s[1] for s in status], dtype=bool),
        "step_satiety": np.asarray([s[2] for s in status], dtype=np.float32),
        "step_modulators": np.asarray([(s[3], s[4]) for s in status], dtype=np.float32).reshape(
            -1,
            2,
        ),
        "world_size_mm": np.array(float(env.continuous.world_size_mm)),
        "body_length_mm": np.array(float(env.continuous.body_length_mm)),
        "step_seconds": np.array(float(env.body_params.step_seconds)),
    }
    field = env.lawn_field
    if field is not None:
        arrays["step_lawn_density"] = np.asarray(
            [s[5] for s in status],
            dtype=np.float32,
        ).reshape(len(status), -1)
        arrays["lawn_cells"] = field.cells.astype(np.float32)
        arrays["lawn_centres"] = field.centres.astype(np.float32)
        arrays["lawn_radius_mm"] = np.array(field.radius_mm)
        arrays["lawn_cell_mm"] = np.array(field.cell_mm)
    return arrays


def load_replay(path: Path) -> Replay:
    """Load a recording; refuse a file that holds no body sub-steps."""
    with np.load(path, allow_pickle=False) as data:
        if "substep_curvature" not in data.files or not len(data["substep_time"]):
            msg = f"{path} holds no kinematic-body sub-steps to replay"
            raise ValueError(msg)
        lawns = "lawn_cells" in data.files

        def optional(name: str) -> np.ndarray | None:
            return np.asarray(data[name]) if name in data.files else None

        return Replay(
            substep_time=np.asarray(data["substep_time"]),
            substep_curvature=np.asarray(data["substep_curvature"]),
            substep_head=np.asarray(data["substep_head"]),
            substep_frame=np.asarray(data["substep_frame"]),
            step_of_substep=np.asarray(data["step_of_substep"]),
            step_reversed=np.asarray(data["step_reversed"]),
            step_intake=np.asarray(data["step_intake"]),
            step_on_lawn=np.asarray(data["step_on_lawn"]),
            step_satiety=np.asarray(data["step_satiety"]),
            step_modulators=np.asarray(data["step_modulators"]),
            step_lawn_density=optional("step_lawn_density"),
            lawn_cells=optional("lawn_cells"),
            lawn_centres=optional("lawn_centres"),
            lawn_radius_mm=float(data["lawn_radius_mm"]) if lawns else None,
            lawn_cell_mm=float(data["lawn_cell_mm"]) if lawns else None,
            world_size_mm=float(data["world_size_mm"]),
            body_length_mm=float(data["body_length_mm"]),
            step_seconds=float(data["step_seconds"]),
            meta=json.loads(str(data["meta"])),
        )
