"""Offline static figures for the continuous-2D substrate.

Headless matplotlib analogues of the live continuous renderer's overlays —
trajectory, concentration-field heatmap, and gradient quiver — for logbook and
behavioural-validation figures. These require no display and no pygame; they read
the environment's field getters directly. Time-series plots stay in
:mod:`quantumnematode.report.plots`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path

    from quantumnematode.env.env import DynamicForagingEnvironment

# Selectable heatmap/quiver fields (mirrors the live renderer's field set).
_FOOD_FIELD = "food"


def _infer_world_size(env: DynamicForagingEnvironment, world_size_mm: float | None) -> float:
    """Resolve the arena side length: explicit value, the continuous param, or grid size."""
    if world_size_mm is not None:
        return float(world_size_mm)
    continuous = getattr(env, "continuous", None)
    if continuous is not None:
        return float(continuous.world_size_mm)
    return float(env.grid_size)


def _field_getter(
    env: DynamicForagingEnvironment,
    field: str,
) -> Callable[[tuple[float, float]], float]:
    """Return a ``(x, y) -> float`` sampler for the named field (food default)."""
    if field == "predator":
        return env.get_predator_concentration
    if field == "temperature":
        return lambda pos: env.get_temperature(pos) or 0.0
    if field == "oxygen":
        return lambda pos: env.get_oxygen_concentration(pos) or 0.0
    return env.get_food_concentration


def _sample_field(
    env: DynamicForagingEnvironment,
    field: str,
    world: float,
    resolution: int,
) -> Any:  # noqa: ANN401 - returns a numpy array
    """Sample ``field`` over a ``resolution`` x ``resolution`` lattice (row 0 = y=0)."""
    getter = _field_getter(env, field)
    values = np.zeros((resolution, resolution), dtype=float)  # [row=y][col=x]
    for j in range(resolution):
        y = world * j / (resolution - 1)
        for i in range(resolution):
            x = world * i / (resolution - 1)
            values[j, i] = float(getter((x, y)))
    return values


def plot_trajectory(
    positions: list[tuple[float, float]],
    output_path: Path,
    *,
    world_size_mm: float,
    foods: list[tuple[float, float]] | None = None,
    title: str = "Worm trajectory",
) -> Path:
    """Plot a worm trajectory (and optional food sources) on the arena.

    Parameters
    ----------
    positions : list[tuple[float, float]]
        Ordered worm ``(x, y)`` positions (mm).
    output_path : Path
        PNG path to write.
    world_size_mm : float
        Arena side length (mm).
    foods : list[tuple[float, float]] | None
        Optional food-source positions to mark.
    title : str
        Figure title.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    if positions:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax.plot(xs, ys, "-", color="#e6b46e", linewidth=1.2, label="path")
        ax.plot(xs[0], ys[0], "o", color="#5fb0ff", markersize=7, label="start")
        ax.plot(xs[-1], ys[-1], "*", color="#ffd24a", markersize=12, label="end")
    if foods:
        ax.scatter(
            [f[0] for f in foods],
            [f[1] for f in foods],
            marker="s",
            color="#78b450",
            s=60,
            label="food",
        )
    ax.set_xlim(0, world_size_mm)
    ax.set_ylim(0, world_size_mm)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_field_heatmap(
    env: DynamicForagingEnvironment,
    output_path: Path,
    *,
    field: str = _FOOD_FIELD,
    resolution: int = 200,
    world_size_mm: float | None = None,
    title: str | None = None,
) -> Path:
    """Render a concentration-field heatmap (matplotlib ``imshow``) over the arena.

    Parameters
    ----------
    env : DynamicForagingEnvironment
        Environment whose field getters are sampled.
    output_path : Path
        PNG path to write.
    field : str
        One of ``food`` (default), ``predator``, ``temperature``, ``oxygen``.
    resolution : int
        Lattice resolution per side.
    world_size_mm : float | None
        Arena side length; inferred from the env when omitted.
    title : str | None
        Figure title (defaults to ``"<field> concentration"``).
    """
    world = _infer_world_size(env, world_size_mm)
    values = _sample_field(env, field, world, resolution)

    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(
        values,
        origin="lower",
        extent=(0.0, world, 0.0, world),
        cmap="viridis",
        aspect="equal",
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title or f"{field} concentration")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_gradient_quiver(
    env: DynamicForagingEnvironment,
    output_path: Path,
    *,
    resolution: int = 20,
    world_size_mm: float | None = None,
    title: str = "Food gradient",
) -> Path:
    """Render an up-gradient food-vector quiver over a coarse lattice.

    Parameters
    ----------
    env : DynamicForagingEnvironment
        Environment whose ``get_separated_gradients`` is sampled.
    output_path : Path
        PNG path to write.
    resolution : int
        Coarse lattice resolution per side.
    world_size_mm : float | None
        Arena side length; inferred from the env when omitted.
    title : str
        Figure title.
    """
    world = _infer_world_size(env, world_size_mm)
    coords = np.linspace(0.0, world, resolution)
    grid_x, grid_y = np.meshgrid(coords, coords)
    u = np.zeros_like(grid_x)
    v = np.zeros_like(grid_y)
    for j in range(resolution):
        for i in range(resolution):
            grad = env.get_separated_gradients(
                (float(grid_x[j, i]), float(grid_y[j, i])),  # type: ignore[arg-type]
                disable_log=True,
            )
            strength = float(grad.get("food_gradient_strength", 0.0))
            direction = float(grad.get("food_gradient_direction", 0.0))
            u[j, i] = strength * np.cos(direction)
            v[j, i] = strength * np.sin(direction)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(grid_x, grid_y, u, v, np.hypot(u, v), cmap="viridis", scale=None)
    ax.set_xlim(0, world)
    ax.set_ylim(0, world)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
