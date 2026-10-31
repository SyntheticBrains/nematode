"""The faster record conversion and block-drawn uniforms reproduce the code they replace exactly.

Covers the batched-evaluation spec's scenarios "A converted animal is the same", "Block draws are
the per-step draws" and "A new run starts a new buffer". Each test compares the new path with a
reference copy of the code it replaced.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch
from quantumnematode.env.primitives import PRIMITIVE_KINDS
from quantumnematode.evaluation import controllers as ctl
from quantumnematode.evaluation.batched import BatchedSenses, BatchedTrajectories
from quantumnematode.report.dtypes import BehaviourStep

from .test_mechanistic_baseline import _BASE


def _reference_steps(tr: BatchedTrajectories, animal: int) -> list[BehaviourStep]:
    """Return an animal's steps by the field-by-field conversion the column-wise one replaced."""
    steps: list[BehaviourStep] = []
    previous: tuple[float, float] | None = None
    for t in range(tr.x.shape[1]):
        x, y, heading = float(tr.x[animal, t]), float(tr.y[animal, t]), float(tr.heading[animal, t])
        if previous is None:
            signed = 0.0
        else:
            signed = (x - previous[0]) * math.cos(heading) + (y - previous[1]) * math.sin(heading)
        previous = (x, y)
        steps.append(
            BehaviourStep(
                step=t,
                x=x,
                y=y,
                heading_rad=heading,
                concentration=float(tr.concentration[animal, t]),
                dc_dt=float(tr.derivative[animal, t]),
                grad_dir=0.0 if tr.grad_dir is None else float(tr.grad_dir[animal, t]),
                grad_strength=0.0
                if tr.grad_strength is None
                else float(tr.grad_strength[animal, t]),
                speed_signed=signed,
                satiety=float(tr.satiety[animal, t]),
                intake=float(tr.intake[animal, t]),
                on_lawn=bool(tr.on_lawn[animal, t]),
                primitive=PRIMITIVE_KINDS[int(tr.primitive[animal, t])],
                phase=None if tr.phase_names is None else tr.phase_names[int(tr.phase[animal, t])],
            ),
        )
    return steps


def _trajectories(*, gradient: bool, phases: bool) -> BatchedTrajectories:
    rng = np.random.default_rng(7)
    n, steps = 3, 400

    def wide() -> np.ndarray:
        return rng.normal(0.0, 5.0, size=(n, steps))

    def narrow() -> np.ndarray:
        return rng.normal(0.0, 1.0, size=(n, steps)).astype(np.float32)

    return BatchedTrajectories(
        seeds=[1, 2, 3],
        x=wide(),
        y=wide(),
        heading=rng.uniform(-math.pi, math.pi, size=(n, steps)),
        concentration=narrow(),
        derivative=narrow(),
        satiety=wide(),
        intake=wide(),
        on_lawn=rng.random((n, steps)) < 0.5,
        primitive=rng.integers(len(PRIMITIVE_KINDS), size=(n, steps)).astype(np.int8),
        phase=rng.integers(2, size=(n, steps)).astype(np.int8),
        phase_names=["acclimation", "assay"] if phases else None,
        assay=None,
        grad_dir=wide() if gradient else None,
        grad_strength=wide() if gradient else None,
    )


@pytest.mark.parametrize(("gradient", "phases"), [(True, True), (False, False)])
def test_a_converted_animal_is_the_same(gradient: bool, phases: bool) -> None:  # noqa: FBT001
    """Covers "A converted animal is the same": every field of every step, its value and type."""
    tr = _trajectories(gradient=gradient, phases=phases)
    names = [f.name for f in dataclasses.fields(BehaviourStep)]
    for animal in range(len(tr.seeds)):
        fast, slow = tr.to_behaviour_steps(animal), _reference_steps(tr, animal)
        assert len(fast) == len(slow)
        for a, b in zip(fast, slow, strict=True):
            for name in names:
                va, vb = getattr(a, name), getattr(b, name)
                assert type(va) is type(vb), name
                assert va == vb, name


@pytest.mark.parametrize("seed", [1, 12345, 5101])
def test_block_draws_are_the_per_step_draws(seed: int) -> None:
    """Covers "Block draws are the per-step draws": the same numbers, then the same next draw."""
    k = ctl.UNIFORM_BLOCK_STEPS
    block, steps = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)
    big = torch.rand(3 * k, generator=block, dtype=torch.float64)
    small = torch.cat([torch.rand(3, generator=steps, dtype=torch.float64) for _ in range(k)])
    assert torch.equal(big, small)
    nxt_block = torch.rand(5, generator=block, dtype=torch.float64)
    nxt_steps = torch.rand(5, generator=steps, dtype=torch.float64)
    assert torch.equal(nxt_block, nxt_steps)


class _PerStep(ctl.BatchedMechanistic):
    """The mechanistic worm drawing three uniforms per animal each second, as it did before."""

    def _uniforms(self, gens: list[torch.Generator]) -> np.ndarray:
        return ctl._uniforms(gens, 3)


def _senses(step: int, n: int) -> BatchedSenses:
    contact = np.full(n, 1.0 if (step // 300) % 2 == 0 else 0.0)
    return BatchedSenses(
        concentration=np.zeros(n),
        lateral=np.sin(np.arange(n) + step / 50.0),
        derivative=np.cos(np.arange(n) + step / 70.0),
        food_contact=contact,
        satiety_fraction=np.full(n, 0.5),
        locomotor=np.zeros(n, dtype=np.int64),
        stam=np.zeros((n, 1)),
        moved=1.0,
        intake=np.zeros(n),
        max_intake=None,
        lateral_scale=1.0,
        derivative_scale=1.0,
    )


def _actions(controller: ctl.BatchedMechanistic, seeds: list[int], steps: int) -> np.ndarray:
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    controller.reset(len(seeds))
    return np.stack([controller.act(_senses(t, len(seeds)), gens) for t in range(steps)])


def test_the_mechanistic_worm_acts_the_same_across_a_block_boundary() -> None:
    """Covers "Block draws are the per-step draws" in the controller, past one block's end."""
    seeds = [11, 12, 13, 14]
    steps = ctl.UNIFORM_BLOCK_STEPS + 50
    fast = _actions(ctl.BatchedMechanistic(_BASE), seeds, steps)
    slow = _actions(_PerStep(_BASE), seeds, steps)
    np.testing.assert_array_equal(fast, slow)


def test_a_new_run_starts_a_new_buffer() -> None:
    """Covers "A new run starts a new buffer": two runs on fresh generators, nothing skipped."""
    seeds = [21, 22]
    fast, slow = ctl.BatchedMechanistic(_BASE), _PerStep(_BASE)
    for _ in range(2):
        np.testing.assert_array_equal(_actions(fast, seeds, 100), _actions(slow, seeds, 100))
    # The same generators across a reset keep their place in the block.
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    reference = [torch.Generator().manual_seed(s) for s in seeds]
    for controller, g in ((fast, gens), (slow, reference)):
        controller.reset(len(seeds))
        for t in range(30):
            controller.act(_senses(t, len(seeds)), g)
        controller.reset(len(seeds))
    np.testing.assert_array_equal(
        fast.act(_senses(0, len(seeds)), gens),
        slow.act(_senses(0, len(seeds)), reference),
    )
