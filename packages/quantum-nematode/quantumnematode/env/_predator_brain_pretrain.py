"""Behavioural-cloning pretrain helper for `MLPPPOPredatorBrain`.

Module-private helper used by `CoevolutionLoop.__init__` (per design.md D7
arm A) to bootstrap gen-0 predator weights from `HeuristicPredatorBrain`'s
decisions before CMA-ES outer-loop weight evolution begins.

Synthesises training pairs `(PredatorBrainParams, heuristic_action)` from
random states (the heuristic teacher is deterministic given params, so the
synthesis is faithful — the brain doesn't care whether params came from a
real env rollout or were sampled), then runs cross-entropy SGD on the
predator brain's actor logits to match the teacher's action distribution.

The helper does NOT interact with `DynamicForagingEnvironment` directly —
it uses synthesised params to keep the pretrain fast, deterministic, and
free of env-side coupling. The 50-"episode" budget per spec
"Imitation Loss Decreases" is interpreted as 50 batches of synthesised
samples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn, optim

from quantumnematode.env.mlpppo_predator_brain import (
    K_NEAREST,
    MLPPPOPredatorBrain,
)

if TYPE_CHECKING:
    from quantumnematode.env.predator_brain import (
        HeuristicPredatorBrain,
        PredatorBrainParams,
    )


# Action enum → index mapping (inverse of `_ACTION_BY_INDEX` in
# `mlpppo_predator_brain.py`). Built lazily inside the train loop to avoid
# eager import of `PredatorAction` here (the import works fine — see
# convention in `mlpppo_predator_brain.py` — but lazy keeps this module
# import-time-cheap).


# Defaults per spec "Imitation Loss Decreases" — 50 episodes (= batches),
# batch size 64 per batch, learning rate 1e-3 with Adam. Window for the
# loss-decrease assertion: first 10 vs last 10 batches.
DEFAULT_NUM_BATCHES = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_GRID_SIZE = 20


def pretrain_against_heuristic(  # noqa: PLR0913 — public helper with several tunable knobs
    brain: MLPPPOPredatorBrain,
    teacher: HeuristicPredatorBrain,
    *,
    num_batches: int = DEFAULT_NUM_BATCHES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    grid_size: int = DEFAULT_GRID_SIZE,
    seed: int | None = None,
) -> list[float]:
    """Run behavioural-cloning pretrain to imitate `teacher` decisions.

    Side effect: `brain.actor` and `brain.critic` are updated in place.

    Parameters
    ----------
    brain
        The `MLPPPOPredatorBrain` to train. Modified in place.
    teacher
        `HeuristicPredatorBrain` providing the supervision signal. Pure
        function of `PredatorBrainParams`; no state.
    num_batches
        Number of training batches (interpreted as the spec's "50
        episodes" budget).
    batch_size
        Number of synthesised `(params, action)` pairs per batch.
    learning_rate
        Adam learning rate.
    grid_size
        Conventional grid size for synthesising params (matches the M3
        lamarckian / pilot scenario default of 20x20).
    seed
        Optional seed for the synthesis RNG (separate from any RNG inside
        the brain). When provided, the same seed produces identical
        training data; useful for test reproducibility.

    Returns
    -------
    list[float]
        Per-batch mean cross-entropy loss values, length `num_batches`.
        Used by tests to assert the windowed loss-decrease invariant
        (final-window mean < initial-window mean per spec).
    """
    # Deferred import: see note in mlpppo_predator_brain.py — top-level
    # `quantumnematode.brain` package eagerly loads arch modules that
    # import from env, which would trigger a cycle if loaded at module
    # import time of this file. Note: `PredatorType` lives in `env.env`,
    # NOT `env.predator_brain` (the latter only re-references it via
    # TYPE_CHECKING).
    from quantumnematode.env.predator_brain import (
        PredatorAction,
    )

    # Build PredatorAction → index mapping inverse to brain's
    # `_ACTION_BY_INDEX`. Mapping must match the brain's output ordering
    # `0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT` per spec "Output Action
    # Mapping".
    action_to_index: dict[PredatorAction, int] = {
        PredatorAction.STAY: 0,
        PredatorAction.UP: 1,
        PredatorAction.DOWN: 2,
        PredatorAction.LEFT: 3,
        PredatorAction.RIGHT: 4,
    }

    rng = np.random.default_rng(seed)
    optimizer = optim.Adam(
        list(brain.actor.parameters()) + list(brain.critic.parameters()),
        lr=learning_rate,
    )
    loss_fn = nn.CrossEntropyLoss()

    losses: list[float] = []
    for _ in range(num_batches):
        # Synthesise a batch of params.
        # Filter to in-pursuit states only — heuristic teacher's
        # random-branch (out-of-pursuit) action is structurally noisy
        # (uniform `rng.integers(4)` draw), so training on those samples
        # provides no learnable signal and would cap held-out accuracy
        # at ~40% even with a perfect classifier. Pretraining on
        # in-pursuit states only teaches the brain "chase the nearest
        # agent on the larger-delta axis" — the meaningful inductive
        # bias for gen-0 predator weights. Out-of-pursuit behaviour
        # emerges from CMA-ES outer-loop weight evolution, not pretrain.
        inputs = np.zeros((batch_size, 11), dtype=np.float32)
        targets = np.zeros(batch_size, dtype=np.int64)
        i = 0
        # Cap synthesis attempts to avoid an infinite loop in pathological
        # configs where in-pursuit states are extremely rare. ~20% in-pursuit
        # at default config; cap of `batch_size * 50` allows for ~2x safety
        # margin even if the rate drops to 1%.
        max_attempts = batch_size * 50
        attempts = 0
        while i < batch_size and attempts < max_attempts:
            attempts += 1
            params = _synthesize_params(rng=rng, grid_size=grid_size)
            if not params.is_pursuing:
                continue
            inputs[i] = brain.encode_observation(params)
            teacher_action = teacher.run_brain(params)
            targets[i] = action_to_index[teacher_action]
            i += 1

        if i < batch_size:
            msg = (
                f"pretrain_against_heuristic: failed to synthesise "
                f"{batch_size} in-pursuit states in {max_attempts} attempts. "
                "Lower `batch_size` or check synthesis distribution."
            )
            raise RuntimeError(msg)

        # Forward + backward.
        x = torch.from_numpy(inputs)
        y = torch.from_numpy(targets)
        logits = brain.actor(x)  # (batch, NUM_ACTIONS)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return losses


def _synthesize_params(
    *,
    rng: np.random.Generator,
    grid_size: int,
) -> PredatorBrainParams:
    """Synthesise a random `PredatorBrainParams` instance.

    Random predator position, 0..`K_NEAREST` random agent positions
    (uniform over the grid), random detection_radius / damage_radius /
    step_index. `chase_target` and `is_pursuing` are computed faithfully
    from the synthesised state so the heuristic teacher's decision tree
    runs in the same semantic regime as a real env rollout.

    Predator type is forced to PURSUIT (the only non-trivial heuristic
    branch — STATIONARY always returns STAY, which would dominate the
    training set if sampled uniformly and bias the brain toward
    no-action). Mixing in some STATIONARY samples is a possible future
    refinement; for the bootstrap helper PURSUIT-only suffices.
    """
    # Deferred imports — see note on `pretrain_against_heuristic`.
    from quantumnematode.env.env import PredatorType
    from quantumnematode.env.predator_brain import PredatorBrainParams

    predator_pos = (int(rng.integers(grid_size)), int(rng.integers(grid_size)))

    # Random agent count in [0, K_NEAREST + 1] so we sometimes test the
    # padding code path (fewer alive agents than k_nearest).
    n_agents = int(rng.integers(0, K_NEAREST + 2))
    agent_positions = tuple(
        (int(rng.integers(grid_size)), int(rng.integers(grid_size))) for _ in range(n_agents)
    )

    detection_radius = int(rng.integers(2, grid_size // 2))
    damage_radius = int(rng.integers(0, 3))
    step_index = int(rng.integers(0, 1000))

    # Resolve chase_target + is_pursuing per the M1 frozen-branch invariant.
    if agent_positions:
        # Nearest by Manhattan, ties broken by env's iteration order (here,
        # synthesised list order).
        chase_target = min(
            agent_positions,
            key=lambda pos: abs(pos[0] - predator_pos[0]) + abs(pos[1] - predator_pos[1]),
        )
        is_pursuing = (
            abs(chase_target[0] - predator_pos[0]) + abs(chase_target[1] - predator_pos[1])
        ) <= detection_radius
    else:
        chase_target = None
        is_pursuing = False

    return PredatorBrainParams(
        predator_id="pretrain_synth",
        predator_position=predator_pos,
        predator_type=PredatorType.PURSUIT,
        detection_radius=detection_radius,
        damage_radius=damage_radius,
        agent_positions=agent_positions,
        chase_target=chase_target,
        is_pursuing=is_pursuing,
        grid_size=grid_size,
        rng=rng,
        step_index=step_index,
    )
