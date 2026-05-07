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
        PredatorAction,
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

    Side effect: `brain.actor` weights are updated in place. The critic
    head is NOT trained (no value targets in the synthesis pipeline);
    its weights remain at their orthogonal-init values until CMA-ES
    outer-loop evolution updates them via `WeightPersistence`.

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

    Raises
    ------
    ValueError
        If any tunable knob is outside its valid range (see "fail-fast"
        validation block at the top of the function body).
    """
    # Fail-fast validation: surface bad config values as ValueError rather
    # than letting them fail mid-loop with cryptic errors (e.g.
    # `np.zeros((-1, 11))` would only blow up inside the train loop).
    if not isinstance(num_batches, int) or num_batches <= 0:
        msg = f"num_batches must be a positive int, got {num_batches!r}"
        raise ValueError(msg)
    if not isinstance(batch_size, int) or batch_size <= 0:
        msg = f"batch_size must be a positive int, got {batch_size!r}"
        raise ValueError(msg)
    if not isinstance(learning_rate, (int, float)) or not (0.0 < learning_rate <= 1.0):
        msg = f"learning_rate must be a positive float in (0, 1], got {learning_rate!r}"
        raise ValueError(msg)
    if not isinstance(grid_size, int) or grid_size < 3:  # noqa: PLR2004
        # grid_size >= 3 is needed because synthesis draws
        # `rng.integers(2, grid_size // 2)` for detection_radius — a
        # grid_size < 3 would either short-circuit (low == high) or
        # produce a degenerate grid that can't fit predator + agents.
        msg = f"grid_size must be an int >= 3, got {grid_size!r}"
        raise ValueError(msg)
    if seed is not None and not isinstance(seed, int):
        msg = f"seed must be None or int, got {seed!r}"
        raise ValueError(msg)

    # Deferred import: see note in mlpppo_predator_brain.py — top-level
    # `quantumnematode.brain` package eagerly loads arch modules that
    # import from env, which would trigger a cycle if loaded at module
    # import time of this file. Note: `PredatorType` lives in `env.env`,
    # NOT `env.predator_brain` (the latter only re-references it via
    # TYPE_CHECKING).
    from quantumnematode.env.mlpppo_predator_brain import _ACTION_BY_INDEX

    # Derive PredatorAction → index inverse from the brain's canonical
    # `_ACTION_BY_INDEX` ordering (`0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT`
    # per spec "Output Action Mapping"). Single source of truth: the
    # brain owns the mapping, pretrain reuses it via inversion. Prevents
    # silent drift if a future PR re-orders `_ACTION_BY_INDEX`.
    action_to_index: dict[PredatorAction, int] = {
        action: idx for idx, action in enumerate(_ACTION_BY_INDEX)
    }

    rng = np.random.default_rng(seed)
    # Actor-only optimizer. The critic head has no supervisory signal in
    # this synthesis pipeline (we only have action labels from the
    # heuristic teacher, not value targets), so including critic params
    # would just allocate unused Adam state. CMA-ES outer-loop evolution
    # handles critic weights at the genome level via WeightPersistence.
    optimizer = optim.Adam(brain.actor.parameters(), lr=learning_rate)
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
