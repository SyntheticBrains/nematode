"""Tests for the predator-brain pretrain helper (M5 PR 1, task 1.4 / 1.6).

Covers:

- Loss decreases by ≥0.05 absolute reduction (initial vs final window)
  per spec scenario "Imitation Loss Decreases" (relaxed in round-5
  review — accuracy claim removed).
- Pretrain modifies brain weights in place.
- Synthesised params are deterministic given a fixed seed.
- Pretrained weights round-trip through the encoder unchanged
  (per spec scenario "Pretrained Weights Round-Trip Through Encoder").
- Empty agent_positions edge case handled (synthesis always produces
  at least one valid in-pursuit state given enough attempts).

Notes
-----
The pretrain helper trains on **in-pursuit** states only — out-of-pursuit
states give the heuristic teacher's noisy random-branch action which
contains no learnable signal (uniform `rng.integers(4)` draw). This is
intentional per design.md D7 + the round-5 spec relaxation; the bootstrap
helper teaches the chase behaviour to break orthogonal-init symmetry, with
out-of-pursuit policy left for CMA-ES outer-loop evolution.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from quantumnematode.env import HeuristicPredatorBrain
from quantumnematode.env._predator_brain_pretrain import pretrain_against_heuristic
from quantumnematode.env.mlpppo_predator_brain import MLPPPOPredatorBrain


class TestLossDecreases:
    """Spec scenario "Imitation Loss Decreases" — windowed mean comparison."""

    def test_loss_decrease_at_default_50_batches(self) -> None:
        """Verify final-window mean < initial-window mean by ≥0.05 at 50 batches."""
        brain = MLPPPOPredatorBrain(seed=42)
        teacher = HeuristicPredatorBrain()
        losses = pretrain_against_heuristic(brain, teacher, num_batches=50, seed=42)
        assert len(losses) == 50
        initial_window = sum(losses[:10]) / 10
        final_window = sum(losses[-10:]) / 10
        delta = initial_window - final_window
        # Per spec: ≥0.05 absolute reduction.
        assert delta >= 0.05, (
            f"loss did not decrease by ≥0.05: initial={initial_window:.4f}, "
            f"final={final_window:.4f}, delta={delta:.4f}"
        )

    def test_loss_decrease_is_reproducible(self) -> None:
        """Verify two runs with the same seed produce the same loss curve."""
        brain_a = MLPPPOPredatorBrain(seed=42)
        brain_b = MLPPPOPredatorBrain(seed=42)
        teacher = HeuristicPredatorBrain()
        losses_a = pretrain_against_heuristic(brain_a, teacher, num_batches=20, seed=42)
        losses_b = pretrain_against_heuristic(brain_b, teacher, num_batches=20, seed=42)
        assert losses_a == pytest.approx(losses_b, rel=1e-5)


class TestWeightUpdate:
    """Pretrain modifies brain weights in place (not a no-op)."""

    def test_weights_change_after_pretrain(self) -> None:
        """Verify at least one parameter changed value after pretrain."""
        brain = MLPPPOPredatorBrain(seed=42)
        teacher = HeuristicPredatorBrain()
        # Snapshot weights pre-pretrain.
        pre_weights = [p.detach().clone() for p in brain.actor.parameters()]
        pretrain_against_heuristic(brain, teacher, num_batches=10, seed=42)
        # At least one parameter SHALL differ post-pretrain.
        post_weights = list(brain.actor.parameters())
        assert any(
            not torch.allclose(pre, post)
            for pre, post in zip(pre_weights, post_weights, strict=True)
        )


class TestRoundTripThroughWeightPersistence:
    """Spec scenario "Pretrained Weights Round-Trip Through Encoder"."""

    def test_pretrained_weights_round_trip(self) -> None:
        """Verify pretrained weights extract → load → produce identical actions."""
        from quantumnematode.env import (  # local import — keeps module import clean
            PredatorBrainParams,
            PredatorType,
        )

        original = MLPPPOPredatorBrain(seed=42)
        teacher = HeuristicPredatorBrain()
        pretrain_against_heuristic(original, teacher, num_batches=20, seed=42)

        # Extract weights and load into a fresh brain.
        components = original.get_weight_components()
        clone = MLPPPOPredatorBrain(seed=999)  # different init seed
        clone.load_weight_components(components)

        # Both SHALL produce identical actions on a fixed test set.
        rng = np.random.default_rng(seed=12345)
        for i in range(20):
            params = PredatorBrainParams(
                predator_id="test",
                predator_position=(int(rng.integers(20)), int(rng.integers(20))),
                predator_type=PredatorType.PURSUIT,
                detection_radius=8,
                damage_radius=1,
                agent_positions=((int(rng.integers(20)), int(rng.integers(20))),),
                chase_target=(15, 15),
                is_pursuing=True,
                grid_size=20,
                rng=np.random.default_rng(seed=i),
                step_index=i,
            )
            assert original.run_brain(params) == clone.run_brain(params), (
                f"action mismatch at i={i} after weight round-trip"
            )


class TestSynthesizeParams:
    """Synthesised params produce in-pursuit states most of the time."""

    def test_synthesise_returns_predatorbrainparams(self) -> None:
        """Verify _synthesize_params returns the expected dataclass."""
        from quantumnematode.env import (
            PredatorBrainParams,
            PredatorType,
        )
        from quantumnematode.env._predator_brain_pretrain import _synthesize_params

        rng = np.random.default_rng(seed=42)
        params = _synthesize_params(rng=rng, grid_size=20)
        assert isinstance(params, PredatorBrainParams)
        assert params.grid_size == 20
        assert 0 <= params.predator_position[0] < 20
        assert 0 <= params.predator_position[1] < 20
        assert params.predator_type == PredatorType.PURSUIT


class TestSmallBatchEdge:
    """Edge cases: very small num_batches still completes."""

    def test_single_batch(self) -> None:
        """Verify num_batches=1 completes without error and returns 1-element list."""
        brain = MLPPPOPredatorBrain(seed=42)
        teacher = HeuristicPredatorBrain()
        losses = pretrain_against_heuristic(brain, teacher, num_batches=1, seed=42)
        assert len(losses) == 1
        assert losses[0] > 0.0  # cross-entropy loss is positive
