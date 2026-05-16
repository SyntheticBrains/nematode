"""Unit tests for :mod:`quantumnematode.agent.transgenerational_memory`.

Covers the four spec scenarios for the substrate (clamp, shape/dtype
validation, decay, apply_to_logits, serialisation round-trip) plus
the placeholder behaviour for ``extract_from_brain``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest
import torch
from quantumnematode.agent.transgenerational_memory import (
    LOGIT_BIAS_CLAMP,
    TransgenerationalMemory,
    extract_from_brain,
    load,
    save,
)

if TYPE_CHECKING:
    from pathlib import Path


def _f0(logit_bias: torch.Tensor, source_genome_id: str = "gid-a") -> TransgenerationalMemory:
    """Construct an F0 substrate (lineage_depth=0) for test convenience."""
    return TransgenerationalMemory(
        logit_bias=logit_bias,
        lineage_depth=0,
        source_genome_id=source_genome_id,
    )


# ---------------------------------------------------------------------------
# Construction: clamp + shape/dtype validation + frozen + no-input-mutation
# ---------------------------------------------------------------------------


def test_construction_clamps_bias_values() -> None:
    """``__post_init__`` SHALL clamp ``|logit_bias[i]| > 2.0`` to ``±2.0``.

    Matches the spec scenario "Substrate construction clamps bias values":
    input ``[5.0, -3.0, 0.5, 1.0]`` → stored ``[2.0, -2.0, 0.5, 1.0]``.
    """
    raw = torch.tensor([5.0, -3.0, 0.5, 1.0], dtype=torch.float32)
    sub = _f0(raw)
    expected = torch.tensor([2.0, -2.0, 0.5, 1.0], dtype=torch.float32)
    assert torch.equal(sub.logit_bias, expected)


def test_construction_does_not_mutate_input_tensor() -> None:
    """The caller-provided input tensor SHALL NOT be mutated in place.

    Critical for callers that retain a reference to the pre-clamp
    tensor (e.g., downstream telemetry that records raw extraction
    values alongside the substrate).
    """
    raw = torch.tensor([5.0, -3.0, 0.5, 1.0], dtype=torch.float32)
    raw_before = raw.clone()
    _ = _f0(raw)
    assert torch.equal(raw, raw_before)


def test_construction_rejects_non_1d_logit_bias() -> None:
    """``logit_bias.ndim != 1`` SHALL raise ``ValueError`` naming the shape."""
    raw = torch.zeros((2, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match=r"must be 1-D"):
        _f0(raw)


def test_construction_rejects_non_float32_logit_bias() -> None:
    """``logit_bias.dtype != float32`` SHALL raise ``ValueError`` naming the dtype."""
    raw = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"must have dtype torch.float32"):
        _f0(raw)


def test_construction_rejects_negative_lineage_depth() -> None:
    """``lineage_depth < 0`` SHALL raise ``ValueError`` naming the offending value."""
    raw = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"lineage_depth must be >= 0"):
        TransgenerationalMemory(logit_bias=raw, lineage_depth=-1, source_genome_id="gid-a")


def test_dataclass_is_frozen() -> None:
    """The dataclass SHALL be frozen (cross-generation aliasing protection).

    Attempting to reassign any field SHALL raise
    ``dataclasses.FrozenInstanceError`` (the specific exception raised
    by the auto-generated ``__setattr__`` on a frozen dataclass).
    """
    sub = _f0(torch.zeros(4, dtype=torch.float32))
    with pytest.raises(FrozenInstanceError):
        sub.lineage_depth = 5  # type: ignore[misc]


def test_clamp_constant_matches_spec() -> None:
    """``LOGIT_BIAS_CLAMP`` SHALL be ``2.0`` per the spec's e^2 ~ 7.4x Boltzmann cap."""
    expected_clamp = 2.0
    assert expected_clamp == LOGIT_BIAS_CLAMP


# ---------------------------------------------------------------------------
# inherit_from: geometric decay + validation
# ---------------------------------------------------------------------------


def test_inherit_from_single_parent_geometric_retention() -> None:
    """``inherit_from`` SHALL produce ``parent.logit_bias * decay_factor`` for each call.

    Matches the spec scenario "Single-parent decay produces geometric retention":
    F0 ``[0.0, 1.0, -0.5, 0.2]`` → F1 ``[0.0, 0.6, -0.3, 0.12]`` →
    F2 ``[0.0, 0.36, -0.18, 0.072]`` → F3 ``[0.0, 0.216, -0.108, 0.0432]``.
    """
    f0 = _f0(torch.tensor([0.0, 1.0, -0.5, 0.2], dtype=torch.float32))
    f1 = TransgenerationalMemory.inherit_from([f0], decay_factor=0.6)
    f2 = TransgenerationalMemory.inherit_from([f1], decay_factor=0.6)
    f3 = TransgenerationalMemory.inherit_from([f2], decay_factor=0.6)

    assert torch.allclose(f1.logit_bias, torch.tensor([0.0, 0.6, -0.3, 0.12]))
    assert torch.allclose(f2.logit_bias, torch.tensor([0.0, 0.36, -0.18, 0.072]))
    assert torch.allclose(f3.logit_bias, torch.tensor([0.0, 0.216, -0.108, 0.0432]))


def test_inherit_from_increments_lineage_depth() -> None:
    """``inherit_from`` SHALL increment ``lineage_depth`` by 1 per call."""
    f0 = _f0(torch.tensor([0.0, 1.0, -0.5, 0.2], dtype=torch.float32))
    f1 = TransgenerationalMemory.inherit_from([f0], decay_factor=0.6)
    f2 = TransgenerationalMemory.inherit_from([f1], decay_factor=0.6)
    f3 = TransgenerationalMemory.inherit_from([f2], decay_factor=0.6)
    assert f0.lineage_depth == 0
    assert f1.lineage_depth == 1
    f2_depth = 2
    assert f2.lineage_depth == f2_depth
    f3_depth = 3
    assert f3.lineage_depth == f3_depth


def test_inherit_from_preserves_source_genome_id() -> None:
    """``inherit_from`` SHALL preserve ``source_genome_id`` across all descendants."""
    f0 = _f0(torch.tensor([0.0, 1.0, -0.5, 0.2], dtype=torch.float32), source_genome_id="gid-a")
    f1 = TransgenerationalMemory.inherit_from([f0], decay_factor=0.6)
    f2 = TransgenerationalMemory.inherit_from([f1], decay_factor=0.6)
    f3 = TransgenerationalMemory.inherit_from([f2], decay_factor=0.6)
    assert f1.source_genome_id == "gid-a"
    assert f2.source_genome_id == "gid-a"
    assert f3.source_genome_id == "gid-a"


def test_inherit_from_empty_parents_raises() -> None:
    """``inherit_from`` SHALL raise on empty parents sequence."""
    with pytest.raises(ValueError, match=r"at least one parent"):
        TransgenerationalMemory.inherit_from([], decay_factor=0.6)


def test_inherit_from_decay_factor_below_zero_raises() -> None:
    """``inherit_from`` SHALL reject ``decay_factor < 0.0``."""
    f0 = _f0(torch.zeros(4, dtype=torch.float32))
    with pytest.raises(ValueError, match=r"decay_factor must be in \[0.0, 1.0\]"):
        TransgenerationalMemory.inherit_from([f0], decay_factor=-0.1)


def test_inherit_from_decay_factor_above_one_raises() -> None:
    """``inherit_from`` SHALL reject ``decay_factor > 1.0``."""
    f0 = _f0(torch.zeros(4, dtype=torch.float32))
    with pytest.raises(ValueError, match=r"decay_factor must be in \[0.0, 1.0\]"):
        TransgenerationalMemory.inherit_from([f0], decay_factor=1.5)


def test_inherit_from_uses_single_elite_when_multi_parents() -> None:
    """``inherit_from`` SHALL use ``parents[0]`` only (single-elite-broadcast).

    Matches ``LamarckianInheritance.assign_parent`` semantics under
    ``elite_count=1``. The multi-parent signature is forward-
    compatible with future strategies but the current single-elite
    rule reads only the first element.
    """
    f0_a = _f0(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32), source_genome_id="gid-a")
    f0_b = _f0(torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32), source_genome_id="gid-b")
    child = TransgenerationalMemory.inherit_from([f0_a, f0_b], decay_factor=0.5)
    # parents[0] is f0_a, so child inherits f0_a's bias scaled by 0.5
    # and f0_a's source_genome_id, NOT f0_b's.
    assert torch.allclose(child.logit_bias, torch.tensor([0.5, 0.0, 0.0, 0.0]))
    assert child.source_genome_id == "gid-a"


# ---------------------------------------------------------------------------
# apply_to_logits: additive, broadcast, no-mutation
# ---------------------------------------------------------------------------


def test_apply_to_logits_additive_without_mutation() -> None:
    """``apply_to_logits`` SHALL return ``logits + bias`` without mutating input.

    Matches the spec scenario "Logits are augmented additively without
    mutation": bias ``[0.5, -0.5, 1.0, 0.0]`` + logits
    ``[[1.0, 2.0, 3.0, 0.0]]`` → ``[[1.5, 1.5, 4.0, 0.0]]``; input
    logits tensor unchanged.
    """
    sub = _f0(torch.tensor([0.5, -0.5, 1.0, 0.0], dtype=torch.float32))
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.0]], dtype=torch.float32)
    logits_before = logits.clone()
    result = sub.apply_to_logits(logits)
    expected = torch.tensor([[1.5, 1.5, 4.0, 0.0]], dtype=torch.float32)
    assert torch.equal(result, expected)
    # Input not mutated:
    assert torch.equal(logits, logits_before)
    # Returned tensor is a distinct object:
    assert result.data_ptr() != logits.data_ptr()


def test_apply_to_logits_broadcasts_over_batch_and_seq() -> None:
    """``apply_to_logits`` SHALL broadcast ``(num_actions,)`` over leading dims.

    Matches the spec scenario "Apply preserves shape and dtype across
    batch dimensions": bias shape ``(4,)`` + logits shape ``(batch, seq, 4)``
    → result shape ``(batch, seq, 4)`` with bias broadcast over batch+seq.
    """
    sub = _f0(torch.tensor([0.5, -0.5, 1.0, 0.0], dtype=torch.float32))
    logits = torch.zeros((3, 5, 4), dtype=torch.float32)
    result = sub.apply_to_logits(logits)
    assert result.shape == (3, 5, 4)
    assert result.dtype == torch.float32
    # Every (batch, seq) cell should equal the bias since logits is zero:
    for b in range(3):
        for s in range(5):
            assert torch.equal(result[b, s], sub.logit_bias)


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    """``save`` / ``load`` SHALL produce a byte-equivalent deserialisation.

    Matches the spec scenario "Round-trip preserves all fields":
    F2 substrate with ``logit_bias=[0.5, -0.3, 1.0, 0.0]``,
    ``lineage_depth=2``, ``source_genome_id="gid-elite-3"`` → save →
    load → identical fields.
    """
    original = TransgenerationalMemory(
        logit_bias=torch.tensor([0.5, -0.3, 1.0, 0.0], dtype=torch.float32),
        lineage_depth=2,
        source_genome_id="gid-elite-3",
    )
    path = tmp_path / "foo.tei.pt"
    save(original, path)
    loaded = load(path)
    assert torch.equal(loaded.logit_bias, original.logit_bias)
    assert loaded.lineage_depth == original.lineage_depth
    assert loaded.source_genome_id == original.source_genome_id


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    """``save`` SHALL create the parent directory if missing (mirrors Lamarckian capture)."""
    sub = _f0(torch.zeros(4, dtype=torch.float32))
    path = tmp_path / "inheritance" / "gen-000" / "genome-gid-a.tei.pt"
    assert not path.parent.exists()
    save(sub, path)
    assert path.parent.exists()
    assert path.exists()


def test_load_missing_file_raises_filenotfounderror(tmp_path: Path) -> None:
    """``load`` SHALL raise ``FileNotFoundError`` naming the missing path."""
    path = tmp_path / "nonexistent.tei.pt"
    with pytest.raises(FileNotFoundError, match=r"substrate file not found"):
        load(path)


# ---------------------------------------------------------------------------
# Telemetry-pass: extract_from_brain
# ---------------------------------------------------------------------------


def _make_probe_brain(seed: int = 42) -> object:
    """Build a tiny LSTMPPO brain for telemetry probing tests.

    Returns an LSTMPPOBrain instance with a freshly-initialised actor
    and untrained weights; tests use this to verify ``extract_from_brain``'s
    determinism + signature contract without depending on a full training
    pipeline.
    """
    from quantumnematode.brain.arch.dtypes import DeviceType
    from quantumnematode.brain.arch.lstmppo import LSTMPPOBrain, LSTMPPOBrainConfig
    from quantumnematode.brain.modules import ModuleName

    config = LSTMPPOBrainConfig(
        sensory_modules=[ModuleName.FOOD_CHEMOTAXIS, ModuleName.PROPRIOCEPTION],
        rollout_buffer_size=32,
        bptt_chunk_length=8,
        lstm_hidden_dim=16,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        num_epochs=2,
        seed=seed,
    )
    return LSTMPPOBrain(config=config, num_actions=4, device=DeviceType.CPU)


def _make_probe_params() -> list[object]:
    """Build a small deterministic probe-params sequence for telemetry tests."""
    import numpy as np
    from quantumnematode.brain.arch import BrainParams
    from quantumnematode.env import Direction

    # Three probe positions with varying gradient strengths — synthetic
    # "near-pathogen" sensory states.
    return [
        BrainParams(
            food_gradient_strength=0.3,
            food_gradient_direction=np.pi / 2,
            agent_direction=Direction.UP,
        ),
        BrainParams(
            food_gradient_strength=0.5,
            food_gradient_direction=np.pi,
            agent_direction=Direction.UP,
        ),
        BrainParams(
            food_gradient_strength=0.1,
            food_gradient_direction=0.0,
            agent_direction=Direction.UP,
        ),
    ]


def test_extract_from_brain_is_deterministic_on_seed() -> None:
    """``extract_from_brain`` SHALL produce identical output for the same seed + brain weights.

    Spec scenario "Extraction is deterministic for a given seed":
    two calls with the same brain (same initialisation seed), same
    probe_params, and same rng_seed SHALL produce element-wise
    equal ``logit_bias`` tensors.
    """
    brain_a = _make_probe_brain(seed=42)
    brain_b = _make_probe_brain(seed=42)
    probe_params = _make_probe_params()

    sub_a = extract_from_brain(
        brain=brain_a,
        probe_params=probe_params,
        rng_seed=123,
        source_genome_id="gid-a",
    )
    sub_b = extract_from_brain(
        brain=brain_b,
        probe_params=probe_params,
        rng_seed=123,
        source_genome_id="gid-a",
    )
    assert torch.equal(sub_a.logit_bias, sub_b.logit_bias)


def test_extract_from_brain_returns_f0_substrate() -> None:
    """``extract_from_brain`` SHALL return a substrate with depth=0 and the supplied id."""
    brain = _make_probe_brain(seed=42)
    probe_params = _make_probe_params()
    sub = extract_from_brain(
        brain=brain,
        probe_params=probe_params,
        rng_seed=123,
        source_genome_id="elite-7",
    )
    assert sub.lineage_depth == 0
    assert sub.source_genome_id == "elite-7"
    # logit_bias has the right shape (4 actions matching the brain).
    assert sub.logit_bias.shape == (4,)
    assert sub.logit_bias.dtype == torch.float32


def test_extract_from_brain_rejects_empty_probe_params() -> None:
    """``extract_from_brain`` SHALL raise ``ValueError`` on empty probe sequence.

    The function requires at least one probe to produce meaningful
    action statistics; an empty sequence would divide by zero in
    the mean-probability computation.
    """

    # Minimal stub brain: only needs prepare_episode (callable) and
    # num_actions to reach the empty-sequence check.
    class _StubBrain:
        num_actions = 4

        def prepare_episode(self) -> None: ...

    with pytest.raises(ValueError, match=r"at least one probe_params element"):
        extract_from_brain(
            brain=_StubBrain(),
            probe_params=[],
            rng_seed=42,
            source_genome_id="gid-a",
        )
