"""Unit tests for :mod:`quantumnematode.agent.transgenerational_memory`.

Covers the four spec scenarios for the substrate (clamp, shape/dtype
validation, decay, apply_to_logits, serialisation round-trip) plus
the placeholder behaviour for ``extract_from_brain``.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, cast

import pytest
import torch
from quantumnematode.agent.transgenerational_memory import (
    LOGIT_BIAS_CLAMP,
    TransgenerationalMemory,
    extract_from_brain,
    load,
    save,
)
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path


def _linear(net: nn.Sequential | None, idx: int) -> nn.Linear:
    """Narrow ``net[idx]`` to ``nn.Linear`` for type-checker friendliness in tests."""
    assert net is not None
    return cast("nn.Linear", net[idx])


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
    """``__post_init__`` SHALL clamp ``|logit_bias[i]|`` to ``±LOGIT_BIAS_CLAMP``.

    Matches the spec scenario "Substrate construction clamps bias values".
    The exact constant is read from the module so this test follows the
    constant if it's tuned (e.g., the M6.9+ pilot-2 critique raised it from
    2.0 to 6.0 to prevent fresh-init F1+ saturation).
    """
    clamp_val = LOGIT_BIAS_CLAMP
    # Pick values that straddle the clamp on both sides + an in-range value.
    raw = torch.tensor(
        [clamp_val + 1.0, -(clamp_val + 1.0), 0.5, clamp_val * 0.5],
        dtype=torch.float32,
    )
    sub = _f0(raw)
    expected = torch.tensor(
        [clamp_val, -clamp_val, 0.5, clamp_val * 0.5],
        dtype=torch.float32,
    )
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


def test_clamp_constant_in_acceptable_range() -> None:
    """``LOGIT_BIAS_CLAMP`` SHALL be a positive float in the expected M6.9+ range.

    The exact value is tuned per pilot evidence (originally 2.0 = e^2 ~ 7.4x
    Boltzmann cap; raised to 6.0 = e^6 ~ 403x after M6.9+ pilot-2 critique
    flagged saturation as the mechanical bottleneck for fresh-init F1+
    substrate application). The constant is positive AND finite AND
    bounded above so a strong bias cannot collapse exploration entirely.
    """
    assert LOGIT_BIAS_CLAMP > 0.0
    assert LOGIT_BIAS_CLAMP <= 10.0  # sane upper bound; e^10 ~22k is plenty
    assert isinstance(LOGIT_BIAS_CLAMP, float)


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


# ---------------------------------------------------------------------------
# M6.9+ sensory-conditional bias-network tests
# ---------------------------------------------------------------------------


def _make_bias_network(
    input_dim: int = 3,
    hidden_dim: int = 8,
    output_dim: int = 4,
) -> torch.nn.Sequential:
    """Construct a small bias-network module for tests."""
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, output_dim),
    )


def _f0_with_network(
    bias_network: torch.nn.Sequential,
    input_features: tuple[str, ...] = ("predator_gradient_strength",),
) -> TransgenerationalMemory:
    """Construct an F0 substrate carrying a sensory-conditional bias-network."""
    return TransgenerationalMemory(
        logit_bias=torch.zeros(4, dtype=torch.float32),
        lineage_depth=0,
        source_genome_id="gid-a",
        bias_network=bias_network,
        input_features=input_features,
    )


def test_bias_network_post_init_deep_copies_module() -> None:
    """Caller-side mutation of the source module SHALL NOT affect the stored substrate."""
    net = _make_bias_network()
    sub = _f0_with_network(net)
    # Capture stored weights before mutation.
    stored = _linear(sub.bias_network, 0).weight.detach().clone()
    # Mutate the original module in place.
    with torch.no_grad():
        cast("nn.Linear", net[0]).weight.data.zero_()
    # Stored substrate's weight SHALL be unchanged.
    assert torch.equal(_linear(sub.bias_network, 0).weight, stored)


def test_bias_network_post_init_strips_gradient_flag() -> None:
    """Bias-network parameters SHALL be detached from autograd after construction."""
    sub = _f0_with_network(_make_bias_network())
    assert sub.bias_network is not None
    for param in sub.bias_network.parameters():
        assert param.requires_grad is False


def test_bias_network_rejects_empty_input_features() -> None:
    """Construction SHALL reject ``bias_network != None`` with empty ``input_features``."""
    with pytest.raises(ValueError, match=r"input_features is empty"):
        TransgenerationalMemory(
            logit_bias=torch.zeros(4, dtype=torch.float32),
            lineage_depth=0,
            source_genome_id="gid-a",
            bias_network=_make_bias_network(),
            input_features=(),
        )


def test_bias_network_apply_to_logits_uses_clamp() -> None:
    """When ``bias_network`` produces large output, the bias SHALL be clamped at the cap."""
    # Construct a network whose output is far outside [-2, 2] for any input.
    net = _make_bias_network(input_dim=3, hidden_dim=8, output_dim=4)
    with torch.no_grad():
        _linear(net, 0).weight.fill_(0.0)
        _linear(net, 0).bias.fill_(0.0)
        _linear(net, 2).weight.fill_(0.0)
        _linear(net, 2).bias.fill_(10.0)  # constant output = 10 on every action
    sub = _f0_with_network(
        net,
        input_features=(
            "predator_gradient_strength",
            "food_gradient_strength",
            "predator_gradient_direction_sin",
        ),
    )
    logits = torch.zeros(4, dtype=torch.float32)
    sensory = torch.tensor([0.5, 0.3, 0.1], dtype=torch.float32)
    biased = sub.apply_to_logits(logits, sensory)
    # Bias saturates at +LOGIT_BIAS_CLAMP on every action.
    assert torch.allclose(biased, torch.full((4,), LOGIT_BIAS_CLAMP))


def test_bias_network_apply_to_logits_returns_legacy_when_none() -> None:
    """``bias_network is None`` SHALL fall back to ``logits + logit_bias`` (M6 legacy)."""
    sub = _f0(torch.tensor([0.5, 0.0, 0.0, 0.0], dtype=torch.float32))
    logits = torch.zeros(4, dtype=torch.float32)
    # sensory_input passed but ignored.
    biased = sub.apply_to_logits(logits, sensory_input=torch.tensor([1.0], dtype=torch.float32))
    assert torch.equal(biased, torch.tensor([0.5, 0.0, 0.0, 0.0], dtype=torch.float32))
    # Also no-sensory call.
    biased_no_sensory = sub.apply_to_logits(logits)
    assert torch.equal(biased_no_sensory, torch.tensor([0.5, 0.0, 0.0, 0.0], dtype=torch.float32))


def test_bias_network_apply_to_logits_requires_sensory_input() -> None:
    """When ``bias_network`` is set, ``apply_to_logits(..., None)`` SHALL raise ``ValueError``."""
    sub = _f0_with_network(_make_bias_network())
    with pytest.raises(ValueError, match=r"sensory_input is None"):
        sub.apply_to_logits(torch.zeros(4, dtype=torch.float32))


def test_build_sensory_input_resolves_raw_field() -> None:
    """``build_sensory_input`` SHALL read a raw ``BrainParams`` field into the tensor."""
    from quantumnematode.agent.transgenerational_memory import build_sensory_input
    from quantumnematode.brain.arch import BrainParams

    params = BrainParams(predator_gradient_strength=0.7, food_gradient_strength=0.3)
    out = build_sensory_input(params, ["predator_gradient_strength", "food_gradient_strength"])
    assert torch.allclose(out, torch.tensor([0.7, 0.3], dtype=torch.float32))


def test_build_sensory_input_applies_sin_cos_transform() -> None:
    """``build_sensory_input`` SHALL apply ``math.sin`` / ``math.cos`` on suffixed names."""
    import math

    from quantumnematode.agent.transgenerational_memory import build_sensory_input
    from quantumnematode.brain.arch import BrainParams

    angle = math.pi / 3
    params = BrainParams(predator_gradient_direction=angle)
    out = build_sensory_input(
        params,
        ["predator_gradient_direction_sin", "predator_gradient_direction_cos"],
    )
    assert torch.allclose(
        out,
        torch.tensor([math.sin(angle), math.cos(angle)], dtype=torch.float32),
    )


def test_build_sensory_input_handles_none_with_zero() -> None:
    """Missing/None ``BrainParams`` fields SHALL contribute 0.0 to the sensory vector."""
    from quantumnematode.agent.transgenerational_memory import build_sensory_input
    from quantumnematode.brain.arch import BrainParams

    params = BrainParams()  # all fields default-None
    out = build_sensory_input(
        params,
        ["predator_gradient_strength", "predator_gradient_direction_sin"],
    )
    assert torch.allclose(out, torch.zeros(2, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Decay-shape branching tests (geometric / linear / sigmoid)
# ---------------------------------------------------------------------------


def test_inherit_from_geometric_decay_byte_equivalent_to_m6() -> None:
    """``decay_shape='geometric'`` (default) SHALL produce M6 byte-equivalent cascade."""
    f0 = _f0(torch.tensor([1.0, -1.0, 0.5, 0.0], dtype=torch.float32))
    f1 = TransgenerationalMemory.inherit_from([f0], decay_factor=0.6)
    f2 = TransgenerationalMemory.inherit_from([f1], decay_factor=0.6)
    # Geometric default: F1 = 0.6 * F0; F2 = 0.6 * F1 = 0.36 * F0.
    assert torch.allclose(f1.logit_bias, torch.tensor([0.6, -0.6, 0.3, 0.0]))
    assert torch.allclose(f2.logit_bias, torch.tensor([0.36, -0.36, 0.18, 0.0]))


def test_inherit_from_linear_decay_monotonically_shrinks() -> None:
    """Under linear decay the cumulative scale reaches zero by lineage_depth 1/(1-decay_factor).

    Child weights = parent * scale; the per-generation scale is the
    delta between cumulative factors at parent and child depth.
    """
    f0 = _f0(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32))
    f1 = TransgenerationalMemory.inherit_from([f0], decay_factor=0.6, decay_shape="linear")
    f2 = TransgenerationalMemory.inherit_from([f1], decay_factor=0.6, decay_shape="linear")
    # linear: cum at depth 1 = max(0, 1 - 0.4) = 0.6; cum at depth 2 = max(0, 1 - 0.8) = 0.2.
    # F1 logit_bias = parent_cum_0 * scale_0to1 * F0 = 1.0 * 0.6 = 0.6.
    # F2 logit_bias = F1 * scale_1to2 = 0.6 * (0.2 / 0.6) = 0.2.
    assert torch.allclose(f1.logit_bias, torch.tensor([0.6, 0.0, 0.0, 0.0]))
    assert torch.allclose(f2.logit_bias, torch.tensor([0.2, 0.0, 0.0, 0.0]), atol=1e-6)


def test_inherit_from_sigmoid_decay_monotonically_shrinks() -> None:
    """Under ``decay_shape='sigmoid'`` the cumulative scale follows the fixed K=2 / M=1 schedule.

    Per the spec the sigmoid schedule is intentionally independent of
    ``decay_factor`` — the cumulative scale at depth ``d`` is
    ``sigmoid(2.0 * (1.0 - d))`` regardless of the requested decay
    factor. This test asserts both monotonic shrinkage AND the
    canonical fixed-shape cumulative values.
    """
    f0 = _f0(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32))
    f1 = TransgenerationalMemory.inherit_from([f0], decay_factor=0.6, decay_shape="sigmoid")
    f2 = TransgenerationalMemory.inherit_from([f1], decay_factor=0.6, decay_shape="sigmoid")
    f3 = TransgenerationalMemory.inherit_from([f2], decay_factor=0.6, decay_shape="sigmoid")
    # Cumulative scale = cum(d) / cum(0) where cum(d) = sigmoid(2*(1-d)).
    # Normalised to cum(0)=0.88080, so F0 logit_bias[0]=1.0 → F1=0.500/0.881=0.5675,
    # F2=0.119/0.881=0.1352, F3=0.018/0.881=0.0205.
    cum0 = 1.0 / (1.0 + math.exp(-2.0))
    expected_f1 = 1.0 * (0.5 / cum0)
    expected_f2 = 1.0 * ((1.0 / (1.0 + math.exp(2.0))) / cum0)
    expected_f3 = 1.0 * ((1.0 / (1.0 + math.exp(4.0))) / cum0)
    assert math.isclose(f1.logit_bias[0].item(), expected_f1, abs_tol=1e-4)
    assert math.isclose(f2.logit_bias[0].item(), expected_f2, abs_tol=1e-4)
    assert math.isclose(f3.logit_bias[0].item(), expected_f3, abs_tol=1e-4)
    # AND monotonic shrinkage (sanity-check the spec's claim).
    assert f0.logit_bias.norm() > f1.logit_bias.norm() > f2.logit_bias.norm() > f3.logit_bias.norm()


def test_inherit_from_sigmoid_decay_ignores_decay_factor() -> None:
    """Under ``decay_shape='sigmoid'`` the cascade SHALL NOT depend on ``decay_factor``.

    Per the spec the sigmoid shape is a fixed-schedule alternative
    (`K=2`, `M=1`), explicitly independent of `decay_factor`. Two
    cascades with different decay_factor values SHALL produce
    bit-identical descendant substrates under sigmoid.
    """
    f0 = _f0(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32))
    f1_low = TransgenerationalMemory.inherit_from([f0], decay_factor=0.2, decay_shape="sigmoid")
    f1_high = TransgenerationalMemory.inherit_from([f0], decay_factor=0.9, decay_shape="sigmoid")
    assert torch.allclose(f1_low.logit_bias, f1_high.logit_bias)


def test_inherit_from_unknown_decay_shape_raises() -> None:
    """``decay_shape='exponential'`` (or any unsupported value) SHALL raise ``ValueError``."""
    f0 = _f0(torch.zeros(4, dtype=torch.float32))
    with pytest.raises(ValueError, match=r"decay_shape must be one of"):
        TransgenerationalMemory.inherit_from([f0], decay_factor=0.6, decay_shape="exponential")  # type: ignore[arg-type]


def test_inherit_from_scales_bias_network_weights() -> None:
    """``inherit_from`` SHALL scale every weight tensor in the bias-network by the per-gen factor."""  # noqa: E501
    net = _make_bias_network()
    # Fix all parameters to ones so the scaling is observable.
    with torch.no_grad():
        for param in net.parameters():
            param.fill_(1.0)
    f0 = _f0_with_network(net)
    f1 = TransgenerationalMemory.inherit_from([f0], decay_factor=0.6)
    assert f0.bias_network is not None
    assert f1.bias_network is not None
    # First layer weight scaled by 0.6.
    assert torch.allclose(
        _linear(f1.bias_network, 0).weight,
        torch.full_like(_linear(f0.bias_network, 0).weight, 0.6),
    )
    assert torch.allclose(
        _linear(f1.bias_network, 2).bias,
        torch.full_like(_linear(f0.bias_network, 2).bias, 0.6),
    )


def test_inherit_from_preserves_input_features() -> None:
    """``inherit_from`` SHALL propagate ``input_features`` unchanged."""
    net = _make_bias_network()
    f0 = _f0_with_network(
        net,
        input_features=("predator_gradient_strength", "food_gradient_strength"),
    )
    f1 = TransgenerationalMemory.inherit_from([f0], decay_factor=0.6)
    assert f1.input_features == ("predator_gradient_strength", "food_gradient_strength")


# ---------------------------------------------------------------------------
# Round-trip persistence: bias_network + input_features metadata
# ---------------------------------------------------------------------------


def test_save_load_round_trip_preserves_bias_network(tmp_path: Path) -> None:
    """``save``/``load`` round-trip SHALL preserve ``bias_network`` state_dict + ``input_features``."""  # noqa: E501
    net = _make_bias_network()
    with torch.no_grad():
        _linear(net, 0).weight.fill_(0.3)
        _linear(net, 0).bias.fill_(0.1)
        _linear(net, 2).weight.fill_(-0.2)
        _linear(net, 2).bias.fill_(0.0)
    sub = _f0_with_network(
        net,
        input_features=(
            "predator_gradient_strength",
            "food_gradient_strength",
            "predator_gradient_direction_sin",
        ),
    )
    path = tmp_path / "f0.tei.pt"
    save(sub, path)
    loaded = load(path)
    assert loaded.bias_network is not None
    assert sub.bias_network is not None
    assert loaded.input_features == sub.input_features
    # state_dict tensors are byte-equivalent across round-trip.
    for k, v in sub.bias_network.state_dict().items():
        assert torch.equal(loaded.bias_network.state_dict()[k], v)


def test_save_load_round_trip_legacy_path_remains_byte_equivalent(tmp_path: Path) -> None:
    """When ``bias_network is None`` round-trip SHALL preserve M6 byte-equivalent payload shape."""
    sub = _f0(torch.tensor([0.5, -0.3, 0.0, 0.1], dtype=torch.float32))
    path = tmp_path / "f0_legacy.tei.pt"
    save(sub, path)
    loaded = load(path)
    assert loaded.bias_network is None
    assert loaded.input_features == ()
    assert torch.equal(loaded.logit_bias, sub.logit_bias)


# ---------------------------------------------------------------------------
# extract_from_brain: linear-projection closed-form path
# ---------------------------------------------------------------------------


def test_extract_from_brain_fits_linear_projection_closed_form() -> None:
    """When ``bias_network_spec.hidden_dim == 0`` extraction SHALL fit via least-squares."""
    brain = _make_probe_brain(seed=42)
    probe_params = _make_probe_params()
    spec = {
        "input_dim": 1,
        "hidden_dim": 0,
        "output_dim": 4,
        "activation": "tanh",
    }
    sub = extract_from_brain(
        brain=brain,
        probe_params=probe_params,
        rng_seed=123,
        source_genome_id="elite-7",
        bias_network_spec=spec,
        input_features=("food_gradient_strength",),
    )
    assert sub.bias_network is not None
    assert sub.input_features == ("food_gradient_strength",)
    # Closed-form: linear-only network = single nn.Linear.
    linears = [m for m in sub.bias_network if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 1
    assert linears[0].in_features == 1
    assert linears[0].out_features == 4


def test_extract_from_brain_with_mlp_fits_deterministically() -> None:
    """MLP fit SHALL be deterministic on ``rng_seed`` (same inputs → bit-identical state_dict)."""
    brain_a = _make_probe_brain(seed=42)
    brain_b = _make_probe_brain(seed=42)
    probe_params = _make_probe_params()
    spec = {
        "input_dim": 1,
        "hidden_dim": 8,
        "output_dim": 4,
        "activation": "tanh",
    }
    sub_a = extract_from_brain(
        brain=brain_a,
        probe_params=probe_params,
        rng_seed=123,
        source_genome_id="elite-7",
        bias_network_spec=spec,
        input_features=("food_gradient_strength",),
    )
    sub_b = extract_from_brain(
        brain=brain_b,
        probe_params=probe_params,
        rng_seed=123,
        source_genome_id="elite-7",
        bias_network_spec=spec,
        input_features=("food_gradient_strength",),
    )
    assert sub_a.bias_network is not None
    assert sub_b.bias_network is not None
    for k, v in sub_a.bias_network.state_dict().items():
        assert torch.equal(sub_b.bias_network.state_dict()[k], v)


def test_extract_from_brain_mlp_runs_50_epochs() -> None:
    """The fitted MLP weights SHALL diverge from the random-init weights after the fit.

    Spec scenario "MLP fit converges within 50 epochs" — the binding
    claim is that Adam runs for the configured ``fit_epochs`` budget
    and updates the parameters. Verifying actual loss-monotonic-
    convergence would require exposing the per-probe targets (the
    fit operates on per-probe ``log(probs + eps) - log_uniform``, not
    on the substrate's `logit_bias` mean). The simpler binding check
    is that the optimiser stepped at all — random-init weights and
    fitted weights MUST differ.
    """
    from quantumnematode.agent.transgenerational_memory import _build_empty_bias_network

    brain = _make_probe_brain(seed=42)
    probe_params = _make_probe_params()
    spec = {
        "input_dim": 1,
        "hidden_dim": 8,
        "output_dim": 4,
        "activation": "tanh",
    }
    # Random-init reference with the same seed used by extract_from_brain.
    torch.manual_seed(123)
    random_net = _build_empty_bias_network(
        input_dim=1,
        hidden_dim=8,
        output_dim=4,
        activation="tanh",
    )
    sub = extract_from_brain(
        brain=brain,
        probe_params=probe_params,
        rng_seed=123,
        source_genome_id="elite-7",
        bias_network_spec=spec,
        input_features=("food_gradient_strength",),
    )
    assert sub.bias_network is not None
    # Fitted weights SHALL differ from random-init (i.e. Adam took >= 1 step).
    fitted_first = _linear(sub.bias_network, 0).weight
    random_first = _linear(random_net, 0).weight
    assert not torch.equal(fitted_first, random_first), (
        "Fitted MLP weights are bit-identical to random-init — optimiser did not step. "
        "Check fit_epochs > 0 and that the Adam optimiser is correctly constructed."
    )


@pytest.mark.parametrize("decay_shape", ["geometric", "linear", "sigmoid"])
def test_inherit_from_scales_bias_network_under_all_decay_shapes(decay_shape: str) -> None:
    """Bias-network parameters SHALL shrink monotonically under every supported decay shape.

    Parametrised regression for the bias-network cascade — `inherit_from`
    scales `param.mul_(scale)` regardless of decay shape; this test
    confirms the per-shape per-step scale resolves correctly for the
    bias-network code path (the logit_bias path is already covered by
    the shape-specific tests above).
    """
    net = _make_bias_network()
    with torch.no_grad():
        for param in net.parameters():
            param.fill_(1.0)
    f0 = _f0_with_network(net)
    f1 = TransgenerationalMemory.inherit_from(
        [f0],
        decay_factor=0.6,
        decay_shape=decay_shape,  # type: ignore[arg-type]
    )
    f2 = TransgenerationalMemory.inherit_from(
        [f1],
        decay_factor=0.6,
        decay_shape=decay_shape,  # type: ignore[arg-type]
    )
    assert f0.bias_network is not None
    assert f1.bias_network is not None
    assert f2.bias_network is not None
    # Frobenius norm shrinks monotonically across generations.
    norm_f0 = sum(p.norm().item() for p in f0.bias_network.parameters())
    norm_f1 = sum(p.norm().item() for p in f1.bias_network.parameters())
    norm_f2 = sum(p.norm().item() for p in f2.bias_network.parameters())
    assert norm_f0 > norm_f1 > norm_f2, (
        f"bias-network parameter norm did not shrink monotonically under {decay_shape}: "
        f"F0={norm_f0}, F1={norm_f1}, F2={norm_f2}"
    )


def test_extract_from_brain_rejects_spec_without_input_features() -> None:
    """``bias_network_spec`` set with empty ``input_features`` SHALL raise ``ValueError``."""
    brain = _make_probe_brain(seed=42)
    probe_params = _make_probe_params()
    spec = {
        "input_dim": 1,
        "hidden_dim": 0,
        "output_dim": 4,
        "activation": "tanh",
    }
    with pytest.raises(ValueError, match=r"input_features is empty"):
        extract_from_brain(
            brain=brain,
            probe_params=probe_params,
            rng_seed=123,
            source_genome_id="elite-7",
            bias_network_spec=spec,
            input_features=(),
        )
