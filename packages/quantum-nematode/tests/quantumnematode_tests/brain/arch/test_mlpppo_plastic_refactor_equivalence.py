"""The matched-rule refactor leaves the MLP-PPO path byte-identical.

Adding a learning-rule selection, a topology that wraps the actor, and
plasticity fields to ``MLPPPOBrain`` must not move a single bit of the
default PPO path: every recorded MLP result depends on its construction
order (which fixes the torch-RNG stream) and its weight-component keys
(which is what saved weight files are keyed by).

The reference is a frozen copy of the brain taken BEFORE the refactor,
constructed in the same process on the same BLAS as the live brain, so
this cannot pass by construction and cannot fail for environment reasons.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.mlpppo import MLPPPOBrain, MLPPPOBrainConfig
from quantumnematode.brain.modules import ModuleName

from ._legacy_mlpppo_reference import LegacyMLPPPOBrain, LegacyMLPPPOBrainConfig

if TYPE_CHECKING:
    # Both classes carry the full MLP surface (learn, weight components);
    # the generic Brain Protocol does not, so type against the concrete pair.
    _MLP = MLPPPOBrain | LegacyMLPPPOBrain

_SEED = 3117
_DRIVE_SEED = 88
# Small buffer so a short drive triggers several PPO updates on both paths.
_COMMON = {
    "sensory_modules": [ModuleName.FOOD_CHEMOTAXIS],
    "rollout_buffer_size": 8,
    "num_minibatches": 2,
    "num_epochs": 2,
}
_STEPS = 30


def _pair(action_mode: str) -> tuple[_MLP, _MLP]:
    live = MLPPPOBrain(
        config=MLPPPOBrainConfig(seed=_SEED, action_mode=action_mode, **_COMMON),  # type: ignore[arg-type]
        device=DeviceType.CPU,
    )
    legacy = LegacyMLPPPOBrain(
        config=LegacyMLPPPOBrainConfig(seed=_SEED, action_mode=action_mode, **_COMMON),  # type: ignore[arg-type]
        device=DeviceType.CPU,
    )
    return live, legacy


def _drive(brain: _MLP) -> None:
    brain.prepare_episode()
    torch.manual_seed(_DRIVE_SEED)
    for step in range(_STEPS):
        brain.run_brain(
            BrainParams(
                food_gradient_strength=0.2 + 0.02 * step,
                food_gradient_direction=0.15 * step - 1.0,
            ),
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )
        brain.learn(
            BrainParams(),
            reward=0.3 * ((step % 4) - 1.5),
            episode_done=(step == _STEPS - 1),
        )


def _params(brain: _MLP) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for prefix in ("actor", "critic"):
        for name, p in getattr(brain, prefix).named_parameters():
            out[f"{prefix}.{name}"] = p.detach()
    log_std = getattr(brain, "log_std", None)
    if isinstance(log_std, torch.nn.Parameter):
        out["log_std"] = log_std.detach()
    return out


@pytest.mark.parametrize("action_mode", ["discrete", "continuous"])
class TestPPOPathIsByteIdenticalToTheFrozenReference:
    """Construction and training match the pre-refactor brain bit for bit."""

    def test_construction_is_identical(self, action_mode: str) -> None:
        live, legacy = _pair(action_mode)
        a, b = _params(live), _params(legacy)
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]), key

    def test_training_is_identical(self, action_mode: str) -> None:
        live, legacy = _pair(action_mode)
        _drive(live)
        _drive(legacy)
        a, b = _params(live), _params(legacy)
        # Guard against a vacuous pass: the drive must have actually updated.
        fresh = _params(_pair(action_mode)[0])
        assert any(not torch.equal(a[k], fresh[k]) for k in a), "drive produced no PPO update"
        for key in a:
            assert torch.equal(a[key], b[key]), key

    def test_weight_component_keys_are_unchanged(self, action_mode: str) -> None:
        live, legacy = _pair(action_mode)
        assert live.get_weight_components().keys() == legacy.get_weight_components().keys()
