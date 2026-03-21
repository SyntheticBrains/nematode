"""Brain state snapshot and restore for plasticity evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from quantumnematode.brain.arch._brain import Brain


def _get_torch_modules(brain: Brain) -> dict[str, Any]:
    """Discover all PyTorch modules and optimizers on a brain instance.

    Returns a dict of {name: module_or_optimizer} for snapshotting.
    """
    modules: dict[str, Any] = {}
    # Common PPO-based attributes (QRH, CRH, MLP PPO)
    for attr in (
        "actor",
        "critic",
        "feature_norm",
        "optimizer",
        # HybridQuantum / HybridClassical
        "cortex_actor",
        "cortex_critic",
        "cortex_actor_optimizer",
        "cortex_critic_optimizer",
        "qsnn_optimizer",
        "reflex_mlp",
    ):
        obj = getattr(brain, attr, None)
        if obj is not None and hasattr(obj, "state_dict"):
            modules[attr] = obj

    # QSNN QLIF layers (stored as nn.Module on HybridQuantum)
    for attr in ("_qlif_sensory", "_qlif_hidden", "_qlif_motor"):
        obj = getattr(brain, attr, None)
        if obj is not None and hasattr(obj, "state_dict"):
            modules[attr] = obj

    return modules


def snapshot_brain_state(brain: Brain) -> dict[str, dict[str, Any]]:
    """Save all brain state_dicts into an in-memory snapshot."""
    modules = _get_torch_modules(brain)
    return {name: deepcopy(mod.state_dict()) for name, mod in modules.items()}


def restore_brain_state(
    brain: Brain,
    snapshot: dict[str, dict[str, Any]],
) -> None:
    """Restore brain state from a snapshot and clear any PPO buffer."""
    modules = _get_torch_modules(brain)

    # Fail-fast: verify all snapshot keys exist before restoring any state
    missing = [name for name in snapshot if name not in modules]
    if missing:
        msg = (
            f"Cannot restore brain state: snapshot contains keys {missing} "
            f"not found on brain (available: {list(modules.keys())})"
        )
        raise ValueError(msg)

    for name, state_dict in snapshot.items():
        modules[name].load_state_dict(deepcopy(state_dict))

    # Clear PPO buffer to prevent eval experience leaking into training
    buffer = getattr(brain, "buffer", None) or getattr(brain, "ppo_buffer", None)
    if buffer is not None and hasattr(buffer, "reset"):
        buffer.reset()


def save_checkpoint(
    brain: Brain,
    checkpoint_dir: Path,
    phase_name: str,
) -> Path:
    """Save brain weights + optimizer state to disk."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_post_{phase_name}.pt"
    snapshot = snapshot_brain_state(brain)
    torch.save(snapshot, path)
    return path
