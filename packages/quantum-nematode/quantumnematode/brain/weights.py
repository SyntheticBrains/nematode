"""Unified weight persistence for brain architectures.

Provides a component-based save/load system. Brains opt in by implementing the
``WeightPersistence`` protocol; brains that do not implement it are silently
skipped on save and raise ``TypeError`` on load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from quantumnematode.brain.arch._brain import Brain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class WeightComponent:
    """A named, serializable weight component.

    Parameters
    ----------
    name : str
        Component identifier (e.g. ``"policy"``, ``"qsnn"``).
    state : dict[str, Any]
        Serializable state -typically a PyTorch ``state_dict()`` or a dict
        of tensors / scalars.
    metadata : dict[str, Any]
        Shape and type information for diagnostics.
    """

    name: str
    state: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class WeightPersistence(Protocol):
    """Protocol for brains with explicit weight component support.

    Brains implement this to declare named weight components that can be
    individually saved, loaded, and filtered.
    """

    def get_weight_components(
        self,
        *,
        components: set[str] | None = None,
    ) -> dict[str, WeightComponent]:
        """Return weight components, optionally filtered.

        Parameters
        ----------
        components : set[str] | None
            If provided, return only the named components.  Unknown names
            raise ``ValueError``.

        Returns
        -------
        dict[str, WeightComponent]
            Mapping of component name → ``WeightComponent``.
        """
        ...

    def load_weight_components(
        self,
        components: dict[str, WeightComponent],
    ) -> None:
        """Load weight components into the brain.

        Parameters
        ----------
        components : dict[str, WeightComponent]
            Components to restore (subset allowed).
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METADATA_KEY = "_metadata"


def _build_shapes(
    components: dict[str, WeightComponent],
) -> dict[str, list[int]]:
    """Extract ``{component.param_name: shape}`` from weight components."""
    shapes: dict[str, list[int]] = {}
    for comp in components.values():
        for key, val in comp.state.items():
            if isinstance(val, torch.Tensor):
                shapes[f"{comp.name}.{key}"] = list(val.shape)
    return shapes


def _extract_episode_count(
    brain: Brain,
    components: dict[str, WeightComponent],
) -> int | None:
    """Best-effort extraction of episode count for metadata."""
    # Prefer the training_state component if present
    if "training_state" in components:
        ts = components["training_state"].state
        if "episode_count" in ts:
            return int(ts["episode_count"])
    # Fallback: look on the brain directly
    count = getattr(brain, "_episode_count", None)
    if count is not None:
        return int(count)
    return None


def _build_metadata(
    brain: Brain,
    components: dict[str, WeightComponent],
) -> dict[str, Any]:
    """Build the ``_metadata`` dict stored alongside components."""
    return {
        "brain_type": type(brain).__name__,
        "saved_at": datetime.now(tz=UTC).isoformat(),
        "components": list(components.keys()),
        "shapes": _build_shapes(components),
        "episode_count": _extract_episode_count(brain, components),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_weights(
    brain: Brain,
    path: Path,
    *,
    components: set[str] | None = None,
) -> Path | None:
    """Save brain weights to a single ``.pt`` file.

    Parameters
    ----------
    brain : Brain
        The brain instance to save.
    path : Path
        Destination file path.  Parent directories are created automatically.
    components : set[str] | None
        Optional filter -save only the named components.

    Returns
    -------
    Path | None
        The path written, or ``None`` if the brain does not implement
        ``WeightPersistence`` (no-op).
    """
    if not isinstance(brain, WeightPersistence):
        logger.debug(
            "Brain %s does not implement WeightPersistence -skipping save.",
            type(brain).__name__,
        )
        return None

    weight_components = brain.get_weight_components(components=components)

    save_dict: dict[str, Any] = {name: comp.state for name, comp in weight_components.items()}
    save_dict[_METADATA_KEY] = _build_metadata(brain, weight_components)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, path)
    logger.info("Weights saved to %s", path)
    return path


def load_weights(
    brain: Brain,
    path: Path,
    *,
    components: set[str] | None = None,
) -> None:
    """Load brain weights from a ``.pt`` file.

    Parameters
    ----------
    brain : Brain
        The brain instance to load weights into.
    path : Path
        Source file path.
    components : set[str] | None
        Optional filter -load only the named components from the file.

    Raises
    ------
    TypeError
        If the brain does not implement ``WeightPersistence``.
    FileNotFoundError
        If *path* does not exist.
    """
    if not isinstance(brain, WeightPersistence):
        msg = (
            f"Brain {type(brain).__name__} does not implement "
            f"WeightPersistence and cannot load weights."
        )
        raise TypeError(msg)

    path = Path(path)
    if not path.exists():
        msg = f"Weight file not found: {path}"
        raise FileNotFoundError(msg)

    checkpoint: dict[str, Any] = torch.load(path, weights_only=True)

    # Brain-type mismatch warning
    metadata = checkpoint.get(_METADATA_KEY, {})
    saved_brain_type = metadata.get("brain_type")
    current_brain_type = type(brain).__name__
    if saved_brain_type and saved_brain_type != current_brain_type:
        logger.warning(
            "Weight file brain_type mismatch: file has '%s', "
            "current brain is '%s'. Proceeding anyway.",
            saved_brain_type,
            current_brain_type,
        )

    # Build WeightComponent objects from loaded data
    available_keys = {k for k in checkpoint if k != _METADATA_KEY}
    loaded_components: dict[str, WeightComponent] = {}
    for key, state in checkpoint.items():
        if key == _METADATA_KEY:
            continue
        if components is not None and key not in components:
            continue
        loaded_components[key] = WeightComponent(name=key, state=state)

    # Warn if requested components were not found in the file
    if components is not None:
        missing = components - set(loaded_components)
        if missing:
            logger.warning(
                "Requested components %s not found in weight file (available: %s).",
                missing,
                available_keys,
            )

    brain.load_weight_components(loaded_components)
    logger.info("Weights loaded from %s", path)
