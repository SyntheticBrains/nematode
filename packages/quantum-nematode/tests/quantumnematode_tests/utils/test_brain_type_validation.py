"""Coverage for the `configuration-system` brain-type validation requirement.

The spec requires that an unregistered brain type is **rejected with an error
naming it**, and never silently substituted with a default or nearest match. The
behaviour existed but nothing tested it: the pre-existing unknown-brain tests cover
the hyperparameter schema and the evolution encoder, not ``configure_brain``.

Added while retiring ``qqlearning`` (#282) — a retired architecture is exactly the
case where a silent fallback would be most damaging, because the config looks
plausible and names something that used to work.
"""

from __future__ import annotations

import pytest
from quantumnematode.brain.arch import _registry
from quantumnematode.utils.config_loader import BRAIN_CONFIG_MAP, configure_brain


class _Brain:
    def __init__(self, name: str | None) -> None:
        self.name = name
        self.config = None


class _Config:
    def __init__(self, name: str | None) -> None:
        self.brain = _Brain(name)


class TestUnregisteredBrainTypeIsRejected:
    """An unknown brain type must fail loudly, never fall back."""

    @pytest.mark.parametrize(
        "name",
        [
            "qqlearning",  # retired 2026-08-23 (#282)
            "qmodular",  # a "legacy alias" the spec used to claim existed
            "mlp",  # ditto
            "definitely-not-a-brain",
        ],
    )
    def test_unknown_name_raises_naming_the_type(self, name: str) -> None:
        with pytest.raises(ValueError, match=f"Unknown brain type: {name}"):
            configure_brain(_Config(name))  # type: ignore[arg-type]

    def test_a_registered_name_is_not_rejected(self) -> None:
        """Guards against the test passing for the wrong reason."""
        registered = sorted(_registry.list_registered_brains())[0]

        assert registered in BRAIN_CONFIG_MAP

    def test_the_retired_brain_is_absent_from_the_config_map(self) -> None:
        """``BRAIN_CONFIG_MAP`` is registry-derived, so retirement propagates."""
        assert "qqlearning" not in BRAIN_CONFIG_MAP
        assert "qqlearning" not in _registry.list_registered_brains()

    def test_missing_brain_name_is_also_rejected(self) -> None:
        with pytest.raises(ValueError, match="No brain name specified"):
            configure_brain(_Config(None))  # type: ignore[arg-type]
