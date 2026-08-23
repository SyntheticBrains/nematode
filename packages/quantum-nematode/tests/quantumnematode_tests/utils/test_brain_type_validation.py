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
from quantumnematode.utils.config_loader import (
    BRAIN_CONFIG_MAP,
    configure_brain,
    configure_brain_from_container,
)


class _Brain:
    def __init__(self, name: str | None) -> None:
        self.name = name
        self.config = None


class _Config:
    def __init__(self, name: str | None) -> None:
        self.brain: _Brain | None = _Brain(name)


def _rejection_message(config: _Config) -> str | None:
    """Return the ``ValueError`` message ``configure_brain`` raises, else ``None``.

    A thin adapter over the real production call — it does not reimplement any of
    the validation. It exists so the assertion can sit outside the ``except`` block
    (ruff PT017) while still allowing the "did not raise at all" case, which
    ``pytest.raises`` would reject outright.
    """
    try:
        configure_brain(config)  # type: ignore[arg-type]
    except ValueError as exc:
        return str(exc)
    return None


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

    @pytest.mark.parametrize("name", sorted(_registry.list_registered_brains()))
    def test_a_registered_name_is_never_reported_as_unknown(self, name: str) -> None:
        """Guards against the parametrized test above passing for the wrong reason.

        The wrong reason is ``configure_brain`` rejecting *everything*. The first
        version of this test asserted ``registered in BRAIN_CONFIG_MAP`` and never
        called ``configure_brain`` at all — both sides of that assertion read the
        same ``_REGISTRY``, so it was tautological and would still have passed
        against a ``configure_brain`` mutated to reject every input.

        Every registered brain currently requires an explicit config block, so
        passing ``config=None`` still raises. What matters is *which* stage raises:
        it must be config resolution, not the registry check. Asserting the absence
        of "Unknown brain type" rather than asserting success is deliberate — it
        states exactly what this test can prove, and will not go stale if a brain
        later grows a usable default config.
        """
        message = _rejection_message(_Config(name)) or ""

        assert "Unknown brain type" not in message, (
            f"registered brain {name!r} was rejected by the registry check"
        )

    def test_absent_brain_section_is_rejected(self) -> None:
        """The sibling branch of the missing-name check, at ``config.brain is None``."""
        config = _Config("mlpppo")
        config.brain = None
        with pytest.raises(ValueError, match="No brain configuration found"):
            configure_brain(config)  # type: ignore[arg-type]

    def test_the_retired_brain_is_absent_from_the_config_map(self) -> None:
        """``BRAIN_CONFIG_MAP`` is registry-derived, so retirement propagates."""
        assert "qqlearning" not in BRAIN_CONFIG_MAP
        assert "qqlearning" not in _registry.list_registered_brains()

    def test_missing_brain_name_is_also_rejected(self) -> None:
        with pytest.raises(ValueError, match="No brain name specified"):
            configure_brain(_Config(None))  # type: ignore[arg-type]


class TestMultiAgentPathEnforcesTheSameRule:
    """``configure_brain_from_container`` is a second, independent implementation.

    The per-agent heterogeneous multi-agent path duplicates the registry check with
    its own code (no logging). The spec requirement covers both, so a change to only
    ``configure_brain`` must not let this one silently diverge.
    """

    def test_unknown_name_is_rejected_here_too(self) -> None:
        container = _Brain("qqlearning")
        with pytest.raises(ValueError, match="Unknown brain type: qqlearning"):
            configure_brain_from_container(container)  # type: ignore[arg-type]

    def test_a_registered_name_is_never_reported_as_unknown(self) -> None:
        try:
            configure_brain_from_container(_Brain("mlpppo"))  # type: ignore[arg-type]
        except ValueError as exc:
            message = str(exc)
        else:
            message = ""

        assert "Unknown brain type" not in message
