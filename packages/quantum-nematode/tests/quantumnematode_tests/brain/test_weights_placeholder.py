"""The ``{seed}`` placeholder in a configured weights path."""

from __future__ import annotations

import pytest
from quantumnematode.brain.weights import resolve_weights_path


def test_placeholder_resolves_to_the_seed() -> None:
    """``{seed}`` becomes the run seed."""
    assert resolve_weights_path("clones/arm_seed{seed}.pt", 5) == "clones/arm_seed5.pt"


def test_plain_path_is_untouched() -> None:
    """A path without the placeholder is returned as written."""
    assert resolve_weights_path("weights/final.pt", 5) == "weights/final.pt"


def test_unresolved_placeholder_is_rejected() -> None:
    """Any other brace is a placeholder nobody resolves."""
    with pytest.raises(ValueError, match="unresolved placeholder"):
        resolve_weights_path("clones/arm_{run}.pt", 5)
