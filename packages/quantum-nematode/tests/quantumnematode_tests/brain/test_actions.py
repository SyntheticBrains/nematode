"""Unit tests for the ActionData payload invariant.

`ActionData` must carry exactly one action payload: the discrete `action`
(grid substrate) or the `continuous` `(speed, turn)` vector (continuous-2D
substrate), never both and never neither.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from quantumnematode.brain.actions import Action, ActionData


class TestActionDataPayloadInvariant:
    """Exactly one of `action` / `continuous` must be set."""

    def test_discrete_action_is_valid(self) -> None:
        """A discrete action with no continuous vector is accepted."""
        data = ActionData(state="forward", action=Action.FORWARD, probability=0.5)
        assert data.action is Action.FORWARD
        assert data.continuous is None

    def test_continuous_action_is_valid(self) -> None:
        """A continuous vector with no discrete action is accepted."""
        data = ActionData(state="continuous", probability=0.5, continuous=(0.3, -0.1))
        assert data.action is None
        assert data.continuous == (0.3, -0.1)

    def test_neither_payload_is_rejected(self) -> None:
        """Setting neither action nor continuous fails validation."""
        with pytest.raises(ValidationError):
            ActionData(state="x", probability=0.5)

    def test_both_payloads_is_rejected(self) -> None:
        """Setting both action and continuous fails validation."""
        with pytest.raises(ValidationError):
            ActionData(
                state="x",
                action=Action.FORWARD,
                probability=0.5,
                continuous=(0.3, -0.1),
            )
