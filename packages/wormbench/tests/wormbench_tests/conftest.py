"""Fixtures for wormbench's own tests: no network, and the data root restored after each test."""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import pytest

from wormbench import data

if TYPE_CHECKING:
    from collections.abc import Iterator


class _NoNetworkSocket(socket.socket):
    def __init__(self, *args: object, **kwargs: object) -> None:
        msg = "wormbench's tests make no network connection"
        raise RuntimeError(msg)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every socket, so a test that reaches for the network fails."""
    monkeypatch.setattr(socket, "socket", _NoNetworkSocket)


@pytest.fixture(autouse=True)
def _restore_data_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Save the host data root and the environment variable, and restore both after the test."""
    saved = data.get_root()
    monkeypatch.delenv(data.ENVIRONMENT_VARIABLE, raising=False)
    yield
    data.set_root(saved)
