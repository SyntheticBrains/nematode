"""Tests for the brain plugin registry."""

# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel
from quantumnematode.brain.arch._registry import (
    _REGISTRY,
    Registration,
    family_members,
    get_all_registrations,
    get_registration,
    instantiate_brain,
    list_registered_brains,
    register_brain,
)
from quantumnematode.brain.arch.dtypes import BrainConfig, BrainType

if TYPE_CHECKING:
    from collections.abc import Generator


class _FakeBrainConfig(BrainConfig):
    """Stand-in config class for isolated registry tests."""

    extra_field: int = 0


class _UnrelatedConfig(BaseModel):
    """A non-BrainConfig type used to test config-type validation."""

    field: str = ""


@pytest.fixture(autouse=True)
def _isolated_registry() -> Generator[None, None, None]:
    """Snapshot and restore the module-level registry per test.

    The registry is populated at import time by every architecture module
    self-registering. Tests that exercise registration mechanics need a
    clean slate; this fixture snapshots, clears, and restores around every
    test in this module via ``autouse``.
    """
    snapshot = dict(_REGISTRY)
    _REGISTRY.clear()
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)


def test_register_and_get_round_trip() -> None:
    """A registered brain is retrievable by name with all fields intact."""

    @register_brain(
        name="mlpppo",
        config_cls=_FakeBrainConfig,
        brain_type=BrainType.MLP_PPO,
        families=("classical",),
    )
    class _FakeBrain:
        def __init__(self, config: _FakeBrainConfig) -> None:
            self.config = config

    reg = get_registration("mlpppo")
    assert isinstance(reg, Registration)
    assert reg.name == "mlpppo"
    assert reg.config_cls is _FakeBrainConfig
    assert reg.brain_cls is _FakeBrain
    assert reg.brain_type is BrainType.MLP_PPO
    assert reg.families == ("classical",)
    assert "mlpppo" in list_registered_brains()


def test_duplicate_name_raises() -> None:
    """A second registration under the same name fails loudly."""

    @register_brain(
        name="mlpppo",
        config_cls=_FakeBrainConfig,
        brain_type=BrainType.MLP_PPO,
        families=("classical",),
    )
    class _FirstBrain:
        def __init__(self, config: _FakeBrainConfig) -> None:
            pass

    with pytest.raises(ValueError, match="already registered"):

        @register_brain(
            name="mlpppo",
            config_cls=_FakeBrainConfig,
            brain_type=BrainType.MLP_PPO,
            families=("classical",),
        )
        class _SecondBrain:
            def __init__(self, config: _FakeBrainConfig) -> None:
                pass


def test_name_must_match_brain_type_value() -> None:
    """Name parameter must equal brain_type.value to pass the consistency check."""
    with pytest.raises(ValueError, match=r"does not match brain_type\.value"):

        @register_brain(
            name="not-the-enum-value",
            config_cls=_FakeBrainConfig,
            brain_type=BrainType.MLP_PPO,
            families=("classical",),
        )
        class _Brain:
            def __init__(self, config: _FakeBrainConfig) -> None:
                pass


def test_get_registration_unknown_name_raises() -> None:
    """Looking up an unregistered name raises with the available names listed."""
    with pytest.raises(ValueError, match="Unknown brain name"):
        get_registration("does_not_exist")


def test_instantiate_brain_round_trip() -> None:
    """instantiate_brain produces an instance of the registered Brain class."""

    @register_brain(
        name="mlpppo",
        config_cls=_FakeBrainConfig,
        brain_type=BrainType.MLP_PPO,
        families=("classical",),
    )
    class _FakeBrain:
        def __init__(self, config: _FakeBrainConfig, **kwargs) -> None:
            self.config = config
            self.kwargs = kwargs

    cfg = _FakeBrainConfig(extra_field=7)
    brain = instantiate_brain("mlpppo", cfg, shots=1024, device="cpu")
    assert isinstance(brain, _FakeBrain)
    assert brain.config is cfg
    assert brain.kwargs == {"shots": 1024, "device": "cpu"}


def test_instantiate_brain_wrong_config_type_raises() -> None:
    """A config that is not the registered config_cls raises with a clear message."""

    @register_brain(
        name="mlpppo",
        config_cls=_FakeBrainConfig,
        brain_type=BrainType.MLP_PPO,
        families=("classical",),
    )
    class _FakeBrain:
        def __init__(self, config: _FakeBrainConfig) -> None:
            pass

    with pytest.raises(ValueError, match="requires a _FakeBrainConfig"):
        instantiate_brain("mlpppo", _UnrelatedConfig())  # type: ignore[arg-type]


def test_instantiate_brain_unknown_name_raises() -> None:
    """instantiate_brain on an unknown name raises with available names listed."""
    with pytest.raises(ValueError, match="Unknown brain name"):
        instantiate_brain("not_a_brain", _FakeBrainConfig())


def test_family_members_partitions_registrations() -> None:
    """family_members returns the brain types whose registrations carry the tag."""

    @register_brain(
        name="mlpppo",
        config_cls=_FakeBrainConfig,
        brain_type=BrainType.MLP_PPO,
        families=("classical",),
    )
    class _ClassicalBrain:
        def __init__(self, config: _FakeBrainConfig) -> None:
            pass

    @register_brain(
        name="qsnnreinforce",
        config_cls=_FakeBrainConfig,
        brain_type=BrainType.QSNN_REINFORCE,
        families=("quantum", "spiking"),
    )
    class _QuantumSpikingBrain:
        def __init__(self, config: _FakeBrainConfig) -> None:
            pass

    assert family_members("classical") == {BrainType.MLP_PPO}
    assert family_members("quantum") == {BrainType.QSNN_REINFORCE}
    assert family_members("spiking") == {BrainType.QSNN_REINFORCE}
    assert family_members("nonexistent") == set()


def test_get_all_registrations_returns_copy() -> None:
    """get_all_registrations returns a copy that does not mutate the registry."""

    @register_brain(
        name="mlpppo",
        config_cls=_FakeBrainConfig,
        brain_type=BrainType.MLP_PPO,
        families=("classical",),
    )
    class _FakeBrain:
        def __init__(self, config: _FakeBrainConfig) -> None:
            pass

    copy = get_all_registrations()
    copy.clear()
    # Original registry untouched
    assert list_registered_brains() == {"mlpppo"}


# Test for the registry-vs-enum invariant lives in a sibling module
# (test_registry_enum_consistency.py) where it can see the production
# registry state without colliding with the autouse fixture above.
