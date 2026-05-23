"""Brain plugin registry.

Single source of truth for the mapping from a brain name to its config
class, Brain class, ``BrainType`` enum member, and family tags. Each
architecture module declares its registration via the ``@register_brain``
decorator at import time; consumers instantiate brains via
``instantiate_brain(name, config, ...)``.

The registry replaces the previous hand-maintained dispatcher
(``setup_brain_model``) and the ``BRAIN_CONFIG_MAP`` literal. Adding a new
architecture is a per-architecture decorator + enum-member addition; no
per-architecture branches survive in the dispatcher or YAML loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from quantumnematode.brain.arch.dtypes import BrainConfig, BrainType
from quantumnematode.logging_config import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from quantumnematode.brain.arch._brain import Brain


_BrainClsT = TypeVar("_BrainClsT", bound=type)


@dataclass(frozen=True)
class Registration:
    """One registry entry binding a brain name to its types and metadata."""

    name: str
    config_cls: type[BrainConfig]
    brain_cls: type
    brain_type: BrainType
    families: tuple[str, ...]


_REGISTRY: dict[str, Registration] = {}


def register_brain(
    name: str,
    config_cls: type[BrainConfig],
    brain_type: BrainType,
    families: tuple[str, ...],
) -> Callable[[_BrainClsT], _BrainClsT]:
    """Class decorator that registers a Brain class in the plugin registry.

    Parameters
    ----------
    name
        The string identifier used in YAML configs and as the registry key.
        Must match ``brain_type.value``.
    config_cls
        The Pydantic config model class the brain accepts.
    brain_type
        The ``BrainType`` enum member that pairs with this brain.
    families
        Family tags for type-alias derivation. Common tags: ``"classical"``,
        ``"quantum"``, ``"spiking"``. An architecture may carry multiple
        tags (e.g. a quantum spiking network uses ``("quantum", "spiking")``).

    Raises
    ------
    ValueError
        If ``name`` is already registered, or if ``name`` does not match
        ``brain_type.value``.
    """
    if name != brain_type.value:
        msg = (
            f"register_brain name '{name}' does not match brain_type.value "
            f"'{brain_type.value}'. The two must agree so the registry-enum "
            f"consistency check succeeds at import time."
        )
        raise ValueError(msg)

    def decorator(cls: _BrainClsT) -> _BrainClsT:
        if name in _REGISTRY:
            existing = _REGISTRY[name]
            msg = (
                f"Brain name '{name}' is already registered to "
                f"{existing.brain_cls.__module__}.{existing.brain_cls.__name__}. "
                f"Cannot re-register to {cls.__module__}.{cls.__name__}."
            )
            raise ValueError(msg)
        _REGISTRY[name] = Registration(
            name=name,
            config_cls=config_cls,
            brain_cls=cls,
            brain_type=brain_type,
            families=families,
        )
        return cls

    return decorator


def get_registration(name: str) -> Registration:
    """Return the registration entry for ``name``.

    Raises ``ValueError`` if ``name`` is unknown, with the available names
    in the error message.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        msg = f"Unknown brain name '{name}'. Available: {available}"
        raise ValueError(msg)
    return _REGISTRY[name]


def list_registered_brains() -> set[str]:
    """Return the set of registered brain names."""
    return set(_REGISTRY.keys())


def get_all_registrations() -> dict[str, Registration]:
    """Return a copy of the full registry table.

    Useful for callers that need to enumerate all registrations (e.g. the
    YAML loader deriving its name-to-config-class map).
    """
    return dict(_REGISTRY)


def family_members(family: str) -> set[BrainType]:
    """Return the set of ``BrainType`` members carrying the given family tag.

    Backs the ``QUANTUM_BRAIN_TYPES`` / ``CLASSICAL_BRAIN_TYPES`` /
    ``SPIKING_BRAIN_TYPES`` sets in ``dtypes.py``.
    """
    return {reg.brain_type for reg in _REGISTRY.values() if family in reg.families}


def instantiate_brain(
    name: str,
    config: BrainConfig,
    **infra_kwargs: Any,  # noqa: ANN401 — kwargs forwarded to arch-specific __init__
) -> Brain:
    """Instantiate a registered brain by name.

    Parameters
    ----------
    name
        The registered brain name (matches ``BrainType.X.value`` for some X).
    config
        The brain config instance. Must be an instance of the registered
        ``config_cls`` (the ``BrainConfig`` subclass declared on registration).
    **infra_kwargs
        Infrastructure-level keyword arguments forwarded to the brain's
        ``__init__`` (e.g. ``shots``, ``device``, ``learning_rate``,
        ``gradient_method``, ``gradient_max_norm``,
        ``parameter_initializer_config``, ``perf_mgmt``). Each architecture
        accepts the subset relevant to it.

    Returns
    -------
    Brain
        A Brain Protocol-conforming instance.

    Raises
    ------
    ValueError
        If ``name`` is unknown or ``config`` is not an instance of the
        registered config class.
    """
    reg = get_registration(name)
    if not isinstance(config, reg.config_cls):
        msg = (
            f"The '{name}' brain architecture requires a {reg.config_cls.__name__}. "
            f"Provided brain config type: {type(config).__name__}."
        )
        logger.error(msg)
        raise ValueError(msg)  # noqa: TRY004 — preserves existing dispatcher contract
    return reg.brain_cls(config=config, **infra_kwargs)


def assert_registry_matches_enum() -> None:
    """Assert that ``BrainType`` enum string values equal the registered names.

    Called from ``brain/arch/__init__.py`` after all architecture modules
    have been imported (so all decorators have run). Raises a descriptive
    exception identifying any names present on one side but not the other.
    """
    enum_values = {bt.value for bt in BrainType}
    registered = list_registered_brains()
    if enum_values != registered:
        missing_in_registry = enum_values - registered
        missing_in_enum = registered - enum_values
        parts = []
        if missing_in_registry:
            parts.append(
                f"BrainType members not registered: {sorted(missing_in_registry)}",
            )
        if missing_in_enum:
            parts.append(
                f"Registered names not in BrainType: {sorted(missing_in_enum)}",
            )
        msg = "Brain plugin registry and BrainType enum are out of sync. " + " | ".join(parts)
        raise RuntimeError(msg)
