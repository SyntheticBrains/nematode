"""YAML compatibility regression for the registry-backed config loader.

After the dispatcher and ``BRAIN_CONFIG_MAP`` were rewired through the
brain plugin registry, every existing scenario YAML under
``configs/scenarios/`` must still load to a valid brain configuration via
``configure_brain(load_simulation_config(...))``. This test enumerates
every YAML in that tree and asserts the load round-trip succeeds.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from quantumnematode.utils.config_loader import (
    BRAIN_CONFIG_MAP,
    configure_brain,
    load_simulation_config,
)

# Resolve the repo-level configs/scenarios/ from this test file's location.
# .../packages/quantum-nematode/tests/quantumnematode_tests/utils/THIS_FILE
# → up five parents to repo root, then configs/scenarios/.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_CONFIGS_DIR = _REPO_ROOT / "configs" / "scenarios"


# Scenario YAMLs that pre-date a config-class field rename and currently
# fail validation on main. These are stale config-archeology, not registry
# regressions; tracked here so the regression test stays meaningful but
# does not block on bitrot unrelated to this change.
_STALE_YAMLS: frozenset[str] = frozenset(
    {
        "configs/scenarios/foraging/qrc_small_oracle.yml",
        "configs/scenarios/foraging/qsnnreinforce_small_oracle.yml",
    },
)


def _iter_scenario_yamls() -> list[Path]:
    """Return the sorted list of scenario YAMLs; raise if none are found.

    Returning an empty list would let parametrize silently collect zero
    cases — a "no tests ran" outcome that pytest reports as passing. We
    want a hard failure if the discovery is broken (e.g. configs/scenarios/
    moved, the relative-paths chain to ``_CONFIGS_DIR`` is wrong).
    """
    if not _CONFIGS_DIR.exists():
        msg = (
            f"Scenario configs directory not found at {_CONFIGS_DIR}. "
            f"The YAML compat regression cannot enumerate cases."
        )
        raise FileNotFoundError(msg)
    paths = sorted(_CONFIGS_DIR.rglob("*.yml"))
    if not paths:
        msg = (
            f"No scenario YAMLs found under {_CONFIGS_DIR}. "
            f"The YAML compat regression would otherwise silently collect "
            f"zero cases."
        )
        raise RuntimeError(msg)
    return paths


def test_brain_config_map_has_every_registered_brain() -> None:
    """The map and the registry are in lock-step (no entries missing on either side)."""
    from quantumnematode.brain.arch._registry import list_registered_brains

    assert set(BRAIN_CONFIG_MAP.keys()) == list_registered_brains(), (
        "BRAIN_CONFIG_MAP and the brain plugin registry have diverged. "
        "The map is derived from the registry at import time; if you see "
        "this, an architecture is missing a `@register_brain(...)` decorator."
    )


@pytest.mark.parametrize(
    "config_path",
    _iter_scenario_yamls(),
    ids=lambda p: p.relative_to(_REPO_ROOT).as_posix(),
)
def test_scenario_yaml_loads_via_registry(config_path: Path) -> None:
    """Every scenario YAML produces a valid brain config via the registry.

    Failure modes this catches:

    - A brain name in the YAML that no architecture registers (typo,
      stale config, or an arch the registry forgot)
    - A YAML field that the registered ``config_cls`` doesn't accept (the
      config-class shape has shifted since the YAML was written)
    - Pydantic validation errors anywhere in the simulation-config tree
    """
    rel = config_path.relative_to(_REPO_ROOT).as_posix()
    if rel in _STALE_YAMLS:
        pytest.xfail(
            f"{rel} pre-dates a config-class field rename; pre-existing breakage on main",
        )

    sim_config = load_simulation_config(str(config_path))

    # Some YAMLs intentionally have ``brain: null`` for non-simulation
    # configs; skip those rather than fail the parametrised case.
    if sim_config.brain is None or sim_config.brain.name is None:
        pytest.skip(f"{config_path.name} has no brain section")

    brain_config = configure_brain(sim_config)
    assert brain_config is not None
    # The resolved config-class type must match the registry's record.
    registered_cls = BRAIN_CONFIG_MAP[sim_config.brain.name]
    assert isinstance(brain_config, registered_cls), (
        f"configure_brain produced {type(brain_config).__name__}, "
        f"expected {registered_cls.__name__} per BRAIN_CONFIG_MAP['{sim_config.brain.name}']"
    )
