"""The shortcut audit walks the cell it is given, from every sensor that cell injects into.

Covers D24's shortcut audit as G.0 runs it: the thermal cell's sources include its thermal sensors,
the food cell's are its food sensors alone, and the gap nulls are walked like the others.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import sensory_motor_hops as hops  # noqa: E402  # pyright: ignore[reportMissingImports]

_THERMAL = (
    _root
    / "configs/scenarios/thermal_foraging"
    / "connectomeppo_small_continuous2d_thermal_klinotaxis_g0_count_wt_t35.yml"
)
_FOOD = {"ASEL", "ASER", "AWCL", "AWCR", "AWAL", "AWAR"}


def _names(topology: object, indices: list[int]) -> set[str]:
    names = topology.neuron_names  # type: ignore[attr-defined]
    return {names[i] for i in indices}


def test_the_thermal_cell_s_sources_include_its_thermal_sensors() -> None:
    """On the thermal cell AFD is injected, so it is a source beside the food sensors."""
    topology = hops._topology("wild_type", 9851, _THERMAL)
    sources = _names(topology, hops.injected_sensors(topology))
    assert sources >= {"AFDL", "AFDR"}
    assert sources >= _FOOD
    assert len(sources) == len(_FOOD) + 2


def test_the_food_cell_s_sources_are_its_food_sensors() -> None:
    """The default food cell injects into its six food sensors and nothing else."""
    topology = hops._topology("wild_type", 9851)
    assert _names(topology, hops.injected_sensors(topology)) == _FOOD


def test_the_gap_nulls_are_walked_on_the_thermal_cell() -> None:
    """Each gap null holds the chemical graph, so only its gap edges can change the reach."""
    for null in (
        "rewired_gap_junctions_only",
        "rewired_gap_outliers_held",
        "rewired_gap_outliers_scattered",
    ):
        result = hops.compare((9851,), null, _THERMAL)
        assert result["null_wiring"] == null
        assert result["config"].endswith(_THERMAL.name)
        reach = hops.short_reach(hops._topology(null, 9851, _THERMAL))
        assert set(reach) == {1, 2}
        assert 0 <= reach[1] <= reach[2] <= result["wild_type"]["n_motor"]
