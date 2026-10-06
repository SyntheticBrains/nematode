"""B.2a's positive controls: the configs, the pilot rule, the reading and the constants.

Covers the connectome-ppo-brain requirement "Across-step leaky-integrator dynamics" through the
configs that run it, and the registration's tau rule and non-inferiority reading.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest
from quantumnematode.utils.config_loader import load_simulation_config

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))
sys.path.insert(0, str(_root / "scripts" / "campaigns"))

import across_step_control as asc  # noqa: E402  # pyright: ignore[reportMissingImports]
import boundary_null as bn  # noqa: E402  # pyright: ignore[reportMissingImports]
import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_across_step_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]

_SCENARIOS = _root / "configs" / "scenarios"
_DIR = {"hard350": "foraging", "thermal": "thermal_foraging"}


@functools.cache
def _loaded(scenario: str, stem: str) -> dict[str, Any]:
    """Load a whole simulation config as a run gets it, through the real loader."""
    return load_simulation_config(str(_SCENARIOS / scenario / f"{stem}.yml")).model_dump()


# ── Configs ──────────────────────────────────────────────────────────────────────────────────


class TestConfigs:
    def test_the_generator_and_the_analysis_agree(self) -> None:
        assert gen.TAUS == asc.TAUS
        assert gen.MLP_STEM == asc.MLP_STEM
        for cell, (_scenario, learn, frozen) in gen.CELLS.items():
            assert asc.CELLS[cell]["settling"] == (learn, frozen)

    @pytest.mark.parametrize("tau", asc.TAUS)
    @pytest.mark.parametrize("cell", sorted(asc.CELLS))
    def test_every_arm_is_a_committed_config(self, cell: str, tau: float) -> None:
        for stem in asc.stems(tau)[cell].values():
            assert (_SCENARIOS / _DIR[cell] / f"{stem}.yml").is_file(), stem

    @pytest.mark.parametrize("tau", asc.TAUS)
    @pytest.mark.parametrize("cell", sorted(asc.CELLS))
    def test_each_leaky_arm_differs_from_its_parent_in_dynamics_alone(
        self,
        cell: str,
        tau: float,
    ) -> None:
        table = asc.stems(tau)[cell]
        for leaky, settling in (("wt_learn", "rn_learn"), ("wt_frozen", "rn_frozen")):
            child = _loaded(_DIR[cell], table[leaky])
            expected = dict(_loaded(_DIR[cell], table[settling]))
            expected["brain"] = {**expected["brain"]}
            expected["brain"]["config"] = {
                **expected["brain"]["config"],
                "dynamics": "leaky",
                "membrane_tau_steps": tau,
            }
            assert child == expected

    def test_the_thermal_mlp_runs_the_connectome_cell(self) -> None:
        mlp = _loaded("thermal_foraging", asc.MLP_STEM)
        cell = _loaded("thermal_foraging", asc.CELLS["thermal"]["settling"][0])
        parent = _loaded("thermal_foraging", gen.MLP_PARENT)
        assert {k: v for k, v in mlp.items() if k != "brain"} == {
            k: v for k, v in cell.items() if k != "brain"
        }
        assert mlp["brain"] == parent["brain"]
        assert mlp["environment"]["foraging"]["target_foods_to_collect"] == 35


# ── Registration constants ───────────────────────────────────────────────────────────────────


class TestRegistration:
    def test_the_margins_are_the_cells_committed_minimums(self) -> None:
        assert asc.CELLS["hard350"]["margin"] == pytest.approx(2 / 3 * 0.02146875)
        assert asc.CELLS["hard350"]["margin"] == bn.MINIMUM
        assert asc.CELLS["thermal"]["margin"] == ts.MINIMUM

    def test_the_bands_are_the_committed_settling_runs(self) -> None:
        assert asc.CELLS["hard350"]["seeds"] == bn.SEEDS
        assert asc.CELLS["thermal"]["seeds"] == ts.SEEDS

    def test_pilot_and_mlp_seeds_are_disjoint_from_the_bands(self) -> None:
        bands = set(bn.SEEDS) | set(ts.SEEDS)
        assert not set(asc.PILOT_SEEDS) & bands
        assert not set(asc.MLP_SEEDS) & bands
        assert not set(asc.PILOT_SEEDS) & set(asc.MLP_SEEDS)

    def test_the_table_fits_the_gate_preflight(self) -> None:
        for arms in asc.STEMS.values():
            assert set(arms) == set(gp.ARMS)

    def test_scoring_refuses_an_unregistered_tau(self, tmp_path: Path) -> None:
        if asc.REGISTERED_TAU is not None:
            pytest.skip("the registered tau is set")
        with pytest.raises(asc.AcrossStepError, match="REGISTERED_TAU"):
            asc.score([tmp_path], tmp_path)


# ── The pilot rule ───────────────────────────────────────────────────────────────────────────


def _candidate(status: str = "readable", hard: float = 0.0, thermal: float = 0.0) -> dict:
    return {
        "status": {"hard350": status, "thermal": "readable"},
        "difference": {"hard350": hard, "thermal": thermal},
    }


class TestSelection:
    def test_tau_one_holds_inside_the_band(self) -> None:
        got = asc.select(
            {0.2: _candidate(hard=0.04), 1.0: _candidate(), 5.0: _candidate(hard=-0.1)},
        )
        assert got["tau"] == 1.0

    def test_a_clear_lead_over_tau_one_wins(self) -> None:
        got = asc.select(
            {0.2: _candidate(hard=0.06, thermal=0.07), 1.0: _candidate(), 5.0: _candidate()},
        )
        assert got["tau"] == 0.2

    def test_the_worse_cell_decides_the_score(self) -> None:
        got = asc.select(
            {0.2: _candidate(hard=0.2, thermal=0.01), 1.0: _candidate(), 5.0: _candidate()},
        )
        assert got["tau"] == 1.0

    def test_without_tau_one_the_best_eligible_wins(self) -> None:
        got = asc.select(
            {
                0.2: _candidate(hard=-0.02),
                1.0: _candidate(status="fails_floor"),
                5.0: _candidate(hard=0.01),
            },
        )
        assert got["tau"] == 5.0

    def test_no_eligible_tau_stops(self) -> None:
        got = asc.select({tau: _candidate(status="near_bar") for tau in asc.TAUS})
        assert got["tau"] is None


# ── The reading ──────────────────────────────────────────────────────────────────────────────


def _gates(*, learns: bool = True, passes: bool = True, saturated: bool = False) -> dict:
    return {
        "wt": {"vs_floor": {"ci_lo": 0.1 if learns else -0.1}},
        "gate_passes": passes,
        "saturated": saturated,
    }


def _t(lo: float, hi: float) -> dict[str, float]:
    return {"ci_lo": lo, "ci_hi": hi}


class TestReading:
    margin = 0.0143

    @pytest.mark.parametrize(
        ("test", "verdict"),
        [
            (_t(-0.010, 0.020), "non_inferior"),
            (_t(0.010, 0.040), "non_inferior"),
            (_t(-0.040, -0.020), "inferior"),
            (_t(-0.030, 0.000), "unresolved"),
        ],
    )
    def test_the_non_inferiority_map(self, test: dict[str, float], verdict: str) -> None:
        assert asc.read_cell(_gates(), test, self.margin) == verdict

    def test_a_dynamical_arm_that_does_not_learn(self) -> None:
        got = asc.read_cell(_gates(learns=False, passes=False), _t(0.0, 0.1), self.margin)
        assert got == "unlearnable"

    def test_a_saturated_cell_is_unreadable(self) -> None:
        assert asc.read_cell(_gates(saturated=True), _t(0.0, 0.1), self.margin) == "unreadable"


class TestMlpGate:
    def test_missing_runs_read_incomplete(self, tmp_path: Path) -> None:
        assert asc.mlp_gate([tmp_path])["verdict"] == "incomplete"
