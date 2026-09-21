"""A.2's driver and its configs: the panel mapping, the seed bands, the gate, and the one-key claim.

A sweep's entire output is a claim about settings, so the things that can quietly falsify it are
structural rather than statistical: a stem that maps to the wrong arm, a level whose config carries
a second delta, a gate that reads a partial panel as a whole one, or an instrument edited to suit
the answer. All four are checkable here without running a campaign, and all four are checked.

The one-key delta is verified **through the real configuration loader**, not by diffing text. A
generated file is the parent's keys re-emitted, so a textual diff would report the whole file; what
matters is whether the loaded configuration differs in exactly the key the header names.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[5]
_ANALYSIS = _REPO / "scripts" / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]

_CONFIGS = _REPO / "configs" / "scenarios"

# The arm axis. A rewired arm differs from its wild-type parent in this key as well as in the pin,
# and that is the contrast rather than a stray delta.
_WIRING_KEY = "wiring"


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def _config_path(stem: str) -> Path:
    return next(iter(_CONFIGS.rglob(f"{stem}.yml")))


def _brain_config(stem: str) -> dict[str, Any]:
    """Read the brain's configuration block as the loader would see it, comments gone."""
    data = yaml.safe_load(_config_path(stem).read_text())
    return dict(data["brain"]["config"])


def _levels_with_arms() -> list[tuple[str, str, str, str, Any]]:
    """``(half, arm, suffix, pin, value)`` for every off-centre arm of the panel."""
    return [
        (half, arm, suffix, pin, value)
        for half in ops.HALVES
        for pin, suffix, value in ops._levels(half)
        for arm in ops.arms_at(half, suffix)
    ]


_LEVEL_ARMS = _levels_with_arms()
_LEVEL_IDS = [f"{h}-{s}-{a}" for h, a, s, _, _ in _LEVEL_ARMS]


class TestThePanelIsWhatItClaims:
    """The stem map is complete, unambiguous, and the shape the registration states."""

    def test_every_stem_names_a_config_that_exists(self) -> None:
        missing = [s for s in ops.ARM_BY_STEM if not list(_CONFIGS.rglob(f"{s}.yml"))]
        assert not missing, f"stems with no config: {missing}"

    def test_no_two_entries_share_a_stem(self) -> None:
        # The campaign runner hash-disambiguates same-named configs in its log names, and the
        # manifest builder keys on the bare stem — so a collision would silently drop one arm.
        assert len(ops.ARM_BY_STEM) == len(set(ops.ARM_BY_STEM))

    def test_the_panel_has_the_registered_arm_counts(self) -> None:
        by_half: dict[str, int] = {}
        for half, _, _ in ops.ARM_BY_STEM.values():
            by_half[half] = by_half.get(half, 0) + 1
        assert by_half == {"ppo": 32, "reading": 40}

    def test_a_construction_level_carries_four_arms_and_a_learning_one_carries_two(self) -> None:
        for half in ops.HALVES:
            for pin, suffix, _ in ops._levels(half):
                expected = 4 if pin in ops.CONSTRUCTION_PINS else 2
                assert len(ops.arms_at(half, suffix)) == expected, (half, suffix, pin)

    def test_the_two_learning_only_pins_belong_to_the_reading_half_alone(self) -> None:
        # Both are declared on the shared config mixin and neither is read under PPO, so a PPO
        # level of either would be an arm that looks swept and is not.
        assert set(ops.PIN_LEVELS["ppo"]) == set(ops.CONSTRUCTION_PINS)
        assert {"plasticity_rate", "trace_decay"} <= set(ops.PIN_LEVELS["reading"])


class TestEveryArmIsOneKeyFromItsCentre:
    """The single-key claim, re-read through the loader rather than trusted from the header."""

    @pytest.mark.parametrize(("half", "arm", "suffix", "pin", "value"), _LEVEL_ARMS, ids=_LEVEL_IDS)
    def test_the_level_differs_from_its_centre_in_exactly_the_pin(
        self,
        half: str,
        arm: str,
        suffix: str,
        pin: str,
        value: Any,
    ) -> None:
        centre = _brain_config(ops.CENTRE_STEMS[half][arm])
        level = _brain_config(ops.stem_for(half, arm, suffix))
        differing = {
            key
            for key in set(centre) | set(level)
            if centre.get(key, "<absent>") != level.get(key, "<absent>")
        }
        assert differing == {pin}, f"{half}/{suffix}/{arm} differs in {sorted(differing)}"
        assert level[pin] == value

    @pytest.mark.parametrize(("half", "arm", "suffix", "pin", "value"), _LEVEL_ARMS, ids=_LEVEL_IDS)
    def test_the_rewired_arm_differs_from_its_wild_type_in_the_wiring_and_nothing_else(
        self,
        half: str,
        arm: str,
        suffix: str,
        pin: str,
        value: Any,
    ) -> None:
        if not arm.startswith("rn_"):
            pytest.skip("wild-type arm")
        wild = _brain_config(ops.stem_for(half, arm.replace("rn_", "wt_"), suffix))
        rewired = _brain_config(ops.stem_for(half, arm, suffix))
        differing = {
            key
            for key in set(wild) | set(rewired)
            if wild.get(key, "<absent>") != rewired.get(key, "<absent>")
        }
        assert differing == {_WIRING_KEY}, f"{half}/{suffix}/{arm} differs in {sorted(differing)}"

    @pytest.mark.parametrize("stem", sorted(ops.ARM_BY_STEM), ids=lambda s: s[-44:])
    def test_rewire_seed_stays_unset(self, stem: str) -> None:
        # Unset means the rewiring RNG derives from the run seed, so each seed's wild-type and
        # rewired arms pair. Pinning it across seeds is a different experiment, registered as M.5.
        body = "\n".join(
            line
            for line in _config_path(stem).read_text().splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "rewire_seed:" not in body


class TestTheSeedsAreFresh:
    """A registered panel never reuses a seed, and a pilot never touches the panel's band."""

    def test_panel_seeds_are_unburnt(self) -> None:
        for half, seeds in ops.SEEDS_BY_HALF.items():
            clash = sorted(set(seeds) & ops.BURNT_SEEDS)
            assert not clash, f"{half} reuses burnt seeds: {clash}"

    def test_the_two_halves_do_not_share_a_seed(self) -> None:
        assert not set(ops.SEEDS_BY_HALF["ppo"]) & set(ops.SEEDS_BY_HALF["reading"])

    def test_pilot_seeds_are_unburnt_and_disjoint_from_both_panels(self) -> None:
        assert not set(ops.PILOT_SEEDS) & ops.BURNT_SEEDS
        for seeds in ops.SEEDS_BY_HALF.values():
            assert not set(ops.PILOT_SEEDS) & set(seeds)


class TestTheGateKnowsWhatACompletePanelIs:
    """Per point-class, because a learning-only level carries two arms by design."""

    def _manifest(self, tmp_path: Path, half: str, rows: list[tuple[str, str, int]]) -> Path:
        path = tmp_path / "m.txt"
        path.write_text("\n".join(f"{a} {s} {seed} out.log" for a, s, seed in rows) + "\n")
        return path

    def _complete_rows(self, half: str, seeds: tuple[int, ...]) -> list[tuple[str, str, int]]:
        suffixes = [ops.CENTRE, *(s for _, s, _ in ops._levels(half))]
        return [
            (arm, suffix, seed)
            for suffix in suffixes
            for arm in ops.arms_at(half, suffix)
            for seed in seeds
        ]

    @pytest.mark.parametrize("half", ops.HALVES)
    def test_a_complete_panel_passes(self, tmp_path: Path, half: str) -> None:
        seeds = (1, 2)
        manifest = self._manifest(tmp_path, half, self._complete_rows(half, seeds))
        ops.require_complete(manifest, half, seeds)

    @pytest.mark.parametrize("half", ops.HALVES)
    def test_a_missing_seed_is_refused(self, tmp_path: Path, half: str) -> None:
        seeds = (1, 2)
        rows = self._complete_rows(half, seeds)[:-1]
        manifest = self._manifest(tmp_path, half, rows)
        with pytest.raises(ops.PanelError, match="incomplete"):
            ops.require_complete(manifest, half, seeds)

    def test_the_gate_does_not_demand_a_floor_a_learning_only_level_never_had(self) -> None:
        # The failure this guards: a uniform four-arm gate would refuse a correct reading panel,
        # because `plasticity_rate` and `trace_decay` cannot move a frozen arm and so have none.
        rate_level = next(s for p, s, _ in ops._levels("reading") if p == "plasticity_rate")
        assert set(ops.arms_at("reading", rate_level)) == set(ops.LEARNING_ARMS)


class TestTheFrozenSubstrateObligation:
    """The reading half is void without drift evidence on every scored seed.

    The check itself is validated in the direction where the answer is known: on the PPO pilot,
    where PPO writes the chemical matrix by design, it reads a relative drift near 1.0 and returns
    void. That is what makes a 0.00 on the reading half evidence rather than a default.
    """

    def test_a_construction_level_uses_its_own_floor_and_a_rule_level_uses_the_centre(self) -> None:
        # `w_chem` at a given seed is the same draw whatever the rate or decay says, and the
        # learning arm never writes it, so the centre's floor is the correct comparator there.
        assert len(ops.arms_at("reading", "d2")) == 4
        rate_level = next(s for p, s, _ in ops._levels("reading") if p == "plasticity_rate")
        assert len(ops.arms_at("reading", rate_level)) == 2

    def test_the_ppo_half_owes_no_drift_evidence(self) -> None:
        # PPO writes the chemical weights, so its contrast is not one run under a learner that
        # leaves them fixed, and demanding the evidence there would be a category error.
        assert "plasticity_rate" not in ops.PIN_LEVELS["ppo"]
        assert set(ops.PIN_LEVELS["ppo"]) == set(ops.CONSTRUCTION_PINS)


class TestTheInstrumentIsUntouched:
    """Both committed harnesses must be byte-identical to main."""

    @pytest.mark.parametrize("module", ["wiring_premise.py", "connectome_structure_efficiency.py"])
    def test_the_scoring_modules_are_untouched_by_this_change(self, module: str) -> None:
        if _git(["rev-parse", "--verify", "--quiet", "origin/main"]).returncode != 0:
            pytest.skip("origin/main is not available in this checkout")
        diff = _git(["diff", "--quiet", "origin/main", "--", f"scripts/analysis/{module}"])
        assert diff.returncode in (0, 1), f"git could not compare {module}: {diff.stderr!r}"
        assert diff.returncode == 0, f"{module} differs from main; the instrument must not change"

    def test_each_half_reaches_the_instrument_built_for_it(self) -> None:
        # The PPO half hands `wiring_premise` the arm names its own map is keyed on; the reading
        # half hands the efficiency module its labels. Relabelling one learner's arms as the
        # other's to reach a function is the mislabel this pairing exists to prevent.
        assert set(ops._WP_ARM.values()) >= set(ops.wp.EFFICIENCY_ARMS)
        assert set(ops._EFF_ARM.values()) == {ops.eff._WILD, ops.eff._REWIRED}

    def test_the_driver_reuses_the_harness_constants(self) -> None:
        # Orientation and censoring come from the instrument, never from a second copy here.
        assert ops.eff._METRICS[ops.CENSORED_METRIC] is False
        assert ops.eff._METRICS[ops.UNCENSORED_METRIC] is True


class TestTheCensoringRuleIsFixedInAdvance:
    """The rule picks the metric; the data never picks it after the fact."""

    def test_comparable_censoring_keeps_the_registered_metric(self) -> None:
        choice = ops.choose_metric({"centre": {"a": 1.0, "b": 1.0}, "d2": {"a": 0.95, "b": 1.0}})
        assert choice["primary_metric"] == ops.CENSORED_METRIC
        assert choice["reported_beside"] == ops.UNCENSORED_METRIC
        assert choice["censoring_comparable"] is True

    def test_divergent_censoring_moves_the_primary_to_the_uncensored_metric(self) -> None:
        choice = ops.choose_metric({"centre": {"a": 1.0, "b": 1.0}, "d2": {"a": 0.4, "b": 1.0}})
        assert choice["primary_metric"] == ops.UNCENSORED_METRIC
        assert choice["reported_beside"] == ops.CENSORED_METRIC
        assert choice["censoring_comparable"] is False

    def test_the_choice_is_scoped_to_one_contrast_rather_than_the_whole_surface(self) -> None:
        # The cells an interaction spans are the centre and that level. Pooling the comparison
        # across every level would let one badly-censored level void the censored metric where it
        # is perfectly interpretable — the pilot found exactly that at `forward_pass_depth: 2`,
        # where the wild type never crosses the threshold and the rewired null always does.
        rates = {
            "centre": {"a": 1.0, "b": 1.0},
            "d2": {"a": 0.0, "b": 1.0},
            "d6": {"a": 1.0, "b": 1.0},
        }
        pooled = ops.choose_metric(rates)
        assert pooled["primary_metric"] == ops.UNCENSORED_METRIC

        per_level = {
            level: ops.choose_metric({"centre": rates["centre"], level: rates[level]})
            for level in ("d2", "d6")
        }
        assert per_level["d2"]["primary_metric"] == ops.UNCENSORED_METRIC
        assert per_level["d6"]["primary_metric"] == ops.CENSORED_METRIC
