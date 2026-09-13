"""The readout-substitution arms, and the two things that make them interpretable.

Three failures these pin:

* a prepared checkpoint that moves more than the readout — making the arm a clone assay, or silently
  changing the action noise, or moving the homeostatic norm targets the rule re-anchors from the
  weights it loads;
* a `rotated` control that is not the same magnitude as the `ppo` readout it is meant to isolate the
  direction of, leaving "this readout helps" inseparable from "the default was bad";
* the load path itself changing something, which would mean R.1c's committed anatomical pair cannot
  serve as the comparator and has to be re-run — the branch this test decides.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.weights import load_weights
from quantumnematode.utils.config_loader import load_simulation_config

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts").is_dir():
    _root = _root.parent
_SCRIPT = _root / "scripts" / "prepare_readout_checkpoints.py"
_CONFIG = (
    _root
    / "configs"
    / "scenarios"
    / "foraging"
    / "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350_nodepert_motor.yml"
)

_spec = importlib.util.spec_from_file_location("prepare_readout_checkpoints", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
prep = importlib.util.module_from_spec(_spec)
sys.modules["prepare_readout_checkpoints"] = prep
_spec.loader.exec_module(prep)

_SEED = 1


def _brain() -> ConnectomePPOBrain:
    simulation = load_simulation_config(str(_CONFIG))
    assert simulation.brain is not None
    config = simulation.brain.config
    assert isinstance(config, ConnectomePPOBrainConfig)
    config.seed = _SEED
    return ConnectomePPOBrain(config=config, device=DeviceType.CPU)


@pytest.fixture(scope="module")
def anatomical(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a prepared checkpoint whose readout is the arm's own: the load-path control."""
    out = tmp_path_factory.mktemp("readouts")
    return prep.prepare(_CONFIG, "anatomical", _SEED, out, None)


class TestTheLoadPathIsInert:
    """What licenses reusing R.1c's committed anatomical pair instead of re-running it."""

    def test_loading_an_anatomical_checkpoint_changes_no_tensor(self, anatomical: Path) -> None:
        # A load also calls reset_state() and buffer.reset(). This establishes they are inert
        # when the loaded tensors match construction, rather than reasoning that they are.
        unloaded, loaded = _brain(), _brain()
        load_weights(loaded, anatomical)
        before, after = unloaded.topology.state_dict(), loaded.topology.state_dict()
        assert set(before) == set(after)
        differing = [k for k in before if not torch.equal(before[k], after[k])]
        assert differing == [], (
            "a load must leave every tensor identical when the checkpoint matches construction; "
            "otherwise R.1c's anatomical arms cannot be the comparator and must be re-run"
        )

    def test_the_homeostatic_targets_survive_the_load(self, anatomical: Path) -> None:
        # reset_state() re-anchors them from the loaded weights. Identical weights must give
        # identical targets, or the loaded arm would be pulled toward different norms.
        from quantumnematode.learning_rules.three_factor import ThreeFactorRule

        unloaded, loaded = _brain(), _brain()
        load_weights(loaded, anatomical)
        assert isinstance(unloaded._rule, ThreeFactorRule)
        assert isinstance(loaded._rule, ThreeFactorRule)
        pairs = zip(unloaded._rule.norm_targets, loaded._rule.norm_targets, strict=True)
        assert all(torch.equal(a, b) for a, b in pairs)


class TestOnlyTheReadoutMoves:
    def test_the_protected_tensors_come_through_untouched(self, anatomical: Path) -> None:
        checkpoint = torch.load(anatomical, weights_only=True)
        fresh = _brain().topology.state_dict()
        for key in prep.PROTECTED:
            if key in fresh:
                assert torch.equal(checkpoint["topology"][key], fresh[key]), key

    def test_a_substituted_readout_is_the_only_difference(self, tmp_path: Path) -> None:
        # Built by hand rather than from a harvest, so the assertion is about the substitution
        # and not about a PPO run being available.
        fresh = _brain()
        target = prep.prepare(_CONFIG, "anatomical", _SEED, tmp_path, None)
        checkpoint = torch.load(target, weights_only=True)
        checkpoint["topology"][prep.READOUT_KEY] = torch.zeros_like(
            checkpoint["topology"][prep.READOUT_KEY],
        )
        torch.save(checkpoint, target)
        loaded = _brain()
        load_weights(loaded, target)
        before, after = fresh.topology.state_dict(), loaded.topology.state_dict()
        differing = [k for k in before if not torch.equal(before[k], after[k])]
        assert differing == [prep.READOUT_KEY]

    def test_a_wrong_shaped_readout_is_refused(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A harvest from a differently-shaped substrate must fail, not be broadcast into place.
        monkeypatch.setattr(prep, "_harvest_readout", lambda *_a, **_k: torch.zeros((3, 7)))
        with pytest.raises(ValueError, match="readout shape"):
            prep.prepare(_CONFIG, "ppo", _SEED, tmp_path, tmp_path)

    def test_ppo_and_rotated_need_a_harvest(self) -> None:
        for source in ("ppo", "rotated"):
            with pytest.raises(ValueError, match="needs --harvest-dir"):
                prep._replacement_readout(source, torch.zeros((2, 4)), _SEED, None)


class TestTheRotatedControl:
    def test_it_matches_the_reference_norm(self) -> None:
        # Same magnitude so the arm isolates DIRECTION: the readout's scale interacts with the
        # action distribution, and changing both would confound the comparison.
        reference = torch.tensor([[1.0, -2.0, 3.0, 0.5], [0.25, 1.5, -1.0, 2.0]])
        rotated = prep._rotate(reference, _SEED)
        assert float(rotated.norm()) == pytest.approx(float(reference.norm()), rel=1e-6)

    def test_it_points_somewhere_else(self) -> None:
        reference = torch.tensor([[1.0, -2.0, 3.0, 0.5], [0.25, 1.5, -1.0, 2.0]])
        rotated = prep._rotate(reference, _SEED)
        cosine = float(
            torch.nn.functional.cosine_similarity(
                rotated.reshape(-1),
                reference.reshape(-1),
                dim=0,
            ),
        )
        assert abs(cosine) < 0.9

    def test_it_is_deterministic_at_a_seed(self) -> None:
        reference = torch.randn((2, 4))
        assert torch.equal(prep._rotate(reference, 7), prep._rotate(reference, 7))

    def test_a_different_seed_gives_a_different_direction(self) -> None:
        reference = torch.randn((2, 4))
        assert not torch.equal(prep._rotate(reference, 7), prep._rotate(reference, 8))


class TestAMissingHarvestFails:
    def test_no_log_for_a_seed_is_an_error(self, tmp_path: Path) -> None:
        # Silence here would produce an unmodified checkpoint reported as a substituted arm.
        (tmp_path / "logs").mkdir()
        with pytest.raises(FileNotFoundError, match="no harvest log"):
            prep._harvest_readout(tmp_path, _SEED)

    def test_two_logs_for_a_seed_is_an_error(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        for name in ("a-seed1.log", "b-seed1.log"):
            (logs / name).write_text("")
        with pytest.raises(ValueError, match="harvest logs for seed"):
            prep._harvest_readout(tmp_path, _SEED)
