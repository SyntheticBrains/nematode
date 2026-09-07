"""Behavioural cloning: self-cloning recovers a scrambled teacher; each set stays in its lane."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.rollouts import RolloutRecorder
from quantumnematode.brain.weights import load_weights

_REPO_ROOT = Path(__file__).resolve().parents[4].parent
_scripts = _REPO_ROOT / "scripts" / "campaigns"
if not _scripts.is_dir():
    msg = f"could not locate scripts/campaigns from {Path(__file__).resolve()}"
    raise RuntimeError(msg)
sys.path.insert(0, str(_scripts))

import l4_behavioural_clone as bc  # noqa: E402  # pyright: ignore[reportMissingImports]

_CONFIG = (
    _REPO_ROOT
    / "configs"
    / "scenarios"
    / "foraging"
    / "connectomeppo_small_continuous2d_klinotaxis.yml"
)
_SEED = 7
_EPISODES = 6
_STEPS = 60


def _observation(rng: np.random.Generator) -> BrainParams:
    return BrainParams(
        food_concentration=float(rng.uniform(0.0, 1.0)),
        food_lateral_gradient=float(rng.uniform(-1.0, 1.0)),
        food_dconcentration_dt=float(rng.uniform(-0.2, 0.2)),
    )


@pytest.fixture(scope="module")
def rollouts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Record a teacher that is the student's own brain with its chemical weights scrambled."""
    teacher = bc.build_student(_CONFIG, _SEED)
    gen = torch.Generator().manual_seed(_SEED)
    with torch.no_grad():
        w = teacher.topology.w_chem
        signs = torch.where(torch.rand(w.shape, generator=gen) < 0.5, -1.0, 1.0)
        w.mul_(signs * 3.0)
    path = tmp_path_factory.mktemp("rollouts") / "teacher.jsonl"
    recorder = RolloutRecorder(path)
    rng = np.random.default_rng(_SEED)
    for _ in range(_EPISODES):
        teacher.prepare_episode()
        for _ in range(_STEPS):
            params = _observation(rng)
            action = teacher.run_brain(
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )[0]
            recorder.record(params, action)
        recorder.end_episode()
    recorder.close()
    return path


def _snapshot(brain: bc.ConnectomePPOBrain) -> dict[str, torch.Tensor]:
    t = brain.topology
    return {
        "w_chem": t.w_chem.detach().clone(),
        "food_gains": t.food_gains.detach().clone(),
        "readout": t.readout.detach().clone(),
        "log_std": t.log_std.detach().clone(),
    }


class TestBuildAndDataset:
    def test_student_is_the_arm_at_its_seed(self) -> None:
        a = bc.build_student(_CONFIG, _SEED)
        b = bc.build_student(_CONFIG, _SEED)
        assert torch.equal(a.topology.w_chem, b.topology.w_chem)
        assert a.seed == _SEED

    def test_dataset_shapes_and_split(self, rollouts: Path) -> None:
        brain = bc.build_student(_CONFIG, _SEED)
        rows = bc.read_rollouts(rollouts)
        states, targets, episodes = bc.load_dataset(rows, brain)
        assert states.shape[0] == _EPISODES * _STEPS
        assert targets.shape == (_EPISODES * _STEPS, 2)
        train, held = bc.split_episodes(episodes, 0.34, np.random.default_rng(1))
        assert len(train) + len(held) == len(episodes)
        assert len(set(episodes[held])) == 2
        assert not set(episodes[held]) & set(episodes[train])

    def test_holdout_never_takes_every_episode(self) -> None:
        episodes = np.array([0, 0, 1, 1])
        train, held = bc.split_episodes(episodes, 0.9, np.random.default_rng(0))
        assert len(train) == 2
        assert len(held) == 2
        with pytest.raises(ValueError, match="holdout must be in"):
            bc.split_episodes(episodes, 1.0, np.random.default_rng(0))

    def test_main_rejects_bad_batch_size_and_holdout(self, rollouts: Path, tmp_path: Path) -> None:
        base = [
            "--config",
            str(_CONFIG),
            "--seed",
            "1",
            "--rollouts",
            str(rollouts),
            "--parameter-set",
            "plastic",
            "--out",
            str(tmp_path / "x.pt"),
        ]
        with pytest.raises(SystemExit):
            bc.main([*base, "--batch-size", "0"])
        with pytest.raises(SystemExit):
            bc.main([*base, "--holdout", "1.0"])

    def test_rows_without_a_mean_are_refused(self, tmp_path: Path) -> None:
        brain = bc.build_student(_CONFIG, _SEED)
        with pytest.raises(ValueError, match="no action mean"):
            bc.load_dataset(
                [
                    {
                        "episode": 0,
                        "step": 0,
                        "params": {},
                        "action": [0.1, 0.2],
                        "action_mean": None,
                    },
                ],
                brain,
            )


class TestSelfCloning:
    @pytest.mark.parametrize("parameter_set", ["plastic", "full"])
    def test_held_out_loss_falls_an_order_of_magnitude(
        self,
        rollouts: Path,
        tmp_path: Path,
        parameter_set: str,
    ) -> None:
        out = tmp_path / f"{parameter_set}.pt"
        code = bc.main(
            [
                "--config",
                str(_CONFIG),
                "--seed",
                str(_SEED),
                "--rollouts",
                str(rollouts),
                "--parameter-set",
                parameter_set,
                "--out",
                str(out),
                "--epochs",
                "150",
                "--lr",
                "0.01",
                "--batch-size",
                "120",
                "--holdout",
                "0.34",
            ],
        )
        assert code == 0
        record = json.loads(out.with_name(f"{parameter_set}.clone.json").read_text())
        assert record["held_out_loss"] < record["initial_held_out_loss"] / 10
        assert record["final_loss"] < record["initial_loss"] / 10
        assert record["n_held_out"] == 2 * _STEPS
        assert record["rollouts_sha256"]
        # The saved file loads into a fresh student and carries the fit: on the first five
        # recorded observations the loaded means sit far closer to the teacher's than a
        # fresh student's do, and differ from the fresh student's element-wise.
        fresh = bc.build_student(_CONFIG, _SEED)
        loaded = bc.build_student(_CONFIG, _SEED)
        load_weights(loaded, out)
        rows = bc.read_rollouts(rollouts)
        states, targets, _eps = bc.load_dataset(rows[:5], loaded)
        with torch.no_grad():
            reproduced = bc.student_mean(loaded, states)
            untrained = bc.student_mean(fresh, states)
        assert reproduced.shape == (5, 2)
        assert not torch.allclose(reproduced, untrained)
        assert torch.mean((reproduced - targets) ** 2) < torch.mean((untrained - targets) ** 2) / 4

    def test_plastic_set_touches_only_the_chemical_weights(self, rollouts: Path) -> None:
        brain = bc.build_student(_CONFIG, _SEED)
        before = _snapshot(brain)
        rows = bc.read_rollouts(rollouts)
        states, targets, episodes = bc.load_dataset(rows, brain)
        train, held = bc.split_episodes(episodes, 0.0, np.random.default_rng(_SEED))
        bc.clone(
            brain,
            states,
            targets,
            train,
            held,
            parameter_set_name="plastic",
            epochs=3,
            lr=0.01,
            batch_size=120,
            rng=np.random.default_rng(_SEED),
        )
        after = _snapshot(brain)
        assert not torch.equal(before["w_chem"], after["w_chem"])
        for key in ("food_gains", "readout", "log_std"):
            assert torch.equal(before[key], after[key]), key
        changed = before["w_chem"] != after["w_chem"]
        assert bool(torch.all(brain.topology.m_chem[changed]))

    def test_full_set_trains_what_ppo_trains(self, rollouts: Path) -> None:
        brain = bc.build_student(_CONFIG, _SEED)
        before = _snapshot(brain)
        rows = bc.read_rollouts(rollouts)
        states, targets, episodes = bc.load_dataset(rows, brain)
        train, held = bc.split_episodes(episodes, 0.0, np.random.default_rng(_SEED))
        bc.clone(
            brain,
            states,
            targets,
            train,
            held,
            parameter_set_name="full",
            epochs=3,
            lr=0.01,
            batch_size=120,
            rng=np.random.default_rng(_SEED),
        )
        after = _snapshot(brain)
        for key in ("w_chem", "food_gains", "readout"):
            assert not torch.equal(before[key], after[key]), key
        # A mean-only record gives the noise parameter no gradient; it is left out of the set.
        assert torch.equal(before["log_std"], after["log_std"])
        assert all(p is not brain.topology.log_std for p in bc.parameter_set(brain, "full"))

    def test_no_improvement_writes_nothing(self, rollouts: Path, tmp_path: Path) -> None:
        out = tmp_path / "none.pt"
        code = bc.main(
            [
                "--config",
                str(_CONFIG),
                "--seed",
                str(_SEED),
                "--rollouts",
                str(rollouts),
                "--parameter-set",
                "plastic",
                "--out",
                str(out),
                "--epochs",
                "0",
            ],
        )
        assert code == 1
        assert not out.exists()
        assert not list(tmp_path.glob("*.json"))
