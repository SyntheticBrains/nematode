"""Warm-start campaign steps: teacher selection, the recording config, clone naming and flags."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from quantumnematode.brain.weights import load_weights, resolve_weights_path

_REPO_ROOT = Path(__file__).resolve().parents[4].parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "campaigns"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "analysis"))

import l4_behavioural_clone as bc  # noqa: E402  # pyright: ignore[reportMissingImports]
import l4_warm_start_campaign as ws  # noqa: E402  # pyright: ignore[reportMissingImports]

_TEACHER = (
    _REPO_ROOT / "configs" / "scenarios" / "foraging_predator_thermal" / f"{ws.TEACHER_STEM}.yml"
)


def _log(path: Path, statuses: list[str], experiment_id: str) -> None:
    lines = [f"  Experiment ID: {experiment_id}"] + [
        f"Run: {i}   Status: {s:<7} Reason: x Steps: 10    "
        f"Eaten: {10 if s == 'SUCCESS' else 1}/10  "
        for i, s in enumerate(statuses, 1)
    ]
    path.write_text("\n".join(lines))


class TestTeacher:
    def test_selection_by_plateau_tail_with_weights(self, tmp_path: Path) -> None:
        logs = tmp_path / "campaign" / "logs"
        logs.mkdir(parents=True)
        experiments = tmp_path / "experiments"
        for seed, rate in ((1, 0.5), (2, 0.9), (3, 0.9), (4, 1.0)):
            eid = f"exp{seed}"
            _log(
                logs / f"{ws.TEACHER_STEM}-seed{seed}.log",
                ["SUCCESS" if i / 40 < rate else "FAILED" for i in range(40)],
                eid,
            )
            folder = experiments / eid
            folder.mkdir(parents=True)
            exports = f"exports/{eid}"
            if seed != 4:  # seed 4 has the best tail but no saved weights
                (tmp_path / exports / "weights").mkdir(parents=True)
                (tmp_path / exports / "weights" / "final.pt").write_bytes(b"x")
            (folder / f"{eid}.json").write_text(json.dumps({"exports_path": exports}))
        records = ws.scan_teacher_campaign(tmp_path / "campaign", experiments)
        assert [r["seed"] for r in records] == [1, 2, 3, 4]
        # weights are resolved against the repository root; make them resolvable for the test
        for r in records:
            r["weights"] = (
                str(tmp_path / r["weights"].split("nematode/", 1)[-1]) if r["weights"] else None
            )
            if r["seed"] == 4:
                r["weights"] = None
        chosen = ws.select_teacher(records)
        assert chosen["seed"] == 2  # ties at 0.9 go to the lower seed; seed 4 lacks weights
        with pytest.raises(ValueError, match="no teacher run"):
            ws.select_teacher([{"seed": 1, "plateau_tail": 1.0, "weights": None}])

    def test_recording_config_is_two_keys_off_the_teacher(self) -> None:
        derived = ws.derive_recording_config(_TEACHER.read_text(), "campaigns/x/teacher.pt")
        assert "freeze_updates: true" in derived
        assert "weights_path: campaigns/x/teacher.pt" in derived
        assert derived.startswith("# Frozen teacher")
        body_parent = "".join(
            line
            for line in _TEACHER.read_text().splitlines(keepends=True)
            if not line.startswith("#")
        )
        body_derived = "".join(
            line for line in derived.splitlines(keepends=True) if not line.startswith("#")
        )
        assert (
            body_derived.replace(
                "    freeze_updates: true\n    weights_path: campaigns/x/teacher.pt\n",
                "",
            )
            == body_parent
        )


class TestClones:
    def test_clone_file_names_match_the_arm_placeholders(self) -> None:
        assert ws.clone_file("plastic", "wt", 3) == "plastic_wt_seed3.pt"
        configs = _REPO_ROOT / "configs" / "scenarios" / "foraging_predator_thermal"
        for parameter_set, wiring in ws.STUDENT_CONFIGS:
            template = f"campaigns/l4-warm-start/clones/{parameter_set}_{wiring}_seed{{seed}}.pt"
            assert any(template in p.read_text() for p in configs.glob("*clone*.yml")), template
            assert resolve_weights_path(template, 7).endswith(
                ws.clone_file(parameter_set, wiring, 7),
            )

    def test_weak_flag(self) -> None:
        assert ws.flag_clone({"initial_held_out_loss": 1.0, "held_out_loss": 0.4}) is False
        assert ws.flag_clone({"initial_held_out_loss": 1.0, "held_out_loss": 0.6}) is True
        assert ws.flag_clone({"initial_held_out_loss": float("nan"), "held_out_loss": 0.1}) is True
        assert ws.flag_clone({}) is True

    def test_rewired_clone_pairs_by_seed(self, tmp_path: Path) -> None:
        """A clone made at seed S loads into the rewired arm at seed S and is refused at S+1."""
        stem = ws.STUDENT_CONFIGS[("plastic", "rn")]
        config = _REPO_ROOT / "configs" / "scenarios" / "foraging_predator_thermal" / f"{stem}.yml"
        student = bc.build_student(config, 5)
        file = tmp_path / "rn_seed5.pt"
        bc.save_weights(student, file, components=bc.SAVED_COMPONENTS)
        same = bc.build_student(config, 5)
        load_weights(same, file)
        assert torch.equal(same.topology.w_chem, student.topology.w_chem)
        other = bc.build_student(config, 6)
        with pytest.raises(ValueError, match="different wiring"):
            load_weights(other, file)
