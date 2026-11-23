"""The project grades with the wormbench package: one copy, the same modules, current records.

Covers the benchmark-package spec's requirements "One grader for the project and for outsiders"
(scenario "The old path is the same module") and "The package's default version and its records"
(scenario "A stale copy or record is caught"), and "Reference data resolved when used" (scenario
"The project reads its vendored files").
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[5]
ALIASES = {
    "quantumnematode.benchmark.readouts": "wormbench.readouts",
    "quantumnematode.benchmark.grading": "wormbench.grading",
    "quantumnematode.benchmark.measurement": "wormbench.measurement",
    "quantumnematode.benchmark.chemotaxis": "wormbench.chemotaxis",
    "quantumnematode.benchmark.track_import": "wormbench.track_import",
    "quantumnematode.benchmark.references": "wormbench.references",
    "quantumnematode.validation.roaming_dwelling": "wormbench.classifier",
    "quantumnematode.report.behaviour": "wormbench.records",
    "quantumnematode.evaluation.odour_field": "wormbench.odour_field",
}


@pytest.mark.parametrize(("old", "new"), sorted(ALIASES.items()))
def test_each_old_path_is_the_moved_module_whichever_is_imported_first(old: str, new: str) -> None:
    """Imported first in a fresh interpreter, each old path is the very wormbench module."""
    code = (
        f"import importlib; old = importlib.import_module({old!r}); "
        f"new = importlib.import_module({new!r}); "
        "assert old is new, (old, new); "
        "assert all(getattr(old, k) is getattr(new, k) for k in vars(new))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603 - fixed argv


def test_the_project_reads_its_vendored_files() -> None:
    """With the project imported, every reference file resolves under the repository's data/."""
    import quantumnematode  # noqa: F401 - sets the data root
    from wormbench import data

    assert data.get_root() == (_REPO / "data").resolve()
    listed = data.manifest()["files"]
    for key in listed:
        namespace, name = key.split("/", 1)
        assert data.path(namespace, name) == (_REPO / "data" / namespace / name).resolve()


def test_the_shipped_records_and_copies_are_current() -> None:
    """The package's records, data copies and manifest match the committed sources."""
    script = _REPO / "scripts" / "analysis" / "wormbench_records.py"
    result = subprocess.run(  # noqa: S603 - fixed argv
        [sys.executable, str(script), "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _bg():
    sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
    import benchmark_grade as bg  # pyright: ignore[reportMissingImports]

    return bg


def test_the_command_grades_as_the_script_does(tmp_path: Path) -> None:
    """Covers "Same input, same grading": the command's file is the script's, plus provenance."""
    import quantumnematode  # noqa: F401 - sets the data root
    from wormbench import cli, data

    bg = _bg()
    example = tmp_path / "example"
    cli.main(["example", "--out", str(example)])
    lines = (example / "manifest.txt").read_text().splitlines()
    manifest = tmp_path / "absolute.txt"
    manifest.write_text(
        "".join(f"{f} {example / n}\n" for f, n in (line.split() for line in lines)),
    )
    script_out, command_out = tmp_path / "script.json", tmp_path / "command.json"
    assert (
        bg.main(["tracks", "--manifest", str(manifest), "--out", str(script_out), "--bands", "v4"])
        == 0
    )
    assert data.get_root() is not None
    grade = ["grade", "--manifest", str(manifest), "--out", str(command_out), "--bands", "v4"]
    assert cli.main(grade) == 0
    command = json.loads(command_out.read_text())
    assert command.pop("wormbench")["bands"] == "v4"
    assert json.dumps(command, indent=2, allow_nan=False) + "\n" == script_out.read_text()


def _simulated_animals(n: int, seconds: int) -> list[list[dict]]:
    """Return n animals' per-second steps on a lawn, with runs and reversals, as plain records."""
    import numpy as np

    rng = np.random.default_rng(20261124)
    animals = []
    for _ in range(n):
        heading = rng.uniform(0, 2 * np.pi)
        x = y = 0.0
        steps = []
        for t in range(seconds):
            kind = "reversal" if rng.random() < 0.04 else "run"
            speed = 0.03 if kind == "reversal" else float(rng.choice([0.01, 0.1]))
            heading += rng.normal(0, 0.3)
            x, y = x + speed * np.cos(heading), y + speed * np.sin(heading)
            steps.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "on_lawn": bool(rng.random() < 0.97),
                    "primitive": kind,
                    "t": t,
                },
            )
        animals.append(steps)
    return animals


def test_simulated_tracks_are_graded_as_the_project_grades_its_own(tmp_path: Path) -> None:
    """Covers "Simulated tracks are graded as the project grades its own"."""
    import dataclasses

    from wormbench import grade as wg
    from wormbench import measurement as mm
    from wormbench.records import BehaviourStep

    bg = _bg()
    seconds, animals = 2400, _simulated_animals(64, 2400)
    seeds = list(range(101, 101 + len(animals)))
    runs = []
    rows = ["animal,t_s,x,y,on_lawn,primitive,seed"]
    for i, (seed, steps) in enumerate(zip(seeds, animals, strict=True)):
        records = [
            BehaviourStep(
                step=s["t"],
                x=s["x"],
                y=s["y"],
                heading_rad=0.0,
                concentration=0.0,
                dc_dt=0.0,
                grad_dir=0.0,
                grad_strength=0.0,
                on_lawn=s["on_lawn"],
                primitive=s["primitive"],
            )
            for s in steps
        ]
        runs.append({"seed": seed, "steps": [dataclasses.asdict(r) for r in records]})
        rows += [
            f"w{i},{float(s['t'])!r},{s['x']!r},{s['y']!r},{int(s['on_lawn'])},{s['primitive']},{seed}"
            for s in steps
        ]
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps({"assay": {"analysis_steps": [0, seconds]}, "runs": runs}))
    captures = tmp_path / "captures.txt"
    captures.write_text(f"small_lawn {capture}\n")
    tracks = tmp_path / "simulated.csv"
    tracks.write_text("\n".join(rows) + "\n")
    sidecar = {
        "family": "small_lawn",
        "analysis_start_s": 0.0,
        "units": "mm",
        "lawn": {"radius": 50.0},
    }
    (tmp_path / "simulated.csv.json").write_text(json.dumps(sidecar))
    manifest = tmp_path / "simulated.txt"
    manifest.write_text(f"small_lawn {tracks}\n")

    bands = bg.load_bands_v4()
    model = mm.TrackerModel.from_dict(bands["measurement"]["lawn"])
    by_capture = bg.capture_populations(captures, model)
    by_import = wg.simulated_track_populations(manifest, model)

    # Lost seconds are NaN, which never equals itself, so the steps are compared as JSON.
    def as_json(populations: dict) -> list[list[str]]:
        return [
            [json.dumps(dataclasses.asdict(s)) for s in animal]
            for animal in populations["small_lawn"]
        ]

    assert as_json(by_import) == as_json(by_capture)
    graded_capture = bg._finite(bg.grade_populations_v4(by_capture, None, bands))
    graded_import = bg._finite(bg.grade_populations_v4(by_import, None, bands))
    assert json.dumps(graded_import) == json.dumps(graded_capture)


def test_the_example_reads_band_cohorts_only() -> None:
    """Covers "No held-out cohort is read": the example's animals are all band-cohort rows."""
    sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
    import wormbench_example  # pyright: ignore[reportMissingImports]
    from wormbench import references as refs

    tracks, rows = wormbench_example.selected_rows()
    assert len(rows) == wormbench_example.ANIMALS
    assert set(tracks.side[rows]) == {refs.BANDS}
    assert set(tracks.arm(wormbench_example.ARM)) >= set(rows)
