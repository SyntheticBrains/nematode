"""The ``wormbench`` command: the worked example, the default version, and fetching the data.

Covers the benchmark-package spec's requirements "A worked example from band cohorts" (scenario
"The example grades with no download"), "The package's default version and its records" (scenario
"The wheel's default needs no repository record"), "A command-line grader with the project's
output" (scenario "Measured grading on v1") and "Reference data resolved when used, checked, and
never fetched implicitly".
"""

from __future__ import annotations

import hashlib
import json
import math
import shlex
from typing import TYPE_CHECKING, Any

import pytest

from wormbench import cli, data
from wormbench import grade as wg
from wormbench.measurement import file_sha256

if TYPE_CHECKING:
    from pathlib import Path

REL_TOL, ABS_TOL = 1e-9, 1e-12


def _close(expected: Any, found: Any, where: str = "") -> None:
    """Compare two gradings: floats within the tolerance, everything else exactly."""
    if isinstance(expected, float) and isinstance(found, float):
        assert math.isclose(expected, found, rel_tol=REL_TOL, abs_tol=ABS_TOL), where
    elif isinstance(expected, dict):
        assert isinstance(found, dict), where
        assert expected.keys() == found.keys(), where
        for key in expected:
            _close(expected[key], found[key], f"{where}/{key}")
    elif isinstance(expected, list):
        assert isinstance(found, list), where
        assert len(expected) == len(found), where
        for i, (e, f) in enumerate(zip(expected, found, strict=True)):
            _close(e, f, f"{where}[{i}]")
    else:
        assert expected == found, where


@pytest.fixture
def example(tmp_path: Path) -> Path:
    """Write the worked example with no data root, as an installed package would."""
    data.set_root(None)
    out = tmp_path / "example"
    assert cli.main(["example", "--out", str(out)]) == 0
    return out


def test_the_example_prints_a_command_a_shell_reads_back(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The printed grading command quotes its paths, so a path with a space survives a shell."""
    data.set_root(None)
    out = tmp_path / "with space"
    assert cli.main(["example", "--out", str(out)]) == 0
    line = capsys.readouterr().out.splitlines()[-1].strip()
    manifest, grade = str(out / "manifest.txt"), str(out / "grade.json")
    assert shlex.split(line) == ["wormbench", "grade", "--manifest", manifest, "--out", grade]


def test_a_verdict_naming_another_versions_bands_validates_nothing() -> None:
    """v4's verdict must name v4's bands file; v3's file and digest leave v4 unvalidated."""
    data.set_root(None)
    v3 = file_sha256(data.path("benchmark", "bands_v3.json"))
    v4 = file_sha256(data.path("benchmark", "bands_v4.json"))
    own = {"v4": {"bands": "bands_v4.json", "verdict": "VALID", "bands_sha256": v4}}
    other = {"v4": {"bands": "bands_v3.json", "verdict": "VALID", "bands_sha256": v3}}
    assert wg.v4_is_valid(own) is True
    assert wg.v4_is_valid(other) is False


def test_the_example_grades_as_recorded_with_no_download(example: Path, tmp_path: Path) -> None:
    """The example grades on v4 from the package alone and matches its stored grading."""
    out = tmp_path / "grade.json"
    assert cli.main(["grade", "--manifest", str(example / "manifest.txt"), "--out", str(out)]) == 0
    graded = json.loads(out.read_text())
    provenance = graded.pop("wormbench")
    graded.pop("manifest")
    expected = json.loads((data.package_directory() / "example" / "expected_v4.json").read_text())
    _close(expected, graded)
    assert set(graded["families"]) == {"small_lawn", "kinematics", "activity"}
    assert graded["D"] is None
    assert graded["D_fit"] is None
    bands = data.package_directory() / "benchmark" / "bands_v4.json"
    assert provenance == {
        "package": cli.__version__,
        "bands": "v4",
        "bands_sha256": hashlib.sha256(bands.read_bytes()).hexdigest(),
    }


def test_the_default_is_refused_when_its_record_is_not_valid(
    example: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With v4 not VALID in the record, no ``--bands`` is refused rather than graded elsewhere."""
    shipped = data.package_record
    monkeypatch.setattr(
        data,
        "package_record",
        lambda name: {} if name == "validity.json" else shipped(name),
    )
    with pytest.raises(SystemExit, match="pass --bands"):
        cli.main(
            ["grade", "--manifest", str(example / "manifest.txt"), "--out", str(tmp_path / "g")],
        )


def test_measured_grading_on_v1_is_refused(example: Path, tmp_path: Path) -> None:
    """v1's bands carry no measurement model, so ``--simulated --bands v1`` is refused."""
    with pytest.raises(ValueError, match="no measurement model"):
        cli.main(
            [
                "grade",
                "--manifest",
                str(example / "manifest.txt"),
                "--out",
                str(tmp_path / "g.json"),
                "--bands",
                "v1",
                "--simulated",
            ],
        )


def _fake_manifest(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, bytes],
    urls: dict[str, str | None],
) -> None:
    listed = {
        key: {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
            "in_package": False,
        }
        for key, content in files.items()
    }
    monkeypatch.setattr(data, "manifest", lambda: {"files": listed})
    shipped = data.package_record
    monkeypatch.setattr(
        data,
        "package_record",
        lambda name: {"record": None, "files": urls} if name == "urls.json" else shipped(name),
    )


def test_fetch_refuses_without_a_published_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Until the deposit exists, every address is null and fetching refuses, naming the files."""
    _fake_manifest(monkeypatch, {"benchmark/tracks.npz": b"x"}, {"benchmark/tracks.npz": None})
    with pytest.raises(SystemExit, match=r"benchmark/tracks\.npz"):
        cli.main(["data", "fetch", "--dir", str(tmp_path / "cache")])


def test_fetch_downloads_and_checks_each_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each file is fetched from its address, checked against the manifest, and kept."""
    source = tmp_path / "deposit" / "tracks.npz"
    source.parent.mkdir()
    source.write_bytes(b"the deposited bytes")
    _fake_manifest(
        monkeypatch,
        {"benchmark/tracks.npz": b"the deposited bytes"},
        {"benchmark/tracks.npz": source.as_uri()},
    )
    cache = tmp_path / "cache"
    assert cli.main(["data", "fetch", "--dir", str(cache)]) == 0
    assert (cache / "benchmark" / "tracks.npz").read_bytes() == b"the deposited bytes"


def test_fetch_refuses_a_download_that_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A download whose SHA-256 differs from the manifest's is discarded and refused."""
    source = tmp_path / "deposit" / "tracks.npz"
    source.parent.mkdir()
    source.write_bytes(b"other bytes")
    _fake_manifest(
        monkeypatch,
        {"benchmark/tracks.npz": b"the deposited bytes"},
        {"benchmark/tracks.npz": source.as_uri()},
    )
    cache = tmp_path / "cache"
    with pytest.raises(SystemExit, match="SHA-256"):
        cli.main(["data", "fetch", "--dir", str(cache)])
    assert not (cache / "benchmark" / "tracks.npz").exists()
