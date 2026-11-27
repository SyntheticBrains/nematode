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
    misnamed = {"v4": {"bands": "bands_v3.json", "verdict": "VALID", "bands_sha256": v4}}
    assert wg.v4_is_valid(own) is True
    assert wg.v4_is_valid(other) is False
    assert wg.v4_is_valid(misnamed) is False


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


class _Response:
    """A stand-in HTTP response: a status, a declared length, and a body that may stop short."""

    def __init__(self, body: bytes, status: int, declared: int) -> None:
        self._body, self.status = body, status
        self.headers = {"Content-Length": str(declared)}

    def __enter__(self) -> _Response:  # noqa: PYI034 - Python 3.10 has no typing.Self
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self, amount: int = -1) -> bytes:
        out, self._body = self._body[:amount], self._body[amount:]
        return out


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    replies: list[Any],
) -> list[str | None]:
    """Answer each request with the next reply, and record each request's Range header."""
    ranges: list[str | None] = []

    def urlopen(request: Any, timeout: float) -> _Response:
        ranges.append(request.get_header("Range"))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(cli.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
    return ranges


BODY = bytes(range(256)) * 40  # 10,240 bytes


def test_fetch_resumes_a_transfer_the_server_cuts_short(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transfer closed before its declared length is resumed from the byte it reached."""
    _fake_manifest(
        monkeypatch,
        {"benchmark/tracks.npz": BODY},
        {"benchmark/tracks.npz": "https://x"},
    )
    ranges = _serve(
        monkeypatch,
        [
            _Response(BODY[:6000], 200, len(BODY)),
            _Response(BODY[6000:9000], 206, len(BODY) - 6000),
            _Response(BODY[9000:], 206, len(BODY) - 9000),
        ],
    )
    cache = tmp_path / "cache"
    assert cli.main(["data", "fetch", "--dir", str(cache)]) == 0
    assert (cache / "benchmark" / "tracks.npz").read_bytes() == BODY
    assert ranges == [None, "bytes=6000-", "bytes=9000-"]


def test_fetch_starts_again_when_the_server_ignores_the_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that answers a range request with the whole file is read from the start."""
    _fake_manifest(
        monkeypatch,
        {"benchmark/tracks.npz": BODY},
        {"benchmark/tracks.npz": "https://x"},
    )
    _serve(
        monkeypatch,
        [_Response(BODY[:6000], 200, len(BODY)), _Response(BODY, 200, len(BODY))],
    )
    cache = tmp_path / "cache"
    assert cli.main(["data", "fetch", "--dir", str(cache)]) == 0
    assert (cache / "benchmark" / "tracks.npz").read_bytes() == BODY


def test_fetch_gives_up_on_a_transfer_that_never_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After its attempts, a download that never completes is refused and nothing is kept."""
    _fake_manifest(
        monkeypatch,
        {"benchmark/tracks.npz": BODY},
        {"benchmark/tracks.npz": "https://x"},
    )
    _serve(monkeypatch, [ConnectionResetError("reset")] * cli.DOWNLOAD_ATTEMPTS)
    cache = tmp_path / "cache"
    with pytest.raises(SystemExit, match=r"stopped at 0 of 10,240 bytes after 8 attempts"):
        cli.main(["data", "fetch", "--dir", str(cache)])
    assert not list((cache / "benchmark").iterdir())


def test_fetch_stops_at_once_on_a_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing file is not retried: the refusal names the HTTP status."""
    _fake_manifest(
        monkeypatch,
        {"benchmark/tracks.npz": BODY},
        {"benchmark/tracks.npz": "https://x"},
    )
    missing = cli.urllib.error.HTTPError("https://x", 404, "NOT FOUND", None, None)  # type: ignore[arg-type]
    ranges = _serve(monkeypatch, [missing])
    with pytest.raises(SystemExit, match=r"HTTP 404"):
        cli.main(["data", "fetch", "--dir", str(tmp_path / "cache")])
    assert len(ranges) == 1


def test_the_shipped_addresses_name_every_fetched_file_in_one_record() -> None:
    """Every fetched file has an address, pinned to the one versioned record the package names."""
    urls = data.package_record(cli.URLS_NAME)
    record = urls["record"]
    assert record.startswith("10.5281/zenodo.")
    recid = record.rsplit(".", 1)[1]
    fetched = sorted(k for k, v in data.manifest()["files"].items() if not v["in_package"])
    assert sorted(urls["files"]) == fetched
    for key, url in urls["files"].items():
        name = key.split("/", 1)[1]
        assert url == f"https://zenodo.org/records/{recid}/files/{name}?download=1"
