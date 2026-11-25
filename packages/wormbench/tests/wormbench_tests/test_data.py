"""The data resolver: its order, its fall-through, and refusing what no source holds.

Covers the benchmark-package spec's requirement "Reference data resolved when used, checked, and
never fetched implicitly" (scenario "Grading at an unvendored size without the tracks").
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from wormbench import data

if TYPE_CHECKING:
    from pathlib import Path


def _make(root: Path, namespace: str, name: str, text: str) -> Path:
    target = root / namespace / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


def test_host_root_outranks_the_environment_variable(tmp_path, monkeypatch):
    """The host's root is read before WORMBENCH_DATA, so the project reads its own files."""
    host = tmp_path / "host"
    env = tmp_path / "env"
    _make(host, "benchmark", "x.json", "host")
    _make(env, "benchmark", "x.json", "env")
    monkeypatch.setenv(data.ENVIRONMENT_VARIABLE, str(env))
    data.set_root(host)
    assert data.path("benchmark", "x.json").read_text() == "host"


def test_a_file_absent_from_the_host_root_falls_through(tmp_path, monkeypatch):
    """A file missing from one source is resolved from the next."""
    host = tmp_path / "host"
    env = tmp_path / "env"
    (host / "benchmark").mkdir(parents=True)
    _make(env, "benchmark", "x.json", "env")
    monkeypatch.setenv(data.ENVIRONMENT_VARIABLE, str(env))
    data.set_root(host)
    assert data.path("benchmark", "x.json").read_text() == "env"


def test_the_fetched_cache_is_the_last_source(tmp_path, monkeypatch):
    """A file held only by the fetched cache resolves there."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    data.set_root(None)
    cached = _make(data.cache_directory(), "benchmark", "only-cached.json", "cached")
    assert data.path("benchmark", "only-cached.json") == cached


def test_a_file_no_source_holds_names_itself_and_the_fetch_command(tmp_path, monkeypatch):
    """A missing file fails with its name and the fetch command, never a download."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    data.set_root(tmp_path / "empty")
    with pytest.raises(data.MissingDataError) as raised:
        data.path("benchmark", "not-a-file.npz")
    assert "benchmark/not-a-file.npz" in str(raised.value)
    assert data.FETCH_COMMAND in str(raised.value)


def test_an_unknown_namespace_is_refused():
    """Only the declared namespaces resolve."""
    with pytest.raises(ValueError, match="unknown data namespace"):
        data.path("elsewhere", "x.json")


def test_importing_the_package_imports_no_host_module():
    """The package imports nothing from the project that hosts it."""
    code = (
        "import sys, wormbench, wormbench.grading, wormbench.measurement, wormbench.track_import, "
        "wormbench.odour_field; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'quantumnematode'))"
    )
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert out == "[]"


def test_no_module_holds_a_data_path_after_import(tmp_path):
    """No module computes a path when imported: none holds a Path among its module-level names."""
    code = (
        "import pathlib, wormbench; from wormbench import data; data.set_root(None); "
        "import wormbench.references, wormbench.classifier, wormbench.measurement, "
        "wormbench.grading, wormbench.track_import, wormbench.odour_field, wormbench.chemotaxis, "
        "wormbench.readouts, wormbench.records; "
        "mods = [m for n, m in __import__('sys').modules.items() if n.startswith('wormbench.')]; "
        "print(sorted(f'{m.__name__}.{k}' for m in mods for k, v in vars(m).items() "
        "if isinstance(v, pathlib.PurePath)))"
    )
    env = {"XDG_CACHE_HOME": str(tmp_path / "cache"), "PATH": "/usr/bin:/bin"}
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()
    assert out == "[]"


def test_a_tampered_copy_is_refused(tmp_path, monkeypatch):
    """A listed file from WORMBENCH_DATA whose SHA-256 differs from the manifest's is refused."""
    env = tmp_path / "env"
    _make(env, "benchmark", "bands_v4.json", '{"tampered": true}\n')
    monkeypatch.setenv(data.ENVIRONMENT_VARIABLE, str(env))
    data.set_root(None)
    with pytest.raises(data.ChecksumError, match=r"benchmark/bands_v4\.json"):
        data.path("benchmark", "bands_v4.json")


def test_a_refused_file_names_the_repair_for_its_source(tmp_path, monkeypatch):
    """The refusal says how to repair the file where it was found, source by source."""
    env = tmp_path / "env"
    _make(env, "benchmark", "bands_v4.json", '{"tampered": true}\n')
    monkeypatch.setenv(data.ENVIRONMENT_VARIABLE, str(env))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    data.set_root(None)
    with pytest.raises(data.ChecksumError, match=rf"unset {data.ENVIRONMENT_VARIABLE}"):
        data.path("benchmark", "bands_v4.json")
    _make(data.cache_directory(), "benchmark", "scheer2023_split.json", "{}\n")
    with pytest.raises(data.ChecksumError, match=r"run `wormbench data fetch`, which replaces it"):
        data.path("benchmark", "scheer2023_split.json")
    shipped = data.package_directory() / "benchmark" / "bands_v4.json"
    assert "reinstall wormbench" in data._repair("benchmark", shipped)


def test_the_host_root_is_read_as_given(tmp_path):
    """The host's root is the host's own: a listed file there is not checked."""
    host = tmp_path / "host"
    target = _make(host, "benchmark", "bands_v4.json", '{"stand-in": true}\n')
    data.set_root(host)
    assert data.path("benchmark", "bands_v4.json") == target


def test_the_shipped_copies_pass_their_own_check():
    """Every file shipped in the package matches the manifest it ships with."""
    data.set_root(None)
    listed = data.manifest()["files"]
    shipped = [key for key, entry in listed.items() if entry["in_package"]]
    assert shipped
    for key in shipped:
        namespace, name = key.split("/", 1)
        assert data.path(namespace, name).parent == data.package_directory() / namespace
