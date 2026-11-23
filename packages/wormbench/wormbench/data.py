"""Where the benchmark's reference files are read from.

Each file belongs to a namespace (``benchmark`` or ``roaming_dwelling``) and is resolved when it is
used, never when a module is imported, from the first of these sources that holds it:

1. a root set by the host program with :func:`set_root`: a directory holding one subdirectory per
   namespace;
2. the directory named by the ``WORMBENCH_DATA`` environment variable, laid out the same way;
3. the files shipped inside the package;
4. the cache that ``wormbench data fetch`` fills.

A file absent from one source falls through to the next. Nothing is ever downloaded here: a file no
source holds raises :class:`MissingDataError`, naming the file and the command that fetches it.
"""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Any

NAMESPACES = ("benchmark", "roaming_dwelling")
ENVIRONMENT_VARIABLE = "WORMBENCH_DATA"
# The version of the fetched files; the cache keeps each version apart.
DATA_VERSION = "1"
FETCH_COMMAND = "wormbench data fetch"

_root: Path | None = None


class MissingDataError(FileNotFoundError):
    """A reference file that no source holds."""


def set_root(path: Path | str | None) -> None:
    """Set the host's data root, or clear it with ``None``."""
    global _root  # noqa: PLW0603 - one process-wide root, set by the host program
    _root = None if path is None else Path(path).resolve()


def get_root() -> Path | None:
    """Return the host's data root, if one is set."""
    return _root


def _check_namespace(namespace: str) -> None:
    if namespace not in NAMESPACES:
        msg = f"unknown data namespace {namespace!r}; expected one of {NAMESPACES}"
        raise ValueError(msg)


def package_directory() -> Path:
    """Return the directory of the files shipped inside the package."""
    return Path(str(resources.files("wormbench").joinpath("_data")))


def package_record(name: str) -> Any:  # noqa: ANN401 - any JSON-shaped record
    """Return a record shipped with the package, outside the namespaces, read from the package."""
    return json.loads((package_directory() / name).read_text())


def stored_path(text: str) -> Path:
    """Resolve a path a bands file stores, ``data/<namespace>/<name>``, through the sources."""
    parts = Path(text).parts
    if len(parts) != 3 or parts[0] != "data":  # noqa: PLR2004 - data, the namespace and the name
        msg = f"a stored reference path must read data/<namespace>/<name>, not {text!r}"
        raise ValueError(msg)
    return path(parts[1], parts[2])


def cache_directory() -> Path:
    """Return the fetched cache's directory for this data version."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "wormbench" / DATA_VERSION


def sources(namespace: str) -> list[Path]:
    """Return the namespace's candidate directories, in resolution order."""
    _check_namespace(namespace)
    found: list[Path] = []
    if _root is not None:
        found.append(_root / namespace)
    environment = os.environ.get(ENVIRONMENT_VARIABLE)
    if environment:
        found.append(Path(environment) / namespace)
    found.append(package_directory() / namespace)
    found.append(cache_directory() / namespace)
    return found


def directory(namespace: str) -> Path:
    """Return a namespace's directory: the host's, else the variable's, else the package's.

    It is the place a caller joining names onto the namespace would read from. Single files are
    resolved by :func:`path`, which also falls through to the fetched cache.
    """
    _check_namespace(namespace)
    if _root is not None:
        return _root / namespace
    environment = os.environ.get(ENVIRONMENT_VARIABLE)
    if environment:
        return Path(environment) / namespace
    return package_directory() / namespace


def locate(namespace: str, name: str) -> Path | None:
    """Return the first source's path to a file, or ``None`` if no source holds it."""
    for candidate in sources(namespace):
        target = candidate / name
        if target.is_file():
            return target
    return None


def path(namespace: str, name: str) -> Path:
    """Return a reference file's path, from the first source that holds it."""
    target = locate(namespace, name)
    if target is None:
        msg = (
            f"the reference file {namespace}/{name} is not available: no data root, "
            f"{ENVIRONMENT_VARIABLE}, package file or fetched copy holds it. Run `{FETCH_COMMAND}`."
        )
        raise MissingDataError(msg)
    return target


def expected_path(namespace: str, name: str) -> Path:
    """Return where a file resolves, or, if no source holds it, where the first source would.

    Kept for callers that read a module-level path constant without loading the file.
    """
    target = locate(namespace, name)
    return target if target is not None else directory(namespace) / name
