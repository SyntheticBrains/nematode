#!/usr/bin/env python
"""Build or check the records that ship inside the ``wormbench`` package.

Two records are generated from committed ones, so the package can decide what it needs without
the repository:

- ``validity.json``: for each benchmark version from v2, its controls' verdict, the SHA-256 of the
  bands they validated, and the committed record it was read from. A version is the default only if
  its verdict is VALID and that SHA-256 equals the bands file's, the rule the grading scripts
  apply;
- ``kappa_v1.json``: v1's calibration factors, which the band builder applies at any size without
  vendored bands, read from the bands' committed record.

The small reference files ship as copies of their ``data/`` sources, and the manifest gives every
reference file's SHA-256 and size, those in the package and those fetched:

- ``build`` writes the two records;
- ``sync`` copies the in-package files and writes the manifest (``urls.json``, the fetched files'
  download addresses, is created with null addresses if absent and otherwise left alone);
- ``check`` regenerates all of it in memory and exits nonzero, naming each file, if a record, a
  copy or the manifest differs from what is shipped.

Usage::

    uv run python scripts/analysis/wormbench_records.py build
    uv run python scripts/analysis/wormbench_records.py sync
    uv run python scripts/analysis/wormbench_records.py check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

DESCRIPTION = "Build or check the records shipped inside the wormbench package."
_REPO = Path(__file__).resolve().parents[2]
SUPPORTING = _REPO / "docs" / "experiments" / "logbooks" / "supporting"
PACKAGE_DATA = _REPO / "packages" / "wormbench" / "wormbench" / "_data"
CONTROLS = {
    "v2": ("bands_v2.json", SUPPORTING / "115-benchmark-v2" / "controls_v2.json"),
    "v3": ("bands_v3.json", SUPPORTING / "120-benchmark-v3" / "controls_v3.json"),
    "v4": ("bands_v4.json", SUPPORTING / "124-benchmark-v4" / "controls_v4.json"),
}
KAPPA_RECORD = SUPPORTING / "109-benchmark-grading" / "bands_record.json"
VALIDITY_NAME = "validity.json"
KAPPA_NAME = "kappa_v1.json"
MANIFEST_NAME = "manifest.json"
URLS_NAME = "urls.json"
DATA = _REPO / "data"
# The files shipped inside the package, by namespace, and the files fetched on request.
IN_PACKAGE = {
    "benchmark": (
        "bands.json",
        "bands_v2.json",
        "bands_v3.json",
        "bands_v4.json",
        "classifier.json",
        "scheer_tracker_noise.npz",
        "lopezcruz2019_reference.json",
        "lopezcruz2019_split.json",
        "PROVENANCE.md",
    ),
    "roaming_dwelling": ("reference_hmm.json", "calibration_retry.json", "PROVENANCE.md"),
}
FETCHED = {
    "benchmark": (
        "scheer2023_tracks.npz",
        "scheer2023_split.json",
        "calhoun2014_tracks.npz",
        "roberts2016_tracks.npz",
        "chen2025_tracks.npz",
        "lopezcruz2019_reorientations.json",
    ),
}


def validity() -> dict[str, Any]:
    """Return each version's verdict and validated bands' SHA-256, from its controls' record."""
    out: dict[str, Any] = {}
    for version, (bands_name, record) in CONTROLS.items():
        controls = json.loads(record.read_text())
        out[version] = {
            "bands": bands_name,
            "verdict": controls.get("verdict"),
            "bands_sha256": controls.get("bands_sha256"),
            "source": record.relative_to(_REPO).as_posix(),
        }
    return out


def kappa_v1() -> dict[str, Any]:
    """Return v1's calibration factors, from the bands' committed record."""
    record = json.loads(KAPPA_RECORD.read_text())
    return {
        "kappa": record["calibration"]["kappa"],
        "source": KAPPA_RECORD.relative_to(_REPO).as_posix(),
    }


def render(obj: dict[str, Any]) -> str:
    """Return a record's text: indented JSON with a trailing newline."""
    return json.dumps(obj, indent=2) + "\n"


def records() -> dict[str, str]:
    """Return each shipped record's name and text."""
    return {VALIDITY_NAME: render(validity()), KAPPA_NAME: render(kappa_v1())}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest() -> dict[str, Any]:
    """Return every reference file's SHA-256 and size, and whether it ships in the package."""
    files: dict[str, Any] = {}
    for in_package, table in ((True, IN_PACKAGE), (False, FETCHED)):
        for namespace, names in table.items():
            for name in names:
                source = DATA / namespace / name
                files[f"{namespace}/{name}"] = {
                    "sha256": _sha256(source),
                    "size": source.stat().st_size,
                    "in_package": in_package,
                }
    return {"files": dict(sorted(files.items()))}


def _copies() -> list[tuple[Path, Path]]:
    """Return each in-package file's source and its copy."""
    return [
        (DATA / namespace / name, PACKAGE_DATA / namespace / name)
        for namespace, names in IN_PACKAGE.items()
        for name in names
    ]


def sync() -> None:
    """Copy the in-package files from ``data/`` and write the manifest."""
    for source, copy in _copies():
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copy)
    (PACKAGE_DATA / MANIFEST_NAME).write_text(render(manifest()))
    urls = PACKAGE_DATA / URLS_NAME
    if not urls.exists():
        names = sorted(f"{ns}/{n}" for ns, ns_names in FETCHED.items() for n in ns_names)
        urls.write_text(render({"record": None, "files": dict.fromkeys(names)}))


def stale() -> list[str]:
    """Return every shipped record, copy or manifest that differs from what it is made from."""
    out = [
        name
        for name, text in records().items()
        if not (PACKAGE_DATA / name).is_file() or (PACKAGE_DATA / name).read_text() != text
    ]
    out += [
        str(copy.relative_to(PACKAGE_DATA))
        for source, copy in _copies()
        if not copy.is_file() or copy.read_bytes() != source.read_bytes()
    ]
    shipped = PACKAGE_DATA / MANIFEST_NAME
    if not shipped.is_file() or shipped.read_text() != render(manifest()):
        out.append(MANIFEST_NAME)
    if not (PACKAGE_DATA / URLS_NAME).is_file():
        out.append(URLS_NAME)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: write the records, or check the shipped copies against their sources."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("action", choices=("build", "sync", "check"))
    args = parser.parse_args(argv)
    if args.action == "build":
        PACKAGE_DATA.mkdir(parents=True, exist_ok=True)
        for name, text in records().items():
            (PACKAGE_DATA / name).write_text(text)
        return 0
    if args.action == "sync":
        sync()
        return 0
    found = stale()
    for name in found:
        print(f"stale or missing: {PACKAGE_DATA / name}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
