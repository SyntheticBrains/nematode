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

``build`` writes both into the package's data directory. ``check`` regenerates both in memory and
exits nonzero if either differs from the shipped copy.

Usage::

    uv run python scripts/analysis/wormbench_records.py build
    uv run python scripts/analysis/wormbench_records.py check
"""

from __future__ import annotations

import argparse
import json
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


def validity() -> dict[str, Any]:
    """Return each version's verdict and validated bands' SHA-256, from its controls' record."""
    out: dict[str, Any] = {}
    for version, (bands_name, record) in CONTROLS.items():
        controls = json.loads(record.read_text())
        out[version] = {
            "bands": bands_name,
            "verdict": controls.get("verdict"),
            "bands_sha256": controls.get("bands_sha256"),
            "source": str(record.relative_to(_REPO)),
        }
    return out


def kappa_v1() -> dict[str, Any]:
    """Return v1's calibration factors, from the bands' committed record."""
    record = json.loads(KAPPA_RECORD.read_text())
    return {
        "kappa": record["calibration"]["kappa"],
        "source": str(KAPPA_RECORD.relative_to(_REPO)),
    }


def render(obj: dict[str, Any]) -> str:
    """Return a record's text: indented JSON with a trailing newline."""
    return json.dumps(obj, indent=2) + "\n"


def records() -> dict[str, str]:
    """Return each shipped record's name and text."""
    return {VALIDITY_NAME: render(validity()), KAPPA_NAME: render(kappa_v1())}


def main(argv: list[str] | None = None) -> int:
    """CLI: write the records, or check the shipped copies against their sources."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("action", choices=("build", "check"))
    args = parser.parse_args(argv)
    expected = records()
    if args.action == "build":
        PACKAGE_DATA.mkdir(parents=True, exist_ok=True)
        for name, text in expected.items():
            (PACKAGE_DATA / name).write_text(text)
        return 0
    stale = [
        name
        for name, text in expected.items()
        if not (PACKAGE_DATA / name).is_file() or (PACKAGE_DATA / name).read_text() != text
    ]
    for name in stale:
        print(f"stale or missing: {PACKAGE_DATA / name}", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
