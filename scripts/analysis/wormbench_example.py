#!/usr/bin/env python
"""Build the worked example that ships inside the ``wormbench`` package.

The example is 64 wild-type animals (PD1074, OD 2) on Scheer & Bargmann's small lawn, written as
CSV tracks with their sidecars and a manifest, in one archive. The selection rule is fixed here,
before any grading:

- band-cohort rows only (``side == "bands"``), each passed through the held-out refusal, so no
  held-out row can enter;
- rows with at least 90% finite positions;
- 64 of them drawn without replacement by ``numpy.random.default_rng(SELECTION_SEED)``, in row
  order.

The animals are inside the bands they are graded against, so the example demonstrates the grader
and validates nothing. ``build`` writes the archive and the expected grading on benchmark v4 (its
``manifest`` path removed, since it depends on where the example is written). ``check`` rebuilds
both in memory and exits nonzero if either differs from the shipped copy.

Usage::

    uv run python scripts/analysis/wormbench_example.py build
    uv run python scripts/analysis/wormbench_example.py check
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import track_import_check as tic

DESCRIPTION = "Build or check the wormbench package's worked example."
_REPO = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = _REPO / "packages" / "wormbench" / "wormbench" / "_data" / "example"
ARCHIVE = EXAMPLE_DIR / "example.zip"
EXPECTED = EXAMPLE_DIR / "expected_v4.json"
ARM = "PD1074_od2_Data"
ANIMALS = 64
FINITE_SHARE = 0.9
SELECTION_SEED = 2050
# A fixed timestamp for every archive member, so the archive's bytes depend on its content only.
ZIP_DATE = (2026, 11, 24, 0, 0, 0)


def selected_rows() -> tuple[Any, np.ndarray]:
    """Return the Scheer store and the example's rows, by the fixed rule."""
    from wormbench import references as refs

    tracks = refs.load_scheer_tracks()
    rows = tracks.arm(ARM)
    rows = rows[tracks.side[rows] == refs.BANDS]
    tic.refuse_held_out(tracks, rows)
    finite = np.isfinite(tracks.x[rows]) & np.isfinite(tracks.y[rows])
    rows = rows[finite.mean(axis=1) >= FINITE_SHARE]
    chosen = np.sort(np.random.default_rng(SELECTION_SEED).choice(rows, ANIMALS, replace=False))
    tic.refuse_held_out(tracks, chosen)
    return tracks, chosen


def archive_bytes() -> bytes:
    """Return the example archive: each animal's CSV and sidecar, a manifest, and a README."""
    tracks, rows = selected_rows()
    t = np.arange(tracks.x.shape[1], dtype=float)
    members: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        names = []
        for row in rows:
            name = f"animal_{int(row)}"
            metadata = tic._lawn_metadata("small_lawn", tracks.boundary_radius[row])
            path = tic.write_track(
                directory,
                name,
                "csv",
                t,
                tracks.x[row],
                tracks.y[row],
                metadata,
            )
            members[path.name] = path.read_bytes()
            sidecar = path.with_name(path.name + ".json")
            members[sidecar.name] = sidecar.read_bytes()
            names.append(path.name)
    members["manifest.txt"] = "".join(f"small_lawn {n}\n" for n in names).encode()
    members["README.txt"] = (
        f"{ANIMALS} wild-type PD1074 animals (OD 2) on Scheer & Bargmann's small lawn, band-cohort\n"
        "rows only, from Scheer & Bargmann 2023 (Dryad doi:10.5061/dryad.47d7wm3jf, CC0-1.0).\n"
        "Grade them with: wormbench grade --manifest manifest.txt --out grade.json\n"
        "These animals are inside the bands they are graded against: the example shows how the\n"
        "grader reads tracks, and validates nothing.\n"
    ).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, members[name])
    return buffer.getvalue()


def expected_grading(archive: bytes) -> str:
    """Return the example's grading on benchmark v4 from the package's own copies, as text."""
    from wormbench import data
    from wormbench import grade as wg

    saved = data.get_root()
    data.set_root(None)
    try:
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                bundle.extractall(directory)
            graded = wg.grade_track_manifest(directory / "manifest.txt", "v4")
    finally:
        data.set_root(saved)
    graded.pop("manifest")
    return json.dumps(graded, indent=2, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI: build the example and its expected grading, or check the shipped copies."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("action", choices=("build", "check"))
    args = parser.parse_args(argv)
    archive = archive_bytes()
    expected = expected_grading(archive)
    if args.action == "build":
        EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        ARCHIVE.write_bytes(archive)
        EXPECTED.write_text(expected)
        print(f"wrote {ARCHIVE} ({len(archive)} bytes) and {EXPECTED}")
        return 0
    stale = [
        str(path)
        for path, content in ((ARCHIVE, archive), (EXPECTED, expected.encode()))
        if not path.is_file() or path.read_bytes() != content
    ]
    for path in stale:
        print(f"stale or missing: {path}", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
