"""The ``wormbench`` command: grade tracks, write the worked example, and manage the data.

- ``wormbench grade --manifest <file> --out <json> [--bands v1|v2|v3|v4] [--simulated]`` grades a
  manifest of ``<family> <tracks.csv | tracks.wcon>`` lines;
- ``wormbench example --out <dir>`` writes a 64-animal real cohort and its manifest;
- ``wormbench data fetch [--dir <path>]`` downloads the tracks the band builder needs;
- ``wormbench data verify`` checks every reference file against the package's manifest;
- ``wormbench info`` prints the versions and where each reference file resolves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from wormbench import __version__, data

DESCRIPTION = "Grade simulated or real C. elegans tracks against real-worm behaviour."
# Every 1.x release grades on benchmark v4 unless told otherwise; only a major release changes it.
DEFAULT_VERSION = "v4"
VERSIONS = ("v1", "v2", "v3", "v4")
URLS_NAME = "urls.json"
EXAMPLE_ARCHIVE = "example/example.zip"


def _grade(args: argparse.Namespace) -> int:
    from wormbench import grade as wg
    from wormbench.measurement import file_sha256

    version = args.bands or DEFAULT_VERSION
    if args.bands is None and not wg._validated(DEFAULT_VERSION):  # noqa: SLF001 - the one rule
        msg = (
            f"the default benchmark {DEFAULT_VERSION} is not marked VALID by this package's "
            "record; pass --bands to choose a version"
        )
        raise SystemExit(msg)
    result: dict[str, Any] = wg.grade_track_manifest(
        args.manifest.resolve(),
        version,
        simulated=args.simulated,
    )
    bands_name = wg.BANDS_FILES[version]
    result["wormbench"] = {
        "package": __version__,
        "bands": version,
        "bands_sha256": file_sha256(data.path("benchmark", bands_name)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("family_passes", "wild_type_passes", "D", "D_fit")}))
    return 0


def _example(args: argparse.Namespace) -> int:
    archive = data.package_directory() / EXAMPLE_ARCHIVE
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(out)
    command = [
        "wormbench",
        "grade",
        "--manifest",
        str(out / "manifest.txt"),
        "--out",
        str(out / "grade.json"),
    ]
    print(f"wrote the example to {out}; grade it with:")
    print(f"  {shlex.join(command)}")
    return 0


def _fetched() -> dict[str, dict[str, Any]]:
    return {k: v for k, v in data.manifest()["files"].items() if not v["in_package"]}


def _fetch(args: argparse.Namespace) -> int:
    urls = data.package_record(URLS_NAME)
    target_root = Path(args.dir) if args.dir else data.cache_directory()
    missing_url = [key for key in _fetched() if not urls["files"].get(key)]
    if missing_url:
        msg = (
            "no download address is recorded for "
            f"{', '.join(missing_url)}: this release's data deposit has no published record yet"
        )
        raise SystemExit(msg)
    for key, entry in _fetched().items():
        namespace, name = key.split("/", 1)
        target = target_root / namespace / name
        if target.is_file() and _sha256(target) == entry["sha256"]:
            print(f"have {key}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with urllib.request.urlopen(urls["files"][key]) as response, partial.open("wb") as handle:  # noqa: S310 - the package's own pinned addresses
            shutil.copyfileobj(response, handle)
        found = _sha256(partial)
        if found != entry["sha256"]:
            partial.unlink()
            msg = f"{key} downloaded with SHA-256 {found}, not the manifest's {entry['sha256']}"
            raise SystemExit(msg)
        partial.replace(target)
        print(f"fetched {key}")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(_args: argparse.Namespace) -> int:
    bad = 0
    for key, entry in data.manifest()["files"].items():
        namespace, name = key.split("/", 1)
        found = data.locate(namespace, name)
        if found is None:
            state = "not fetched" if not entry["in_package"] else "MISSING"
            bad += state == "MISSING"
        elif _sha256(found) != entry["sha256"]:
            state = "DIFFERS"
            bad += 1
        else:
            state = "ok"
        print(f"{state:12s} {key}  {found or ''}")
    return 1 if bad else 0


def _info(_args: argparse.Namespace) -> int:
    print(f"wormbench {__version__}; default benchmark {DEFAULT_VERSION}; data {data.DATA_VERSION}")
    print(f"host root: {data.get_root()}")
    for key in data.manifest()["files"]:
        namespace, name = key.split("/", 1)
        print(f"  {key}: {data.locate(namespace, name)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the ``wormbench`` command."""
    parser = argparse.ArgumentParser(prog="wormbench", description=DESCRIPTION)
    sub = parser.add_subparsers(dest="command", required=True)
    grade = sub.add_parser("grade", help="grade a manifest of tracks")
    grade.add_argument("--manifest", type=Path, required=True)
    grade.add_argument("--out", type=Path, required=True)
    grade.add_argument("--bands", choices=VERSIONS, default=None)
    grade.add_argument(
        "--simulated",
        action="store_true",
        help="measure each animal as the tracker would, from its primitive labels and seed",
    )
    grade.set_defaults(run=_grade)
    example = sub.add_parser("example", help="write the worked example's cohort and manifest")
    example.add_argument("--out", type=Path, required=True)
    example.set_defaults(run=_example)
    data_parser = sub.add_parser("data", help="fetch or verify the reference data")
    data_sub = data_parser.add_subparsers(dest="action", required=True)
    fetch = data_sub.add_parser("fetch", help="download the tracks the band builder needs")
    fetch.add_argument("--dir", type=Path, default=None)
    fetch.set_defaults(run=_fetch)
    verify = data_sub.add_parser("verify", help="check every reference file against the manifest")
    verify.set_defaults(run=_verify)
    info = sub.add_parser("info", help="print versions and where each file resolves")
    info.set_defaults(run=_info)
    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
