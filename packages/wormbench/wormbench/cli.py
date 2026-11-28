"""The ``wormbench`` command: grade tracks, write the worked example, and manage the data.

- ``wormbench grade --manifest <file> --out <json> [--bands v1|v2|v3|v4] [--simulated]`` grades a
  manifest of ``<family> <tracks.csv | tracks.wcon>`` lines;
- ``wormbench example --out <dir>`` writes a 64-animal real cohort and its manifest;
- ``wormbench wcon --tracks <tracks.csv> --out <tracks.wcon> [--simulated]`` writes a CSV's tracks
  (with its sidecar) as WCON 1.0, and checks the WCON reads back the same;
- ``wormbench data fetch [--dir <path>]`` downloads the tracks the band builder needs;
- ``wormbench data verify`` checks every reference file against the package's manifest;
- ``wormbench info`` prints the versions and where each reference file resolves.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import shlex
import sys
import time
import urllib.error
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
# Requests per file: a transfer the server cuts short is resumed by the next request, from the byte
# it reached, so a few requests complete even an unreliable download.
DOWNLOAD_ATTEMPTS = 8
DOWNLOAD_TIMEOUT_S = 60
HTTP_PARTIAL_CONTENT = 206
HTTP_SERVER_ERROR = 500
# Client errors worth retrying: a timed-out request, and a rate limit.
RETRIED_CLIENT_ERRORS = (408, 429)


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
    # A manifest, track or reference file that cannot be read, and a grading the inputs do not
    # allow, are reported as one line, as `wcon` reports them.
    try:
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
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({k: result[k] for k in ("family_passes", "wild_type_passes", "D", "D_fit")}))
    return 0


def _example(args: argparse.Namespace) -> int:
    archive = data.package_directory() / EXAMPLE_ARCHIVE
    out: Path = args.out
    try:
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(out)
    except OSError as error:
        raise SystemExit(str(error)) from error
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


def _wcon(args: argparse.Namespace) -> int:
    from wormbench import track_export as te

    try:
        te.csv_to_wcon(args.tracks, args.out, simulated=args.simulated)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"wrote {args.out}")
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
        _download(key, urls["files"][key], partial, int(entry["size"]))
        found = _sha256(partial)
        if found != entry["sha256"]:
            partial.unlink()
            msg = f"{key} downloaded with SHA-256 {found}, not the manifest's {entry['sha256']}"
            raise SystemExit(msg)
        partial.replace(target)
        print(f"fetched {key}")
    return 0


def _download(key: str, url: str, partial: Path, size: int) -> None:
    """Download ``url`` into ``partial``, resuming whenever a transfer stops short.

    The file is complete when it reaches ``size``, the manifest's. Until then each request asks for
    the rest with a byte range: after a server closes the connection before the body it declared is
    complete, or answers a range with only part of the rest. A server that ignores the range is read
    from the start. A failed connection or a server error is retried after a pause; a missing or
    refused file is not. The caller checks the finished file's SHA-256.
    """
    partial.write_bytes(b"")
    last = "no response"
    for attempt in range(DOWNLOAD_ATTEMPTS):
        have = partial.stat().st_size
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})  # noqa: S310 - the package's own pinned addresses
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:  # noqa: S310 - the package's own pinned addresses
                resumed = have > 0 and getattr(response, "status", None) == HTTP_PARTIAL_CONTENT
                declared = response.headers.get("Content-Length")
                received = 0
                with partial.open("ab" if resumed else "wb") as handle:
                    for block in iter(lambda: response.read(1 << 16), b""):
                        handle.write(block)
                        received += len(block)
        except urllib.error.HTTPError as error:
            if error.code < HTTP_SERVER_ERROR and error.code not in RETRIED_CLIENT_ERRORS:
                partial.unlink(missing_ok=True)
                msg = f"{key}: {url} answered HTTP {error.code} ({error.reason})"
                raise SystemExit(msg) from error
            last = f"HTTP {error.code}"
            time.sleep(attempt + 1)
            continue
        except (OSError, http.client.HTTPException) as error:
            last = f"{type(error).__name__}: {error}"
            time.sleep(attempt + 1)
            continue
        total = partial.stat().st_size
        cut = declared is not None and received < int(declared)
        # Complete once the file reaches the manifest's size. A whole fresh response of another
        # size is the server's file as it stands, which more ranges cannot change; a partial
        # response that ends early, or a range answered with less than the rest, asks again.
        if total >= size or (not resumed and not cut):
            return
        last = (
            f"the server closed the transfer after {received:,} of {int(declared or 0):,} bytes"
            if cut
            else f"a range response left the file at {total:,} of {size:,} bytes"
        )
    got = partial.stat().st_size
    partial.unlink(missing_ok=True)
    msg = (
        f"{key}: the download from {url} stopped at {got:,} of {size:,} bytes after "
        f"{DOWNLOAD_ATTEMPTS} attempts ({last}); run `{data.FETCH_COMMAND}` again"
    )
    raise SystemExit(msg)


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
    wcon = sub.add_parser("wcon", help="write a CSV's tracks, with its sidecar, as WCON 1.0")
    wcon.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="the CSV; its sidecar is <file>.json",
    )
    wcon.add_argument("--out", type=Path, required=True)
    wcon.add_argument(
        "--simulated",
        action="store_true",
        help="carry each animal's primitive labels and seed, as --simulated grading reads them",
    )
    wcon.set_defaults(run=_wcon)
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
