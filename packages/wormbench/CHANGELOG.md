# Changelog

All notable changes to `wormbench` are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/). Every 1.x release grades on benchmark v4 by default; only a major release changes the default.

`wormbench` is released on its own, as `wormbench-vX.Y.Z` tags. The project it comes from keeps its own [changelog](../../CHANGELOG.md).

## [Unreleased]

## [1.0.1] - 2026-11-28

The first release on PyPI (`pip install wormbench`).

### Added

- `wormbench grade` grades a manifest of tracks, CSV with a JSON sidecar or WCON 1.0, on benchmark v4 by default: Scheer & Bargmann's small lawn and uniform well, kinematics and activity on the lawn, and López-Cruz et al.'s local search. `--bands v1|v2|v3` reproduces the frozen versions, and `--simulated` measures simulated tracks the way the tracker would. Every grading records the package version, the bands version and the bands' SHA-256.
- `wormbench example` writes 64 real band-cohort animals on the small lawn that grade with no download.
- `wormbench wcon` writes a CSV's tracks as WCON 1.0 and replaces the output only if the file reads back unchanged. From Python, `wormbench.track_export.write_csv` and `write_wcon` write tracks as `wormbench.track_import` reads them.
- `wormbench data fetch` downloads the real-worm tracks from their Zenodo deposit, [doi:10.5281/zenodo.24585652](https://doi.org/10.5281/zenodo.24585652), resuming a transfer the server cuts short, and `wormbench data verify` checks every reference file against its SHA-256. `wormbench info` shows where each file is read from.

### Fixed

- Against 1.0.0: `wormbench grade` and `wormbench example` report a manifest, track or reference file they cannot read, or a grading the inputs do not allow, in one line with exit status 1, instead of a traceback.

## [1.0.0] - 2026-11-28

Uploaded to TestPyPI only. Its check there found the tracebacks that 1.0.1 fixes, and an index never takes the same version twice, so PyPI's first version is 1.0.1.

[1.0.1]: https://github.com/SyntheticBrains/nematode/releases/tag/wormbench-v1.0.1
[unreleased]: https://github.com/SyntheticBrains/nematode/compare/wormbench-v1.0.1...HEAD
