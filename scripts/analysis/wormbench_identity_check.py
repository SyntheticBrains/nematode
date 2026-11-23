#!/usr/bin/env python
"""Byte-identity check for moving the benchmark grader into its own package.

Grades a fixed set of cases in one source tree and writes each case's output as JSON. The
``baseline`` stage runs in a tree before the move and the ``check`` stage in a tree after it, each
under that tree's own environment; ``check`` then compares every output with the baseline's, byte
for byte, and exits nonzero on any difference.

Run each stage from its tree, with that tree's interpreter in isolated mode, so that neither this
file's directory nor the working directory enters ``sys.path``::

    cd <tree> && .venv/bin/python -I <this file> baseline --tree . --build <shared dir>
    cd <tree> && .venv/bin/python -I <this file> check --tree . --build <shared dir>

Each stage records its tree's commit and the file of every module it grades with, and refuses to
run if any of them lies outside its tree, so two stages cannot grade the same code. ``check`` also
refuses if its commit is the baseline's.

The shared inputs (imported tracks and a capture set) are generated once, by the first baseline,
and pinned by SHA256; every later stage verifies them. Every Scheer row comes from the band
cohorts and passes ``track_import_check.refuse_held_out``. Every case runs on one thread.

Cases:

- **tracks:** band-cohort small-lawn and uniform-well animals, written as CSV and imported, at a
  vendored and an unvendored size, on every benchmark version;
- **batched:** the constant-hazard and pirouette negatives on fixed seeds, on every version;
- **captures:** a pinned capture set on v1 (exact positions) and v4 (measured);
- **no presence:** v2 and v4 gradings without the presence resamples;
- **mutants:** each mutant's bands at a vendored and an unvendored size, and a v4 mutant reading;
- **builder:** bands at an unvendored size on every builder path, and the builder at the vendored
  sizes against the vendored v1 bands;
- **c1 record:** C.1's v3 answer re-graded at its grading base, compared with Logbook 124's
  committed values as well as with the baseline.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DESCRIPTION = "Byte-identity check for moving the benchmark grader into its own package."
CASES = ("tracks", "batched", "captures", "no_presence", "mutants", "builder", "c1_record")
VERSIONS = ("v1", "v2", "v3", "v4")
CONTROLLERS = ("constant", "pirouette")
SEEDS_A = "9101-9164"
SEEDS_B = "9201-9264"
CAPTURE_SEEDS = list(range(9301, 9317))
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
SMALL_LAWN_ARM = "PD1074_od2_Data"
UNIFORM_ARM = "PD1074_od2_uniformlawn_Data"
UNVENDORED = 100
C1_POPULATIONS = 5
LOGBOOK_124_READ = Path("docs/experiments/logbooks/supporting/124-benchmark-v4/read.json")
C1_V3_FIT = Path("docs/experiments/logbooks/supporting/121-c1-v3-fit/fit.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, obj: Any) -> None:  # noqa: ANN401 - any JSON-shaped value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def _import_tree(tree: Path) -> dict[str, Any]:
    """Import the tree's scripts and package, and return the modules the cases grade with."""
    sys.path.insert(0, str(tree / "scripts" / "analysis"))
    import benchmark_grade as bg
    import c1_v3_fit
    import quantumnematode
    import track_import_check as tic

    modules: dict[str, Any] = {
        "quantumnematode": quantumnematode,
        "benchmark_grade": bg,
        "track_import_check": tic,
        "c1_v3_fit": c1_v3_fit,
    }
    try:
        # Absent before the move, by design: the baseline stage grades without it.
        import wormbench  # pyright: ignore[reportMissingImports]

        modules["wormbench"] = wormbench
    except ImportError:
        pass
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    return modules


def _guard(tree: Path, stage: str, modules: dict[str, Any]) -> dict[str, Any]:
    """Refuse a run whose modules lie outside its tree; return the tree's identity."""
    root = tree.resolve()
    files = {name: str(Path(m.__file__).resolve()) for name, m in modules.items()}
    outside = {n: f for n, f in files.items() if not Path(f).is_relative_to(root)}
    if outside:
        msg = f"modules outside the tree {root}: {outside}"
        raise SystemExit(msg)
    if stage == "check" and "wormbench" not in modules:
        msg = "the check stage needs wormbench importable from its tree"
        raise SystemExit(msg)
    head = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"tree": str(root), "commit": head, "modules": files}


# ---------------------------------------------------------------------------------------------
# Shared inputs: generated once, pinned by SHA256
# ---------------------------------------------------------------------------------------------


def _band_rows(tracks: Any, arm: str, tic: Any) -> Any:  # noqa: ANN401 - numpy rows
    from quantumnematode.benchmark import references as refs

    rows = tracks.arm(arm)
    rows = rows[tracks.side[rows] == refs.BANDS]
    tic.refuse_held_out(tracks, rows)
    return rows


def _make_inputs(inputs: Path, modules: dict[str, Any]) -> None:
    """Write the imported tracks, their manifests and the capture set."""
    import numpy as np
    from quantumnematode.benchmark.references import load_scheer_tracks
    from quantumnematode.evaluation.batched import BatchedEvaluator
    from quantumnematode.evaluation.controllers import BatchedConstantHazards
    from quantumnematode.utils.config_loader import load_simulation_config

    bg, tic = modules["benchmark_grade"], modules["track_import_check"]
    tracks = load_scheer_tracks()
    directory = inputs / "tracks"
    directory.mkdir(parents=True, exist_ok=True)
    t = np.arange(tracks.x.shape[1], dtype=float)
    written: dict[str, list[str]] = {}
    for family, arm in (("small_lawn", SMALL_LAWN_ARM), ("uniform_lawn", UNIFORM_ARM)):
        rows = _band_rows(tracks, arm, tic)
        for i in rows[:200]:
            name = f"{family}_{int(i)}"
            metadata = tic._lawn_metadata(family, tracks.boundary_radius[i])
            path = tic.write_track(directory, name, "csv", t, tracks.x[i], tracks.y[i], metadata)
            written.setdefault(family, []).append(str(path.resolve()))
    manifests = {
        "vendored": [("small_lawn", p) for p in written["small_lawn"][:200]],
        "unvendored": [("small_lawn", p) for p in written["small_lawn"][:UNVENDORED]]
        + [("uniform_lawn", p) for p in written["uniform_lawn"]],
    }
    for label, lines in manifests.items():
        (inputs / f"tracks_{label}.txt").write_text("".join(f"{f} {p}\n" for f, p in lines))
    captures = inputs / "captures"
    captures.mkdir(parents=True, exist_ok=True)
    lines = []
    for family, protocol in bg.PROTOCOLS.items():
        config_path = bg.FORAGING / f"{bg.ARMS['mlp']}_assay_{protocol}.yml"
        config = load_simulation_config(str(config_path))
        controller = BatchedConstantHazards(bg.RUN_SPEED, bg.HAZARDS)
        trajectories = BatchedEvaluator(config, controller, CAPTURE_SEEDS).run()
        runs = [
            {
                "seed": seed,
                "steps": [dataclasses.asdict(s) for s in trajectories.to_behaviour_steps(i)],
            }
            for i, seed in enumerate(CAPTURE_SEEDS)
        ]
        path = captures / f"{family}.json"
        path.write_text(json.dumps({"assay": trajectories.assay, "runs": runs}))
        lines.append(f"{family} {path.resolve()}\n")
    (inputs / "captures.txt").write_text("".join(lines))
    pins = {
        str(p.relative_to(inputs)): _sha256(p)
        for p in sorted(inputs.rglob("*"))
        if p.is_file() and p.name != "pins.json"
    }
    _write(inputs / "pins.json", pins)


def _verify_inputs(inputs: Path) -> None:
    pins = json.loads((inputs / "pins.json").read_text())
    bad = [name for name, sha in pins.items() if _sha256(inputs / name) != sha]
    if bad:
        msg = f"shared inputs changed since they were pinned: {bad[:5]}"
        raise SystemExit(msg)


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------


def _band_dict(bands: dict[str, Any]) -> dict[str, Any]:
    return {name: dataclasses.asdict(band) for name, band in sorted(bands.items())}


def _case_tracks(out: Path, inputs: Path, modules: dict[str, Any]) -> None:
    bg = modules["benchmark_grade"]
    for label in ("vendored", "unvendored"):
        for version in VERSIONS:
            manifest = inputs / f"tracks_{label}.txt"
            target = out / f"tracks_{label}_{version}.json"
            bg.main(
                ["tracks", "--manifest", str(manifest), "--out", str(target), "--bands", version],
            )


def _case_batched(out: Path, _inputs: Path, modules: dict[str, Any]) -> None:
    bg = modules["benchmark_grade"]
    for controller in CONTROLLERS:
        for version in VERSIONS:
            target = out / f"batched_{controller}_{version}.json"
            bg.main(
                [
                    "batched",
                    "--controller",
                    controller,
                    "--seeds",
                    SEEDS_A,
                    "--out",
                    str(target),
                    "--bands",
                    version,
                ],
            )


def _case_captures(out: Path, inputs: Path, modules: dict[str, Any]) -> None:
    bg = modules["benchmark_grade"]
    for version in ("v1", "v4"):
        target = out / f"captures_{version}.json"
        manifest = inputs / "captures.txt"
        bg.main(["captures", "--manifest", str(manifest), "--out", str(target), "--bands", version])


def _constant(bg: Any) -> Any:  # noqa: ANN401 - a controller factory
    from quantumnematode.evaluation.controllers import BatchedConstantHazards

    return lambda _path: BatchedConstantHazards(bg.RUN_SPEED, bg.HAZARDS)


def _case_no_presence(out: Path, _inputs: Path, modules: dict[str, Any]) -> None:
    bg = modules["benchmark_grade"]
    seeds = bg.parse_seeds(SEEDS_A)
    for version, load, grade in (
        ("v2", bg.load_bands_v2, bg.grade_populations_v2),
        ("v4", bg.load_bands_v4, bg.grade_populations_v4),
    ):
        bands = load()
        populations, chemotaxis = bg.simulate_v2("mlp", _constant(bg), seeds, bands)
        graded = grade(populations, chemotaxis, bands, presence=False)
        _write(out / f"no_presence_{version}.json", bg._finite(graded))


def _case_mutants(out: Path, _inputs: Path, modules: dict[str, Any]) -> None:
    from quantumnematode.benchmark import grading as g

    bg = modules["benchmark_grade"]
    mutants = ("tph-1", "pdfr-1", "tax-4")
    record: dict[str, Any] = {}
    for version, load in (("v1", bg.load_bands), ("v4", bg.load_bands_v4)):
        bands = load()
        for n in (64, UNVENDORED):
            for mutant in mutants:
                record[f"{version}/{n}/{mutant}"] = _band_dict(bg.mutant_bands(bands, mutant, n))
    bands = bg.load_bands_v4()
    clf = bg.classifier(bands)
    wild, _ = bg.simulate_v2("mlp", _constant(bg), bg.parse_seeds(SEEDS_A), bands, ("small_lawn",))
    other, _ = bg.simulate_v2("mlp", _constant(bg), bg.parse_seeds(SEEDS_B), bands, ("small_lawn",))
    wild_values = bg.family_values("small_lawn", wild["small_lawn"], clf)
    other_values = bg.family_values("small_lawn", other["small_lawn"], clf)
    delta = {s: other_values[s] - wild_values[s] for s in g.MUTANT_LAWN_STATISTICS}
    for mutant in mutants:
        mutant_bands = bg.mutant_bands(bands, mutant, 64)
        readings = g.grade(delta, mutant_bands, float(bands["bars"][mutant]), directions=True)
        record[f"reading/{mutant}"] = {
            s: {"value": r.value, "z": r.z, "passes": r.passes} for s, r in sorted(readings.items())
        }
    _write(out / "mutants.json", bg._finite(record))


def _case_builder(out: Path, _inputs: Path, modules: dict[str, Any]) -> None:
    bg = modules["benchmark_grade"]
    v1, v4 = bg.load_bands(), bg.load_bands_v4()
    record: dict[str, Any] = {
        "v1/small_lawn": _band_dict(bg.family_bands(v1, "small_lawn", UNVENDORED)),
        "v1/uniform_lawn": _band_dict(bg.family_bands(v1, "uniform_lawn", UNVENDORED)),
        "v4/kinematics": _band_dict(bg.family_bands_v2(v4, "kinematics", UNVENDORED)),
        "v4/activity": _band_dict(bg.family_bands_v2(v4, "activity", UNVENDORED)),
    }
    builder = bg._builder()
    for n in (64, 200):
        for family in ("small_lawn", "uniform_lawn"):
            built = _band_dict(
                builder.small_lawn(n) if family == "small_lawn" else builder.uniform_lawn(n),
            )
            vendored = _band_dict(
                {k: bg._band(e) for k, e in v1["bands"][str(n)][family].items()},
            )
            record[f"reproduces/{n}/{family}"] = built == vendored
    _write(out / "builder.json", bg._finite(record))


def _case_c1_record(out: Path, _inputs: Path, modules: dict[str, Any]) -> None:
    bg, c1 = modules["benchmark_grade"], modules["c1_v3_fit"]
    fit = json.loads(C1_V3_FIT.read_text())
    z = fit["restarts"][fit["chosen_restart"]]["answer"]
    committed = json.loads(LOGBOOK_124_READ.read_text())["c1_v3_answer"]
    base = int(committed["base"])
    record: dict[str, Any] = {"base": base, "populations": [], "matches_logbook_124": []}
    for k in range(C1_POPULATIONS):
        graded = c1.grade_answer(k, z, base)
        record["populations"].append(graded)
        reference = committed["populations"][k]["families"]
        matches = {}
        for family, readings in graded.get("families", {}).items():
            for stat, reading in readings.items():
                ref = reference.get(family, {}).get(stat)
                if ref is not None:
                    matches[f"{family}/{stat}"] = reading.get("value") == ref.get("value")
        record["matches_logbook_124"].append(matches)
    _write(out / "c1_record.json", bg._finite(record))


RUNNERS = {
    "tracks": _case_tracks,
    "batched": _case_batched,
    "captures": _case_captures,
    "no_presence": _case_no_presence,
    "mutants": _case_mutants,
    "builder": _case_builder,
    "c1_record": _case_c1_record,
}


# ---------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------


# Each case's outputs, by the prefix of their file names.
CASE_OUTPUTS = {
    "tracks": "tracks_",
    "batched": "batched_",
    "captures": "captures_",
    "no_presence": "no_presence_",
    "mutants": "mutants",
    "builder": "builder",
    "c1_record": "c1_record",
}


def _compare(build: Path, cases: list[str]) -> dict[str, Any]:
    """Compare the run cases' outputs with the baseline's, byte for byte."""
    baseline, check = build / "baseline", build / "check"
    prefixes = tuple(CASE_OUTPUTS[c] for c in cases)
    names = sorted(
        str(p.relative_to(baseline))
        for p in baseline.rglob("*.json")
        if p.name != "identity.json" and p.name.startswith(prefixes)
    )
    missing = [n for n in names if not (check / n).is_file()]
    differ = [
        n
        for n in names
        if n not in missing and (check / n).read_bytes() != (baseline / n).read_bytes()
    ]
    return {
        "cases": cases,
        "compared": len(names) - len(missing),
        "differ": differ,
        "missing": missing,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: run one stage's cases in one tree, and on ``check`` compare with the baseline."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("stage", choices=("baseline", "check"))
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(CASES))
    args = parser.parse_args(argv)
    tree, build = args.tree.resolve(), args.build.resolve()
    if build.is_relative_to(tree):
        msg = "the shared build directory must lie outside the tree"
        raise SystemExit(msg)
    cases = [c for c in args.cases.split(",") if c]
    unknown = set(cases) - set(CASES)
    if unknown:
        msg = f"unknown cases {sorted(unknown)}"
        raise SystemExit(msg)
    # One thread per run: set before anything imports numpy or torch.
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    os.chdir(tree)
    modules = _import_tree(tree)
    identity = _guard(tree, args.stage, modules)
    inputs = build / "inputs"
    if not (inputs / "pins.json").exists():
        if args.stage != "baseline":
            msg = "no pinned inputs: run the baseline first"
            raise SystemExit(msg)
        _make_inputs(inputs, modules)
    _verify_inputs(inputs)
    if args.stage == "check":
        base_identity = json.loads((build / "baseline" / "identity.json").read_text())
        if (
            base_identity["commit"] == identity["commit"]
            or base_identity["tree"] == identity["tree"]
        ):
            msg = "the check runs on the baseline's own tree or commit"
            raise SystemExit(msg)
    out = build / args.stage
    timings = {}
    for case in cases:
        start = time.perf_counter()
        RUNNERS[case](out, inputs, modules)
        timings[case] = round(time.perf_counter() - start, 1)
        print(f"{args.stage} {case}: {timings[case]} s", flush=True)
    _write(out / "identity.json", {**identity, "cases": cases, "seconds": timings})
    if args.stage == "baseline":
        return 0
    report = _compare(build, cases)
    _write(build / "report.json", report)
    print(json.dumps(report, indent=1))
    return 1 if report["differ"] or report["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
