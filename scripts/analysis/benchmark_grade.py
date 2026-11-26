#!/usr/bin/env python
r"""Grade simulated worms on the benchmark: each statistic's value, z and pass, and the distance.

Two sources, one grading:

``batched``
    Run a non-learning hazard controller through the batched evaluator on the three wild-type
    assays (Scheer's small lawn, Scheer's uniform well and López-Cruz's food removal) for one MLP
    or connectome primitives arm, on the given seeds.

``captures``
    Read behaviour captures, one per line of a manifest: ``<family> <behaviour_capture.json>``,
    family ``small_lawn``, ``uniform_lawn`` or ``local_search``. Each captured run is one animal.

``tracks``
    Read imported tracks, one file per line of a manifest: ``<family> <tracks.csv | tracks.wcon>``
    (``quantumnematode.benchmark.track_import``; the CSV's metadata sit in ``<file>.json``). Each
    track is one animal, cut from its metadata's analysis start.

Every animal is read only over its assay's analysis window: the small lawn's and the uniform
well's last 40 min, and López-Cruz's 45 min from removal. The output gives
every statistic's value, band, capped z and pass; the wild type's pass at its Bonferroni bar; and
the distance D and the fitted distance D_fit. Each family is graded at its own size: on the vendored
bands where they were built at it, and otherwise on bands built now by ``benchmark_bands.py``'s
builder, each band on its own generator, which gives the vendored values at the vendored sizes.

Usage::

    uv run python scripts/analysis/benchmark_grade.py batched --controller constant \
        --seeds 9101-9164 --out <grade.json>
    uv run python scripts/analysis/benchmark_grade.py captures --manifest <manifest.txt> \
        --out <grade.json>
    uv run python scripts/analysis/benchmark_grade.py tracks --manifest <manifest.txt> \
        --out <grade.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import quantumnematode.brain  # noqa: F401  - registers the brains the configs name
from quantumnematode.benchmark import grading as g
from quantumnematode.evaluation.batched import BatchedEvaluator
from quantumnematode.evaluation.controllers import (
    BatchedConstantHazards,
    BatchedPirouetteHazards,
)
from quantumnematode.report.dtypes import BehaviourStep
from quantumnematode.utils.config_loader import load_simulation_config
from wormbench.bands import (  # noqa: F401 - re-exported: fits and probes reach them here
    _activity_bands_at,
    _band,
    _builder,
    _kinematic_bands_at,
    _v2_family_map,
    classifier,
    family_bands,
    family_bands_v2,
    load_bands,
    load_bands_v2,
    load_bands_v3,
    load_bands_v4,
    mutant_bands,
)
from wormbench.grade import (  # noqa: F401 - re-exported: fits and probes reach them here
    _chemotaxis_reading,
    _finite,
    _grade_measured,
    _grade_with_share,
    _small_lawn_derived,
    _validated,
    activity_values,
    chemotaxis_fragments,
    family_values,
    grade_populations,
    grade_populations_v2,
    grade_populations_v3,
    grade_populations_v4,
    grade_track_manifest,
    kinematic_values,
    v2_is_valid,
    v3_is_valid,
    v4_is_valid,
    window,
)
from wormbench.grade import track_populations as _track_populations

if TYPE_CHECKING:
    from collections.abc import Callable

DESCRIPTION = "Grade simulated worms on the benchmark's wild-type statistics."
_REPO = Path(__file__).resolve().parents[2]
BANDS_PATH = _REPO / "data" / "benchmark" / "bands.json"
FORAGING = _REPO / "configs" / "scenarios" / "foraging"
ARMS = {
    "mlp": "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1_primitives",
    "connectome": (
        "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_modulated_s1_primitives"
    ),
    # C.2's arm: the data-layer connectome under the compact encoder.
    "connectome_c2": (
        "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_encoded_s1_primitives"
    ),
    # C.2's registered body: that arm with on-food reversals following the worm's run speed.
    "connectome_c2_following": (
        "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_encoded_s1_primitives"
        "_following"
    ),
    # The MLP arm's environment with the same body, for controllers that replace its brain.
    "mlp_following": (
        "mlpppo_small_continuous2d_fick_adaptive_klinotaxis_lawns_internal_s1_primitives_following"
    ),
}
PROTOCOLS = {
    "small_lawn": "scheer_lawn",
    "uniform_lawn": "uniform_lawn_scheer",
    "local_search": "food_removal_lopezcruz",
}
# The preflight's registered settings: the parity suite's constant controller, and the pirouette
# rule at the same rates.
RUN_SPEED = 0.15
HAZARDS = (0.02, 0.01, 0.01)
PIROUETTE_BETA = 3.0


def track_populations(manifest: Path) -> dict[str, list[list[BehaviourStep]]]:
    """Read ``<family> <tracks>`` lines; a relative file resolves against the repository root."""
    return _track_populations(manifest, base=_REPO)


def batched_populations(arm: str, controller: str, seeds: list[int]) -> dict[str, Any]:
    """Run a hazard controller on each wild-type assay; return each family's windowed animals."""
    populations: dict[str, list[list[BehaviourStep]]] = {}
    for family, protocol in PROTOCOLS.items():
        config = load_simulation_config(str(FORAGING / f"{ARMS[arm]}_assay_{protocol}.yml"))
        batched = (
            BatchedConstantHazards(RUN_SPEED, HAZARDS)
            if controller == "constant"
            else BatchedPirouetteHazards(RUN_SPEED, HAZARDS, beta=PIROUETTE_BETA)
        )
        trajectories = BatchedEvaluator(config, batched, seeds).run()
        analysis = tuple(trajectories.assay["analysis_steps"])  # type: ignore[index]
        populations[family] = [
            window(family, trajectories.to_behaviour_steps(i), analysis)  # type: ignore[arg-type]
            for i in range(len(seeds))
        ]
    return populations


def capture_populations(
    manifest: Path,
    model: Any = None,  # noqa: ANN401 - a TrackerModel, or None to grade exact positions
) -> dict[str, list[list[BehaviourStep]]]:
    """Read ``<family> <behaviour_capture.json>`` lines; each captured run is one animal.

    With a tracker model, each run on Scheer's rig is measured on its own seed's stream before it is
    windowed; local search stays exact.
    """
    populations: dict[str, list[list[BehaviourStep]]] = {}
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        family, path = line.split(maxsplit=1)
        if family not in g.FAMILIES:
            msg = f"unknown family {family!r}; expected one of {sorted(g.FAMILIES)}"
            raise ValueError(msg)
        data = json.loads((_REPO / path.strip()).read_text())
        if "assay" not in data:
            msg = f"{path} has no assay timing; the benchmark reads assay captures only"
            raise ValueError(msg)
        analysis = tuple(data["assay"]["analysis_steps"])
        for run in data["runs"]:
            steps = [BehaviourStep(**s) for s in run["steps"]]
            if model is not None and family in LAWN_MEASURED:
                from quantumnematode.benchmark import measurement as mm

                steps = mm.measure(steps, int(run["seed"]), model)
            populations.setdefault(family, []).append(window(family, steps, analysis))  # type: ignore[arg-type]
    return populations


# ---------------------------------------------------------------------------------------------
# B v2: measured simulated tracks, the kinematics and chemotaxis families
# ---------------------------------------------------------------------------------------------

BANDS_V2_PATH = _REPO / "data" / "benchmark" / "bands_v2.json"
CONTROLS_V2_PATH = (
    _REPO
    / "docs"
    / "experiments"
    / "logbooks"
    / "supporting"
    / "115-benchmark-v2"
    / "controls_v2.json"
)


CHEMOTAXIS_PROTOCOL = "chemotaxis_chen"
# The families graded against Scheer's rig, whose tracker the lawn model measures. Local search is
# graded against López-Cruz's reorientation counts from another rig, whose tracker no track
# measures, so its simulated tracks stay exact.
LAWN_MEASURED = ("small_lawn", "uniform_lawn")


BANDS_V3_PATH = _REPO / "data" / "benchmark" / "bands_v3.json"
CONTROLS_V3_PATH = (
    _REPO
    / "docs"
    / "experiments"
    / "logbooks"
    / "supporting"
    / "120-benchmark-v3"
    / "controls_v3.json"
)


BANDS_V4_PATH = _REPO / "data" / "benchmark" / "bands_v4.json"
CONTROLS_V4_PATH = (
    _REPO
    / "docs"
    / "experiments"
    / "logbooks"
    / "supporting"
    / "124-benchmark-v4"
    / "controls_v4.json"
)


@overload
def simulate_v2(
    arm: str,
    controller: Callable[[Path], Any],
    seeds: list[int],
    bands: dict[str, Any],
    assays: None = None,
) -> tuple[dict[str, list[list[BehaviourStep]]], list[list[BehaviourStep]]]: ...


@overload
def simulate_v2(
    arm: str,
    controller: Callable[[Path], Any],
    seeds: list[int],
    bands: dict[str, Any],
    assays: tuple[str, ...],
) -> tuple[dict[str, list[list[BehaviourStep]]], list[list[BehaviourStep]] | None]: ...


def simulate_v2(
    arm: str,
    controller: Callable[[Path], Any],
    seeds: list[int],
    bands: dict[str, Any],
    assays: tuple[str, ...] | None = None,
) -> tuple[dict[str, list[list[BehaviourStep]]], list[list[BehaviourStep]] | None]:
    """Run a controller on the v2 assays; measure each rig's tracks before they are windowed.

    Returns the lawn families' windowed, measured animals and the chemotaxis assay's measured
    animals over its analysis window. ``controller`` builds a fresh
    controller for each assay from that assay's config path. ``assays`` names the families to run
    (``chemotaxis`` among them), all by default; each assay is its own run on the animals' own
    streams, so a subset gives those families' animals exactly as the full run does. Without
    ``chemotaxis`` among them, no chemotaxis population is returned (None), so the grading leaves
    that family out.
    """
    from quantumnematode.benchmark import measurement as mm
    from quantumnematode.evaluation.odour_field import chen_field, chen_starts

    lawn_model = mm.TrackerModel.from_dict(bands["measurement"]["lawn"])
    chem_model = mm.TrackerModel.from_dict(bands["measurement"]["chemotaxis"])
    populations: dict[str, list[list[BehaviourStep]]] = {}

    for family, protocol in PROTOCOLS.items():
        if assays is not None and family not in assays:
            continue
        path = FORAGING / f"{ARMS[arm]}_assay_{protocol}.yml"
        config = load_simulation_config(str(path))
        trajectories = BatchedEvaluator(config, controller(path), seeds).run()
        analysis = tuple(trajectories.assay["analysis_steps"])  # type: ignore[index]
        # Only each family's window is built. A lawn family's noise and losses are drawn over the
        # whole track, so its window equals the whole track measured and then windowed.
        samples = g.SEARCH_SAMPLES if family == "local_search" else g.LAWN_SAMPLES
        first, end = g.analysis_bounds(analysis, samples, trajectories.x.shape[1])  # type: ignore[arg-type]
        populations[family] = [
            mm.measure_window(
                trajectories.to_behaviour_steps(i, first, end),
                first,
                trajectories.primitive_kinds(i),
                seed,
                lawn_model,
            )
            if family in LAWN_MEASURED
            else trajectories.to_behaviour_steps(i, first, end)
            for i, seed in enumerate(seeds)
        ]
    if assays is not None and "chemotaxis" not in assays:
        return populations, None
    path = FORAGING / f"{ARMS[arm]}_assay_{CHEMOTAXIS_PROTOCOL}.yml"
    config = load_simulation_config(str(path))
    field = chen_field()
    trajectories = BatchedEvaluator(
        config,
        controller(path),
        seeds,
        odour_field=field,
        field_starts=chen_starts(seeds, field),
    ).run()
    first, stop = trajectories.assay["analysis_steps"]  # type: ignore[index]
    chemotaxis = [
        mm.measure(trajectories.to_behaviour_steps(i, first, stop), seed, chem_model)
        for i, seed in enumerate(seeds)
    ]
    return populations, chemotaxis


def parse_seeds(text: str) -> list[int]:
    """Parse ``a-b`` or ``a,b,c`` into seeds."""
    if "-" in text:
        low, high = (int(v) for v in text.split("-"))
        return list(range(low, high + 1))
    return [int(v) for v in text.split(",")]


def main(argv: list[str] | None = None) -> int:
    """CLI: grade a batched run or a set of captures and write the grading."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest="source", required=True)
    batched = sub.add_parser("batched")
    batched.add_argument("--controller", choices=("constant", "pirouette"), required=True)
    batched.add_argument("--arm", choices=tuple(ARMS), default="mlp")
    batched.add_argument("--seeds", required=True)
    batched.add_argument("--out", type=Path, required=True)
    captures = sub.add_parser("captures")
    captures.add_argument("--manifest", type=Path, required=True)
    captures.add_argument("--out", type=Path, required=True)
    tracks = sub.add_parser("tracks")
    tracks.add_argument("--manifest", type=Path, required=True)
    tracks.add_argument("--out", type=Path, required=True)
    for p in (batched, captures, tracks):
        p.add_argument(
            "--bands",
            choices=("v1", "v2", "v3", "v4"),
            default=None,
            help="the newest version whose controls find it VALID is the default: v4, v3 with the "
            "on-lawn activity family; v3, 21 statistics with reorientation graded; v2, measured "
            "simulated tracks, 19 statistics; v1 frozen for past verdicts",
        )
    args = parser.parse_args(argv)
    if args.bands is None:
        args.bands = (
            "v4" if v4_is_valid() else "v3" if v3_is_valid() else "v2" if v2_is_valid() else "v1"
        )
    if args.source == "tracks":
        # Imported tracks are graded by the package's own function, which `wormbench grade` calls.
        result = grade_track_manifest(args.manifest, args.bands, base=_REPO)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps({k: result[k] for k in ("family_passes", "wild_type_passes", "D", "D_fit")}),
        )
        return 0
    if args.bands in ("v2", "v3", "v4"):
        return _main_measured(args, int(args.bands[1]))
    bands = load_bands()
    if args.source == "batched":
        seeds = parse_seeds(args.seeds)
        populations = batched_populations(args.arm, args.controller, seeds)
        source: dict[str, Any] = {
            "source": "batched",
            "controller": args.controller,
            "arm": args.arm,
            "seeds": seeds,
            "run_speed_mm_per_s": RUN_SPEED,
            "hazards_per_s": list(HAZARDS),
            **({"beta": PIROUETTE_BETA} if args.controller == "pirouette" else {}),
        }
    else:
        populations = capture_populations(args.manifest)
        source = {"source": "captures", "manifest": str(args.manifest)}
    result = _finite({**source, **grade_populations(populations, bands)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("family_passes", "wild_type_passes", "D", "D_fit")}))
    return 0


def _hazard_controller(controller: str) -> Any:  # noqa: ANN401 - a batched controller
    if controller == "constant":
        return BatchedConstantHazards(RUN_SPEED, HAZARDS)
    return BatchedPirouetteHazards(RUN_SPEED, HAZARDS, beta=PIROUETTE_BETA)


def _main_measured(args: argparse.Namespace, version: int) -> int:
    """Grade on B v2 or v3: simulated tracks are measured; imported real tracks are not.

    The chemotaxis assay needs the measured odour field, which only the batched evaluator reads, so
    captures and imported tracks are graded without that family, and their D is not defined.
    """
    from quantumnematode.benchmark import measurement as mm

    bands = {2: load_bands_v2, 3: load_bands_v3, 4: load_bands_v4}[version]()
    chemotaxis: list[list[BehaviourStep]] | None = None
    if args.source == "batched":
        seeds = parse_seeds(args.seeds)
        populations, chemotaxis = simulate_v2(
            args.arm,
            lambda _path: _hazard_controller(args.controller),
            seeds,
            bands,
        )
        source: dict[str, Any] = {
            "source": "batched",
            "controller": args.controller,
            "arm": args.arm,
            "seeds": seeds,
            "run_speed_mm_per_s": RUN_SPEED,
            "hazards_per_s": list(HAZARDS),
            **({"beta": PIROUETTE_BETA} if args.controller == "pirouette" else {}),
        }
    else:
        model = mm.TrackerModel.from_dict(bands["measurement"]["lawn"])
        populations = capture_populations(args.manifest, model)
        source = {"source": "captures", "manifest": str(args.manifest)}
    grade = {2: grade_populations_v2, 3: grade_populations_v3, 4: grade_populations_v4}[version]
    graded = grade(populations, chemotaxis, bands)
    result = _finite({**source, "bands": f"v{version}", **graded})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("family_passes", "wild_type_passes", "D", "D_fit")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
