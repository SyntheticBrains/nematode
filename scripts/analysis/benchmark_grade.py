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
import functools
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import quantumnematode.brain  # noqa: F401  - registers the brains the configs name
from quantumnematode.benchmark import grading as g
from quantumnematode.evaluation.batched import BatchedEvaluator
from quantumnematode.evaluation.controllers import (
    BatchedConstantHazards,
    BatchedPirouetteHazards,
)
from quantumnematode.report.dtypes import BehaviourStep
from quantumnematode.utils.config_loader import load_simulation_config
from quantumnematode.validation import roaming_dwelling as rd

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


def load_bands(path: Path = BANDS_PATH) -> dict[str, Any]:
    """Load the bands file."""
    return json.loads(path.read_text())


@functools.cache
def _builder() -> Any:  # noqa: ANN401 - the bands script's builder, imported on first use
    """Return the process's band builder, created once so its classified tallies are reused."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import benchmark_bands

    return benchmark_bands.BandBuilder()


def _band(entry: dict[str, Any]) -> g.Band:
    return g.Band(
        centre=float("nan") if entry["centre"] is None else float(entry["centre"]),
        scale=float("nan") if entry["scale"] is None else float(entry["scale"]),
        inflated=bool(entry["inflated"]),
        direction=int(entry.get("direction", 0)),
    )


def family_bands(bands: dict[str, Any], family: str, n: int) -> dict[str, g.Band]:
    """Return a family's bands at n: the vendored ones where built at n, else built now.

    Local search's band does not depend on n.
    """
    if family == "local_search":
        any_size = next(iter(bands["bands"].values()))
        return {name: _band(entry) for name, entry in any_size[family].items()}
    if str(n) in bands["bands"]:
        return {name: _band(entry) for name, entry in bands["bands"][str(n)][family].items()}
    builder = _builder()
    return builder.small_lawn(n) if family == "small_lawn" else builder.uniform_lawn(n)


def mutant_bands(bands: dict[str, Any], mutant: str, n: int) -> dict[str, g.Band]:
    """Return a mutant's Δ bands at n, its directions resolved over its paired band dates."""
    if str(n) in bands["bands"]:
        entries = bands["bands"][str(n)]["mutants"][mutant]
        return {name: _band(entry) for name, entry in entries.items()}
    return _builder().mutant(mutant, n)[0]


def classifier(bands: dict[str, Any]) -> rd.GaussianClassifier:
    """Return the classifier the bands were read with."""
    return rd.GaussianClassifier(hmm=rd.load_calibrated_hmm(_REPO / bands["classifier"]))


def family_values(
    family: str,
    animals: list[list[BehaviourStep]],
    clf: rd.GaussianClassifier,
) -> dict[str, float]:
    """Return a family's statistics over a population, each animal already in its window."""
    if family == "local_search":
        return g.simulated_search(animals)
    uniform = family == "uniform_lawn"
    total = sum(
        (g.tally(g.simulated_windows(steps, clf, uniform=uniform)) for steps in animals),
        start=np.zeros(len(g.TALLY_FIELDS)),
    )
    return g.lawn_statistics(total)


def window(
    family: str,
    steps: list[BehaviourStep],
    analysis: tuple[int, int],
) -> list[BehaviourStep]:
    """Return one animal's steps over its family's registered window."""
    samples = g.SEARCH_SAMPLES if family == "local_search" else g.LAWN_SAMPLES
    return g.analysis_steps(steps, analysis, samples)


def grade_populations(
    populations: dict[str, list[list[BehaviourStep]]],
    bands: dict[str, Any],
) -> dict[str, Any]:
    """Grade each family's population at its own size; return values, z, passes and distances."""
    clf = classifier(bands)
    bar = float(bands["bars"]["wild_type"])
    readings: dict[str, dict[str, g.Reading]] = {}
    sizes: dict[str, int] = {}
    presence: dict[str, Any] | None = None
    for family, animals in populations.items():
        sizes[family] = len(animals)
        values = family_values(family, animals, clf)
        readings[family] = g.grade(values, family_bands(bands, family, len(animals)), bar)
        if family == "local_search":
            presence = g.simulated_search_presence(animals)
    graded = list(readings)
    complete = set(graded) == set(g.FAMILIES)
    out: dict[str, Any] = {
        "n": sizes,
        "bar": bar,
        "families": {
            family: {
                name: {
                    "value": r.value,
                    "centre": r.band.centre,
                    "scale": r.band.scale,
                    "z": r.z,
                    "passes": r.passes,
                }
                for name, r in rs.items()
            }
            for family, rs in readings.items()
        },
        "family_passes": {f: all(r.passes for r in rs.values()) for f, rs in readings.items()},
    }
    if presence is not None:
        # Local search passes only where it is present: early/late above 1 at its lower bound.
        out["local_search_presence"] = presence
        out["family_passes"]["local_search"] = bool(
            out["family_passes"]["local_search"] and presence["present"],
        )
    out["wild_type_passes"] = complete and all(out["family_passes"].values())
    out["D"] = g.distance(readings, list(g.FAMILIES)) if complete else None
    fit = [f for f in g.FIT_FAMILIES if f in readings]
    out["D_fit"] = g.distance(readings, fit) if set(fit) == set(g.FIT_FAMILIES) else None
    return out


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

    With a tracker model, each run is measured on its own seed's stream before it is windowed.
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
            if model is not None:
                from quantumnematode.benchmark import measurement as mm

                steps = mm.measure(steps, int(run["seed"]), model)
            populations.setdefault(family, []).append(window(family, steps, analysis))  # type: ignore[arg-type]
    return populations


def track_populations(manifest: Path) -> dict[str, list[list[BehaviourStep]]]:
    """Read ``<family> <tracks>`` lines: CSV (with its sidecar) or WCON files, imported tracks."""
    from quantumnematode.benchmark.track_import import import_population

    populations: dict[str, list[list[BehaviourStep]]] = {}
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        family, path_text = line.split(maxsplit=1)
        path = Path(path_text.strip())
        found, animals = import_population(path if path.is_absolute() else _REPO / path)
        if found != family:
            msg = f"{path} declares family {found!r}, but the manifest line names {family!r}"
            raise ValueError(msg)
        populations.setdefault(family, []).extend(animals)
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


def v2_is_valid() -> bool:
    """Return whether B v2's bands are vendored and its controls' verdict is VALID."""
    if not (BANDS_V2_PATH.exists() and CONTROLS_V2_PATH.exists()):
        return False
    return json.loads(CONTROLS_V2_PATH.read_text()).get("verdict") == "VALID"


CHEMOTAXIS_PROTOCOL = "chemotaxis_chen"


def load_bands_v2(path: Path = BANDS_V2_PATH) -> dict[str, Any]:
    """Load B v2's bands, with its families, bar and measurement models."""
    return json.loads(path.read_text())


def _v2_family_map(bands: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    return {f: tuple(st) for f, st in bands["families"].items()}


def family_bands_v2(bands: dict[str, Any], family: str, n: int) -> dict[str, g.Band]:
    """Return a v2 family's bands at n; chemotaxis and local search do not depend on n."""
    if family in ("chemotaxis", "local_search"):
        any_size = next(iter(bands["bands"].values()))
        return {name: _band(entry) for name, entry in any_size[family].items()}
    if str(n) in bands["bands"]:
        return {name: _band(entry) for name, entry in bands["bands"][str(n)][family].items()}
    if family == "kinematics":
        return _kinematic_bands_at(n)
    return family_bands(bands, family, n)


@functools.cache
def _kinematic_bands_at(n: int) -> dict[str, g.Band]:
    import benchmark_bands as bb

    builder = _builder()
    per_date = bb.per_date_kinematic_tallies(builder.tracks, bb.WILD_TYPE, builder._classifier)
    kappa = load_bands_v2()["kinematics_kappa"]
    return bb.lawn_bands(
        per_date,
        g.KINEMATIC_STATISTICS,
        n,
        np.random.default_rng([bb.BAND_SEED, bb.KINEMATICS_INDEX, n]),
        kappa=kappa,
        statistic=g.kinematic_statistics,
    )


def kinematic_values(
    animals: list[list[BehaviourStep]],
    clf: rd.GaussianClassifier,
) -> dict[str, float]:
    """Return the kinematics family's statistics for a small-lawn population."""
    total = np.zeros(g.KINEMATIC_FIELDS)
    for steps in animals:
        x = np.array([s.x for s in steps], dtype=float)
        y = np.array([s.y for s in steps], dtype=float)
        on = np.array([bool(s.on_lawn) for s in steps])
        total += g.kinematic_tally(g.simulated_windows(steps, clf), x, y, on)
    return g.kinematic_statistics(total)


def chemotaxis_fragments(
    animals: list[list[BehaviourStep]],
    field: Any,  # noqa: ANN401 - an OdourField
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a measured chemotaxis population's fragments in the field's frame.

    Samples outside the measured field are missing, as a real worm's track ends when it leaves the
    imaged field.
    """
    from quantumnematode.benchmark import chemotaxis as ch

    out = []
    for steps in animals:
        xy = np.array([(s.x, s.y) for s in steps], dtype=float)
        finite = np.isfinite(xy).all(axis=1)
        inside = np.zeros(len(xy), dtype=bool)
        inside[finite] = field.inside(xy[finite])
        xy[~inside] = np.nan
        local = xy - np.asarray(field.origin_mm)
        out.extend(ch.split_fragments(local[:, 0], local[:, 1]))
    return out


def simulate_v2(
    arm: str,
    controller: Callable[[Path], Any],
    seeds: list[int],
    bands: dict[str, Any],
) -> tuple[dict[str, list[list[BehaviourStep]]], list[list[BehaviourStep]]]:
    """Run a controller on the v2 assays; measure every track before it is windowed.

    Returns the lawn families' windowed, measured animals and the chemotaxis assay's measured
    animals over its analysis window. ``controller`` builds a fresh
    controller for each assay from that assay's config path.
    """
    from quantumnematode.benchmark import measurement as mm
    from quantumnematode.evaluation.odour_field import chen_field, chen_starts

    lawn_model = mm.TrackerModel.from_dict(bands["measurement"]["lawn"])
    chem_model = mm.TrackerModel.from_dict(bands["measurement"]["chemotaxis"])
    populations: dict[str, list[list[BehaviourStep]]] = {}

    for family, protocol in PROTOCOLS.items():
        path = FORAGING / f"{ARMS[arm]}_assay_{protocol}.yml"
        config = load_simulation_config(str(path))
        trajectories = BatchedEvaluator(config, controller(path), seeds).run()
        analysis = tuple(trajectories.assay["analysis_steps"])  # type: ignore[index]
        populations[family] = [
            window(
                family,
                mm.measure(trajectories.to_behaviour_steps(i), seed, lawn_model),
                analysis,  # type: ignore[arg-type]
            )
            for i, seed in enumerate(seeds)
        ]
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
        mm.measure(trajectories.to_behaviour_steps(i)[first:stop], seed, chem_model)
        for i, seed in enumerate(seeds)
    ]
    return populations, chemotaxis


def grade_populations_v2(
    populations: dict[str, list[list[BehaviourStep]]],
    chemotaxis: list[list[BehaviourStep]] | None,
    bands: dict[str, Any],
) -> dict[str, Any]:
    """Grade on B v2: v1's families, kinematics from the small lawn's animals, and chemotaxis.

    Chemotaxis, like local search, passes only where it is present.
    """
    from quantumnematode.benchmark import chemotaxis as ch
    from quantumnematode.benchmark.references import sample_field
    from quantumnematode.evaluation.odour_field import chen_field

    clf = classifier(bands)
    bar = float(bands["bars"]["wild_type"])
    family_map = _v2_family_map(bands)
    readings: dict[str, dict[str, g.Reading]] = {}
    sizes: dict[str, int] = {}
    presence: dict[str, Any] | None = None
    for family, animals in populations.items():
        sizes[family] = len(animals)
        values = family_values(family, animals, clf)
        readings[family] = g.grade(values, family_bands_v2(bands, family, len(animals)), bar)
        if family == "local_search":
            presence = g.simulated_search_presence(animals)
        if family == "small_lawn":
            sizes["kinematics"] = len(animals)
            readings["kinematics"] = g.grade(
                kinematic_values(animals, clf),
                family_bands_v2(bands, "kinematics", len(animals)),
                bar,
            )
    chemotaxis_presence: dict[str, Any] | None = None
    if chemotaxis is not None:
        field = chen_field()

        def concentration(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return sample_field(field.field, field.cell_mm, x, y)

        # One group per animal: presence resamples animals, as local search's does.
        groups = [
            ch.fragment_samples(chemotaxis_fragments([animal], field), concentration)
            for animal in chemotaxis
        ]
        sizes["chemotaxis"] = len(chemotaxis)
        readings["chemotaxis"] = g.grade(
            ch.pooled_statistics([sample for group in groups for sample in group]),
            family_bands_v2(bands, "chemotaxis", len(chemotaxis)),
            bar,
        )
        chemotaxis_presence = ch.chemotaxis_presence(groups)
    out: dict[str, Any] = {
        "version": 2,
        "n": sizes,
        "bar": bar,
        "families": {
            family: {
                name: {
                    "value": r.value,
                    "centre": r.band.centre,
                    "scale": r.band.scale,
                    "z": r.z,
                    "passes": r.passes,
                }
                for name, r in rs.items()
            }
            for family, rs in readings.items()
        },
        "family_passes": {f: all(r.passes for r in rs.values()) for f, rs in readings.items()},
    }
    if presence is not None:
        out["local_search_presence"] = presence
        out["family_passes"]["local_search"] = bool(
            out["family_passes"]["local_search"] and presence["present"],
        )
    if chemotaxis_presence is not None:
        # Chemotaxis passes only where it is present: drift and the log ratio above 0.
        out["chemotaxis_presence"] = chemotaxis_presence
        out["family_passes"]["chemotaxis"] = bool(
            out["family_passes"]["chemotaxis"] and chemotaxis_presence["present"],
        )
    complete = set(readings) == set(family_map)
    out["wild_type_passes"] = complete and all(out["family_passes"].values())
    out["D"] = g.distance(readings, list(family_map), family_map) if complete else None
    fit = list(bands["fit_families"])
    out["D_fit"] = g.distance(readings, fit, family_map) if set(fit) <= set(readings) else None
    return out


def parse_seeds(text: str) -> list[int]:
    """Parse ``a-b`` or ``a,b,c`` into seeds."""
    if "-" in text:
        low, high = (int(v) for v in text.split("-"))
        return list(range(low, high + 1))
    return [int(v) for v in text.split(",")]


def _finite(value: Any) -> Any:  # noqa: ANN401 - any JSON-shaped value
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


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
            choices=("v1", "v2"),
            default=None,
            help="v2 (the default once its controls find it VALID): measured simulated tracks, "
            "20 statistics; v1 is frozen for past verdicts",
        )
    args = parser.parse_args(argv)
    if args.bands is None:
        args.bands = "v2" if v2_is_valid() else "v1"
    if args.bands == "v2":
        return _main_v2(args)
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
    elif args.source == "tracks":
        populations = track_populations(args.manifest)
        source = {"source": "tracks", "manifest": str(args.manifest)}
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


def _main_v2(args: argparse.Namespace) -> int:
    """Grade on B v2: simulated tracks are measured; imported real tracks are not.

    The chemotaxis assay needs the measured odour field, which only the batched evaluator reads, so
    captures and imported tracks are graded without that family, and their D is not defined.
    """
    from quantumnematode.benchmark import measurement as mm

    bands = load_bands_v2()
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
    elif args.source == "tracks":
        populations = track_populations(args.manifest)
        source = {"source": "tracks", "manifest": str(args.manifest)}
    else:
        model = mm.TrackerModel.from_dict(bands["measurement"]["lawn"])
        populations = capture_populations(args.manifest, model)
        source = {"source": "captures", "manifest": str(args.manifest)}
    graded = grade_populations_v2(populations, chemotaxis, bands)
    result = _finite({**source, "bands": "v2", **graded})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("family_passes", "wild_type_passes", "D", "D_fit")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
