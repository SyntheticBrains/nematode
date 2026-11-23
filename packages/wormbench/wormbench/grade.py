"""Grade populations of tracks on every benchmark version.

Each family is read over its registered window and graded at its population's own size. The
result gives every statistic's value, band, capped z and pass; the wild type's pass at its
Bonferroni bar; and the distance D and the fitted distance D_fit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from wormbench import classifier as rd
from wormbench import data
from wormbench import grading as g
from wormbench.bands import _v2_family_map, classifier, family_bands, family_bands_v2

if TYPE_CHECKING:
    from wormbench.records import BehaviourStep

# The record of each version's controls' verdict, shipped with the package.
VALIDITY_RECORD = "validity.json"


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


def track_populations(
    manifest: Path,
    base: Path | None = None,
) -> dict[str, list[list[BehaviourStep]]]:
    """Read ``<family> <tracks>`` lines: CSV (with its sidecar) or WCON files, imported tracks.

    A relative file resolves against ``base``, by default the manifest's own directory.
    """
    from wormbench.track_import import import_population

    root = Path(manifest).parent if base is None else Path(base)

    populations: dict[str, list[list[BehaviourStep]]] = {}
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        family, path_text = line.split(maxsplit=1)
        path = Path(path_text.strip())
        found, animals = import_population(path if path.is_absolute() else root / path)
        if found != family:
            msg = f"{path} declares family {found!r}, but the manifest line names {family!r}"
            raise ValueError(msg)
        populations.setdefault(family, []).extend(animals)
    return populations


def _validated(version: str, validity: dict[str, Any] | None = None) -> bool:
    """Return whether a version's controls found VALID the very bands file now resolved.

    ``validity`` is the record of each version's verdict and validated bands' SHA-256, by default
    the one shipped with the package, which is generated from the controls' committed records.
    """
    try:
        record = (data.package_record(VALIDITY_RECORD) if validity is None else validity)[version]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False  # an unreadable verdict leaves the default where it was
    if not isinstance(record, dict) or record.get("verdict") != "VALID":
        return False
    bands_path = data.locate("benchmark", str(record.get("bands")))
    if bands_path is None:
        return False
    from wormbench.measurement import file_sha256

    try:
        return record.get("bands_sha256") == file_sha256(bands_path)
    except OSError:
        return False  # unreadable bands leave the default where it was


def v2_is_valid(validity: dict[str, Any] | None = None) -> bool:
    """Return whether benchmark v2's controls found VALID the very bands file now resolved.

    The verdict records the SHA-256 of the bands it validated; a bands file that differs from it, a
    verdict that records none, or one that cannot be read, does not make v2 the default.
    """
    return _validated("v2", validity)


def v3_is_valid(validity: dict[str, Any] | None = None) -> bool:
    """Return whether benchmark v3's controls found VALID the very bands file now resolved."""
    return _validated("v3", validity)


def v4_is_valid(validity: dict[str, Any] | None = None) -> bool:
    """Return whether benchmark v4's controls found VALID the very bands file now resolved."""
    return _validated("v4", validity)


def activity_values(animals: list[list[BehaviourStep]]) -> dict[str, float]:
    """Return the activity family's statistics for a small-lawn population, from its positions."""
    total = np.zeros(len(g.ACTIVITY_FIELDS))
    for steps in animals:
        x = np.array([s.x for s in steps], dtype=float)
        y = np.array([s.y for s in steps], dtype=float)
        on = np.array([bool(s.on_lawn) for s in steps])
        total += g.activity_tally(x, y, on)
    return g.activity_statistics(total)


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
    from wormbench import chemotaxis as ch

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


def _chemotaxis_reading(
    chemotaxis: list[list[BehaviourStep]],
    bands: dict[str, Any],
    bar: float,
    *,
    presence: bool,
    version: int = 2,
) -> tuple[dict[str, g.Reading], dict[str, Any] | None]:
    """Return the chemotaxis family's readings, and its presence test where asked for.

    v3 reads five statistics from the same samples, v2 three.
    """
    from wormbench import chemotaxis as ch
    from wormbench.odour_field import chen_field
    from wormbench.references import sample_field

    field = chen_field()

    def concentration(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return sample_field(field.field, field.cell_mm, x, y)

    # One group per animal: presence resamples animals, as local search's does.
    groups = [
        ch.fragment_samples(chemotaxis_fragments([animal], field), concentration)
        for animal in chemotaxis
    ]
    pooled = ch.pooled_statistics_v3 if version >= 3 else ch.pooled_statistics  # noqa: PLR2004 - v3
    readings = g.grade(
        pooled([sample for group in groups for sample in group]),
        family_bands_v2(bands, "chemotaxis", len(chemotaxis)),
        bar,
    )
    return readings, (ch.chemotaxis_presence(groups) if presence else None)


def grade_populations_v2(
    populations: dict[str, list[list[BehaviourStep]]],
    chemotaxis: list[list[BehaviourStep]] | None,
    bands: dict[str, Any],
    *,
    presence: bool = True,
) -> dict[str, Any]:
    """Grade on benchmark v2: v1's families, kinematics from the small lawn, and chemotaxis.

    Chemotaxis, like local search, passes only where it is present. With ``presence=False``, as a
    fit's objective reads only statistics and distances, the presence tests are skipped: every
    statistic, D and D_fit are unchanged, and the passes, which need presence, are not given.
    """
    return _grade_measured(populations, chemotaxis, bands, presence=presence, version=2)


def grade_populations_v3(
    populations: dict[str, list[list[BehaviourStep]]],
    chemotaxis: list[list[BehaviourStep]] | None,
    bands: dict[str, Any],
    *,
    presence: bool = True,
) -> dict[str, Any]:
    """Grade on benchmark v3: v2's grading, with local search's log ratio and chemotaxis's five.

    Presence is read as on v2. The share of seconds in a reversal bout, over the local-search and
    chemotaxis windows, is reported beside the grade where the steps carry primitive labels, with
    its real reference; it takes no part in any pass, z or distance.
    """
    return _grade_with_share(populations, chemotaxis, bands, presence=presence, version=3)


def _grade_with_share(
    populations: dict[str, list[list[BehaviourStep]]],
    chemotaxis: list[list[BehaviourStep]] | None,
    bands: dict[str, Any],
    *,
    presence: bool,
    version: int,
) -> dict[str, Any]:
    """Grade on v3 or v4, with the share of seconds reversing reported beside the grade."""
    out = _grade_measured(populations, chemotaxis, bands, presence=presence, version=version)
    labelled = [*populations.get("local_search", []), *(chemotaxis or [])]
    share = g.reversal_share(labelled)
    out["reversal_share"] = (
        None
        if share is None
        else {
            "value": share,
            "windows": [
                w
                for w, animals in (
                    ("local_search", populations.get("local_search")),
                    ("chemotaxis", chemotaxis),
                )
                if animals
            ],
            "reference": bands["reported"]["reversal_share"],
        }
    )
    return out


def grade_populations_v4(
    populations: dict[str, list[list[BehaviourStep]]],
    chemotaxis: list[list[BehaviourStep]] | None,
    bands: dict[str, Any],
    *,
    presence: bool = True,
) -> dict[str, Any]:
    """Grade on benchmark v4: v3's grading, with the on-lawn activity family from the small lawn.

    The activity statistics the bands grade are read on their bands; the onset rate, where the
    bands report it rather than grade it, is reported beside the grade. Refuses bands whose onset
    rate is not yet decided.
    """
    if bands.get("activity_graded") is None:
        msg = (
            "these v4 bands have not recorded whether the onset rate is graded; set it with "
            "benchmark_bands.py --version v4 --onset once the oracle's check is read"
        )
        raise ValueError(msg)
    return _grade_with_share(populations, chemotaxis, bands, presence=presence, version=4)


def _small_lawn_derived(
    animals: list[list[BehaviourStep]],
    clf: rd.GaussianClassifier,
    bands: dict[str, Any],
    bar: float,
    version: int,
) -> tuple[dict[str, dict[str, g.Reading]], dict[str, float], dict[str, float] | None]:
    """Return the families read from the small lawn's animals, and what is reported beside them.

    Kinematics on every measured version, with the on-lawn reversal rate reported; from v4, the
    activity family, its ungraded statistics reported.
    """
    kinematics = kinematic_values(animals, clf)
    readings = {
        "kinematics": g.grade(kinematics, family_bands_v2(bands, "kinematics", len(animals)), bar),
    }
    # Computed beside the graded kinematics and reported, never read on a band.
    reported = {name: kinematics[name] for name in g.KINEMATIC_REPORTED}
    if version < 4:  # noqa: PLR2004 - activity is graded from benchmark v4
        return readings, reported, None
    activity = activity_values(animals)
    graded = tuple(bands["families"]["activity"])
    # The bands hold every activity statistic, graded or not: grade only the graded ones.
    activity_bands = family_bands_v2(bands, "activity", len(animals))
    readings["activity"] = g.grade(
        {name: activity[name] for name in graded},
        {name: activity_bands[name] for name in graded},
        bar,
    )
    return readings, reported, {k: v for k, v in activity.items() if k not in graded}


def _grade_measured(
    populations: dict[str, list[list[BehaviourStep]]],
    chemotaxis: list[list[BehaviourStep]] | None,
    bands: dict[str, Any],
    *,
    presence: bool,
    version: int,
) -> dict[str, Any]:
    """Grade on benchmark v2 to v4, which differ in local search's ratio and chemotaxis's reads."""
    clf = classifier(bands)
    bar = float(bands["bars"]["wild_type"])
    family_map = _v2_family_map(bands)
    readings: dict[str, dict[str, g.Reading]] = {}
    sizes: dict[str, int] = {}
    search_presence: dict[str, Any] | None = None
    reported: dict[str, float] | None = None
    activity_reported: dict[str, float] | None = None
    for family, animals in populations.items():
        sizes[family] = len(animals)
        values = family_values(family, animals, clf)
        if family == "local_search" and version >= 3:  # noqa: PLR2004 - the v3 log ratio
            values = g.log_search_statistics(values)
        readings[family] = g.grade(values, family_bands_v2(bands, family, len(animals)), bar)
        if family == "local_search" and presence:
            search_presence = g.simulated_search_presence(animals)
        if family == "small_lawn":
            derived, reported, activity_reported = _small_lawn_derived(
                animals,
                clf,
                bands,
                bar,
                version,
            )
            readings.update(derived)
            sizes.update(dict.fromkeys(derived, len(animals)))
    chemotaxis_presence: dict[str, Any] | None = None
    if chemotaxis is not None:
        sizes["chemotaxis"] = len(chemotaxis)
        readings["chemotaxis"], chemotaxis_presence = _chemotaxis_reading(
            chemotaxis,
            bands,
            bar,
            presence=presence,
            version=version,
        )
    out: dict[str, Any] = {
        "version": version,
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
    out |= {
        key: value
        for key, value in (
            ("kinematics_reported", reported),
            ("activity_reported", activity_reported),
        )
        if value
    }
    if search_presence is not None:
        out["local_search_presence"] = search_presence
        out["family_passes"]["local_search"] = bool(
            out["family_passes"]["local_search"] and search_presence["present"],
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
    if not presence:
        # Passes need presence; a grading without it gives statistics and distances only.
        out["family_passes"] = None
        out["wild_type_passes"] = None
        out["presence"] = None
    return out


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
