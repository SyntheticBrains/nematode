"""The benchmark's bands: the vendored files, and bands built at any other size.

A family is graded at its population's own size: on the vendored bands where they were built at
that size, and otherwise on bands built now from the real tracks by :mod:`wormbench.builder`, each
band on its own generator, which gives the vendored values at the vendored sizes.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from wormbench import builder as bb
from wormbench import classifier as rd
from wormbench import data
from wormbench import grading as g
from wormbench.builder import BandBuilder


def _bands_path(path: Path | None, name: str) -> Path:
    """Return an explicit bands path as given, or the named bands file from the data sources."""
    return Path(path) if path is not None else data.path("benchmark", name)


def _sources_key() -> tuple[str | None, str | None]:
    """Return what decides where the tracks resolve, so a cache is kept per data sources."""
    root = data.get_root()
    return (None if root is None else str(root), os.environ.get(data.ENVIRONMENT_VARIABLE))


def load_bands(path: Path | None = None) -> dict[str, Any]:
    """Load the bands file."""
    return json.loads(_bands_path(path, "bands.json").read_text())


def _builder() -> BandBuilder:
    """Return the band builder for the current data sources, created once per sources.

    Its classified tallies are reused across every band it builds.
    """
    return _builder_for(_sources_key())


@functools.cache
def _builder_for(_key: tuple[str | None, str | None]) -> BandBuilder:
    return BandBuilder()


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
    return rd.GaussianClassifier(hmm=rd.load_calibrated_hmm(data.stored_path(bands["classifier"])))


def load_bands_v2(path: Path | None = None) -> dict[str, Any]:
    """Load benchmark v2's bands, with its families, bar and measurement models."""
    return json.loads(_bands_path(path, "bands_v2.json").read_text())


def load_bands_v3(path: Path | None = None) -> dict[str, Any]:
    """Load benchmark v3's bands: v2's layout, with v3's families, bar and reported reference."""
    return json.loads(_bands_path(path, "bands_v3.json").read_text())


def load_bands_v4(path: Path | None = None) -> dict[str, Any]:
    """Load benchmark v4's bands: v3's, with the on-lawn activity family."""
    return json.loads(_bands_path(path, "bands_v4.json").read_text())


def _v2_family_map(  # pyright: ignore[reportUnusedFunction] - read by wormbench.grade
    bands: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
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
    if family == "activity":
        return _activity_bands_at(n, tuple(sorted(bands["activity_kappa"].items())))
    return family_bands(bands, family, n)


def _kinematic_bands_at(n: int) -> dict[str, g.Band]:
    return _kinematic_bands_for(n, _sources_key())


@functools.cache
def _kinematic_bands_for(n: int, _key: tuple[str | None, str | None]) -> dict[str, g.Band]:
    builder = _builder()
    per_date = bb.per_date_kinematic_tallies(
        builder.tracks,
        bb.WILD_TYPE,
        builder._classifier,  # noqa: SLF001 - the builder's own classifier, loaded once
    )
    kappa = load_bands_v2()["kinematics_kappa"]
    return bb.lawn_bands(
        per_date,
        g.KINEMATIC_STATISTICS,
        n,
        np.random.default_rng([bb.BAND_SEED, bb.KINEMATICS_INDEX, n]),
        kappa=kappa,
        statistic=g.kinematic_statistics,
    )


def _activity_bands_at(n: int, kappa: tuple[tuple[str, float], ...]) -> dict[str, g.Band]:
    """Return the activity bands at n on the given bands' kappa, by the registered build's rule."""
    return _activity_bands_for(n, kappa, _sources_key())


@functools.cache
def _activity_bands_for(
    n: int,
    kappa: tuple[tuple[str, float], ...],
    _key: tuple[str | None, str | None],
) -> dict[str, g.Band]:
    builder = _builder()
    per_date = bb.per_date_activity_tallies(builder.tracks, bb.WILD_TYPE)
    return bb.lawn_bands(
        per_date,
        g.ACTIVITY_STATISTICS,
        n,
        np.random.default_rng([bb.BAND_SEED, bb.ACTIVITY_INDEX, n]),
        kappa=dict(kappa),
        statistic=g.activity_statistics,
    )
