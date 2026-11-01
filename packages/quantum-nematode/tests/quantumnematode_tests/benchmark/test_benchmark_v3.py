"""B v3: local search on a log scale, chemotaxis's onset rate and direction flips, and the share.

Covers the benchmark-grading spec's scenarios "A flat search curve fails", "Frequent direction flips
fail", "v2 is unchanged" and "An imported track reports no share", and the benchmark-controls
spec's "v3 becomes the default" (the default chain). The v3 bands here stand in for the vendored
file: v2's, with local search rebuilt from López-Cruz's band batches and chemotaxis's two new
statistics on stand-in bands.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from quantumnematode.benchmark import chemotaxis as ch
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import measurement as mm
from quantumnematode.benchmark import references as refs
from quantumnematode.report.behaviour import BehaviourStep

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import benchmark_bands as bb  # noqa: E402  # pyright: ignore[reportMissingImports]
import benchmark_grade as bg  # noqa: E402  # pyright: ignore[reportMissingImports]

V3_BAR = 3.0380743049242103


def _search_band() -> dict[str, g.Band]:
    return bb.search_band_v3(list(bb.search_band_batches_v3(refs.load_lopezcruz()).values()))


def _band_entry(band: g.Band) -> dict[str, Any]:
    return {"centre": band.centre, "scale": band.scale, "inflated": False}


def _bands_v3() -> dict[str, Any]:
    v2 = bg.load_bands_v2()
    search = {k: _band_entry(v) for k, v in _search_band().items()}
    bands = json.loads(json.dumps(v2["bands"]))
    for size in bands.values():
        size["local_search"] = search
        size["chemotaxis"] = {
            **size["chemotaxis"],
            "onset_per_min": {"centre": 1.1, "scale": 0.94, "inflated": False},
            "flip_fraction": {"centre": 0.023, "scale": 0.020, "inflated": False},
        }
    families = bb.v3_families()
    return {
        **v2,
        "version": 3,
        "families": {f: list(st) for f, st in families.items()},
        "bars": {**v2["bars"], "wild_type": g.z_star(bb.v3_statistics())},
        "bands": bands,
        "reported": {"reversal_share": {"value": 0.173}},
    }


def _steps(n: int, primitive: str | None = "run") -> list[BehaviourStep]:
    return [
        BehaviourStep(
            step=i,
            x=0.1 * i,
            y=0.0,
            heading_rad=0.0,
            concentration=0.0,
            dc_dt=0.0,
            grad_dir=0.0,
            grad_strength=0.0,
            on_lawn=True,
            primitive=primitive,
        )
        for i in range(n)
    ]


def test_direction_flips_count_only_pairs_above_the_floor() -> None:
    """A pair counts when both 1-s displacements reach the floor; it flips beyond 135 degrees."""
    t = np.arange(20, dtype=float)
    assert ch.direction_flips(0.1 * t, 0.0 * t) == (0, 18)
    back_and_forth = 0.1 * (t % 2)
    assert ch.direction_flips(back_and_forth, 0.0 * t) == (18, 18)
    assert ch.direction_flips(0.01 * t, 0.0 * t) == (0, 0)
    # A right angle is a turn, not a flip; a further 150 degrees is a flip.
    angle = math.radians(90.0 + 150.0)
    x = np.array([0.0, 0.1, 0.1, 0.1 + 0.1 * math.cos(angle)])
    y = np.array([0.0, 0.0, 0.1, 0.1 + 0.1 * math.sin(angle)])
    assert ch.direction_flips(x, y) == (1, 2)
    # A missing sample removes both pairs it touches.
    gapped = 0.1 * t
    gapped[10] = math.nan
    assert ch.direction_flips(gapped, 0.0 * t) == (0, 15)


def _walk(rng: np.random.Generator, n: int, reverse_per_s: float) -> tuple[np.ndarray, np.ndarray]:
    heading = np.cumsum(rng.normal(0.0, 0.3, size=n) + math.pi * (rng.random(n) < reverse_per_s))
    x = np.concatenate([[0.0], np.cumsum(0.1 * np.cos(heading))])
    y = np.concatenate([[0.0], np.cumsum(0.1 * np.sin(heading))])
    return x, y


def _concentration(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(x) + 0.5 * np.asarray(y)


def _recording(rng: np.random.Generator, reverse_per_s: float) -> dict[str, float]:
    fragments = [_walk(rng, 120, reverse_per_s) for _ in range(30)]
    return ch.pooled_statistics_v3(ch.fragment_samples(fragments, _concentration))


def test_pooled_v3_statistics_extend_v2s() -> None:
    """v3's first three statistics are v2's; the onset rate and flip fraction pool their parts."""
    rng = np.random.default_rng(11)
    fragments = [_walk(rng, 120, 0.05) for _ in range(10)]
    samples = ch.fragment_samples(fragments, _concentration)
    v2, v3 = ch.pooled_statistics(samples), ch.pooled_statistics_v3(samples)
    assert tuple(v3) == ch.CHEMOTAXIS_STATISTICS_V3
    assert {k: v3[k] for k in ch.CHEMOTAXIS_STATISTICS} == v2
    onsets = np.concatenate([s["onset"] for s in samples])
    assert v3["onset_per_min"] == pytest.approx(60.0 * onsets.mean())
    flips = [ch.direction_flips(x, y) for x, y in fragments]
    assert v3["flip_fraction"] == sum(f for f, _ in flips) / sum(p for _, p in flips)
    empty = ch.pooled_statistics_v3([])
    assert math.isnan(empty["onset_per_min"])
    assert math.isnan(empty["flip_fraction"])


def test_frequent_direction_flips_fail() -> None:
    """Covers "Frequent direction flips fail": a worm turning back several times as often fails.

    Six recordings of a walker that reverses at 0.02/s make the band, as Chen's six standard
    recordings do; a population reversing at 0.25/s fails the flip fraction at v3's bar.
    """
    rng = np.random.default_rng(12)
    band = bb.chemotaxis_band_v3([_recording(rng, 0.02) for _ in range(6)])
    faithful = g.grade(_recording(rng, 0.02), band, V3_BAR)
    reverser = g.grade(_recording(rng, 0.25), band, V3_BAR)
    assert faithful["flip_fraction"].passes
    assert reverser["flip_fraction"].z > V3_BAR
    assert not reverser["flip_fraction"].passes


def test_a_flat_search_curve_fails() -> None:
    """Covers "A flat search curve fails": a ratio near 1 fails v3's log band, though v2's passes.

    The band is the López-Cruz band batches' at v3's factor; their log ratio's mean is about 1.39.
    """
    band = _search_band()
    assert band["log_early_late_ratio"].centre == pytest.approx(1.3928, abs=1e-4)
    gamma = band["gamma_per_min"].centre
    flat = g.log_search_statistics({"gamma_per_min": gamma, "early_late_ratio": 1.05})
    readings = g.grade(flat, band, V3_BAR)
    assert readings["log_early_late_ratio"].z < -V3_BAR
    assert not readings["log_early_late_ratio"].passes
    real = {"gamma_per_min": gamma, "early_late_ratio": math.exp(1.3928)}
    assert all(r.passes for r in g.grade(g.log_search_statistics(real), band, V3_BAR).values())
    v2 = bg.family_bands_v2(bg.load_bands_v2(), "local_search", 200)
    assert g.grade({"gamma_per_min": gamma, "early_late_ratio": 1.05}, v2, 3.0078)[
        "early_late_ratio"
    ].passes


def test_the_log_ratio_is_undefined_without_a_positive_ratio() -> None:
    """A ratio that is zero, negative or NaN has no log; gamma passes through."""
    for ratio in (0.0, -1.0, math.nan):
        out = g.log_search_statistics({"gamma_per_min": 0.1, "early_late_ratio": ratio})
        assert out["gamma_per_min"] == 0.1
        assert math.isnan(out["log_early_late_ratio"])


def test_v3_weights_give_each_of_five_families_a_fifth() -> None:
    """D weights v3's five families equally over 21 statistics; D_fit keeps v2's three families."""
    families = bb.v3_families()
    assert bb.v3_statistics() == 21
    assert g.z_star(21) == pytest.approx(V3_BAR)
    w = g.weights(list(families), families)
    for family in families:
        assert sum(v for (f, _), v in w.items() if f == family) == pytest.approx(0.2)
    assert w[("chemotaxis", "flip_fraction")] == pytest.approx(0.04)
    assert families["small_lawn"] == g.LAWN_STATISTICS
    assert families["kinematics"] == g.KINEMATIC_STATISTICS
    assert bb.prediction_factor_v3(17) > bb.prediction_factor(17)


def test_v2_is_unchanged() -> None:
    """Covers "v2 is unchanged": v2's tuples, its count, and its grading's shape stay as they were.

    Every v2 statistic, D and presence reading was compared byte for byte against a baseline from
    the code before v3 (`speedup_identity_check.py check`); this pins the surface that check reads.
    """
    assert ch.CHEMOTAXIS_STATISTICS == ("drift_mm_s", "log_quintile_ratio", "weathervane_deg_s")
    assert g.LOCAL_SEARCH_STATISTICS == ("gamma_per_min", "early_late_ratio")
    assert bb.v2_statistics() == 19
    assert bb.v2_families()["chemotaxis"] == ch.CHEMOTAXIS_STATISTICS
    populations = {"small_lawn": [_steps(g.LAWN_SAMPLES) for _ in range(64)]}
    v2 = bg.grade_populations_v2(populations, None, bg.load_bands_v2())
    v3 = bg.grade_populations_v3(populations, None, _bands_v3())
    assert v2["version"] == 2
    assert "reversal_share" not in v2
    assert v3["version"] == 3
    for family in ("small_lawn", "kinematics"):
        assert bg._finite(v3["families"][family]) == bg._finite(v2["families"][family])


def test_the_reversal_share_is_reported_where_labelled() -> None:
    """The share of labelled seconds reversing is reported over local search and chemotaxis."""
    bands = _bands_v3()
    half = [_steps(10, "run") + _steps(10, "reversal")]
    out = bg.grade_populations_v3({"local_search": half}, [_steps(20, "run")], bands)
    share = out["reversal_share"]
    assert share["value"] == pytest.approx(10 / 40)
    assert share["windows"] == ["local_search", "chemotaxis"]
    assert share["reference"] == {"value": 0.173}
    assert g.reversal_share([_steps(5, None)]) is None
    assert g.reversal_share([]) is None


def _track_manifest(tmp_path: Path, animals: int = 64) -> Path:
    """Write one band-cohort Scheer animal as a CSV track; list it ``animals`` times."""
    import track_import_check as tic  # pyright: ignore[reportMissingImports]

    tracks = refs.load_scheer_tracks()
    row = int(bb.band_rows(tracks, bb.WILD_TYPE)[0])
    t = np.arange(tracks.x.shape[1], dtype=float)
    path = tic.write_track(
        tmp_path,
        "animal",
        "csv",
        t,
        tracks.x[row],
        tracks.y[row],
        tic._lawn_metadata("small_lawn", tracks.boundary_radius[row]),
    )
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"small_lawn {path}\n" * animals)
    return manifest


def _write_bands_v3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "bands_v3.json"
    path.write_text(json.dumps(_bands_v3(), indent=2) + "\n")
    monkeypatch.setattr(bg, "BANDS_V3_PATH", path)
    return path


def test_an_imported_track_reports_no_share(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers "An imported track reports no share": imported positions grade on v3 with no share."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        msg = "imported tracks must not be measured"
        raise AssertionError(msg)

    monkeypatch.setattr(mm, "measure", refuse)
    _write_bands_v3(tmp_path, monkeypatch)
    out = tmp_path / "grade.json"
    manifest = _track_manifest(tmp_path)
    assert bg.main(["tracks", "--manifest", str(manifest), "--out", str(out), "--bands", "v3"]) == 0
    result = json.loads(out.read_text())
    assert result["bands"] == "v3"
    assert result["version"] == 3
    assert result["reversal_share"] is None
    assert set(result["families"]) == {"small_lawn", "kinematics"}
    assert set(result["families"]["small_lawn"]) == set(g.LAWN_STATISTICS)
    assert result["bar"] == pytest.approx(V3_BAR)


def test_v3_is_the_default_only_once_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers "v3 becomes the default": a VALID verdict for v3's very bands makes it the default."""
    bands = _write_bands_v3(tmp_path, monkeypatch)
    controls = tmp_path / "controls_v3.json"
    monkeypatch.setattr(bg, "CONTROLS_V3_PATH", controls)
    assert bg.v3_is_valid() is False
    controls.write_text(json.dumps({"verdict": "VOID", "bands_sha256": mm.file_sha256(bands)}))
    assert bg.v3_is_valid() is False
    controls.write_text(json.dumps({"verdict": "VALID", "bands_sha256": mm.file_sha256(bands)}))
    assert bg.v3_is_valid() is True
    out = tmp_path / "grade.json"
    manifest = _track_manifest(tmp_path)
    assert bg.main(["tracks", "--manifest", str(manifest), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["bands"] == "v3"
    bands.write_text(bands.read_text().replace('"version": 3', '"version": 3 '))
    assert bg.v3_is_valid() is False


def test_the_v3_build_refuses_before_part_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no committed launch record the v3 build writes nothing."""
    monkeypatch.setattr(bb, "V3_LAUNCH_PATH", _REPO / "no-such-dir" / "launch.md")
    out, record = tmp_path / "bands.json", tmp_path / "record.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_bands.py", "--version", "v3", "--out", str(out), "--record", str(record)],
    )
    assert bb.main() == 1
    assert not out.exists()
    assert not record.exists()


def test_the_reported_reference_is_roberts_reverse_share() -> None:
    """The reported share's reference is Roberts' tracked frames in reverse: about 17.3%."""
    reference = bb.roberts_reversal_share()
    assert reference["value"] == pytest.approx(0.1734, abs=1e-4)
    assert reference["frames"] > 1_000_000


def test_the_vendored_v3_bands_follow_their_rules() -> None:
    """The vendored v3 bands: v2's lawn and kinematics, local search and chemotaxis at v3's factor.

    Local search is rebuilt from the López-Cruz band batches, and chemotaxis's centres are the means
    of the standard recordings in the bands' record; the verdict validates these very bands.
    """
    v2, v3 = bg.load_bands_v2(), bg.load_bands_v3()
    assert v3["version"] == 3
    assert v3["bars"]["wild_type"] == pytest.approx(V3_BAR)
    for size in ("64", "200"):
        for family in ("small_lawn", "uniform_lawn", "mutants", "kinematics"):
            assert v3["bands"][size][family] == v2["bands"][size][family]
    search = _search_band()
    for name, band in search.items():
        assert v3["bands"]["200"]["local_search"][name]["centre"] == pytest.approx(band.centre)
        assert v3["bands"]["200"]["local_search"][name]["scale"] == pytest.approx(band.scale)
    record = json.loads((bb.V3_SUPPORT_PATH / "bands_v3_record.json").read_text())
    standard = [
        r["statistics"]
        for r in record["chemotaxis_recordings"].values()
        if r["protocol"] == "standard"
    ]
    assert len(standard) == 6
    for name in ch.CHEMOTAXIS_STATISTICS_V3:
        centre = v3["bands"]["200"]["chemotaxis"][name]["centre"]
        assert centre == pytest.approx(float(np.mean([s[name] for s in standard])))
    controls = json.loads(bg.CONTROLS_V3_PATH.read_text())
    assert controls["verdict"] == "VALID"
    assert controls["bands_sha256"] == mm.file_sha256(bg.BANDS_V3_PATH)
