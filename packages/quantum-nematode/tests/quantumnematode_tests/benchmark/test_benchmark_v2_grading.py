"""B v2's grading path: measured captures, the five families' weights, and the field's frame.

Covers the benchmark-grading spec's scenarios "Real tracks are untouched" and "A past verdict is
reproduced" (v1's grading surface is unchanged), and the benchmark-controls spec's "A held-out
PD1074 cohort is requested" (the v2 build refuses before Part A is committed).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from quantumnematode.benchmark import grading as g
from quantumnematode.benchmark import measurement as mm
from quantumnematode.evaluation.odour_field import CHEN_PLATE_MM, OdourField, chen_field
from quantumnematode.report.behaviour import BehaviourStep

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
import benchmark_bands as bb  # noqa: E402  # pyright: ignore[reportMissingImports]
import benchmark_grade as bg  # noqa: E402  # pyright: ignore[reportMissingImports]


def _steps(n: int, x0: float = 0.0, dx: float = 0.1) -> list[BehaviourStep]:
    return [
        BehaviourStep(
            step=i,
            x=x0 + dx * i,
            y=0.0,
            heading_rad=0.0,
            concentration=0.0,
            dc_dt=0.0,
            grad_dir=0.0,
            grad_strength=0.0,
            on_lawn=True,
            primitive="run",
        )
        for i in range(n)
    ]


def _capture(tmp_path: Path, seeds: list[int], family: str = "small_lawn") -> Path:
    n = g.SEARCH_SAMPLES if family == "local_search" else g.LAWN_SAMPLES
    runs = [
        {
            "run": k,
            "seed": seed,
            "steps": [
                {k2: v for k2, v in dataclasses.asdict(s).items() if v is not None}
                for s in _steps(n)
            ],
        }
        for k, seed in enumerate(seeds)
    ]
    path = tmp_path / "behaviour_capture.json"
    path.write_text(json.dumps({"runs": runs, "assay": {"analysis_steps": [0, n]}}))
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{family} {path}\n")
    return manifest


def test_captures_are_measured_on_their_own_seed(tmp_path: Path) -> None:
    """Under v2 each captured run is measured on its seed's stream; without a model it is exact."""
    manifest = _capture(tmp_path, [11, 12])
    model = mm.TrackerModel(jitter_mm=0.01, p_other=0.01, runs_other_s=(5,))
    exact = bg.capture_populations(manifest)["small_lawn"]
    measured = bg.capture_populations(manifest, model)["small_lawn"]
    assert [s.x for s in exact[0]] == [s.x for s in _steps(g.LAWN_SAMPLES)]
    for seed, animal in zip((11, 12), measured, strict=True):
        expected = mm.measure(_steps(g.LAWN_SAMPLES), seed, model)
        np.testing.assert_array_equal([s.x for s in animal], [s.x for s in expected])
    assert any(math.isnan(s.x) for s in measured[0])


def _refuse_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        msg = "these tracks must not be measured"
        raise AssertionError(msg)

    monkeypatch.setattr(mm, "measure", refuse)


def _track_manifest(tmp_path: Path, animals: int = 64) -> Path:
    """Write one band-cohort Scheer animal as a CSV track; list it ``animals`` times."""
    import track_import_check as tic  # pyright: ignore[reportMissingImports]
    from quantumnematode.benchmark import references as refs

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


def test_imported_tracks_are_never_measured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers "Real tracks are untouched": grading imported tracks on v2 never calls the model.

    Sixty-four copies of one band-cohort Scheer animal are graded through the CLI with
    ``--bands v2``, the vendored bands' size; the reversal rate is reported beside the grade.
    """
    _refuse_measurement(monkeypatch)
    manifest = _track_manifest(tmp_path)
    out = tmp_path / "grade.json"
    assert bg.main(["tracks", "--manifest", str(manifest), "--out", str(out), "--bands", "v2"]) == 0
    result = json.loads(out.read_text())
    assert result["bands"] == "v2"
    assert result["n"] == {"small_lawn": 64, "kinematics": 64}
    assert set(result["families"]["kinematics"]) == set(g.KINEMATIC_STATISTICS)
    assert set(result["kinematics_reported"]) == set(g.KINEMATIC_REPORTED)
    reversals = result["kinematics_reported"]["reversals_per_on_lawn_min"]
    assert isinstance(reversals, float)
    assert math.isfinite(reversals)
    assert result["D"] is None


def test_the_default_grades_on_v2_once_it_is_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no ``--bands``, a VALID verdict for the vendored bands makes the grade a v2 grade."""
    controls = tmp_path / "controls_v2.json"
    controls.write_text(
        json.dumps({"verdict": "VALID", "bands_sha256": mm.file_sha256(bg.BANDS_V2_PATH)}),
    )
    monkeypatch.setattr(bg, "CONTROLS_V2_PATH", controls)
    _refuse_measurement(monkeypatch)
    out = tmp_path / "grade.json"
    assert bg.main(["tracks", "--manifest", str(_track_manifest(tmp_path)), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["bands"] == "v2"


def test_v1_captures_are_graded_without_the_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers "A past verdict is reproduced": ``--bands v1`` grades captures on exact positions."""
    _refuse_measurement(monkeypatch)
    manifest = _capture(tmp_path, list(range(64)))
    out = tmp_path / "grade.json"
    assert (
        bg.main(["captures", "--manifest", str(manifest), "--out", str(out), "--bands", "v1"]) == 0
    )
    result = json.loads(out.read_text())
    assert "bands" not in result
    assert set(result["families"]) == {"small_lawn"}
    assert result["bar"] == pytest.approx(2.8375969129437877)


def test_v2_weights_give_each_of_five_families_a_fifth() -> None:
    """D weights the five families equally (19 statistics), and D_fit its three fitted families."""
    families = bb.v2_families()
    assert bb.v2_statistics() == sum(len(v) for v in families.values()) == 19
    w = g.weights(list(families), families)
    for family in families:
        assert sum(v for (f, _), v in w.items() if f == family) == pytest.approx(0.2)
    fit = g.weights(list(bb.V2_FIT_FAMILIES), families)
    assert {f for f, _ in fit} == {"small_lawn", "uniform_lawn", "kinematics"}
    assert sum(fit.values()) == pytest.approx(1.0)
    assert g.z_star(19) == pytest.approx(3.0078, abs=1e-4)


def test_v1_weights_and_bar_are_unchanged() -> None:
    """Covers "A past verdict is reproduced": v1 reads its own three families at its own bar."""
    w = g.weights(list(g.FAMILIES))
    assert {f for f, _ in w} == set(g.FAMILIES)
    assert sum(w.values()) == pytest.approx(1.0)
    assert bg.load_bands()["bars"]["wild_type"] == pytest.approx(2.8375969129437877)


def test_v2_is_the_default_only_for_the_validated_bands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default is v2 only when the verdict is VALID for the very bands file now vendored."""
    bands = tmp_path / "bands_v2.json"
    bands.write_text("{}\n")
    controls = tmp_path / "controls_v2.json"
    monkeypatch.setattr(bg, "BANDS_V2_PATH", bands)
    monkeypatch.setattr(bg, "CONTROLS_V2_PATH", controls)
    controls.write_text(json.dumps({"verdict": "VALID", "bands_sha256": mm.file_sha256(bands)}))
    assert bg.v2_is_valid() is True
    bands.write_text('{"changed": true}\n')
    assert bg.v2_is_valid() is False
    controls.write_text(json.dumps({"verdict": "VALID"}))
    assert bg.v2_is_valid() is False
    controls.write_text("{not json")
    assert bg.v2_is_valid() is False


def test_samples_outside_the_field_are_missing() -> None:
    """A chemotaxis animal's samples outside the imaged field drop out, cutting its fragments."""
    field = OdourField(field=np.ones((11, 11)), cell_mm=1.0, origin_mm=(5.0, 5.0), c_max=1.0)
    # From x = 2 to x = 19 along y = 10: inside the field from x = 5 to x = 15.
    animal = [dataclasses.replace(s, y=10.0) for s in _steps(18, x0=2.0, dx=1.0)]
    fragments = bg.chemotaxis_fragments([animal], field)
    assert len(fragments) == 1
    x, y = fragments[0]
    np.testing.assert_allclose(x, np.arange(0.0, 11.0))
    np.testing.assert_allclose(y, 5.0)


def test_chen_field_is_centred_on_its_plate() -> None:
    """Chen's field sits centred on the 100 mm plate, at the probed scale and registration."""
    field = chen_field()
    width, height = field.extent_mm
    assert field.origin_mm == pytest.approx(
        ((CHEN_PLATE_MM - width) / 2, (CHEN_PLATE_MM - height) / 2),
    )
    probes = json.loads(bb.PROBES_PATH.read_text())["chen"]
    assert probes["field_registration"]["correlation"] > 0.9999
    assert probes["mm_per_px"] == pytest.approx(0.03009, abs=1e-4)


def test_the_v2_build_refuses_before_part_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no committed launch record the v2 build writes nothing."""
    monkeypatch.setattr(bb, "V2_LAUNCH_PATH", _REPO / "no-such-dir" / "launch.md")
    out, record = tmp_path / "bands.json", tmp_path / "record.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_bands.py", "--version", "v2", "--out", str(out), "--record", str(record)],
    )
    assert bb.main() == 1
    assert not out.exists()
    assert not record.exists()


def _samples(rng: np.random.Generator, drift: float, bias: float) -> dict[str, np.ndarray]:
    n = 600
    dcdt = rng.normal(size=n)
    # Onsets more likely when concentration falls, by ``bias``.
    onset = rng.random(n) < 0.02 * np.exp(-bias * dcdt)
    return {
        "drift": rng.normal(drift, 0.05, size=n),
        "dcdt": dcdt,
        "onset": onset,
        "turning": rng.normal(size=n),
        "sin_bearing": rng.uniform(-1, 1, size=n),
    }


def test_a_worm_that_ignores_the_gradient_is_not_present() -> None:
    """Covers the benchmark-grading spec's scenario "A worm that ignores the gradient".

    Drift and the onset bias of a random walk give no chemotaxis; a biased, drifting population
    has it.
    """
    from quantumnematode.benchmark import chemotaxis as ch

    rng = np.random.default_rng(5)
    blind = [[_samples(rng, 0.0, 0.0)] for _ in range(64)]
    worm = [[_samples(rng, 0.02, 1.0)] for _ in range(64)]
    assert ch.chemotaxis_presence(blind)["present"] is False
    present = ch.chemotaxis_presence(worm)
    assert present["present"] is True
    assert set(present["lower_bounds"]) == set(ch.PRESENCE_STATISTICS)


def test_presence_counts_an_undefined_resample_as_absent() -> None:
    """A population with no onsets has no quintile ratio, so its chemotaxis is not present."""
    from quantumnematode.benchmark import chemotaxis as ch

    rng = np.random.default_rng(6)
    groups = [[{**_samples(rng, 0.05, 0.0), "onset": np.zeros(600, dtype=bool)}] for _ in range(8)]
    out = ch.chemotaxis_presence(groups)
    assert out["present"] is False
    assert out["lower_bounds"]["log_quintile_ratio"] == -math.inf


def test_pooled_statistics_match_the_fragment_path() -> None:
    """Pooling precomputed samples gives the statistics computed from the fragments directly."""
    from quantumnematode.benchmark import chemotaxis as ch

    rng = np.random.default_rng(7)
    fragments = [tuple(np.cumsum(rng.normal(0.0, 0.1, size=(2, 120)), axis=1)) for _ in range(10)]

    def concentration(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(x) + 0.5 * np.asarray(y)

    direct = ch.chemotaxis_statistics(fragments, concentration)
    pooled = ch.pooled_statistics(ch.fragment_samples(fragments, concentration))
    assert direct == pooled


def test_local_search_tracks_stay_exact(tmp_path: Path) -> None:
    """Local search is graded against another rig's counts, so its captures are never measured."""
    manifest = _capture(tmp_path, [11], family="local_search")
    model = mm.TrackerModel(jitter_mm=0.01, p_other=0.01, runs_other_s=(5,))
    animal = bg.capture_populations(manifest, model)["local_search"][0]
    assert [s.x for s in animal] == [s.x for s in _steps(g.SEARCH_SAMPLES)]
