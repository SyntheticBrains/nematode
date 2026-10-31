"""The measurement model: noise, lost seconds, fragmentation and per-animal streams.

Covers the benchmark-grading spec's scenario "An omega second can go missing".
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from quantumnematode.benchmark import measurement as mm
from quantumnematode.report.behaviour import BehaviourStep

if TYPE_CHECKING:
    from pathlib import Path


def _steps(n: int, kinds: list[str] | None = None) -> list[BehaviourStep]:
    kinds = kinds or ["run"] * n
    return [
        BehaviourStep(
            step=i,
            x=float(i),
            y=0.0,
            heading_rad=0.0,
            concentration=0.0,
            dc_dt=0.0,
            grad_dir=0.0,
            grad_strength=0.0,
            on_lawn=True,
            primitive=kinds[i],
        )
        for i in range(n)
    ]


def test_noise_has_the_stated_sd() -> None:
    """Positions gain normal noise of jitter_mm per axis; nothing else changes."""
    model = mm.TrackerModel(jitter_mm=0.01)
    out = mm.measure(_steps(20_000), seed=3, model=model)
    dx = np.array([o.x - i for i, o in enumerate(out)])
    assert np.std(dx) == pytest.approx(0.01, rel=0.03)
    assert all(o.on_lawn for o in out)


def test_an_omega_second_can_go_missing() -> None:
    """A loss starts in omega seconds at p_omega and lasts a run drawn from the omega runs."""
    model = mm.TrackerModel(jitter_mm=0.0, p_omega=1.0, p_other=0.0, runs_omega_s=(3,))
    kinds = ["run"] * 10 + ["omega"] + ["run"] * 10
    out = mm.measure(_steps(21, kinds), seed=1, model=model)
    lost = [math.isnan(o.x) for o in out]
    assert lost == [False] * 10 + [True] * 3 + [False] * 8


def test_without_runs_a_loss_is_one_second() -> None:
    """Without measured runs, a start loses that second alone."""
    rng = np.random.default_rng(0)
    model = mm.TrackerModel(jitter_mm=0.0, p_omega=0.0, p_other=1.0)
    lost = mm.lost_seconds(["run"] * 5, model, rng)
    assert lost.tolist() == [True] * 5


def test_an_animal_is_measured_the_same_alone_or_again() -> None:
    """The animal's own stream makes the measurement reproducible from its seed."""
    model = mm.TrackerModel(jitter_mm=0.01, p_other=0.01, runs_other_s=(5, 10))
    a = mm.measure(_steps(500), seed=42, model=model)
    b = mm.measure(_steps(500), seed=42, model=model)
    c = mm.measure(_steps(500), seed=43, model=model)
    assert [(s.x, s.y) for s in a] == [(s.x, s.y) for s in b]
    assert [(s.x, s.y) for s in a] != [(s.x, s.y) for s in c]


def test_fragments_are_cut_with_one_lost_second() -> None:
    """A fragmenting tracker cuts the track at drawn lengths, one lost second at each cut."""
    model = mm.TrackerModel(jitter_mm=0.0, fragment_lengths_s=(10,))
    out = mm.measure(_steps(35), seed=0, model=model)
    lost = [i for i, o in enumerate(out) if math.isnan(o.x)]
    assert lost == [10, 21, 32]


def test_jitter_from_steps_inverts_the_still_worm() -> None:
    """Steps between noisy readings of one point give back the per-axis SD."""
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 0.005, size=(2, 100_000, 2))
    steps = np.hypot(*(a - b).T)
    assert mm.jitter_from_steps(steps) == pytest.approx(0.005, rel=0.02)


def test_the_model_round_trips_through_its_record() -> None:
    """A recorded model reads back as the same model."""
    model = mm.TrackerModel(
        jitter_mm=0.004,
        p_omega=0.05,
        runs_omega_s=(16, 20),
        fragment_lengths_s=(39,),
    )
    assert mm.TrackerModel.from_dict(model.to_dict()) == model


def _library() -> mm.NoiseLibrary:
    return mm.NoiseLibrary(
        x_mm=np.array([1.0, 2.0, 3.0, 10.0, 20.0]),
        y_mm=np.array([-1.0, -2.0, -3.0, -10.0, -20.0]),
        lengths=np.array([3, 2]),
    )


def test_a_noise_track_is_spliced_at_the_nearest_offset() -> None:
    """Covers "The lawn rig's tracker noise is resampled real noise": tracks splice, never jump.

    Every drawn second is a stored offset. Where a stretch ends, the track continues from the
    offset after the nearest match in another stretch, a stretch's last offset never matching.
    """
    lib = mm.NoiseLibrary(
        x_mm=np.array([0.0, 1.0, 2.0, 10.0, 2.1, 5.0, 6.0]),
        y_mm=np.zeros(7),
        lengths=np.array([3, 4]),
    )
    for seed in range(50):
        track = lib.draw(6, np.random.default_rng(seed))
        assert set(track[:, 0]) <= set(lib.x_mm.tolist())
        if track[0, 0] == 0.0:
            # Stretch A ends at 2.0; the nearest usable match is B's 2.1, so it runs on at 5.0, 6.0.
            np.testing.assert_array_equal(track[:5, 0], [0.0, 1.0, 2.0, 5.0, 6.0])
            break
    else:
        pytest.fail("no draw started at the first stretch")


def test_a_library_model_measures_an_animal_the_same_alone_or_in_a_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers "An animal is measured the same way alone or in a population" for the library."""
    lib = _library()
    monkeypatch.setattr(mm, "load_noise_library", lambda _name: lib)
    model = mm.TrackerModel(jitter_mm=0.0, noise_library="toy.npz")
    first = mm.measure(_steps(30), seed=7, model=model)
    again = mm.measure(_steps(30), seed=7, model=model)
    other = mm.measure(_steps(30), seed=8, model=model)
    assert [s.x for s in first] == [s.x for s in again]
    assert [s.x for s in first] != [s.x for s in other]
    offsets = {round(s.x - i, 9) for i, s in enumerate(first)}
    assert offsets <= {1.0, 2.0, 3.0, 10.0, 20.0}


def test_a_saved_library_round_trips_to_the_micrometre(tmp_path: Path) -> None:
    """A library is stored in whole micrometres and read back in millimetres."""
    path = tmp_path / "lib.npz"
    mm.save_noise_library(path, np.array([0.0012, -0.0034]), np.array([0.0, 0.0101]), np.array([2]))
    with np.load(path) as store:
        assert store["x_um"].tolist() == [1, -3]
        assert store["y_um"].tolist() == [0, 10]
        assert store["lengths"].tolist() == [2]


def test_a_changed_library_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model whose library file no longer matches its recorded digest refuses to measure."""
    monkeypatch.setattr(mm, "_library_sha256", lambda _name: "f" * 64)
    monkeypatch.setattr(mm, "load_noise_library", lambda _name: _library())
    model = mm.TrackerModel(jitter_mm=0.0, noise_library="toy.npz", noise_sha256="0" * 64)
    with pytest.raises(ValueError, match="digest"):
        mm.measure(_steps(10), seed=1, model=model)
    unchecked = mm.TrackerModel(jitter_mm=0.0, noise_library="toy.npz")
    assert len(mm.measure(_steps(10), seed=1, model=unchecked)) == 10


def test_a_rewritten_library_is_refused_without_clearing_a_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A library rewritten on disk is seen at once: its digest no longer matches its model's."""
    monkeypatch.setattr(mm, "BENCHMARK_DATA", tmp_path)
    path = tmp_path / "lib.npz"
    mm.save_noise_library(path, np.array([0.001, 0.002, 0.003]), np.zeros(3), np.array([3]))
    model = mm.TrackerModel(
        jitter_mm=0.0,
        noise_library="lib.npz",
        noise_sha256=mm.file_sha256(path),
    )
    assert len(mm.measure(_steps(10), seed=1, model=model)) == 10
    # Rewritten behind the model's back, not through save_noise_library.
    np.savez_compressed(
        path,
        x_um=np.array([5, 6, 7, 8], dtype=np.int32),
        y_um=np.zeros(4, dtype=np.int32),
        lengths=np.array([4], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="digest"):
        mm.measure(_steps(10), seed=1, model=model)
    assert mm.load_noise_library("lib.npz").lengths.tolist() == [4]


def _measure_by_replace(steps: list[BehaviourStep], seed: int, model: mm.TrackerModel) -> list:
    """Measure as first written, with dataclasses.replace: the parity reference."""
    import dataclasses

    rng = np.random.default_rng([seed, model.stream_key])
    n = len(steps)
    if model.noise_library is not None:
        noise = mm.checked_noise_library(model).draw(n, rng)
    else:
        noise = rng.normal(0.0, model.jitter_mm, size=(n, 2))
    lost = mm.lost_seconds([s.primitive for s in steps], model, rng)
    if model.fragment_lengths_s is not None:
        lost[mm.fragment_cuts(n, model.fragment_lengths_s, rng)] = True
    return [
        dataclasses.replace(s, x=math.nan, y=math.nan)
        if lost[i]
        else dataclasses.replace(s, x=s.x + noise[i, 0], y=s.y + noise[i, 1])
        for i, s in enumerate(steps)
    ]


@pytest.mark.parametrize("vendored", [True, False])
def test_measured_steps_are_unchanged_by_the_faster_construction(vendored: bool) -> None:  # noqa: FBT001
    """Measured steps are equal field for field to the dataclasses.replace measurement."""
    import json

    from quantumnematode.benchmark import references as refs

    if vendored:
        bands = json.loads((refs.BENCHMARK_DATA / "bands_v2.json").read_text())
        model = mm.TrackerModel.from_dict(bands["measurement"]["lawn"])
    else:
        model = mm.TrackerModel(jitter_mm=0.005, p_omega=0.3, runs_omega_s=(2, 5), p_other=0.01)
    import dataclasses

    kinds = (["run"] * 40 + ["omega"] * 3) * 20
    # Every field carries a value of its own, so a regression dropping or swapping any shows.
    steps = [
        dataclasses.replace(
            s,
            heading_rad=0.1 * i,
            concentration=0.5 + 0.001 * i,
            dc_dt=-0.002 * i,
            grad_dir=0.3,
            grad_strength=0.7,
            speed_signed=0.05,
            satiety=0.9 - 0.0001 * i,
            intake=0.01,
            phase="assay",
        )
        for i, s in enumerate(_steps(len(kinds), kinds))
    ]
    names = [f.name for f in dataclasses.fields(BehaviourStep) if f.init]
    for seed in (1, 2, 3):
        fast = mm.measure(steps, seed, model)
        slow = _measure_by_replace(steps, seed, model)
        assert len(fast) == len(slow)
        for a, b in zip(fast, slow, strict=True):
            assert type(a) is type(b)
            for name in names:
                va, vb = getattr(a, name), getattr(b, name)
                assert type(va) is type(vb), name
                assert va == vb or (isinstance(va, float) and math.isnan(va) and math.isnan(vb))


def _reference_splice(
    lib: mm.NoiseLibrary,
    x: float,
    y: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Return a splice by the padded grid the end-to-end splice replaced."""
    ks = rng.integers(len(lib.lengths), size=mm.SPLICE_CANDIDATES)
    starts, lengths = lib.starts[ks], lib.lengths[ks]
    width = int(lengths.max()) - 1
    index = starts[:, None] + np.arange(width)[None, :]
    usable = np.arange(width)[None, :] < (lengths - 1)[:, None]
    index = np.where(usable, index, 0)
    distance = np.where(usable, np.hypot(lib.x_mm[index] - x, lib.y_mm[index] - y), np.inf)
    row, column = np.unravel_index(int(np.argmin(distance)), distance.shape)
    return int(index[row, column]) + 1, int(starts[row] + lengths[row])


@pytest.mark.parametrize("vendored", [True, False])
def test_a_splice_continues_where_the_padded_grid_did(vendored: bool) -> None:  # noqa: FBT001
    """Covers "A measured track is the same": each splice makes the same draws and choice.

    The synthetic library has ties and one-offset stretches, so the first-minimum rule and the
    stretches with no usable offset are exercised.
    """
    import json

    from quantumnematode.benchmark import references as refs

    if vendored:
        bands = json.loads((refs.BENCHMARK_DATA / "bands_v2.json").read_text())
        lib = mm.checked_noise_library(mm.TrackerModel.from_dict(bands["measurement"]["lawn"]))
    else:
        lib = mm.NoiseLibrary(
            x_mm=np.array([0.0, 1.0, 1.0, 5.0, 1.0, 2.0, 1.0, 3.0, 0.0]),
            y_mm=np.zeros(9),
            lengths=np.array([3, 1, 2, 1, 2]),
        )
    assert lib is not None
    probe = np.random.default_rng(0)
    for seed in range(200):
        x, y = probe.normal(0.0, 0.01 if vendored else 2.0, size=2)
        fast = lib._splice(float(x), float(y), np.random.default_rng(seed))
        slow = _reference_splice(lib, float(x), float(y), np.random.default_rng(seed))
        assert fast == slow
    for seed in range(20):
        np.testing.assert_array_equal(
            lib.draw(500, np.random.default_rng(seed)),
            _reference_draw(lib, 500, np.random.default_rng(seed)),
        )


def _reference_draw(lib: mm.NoiseLibrary, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a noise track drawn as before, splicing by the padded grid."""
    out = np.empty((n, 2))
    k = int(rng.integers(len(lib.lengths)))
    j, end = int(lib.starts[k]), int(lib.starts[k] + lib.lengths[k])
    t = 0
    while t < n:
        take = min(end - j, n - t)
        out[t : t + take, 0] = lib.x_mm[j : j + take]
        out[t : t + take, 1] = lib.y_mm[j : j + take]
        t += take
        if t < n:
            j, end = _reference_splice(lib, out[t - 1, 0], out[t - 1, 1], rng)
    return out


def test_the_library_computes_its_offsets_once() -> None:
    """A library's stretch offsets are computed once and reused at every splice."""
    lib = _library()
    assert lib.starts is lib.starts
    assert lib.starts.tolist() == [0, 3]
